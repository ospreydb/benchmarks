#!/usr/bin/env python3
"""
Controlled, recall-matched benchmark protocol for OspreyDB vs Qdrant vs pgvector.

Key properties:
  * Calibrates each engine on a validation split until Recall@10 is within
    ``recall_target +/- 0.005`` (or reports ``below_target``).
  * Uses an HTTP client-server transport for OspreyDB (local ``serve`` mode of
    ``osprey-bench-runner``) and Qdrant, and SQL client-server for pgvector.
  * Reports client round-trip latency, pure server-side latency, concurrent QPS
    (16 async workers), Peak Process RSS (``VmHWM``), and artifact sizes.
  * Runs 200 warmup queries followed by 1000 measured queries in 3 independent
    rounds, reporting the median and standard deviation of P50/P95/P99.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import psycopg
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts" / "controlled"
RESULTS_DIR = ROOT / "results"
COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"
TMP_DIR = Path("/tmp/kilo")

QDRANT_FP32_URL = "http://127.0.0.1:6333"
QDRANT_BQ_NAME = "osprey-bench-qdrant-bq"
QDRANT_BQ_PORT = 6335
QDRANT_BQ_DATA = TMP_DIR / "qdrant-bq-data"
PGVECTOR_CONTAINER = "osprey-bench-pgvector"
PGVECTOR_DSN = "postgresql://osprey:ospreypassword123@127.0.0.1:5433/osprey_bench"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def percentile(sorted_vals: Sequence[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = ((len(sorted_vals) - 1) * p) // 100
    return float(sorted_vals[idx])


def latency_stats(vals: Sequence[float]) -> Tuple[float, float, float]:
    ordered = sorted(float(v) for v in vals)
    return percentile(ordered, 50), percentile(ordered, 95), percentile(ordered, 99)


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


def recall_at_k(pred_ids: Sequence[Sequence[int]], gt: np.ndarray, k: int = 10) -> float:
    hits = 0
    denom = 0
    for pred, row in zip(pred_ids, gt):
        gt_set = {int(x) for x in row[:k]}
        hits += len({int(x) for x in pred[:k]} & gt_set)
        denom += k
    return (hits / denom) if denom else 0.0


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


def read_vmhwm_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def container_pids(name: str) -> List[int]:
    try:
        out = subprocess.check_output(
            ["docker", "top", name, "-eo", "pid"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    pids: List[int] = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def container_max_vmhwm_kb(name: str) -> int:
    return max((read_vmhwm_kb(pid) for pid in container_pids(name)), default=0)


class RssSampler:
    """Samples Peak Process RSS (VmHWM) in a background thread."""

    def __init__(self, read_fn: Callable[[], int], interval: float = 0.25):
        self.read_fn = read_fn
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.max_kb = 0

    def start(self) -> None:
        self._stop.clear()
        self.max_kb = max(self.max_kb, self.read_fn())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.max_kb = max(self.max_kb, self.read_fn())
            except Exception:
                pass

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self.max_kb = max(self.max_kb, self.read_fn())
        except Exception:
            pass
        return self.max_kb


def wait_http(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 - readiness probe
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {url}: {last_err}")


def measure_cmd_peak_rss(
    cmd: Sequence[str], env: Optional[Dict[str, str]] = None
) -> Tuple[float, float]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    time_bin = "/usr/bin/time"
    started = time.perf_counter()
    if os.path.exists(time_bin):
        proc = subprocess.run(
            [time_bin, "-v", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
            )
        for line in proc.stderr.splitlines():
            if "Maximum resident set size (kbytes):" in line:
                return elapsed, int(line.split(":")[-1].strip()) / 1024.0
        return elapsed, 0.0
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    return elapsed, 0.0


def stop_container(name: str) -> None:
    subprocess.run(
        ["docker", "stop", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_container(name: str) -> None:
    subprocess.run(
        ["docker", "start", name],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def ensure_compose_services() -> None:
    # Create the containers without starting them: each engine must be measured
    # in isolation, so engines are started and stopped around their own stage.
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "--no-start",
            "qdrant",
            "pgvector",
        ],
        check=True,
    )
    stop_all_engines()


def start_qdrant_fp32() -> None:
    start_container("osprey-bench-qdrant")
    wait_http(f"{QDRANT_FP32_URL}/", timeout=180)


def stop_qdrant_fp32() -> None:
    stop_container("osprey-bench-qdrant")


def start_pgvector() -> None:
    start_container(PGVECTOR_CONTAINER)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with psycopg.connect(PGVECTOR_DSN, connect_timeout=3):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("timeout waiting for pgvector")


def stop_pgvector() -> None:
    stop_container(PGVECTOR_CONTAINER)


def stop_all_engines() -> None:
    stop_qdrant_fp32()
    stop_pgvector()
    stop_qdrant_bq_container()


def clear_pg_tables() -> None:
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        cur = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
        )
        for (table,) in cur.fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')


def start_qdrant_bq_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", QDRANT_BQ_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    QDRANT_BQ_DATA.mkdir(parents=True, exist_ok=True)
    os.chmod(QDRANT_BQ_DATA, 0o777)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            QDRANT_BQ_NAME,
            "-p",
            f"{QDRANT_BQ_PORT}:6333",
            "-p",
            f"{QDRANT_BQ_PORT + 1}:6334",
            "-v",
            f"{QDRANT_BQ_DATA}:/qdrant/storage",
            "qdrant/qdrant:latest",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    wait_http(f"http://127.0.0.1:{QDRANT_BQ_PORT}/", timeout=180)


def stop_qdrant_bq_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", QDRANT_BQ_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
def load_dataset(name: str) -> Tuple[Path, np.ndarray, np.ndarray, Dict[str, Any]]:
    ds_dir = PREP_DIR / name
    queries = np.load(ds_dir / "queries.npy").astype(np.float32)
    gt = np.load(ds_dir / "gt_ip.npy").astype(np.int64)
    meta = json.loads((ds_dir / "meta.json").read_text())
    return ds_dir, queries, gt, meta


def make_splits(
    queries: np.ndarray,
    gt: np.ndarray,
    n_calib: int,
    n_warmup: int,
    n_measure: int,
) -> Dict[str, Any]:
    total = len(queries)
    n_calib = min(n_calib, total)
    n_warmup = min(n_warmup, max(0, total - n_calib))
    calib_slice = slice(0, n_calib)
    warm_slice = slice(n_calib, n_calib + n_warmup)
    pool_idx = np.arange(n_calib + n_warmup, total, dtype=np.int64)
    if len(pool_idx) == 0:
        pool_idx = np.arange(total, dtype=np.int64)
    if len(pool_idx) >= n_measure:
        test_idx = pool_idx[:n_measure]
        unique_test = n_measure
    else:
        test_idx = np.resize(pool_idx, n_measure)
        unique_test = len(pool_idx)
    return {
        "calib": queries[calib_slice],
        "calib_gt": gt[calib_slice],
        "warmup": queries[warm_slice],
        "warmup_gt": gt[warm_slice],
        "test": queries[test_idx],
        "test_gt": gt[test_idx],
        "n_total": total,
        "n_calib": n_calib,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "unique_test": unique_test,
    }


# ---------------------------------------------------------------------------
# OspreyDB HTTP engine
# ---------------------------------------------------------------------------
class OspreyHttpEngine:
    def __init__(
        self,
        runner_bin: str,
        osi_path: Path,
        osv_path: Path,
        bind: str,
        cache_mb: int,
    ):
        self.runner_bin = runner_bin
        self.osi_path = osi_path
        self.osv_path = osv_path
        self.bind = bind
        self.cache_mb = cache_mb
        self.base_url = f"http://{bind}"
        self.proc: Optional[subprocess.Popen[str]] = None
        self.client: Optional[httpx.Client] = None
        self.load_time_s = 0.0

    def start(self) -> float:
        cmd = [
            self.runner_bin,
            "serve",
            "--osi",
            str(self.osi_path),
            "--slabs",
            str(self.osv_path),
            "--bind",
            self.bind,
            "--cache-mb",
            str(self.cache_mb),
        ]
        started = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            wait_http(f"{self.base_url}/health", timeout=300)
        except Exception:
            self.stop()
            raise
        self.load_time_s = time.perf_counter() - started
        self.client = httpx.Client(timeout=120.0)
        return self.load_time_s

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None

    def vmhwm_kb(self) -> int:
        return read_vmhwm_kb(self.proc.pid) if self.proc is not None else 0

    def _payload(
        self, query: np.ndarray, nprobe: int, refine_m: int, top_k: int, refine_multiplier: int
    ) -> Dict[str, Any]:
        return {
            "query": query.tolist(),
            "nprobe": nprobe,
            "refine_m": refine_m,
            "top_k": top_k,
            "refine_multiplier": refine_multiplier,
        }

    def query(
        self,
        query: np.ndarray,
        nprobe: int,
        refine_m: int,
        top_k: int = 10,
        refine_multiplier: int = 0,
    ) -> Tuple[List[int], float, float, Dict[str, Any]]:
        assert self.client is not None
        started = time.perf_counter()
        resp = self.client.post(
            f"{self.base_url}/search",
            json=self._payload(query, nprobe, refine_m, top_k, refine_multiplier),
        )
        client_ms = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Osprey HTTP {resp.status_code} for nprobe={nprobe} "
                f"refine_m={refine_m}: {resp.text}"
            )
        resp.raise_for_status()
        body = resp.json()
        ids = [int(x) for x in body["doc_ids"]]
        return ids, client_ms, float(body["server_ms"]), body

    async def _concurrent(
        self,
        queries: np.ndarray,
        nprobe: int,
        refine_m: int,
        top_k: int,
        refine_multiplier: int,
        workers: int,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        limits = httpx.Limits(
            max_connections=workers, max_keepalive_connections=workers
        )
        results: List[Optional[Tuple[List[int], float, float]]] = [None] * len(queries)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(queries)):
            queue.put_nowait(idx)

        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:

            async def worker() -> None:
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    started = time.perf_counter()
                    resp = await client.post(
                        f"{self.base_url}/search",
                        json=self._payload(
                            queries[idx], nprobe, refine_m, top_k, refine_multiplier
                        ),
                    )
                    client_ms = (time.perf_counter() - started) * 1000.0
                    resp.raise_for_status()
                    body = resp.json()
                    results[idx] = (
                        [int(x) for x in body["doc_ids"]],
                        client_ms,
                        float(body["server_ms"]),
                    )

            started = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(workers)])
            wall_s = time.perf_counter() - started

        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        server_ms = [r[2] for r in results if r is not None]
        return ids, client_ms, server_ms, wall_s

    def concurrent(
        self,
        queries: np.ndarray,
        nprobe: int,
        refine_m: int,
        top_k: int = 10,
        refine_multiplier: int = 0,
        workers: int = 16,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        return asyncio.run(
            self._concurrent(
                queries, nprobe, refine_m, top_k, refine_multiplier, workers
            )
        )


# ---------------------------------------------------------------------------
# Qdrant engine (REST for queries, client for setup)
# ---------------------------------------------------------------------------
class QdrantEngine:
    def __init__(self, url: str, container: str, data_dir: Path, dataset: str):
        self.url = url.rstrip("/")
        self.container = container
        self.data_dir = data_dir
        self.dataset = dataset
        self.client = httpx.Client(timeout=120.0)

    def close(self) -> None:
        self.client.close()

    def vmhwm_kb(self) -> int:
        return container_max_vmhwm_kb(self.container)

    def collection(self, variant: str) -> str:
        return f"ctrl_{self.dataset}_{variant}"

    def _wait_ready(self, client: QdrantClient, coll: str, expected: int) -> None:
        deadline = time.time() + 900
        last = ""
        while time.time() < deadline:
            info = client.get_collection(coll)
            count = client.count(coll, exact=True).count
            opt = str(info.optimizer_status).lower()
            st = str(info.status).lower()
            last = f"status={info.status} optimizer={info.optimizer_status} count={count}/{expected}"
            if (
                count == expected
                and st.endswith("green")
                and "in_progress" not in opt
                and "indexing" not in opt
                and "optimizing" not in opt
            ):
                return
            time.sleep(1.0)
        raise RuntimeError(f"Qdrant collection {coll} not ready: {last}")

    def setup(
        self,
        base: np.ndarray,
        dim: int,
        variant: str,
        bq: bool,
    ) -> Tuple[float, str]:
        client = QdrantClient(url=self.url, timeout=900)
        coll = self.collection(variant)
        for existing in client.get_collections().collections:
            try:
                client.delete_collection(existing.name)
            except Exception:
                pass
        try:
            client.delete_collection(coll)
        except Exception:
            pass
        kwargs: Dict[str, Any] = {}
        if bq:
            kwargs["quantization_config"] = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
            **kwargs,
        )

        started = time.perf_counter()
        total = len(base)
        # Keep each JSON upsert below Qdrant's 32 MiB REST payload limit.
        batch = 10000 if dim <= 256 else 2000
        for start in range(0, total, batch):
            end = min(start + batch, total)
            points = [
                PointStruct(id=int(i), vector=base[i].tolist())
                for i in range(start, end)
            ]
            client.upsert(collection_name=coll, points=points, wait=False)
        self._wait_ready(client, coll, total)
        build_s = time.perf_counter() - started
        try:
            client.close()
        except Exception:
            pass
        return build_s, coll

    def query(
        self,
        coll: str,
        query: np.ndarray,
        ef: int,
        oversampling: Optional[float] = None,
    ) -> Tuple[List[int], float, float]:
        params: Dict[str, Any] = {"hnsw_ef": int(ef)}
        if oversampling is not None:
            params["quantization"] = {
                "rescore": True,
                "oversampling": float(oversampling),
            }
        body = {
            "query": query.tolist(),
            "limit": 10,
            "with_payload": False,
            "params": params,
        }
        started = time.perf_counter()
        resp = self.client.post(
            f"{self.url}/collections/{coll}/points/query", json=body
        )
        client_ms = (time.perf_counter() - started) * 1000.0
        resp.raise_for_status()
        payload = resp.json()
        ids = [int(p["id"]) for p in payload["result"]["points"]]
        server_ms = float(payload.get("time", 0.0)) * 1000.0
        return ids, client_ms, server_ms

    async def _concurrent(
        self,
        coll: str,
        queries: np.ndarray,
        ef: int,
        oversampling: Optional[float],
        workers: int,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        params: Dict[str, Any] = {"hnsw_ef": int(ef)}
        if oversampling is not None:
            params["quantization"] = {
                "rescore": True,
                "oversampling": float(oversampling),
            }
        limits = httpx.Limits(
            max_connections=workers, max_keepalive_connections=workers
        )
        results: List[Optional[Tuple[List[int], float, float]]] = [None] * len(queries)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(queries)):
            queue.put_nowait(idx)

        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:

            async def worker() -> None:
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    body = {
                        "query": queries[idx].tolist(),
                        "limit": 10,
                        "with_payload": False,
                        "params": params,
                    }
                    started = time.perf_counter()
                    resp = await client.post(
                        f"{self.url}/collections/{coll}/points/query", json=body
                    )
                    client_ms = (time.perf_counter() - started) * 1000.0
                    resp.raise_for_status()
                    payload = resp.json()
                    results[idx] = (
                        [int(p["id"]) for p in payload["result"]["points"]],
                        client_ms,
                        float(payload.get("time", 0.0)) * 1000.0,
                    )

            started = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(workers)])
            wall_s = time.perf_counter() - started

        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        server_ms = [r[2] for r in results if r is not None]
        return ids, client_ms, server_ms, wall_s

    def concurrent(
        self,
        coll: str,
        queries: np.ndarray,
        ef: int,
        oversampling: Optional[float],
        workers: int = 16,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        return asyncio.run(
            self._concurrent(coll, queries, ef, oversampling, workers)
        )

    def disk_mb(self, coll: str) -> float:
        return dir_size_mb(self.data_dir / "collections" / coll)


# ---------------------------------------------------------------------------
# pgvector engine
# ---------------------------------------------------------------------------
class PgvectorEngine:
    def __init__(self, dataset: str, dim: int):
        self.dataset = dataset
        self.dim = dim
        self.table = f"ctrl_vecs_{dataset}"
        self._ef_by_conn: Dict[int, int] = {}

    def vmhwm_kb(self) -> int:
        return container_max_vmhwm_kb(PGVECTOR_CONTAINER)

    @staticmethod
    def vector_literal(vec: np.ndarray) -> str:
        return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"

    def setup(self, base: np.ndarray) -> float:
        started = time.perf_counter()
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"DROP TABLE IF EXISTS {self.table};")
                cur.execute(
                    f"CREATE TABLE {self.table} (id BIGINT PRIMARY KEY, emb vector({self.dim}));"
                )
                cur.execute("SET maintenance_work_mem = '1GB';")
            with conn.cursor() as cur:
                with cur.copy(f"COPY {self.table} (id, emb) FROM STDIN") as copy:
                    for i in range(len(base)):
                        copy.write_row(
                            (i, self.vector_literal(np.asarray(base[i], dtype=np.float32)))
                        )
                cur.execute(
                    f"CREATE INDEX {self.table}_hnsw_idx ON {self.table} "
                    "USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 100);"
                )
        return time.perf_counter() - started

    def set_ef(self, conn: psycopg.Connection, ef: int) -> None:
        key = id(conn)
        if self._ef_by_conn.get(key) != int(ef):
            conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            self._ef_by_conn[key] = int(ef)

    def query(
        self, conn: psycopg.Connection, q_literal: str, ef: int
    ) -> Tuple[List[int], float, None]:
        self.set_ef(conn, ef)
        started = time.perf_counter()
        cur = conn.execute(
            f"SELECT id FROM {self.table} ORDER BY emb <=> %s::vector LIMIT 10;",
            (q_literal,),
        )
        rows = cur.fetchall()
        client_ms = (time.perf_counter() - started) * 1000.0
        return [int(r[0]) for r in rows], client_ms, None

    async def _concurrent(
        self,
        q_literals: Sequence[str],
        ef: int,
        workers: int,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        results: List[Optional[Tuple[List[int], float]]] = [None] * len(q_literals)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(q_literals)):
            queue.put_nowait(idx)

        async def worker() -> None:
            conn = await psycopg.AsyncConnection.connect(PGVECTOR_DSN)
            try:
                await conn.execute(f"SET hnsw.ef_search = {int(ef)}")
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    started = time.perf_counter()
                    cur = await conn.execute(
                        f"SELECT id FROM {self.table} ORDER BY emb <=> %s::vector LIMIT 10;",
                        (q_literals[idx],),
                    )
                    rows = await cur.fetchall()
                    client_ms = (time.perf_counter() - started) * 1000.0
                    results[idx] = ([int(r[0]) for r in rows], client_ms)
            finally:
                await conn.close()

        started = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(workers)])
        wall_s = time.perf_counter() - started
        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        return ids, client_ms, [], wall_s

    def concurrent(
        self, q_literals: Sequence[str], ef: int, workers: int = 16
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        return asyncio.run(self._concurrent(q_literals, ef, workers))

    def server_latency_sample(
        self, q_literals: Sequence[str], ef: int, sample: int = 100
    ) -> Tuple[float, float, float]:
        values: List[float] = []
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
            self.set_ef(conn, ef)
            for q_literal in list(q_literals)[:sample]:
                cur = conn.execute(
                    f"EXPLAIN (ANALYZE, FORMAT JSON) "
                    f"SELECT id FROM {self.table} ORDER BY emb <=> %s::vector LIMIT 10;",
                    (q_literal,),
                )
                row = cur.fetchone()
                payload = row[0] if row else None
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, list) and payload:
                    values.append(float(payload[0].get("Execution Time", 0.0)))
        return latency_stats(values)

    def disk_mb(self) -> float:
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
            cur = conn.execute(
                f"SELECT pg_total_relation_size('{self.table}'), "
                f"pg_relation_size('{self.table}_hnsw_idx');"
            )
            row = cur.fetchone()
            if row:
                return (int(row[0]) + int(row[1])) / (1024.0 * 1024.0)
        return 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def select_for_targets(
    results: Sequence[Dict[str, Any]],
    targets: Sequence[float],
    cost_fn: Callable[[Dict[str, Any]], Tuple[float, ...]],
) -> Dict[float, Dict[str, Any]]:
    selected: Dict[float, Dict[str, Any]] = {}
    for target in targets:
        candidates = [r for r in results if r["recall"] >= target - 0.005]
        if candidates:
            best = min(candidates, key=cost_fn)
            selected[target] = {**best, "status": "ok"}
        else:
            best = max(results, key=lambda r: r["recall"])
            selected[target] = {**best, "status": "below_target"}
    return selected


def calibrate_osprey(
    engine: OspreyHttpEngine,
    queries: np.ndarray,
    gt: np.ndarray,
    nprobe_grid: Sequence[int],
    refine_grid: Sequence[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for nprobe in nprobe_grid:
        for refine_m in refine_grid:
            ids: List[List[int]] = []
            latencies: List[float] = []
            meta: Dict[str, Any] = {}
            for query in queries:
                found, client_ms, _server_ms, meta = engine.query(
                    query, nprobe, refine_m, 10, 0
                )
                ids.append(found)
                latencies.append(client_ms)
            p50, _, _ = latency_stats(latencies)
            recall = recall_at_k(ids, gt)
            print(
                f"  [calib][osprey] nprobe={nprobe} refine={refine_m} "
                f"effective={int(meta.get('effective_nprobe', 0))}/"
                f"{int(meta.get('effective_candidates', 0))} "
                f"recall={recall:.4f} p50={p50:.3f}ms",
                flush=True,
            )
            results.append(
                {
                    "nprobe": nprobe,
                    "refine_m": refine_m,
                    "effective_nprobe": int(meta.get("effective_nprobe", 0)),
                    "effective_candidates": int(meta.get("effective_candidates", 0)),
                    "recall": recall,
                    "p50_ms": p50,
                }
            )
    return results


def calibrate_qdrant(
    engine: QdrantEngine,
    coll: str,
    queries: np.ndarray,
    gt: np.ndarray,
    ef_grid: Sequence[int],
    oversampling_grid: Sequence[float],
    bq: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for ef in ef_grid:
        values = oversampling_grid if bq else [None]
        for oversampling in values:
            ids: List[List[int]] = []
            latencies: List[float] = []
            for query in queries:
                found, client_ms, _server_ms = engine.query(
                    coll, query, ef, oversampling
                )
                ids.append(found)
                latencies.append(client_ms)
            p50, _, _ = latency_stats(latencies)
            recall = recall_at_k(ids, gt)
            print(
                f"  [calib][qdrant] ef={ef} oversampling={oversampling} "
                f"recall={recall:.4f} p50={p50:.3f}ms",
                flush=True,
            )
            results.append(
                {
                    "hnsw_ef": ef,
                    "oversampling": oversampling,
                    "recall": recall,
                    "p50_ms": p50,
                }
            )
    return results


def calibrate_pgvector(
    engine: PgvectorEngine,
    q_literals: Sequence[str],
    gt: np.ndarray,
    ef_grid: Sequence[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        for ef in ef_grid:
            ids: List[List[int]] = []
            latencies: List[float] = []
            for q_literal in q_literals:
                found, client_ms, _server_ms = engine.query(conn, q_literal, ef)
                ids.append(found)
                latencies.append(client_ms)
            p50, _, _ = latency_stats(latencies)
            recall = recall_at_k(ids, gt)
            print(
                f"  [calib][pgvector] ef={ef} recall={recall:.4f} p50={p50:.3f}ms",
                flush=True,
            )
            results.append(
                {"hnsw_ef_search": ef, "recall": recall, "p50_ms": p50}
            )
    return results


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_generic(
    selected: Dict[float, Dict[str, Any]],
    warmup_queries: np.ndarray,
    test_queries: np.ndarray,
    test_gt: np.ndarray,
    rounds: int,
    workers: int,
    sequential_fn: Callable[[Any, Dict[str, Any]], Tuple[List[int], float, Optional[float]]],
    warmup_fn: Callable[[Any, Dict[str, Any]], Any],
    concurrent_fn: Callable[
        [Sequence[Any], Dict[str, Any], int],
        Tuple[List[List[int]], List[float], List[float], float],
    ],
    prepared_queries: Optional[Sequence[Any]] = None,
    server_latency_fn: Optional[Callable[[Dict[str, Any]], Tuple[float, float, float]]] = None,
) -> Dict[float, Dict[str, Any]]:
    measured: Dict[float, Dict[str, Any]] = {}
    for target, cfg in selected.items():
        for query in warmup_queries:
            warmup_fn(query, cfg)

        round_stats: List[Dict[str, Any]] = []
        for round_idx in range(rounds):
            print(
                f"  [measure] target={target:.3f} round={round_idx + 1}/{rounds} "
                f"queries={len(prepared_queries if prepared_queries is not None else test_queries)}",
                flush=True,
            )
            ids: List[List[int]] = []
            latencies: List[float] = []
            server_latencies: List[float] = []
            source = prepared_queries if prepared_queries is not None else test_queries
            for item in source:
                found, client_ms, server_ms = sequential_fn(item, cfg)
                ids.append(found)
                latencies.append(client_ms)
                if server_ms is not None:
                    server_latencies.append(server_ms)
            p50, p95, p99 = latency_stats(latencies)
            s50, s95, s99 = latency_stats(server_latencies)

            c_ids, c_latencies, c_server, wall_s = concurrent_fn(
                list(source), cfg, workers
            )
            c50, c95, c99 = latency_stats(c_latencies)
            cs50, cs95, cs99 = latency_stats(c_server)
            qps = (len(source) / wall_s) if wall_s > 0 else 0.0

            round_recall = recall_at_k(ids, test_gt)
            round_concurrent_recall = recall_at_k(c_ids, test_gt)
            print(
                f"  [measure] target={target:.3f} round={round_idx + 1}/{rounds} "
                f"recall={round_recall:.4f} concurrent_recall={round_concurrent_recall:.4f} "
                f"p50={p50:.3f}ms qps={qps:.1f}",
                flush=True,
            )
            round_stats.append(
                {
                    "recall": round_recall,
                    "concurrent_recall": round_concurrent_recall,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "server_p50": s50,
                    "server_p95": s95,
                    "server_p99": s99,
                    "concurrent_p50": c50,
                    "concurrent_p95": c95,
                    "concurrent_p99": c99,
                    "concurrent_server_p50": cs50,
                    "concurrent_server_p95": cs95,
                    "concurrent_server_p99": cs99,
                    "qps": qps,
                }
            )

        def med_std(key: str) -> Tuple[float, float]:
            return mean_std([r[key] for r in round_stats])

        summary: Dict[str, Any] = {"rounds": round_stats}
        for key in (
            "recall",
            "concurrent_recall",
            "p50",
            "p95",
            "p99",
            "concurrent_p50",
            "concurrent_p95",
            "concurrent_p99",
            "qps",
        ):
            summary[f"{key}_median"], summary[f"{key}_std"] = med_std(key)
        for key in ("server_p50", "server_p95", "server_p99"):
            summary[f"{key}_median"], summary[f"{key}_std"] = med_std(key)
        for key in (
            "concurrent_server_p50",
            "concurrent_server_p95",
            "concurrent_server_p99",
        ):
            summary[f"{key}_median"], summary[f"{key}_std"] = med_std(key)
        if server_latency_fn is not None:
            s50, s95, s99 = server_latency_fn(cfg)
            summary["server_p50_median"] = s50
            summary["server_p95_median"] = s95
            summary["server_p99_median"] = s99
            summary["server_p50_std"] = 0.0
            summary["server_p95_std"] = 0.0
            summary["server_p99_std"] = 0.0
        measured[target] = summary
    return measured


def build_row(
    dataset: str,
    meta: Dict[str, Any],
    splits: Dict[str, Any],
    engine: str,
    variant: str,
    transport: str,
    target: float,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    load_peak_kb: int,
    query_peak_kb: int,
    build_peak_kb: int,
    build_time_s: float,
    load_time_s: float,
    disk_mb: float,
    rounds: int,
    workers: int,
    notes: str,
) -> Dict[str, Any]:
    public_config = {k: v for k, v in config.items() if k not in ("recall", "p50_ms", "status")}
    return {
        "dataset": dataset,
        "dataset_size": int(meta.get("num_vectors", 0)),
        "dim": int(meta.get("dim", 0)),
        "engine": engine,
        "variant": variant,
        "transport": transport,
        "recall_target": target,
        "calibration_status": config.get("status", ""),
        "calibration_params": json.dumps(public_config, sort_keys=True),
        "validation_recall": round(float(config.get("recall", 0.0)), 4),
        "test_recall": round(float(stats.get("recall_median", 0.0)), 4),
        "test_recall_std": round(float(stats.get("recall_std", 0.0)), 4),
        "p50_ms_median": round(float(stats.get("p50_median", 0.0)), 3),
        "p50_ms_std": round(float(stats.get("p50_std", 0.0)), 3),
        "p95_ms_median": round(float(stats.get("p95_median", 0.0)), 3),
        "p95_ms_std": round(float(stats.get("p95_std", 0.0)), 3),
        "p99_ms_median": round(float(stats.get("p99_median", 0.0)), 3),
        "p99_ms_std": round(float(stats.get("p99_std", 0.0)), 3),
        "server_p50_ms_median": round(float(stats.get("server_p50_median", 0.0)), 3),
        "server_p95_ms_median": round(float(stats.get("server_p95_median", 0.0)), 3),
        "server_p99_ms_median": round(float(stats.get("server_p99_median", 0.0)), 3),
        "qps_concurrent_median": round(float(stats.get("qps_median", 0.0)), 1),
        "qps_concurrent_std": round(float(stats.get("qps_std", 0.0)), 1),
        "concurrent_recall": round(float(stats.get("concurrent_recall_median", 0.0)), 4),
        "load_peak_rss_mb": round(load_peak_kb / 1024.0, 1),
        "query_peak_rss_mb": round(query_peak_kb / 1024.0, 1),
        "build_peak_rss_mb": round(build_peak_kb / 1024.0, 1),
        "build_time_s": round(build_time_s, 3),
        "load_time_s": round(load_time_s, 3),
        "disk_size_mb": round(disk_mb, 2),
        "queries_measured": int(splits["n_measure"]),
        "unique_test_queries": int(splits["unique_test"]),
        "warmup_queries": int(splits["n_warmup"]),
        "validation_queries": int(splits["n_calib"]),
        "rounds": rounds,
        "workers": workers,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Top-level controlled driver
# ---------------------------------------------------------------------------
def run_controlled(args: Any) -> int:
    targets = [float(t) for t in args.recall_target]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    rounds = int(args.rounds)
    workers = int(args.workers)
    n_calib = 200
    n_warmup = int(args.warmup_queries)
    n_measure = int(args.measure_queries)
    if args.quick:
        n_calib = min(40, n_calib)
        n_warmup = min(20, n_warmup)
        n_measure = min(40, n_measure)
        rounds = min(1, rounds)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_compose_services()

    nprobe_grid = [8, 16, 32, 64] if not args.quick else [16, 32]
    refine_grid = (
        [300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000]
        if not args.quick
        else [500, 1200]
    )
    ef_grid = (
        [16, 32, 64, 96, 128, 192, 256, 384, 512]
        if not args.quick
        else [64, 128]
    )
    oversampling_grid = [1.0, 1.5, 2.0, 3.0] if not args.quick else [1.0, 2.0]

    rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        ds_dir, queries, gt, meta = load_dataset(dataset)
        base = np.load(ds_dir / "base.npy", mmap_mode="r")
        dim = int(meta["dim"])
        low_dim = dim <= 256
        max_stage1 = 4096
        refine_grid_ds = [
            r
            for r in refine_grid
            if r * (2 if low_dim else 1) <= max_stage1
        ]
        if len(refine_grid_ds) != len(refine_grid):
            print(
                f"[controlled] {dataset}: refine grid filtered to {refine_grid_ds} "
                f"(Stage-1 cap {max_stage1}, low_dim={low_dim})"
            )
        splits = make_splits(
            queries, gt, n_calib=n_calib, n_warmup=n_warmup, n_measure=n_measure
        )
        print(
            f"[controlled] {dataset}: {meta.get('num_vectors')} x {dim}d, "
            f"queries={splits['n_total']} (calib={splits['n_calib']}, "
            f"warmup={splits['n_warmup']}, measure={splits['n_measure']}, "
            f"unique_test={splits['unique_test']})"
        )

        if "osprey" in engines:
            stop_all_engines()
            print(f"[controlled] {dataset}: building OspreyDB index")
            art_dir = ARTIFACTS_DIR / dataset
            art_dir.mkdir(parents=True, exist_ok=True)
            build_s, build_peak_mb = measure_cmd_peak_rss(
                [
                    args.osprey_bin,
                    "import",
                    "--input",
                    str(ds_dir / "base.parquet"),
                    "--output",
                    str(art_dir),
                    "--id-col",
                    "doc_id",
                    "--vec-col",
                    "embedding",
                    "--clusters",
                    str(args.clusters),
                    "--json",
                ]
            )
            bind = f"127.0.0.1:{args.osprey_port}"
            engine = OspreyHttpEngine(
                args.runner_bin,
                art_dir / "segment.osi",
                art_dir / "segment.osv",
                bind,
                args.controlled_cache_mb,
            )
            print(f"[controlled] {dataset}: starting OspreyDB HTTP server on {bind}")
            load_s = engine.start()
            load_peak_kb = engine.vmhwm_kb()
            sampler = RssSampler(engine.vmhwm_kb, interval=0.25)
            sampler.start()
            try:
                print(f"[controlled] {dataset}: calibrating OspreyDB")
                calib = calibrate_osprey(
                    engine,
                    splits["calib"],
                    splits["calib_gt"],
                    nprobe_grid,
                    refine_grid_ds,
                )
                selected = select_for_targets(
                    calib,
                    targets,
                    cost_fn=lambda r: (
                        r["effective_candidates"],
                        r["effective_nprobe"],
                        r["p50_ms"],
                    ),
                )
                for target, cfg in selected.items():
                    print(
                        f"  target={target:.3f} -> requested {cfg['nprobe']}/"
                        f"{cfg['refine_m']} effective {cfg['effective_nprobe']}/"
                        f"{cfg['effective_candidates']} val_recall={cfg['recall']:.4f} "
                        f"status={cfg['status']}"
                    )

                def seq(item: np.ndarray, cfg: Dict[str, Any]):
                    ids, cms, sms, _meta = engine.query(
                        item, cfg["nprobe"], cfg["refine_m"], 10, 0
                    )
                    return ids, cms, sms

                def warm(item: np.ndarray, cfg: Dict[str, Any]):
                    engine.query(item, cfg["nprobe"], cfg["refine_m"], 10, 0)

                def conc(
                    items: Sequence[np.ndarray], cfg: Dict[str, Any], nworkers: int
                ):
                    return engine.concurrent(
                        np.asarray(items),
                        cfg["nprobe"],
                        cfg["refine_m"],
                        10,
                        0,
                        nworkers,
                    )

                measured = measure_generic(
                    selected,
                    splits["warmup"],
                    splits["test"],
                    splits["test_gt"],
                    rounds,
                    workers,
                    sequential_fn=seq,
                    warmup_fn=warm,
                    concurrent_fn=conc,
                )
            finally:
                query_peak_kb = sampler.stop()
                engine.stop()

            for target, stats in measured.items():
                rows.append(
                    build_row(
                        dataset,
                        meta,
                        splits,
                        "OspreyDB (HTTP, 1-bit .osi + f16 .osv)",
                        "osprey_http",
                        "HTTP localhost",
                        target,
                        selected[target],
                        stats,
                        load_peak_kb,
                        query_peak_kb,
                        int(build_peak_mb * 1024),
                        build_s,
                        load_s,
                        (art_dir / "segment.osi").stat().st_size
                        / (1024.0 * 1024.0)
                        + (art_dir / "segment.osv").stat().st_size
                        / (1024.0 * 1024.0),
                        rounds,
                        workers,
                        "HTTP server in osprey-bench-runner; client and server share localhost.",
                    )
                )

        if "qdrant_fp32" in engines:
            stop_all_engines()
            print(f"[controlled] {dataset}: starting Qdrant FP32 and building index")
            start_qdrant_fp32()
            fp32 = QdrantEngine(
                QDRANT_FP32_URL, "osprey-bench-qdrant", ROOT / "docker" / "qdrant-data", dataset
            )
            sampler = RssSampler(fp32.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                build_s, coll = fp32.setup(base, dim, "fp32", bq=False)
            finally:
                build_peak_kb = sampler.stop()
            load_peak_kb = fp32.vmhwm_kb()
            sampler = RssSampler(fp32.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                print(f"[controlled] {dataset}: calibrating Qdrant FP32")
                calib = calibrate_qdrant(
                    fp32, coll, splits["calib"], splits["calib_gt"], ef_grid, [None], False
                )
                selected = select_for_targets(
                    calib, targets, cost_fn=lambda r: (r["hnsw_ef"], r["p50_ms"])
                )
                for target, cfg in selected.items():
                    print(
                        f"  target={target:.3f} -> hnsw_ef={cfg['hnsw_ef']} "
                        f"val_recall={cfg['recall']:.4f} status={cfg['status']}"
                    )

                def seq(item: np.ndarray, cfg: Dict[str, Any]):
                    return fp32.query(coll, item, cfg["hnsw_ef"], None)

                def warm(item: np.ndarray, cfg: Dict[str, Any]):
                    fp32.query(coll, item, cfg["hnsw_ef"], None)

                def conc(
                    items: Sequence[np.ndarray], cfg: Dict[str, Any], nworkers: int
                ):
                    return fp32.concurrent(
                        coll, np.asarray(items), cfg["hnsw_ef"], None, nworkers
                    )

                measured = measure_generic(
                    selected,
                    splits["warmup"],
                    splits["test"],
                    splits["test_gt"],
                    rounds,
                    workers,
                    sequential_fn=seq,
                    warmup_fn=warm,
                    concurrent_fn=conc,
                )
            finally:
                query_peak_kb = sampler.stop()
            disk_mb = fp32.disk_mb(coll)
            fp32.close()
            stop_qdrant_fp32()
            for target, stats in measured.items():
                rows.append(
                    build_row(
                        dataset,
                        meta,
                        splits,
                        "Qdrant (HNSW FP32)",
                        "qdrant_fp32",
                        "HTTP localhost",
                        target,
                        selected[target],
                        stats,
                        load_peak_kb,
                        query_peak_kb,
                        build_peak_kb,
                        build_s,
                        build_s,
                        disk_mb,
                        rounds,
                        workers,
                        "Qdrant compose container; HNSW m=16 ef_construct=100.",
                    )
                )

        if "qdrant_bq" in engines:
            stop_all_engines()
            print(f"[controlled] {dataset}: starting dedicated Qdrant BQ and building index")
            start_qdrant_bq_container()
            bq_url = f"http://127.0.0.1:{QDRANT_BQ_PORT}"
            bq = QdrantEngine(bq_url, QDRANT_BQ_NAME, QDRANT_BQ_DATA, dataset)
            sampler = RssSampler(bq.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                build_s, coll = bq.setup(base, dim, "bq", bq=True)
            finally:
                build_peak_kb = sampler.stop()
            load_peak_kb = bq.vmhwm_kb()
            sampler = RssSampler(bq.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                print(f"[controlled] {dataset}: calibrating Qdrant BQ")
                calib = calibrate_qdrant(
                    bq,
                    coll,
                    splits["calib"],
                    splits["calib_gt"],
                    ef_grid,
                    oversampling_grid,
                    True,
                )
                selected = select_for_targets(
                    calib,
                    targets,
                    cost_fn=lambda r: (
                        r["hnsw_ef"] * float(r["oversampling"] or 1.0),
                        r["hnsw_ef"],
                        r["p50_ms"],
                    ),
                )
                for target, cfg in selected.items():
                    print(
                        f"  target={target:.3f} -> hnsw_ef={cfg['hnsw_ef']} "
                        f"oversampling={cfg['oversampling']} val_recall={cfg['recall']:.4f} "
                        f"status={cfg['status']}"
                    )

                def seq(item: np.ndarray, cfg: Dict[str, Any]):
                    return bq.query(coll, item, cfg["hnsw_ef"], cfg["oversampling"])

                def warm(item: np.ndarray, cfg: Dict[str, Any]):
                    bq.query(coll, item, cfg["hnsw_ef"], cfg["oversampling"])

                def conc(
                    items: Sequence[np.ndarray], cfg: Dict[str, Any], nworkers: int
                ):
                    return bq.concurrent(
                        coll,
                        np.asarray(items),
                        cfg["hnsw_ef"],
                        cfg["oversampling"],
                        nworkers,
                    )

                measured = measure_generic(
                    selected,
                    splits["warmup"],
                    splits["test"],
                    splits["test_gt"],
                    rounds,
                    workers,
                    sequential_fn=seq,
                    warmup_fn=warm,
                    concurrent_fn=conc,
                )
            finally:
                query_peak_kb = sampler.stop()
            disk_mb = bq.disk_mb(coll)
            bq.close()
            stop_qdrant_bq_container()
            for target, stats in measured.items():
                rows.append(
                    build_row(
                        dataset,
                        meta,
                        splits,
                        "Qdrant (HNSW 1-bit BQ + rescore)",
                        "qdrant_bq",
                        "HTTP localhost",
                        target,
                        selected[target],
                        stats,
                        load_peak_kb,
                        query_peak_kb,
                        build_peak_kb,
                        build_s,
                        build_s,
                        disk_mb,
                        rounds,
                        workers,
                        "Dedicated Qdrant container; binary quantization always_ram, rescore=true.",
                    )
                )

        if "pgvector" in engines:
            stop_all_engines()
            print(f"[controlled] {dataset}: starting pgvector and building index")
            start_pgvector()
            try:
                clear_pg_tables()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                print(f"  [warn] could not clear pgvector tables: {exc}")
            pg = PgvectorEngine(dataset, dim)
            sampler = RssSampler(pg.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                build_s = pg.setup(base)
            finally:
                build_peak_kb = sampler.stop()
            load_peak_kb = pg.vmhwm_kb()
            calib_q = [pg.vector_literal(q) for q in splits["calib"]]
            warm_q = [pg.vector_literal(q) for q in splits["warmup"]]
            test_q = [pg.vector_literal(q) for q in splits["test"]]
            sampler = RssSampler(pg.vmhwm_kb, interval=0.5)
            sampler.start()
            try:
                print(f"[controlled] {dataset}: calibrating pgvector")
                calib = calibrate_pgvector(pg, calib_q, splits["calib_gt"], ef_grid)
                selected = select_for_targets(
                    calib, targets, cost_fn=lambda r: (r["hnsw_ef_search"], r["p50_ms"])
                )
                for target, cfg in selected.items():
                    print(
                        f"  target={target:.3f} -> hnsw.ef_search={cfg['hnsw_ef_search']} "
                        f"val_recall={cfg['recall']:.4f} status={cfg['status']}"
                    )

                with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:

                    def seq(item: str, cfg: Dict[str, Any]):
                        return pg.query(conn, item, cfg["hnsw_ef_search"])

                    def warm(item: str, cfg: Dict[str, Any]):
                        pg.query(conn, item, cfg["hnsw_ef_search"])

                    def conc(
                        items: Sequence[str], cfg: Dict[str, Any], nworkers: int
                    ):
                        return pg.concurrent(
                            list(items), cfg["hnsw_ef_search"], nworkers
                        )

                    measured = measure_generic(
                        selected,
                        warm_q,
                        test_q,
                        splits["test_gt"],
                        rounds,
                        workers,
                        sequential_fn=seq,
                        warmup_fn=warm,
                        concurrent_fn=conc,
                        prepared_queries=test_q,
                        server_latency_fn=lambda cfg: pg.server_latency_sample(
                            test_q, cfg["hnsw_ef_search"], sample=100
                        ),
                    )
            finally:
                query_peak_kb = sampler.stop()
            disk_mb = pg.disk_mb()
            stop_pgvector()
            for target, stats in measured.items():
                rows.append(
                    build_row(
                        dataset,
                        meta,
                        splits,
                        "pgvector (HNSW FP32)",
                        "pgvector",
                        "SQL localhost",
                        target,
                        selected[target],
                        stats,
                        load_peak_kb,
                        query_peak_kb,
                        build_peak_kb,
                        build_s,
                        build_s,
                        disk_mb,
                        rounds,
                        workers,
                        "PostgreSQL container; HNSW m=16 ef_construction=100; server time via EXPLAIN ANALYZE (sample=100).",
                    )
                )

        stop_all_engines()

    out_csv = Path(args.out_controlled_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[controlled] saved {out_csv} ({len(rows)} rows)")
    return 0
