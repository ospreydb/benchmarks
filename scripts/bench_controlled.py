#!/usr/bin/env python3
"""
Controlled, recall-matched benchmark protocol for OspreyDB vs Qdrant vs pgvector vs FAISS.

Key properties:
  * Calibrates each engine on a validation split requiring ``recall >= recall_target``
    (or reports ``below_target`` with the highest-recall grid point).
  * Uses HTTP/1.1 keep-alive client-server transport via ``osprey-bench-runner http-load``
    for OspreyDB (``serve``) and Qdrant (FP32, SQ8, BQ), SQL client-server via ``psycopg``
    for pgvector (FP32, halfvec), and optional in-process FAISS IVF-Flat.
  * Reports client round-trip latency, server-side latency, concurrent QPS,
    query-only cgroup v2 ``memory.peak`` (RSS + Linux page cache) alongside process
    ``VmHWM``, load peak, build peak, separated ``ingest_time_s`` / ``index_build_time_s``,
    and allocated disk size.
  * Runs configurable warmup and measured query splits across independent rounds,
    reporting median and std of P50/P95/P99 and 95% bootstrap confidence intervals
    for ``test_recall``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

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
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

try:
    from osprey_docker_engine import (
        OSPREY_DOCKER_CONTAINER,
        OSPREY_DOCKER_PORT,
        OspreyDockerServerEngine,
        container_max_vmhwm_kb,
        container_peak_memory_kb,
        container_pids,
        dir_size_mb,
        drop_docker_dir_page_cache,
        read_vmhwm_kb,
        run_rust_http_load,
        start_container,
        stop_container,
        stop_osprey_docker_container,
        wait_http,
    )
except ImportError:
    from scripts.osprey_docker_engine import (
        OSPREY_DOCKER_CONTAINER,
        OSPREY_DOCKER_PORT,
        OspreyDockerServerEngine,
        container_max_vmhwm_kb,
        container_peak_memory_kb,
        container_pids,
        dir_size_mb,
        drop_docker_dir_page_cache,
        read_vmhwm_kb,
        run_rust_http_load,
        start_container,
        stop_container,
        stop_osprey_docker_container,
        wait_http,
    )

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts" / "controlled"
RESULTS_DIR = ROOT / "results"
COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"
QDRANT_IMAGE = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.20.2")
PGVECTOR_IMAGE = os.environ.get("PGVECTOR_IMAGE", "pgvector/pgvector:0.8.7-pg16")

QDRANT_FP32_CONTAINER = "osprey-bench-qdrant"
QDRANT_FP32_URL = "http://127.0.0.1:6333"
QDRANT_FP32_DATA = ROOT / "docker" / "qdrant-data"
QDRANT_BQ_NAME = "osprey-bench-qdrant-bq"
QDRANT_BQ_PORT = 6335
QDRANT_BQ_DATA = ROOT / "docker" / "qdrant-bq-data"
QDRANT_SQ8_NAME = "osprey-bench-qdrant-sq8"
QDRANT_SQ8_PORT = 6337
QDRANT_SQ8_DATA = ROOT / "docker" / "qdrant-sq8-data"
PGVECTOR_CONTAINER = "osprey-bench-pgvector"
PGVECTOR_DATA = ROOT / "docker" / "pgvector-data"
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


def median_std(vals: Sequence[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    return float(statistics.median(vals)), float(statistics.pstdev(vals))


def recall_at_k(pred_ids: Sequence[Sequence[int]], gt: np.ndarray, k: int = 10) -> float:
    hits = 0
    denom = 0
    for pred, row in zip(pred_ids, gt):
        gt_set = {int(x) for x in row[:k]}
        hits += len({int(x) for x in pred[:k]} & gt_set)
        denom += k
    return (hits / denom) if denom else 0.0


def bootstrap_recall_ci95(
    pred_ids: Sequence[Sequence[int]],
    gt: np.ndarray,
    k: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute a deterministic 95% bootstrap confidence interval across test queries."""
    if len(pred_ids) == 0 or len(gt) == 0:
        return 0.0, 0.0
    per_q = np.array(
        [
            len({int(x) for x in pred[:k]} & {int(x) for x in row[:k]}) / float(k)
            for pred, row in zip(pred_ids, gt)
        ],
        dtype=np.float64,
    )
    n = len(per_q)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    chunk = 500
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        boot_means[start : start + b] = per_q[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


_SYSTEMD_SCOPE_OK: Optional[bool] = None


def has_systemd_user_scope() -> bool:
    global _SYSTEMD_SCOPE_OK
    if _SYSTEMD_SCOPE_OK is None:
        try:
            res = subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--scope",
                    "--quiet",
                    "-p",
                    "MemoryAccounting=yes",
                    "true",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            _SYSTEMD_SCOPE_OK = res.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _SYSTEMD_SCOPE_OK = False
    return _SYSTEMD_SCOPE_OK


def _find_unit_scope_peak_path(unit: str) -> Optional[Path]:
    """Locate `/sys/fs/cgroup/.../{unit}.scope/memory.peak` if systemd has created it."""
    uid = os.getuid()
    direct = Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope/memory.peak"
    )
    if direct.exists():
        return direct
    user_slice = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice")
    if user_slice.exists():
        for match in user_slice.glob(f"**/{unit}.scope/memory.peak"):
            return match
    return None


def read_pid_cgroup_peak_kb(pid: int, expected_unit: Optional[str] = None) -> int:
    """Read cgroup v2 memory.peak (RSS + page cache) for a live process PID.

    When ``expected_unit`` is given, refuses to read any cgroup whose path does
    not belong to ``{expected_unit}.scope``, preventing the systemd-run race
    where ``proc.pid`` still sits in the caller's session cgroup.
    """
    if expected_unit:
        direct_peak = _find_unit_scope_peak_path(expected_unit)
        if direct_peak is not None:
            try:
                return int(direct_peak.read_text().strip()) // 1024
            except (OSError, ValueError):
                pass
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[2]:
                cgroup_rel = parts[2]
                if expected_unit and f"{expected_unit}.scope" not in cgroup_rel:
                    return 0
                peak_file = Path(f"/sys/fs/cgroup{cgroup_rel}/memory.peak")
                if peak_file.exists():
                    return int(peak_file.read_text().strip()) // 1024
    except (OSError, ValueError):
        pass
    return 0


def drop_path_page_cache(target: Path) -> None:
    """Evict readable regular files under `target` from the Linux host page cache."""
    if not target.exists():
        return
    files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as exc:
            print(
                f"  [warn] failed to evict page cache for {path}: {exc}",
                flush=True,
            )


def runner_supports_pg_load(runner_bin: str) -> bool:
    """Probe whether `runner_bin` supports the `pg-load` subcommand."""
    if not runner_bin:
        return False
    try:
        proc = subprocess.run(
            [runner_bin, "pg-load", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def run_rust_pg_load(
    runner_bin: str,
    dsn: str,
    table: str,
    col_type: str,
    ef_search: int,
    q_literals: Sequence[str],
    workers: int,
    warmup_q_literals: Optional[Sequence[str]] = None,
    limit: int = 10,
) -> Tuple[List[List[int]], List[float], List[float], float]:
    """Run sequential (workers=1) or concurrent (workers=N) pgvector SQL load via Rust."""
    with tempfile.TemporaryDirectory(prefix="osprey_pg_load_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload_path = tmp_path / "pg_load_payloads.txt"
        out_path = tmp_path / "pg_load_out.json"
        payload_path.write_text(
            "".join(f"{q}\n" for q in q_literals), encoding="utf-8"
        )
        cmd = [
            runner_bin,
            "pg-load",
            "--dsn",
            dsn,
            "--table",
            table,
            "--col-type",
            col_type,
            "--ef-search",
            str(int(ef_search)),
            "--limit",
            str(int(limit)),
            "--payloads-file",
            str(payload_path),
            "--workers",
            str(int(workers)),
            "--out-json",
            str(out_path),
        ]
        if warmup_q_literals:
            warmup_path = tmp_path / "pg_load_warmup.txt"
            warmup_path.write_text(
                "".join(f"{q}\n" for q in warmup_q_literals), encoding="utf-8"
            )
            cmd.extend(["--warmup-payloads-file", str(warmup_path)])
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        report = json.loads(out_path.read_text(encoding="utf-8"))
        return (
            [[int(x) for x in row] for row in report["doc_ids"]],
            [float(x) for x in report["client_ms"]],
            [float(x) for x in report.get("server_ms", [])],
            float(report["wall_s"]),
        )


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


_UNIT_SEQ = 0


def _next_unit_name(prefix: str) -> str:
    global _UNIT_SEQ
    _UNIT_SEQ += 1
    return f"{prefix}-{os.getpid()}-{int(time.time() * 1000)}-{_UNIT_SEQ}"


def measure_cmd_peak_rss(
    cmd: Sequence[str], env: Optional[Dict[str, str]] = None
) -> Tuple[float, float]:
    """Run command inside a user cgroup scope (if available) and return (elapsed_s, peak_mb)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if has_systemd_user_scope():
        unit = _next_unit_name("osprey-cmd")
        scoped_cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            f"--unit={unit}",
            "-p",
            "MemoryAccounting=yes",
            *cmd,
        ]
        started = time.perf_counter()
        proc = subprocess.Popen(
            scoped_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )

        def _sample_scope_or_vmhwm() -> int:
            cgroup_kb = read_pid_cgroup_peak_kb(proc.pid, expected_unit=unit)
            if cgroup_kb > 0:
                return cgroup_kb
            return read_vmhwm_kb(proc.pid)

        sampler = RssSampler(_sample_scope_or_vmhwm, interval=0.05)
        sampler.start()
        stdout, stderr = proc.communicate()
        peak_kb = sampler.stop()
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}\n{stderr}\n{stdout}"
            )
        return elapsed, peak_kb / 1024.0
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
    stop_qdrant_dedicated_container(QDRANT_BQ_NAME)
    stop_qdrant_dedicated_container(QDRANT_SQ8_NAME)
    stop_osprey_docker_container()


def restart_qdrant_fp32() -> None:
    stop_qdrant_fp32()
    drop_docker_dir_page_cache(QDRANT_FP32_DATA, QDRANT_IMAGE, "/qdrant/storage")
    start_qdrant_fp32()


def drop_pgvector_page_cache_and_restart() -> None:
    """Checkpoint PostgreSQL, stop container, drop host page cache on pgdata, and start fresh cgroup."""
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        conn.execute("CHECKPOINT;")
    stop_pgvector()
    drop_docker_dir_page_cache(PGVECTOR_DATA, PGVECTOR_IMAGE, "/pgdata")
    start_pgvector()


def clear_pg_tables() -> None:
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        cur = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
        )
        for (table,) in cur.fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')


def start_qdrant_dedicated_container(name: str, port: int, data_dir: Path) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(data_dir, 0o777)
    # Clean root-owned files from previous container runs before starting fresh.
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{data_dir.resolve()}:/qdrant/storage",
            "--entrypoint=",
            QDRANT_IMAGE,
            "sh",
            "-c",
            "rm -rf /qdrant/storage/* /qdrant/storage/.[!.]*",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Create without --rm so stop+start isolates build vs query cgroups cleanly.
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{port}:6333",
            "-p",
            f"{port + 1}:6334",
            "-v",
            f"{data_dir}:/qdrant/storage",
            QDRANT_IMAGE,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    wait_http(f"http://127.0.0.1:{port}/", timeout=180)


def restart_qdrant_dedicated_container(name: str, port: int, data_dir: Path) -> None:
    stop_container(name)
    drop_docker_dir_page_cache(data_dir, QDRANT_IMAGE, "/qdrant/storage")
    start_container(name)
    wait_http(f"http://127.0.0.1:{port}/", timeout=180)


def stop_qdrant_dedicated_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Software / environment metadata helpers
# ---------------------------------------------------------------------------
def _git_short_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out
    except (OSError, subprocess.SubprocessError):
        return ""


def get_osprey_version_meta(osprey_bin: str, runner_bin: str) -> str:
    parts: List[str] = []
    for label, bin_path in (("osprey", osprey_bin), ("runner", runner_bin)):
        try:
            out = subprocess.check_output(
                [bin_path, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).strip()
            if out:
                parts.append(f"{label}={out.splitlines()[0]}")
        except (OSError, subprocess.SubprocessError):
            pass
    bin_sha = _git_short_sha(Path(osprey_bin).resolve().parent)
    if bin_sha:
        parts.append(f"osprey_sha={bin_sha}")
    bench_sha = _git_short_sha(ROOT)
    if bench_sha:
        parts.append(f"bench_sha={bench_sha}")
    return " ".join(parts) or "osprey=unknown"


def get_qdrant_version_meta(url: str) -> str:
    try:
        resp = httpx.get(f"{url.rstrip('/')}/", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        ver = data.get("version", "unknown")
        commit = str(data.get("commit", ""))[:8]
        return f"qdrant={ver}" + (f"({commit})" if commit else "")
    except Exception:  # noqa: BLE001
        return "qdrant=unknown"


def get_pgvector_version_meta(dsn: str = PGVECTOR_DSN) -> str:
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            pg_ver = conn.execute("SHOW server_version;").fetchone()
            ext_ver = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
            ).fetchone()
            pg_s = str(pg_ver[0]).split()[0] if pg_ver else "unknown"
            ext_s = str(ext_ver[0]) if ext_ver else "unknown"
            return f"postgres={pg_s} pgvector={ext_s}"
    except Exception:  # noqa: BLE001
        return "postgres=unknown pgvector=unknown"


def host_env_meta() -> str:
    return (
        f"cpu={os.cpu_count()} python={platform.python_version()} "
        f"arch={platform.machine()}"
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
    if n_calib < 1:
        raise ValueError("n_calib must be >= 1")
    if n_calib + n_warmup >= total:
        raise ValueError(
            f"need at least one test query: total={total} calib={n_calib} warmup={n_warmup}"
        )
    calib_slice = slice(0, n_calib)
    warm_slice = slice(n_calib, n_calib + n_warmup)
    pool_idx = np.arange(n_calib + n_warmup, total, dtype=np.int64)
    # Never repeat test queries: measure at most the unique remaining pool.
    requested = n_measure if n_measure > 0 else len(pool_idx)
    test_idx = pool_idx[: min(requested, len(pool_idx))]
    measured = len(test_idx)
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
        "n_measure": measured,
        "unique_test": measured,
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
        self.unit: Optional[str] = None
        self.client: Optional[httpx.Client] = None
        self.load_time_s = 0.0

    def start(self) -> float:
        base_cmd = [
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
        if has_systemd_user_scope():
            unit = _next_unit_name("osprey-serve")
            self.unit = unit
            cmd = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                f"--unit={unit}",
                "-p",
                "MemoryAccounting=yes",
                *base_cmd,
            ]
        else:
            self.unit = None
            cmd = base_cmd
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
        self.unit = None

    def vmhwm_kb(self) -> int:
        return read_vmhwm_kb(self.proc.pid) if self.proc is not None else 0

    def cgroup_peak_kb(self) -> int:
        return (
            read_pid_cgroup_peak_kb(self.proc.pid, expected_unit=self.unit)
            if self.proc is not None
            else 0
        )

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

    def prepare(self, query: np.ndarray, cfg: Dict[str, Any]) -> bytes:
        return json.dumps(
            self._payload(query, cfg["nprobe"], cfg["refine_m"], 10, 0)
        ).encode()

    def query_prepared(
        self, payload: bytes
    ) -> Tuple[List[int], float, float, Dict[str, Any]]:
        assert self.client is not None
        started = time.perf_counter()
        resp = self.client.post(
            f"{self.base_url}/search",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        client_ms = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(f"Osprey HTTP {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        body = resp.json()
        ids = [int(x) for x in body["doc_ids"]]
        return ids, client_ms, float(body["server_ms"]), body

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

    async def _concurrent_payloads(
        self, payloads: Sequence[bytes], workers: int
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        limits = httpx.Limits(
            max_connections=workers, max_keepalive_connections=workers
        )
        results: List[Optional[Tuple[List[int], float, float]]] = [None] * len(payloads)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(payloads)):
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
                        content=payloads[idx],
                        headers={"Content-Type": "application/json"},
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

    def sequential_prepared(
        self,
        payloads: Sequence[bytes],
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float]]:
        ids, client_ms, server_ms, _wall = run_rust_http_load(
            self.runner_bin,
            f"{self.base_url}/search",
            "osprey",
            payloads,
            1,
            warmup_payloads=warmup_payloads,
        )
        return ids, client_ms, server_ms

    def concurrent_prepared(
        self,
        payloads: Sequence[bytes],
        workers: int = 16,
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        return run_rust_http_load(
            self.runner_bin,
            f"{self.base_url}/search",
            "osprey",
            payloads,
            workers,
            warmup_payloads=warmup_payloads,
        )

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
    def __init__(
        self,
        url: str,
        container: str,
        data_dir: Path,
        dataset: str,
        runner_bin: str = "",
        on_disk: bool = False,
    ):
        self.url = url.rstrip("/")
        self.container = container
        self.data_dir = data_dir
        self.dataset = dataset
        self.runner_bin = runner_bin
        self.on_disk = on_disk
        self.client = httpx.Client(timeout=120.0)
        self.current_coll = ""

    def _make_client(
        self, prefer_grpc: bool = False, timeout: int = 900
    ) -> QdrantClient:
        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6333
        return QdrantClient(
            host=host,
            port=port,
            grpc_port=port + 1,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
            check_compatibility=False,
        )

    def close(self) -> None:
        self.client.close()

    def vmhwm_kb(self) -> int:
        return container_max_vmhwm_kb(self.container)

    def collection(self, variant: str) -> str:
        return f"ctrl_{self.dataset}_{variant}"

    def _wait_ready(
        self, client: QdrantClient, coll: str, expected: int
    ) -> None:
        deadline = time.time() + 900
        last = ""
        while time.time() < deadline:
            info = client.get_collection(coll)
            count = client.count(coll, exact=True).count
            indexed = int(info.indexed_vectors_count or 0)
            opt = str(info.optimizer_status).lower()
            st = str(info.status).lower()
            last = (
                f"status={info.status} optimizer={info.optimizer_status} "
                f"count={count}/{expected} indexed={indexed}/{expected}"
            )
            if (
                count == expected
                and indexed >= expected
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
        quantization: Optional[str] = None,
    ) -> Tuple[float, float, str]:
        rest_client = self._make_client(prefer_grpc=False, timeout=900)
        grpc_client = self._make_client(prefer_grpc=True, timeout=900)
        coll = self.collection(variant)
        for existing in rest_client.get_collections().collections:
            try:
                rest_client.delete_collection(existing.name)
            except Exception:
                pass
        try:
            rest_client.delete_collection(coll)
        except Exception:
            pass
        kwargs: Dict[str, Any] = {}
        if quantization == "bq":
            kwargs["quantization_config"] = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )
        elif quantization == "sq8":
            kwargs["quantization_config"] = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True
                )
            )
        # Defer HNSW indexing during bulk upload (`indexing_threshold=0`), and pin
        # `default_segment_number=2` so 100k-scale datasets don't fragment into small
        # unindexed segments below `indexing_threshold`.
        rest_client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(
                size=dim, distance=Distance.COSINE, on_disk=self.on_disk
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0, default_segment_number=2
            ),
            **kwargs,
        )

        total = len(base)
        vectors = base if base.dtype == np.float32 else np.asarray(base, dtype=np.float32)
        batch = 4096 if dim <= 256 else 1024

        t_ingest = time.perf_counter()
        grpc_client.upload_collection(
            collection_name=coll,
            vectors=vectors,
            ids=range(total),
            batch_size=batch,
            parallel=4,
            wait=True,
        )
        ingest_s = time.perf_counter() - t_ingest

        t_index = time.perf_counter()
        rest_client.update_collection(
            collection_name=coll,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000, default_segment_number=2
            ),
        )
        self._wait_ready(rest_client, coll, total)
        index_build_s = time.perf_counter() - t_index
        for c in (grpc_client, rest_client):
            try:
                c.close()
            except Exception:
                pass
        self.current_coll = coll
        return ingest_s, index_build_s, coll

    def wait_after_restart(self, coll: str, expected: int) -> None:
        self.current_coll = coll
        client = self._make_client(prefer_grpc=False, timeout=900)
        try:
            self._wait_ready(client, coll, expected)
        finally:
            try:
                client.close()
            except Exception:
                pass

    def prepare(self, query: np.ndarray, cfg: Dict[str, Any]) -> bytes:
        params: Dict[str, Any] = {"hnsw_ef": int(cfg["hnsw_ef"])}
        if cfg.get("oversampling") is not None:
            params["quantization"] = {
                "rescore": True,
                "oversampling": float(cfg["oversampling"]),
            }
        body = {
            "query": query.tolist(),
            "limit": 10,
            "with_payload": False,
            "params": params,
        }
        return json.dumps(body).encode()

    def query_prepared(self, payload: bytes) -> Tuple[List[int], float, float]:
        started = time.perf_counter()
        resp = self.client.post(
            f"{self.url}/collections/{self.current_coll}/points/query",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        client_ms = (time.perf_counter() - started) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        ids = [int(p["id"]) for p in body["result"]["points"]]
        server_ms = float(body.get("time", 0.0)) * 1000.0
        return ids, client_ms, server_ms

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

    async def _concurrent_payloads(
        self, payloads: Sequence[bytes], workers: int
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        limits = httpx.Limits(
            max_connections=workers, max_keepalive_connections=workers
        )
        results: List[Optional[Tuple[List[int], float, float]]] = [None] * len(payloads)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(payloads)):
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
                        f"{self.url}/collections/{self.current_coll}/points/query",
                        content=payloads[idx],
                        headers={"Content-Type": "application/json"},
                    )
                    client_ms = (time.perf_counter() - started) * 1000.0
                    resp.raise_for_status()
                    body = resp.json()
                    results[idx] = (
                        [int(p["id"]) for p in body["result"]["points"]],
                        client_ms,
                        float(body.get("time", 0.0)) * 1000.0,
                    )

            started = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(workers)])
            wall_s = time.perf_counter() - started

        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        server_ms = [r[2] for r in results if r is not None]
        return ids, client_ms, server_ms, wall_s

    def sequential_prepared(
        self,
        payloads: Sequence[bytes],
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float]]:
        if self.runner_bin:
            ids, client_ms, server_ms, _wall = run_rust_http_load(
                self.runner_bin,
                f"{self.url}/collections/{self.current_coll}/points/query",
                "qdrant",
                payloads,
                1,
                warmup_payloads=warmup_payloads,
            )
            return ids, client_ms, server_ms
        ids_out: List[List[int]] = []
        cms_out: List[float] = []
        sms_out: List[float] = []
        for p in payloads:
            ids, cms, sms = self.query_prepared(p)
            ids_out.append(ids)
            cms_out.append(cms)
            sms_out.append(sms)
        return ids_out, cms_out, sms_out

    def concurrent_prepared(
        self,
        payloads: Sequence[bytes],
        workers: int = 16,
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        if self.runner_bin:
            return run_rust_http_load(
                self.runner_bin,
                f"{self.url}/collections/{self.current_coll}/points/query",
                "qdrant",
                payloads,
                workers,
                warmup_payloads=warmup_payloads,
            )
        return asyncio.run(self._concurrent_payloads(payloads, workers))

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
    def __init__(
        self,
        dataset: str,
        dim: int,
        variant: str = "pgvector",
        runner_bin: str = "",
    ):
        self.dataset = dataset
        self.dim = dim
        self.variant = variant
        self.runner_bin = runner_bin
        self.col_type = "halfvec" if variant == "pgvector_halfvec" else "vector"
        self.ops_class = (
            "halfvec_cosine_ops"
            if variant == "pgvector_halfvec"
            else "vector_cosine_ops"
        )
        self.table = (
            f"ctrl_vecs_{dataset}"
            if variant == "pgvector"
            else f"ctrl_vecs_{dataset}_halfvec"
        )
        self._ef_by_conn: Dict[int, int] = {}
        self._rust_pg_ok: Optional[bool] = None

    def uses_rust_pg_load(self) -> bool:
        if self._rust_pg_ok is None:
            self._rust_pg_ok = runner_supports_pg_load(self.runner_bin)
        return self._rust_pg_ok

    def vmhwm_kb(self) -> int:
        return container_max_vmhwm_kb(PGVECTOR_CONTAINER)

    @staticmethod
    def vector_literal(vec: np.ndarray) -> str:
        return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"

    def setup(self, base: np.ndarray) -> Tuple[float, float]:
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"DROP TABLE IF EXISTS {self.table};")
                cur.execute(
                    f"CREATE TABLE {self.table} (id BIGINT PRIMARY KEY, emb {self.col_type}({self.dim}));"
                )
                cur.execute("SET maintenance_work_mem = '2GB';")
                cur.execute("SET max_parallel_maintenance_workers = 4;")
            with conn.cursor() as cur:
                t_ingest = time.perf_counter()
                total = len(base)
                batch_rows = 4096
                with cur.copy(f"COPY {self.table} (id, emb) FROM STDIN") as copy:
                    for start in range(0, total, batch_rows):
                        end = min(start + batch_rows, total)
                        chunk = base[start:end]
                        lines = [
                            f"{start + offset}\t[{','.join(f'{v:.6f}' for v in row)}]"
                            for offset, row in enumerate(chunk.tolist())
                        ]
                        copy.write("\n".join(lines) + "\n")
                ingest_s = time.perf_counter() - t_ingest

                t_index = time.perf_counter()
                cur.execute(
                    f"CREATE INDEX {self.table}_hnsw_idx ON {self.table} "
                    f"USING hnsw (emb {self.ops_class}) WITH (m = 16, ef_construction = 100);"
                )
                index_build_s = time.perf_counter() - t_index
        return ingest_s, index_build_s

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
            f"SELECT id FROM {self.table} ORDER BY emb <=> %s::{self.col_type} LIMIT 10;",
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

        # Pre-connect all worker sessions before starting wall-clock QPS timing.
        conns = []
        for _ in range(workers):
            conn = await psycopg.AsyncConnection.connect(PGVECTOR_DSN)
            await conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            conns.append(conn)

        async def worker(conn: psycopg.AsyncConnection) -> None:
            while True:
                try:
                    idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                started = time.perf_counter()
                cur = await conn.execute(
                    f"SELECT id FROM {self.table} ORDER BY emb <=> %s::{self.col_type} LIMIT 10;",
                    (q_literals[idx],),
                )
                rows = await cur.fetchall()
                client_ms = (time.perf_counter() - started) * 1000.0
                results[idx] = ([int(r[0]) for r in rows], client_ms)

        try:
            started = time.perf_counter()
            await asyncio.gather(*[worker(c) for c in conns])
            wall_s = time.perf_counter() - started
        finally:
            for c in conns:
                await c.close()
        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        return ids, client_ms, [], wall_s

    def sequential_prepared(
        self,
        q_literals: Sequence[str],
        ef: int,
        warmup_q_literals: Optional[Sequence[str]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float]]:
        if self.uses_rust_pg_load():
            ids, client_ms, server_ms, _wall = run_rust_pg_load(
                self.runner_bin,
                PGVECTOR_DSN,
                self.table,
                self.col_type,
                ef,
                q_literals,
                1,
                warmup_q_literals=warmup_q_literals,
            )
            return ids, client_ms, server_ms
        ids_out: List[List[int]] = []
        cms_out: List[float] = []
        with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
            for q_literal in q_literals:
                found, cms, _ = self.query(conn, q_literal, ef)
                ids_out.append(found)
                cms_out.append(cms)
        return ids_out, cms_out, []

    def concurrent(
        self,
        q_literals: Sequence[str],
        ef: int,
        workers: int = 16,
        warmup_q_literals: Optional[Sequence[str]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        if self.uses_rust_pg_load():
            return run_rust_pg_load(
                self.runner_bin,
                PGVECTOR_DSN,
                self.table,
                self.col_type,
                ef,
                q_literals,
                workers,
                warmup_q_literals=warmup_q_literals,
            )
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
                    f"SELECT id FROM {self.table} ORDER BY emb <=> %s::{self.col_type} LIMIT 10;",
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
            # pg_total_relation_size already includes heap, TOAST, and all indexes (incl. HNSW).
            cur = conn.execute(f"SELECT pg_total_relation_size('{self.table}');")
            row = cur.fetchone()
            if row:
                return int(row[0]) / (1024.0 * 1024.0)
        return 0.0


# ---------------------------------------------------------------------------
# FAISS IVF-Flat optional engine (in-process)
# ---------------------------------------------------------------------------
class FaissIvfEngine:
    def __init__(self, dataset: str, dim: int, clusters: int, index_path: Path):
        import faiss  # type: ignore[import-not-found]

        self._faiss = faiss
        self.dataset = dataset
        self.dim = dim
        self.clusters = clusters
        self.index_path = index_path
        self.index: Any = None
        self.nlist = clusters

    def setup(self, base: np.ndarray) -> Tuple[float, float]:
        vecs = np.ascontiguousarray(base, dtype=np.float32)
        n_vecs = len(vecs)
        self.nlist = max(1, min(int(self.clusters), max(1, n_vecs // 39)))
        quantizer = self._faiss.IndexFlatIP(self.dim)
        index = self._faiss.IndexIVFFlat(
            quantizer, self.dim, self.nlist, self._faiss.METRIC_INNER_PRODUCT
        )
        t_idx = time.perf_counter()
        index.train(vecs)
        index.add(vecs)
        index_build_s = time.perf_counter() - t_idx
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(index, str(self.index_path))
        self.index = index
        return 0.0, index_build_s

    def query(self, q: np.ndarray, nprobe: int) -> Tuple[List[int], float, float]:
        assert self.index is not None
        self._faiss.omp_set_num_threads(1)
        self.index.nprobe = int(nprobe)
        q2d = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)
        started = time.perf_counter()
        _dist, idx = self.index.search(q2d, 10)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [int(x) for x in idx[0] if x >= 0], elapsed_ms, elapsed_ms

    def concurrent(
        self, queries: Sequence[np.ndarray], nprobe: int, workers: int = 16
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        assert self.index is not None
        self._faiss.omp_set_num_threads(1)
        self.index.nprobe = int(nprobe)
        q_arr = [np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1) for q in queries]

        def run_one(q2d: np.ndarray) -> Tuple[List[int], float]:
            t0 = time.perf_counter()
            _dist, idx = self.index.search(q2d, 10)
            ms = (time.perf_counter() - t0) * 1000.0
            return [int(x) for x in idx[0] if x >= 0], ms

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(run_one, q_arr))
        wall_s = time.perf_counter() - started
        ids = [r[0] for r in res]
        lat = [r[1] for r in res]
        return ids, lat, list(lat), wall_s

    def disk_mb(self) -> float:
        return dir_size_mb(self.index_path)


def calibrate_faiss(
    engine: FaissIvfEngine,
    queries: np.ndarray,
    gt: np.ndarray,
    nprobe_grid: Sequence[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    valid_nprobes = [n for n in nprobe_grid if n <= engine.nlist]
    if not valid_nprobes:
        valid_nprobes = [engine.nlist]
    for nprobe in valid_nprobes:
        ids: List[List[int]] = []
        latencies: List[float] = []
        for q in queries:
            found, ms, _ = engine.query(q, nprobe)
            ids.append(found)
            latencies.append(ms)
        p50, _, _ = latency_stats(latencies)
        recall = recall_at_k(ids, gt)
        print(
            f"  [calib][faiss_ivf] nprobe={nprobe}/{engine.nlist} "
            f"recall={recall:.4f} p50={p50:.3f}ms",
            flush=True,
        )
        results.append({"nprobe": nprobe, "nlist": engine.nlist, "recall": recall, "p50_ms": p50})
    return results


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
        # Require the validation recall itself to meet the target. The previous
        # `target - 0.005` tolerance let configurations through that missed the
        # target on the independent test split.
        candidates = [r for r in results if r["recall"] >= target]
        if candidates:
            best = min(candidates, key=cost_fn)
            selected[target] = {**best, "status": "ok"}
        else:
            best = max(results, key=lambda r: r["recall"])
            selected[target] = {**best, "status": "below_target"}
    return selected


def calibrate_osprey(
    engine: Any,
    queries: np.ndarray,
    gt: np.ndarray,
    nprobe_grid: Sequence[int],
    refine_grid: Sequence[int],
    label: str = "osprey",
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
                f"  [calib][{label}] nprobe={nprobe} refine={refine_m} "
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
        [Sequence[Any], Dict[str, Any], int, Optional[Sequence[Any]]],
        Tuple[List[List[int]], List[float], List[float], float],
    ],
    prepared_queries: Optional[Sequence[Any]] = None,
    prepare_fn: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
    server_latency_fn: Optional[Callable[[Dict[str, Any]], Tuple[float, float, float]]] = None,
    sequential_batch_fn: Optional[
        Callable[
            [Sequence[Any], Dict[str, Any], Optional[Sequence[Any]]],
            Tuple[List[List[int]], List[float], List[float]],
        ]
    ] = None,
) -> Dict[float, Dict[str, Any]]:
    measured: Dict[float, Dict[str, Any]] = {}
    for target, cfg in selected.items():
        if prepared_queries is not None:
            warm_items = list(warmup_queries)
            test_items = list(prepared_queries)
        elif prepare_fn is not None:
            # Serialize/format requests once, outside the timed loop, so the
            # HTTP engines do not pay Python JSON cost per query.
            warm_items = [prepare_fn(query, cfg) for query in warmup_queries]
            test_items = [prepare_fn(query, cfg) for query in test_queries]
        else:
            warm_items = list(warmup_queries)
            test_items = list(test_queries)
        for item in warm_items:
            warmup_fn(item, cfg)

        round_stats: List[Dict[str, Any]] = []
        for round_idx in range(rounds):
            print(
                f"  [measure] target={target:.3f} round={round_idx + 1}/{rounds} "
                f"queries={len(test_items)}",
                flush=True,
            )
            source = test_items
            if sequential_batch_fn is not None:
                ids, latencies, server_latencies = sequential_batch_fn(
                    source, cfg, warm_items
                )
            else:
                ids = []
                latencies = []
                server_latencies = []
                for item in source:
                    found, client_ms, server_ms = sequential_fn(item, cfg)
                    ids.append(found)
                    latencies.append(client_ms)
                    if server_ms is not None:
                        server_latencies.append(server_ms)
            p50, p95, p99 = latency_stats(latencies)
            s50, s95, s99 = latency_stats(server_latencies)

            c_ids, c_latencies, c_server, wall_s = concurrent_fn(
                list(source), cfg, workers, warm_items
            )
            c50, c95, c99 = latency_stats(c_latencies)
            cs50, cs95, cs99 = latency_stats(c_server)
            qps = (len(source) / wall_s) if wall_s > 0 else 0.0

            round_recall = recall_at_k(ids, test_gt)
            ci_low, ci_high = bootstrap_recall_ci95(ids, test_gt)
            round_concurrent_recall = recall_at_k(c_ids, test_gt)
            print(
                f"  [measure] target={target:.3f} round={round_idx + 1}/{rounds} "
                f"recall={round_recall:.4f} (95%CI [{ci_low:.4f}, {ci_high:.4f}]) "
                f"concurrent_recall={round_concurrent_recall:.4f} "
                f"p50={p50:.3f}ms qps={qps:.1f}",
                flush=True,
            )
            round_stats.append(
                {
                    "recall": round_recall,
                    "recall_ci95_low": ci_low,
                    "recall_ci95_high": ci_high,
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
            return median_std([r[key] for r in round_stats])

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
        median_round = sorted(round_stats, key=lambda r: r["recall"])[
            len(round_stats) // 2
        ]
        summary["recall_ci95_low"] = median_round["recall_ci95_low"]
        summary["recall_ci95_high"] = median_round["recall_ci95_high"]
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
    query_vmhwm_kb: int,
    build_peak_kb: int,
    build_time_s: float,
    load_time_s: float,
    disk_mb: float,
    rounds: int,
    workers: int,
    memory_metric: str,
    notes: str,
    ingest_time_s: float = 0.0,
    index_build_time_s: float = 0.0,
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
        "test_recall_ci95_low": round(float(stats.get("recall_ci95_low", 0.0)), 4),
        "test_recall_ci95_high": round(float(stats.get("recall_ci95_high", 0.0)), 4),
        "test_target_met": bool(
            float(stats.get("recall_median", 0.0)) >= float(target)
        ),
        "test_recall_delta": round(
            float(stats.get("recall_median", 0.0)) - float(target), 4
        ),
        "p50_ms_median": round(float(stats.get("p50_median", 0.0)), 3),
        "p50_ms_std": round(float(stats.get("p50_std", 0.0)), 3),
        "p95_ms_median": round(float(stats.get("p95_median", 0.0)), 3),
        "p95_ms_std": round(float(stats.get("p95_std", 0.0)), 3),
        "p99_ms_median": round(float(stats.get("p99_median", 0.0)), 3),
        "p99_ms_std": round(float(stats.get("p99_std", 0.0)), 3),
        "server_p50_ms_median": round(float(stats.get("server_p50_median", 0.0)), 3),
        "server_p95_ms_median": round(float(stats.get("server_p95_median", 0.0)), 3),
        "server_p99_ms_median": round(float(stats.get("server_p99_median", 0.0)), 3),
        "concurrent_p50_ms_median": round(
            float(stats.get("concurrent_p50_median", 0.0)), 3
        ),
        "concurrent_p50_ms_std": round(
            float(stats.get("concurrent_p50_std", 0.0)), 3
        ),
        "concurrent_p95_ms_median": round(
            float(stats.get("concurrent_p95_median", 0.0)), 3
        ),
        "concurrent_p99_ms_median": round(
            float(stats.get("concurrent_p99_median", 0.0)), 3
        ),
        "concurrent_server_p50_ms_median": round(
            float(stats.get("concurrent_server_p50_median", 0.0)), 3
        ),
        "concurrent_server_p95_ms_median": round(
            float(stats.get("concurrent_server_p95_median", 0.0)), 3
        ),
        "concurrent_server_p99_ms_median": round(
            float(stats.get("concurrent_server_p99_median", 0.0)), 3
        ),
        "qps_concurrent_median": round(float(stats.get("qps_median", 0.0)), 1),
        "qps_concurrent_std": round(float(stats.get("qps_std", 0.0)), 1),
        "concurrent_recall": round(float(stats.get("concurrent_recall_median", 0.0)), 4),
        "memory_metric": memory_metric,
        "load_peak_rss_mb": round(load_peak_kb / 1024.0, 1),
        "query_peak_rss_mb": round(query_peak_kb / 1024.0, 1),
        "process_vmhwm_mb": round(query_vmhwm_kb / 1024.0, 1),
        "build_peak_rss_mb": round(build_peak_kb / 1024.0, 1),
        "ingest_time_s": round(ingest_time_s, 3),
        "index_build_time_s": round(index_build_time_s, 3),
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


def _run_qdrant_variant(
    dataset: str,
    meta: Dict[str, Any],
    base: np.ndarray,
    dim: int,
    splits: Dict[str, Any],
    targets: Sequence[float],
    ef_grid: Sequence[int],
    oversampling_grid: Sequence[float],
    rounds: int,
    workers: int,
    runner_bin: str,
    variant: str,
    on_disk: bool = False,
) -> List[Dict[str, Any]]:
    """Run one Qdrant variant (qdrant_fp32, qdrant_sq8, qdrant_bq) with isolated build vs query cgroups."""
    stop_all_engines()
    storage_mode = "on_disk=True" if on_disk else "on_disk=False"
    if variant == "qdrant_fp32":
        label = "Qdrant (HNSW FP32)"
        container = QDRANT_FP32_CONTAINER
        url = QDRANT_FP32_URL
        data_dir = QDRANT_FP32_DATA
        quantization: Optional[str] = None
        short_tag = "fp32"
        start_qdrant_fp32()
        restart_fn = restart_qdrant_fp32
        stop_fn = stop_qdrant_fp32
        notes = (
            f"Qdrant compose container ({storage_mode}, default_segment_number=2); "
            "HNSW m=16 ef_construct=100; restarted + root dd page cache evicted after build; "
            "Rust HTTP load client; query peak memory is query-only cgroup memory.peak."
        )
    elif variant == "qdrant_sq8":
        label = "Qdrant (HNSW int8 SQ8 + rescore)"
        container = QDRANT_SQ8_NAME
        url = f"http://127.0.0.1:{QDRANT_SQ8_PORT}"
        data_dir = QDRANT_SQ8_DATA
        quantization = "sq8"
        short_tag = "sq8"
        start_qdrant_dedicated_container(container, QDRANT_SQ8_PORT, data_dir)
        restart_fn = lambda: restart_qdrant_dedicated_container(container, QDRANT_SQ8_PORT, data_dir)
        stop_fn = lambda: stop_qdrant_dedicated_container(container)
        notes = (
            f"Dedicated Qdrant container ({storage_mode}, default_segment_number=2); "
            "scalar quantization INT8 always_ram, rescore=true; restarted + root dd page cache evicted after build; "
            "Rust HTTP load client; query peak memory is query-only cgroup memory.peak."
        )
    elif variant == "qdrant_bq":
        label = "Qdrant (HNSW 1-bit BQ + rescore)"
        container = QDRANT_BQ_NAME
        url = f"http://127.0.0.1:{QDRANT_BQ_PORT}"
        data_dir = QDRANT_BQ_DATA
        quantization = "bq"
        short_tag = "bq"
        start_qdrant_dedicated_container(container, QDRANT_BQ_PORT, data_dir)
        restart_fn = lambda: restart_qdrant_dedicated_container(container, QDRANT_BQ_PORT, data_dir)
        stop_fn = lambda: stop_qdrant_dedicated_container(container)
        notes = (
            f"Dedicated Qdrant container ({storage_mode}, default_segment_number=2); "
            "binary quantization always_ram, rescore=true; restarted + root dd page cache evicted after build; "
            "Rust HTTP load client; query peak memory is query-only cgroup memory.peak."
        )
    else:
        raise ValueError(f"unknown Qdrant variant: {variant}")

    qdrant_ver = get_qdrant_version_meta(url)
    notes = f"{notes} versions=[{qdrant_ver} {host_env_meta()}]"

    print(f"[controlled] {dataset}: starting {label} ({qdrant_ver}) and building index")
    eng = QdrantEngine(
        url, container, data_dir, dataset, runner_bin=runner_bin, on_disk=on_disk
    )
    ingest_s = 0.0
    index_build_s = 0.0
    try:
        ingest_s, index_build_s, coll = eng.setup(
            base, dim, short_tag, quantization=quantization
        )
    finally:
        build_peak_kb = container_peak_memory_kb(container)
    build_s = ingest_s + index_build_s

    # Isolate query-stage cgroup memory.peak from build/ingestion spikes.
    restart_fn()
    eng.wait_after_restart(coll, len(base))
    sampler = RssSampler(eng.vmhwm_kb, interval=0.25)
    sampler.start()
    load_peak_kb = 0
    try:
        print(f"[controlled] {dataset}: calibrating {label}")
        use_os = quantization is not None
        calib = calibrate_qdrant(
            eng,
            coll,
            splits["calib"],
            splits["calib_gt"],
            ef_grid,
            oversampling_grid if use_os else [None],
            use_os,
        )
        if use_os:
            selected = select_for_targets(
                calib,
                targets,
                cost_fn=lambda r: (
                    r["hnsw_ef"] * float(r["oversampling"] or 1.0),
                    r["hnsw_ef"],
                    r["p50_ms"],
                ),
            )
        else:
            selected = select_for_targets(
                calib, targets, cost_fn=lambda r: (r["hnsw_ef"], r["p50_ms"])
            )
        for target, cfg in selected.items():
            print(
                f"  target={target:.3f} -> hnsw_ef={cfg['hnsw_ef']} "
                f"oversampling={cfg['oversampling']} val_recall={cfg['recall']:.4f} "
                f"status={cfg['status']}"
            )

        def seq(item: bytes, cfg: Dict[str, Any]):
            return eng.query_prepared(item)

        def seq_batch(
            items: Sequence[bytes],
            cfg: Dict[str, Any],
            warm_items: Optional[Sequence[Any]] = None,
        ):
            return eng.sequential_prepared(items, warmup_payloads=warm_items)

        def warm(item: bytes, cfg: Dict[str, Any]):
            eng.query_prepared(item)

        def conc(
            items: Sequence[bytes],
            cfg: Dict[str, Any],
            nworkers: int,
            warm_items: Optional[Sequence[Any]] = None,
        ):
            return eng.concurrent_prepared(
                items, nworkers, warmup_payloads=warm_items
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
            prepare_fn=eng.prepare,
            sequential_batch_fn=seq_batch,
        )
    finally:
        query_vmhwm_kb = sampler.stop()
        query_peak_kb = container_peak_memory_kb(container)
    eng.close()
    stop_fn()
    disk_mb = eng.disk_mb(coll)

    out_rows: List[Dict[str, Any]] = []
    for target, stats in measured.items():
        out_rows.append(
            build_row(
                dataset,
                meta,
                splits,
                label,
                variant,
                "HTTP localhost (Rust client)",
                target,
                selected[target],
                stats,
                load_peak_kb,
                query_peak_kb,
                query_vmhwm_kb,
                build_peak_kb,
                build_s,
                0.0,
                disk_mb,
                rounds,
                workers,
                "cgroup v2 memory.peak (query-only)",
                notes,
                ingest_time_s=ingest_s,
                index_build_time_s=index_build_s,
            )
        )
    return out_rows


def _calibrate_and_measure_osprey(
    engine: Any,
    splits: Dict[str, Any],
    targets: Sequence[float],
    nprobe_grid: Sequence[int],
    refine_grid_ds: Sequence[int],
    rounds: int,
    workers: int,
    label: str = "osprey",
) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    print(f"[controlled] calibrating {label}")
    calib = calibrate_osprey(
        engine,
        splits["calib"],
        splits["calib_gt"],
        nprobe_grid,
        refine_grid_ds,
        label=label,
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

    def seq(item: bytes, cfg: Dict[str, Any]):
        res = engine.query_prepared(item)
        return res[0], res[1], res[2]

    def seq_batch(
        items: Sequence[bytes],
        cfg: Dict[str, Any],
        warm_items: Optional[Sequence[Any]] = None,
    ):
        return engine.sequential_prepared(items, warmup_payloads=warm_items)

    def warm(item: bytes, cfg: Dict[str, Any]):
        engine.query_prepared(item)

    def conc(
        items: Sequence[bytes],
        cfg: Dict[str, Any],
        nworkers: int,
        warm_items: Optional[Sequence[Any]] = None,
    ):
        return engine.concurrent_prepared(
            items, nworkers, warmup_payloads=warm_items
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
        prepare_fn=engine.prepare,
        sequential_batch_fn=seq_batch,
    )
    return selected, measured


def _run_osprey_docker_variant(
    dataset: str,
    ds_dir: Path,
    meta: Dict[str, Any],
    dim: int,
    clusters: int,
    splits: Dict[str, Any],
    targets: Sequence[float],
    nprobe_grid: Sequence[int],
    refine_grid_ds: Sequence[int],
    rounds: int,
    workers: int,
    osprey_bin: str,
    runner_bin: str,
    osprey_ver: str,
    env_ver: str,
    port: int = OSPREY_DOCKER_PORT,
) -> List[Dict[str, Any]]:
    """Run production `osprey serve` inside a symmetric Docker container."""
    stop_all_engines()
    print(f"[controlled] {dataset}: building OspreyDB Docker index ({osprey_ver})")
    eng = OspreyDockerServerEngine(
        dataset=dataset,
        dim=dim,
        osprey_bin=osprey_bin,
        runner_bin=runner_bin,
        port=port,
    )
    store_dir, _wal_dir = eng.prepare_data_dir()
    build_s, build_peak_mb = measure_cmd_peak_rss(
        [
            osprey_bin,
            "import",
            "--input",
            str(ds_dir / "base.parquet"),
            "--output",
            str(store_dir),
            "--id-col",
            "doc_id",
            "--vec-col",
            "embedding",
            "--clusters",
            str(clusters),
            "--json",
        ]
    )
    eng.write_manifest()
    eng.drop_page_cache()
    print(f"[controlled] {dataset}: starting OspreyDB Docker server on {eng.url}")
    load_s = eng.start()
    load_cgroup_kb = eng.cgroup_peak_kb()
    load_peak_kb = load_cgroup_kb if load_cgroup_kb > 0 else eng.vmhwm_kb()
    sampler = RssSampler(eng.vmhwm_kb, interval=0.25)
    sampler.start()
    try:
        selected, measured = _calibrate_and_measure_osprey(
            eng,
            splits,
            targets,
            nprobe_grid,
            refine_grid_ds,
            rounds,
            workers,
            label="osprey_docker",
        )
    finally:
        query_vmhwm_kb = sampler.stop()
        query_cgroup_kb = eng.cgroup_peak_kb()
        eng.stop()
    use_cgroup = query_cgroup_kb > 0
    query_peak_kb = query_cgroup_kb if use_cgroup else query_vmhwm_kb
    mem_metric = (
        "cgroup v2 memory.peak (query-only)" if use_cgroup else "VmHWM (process)"
    )
    disk_mb = eng.disk_mb()
    transport = (
        "HTTP Docker bridge (Rust client)"
        if eng.uses_rust_http_load()
        else "HTTP Docker bridge (Python httpx fallback)"
    )

    out_rows: List[Dict[str, Any]] = []
    for target, stats in measured.items():
        out_rows.append(
            build_row(
                dataset,
                meta,
                splits,
                "OspreyDB Server (Docker, WAL+LSM, 1-bit .osi + f16 .osv)",
                "osprey_docker",
                transport,
                target,
                selected[target],
                stats,
                load_peak_kb,
                query_peak_kb,
                query_vmhwm_kb,
                int(build_peak_mb * 1024),
                build_s,
                load_s,
                disk_mb,
                rounds,
                workers,
                mem_metric,
                (
                    f"Production osprey serve --role unified in Docker container ({OSPREY_DOCKER_CONTAINER}); "
                    "WAL+LSM+CoW compactor active; manifest.json from .osi header; "
                    "root dd page cache evicted before serve; query peak memory is query-only cgroup memory.peak. "
                    f"versions=[{osprey_ver} {env_ver}]"
                ),
                ingest_time_s=0.0,
                index_build_time_s=build_s,
            )
        )
    return out_rows


def _run_pgvector_variant(
    dataset: str,
    meta: Dict[str, Any],
    base: np.ndarray,
    dim: int,
    splits: Dict[str, Any],
    targets: Sequence[float],
    ef_grid: Sequence[int],
    rounds: int,
    workers: int,
    variant: str,
    runner_bin: str = "",
) -> List[Dict[str, Any]]:
    """Run pgvector FP32 or halfvec (FP16) with isolated build vs query cgroups."""
    stop_all_engines()
    label = (
        "pgvector (HNSW halfvec FP16)"
        if variant == "pgvector_halfvec"
        else "pgvector (HNSW FP32)"
    )
    print(f"[controlled] {dataset}: starting {label} and building index")
    start_pgvector()
    try:
        clear_pg_tables()
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        print(f"  [warn] could not clear pgvector tables: {exc}")
    pg = PgvectorEngine(dataset, dim, variant=variant, runner_bin=runner_bin)
    use_rust_pg = pg.uses_rust_pg_load()
    transport = (
        "SQL Docker bridge (Rust tokio-postgres)"
        if use_rust_pg
        else "SQL localhost (psycopg)"
    )
    client_note = (
        "Rust tokio-postgres load client" if use_rust_pg else "psycopg SQL client"
    )
    ingest_s = 0.0
    index_build_s = 0.0
    try:
        ingest_s, index_build_s = pg.setup(base)
    finally:
        build_peak_kb = container_peak_memory_kb(PGVECTOR_CONTAINER)
    build_s = ingest_s + index_build_s
    pg_ver = get_pgvector_version_meta(PGVECTOR_DSN)

    # Flush WAL/buffers, drop host page cache, and restart into a fresh cgroup.
    drop_pgvector_page_cache_and_restart()
    sampler = RssSampler(pg.vmhwm_kb, interval=0.25)
    sampler.start()
    load_peak_kb = 0
    calib_q = [pg.vector_literal(q) for q in splits["calib"]]
    warm_q = [pg.vector_literal(q) for q in splits["warmup"]]
    test_q = [pg.vector_literal(q) for q in splits["test"]]
    try:
        print(f"[controlled] {dataset}: calibrating {label} ({pg_ver})")
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

            def seq_batch(
                items: Sequence[str],
                cfg: Dict[str, Any],
                warm_items: Optional[Sequence[Any]] = None,
            ):
                return pg.sequential_prepared(
                    list(items),
                    cfg["hnsw_ef_search"],
                    warmup_q_literals=list(warm_items) if warm_items else None,
                )

            def conc(
                items: Sequence[str],
                cfg: Dict[str, Any],
                nworkers: int,
                warm_items: Optional[Sequence[Any]] = None,
            ):
                return pg.concurrent(
                    list(items),
                    cfg["hnsw_ef_search"],
                    nworkers,
                    warmup_q_literals=list(warm_items) if warm_items else None,
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
                sequential_batch_fn=seq_batch if use_rust_pg else None,
            )
    finally:
        query_vmhwm_kb = sampler.stop()
        query_peak_kb = container_peak_memory_kb(PGVECTOR_CONTAINER)
    disk_mb = pg.disk_mb()
    stop_pgvector()

    out_rows: List[Dict[str, Any]] = []
    for target, stats in measured.items():
        out_rows.append(
            build_row(
                dataset,
                meta,
                splits,
                label,
                variant,
                transport,
                target,
                selected[target],
                stats,
                load_peak_kb,
                query_peak_kb,
                query_vmhwm_kb,
                build_peak_kb,
                build_s,
                0.0,
                disk_mb,
                rounds,
                workers,
                "cgroup v2 memory.peak (query-only)",
                (
                    f"PostgreSQL container ({pg.col_type}); HNSW m=16 ef_construction=100; "
                    f"{client_note}; restarted + page cache dropped after build; "
                    "server time via EXPLAIN ANALYZE (sample=100); "
                    f"query peak memory is query-only cgroup memory.peak incl. page cache. "
                    f"versions=[{pg_ver} {host_env_meta()}]"
                ),
                ingest_time_s=ingest_s,
                index_build_time_s=index_build_s,
            )
        )
    return out_rows


def _run_faiss_ivf_variant(
    dataset: str,
    meta: Dict[str, Any],
    base: np.ndarray,
    dim: int,
    clusters: int,
    splits: Dict[str, Any],
    targets: Sequence[float],
    rounds: int,
    workers: int,
    quick: bool,
) -> List[Dict[str, Any]]:
    """Run optional in-process FAISS IVF-Flat baseline, skipping cleanly if faiss is absent."""
    if importlib.util.find_spec("faiss") is None:
        print(f"[skip] {dataset}: faiss_ivf skipped (faiss not installed)")
        return []

    import faiss  # type: ignore[import-not-found]

    stop_all_engines()
    print(f"[controlled] {dataset}: building FAISS IVF-Flat baseline")
    index_path = ARTIFACTS_DIR / dataset / "faiss_ivf.index"
    eng = FaissIvfEngine(dataset, dim, clusters, index_path)
    vmhwm_before = read_vmhwm_kb(os.getpid())
    sampler = RssSampler(lambda: read_vmhwm_kb(os.getpid()), interval=0.25)
    sampler.start()
    ingest_s, index_build_s = eng.setup(base)
    build_peak_kb = max(sampler.stop(), vmhwm_before)
    build_s = ingest_s + index_build_s

    faiss_nprobe_grid = [4, 16, 64] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    q_sampler = RssSampler(lambda: read_vmhwm_kb(os.getpid()), interval=0.25)
    q_sampler.start()
    try:
        calib = calibrate_faiss(eng, splits["calib"], splits["calib_gt"], faiss_nprobe_grid)
        selected = select_for_targets(
            calib, targets, cost_fn=lambda r: (r["nprobe"], r["p50_ms"])
        )
        for target, cfg in selected.items():
            print(
                f"  target={target:.3f} -> nprobe={cfg['nprobe']}/{cfg['nlist']} "
                f"val_recall={cfg['recall']:.4f} status={cfg['status']}"
            )

        def seq(item: np.ndarray, cfg: Dict[str, Any]):
            return eng.query(item, cfg["nprobe"])

        def warm(item: np.ndarray, cfg: Dict[str, Any]):
            eng.query(item, cfg["nprobe"])

        def conc(
            items: Sequence[np.ndarray],
            cfg: Dict[str, Any],
            nworkers: int,
            _warm_items: Optional[Sequence[Any]] = None,
        ):
            return eng.concurrent(items, cfg["nprobe"], nworkers)

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
        query_vmhwm_kb = q_sampler.stop()
    disk_mb = eng.disk_mb()
    faiss_ver = getattr(faiss, "__version__", "unknown")

    out_rows: List[Dict[str, Any]] = []
    for target, stats in measured.items():
        out_rows.append(
            build_row(
                dataset,
                meta,
                splits,
                "FAISS (IVF-Flat)",
                "faiss_ivf",
                "In-process (FAISS C++ ThreadPool)",
                target,
                selected[target],
                stats,
                0,
                query_vmhwm_kb,
                query_vmhwm_kb,
                build_peak_kb,
                build_s,
                0.0,
                disk_mb,
                rounds,
                workers,
                "VmHWM (in-process)",
                (
                    f"In-process FAISS IndexIVFFlat (nlist={eng.nlist}, METRIC_INNER_PRODUCT); "
                    f"versions=[faiss={faiss_ver} {host_env_meta()}]"
                ),
                ingest_time_s=ingest_s,
                index_build_time_s=index_build_s,
            )
        )
    return out_rows


# ---------------------------------------------------------------------------
# Top-level controlled driver
# ---------------------------------------------------------------------------
def run_controlled(args: Any) -> int:
    targets = [float(t) for t in args.recall_target]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    rounds = int(args.rounds)
    workers = int(args.workers)
    n_warmup = int(args.warmup_queries)
    n_measure = int(args.measure_queries)
    requested_calib = int(getattr(args, "calib_queries", 0) or 0)
    qdrant_on_disk = bool(getattr(args, "qdrant_on_disk", False))
    if args.quick:
        n_warmup = min(20, n_warmup)
        n_measure = min(40, n_measure)
        rounds = min(1, rounds)

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
        [16, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512]
        if not args.quick
        else [48, 80, 128]
    )
    oversampling_grid = (
        [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
        if not args.quick
        else [1.0, 5.0]
    )

    osprey_ver = get_osprey_version_meta(args.osprey_bin, args.runner_bin)
    env_ver = host_env_meta()
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
        if args.quick:
            ds_calib = 40
        elif requested_calib > 0:
            ds_calib = requested_calib
        else:
            # Larger validation split than the old fixed 200 reduces the
            # validation/test recall gap; 25% of the query file, floor 200.
            ds_calib = max(200, int(round(len(queries) * 0.25)))
        splits = make_splits(
            queries,
            gt,
            n_calib=ds_calib,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        print(
            f"[controlled] {dataset}: {meta.get('num_vectors')} x {dim}d, "
            f"queries={splits['n_total']} (calib={splits['n_calib']}, "
            f"warmup={splits['n_warmup']}, measure={splits['n_measure']}, "
            f"unique_test={splits['unique_test']})"
        )

        if "osprey_docker" in engines:
            rows.extend(
                _run_osprey_docker_variant(
                    dataset,
                    ds_dir,
                    meta,
                    dim,
                    int(args.clusters),
                    splits,
                    targets,
                    nprobe_grid,
                    refine_grid_ds,
                    rounds,
                    workers,
                    args.osprey_bin,
                    args.runner_bin,
                    osprey_ver,
                    env_ver,
                )
            )

        if "osprey" in engines:
            stop_all_engines()
            print(f"[controlled] {dataset}: building OspreyDB index ({osprey_ver})")
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
            # Evict build-written .osi/.osv from Linux page cache so the serve cgroup
            # pays the full OS page-cache cost symmetrically with Docker containers.
            drop_path_page_cache(art_dir)
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
            load_cgroup_kb = engine.cgroup_peak_kb()
            load_peak_kb = load_cgroup_kb if load_cgroup_kb > 0 else engine.vmhwm_kb()
            sampler = RssSampler(engine.vmhwm_kb, interval=0.25)
            sampler.start()
            try:
                selected, measured = _calibrate_and_measure_osprey(
                    engine,
                    splits,
                    targets,
                    nprobe_grid,
                    refine_grid_ds,
                    rounds,
                    workers,
                    label="osprey",
                )
            finally:
                query_vmhwm_kb = sampler.stop()
                query_cgroup_kb = engine.cgroup_peak_kb()
                engine.stop()
            use_cgroup = query_cgroup_kb > 0
            query_peak_kb = query_cgroup_kb if use_cgroup else query_vmhwm_kb
            mem_metric = (
                "cgroup v2 memory.peak (query-only)" if use_cgroup else "VmHWM (process)"
            )
            disk_mb = dir_size_mb(art_dir / "segment.osi") + dir_size_mb(
                art_dir / "segment.osv"
            )

            for target, stats in measured.items():
                rows.append(
                    build_row(
                        dataset,
                        meta,
                        splits,
                        "OspreyDB (HTTP, 1-bit .osi + f16 .osv)",
                        "osprey_http",
                        "HTTP localhost (Rust client)",
                        target,
                        selected[target],
                        stats,
                        load_peak_kb,
                        query_peak_kb,
                        query_vmhwm_kb,
                        int(build_peak_mb * 1024),
                        build_s,
                        load_s,
                        disk_mb,
                        rounds,
                        workers,
                        mem_metric,
                        (
                            "HTTP server in osprey-bench-runner; Rust HTTP load client; "
                            ".osi/.osv page cache dropped before serve; peak memory includes RSS + Linux page cache via cgroup v2 memory.peak. "
                            f"versions=[{osprey_ver} {env_ver}]"
                        ),
                        ingest_time_s=0.0,
                        index_build_time_s=build_s,
                    )
                )

        for q_variant in ("qdrant_fp32", "qdrant_sq8", "qdrant_bq"):
            if q_variant in engines:
                rows.extend(
                    _run_qdrant_variant(
                        dataset,
                        meta,
                        base,
                        dim,
                        splits,
                        targets,
                        ef_grid,
                        oversampling_grid,
                        rounds,
                        workers,
                        args.runner_bin,
                        q_variant,
                        on_disk=qdrant_on_disk,
                    )
                )

        for pg_variant in ("pgvector", "pgvector_halfvec"):
            if pg_variant in engines:
                rows.extend(
                    _run_pgvector_variant(
                        dataset,
                        meta,
                        base,
                        dim,
                        splits,
                        targets,
                        ef_grid,
                        rounds,
                        workers,
                        pg_variant,
                        runner_bin=args.runner_bin,
                    )
                )

        if "faiss_ivf" in engines:
            rows.extend(
                _run_faiss_ivf_variant(
                    dataset,
                    meta,
                    base,
                    dim,
                    int(args.clusters),
                    splits,
                    targets,
                    rounds,
                    workers,
                    bool(args.quick),
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
