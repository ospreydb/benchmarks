#!/usr/bin/env python3
"""
Docker-containerized production `osprey serve` engine and shared Docker/cgroup helpers.

Runs the host-compiled native `osprey` binary inside a symmetric Docker container
(`osprey-bench-osprey`, built from `docker/Dockerfile.osprey`) with:
  * Host-mounted `/data` volume (`docker/osprey-data`) containing `store/` and `wal/`
  * `manifest.json` (`ClusterManifest`) generated directly from the `.osi` binary header
  * Symmetric post-build `dd iflag=nocache` host page-cache eviction via root container
  * Docker bridge networking (`127.0.0.1:16333 -> 6333`, `127.0.0.1:18815 -> 8815`)
  * `cgroup v2 memory.peak`, `VmHWM`, and allocated `st_blocks * 512` disk accounting
  * Rust `osprey-bench-runner http-load --engine osprey_server` (with Python `httpx` fallback)
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCKER_DIR = ROOT / "docker"
OSPREY_DOCKERFILE = DOCKER_DIR / "Dockerfile.osprey"
OSPREY_DOCKER_IMAGE = os.environ.get("OSPREY_DOCKER_IMAGE", "osprey-bench-osprey:latest")
OSPREY_DOCKER_CONTAINER = "osprey-bench-osprey"
OSPREY_DOCKER_PORT = 16333
OSPREY_DOCKER_FLIGHT_PORT = 18815
OSPREY_DOCKER_DATA = DOCKER_DIR / "osprey-data"

OSI_HEADER_STRUCT = struct.Struct("<4sBBHHHIQQQQQQ")
OSI_MAGIC = b"OSPR"
OSI_FORMAT_KIND = 1
OSI_FORMAT_VERSION = 1

# Mirrors `osprey_core::search::SearchOptions::resolve_for_bits`
LOW_DIM_OVERSAMPLE_THRESHOLD = 256
DEFAULT_LOW_DIM_OVERSAMPLE_MULTIPLIER = 2
LOW_DIM_REFINE_POOL = 128
LOW_DIM_DEFAULT_CANDIDATES = 2000
HIGH_DIM_DEFAULT_CANDIDATES = 300
FOUR_BIT_LOW_DIM_DEFAULT_CANDIDATES = 100
FOUR_BIT_HIGH_DIM_DEFAULT_CANDIDATES = 40
MAX_STAGE1_CANDIDATES = 4096
OSI_FLAG_RABITQ_4BIT = 0x04


# ---------------------------------------------------------------------------
# Shared OS, cgroup, Docker, and Rust HTTP load helpers
# ---------------------------------------------------------------------------
def dir_size_mb(path: Path) -> float:
    """Measure allocated on-disk bytes (`st_blocks * 512`) to avoid overcounting sparse holes."""
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
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


def container_peak_memory_kb(name: str) -> int:
    """Peak cgroup memory for a container, including its page cache.

    Process ``VmHWM`` undercounts page-cache resident indexes (notably
    pgvector). cgroup v2 ``memory.peak`` is the comparable container metric.
    """
    try:
        cid = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.Id}}", name],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        candidates = [
            Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.peak"),
            Path(f"/sys/fs/cgroup/docker/{cid}/memory.peak"),
            Path(
                f"/sys/fs/cgroup/memory/docker/{cid}/memory.max_usage_in_bytes"
            ),
        ]
        for path in candidates:
            if path.exists():
                return int(path.read_text().strip()) // 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return container_max_vmhwm_kb(name)


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


def drop_docker_dir_page_cache(
    host_dir: Path, image: str, mount_path: str = "/data"
) -> None:
    """Evict host page cache on root-owned Docker data directories via `dd iflag=nocache`."""
    if not host_dir.exists():
        return
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0:0",
            "-v",
            f"{host_dir.resolve()}:{mount_path}",
            "--entrypoint=",
            image,
            "sh",
            "-c",
            (
                f"find {mount_path} -type f -exec "
                "dd if={} of=/dev/null iflag=nocache count=0 status=none \\;"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(
            f"  [warn] root container page-cache eviction failed for {host_dir} "
            f"(exit={proc.returncode}): {proc.stderr.strip()}",
            flush=True,
        )


def run_rust_http_load(
    runner_bin: str,
    url: str,
    engine: str,
    payloads: Sequence[bytes],
    workers: int,
    warmup_payloads: Optional[Sequence[bytes]] = None,
) -> Tuple[List[List[int]], List[float], List[float], float]:
    """Run sequential (workers=1) or concurrent (workers=N) HTTP load via Rust."""
    with tempfile.TemporaryDirectory(prefix="osprey_http_load_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload_path = tmp_path / f"http_load_{engine}.jsonl"
        out_path = tmp_path / f"http_load_{engine}_out.json"
        with open(payload_path, "wb") as fh:
            for p in payloads:
                fh.write(p)
                fh.write(b"\n")
        cmd = [
            runner_bin,
            "http-load",
            "--url",
            url,
            "--engine",
            engine,
            "--payloads-file",
            str(payload_path),
            "--workers",
            str(workers),
            "--out-json",
            str(out_path),
        ]
        if warmup_payloads:
            warmup_path = tmp_path / f"http_load_{engine}_warmup.jsonl"
            with open(warmup_path, "wb") as fh:
                for p in warmup_payloads:
                    fh.write(p)
                    fh.write(b"\n")
            cmd.extend(["--warmup-payloads-file", str(warmup_path)])
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        report = json.loads(out_path.read_text())
        return (
            [[int(x) for x in row] for row in report["doc_ids"]],
            [float(x) for x in report["client_ms"]],
            [float(x) for x in report["server_ms"]],
            float(report["wall_s"]),
        )


# ---------------------------------------------------------------------------
# .osi header parsing and ClusterManifest generation
# ---------------------------------------------------------------------------
def read_osi_header(osi_path: Path) -> Dict[str, Any]:
    """Parse the 64-byte `OsiHeader` from a `.osi` artifact."""
    with open(osi_path, "rb") as fh:
        raw = fh.read(OSI_HEADER_STRUCT.size)
    if len(raw) < OSI_HEADER_STRUCT.size:
        raise ValueError(
            f"{osi_path}: expected at least {OSI_HEADER_STRUCT.size} bytes, got {len(raw)}"
        )
    (
        magic,
        format_kind,
        flags,
        version,
        orig_dim,
        padded_dim,
        num_clusters,
        num_vectors,
        fwht_seed,
        centroids_offset,
        entries_offset,
        tombstones_offset,
        header_xxh64,
    ) = OSI_HEADER_STRUCT.unpack(raw)
    if magic != OSI_MAGIC:
        raise ValueError(f"{osi_path}: invalid magic {magic!r}, expected {OSI_MAGIC!r}")
    if format_kind != OSI_FORMAT_KIND:
        raise ValueError(
            f"{osi_path}: invalid format_kind {format_kind}, expected {OSI_FORMAT_KIND}"
        )
    if version != OSI_FORMAT_VERSION:
        raise ValueError(
            f"{osi_path}: unsupported version {version}, expected {OSI_FORMAT_VERSION}"
        )
    return {
        "flags": int(flags),
        "version": int(version),
        "orig_dim": int(orig_dim),
        "padded_dim": int(padded_dim),
        "num_clusters": int(num_clusters),
        "num_vectors": int(num_vectors),
        "fwht_seed": int(fwht_seed),
        "centroids_offset": int(centroids_offset),
        "entries_offset": int(entries_offset),
        "tombstones_offset": int(tombstones_offset),
        "header_xxh64": int(header_xxh64),
    }


def build_cluster_manifest(
    osi_path: Path,
    osi_key: str = "segment.osi",
    osv_key: str = "segment.osv",
    segment_id: int = 1,
    updated_at_unix: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a `ClusterManifest` v1 JSON structure from an `.osi` segment header."""
    hdr = read_osi_header(osi_path)
    ts = int(time.time()) if updated_at_unix is None else int(updated_at_unix)
    return {
        "schema_version": 1,
        "epoch": 1,
        "updated_at_unix": ts,
        "next_segment_id": int(segment_id) + 1,
        "segments": [
            {
                "segment_id": int(segment_id),
                "osi_key": osi_key,
                "osv_key": osv_key,
                "orig_dim": hdr["orig_dim"],
                "padded_dim": hdr["padded_dim"],
                "fwht_seed": hdr["fwht_seed"],
                "num_vectors": hdr["num_vectors"],
                "tombstones_b64": "",
                "tombstone_count": 0,
            }
        ],
        "global_tombstones_b64": "",
    }


def write_cluster_manifest(
    store_dir: Path,
    osi_path: Path,
    osi_key: str = "segment.osi",
    osv_key: str = "segment.osv",
    segment_id: int = 1,
    updated_at_unix: Optional[int] = None,
) -> Path:
    """Write `manifest.json` inside `store_dir` matching `ospreydb::ClusterManifest`."""
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_cluster_manifest(
        osi_path=osi_path,
        osi_key=osi_key,
        osv_key=osv_key,
        segment_id=segment_id,
        updated_at_unix=updated_at_unix,
    )
    manifest_path = store_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Adaptive SearchOptions::resolve & runner probing
# ---------------------------------------------------------------------------
def resolve_search_params(
    dim: int,
    nprobe: int,
    refine_m: int,
    top_k: int = 10,
    refine_multiplier: int = 0,
    bits: int = 1,
) -> Dict[str, Any]:
    """Mirror `osprey_core::search::SearchOptions::resolve_for_bits` with Stage-1 cap 4096."""
    low_dim = int(dim) <= LOW_DIM_OVERSAMPLE_THRESHOLD
    is_4bit = int(bits) >= 4
    if int(refine_multiplier) == 0:
        resolved_mult = DEFAULT_LOW_DIM_OVERSAMPLE_MULTIPLIER if (low_dim and not is_4bit) else 1
    else:
        resolved_mult = max(1, int(refine_multiplier))

    auto_candidates = int(refine_m) == 0
    if auto_candidates:
        if is_4bit:
            base_candidates = (
                FOUR_BIT_LOW_DIM_DEFAULT_CANDIDATES
                if low_dim
                else FOUR_BIT_HIGH_DIM_DEFAULT_CANDIDATES
            )
        else:
            base_candidates = (
                LOW_DIM_DEFAULT_CANDIDATES if low_dim else HIGH_DIM_DEFAULT_CANDIDATES
            )
    else:
        base_candidates = int(refine_m)

    low_dim_oversample = (not is_4bit) and low_dim and resolved_mult > 1
    refine_floor = LOW_DIM_REFINE_POOL if low_dim_oversample else 0
    refine_limit = max(int(top_k), refine_floor)
    effective_candidates = min(base_candidates * resolved_mult, MAX_STAGE1_CANDIDATES)
    stage1_limit = min(max(effective_candidates, refine_limit), MAX_STAGE1_CANDIDATES)
    effective_nprobe = int(nprobe) * resolved_mult

    return {
        "refine_multiplier": resolved_mult,
        "effective_nprobe": effective_nprobe,
        "effective_candidates": effective_candidates,
        "refine_limit": refine_limit,
        "stage1_limit": stage1_limit,
        "low_dim_oversample": low_dim_oversample,
        "auto_candidates": auto_candidates,
    }


def runner_supports_http_engine(runner_bin: str, engine: str = "osprey_server") -> bool:
    """Probe whether `runner_bin http-load` documents support for `engine`."""
    if not runner_bin:
        return False
    try:
        proc = subprocess.run(
            [runner_bin, "http-load", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return False
        combined = f"{proc.stdout}\n{proc.stderr}"
        return engine in combined
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_osprey_docker_image(
    dockerfile: Path = OSPREY_DOCKERFILE,
    context_dir: Path = DOCKER_DIR,
    image: str = OSPREY_DOCKER_IMAGE,
) -> None:
    """Build `osprey-bench-osprey` image from `docker/Dockerfile.osprey` if absent."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image,
            str(context_dir),
        ],
        check=True,
    )


def stop_osprey_docker_container(name: str = OSPREY_DOCKER_CONTAINER) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# OspreyDockerServerEngine
# ---------------------------------------------------------------------------
class OspreyDockerServerEngine:
    """Manages production `osprey serve` inside a symmetric Docker container."""

    def __init__(
        self,
        dataset: str,
        dim: int,
        osprey_bin: str,
        runner_bin: str = "",
        data_dir: Path = OSPREY_DOCKER_DATA,
        container: str = OSPREY_DOCKER_CONTAINER,
        image: str = OSPREY_DOCKER_IMAGE,
        port: int = OSPREY_DOCKER_PORT,
        flight_port: int = OSPREY_DOCKER_FLIGHT_PORT,
        collection: Optional[str] = None,
        role: str = "unified",
    ):
        self.dataset = dataset
        self.dim = int(dim)
        self.osprey_bin = osprey_bin
        self.runner_bin = runner_bin
        self.data_dir = Path(data_dir)
        self.store_dir = self.data_dir / "store"
        self.wal_dir = self.data_dir / "wal"
        self.container = container
        self.image = image
        self.port = int(port)
        self.flight_port = int(flight_port)
        self.collection = collection or f"ctrl_{dataset}_osprey"
        self.role = role
        self.url = f"http://127.0.0.1:{self.port}"
        self.client: Optional[httpx.Client] = None
        self.fwht_seed: Optional[int] = None
        self.osi_meta: Optional[Dict[str, Any]] = None
        self.bits: int = 1
        self.load_time_s = 0.0
        self._rust_http_ok: Optional[bool] = None
        self._warned_httpx_fallback = False

    def uses_rust_http_load(self) -> bool:
        if self._rust_http_ok is None:
            self._rust_http_ok = runner_supports_http_engine(
                self.runner_bin, "osprey_server"
            )
            if not self._rust_http_ok and not self._warned_httpx_fallback:
                self._warned_httpx_fallback = True
                print(
                    "  [warn] osprey-bench-runner http-load does not advertise "
                    "--engine osprey_server; falling back to Python httpx client",
                    flush=True,
                )
        return self._rust_http_ok

    def prepare_data_dir(self) -> Tuple[Path, Path]:
        """Reset `docker/osprey-data/{store,wal}` cleanly before importing a dataset."""
        ensure_osprey_docker_image(image=self.image)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.data_dir, 0o777)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0:0",
                "-v",
                f"{self.data_dir.resolve()}:/data",
                "--entrypoint=",
                self.image,
                "sh",
                "-c",
                "rm -rf /data/* /data/.[!.]* && mkdir -p /data/store /data/wal && chmod 777 /data /data/store /data/wal",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        return self.store_dir, self.wal_dir

    def write_manifest(self) -> Path:
        osi_path = self.store_dir / "segment.osi"
        self.osi_meta = read_osi_header(osi_path)
        self.fwht_seed = int(self.osi_meta["fwht_seed"])
        self.bits = 4 if (int(self.osi_meta.get("flags", 0)) & OSI_FLAG_RABITQ_4BIT) != 0 else 1
        return write_cluster_manifest(self.store_dir, osi_path)

    def drop_page_cache(self) -> None:
        ensure_osprey_docker_image(image=self.image)
        drop_docker_dir_page_cache(self.data_dir, self.image, "/data")

    def _wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        wait_http(f"{self.url}/healthz", timeout=timeout)
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                resp = httpx.get(
                    f"{self.url}/collections/{self.collection}", timeout=3.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    segs = int(data.get("result", {}).get("segments_count", 0))
                    if segs >= 1:
                        return
                    last_err = f"segments_count={segs}"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            time.sleep(0.5)
        raise RuntimeError(
            f"timeout waiting for osprey_docker collection {self.collection}: {last_err}"
        )

    def start(self) -> float:
        """Create and start `osprey-bench-osprey`, waiting for `/healthz` and loaded segments."""
        stop_osprey_docker_container(self.container)
        ensure_osprey_docker_image(image=self.image)
        if self.fwht_seed is None and (self.store_dir / "segment.osi").exists():
            self.osi_meta = read_osi_header(self.store_dir / "segment.osi")
            self.fwht_seed = int(self.osi_meta["fwht_seed"])
            self.bits = 4 if (int(self.osi_meta.get("flags", 0)) & OSI_FLAG_RABITQ_4BIT) != 0 else 1
        if not (self.store_dir / "manifest.json").exists() and (
            self.store_dir / "segment.osi"
        ).exists():
            self.write_manifest()

        osprey_bin_resolved = str(Path(self.osprey_bin).resolve())
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container,
            "-p",
            f"{self.port}:6333",
            "-p",
            f"{self.flight_port}:8815",
            "-v",
            f"{self.data_dir.resolve()}:/data",
            "-v",
            f"{osprey_bin_resolved}:/usr/local/bin/osprey:ro",
            self.image,
            "serve",
            "--role",
            self.role,
            "--collection",
            self.collection,
            "--dim",
            str(self.dim),
            "--data-dir",
            "/data/wal",
            "--object-store-uri",
            "file:///data/store",
            "--http-addr",
            "0.0.0.0:6333",
            "--flight-addr",
            "0.0.0.0:8815",
        ]
        if self.fwht_seed is not None:
            cmd.extend(["--fwht-seed", str(self.fwht_seed)])

        started = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        try:
            self._wait_ready(timeout=300.0)
        except Exception:
            self.stop()
            raise
        self.load_time_s = time.perf_counter() - started
        self.client = httpx.Client(timeout=120.0)
        return self.load_time_s

    def restart(self) -> float:
        """Symmetric stop + `dd iflag=nocache` page-cache eviction + container start + `/healthz`."""
        if self.client is not None:
            self.client.close()
            self.client = None
        stop_container(self.container)
        self.drop_page_cache()
        started = time.perf_counter()
        start_container(self.container)
        try:
            self._wait_ready(timeout=300.0)
        except Exception:
            self.stop()
            raise
        self.client = httpx.Client(timeout=120.0)
        return time.perf_counter() - started

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        stop_osprey_docker_container(self.container)

    def vmhwm_kb(self) -> int:
        return container_max_vmhwm_kb(self.container)

    def cgroup_peak_kb(self) -> int:
        return container_peak_memory_kb(self.container)

    def disk_mb(self) -> float:
        return dir_size_mb(self.data_dir)

    def _search_url(self) -> str:
        return f"{self.url}/collections/{self.collection}/points/search"

    def _build_payload_dict(
        self,
        query: np.ndarray,
        nprobe: int,
        refine_m: int,
        top_k: int = 10,
        refine_multiplier: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        plan = resolve_search_params(
            self.dim,
            nprobe,
            refine_m,
            top_k=top_k,
            refine_multiplier=refine_multiplier,
            bits=self.bits,
        )
        vec = query.tolist() if hasattr(query, "tolist") else list(query)
        payload = {
            "vector": vec,
            "limit": int(top_k),
            "with_payload": False,
            "params": {
                "nprobe": int(plan["effective_nprobe"]),
                "hnsw_ef": int(plan["stage1_limit"]),
            },
        }
        return payload, plan

    def prepare(self, query: np.ndarray, cfg: Dict[str, Any]) -> bytes:
        payload, _plan = self._build_payload_dict(
            query,
            nprobe=int(cfg["nprobe"]),
            refine_m=int(cfg["refine_m"]),
            top_k=int(cfg.get("top_k", 10)),
            refine_multiplier=int(cfg.get("refine_multiplier", 0)),
        )
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _parse_response(body: Dict[str, Any]) -> Tuple[List[int], float]:
        ids = [int(p["id"]) for p in body["result"]]
        server_ms = float(body.get("time", 0.0)) * 1000.0
        return ids, server_ms

    def query_prepared(self, payload: bytes) -> Tuple[List[int], float, float]:
        assert self.client is not None
        started = time.perf_counter()
        resp = self.client.post(
            self._search_url(),
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        client_ms = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(f"Osprey Docker HTTP {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        ids, server_ms = self._parse_response(resp.json())
        return ids, client_ms, server_ms

    def query(
        self,
        query: np.ndarray,
        nprobe: int,
        refine_m: int,
        top_k: int = 10,
        refine_multiplier: int = 0,
    ) -> Tuple[List[int], float, float, Dict[str, Any]]:
        assert self.client is not None
        payload, plan = self._build_payload_dict(
            query,
            nprobe=nprobe,
            refine_m=refine_m,
            top_k=top_k,
            refine_multiplier=refine_multiplier,
        )
        started = time.perf_counter()
        resp = self.client.post(self._search_url(), json=payload)
        client_ms = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Osprey Docker HTTP {resp.status_code} for nprobe={nprobe} "
                f"refine_m={refine_m}: {resp.text}"
            )
        resp.raise_for_status()
        ids, server_ms = self._parse_response(resp.json())
        return ids, client_ms, server_ms, plan

    async def _concurrent_payloads(
        self,
        payloads: Sequence[bytes],
        workers: int,
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float], float]:
        limits = httpx.Limits(
            max_connections=workers, max_keepalive_connections=workers
        )
        results: List[Optional[Tuple[List[int], float, float]]] = [None] * len(payloads)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(payloads)):
            queue.put_nowait(idx)

        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
            if warmup_payloads:
                for p in list(warmup_payloads)[:workers]:
                    resp = await client.post(
                        self._search_url(),
                        content=p,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()

            async def worker() -> None:
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    started = time.perf_counter()
                    resp = await client.post(
                        self._search_url(),
                        content=payloads[idx],
                        headers={"Content-Type": "application/json"},
                    )
                    client_ms = (time.perf_counter() - started) * 1000.0
                    resp.raise_for_status()
                    ids, server_ms = self._parse_response(resp.json())
                    results[idx] = (ids, client_ms, server_ms)

            started = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(workers)])
            wall_s = time.perf_counter() - started

        ids = [r[0] for r in results if r is not None]
        client_ms = [r[1] for r in results if r is not None]
        server_ms = [r[2] for r in results if r is not None]
        return ids, client_ms, server_ms, wall_s

    def _run_rust_with_fallback(
        self,
        payloads: Sequence[bytes],
        workers: int,
        warmup_payloads: Optional[Sequence[bytes]],
    ) -> Optional[Tuple[List[List[int]], List[float], List[float], float]]:
        if not self.uses_rust_http_load():
            return None
        try:
            return run_rust_http_load(
                self.runner_bin,
                self._search_url(),
                "osprey_server",
                payloads,
                workers,
                warmup_payloads=warmup_payloads,
            )
        except subprocess.CalledProcessError as exc:
            err_text = f"{exc.stderr or ''}\n{exc.stdout or ''}"
            if "unsupported --engine" in err_text or "osprey_server" in err_text:
                self._rust_http_ok = False
                if not self._warned_httpx_fallback:
                    self._warned_httpx_fallback = True
                    print(
                        "  [warn] osprey-bench-runner rejected --engine osprey_server; "
                        "falling back to Python httpx client",
                        flush=True,
                    )
                return None
            raise

    def sequential_prepared(
        self,
        payloads: Sequence[bytes],
        warmup_payloads: Optional[Sequence[bytes]] = None,
    ) -> Tuple[List[List[int]], List[float], List[float]]:
        rust_res = self._run_rust_with_fallback(payloads, 1, warmup_payloads)
        if rust_res is not None:
            ids, client_ms, server_ms, _wall = rust_res
            return ids, client_ms, server_ms

        if warmup_payloads and self.client is not None:
            for p in list(warmup_payloads)[:4]:
                self.query_prepared(p)
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
        rust_res = self._run_rust_with_fallback(payloads, workers, warmup_payloads)
        if rust_res is not None:
            return rust_res
        return asyncio.run(
            self._concurrent_payloads(
                payloads, workers, warmup_payloads=warmup_payloads
            )
        )
