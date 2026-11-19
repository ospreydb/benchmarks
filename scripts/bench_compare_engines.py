#!/usr/bin/env python3
"""
Task 173: Comparative Benchmark — OspreyDB vs Qdrant (HNSW) vs pgvector (HNSW).

Measures for each engine on identical prepared datasets (e.g. SIFT1M, Cohere 768d, GIST):
  - Index build time (seconds)
  - Peak resident RAM (VmHWM / RSS in MiB)
  - Disk footprint (MiB) & resident index RAM (MiB)
  - Recall@10, QPS, and Latency (P50 / P95 / P99 ms)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"


def get_container_rss_mb(container_name: str) -> float:
    """Reads live RSS (VmRSS / memory.current) of a running Docker container in MiB."""
    try:
        out = subprocess.check_output(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                container_name,
            ],
            text=True,
            timeout=10,
        ).strip()
        # Format: "123.4MiB / 31.3GiB" or "1.23GiB / 31.3GiB"
        used = out.split("/")[0].strip()
        m = re.match(r"([0-9.]+)\s*([KMG]i?B)", used, re.IGNORECASE)
        if not m:
            return 0.0
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("k"):
            return val / 1024.0
        if unit.startswith("m"):
            return val
        if unit.startswith("g"):
            return val * 1024.0
    except Exception:
        pass
    return 0.0


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


def measure_cmd_peak_rss(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[float, float]:
    """Runs subprocess with /usr/bin/time -v (if present) and returns (elapsed_s, peak_rss_mb)."""
    t0 = time.perf_counter()
    time_bin = "/usr/bin/time"
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if os.path.exists(time_bin):
        proc = subprocess.run(
            [time_bin, "-v", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", proc.stderr)
        rss_mb = (int(m.group(1)) / 1024.0) if m else 0.0
        return elapsed, rss_mb

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=merged_env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return elapsed, 0.0


def build_osprey_index(
    ds_dir: Path,
    osprey_bin: str,
    clusters: int,
    art_dir: Path,
) -> Tuple[float, float]:
    art_dir.mkdir(parents=True, exist_ok=True)
    build_s, build_rss_mb = measure_cmd_peak_rss(
        [
            osprey_bin,
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
            str(clusters),
            "--json",
        ]
    )
    return build_s, build_rss_mb


def eval_osprey(
    dataset: str,
    ds_dir: Path,
    runner_bin: str,
    art_dir: Path,
    nprobe: int,
    refine_m: int,
    refine_multiplier: int,
    max_queries: int,
    build_s: float,
    build_rss_mb: float,
) -> Dict[str, object]:
    queries_npy = ds_dir / "queries.npy"
    gt_npy = ds_dir / "gt_ip.npy"

    out_json = art_dir / f"osprey_eval_rm{refine_multiplier}.json"
    subprocess.run(
        [
            runner_bin,
            "search-eval",
            "--osi",
            str(art_dir / "segment.osi"),
            "--slabs",
            str(art_dir / "segment.osv"),
            "--queries-npy",
            str(queries_npy),
            "--gt-npy",
            str(gt_npy),
            "--nprobe-list",
            str(nprobe),
            "--refine-list",
            str(refine_m),
            "--refine-multiplier",
            str(refine_multiplier),
            "--top-k",
            "10",
            "--max-queries",
            str(max_queries),
            "--cache-mb",
            "16",
            "--out-json",
            str(out_json),
        ],
        check=True,
    )
    res = json.loads(out_json.read_text())[0]
    osi_mb = res["osi_bytes"] / (1024.0 * 1024.0)
    osv_mb = res["osv_bytes"] / (1024.0 * 1024.0)

    if refine_multiplier == 0:
        refine_policy = "adaptive"
    elif refine_multiplier == 1:
        refine_policy = "baseline"
    else:
        refine_policy = f"forced-{refine_multiplier}"

    return {
        "dataset": dataset,
        "engine": "OspreyDB (1-bit .osi + .osv)",
        "build_time_s": round(build_s, 3),
        "peak_rss_mb": round(max(res["vm_hwm_mb"], osi_mb), 2),
        "build_peak_rss_mb": round(build_rss_mb, 2),
        "ram_index_mb": round(osi_mb, 2),
        "disk_size_mb": round(osi_mb + osv_mb, 2),
        "recall_at_10": round(res["recall_at_10"], 4),
        "p50_ms": round(res["p50_ms"], 3),
        "p95_ms": round(res["p95_ms"], 3),
        "p99_ms": round(res["p99_ms"], 3),
        "qps": round(res["qps"], 1),
        "refine_policy": refine_policy,
        "requested_refine_multiplier": refine_multiplier,
        "refine_multiplier": res["refine_multiplier"],
        "orig_dim": res["orig_dim"],
        "requested_nprobe": res["requested_nprobe"],
        "effective_nprobe": res["effective_nprobe"],
        "requested_candidates": res["requested_candidates"],
        "effective_candidates": res["effective_candidates"],
        "refine_limit": res["refine_limit"],
        "stage1_limit": res["stage1_limit"],
        "low_dim_oversample": res["low_dim_oversample"],
        "auto_candidates": res["auto_candidates"],
        "candidates_mode": "auto" if res["auto_candidates"] else "explicit",
    }


def bench_qdrant(
    dataset: str,
    ds_dir: Path,
    max_queries: int,
    qdrant_url: str = "http://127.0.0.1:6333",
) -> Dict[str, object]:
    base = np.load(ds_dir / "base.npy")
    queries = np.load(ds_dir / "queries.npy")[:max_queries]
    gt = np.load(ds_dir / "gt_ip.npy")[:max_queries, :10]

    n_vecs, dim = base.shape
    client = QdrantClient(url=qdrant_url, timeout=120)
    coll = f"bench_{dataset}"
    if client.collection_exists(coll):
        client.delete_collection(coll)

    rss_before = get_container_rss_mb("osprey-bench-qdrant")
    t0 = time.perf_counter()
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    batch_size = 2000
    for start in range(0, n_vecs, batch_size):
        end = min(start + batch_size, n_vecs)
        points = [
            PointStruct(id=int(i), vector=base[i].tolist()) for i in range(start, end)
        ]
        client.upsert(collection_name=coll, points=points, wait=True)

    # Wait until HNSW optimizer finishes indexing
    while True:
        info = client.get_collection(coll)
        if str(info.status).lower().endswith("green"):
            break
        time.sleep(0.5)
    build_s = time.perf_counter() - t0
    rss_after = get_container_rss_mb("osprey-bench-qdrant")

    latencies_ms: List[float] = []
    recall_hits = 0
    wall_t0 = time.perf_counter()
    for qi in range(len(queries)):
        qt0 = time.perf_counter()
        resp = client.query_points(
            collection_name=coll,
            query=queries[qi].tolist(),
            limit=10,
            with_payload=False,
        )
        latencies_ms.append((time.perf_counter() - qt0) * 1000.0)
        got = {int(p.id) for p in resp.points}
        gt_set = {int(x) for x in gt[qi]}
        recall_hits += len(got & gt_set)
    wall_s = time.perf_counter() - wall_t0

    rss_query = get_container_rss_mb("osprey-bench-qdrant")
    disk_mb = dir_size_mb(ROOT / "docker" / "qdrant-data" / "collections" / coll)
    arr = np.sort(np.array(latencies_ms))

    client.delete_collection(coll)
    return {
        "dataset": dataset,
        "engine": "Qdrant (HNSW FP32)",
        "build_time_s": round(build_s, 3),
        "peak_rss_mb": round(max(rss_after, rss_query, rss_after - rss_before), 2),
        "build_peak_rss_mb": round(rss_after, 2),
        "ram_index_mb": round(max(rss_after - rss_before, 0.0), 2),
        "disk_size_mb": round(disk_mb, 2),
        "recall_at_10": round(recall_hits / (len(queries) * 10.0), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "qps": round(len(queries) / max(wall_s, 1e-9), 1),
    }


def bench_pgvector(
    dataset: str,
    ds_dir: Path,
    max_queries: int,
    dsn: str = "postgresql://osprey:ospreypassword123@127.0.0.1:5433/osprey_bench",
) -> Dict[str, object]:
    base = np.load(ds_dir / "base.npy")
    queries = np.load(ds_dir / "queries.npy")[:max_queries]
    gt = np.load(ds_dir / "gt_ip.npy")[:max_queries, :10]
    n_vecs, dim = base.shape

    table = f"vecs_{dataset}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"DROP TABLE IF EXISTS {table};")
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, emb vector({dim}));")
            cur.execute("SET maintenance_work_mem = '1GB';")

        rss_before = get_container_rss_mb("osprey-bench-pgvector")
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            with cur.copy(f"COPY {table} (id, emb) FROM STDIN") as copy:
                for i in range(n_vecs):
                    vec_str = "[" + ",".join(f"{v:.6f}" for v in base[i].tolist()) + "]"
                    copy.write_row((i, vec_str))
            cur.execute(
                f"CREATE INDEX {table}_hnsw_idx ON {table} USING hnsw (emb vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64);"
            )
        build_s = time.perf_counter() - t0
        rss_after = get_container_rss_mb("osprey-bench-pgvector")

        latencies_ms: List[float] = []
        recall_hits = 0
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 64;")
            wall_t0 = time.perf_counter()
            for qi in range(len(queries)):
                q_str = "[" + ",".join(f"{v:.6f}" for v in queries[qi].tolist()) + "]"
                qt0 = time.perf_counter()
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY emb <=> %s::vector LIMIT 10;",
                    (q_str,),
                )
                rows = cur.fetchall()
                latencies_ms.append((time.perf_counter() - qt0) * 1000.0)
                got = {int(r[0]) for r in rows}
                gt_set = {int(x) for x in gt[qi]}
                recall_hits += len(got & gt_set)
            wall_s = time.perf_counter() - wall_t0

            cur.execute(f"SELECT pg_total_relation_size('{table}'), pg_relation_size('{table}_hnsw_idx');")
            total_bytes, idx_bytes = cur.fetchone()
            cur.execute(f"DROP TABLE IF EXISTS {table};")

    rss_query = get_container_rss_mb("osprey-bench-pgvector")
    arr = np.sort(np.array(latencies_ms))
    return {
        "dataset": dataset,
        "engine": "pgvector (HNSW)",
        "build_time_s": round(build_s, 3),
        "peak_rss_mb": round(max(rss_after, rss_query, rss_after - rss_before), 2),
        "build_peak_rss_mb": round(rss_after, 2),
        "ram_index_mb": round(idx_bytes / (1024.0 * 1024.0), 2),
        "disk_size_mb": round(total_bytes / (1024.0 * 1024.0), 2),
        "recall_at_10": round(recall_hits / (len(queries) * 10.0), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "qps": round(len(queries) / max(wall_s, 1e-9), 1),
    }


def main() -> int:
    # Keep progress visible when stdout is piped (e.g. `make ... | tee run.log`).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sift1m", "cohere768d"])
    parser.add_argument("--osprey-bin", required=True)
    parser.add_argument("--runner-bin", required=True)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument(
        "--refine-m",
        type=int,
        default=None,
        help=(
            "Explicit Stage-1 candidate pool. Omit to use the adaptive auto default "
            "(2000 for dim<=256, 300 otherwise)."
        ),
    )
    parser.add_argument(
        "--refine-multiplier",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Adaptive oversampling policy: 0 = adaptive (2x + 128 f16 floor for dim<=256), "
            "1 = disabled/baseline, N>1 = forced Nx. May pass several values to run an A/B sweep."
        ),
    )
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument(
        "--controlled",
        action="store_true",
        help="Run the recall-target controlled protocol (HTTP transport, warmup, 3 rounds, concurrent QPS).",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        nargs="+",
        default=[0.95, 0.97],
        help="Recall@10 targets calibrated on the validation split.",
    )
    parser.add_argument(
        "--engines",
        default="osprey,qdrant_fp32,qdrant_sq8,qdrant_bq,pgvector,pgvector_halfvec",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--calib-queries",
        type=int,
        default=0,
        help="Validation queries for recall calibration (0 = 25%% of the query set, minimum 200).",
    )
    parser.add_argument("--warmup-queries", type=int, default=200)
    parser.add_argument("--measure-queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--osprey-port", type=int, default=18080)
    parser.add_argument("--controlled-cache-mb", type=int, default=256)
    parser.add_argument(
        "--qdrant-on-disk",
        action="store_true",
        help="Store Qdrant vectors on disk (VectorParams.on_disk=True) instead of in-RAM.",
    )
    parser.add_argument("--quick", action="store_true", help="Small grids/splits for smoke testing.")
    parser.add_argument(
        "--out-controlled-csv",
        default=str(RESULTS_DIR / "compare_engines_controlled.csv"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(RESULTS_DIR / "compare_engines.csv"),
        help="Output CSV (use results/compare_engines_adaptive.csv to preserve the published file).",
    )
    parser.add_argument("--skip-pgvector", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()

    if args.controlled:
        from bench_controlled import run_controlled

        return run_controlled(args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    refine_m = args.refine_m if args.refine_m is not None else 0
    refine_m_label = "auto" if refine_m == 0 else str(refine_m)

    for ds in args.datasets:
        ds_dir = PREP_DIR / ds
        if not (ds_dir / "base.npy").exists():
            print(f"[skip] {ds}: {ds_dir / 'base.npy'} not found")
            continue

        art_dir = ARTIFACTS_DIR / "compare" / ds
        print(f"=== Building OspreyDB index on {ds} ===")
        build_s, build_rss_mb = build_osprey_index(
            ds_dir, args.osprey_bin, args.clusters, art_dir
        )

        for refine_multiplier in args.refine_multiplier:
            print(
                f"=== Benchmarking OspreyDB on {ds} "
                f"(nprobe={args.nprobe}, refine_m={refine_m_label}, "
                f"refine_multiplier={refine_multiplier}) ==="
            )
            rows.append(
                eval_osprey(
                    ds,
                    ds_dir,
                    args.runner_bin,
                    art_dir,
                    args.nprobe,
                    refine_m,
                    refine_multiplier,
                    args.max_queries,
                    build_s,
                    build_rss_mb,
                )
            )

        if not args.skip_qdrant:
            print(f"=== Benchmarking Qdrant on {ds} ===")
            rows.append(bench_qdrant(ds, ds_dir, args.max_queries))

        if not args.skip_pgvector:
            print(f"=== Benchmarking pgvector on {ds} ===")
            rows.append(bench_pgvector(ds, ds_dir, args.max_queries))

    out_csv = Path(args.out_csv)
    if rows:
        fieldnames: List[str] = []
        for r in rows:
            for key in r:
                if key not in fieldnames:
                    fieldnames.append(key)
        normalized = [{k: r.get(k, "") for k in fieldnames} for r in rows]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
        print(f"[saved] {out_csv}")
        for r in normalized:
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
