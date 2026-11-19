#!/usr/bin/env python3
"""
Tasks 172 & 178: MinIO + tc netem S3 Latency Simulation & DP Range Coalescing Pareto Sweep.

Uploads `.osv` slabs to local MinIO (behind `minio-netem` in `docker/docker-compose.yml`,
which applies `tc qdisc replace dev eth0 root netem delay ${NETEM_DELAY}` with default `15ms`),
verifies TTFB latency, and sweeps `(nprobe, refine_m)` with DP range coalescing ON vs OFF
across sequential (`--workers 1`) and optional concurrent (`--concurrent-workers`) load
to produce `Recall@10 vs Latency (ms) vs S3 Range-GETs` Pareto frontiers.
"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import boto3
import numpy as np
from botocore.client import Config

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

METRIC_SCHEMA_KEYS = (
    "nprobe",
    "refine_m",
    "requested_nprobe",
    "effective_nprobe",
    "requested_candidates",
    "effective_candidates",
    "refine_limit",
    "stage1_limit",
    "refine_multiplier",
    "low_dim_oversample",
    "auto_candidates",
    "orig_dim",
    "coalesce_enabled",
    "queries_evaluated",
    "recall_at_10",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "qps",
    "stage1_mean_ms",
    "range_plan_mean_us",
    "stage2_3_mean_ms",
    "avg_raw_pages",
    "avg_s3_gets",
    "coalesce_factor",
    "avg_bytes_fetched",
    "osi_bytes",
    "osv_bytes",
    "vm_hwm_mb",
    "simd_tier",
)


def parse_int_csv(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def percentile(sorted_vals: Sequence[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = ((len(sorted_vals) - 1) * p) // 100
    return float(sorted_vals[idx])


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_proc_vm_hwm_mb(pid: int) -> float:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return 0.0
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return 0.0
    return 0.0


def ensure_minio_bucket_and_upload(
    endpoint: str,
    bucket: str,
    local_osv: Path,
    object_key: str,
) -> float:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="osprey",
        aws_secret_access_key="ospreypassword123",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)

    print(f"[minio] uploading {local_osv} -> s3://{bucket}/{object_key}")
    s3.upload_file(str(local_osv), bucket, object_key)

    # Probe TTFB on 64-byte Range-GET to confirm tc netem delay
    rtts = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = s3.get_object(Bucket=bucket, Key=object_key, Range="bytes=0-63")
        _ = resp["Body"].read()
        rtts.append((time.perf_counter() - t0) * 1000.0)
    mean_rtt = sum(rtts) / len(rtts)
    print(f"[minio] verified S3 64B Range-GET mean RTT (tc netem): {mean_rtt:.2f} ms")
    return mean_rtt


def run_search_eval(
    runner_bin: str,
    osi_path: Path,
    slabs_uri: str,
    ds_dir: Path,
    nprobe_list: str,
    refine_list: str,
    refine_multiplier: int,
    max_queries: int,
    no_coalesce: bool,
    out_json: Path,
    env: Dict[str, str],
) -> List[Dict[str, object]]:
    cmd = [
        runner_bin,
        "search-eval",
        "--osi",
        str(osi_path),
        "--slabs",
        slabs_uri,
        "--queries-npy",
        str(ds_dir / "queries.npy"),
        "--gt-npy",
        str(ds_dir / "gt_ip.npy"),
        "--nprobe-list",
        nprobe_list,
        "--refine-list",
        refine_list,
        "--refine-multiplier",
        str(refine_multiplier),
        "--top-k",
        "10",
        "--max-queries",
        str(max_queries),
        "--cache-mb",
        "0",
        "--out-json",
        str(out_json),
    ]
    if no_coalesce:
        cmd.append("--no-coalesce")
    subprocess.run(cmd, env=env, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def start_osprey_server(
    runner_bin: str,
    osi_path: Path,
    s3_uri: str,
    no_coalesce: bool,
    env: Dict[str, str],
    log_path: Path,
) -> Tuple[subprocess.Popen, str]:
    port = pick_free_port()
    bind = f"127.0.0.1:{port}"
    cmd = [
        runner_bin,
        "serve",
        "--osi",
        str(osi_path),
        "--slabs",
        s3_uri,
        "--bind",
        bind,
        "--cache-mb",
        "0",
    ]
    if no_coalesce:
        cmd.append("--no-coalesce")

    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_fh.close()
    health_url = f"http://{bind}/health"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_txt = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"osprey-bench-runner serve exited prematurely ({proc.returncode}): {stderr_txt}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    return proc, f"http://{bind}/search"
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.1)

    proc.terminate()
    proc.wait(timeout=5)
    raise RuntimeError(f"Timed out waiting for osprey-bench-runner serve on {health_url}")


def compute_recall_at_10(doc_ids_per_query: Sequence[Sequence[int]], gt_top10: np.ndarray) -> float:
    hits = 0
    denom = 0
    for preds, gt_row in zip(doc_ids_per_query, gt_top10):
        gt_set = {int(v) for v in gt_row if int(v) >= 0}
        hits += sum(1 for doc_id in preds[:10] if int(doc_id) in gt_set)
        denom += len(gt_set)
    return float(hits) / float(max(denom, 1))


def run_concurrent_eval(
    runner_bin: str,
    osi_path: Path,
    osv_path: Path,
    s3_uri: str,
    ds_dir: Path,
    art_dir: Path,
    nprobe_list: str,
    refine_list: str,
    refine_multiplier: int,
    max_queries: int,
    no_coalesce: bool,
    workers: int,
    env: Dict[str, str],
    meta_cache: Dict[Tuple[int, bool, int, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    """Evaluate S3 Range-GET coalescing under concurrent HTTP worker load."""
    coalesce_enabled = not no_coalesce
    nprobes = parse_int_csv(nprobe_list)
    refines = parse_int_csv(refine_list)

    # Populate deterministic per-point planner/GET/artifact metadata from local slab if not cached.
    missing_points = [
        (np_val, rm_val)
        for np_val in nprobes
        for rm_val in refines
        if (refine_multiplier, coalesce_enabled, np_val, rm_val) not in meta_cache
    ]
    if missing_points:
        local_meta_json = art_dir / f"pareto_local_meta_c{int(coalesce_enabled)}_rm{refine_multiplier}.json"
        local_rows = run_search_eval(
            runner_bin,
            osi_path,
            str(osv_path),
            ds_dir,
            nprobe_list,
            refine_list,
            refine_multiplier,
            max_queries,
            no_coalesce,
            local_meta_json,
            env,
        )
        for pt in local_rows:
            meta_cache[(refine_multiplier, coalesce_enabled, int(pt["nprobe"]), int(pt["refine_m"]))] = pt

    queries = np.load(ds_dir / "queries.npy")
    gt = np.load(ds_dir / "gt_ip.npy")
    q_count = min(int(queries.shape[0]), int(gt.shape[0]), max_queries)
    gt_top10 = gt[:q_count, : min(10, int(gt.shape[1]))]
    query_lists = [queries[i].tolist() for i in range(q_count)]

    serve_log = art_dir / f"serve_w{workers}_c{int(coalesce_enabled)}_rm{refine_multiplier}.log"
    proc, search_url = start_osprey_server(
        runner_bin, osi_path, s3_uri, no_coalesce, env, serve_log
    )
    results: List[Dict[str, object]] = []
    try:
        for nprobe in nprobes:
            for refine_m in refines:
                payload_path = (
                    art_dir
                    / f"payloads_w{workers}_c{int(coalesce_enabled)}_rm{refine_multiplier}_np{nprobe}_m{refine_m}.jsonl"
                )
                http_out = (
                    art_dir
                    / f"http_w{workers}_c{int(coalesce_enabled)}_rm{refine_multiplier}_np{nprobe}_m{refine_m}.json"
                )
                with open(payload_path, "w", encoding="utf-8") as fh:
                    for q_vec in query_lists:
                        fh.write(
                            json.dumps(
                                {
                                    "query": q_vec,
                                    "nprobe": nprobe,
                                    "refine_m": refine_m,
                                    "top_k": 10,
                                    "refine_multiplier": refine_multiplier,
                                }
                            )
                            + "\n"
                        )

                subprocess.run(
                    [
                        runner_bin,
                        "http-load",
                        "--url",
                        search_url,
                        "--engine",
                        "osprey",
                        "--payloads-file",
                        str(payload_path),
                        "--workers",
                        str(workers),
                        "--out-json",
                        str(http_out),
                    ],
                    env=env,
                    check=True,
                )
                report = json.loads(http_out.read_text(encoding="utf-8"))
                client_ms = [float(x) for x in report["client_ms"]]
                sorted_ms = sorted(client_ms)
                mean_ms = sum(client_ms) / float(max(len(client_ms), 1))
                wall_s = max(float(report["wall_s"]), 1e-9)
                recall_at_10 = compute_recall_at_10(report["doc_ids"], gt_top10)

                base = meta_cache[(refine_multiplier, coalesce_enabled, nprobe, refine_m)]
                row = {k: base[k] for k in METRIC_SCHEMA_KEYS}
                row.update(
                    {
                        "coalesce_enabled": coalesce_enabled,
                        "queries_evaluated": len(client_ms),
                        "recall_at_10": recall_at_10,
                        "p50_ms": percentile(sorted_ms, 50),
                        "p95_ms": percentile(sorted_ms, 95),
                        "p99_ms": percentile(sorted_ms, 99),
                        "mean_ms": mean_ms,
                        "qps": float(len(client_ms)) / wall_s,
                        "stage2_3_mean_ms": max(0.0, mean_ms - float(base["stage1_mean_ms"])),
                        "vm_hwm_mb": read_proc_vm_hwm_mb(proc.pid),
                    }
                )
                results.append(row)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    return results


def annotate_rows(
    points: Sequence[Dict[str, object]],
    dataset: str,
    measured_rtt_ms: float,
    netem_delay: str,
    workers: int,
    refine_policy: str,
    refine_multiplier: int,
) -> List[Dict[str, object]]:
    annotated: List[Dict[str, object]] = []
    for p in points:
        row = {k: p[k] for k in METRIC_SCHEMA_KEYS}
        row["dataset"] = dataset
        row["minio_netem_rtt_ms"] = round(measured_rtt_ms, 2)
        row["netem_delay"] = netem_delay
        row["workers"] = workers
        row["refine_policy"] = refine_policy
        row["requested_refine_multiplier"] = refine_multiplier
        annotated.append(row)
    return annotated


def policy_label(refine_multiplier: int) -> str:
    if refine_multiplier == 0:
        return "adaptive"
    if refine_multiplier == 1:
        return "baseline"
    return f"forced-{refine_multiplier}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cohere768d")
    parser.add_argument("--osprey-bin", required=True)
    parser.add_argument("--runner-bin", required=True)
    parser.add_argument("--minio-endpoint", default="http://127.0.0.1:9000")
    parser.add_argument("--bucket", default="osprey-bench")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--nprobe-list", default="4,8,16,24,32,48")
    parser.add_argument("--refine-list", default="50,100,150,200,300")
    parser.add_argument("--off-nprobe-list", default="16,24,32")
    parser.add_argument("--off-refine-list", default="100,150,200")
    parser.add_argument(
        "--refine-multiplier",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Adaptive oversampling policy: 0 = adaptive (2x + 128 f16 floor for dim<=256), "
            "1 = disabled/baseline, N>1 = forced Nx. Pass several values to sweep both policies."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Primary worker concurrency (1 = sequential search-eval; >1 = serve + http-load).",
    )
    parser.add_argument(
        "--concurrent-workers",
        type=int,
        nargs="*",
        default=[],
        help="Optional additional concurrency levels to sweep alongside --workers (e.g. --concurrent-workers 8).",
    )
    parser.add_argument(
        "--netem-delay",
        default=os.environ.get("NETEM_DELAY", "15ms"),
        help="tc netem delay label recorded as CSV metadata (defaults to NETEM_DELAY env or '15ms').",
    )
    parser.add_argument(
        "--out-csv",
        default=str(RESULTS_DIR / "pareto_s3.csv"),
        help="Output CSV (use results/pareto_s3_adaptive.csv to preserve the published file).",
    )
    parser.add_argument("--max-queries", type=int, default=200)
    args = parser.parse_args()

    worker_counts: List[int] = []
    for w in [args.workers, *args.concurrent_workers]:
        if w < 1:
            raise ValueError(f"Worker count must be >= 1, got {w}")
        if w not in worker_counts:
            worker_counts.append(w)

    ds_dir = PREP_DIR / args.dataset
    art_dir = ARTIFACTS_DIR / "pareto" / args.dataset
    art_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    osi_path = art_dir / "segment.osi"
    osv_path = art_dir / "segment.osv"
    if not (osi_path.exists() and osv_path.exists()):
        subprocess.run(
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
            ],
            check=True,
        )

    object_key = f"{args.dataset}/segment.osv"
    s3_uri = f"s3://{args.bucket}/{object_key}"
    measured_rtt_ms = ensure_minio_bucket_and_upload(
        args.minio_endpoint, args.bucket, osv_path, object_key
    )

    env = os.environ.copy()
    env.update(
        {
            "AWS_ACCESS_KEY_ID": "osprey",
            "AWS_SECRET_ACCESS_KEY": "ospreypassword123",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ENDPOINT_URL": args.minio_endpoint,
            "AWS_ALLOW_HTTP": "true",
        }
    )

    rows: List[Dict[str, object]] = []
    meta_cache: Dict[Tuple[int, bool, int, int], Dict[str, object]] = {}

    for workers in worker_counts:
        for refine_multiplier in args.refine_multiplier:
            refine_policy = policy_label(refine_multiplier)
            for no_coalesce, np_list, rf_list in (
                (False, args.nprobe_list, args.refine_list),
                (True, args.off_nprobe_list, args.off_refine_list),
            ):
                mode_label = "OFF" if no_coalesce else "ON"
                print(
                    f"=== DP Coalescing {mode_label} sweep (workers={workers}, "
                    f"refine_multiplier={refine_multiplier}, policy={refine_policy}, "
                    f"max_queries={args.max_queries}) ==="
                )
                if workers == 1:
                    out_json = (
                        art_dir
                        / f"pareto_coalesce_{mode_label.lower()}_w1_rm{refine_multiplier}.json"
                    )
                    points = run_search_eval(
                        args.runner_bin,
                        osi_path,
                        s3_uri,
                        ds_dir,
                        np_list,
                        rf_list,
                        refine_multiplier,
                        args.max_queries,
                        no_coalesce,
                        out_json,
                        env,
                    )
                    for pt in points:
                        meta_cache[
                            (refine_multiplier, not no_coalesce, int(pt["nprobe"]), int(pt["refine_m"]))
                        ] = pt
                else:
                    points = run_concurrent_eval(
                        args.runner_bin,
                        osi_path,
                        osv_path,
                        s3_uri,
                        ds_dir,
                        art_dir,
                        np_list,
                        rf_list,
                        refine_multiplier,
                        args.max_queries,
                        no_coalesce,
                        workers,
                        env,
                        meta_cache,
                    )

                rows.extend(
                    annotate_rows(
                        points,
                        args.dataset,
                        measured_rtt_ms,
                        args.netem_delay,
                        workers,
                        refine_policy,
                        refine_multiplier,
                    )
                )

    if not rows:
        raise RuntimeError("No Pareto evaluation rows produced")

    # Enforce identical schema columns and identical queries_evaluated across coalesce ON vs OFF.
    fieldnames = list(rows[0].keys())
    expected_q_counts = {int(r["queries_evaluated"]) for r in rows}
    if len(expected_q_counts) != 1:
        raise RuntimeError(
            f"Inconsistent queries_evaluated across rows: {sorted(expected_q_counts)}"
        )
    for idx, r in enumerate(rows):
        if list(r.keys()) != fieldnames:
            raise RuntimeError(f"Schema drift on row {idx}: {list(r.keys())} != {fieldnames}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_csv} ({len(rows)} Pareto points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
