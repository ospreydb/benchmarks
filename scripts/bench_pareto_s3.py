#!/usr/bin/env python3
"""
Tasks 172 & 178: MinIO + tc netem S3 Latency Simulation & WaRP Pareto Sweep.

Uploads `.osv` slabs to local MinIO (with `tc netem delay 15ms 3ms` active),
verifies TTFB latency, and sweeps `(nprobe, refine_m)` with WaRP coalescing ON vs OFF
to produce `Recall@10 vs Latency (ms) vs S3 Range-GETs` Pareto frontiers.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import boto3
from botocore.client import Config

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"


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

    # Probe TTFB on 64-byte Range-GET to confirm tc netem delay (~15ms)
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
    s3_uri: str,
    ds_dir: Path,
    nprobe_list: str,
    refine_list: str,
    refine_multiplier: int,
    max_queries: int,
    no_warp: bool,
    out_json: Path,
    env: Dict[str, str],
) -> None:
    cmd = [
        runner_bin,
        "search-eval",
        "--osi",
        str(osi_path),
        "--slabs",
        s3_uri,
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
    if no_warp:
        cmd.append("--no-warp")
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sift1m")
    parser.add_argument("--osprey-bin", required=True)
    parser.add_argument("--runner-bin", required=True)
    parser.add_argument("--minio-endpoint", default="http://127.0.0.1:9000")
    parser.add_argument("--bucket", default="osprey-bench")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--nprobe-list", default="4,8,16,24,32,48")
    parser.add_argument("--refine-list", default="50,100,150,200,300")
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
        "--out-csv",
        default=str(RESULTS_DIR / "pareto_s3.csv"),
        help="Output CSV (use results/pareto_s3_adaptive.csv to preserve the published file).",
    )
    parser.add_argument("--max-queries", type=int, default=50)
    args = parser.parse_args()

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
    for refine_multiplier in args.refine_multiplier:
        if refine_multiplier == 0:
            refine_policy = "adaptive"
        elif refine_multiplier == 1:
            refine_policy = "baseline"
        else:
            refine_policy = f"forced-{refine_multiplier}"

        warp_on_json = art_dir / f"pareto_warp_on_rm{refine_multiplier}.json"
        print(
            f"=== WaRP ON sweep (refine_multiplier={refine_multiplier}, "
            f"policy={refine_policy}) ==="
        )
        run_search_eval(
            args.runner_bin,
            osi_path,
            s3_uri,
            ds_dir,
            args.nprobe_list,
            args.refine_list,
            refine_multiplier,
            args.max_queries,
            False,
            warp_on_json,
            env,
        )

        # Also run WaRP OFF baseline on representative points to quantify exact S3 GET reduction
        warp_off_json = art_dir / f"pareto_warp_off_rm{refine_multiplier}.json"
        print(
            f"=== WaRP OFF sweep (refine_multiplier={refine_multiplier}, "
            f"policy={refine_policy}) ==="
        )
        run_search_eval(
            args.runner_bin,
            osi_path,
            s3_uri,
            ds_dir,
            "16,24,32",
            "100,150,200",
            refine_multiplier,
            min(25, args.max_queries),
            True,
            warp_off_json,
            env,
        )

        for p in json.loads(warp_on_json.read_text()) + json.loads(warp_off_json.read_text()):
            p["dataset"] = args.dataset
            p["minio_netem_rtt_ms"] = round(measured_rtt_ms, 2)
            p["refine_policy"] = refine_policy
            p["requested_refine_multiplier"] = refine_multiplier
            rows.append(p)

    out_csv = Path(args.out_csv)
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
    print(f"[saved] {out_csv} ({len(rows)} Pareto points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
