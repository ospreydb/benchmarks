#!/usr/bin/env python3
"""
Task 174: LUT-Rank (CPU AVX-512/AVX2) vs ColBERTv2 (PyTorch CPU / PyTorch CUDA GPU / ONNX CPU).

Measures per-query latency (P50 / P95 / P99 / mean ms), throughput (QPS),
and ranking quality (NDCG@10, MRR@10) on MS MARCO Passages, BEIR SciFact, and BEIR FiQA.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"


def dcg_at_k(rels: List[float], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r > 0.0)


def eval_ndcg_mrr(ranked_ids: List[int], qrels: Dict[str, int], k: int = 10) -> Tuple[float, float]:
    if not qrels:
        return 0.0, 0.0
    rels = [float(max(0, qrels.get(str(did), 0))) for did in ranked_ids[:k]]
    ideal = sorted((float(max(0, v)) for v in qrels.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    ndcg = (dcg_at_k(rels, k) / idcg) if idcg > 0.0 else 0.0
    mrr = 0.0
    for idx, r in enumerate(rels):
        if r > 0.0:
            mrr = 1.0 / (idx + 1)
            break
    return ndcg, mrr


def load_corpus_tokens(jsonl_path: Path) -> Dict[int, np.ndarray]:
    tokens_by_doc: Dict[int, np.ndarray] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            did = int(obj["doc_id"])
            tokens_by_doc[did] = np.asarray(obj["tokens"], dtype=np.float32)
    return tokens_by_doc


def build_candidate_ids(
    queries: List[Dict], num_docs: int, candidates_per_query: int
) -> List[List[int]]:
    all_cands: List[List[int]] = []
    for qi, q in enumerate(queries):
        cands: List[int] = []
        seen = set()
        for d_str in q.get("qrels", {}).keys():
            did = int(d_str)
            if did not in seen:
                seen.add(did)
                cands.append(did)
        cursor = (qi * 7919) % max(1, num_docs)
        while len(cands) < candidates_per_query and len(cands) < num_docs:
            if cursor not in seen:
                seen.add(cursor)
                cands.append(cursor)
            cursor = (cursor + 17) % num_docs
        all_cands.append(cands)
    return all_cands


def bench_osprey_lutrank(
    dataset: str,
    ds_dir: Path,
    osprey_bin: str,
    runner_bin: str,
    candidates_per_query: int,
    max_queries: int,
) -> Dict[str, object]:
    art_dir = ARTIFACTS_DIR / "lutrank" / dataset
    art_dir.mkdir(parents=True, exist_ok=True)
    osr_path = art_dir / "segment.osr"

    if not osr_path.exists():
        subprocess.run(
            [
                osprey_bin,
                "import",
                "--input",
                str(ds_dir / "corpus_tokens.jsonl"),
                "--output",
                str(art_dir),
                "--id-col",
                "doc_id",
                "--vec-col",
                "embedding",
                "--token-col",
                "tokens",
                "--with-lutrank",
                "--clusters",
                "64",
            ],
            check=True,
        )

    out_json = art_dir / "lutrank_eval.json"
    subprocess.run(
        [
            runner_bin,
            "lutrank-eval",
            "--osr",
            str(osr_path),
            "--queries-json",
            str(ds_dir / "queries.json"),
            "--candidates-per-query",
            str(candidates_per_query),
            "--top-k",
            "10",
            "--max-queries",
            str(max_queries),
            "--out-json",
            str(out_json),
        ],
        check=True,
    )
    res = json.loads(out_json.read_text())
    return {
        "dataset": dataset,
        "engine": f"OspreyDB LUT-Rank ({res['simd_tier']} CPU)",
        "device": "CPU (AVX-512 vpshufb)",
        "queries": res["queries_evaluated"],
        "candidates_per_query": res["candidates_per_query"],
        "p50_ms": round(res["p50_ms"], 3),
        "p95_ms": round(res["p95_ms"], 3),
        "p99_ms": round(res["p99_ms"], 3),
        "mean_ms": round(res["mean_ms"], 3),
        "qps": round(res["qps"], 1),
        "ndcg_at_10": round(res["ndcg_at_10"], 4),
        "mrr_at_10": round(res["mrr_at_10"], 4),
        "index_mb": round(res["osr_bytes"] / (1024.0 * 1024.0), 2),
        "lut_synth_us": round(res["lut_synth_mean_us"], 2),
    }


def bench_colbert_numpy_cpu(
    dataset: str,
    queries: List[Dict],
    tokens_by_doc: Dict[int, np.ndarray],
    cands_list: List[List[int]],
) -> Dict[str, object]:
    latencies_ms = []
    ndcg_sum = 0.0
    mrr_sum = 0.0
    qrel_cnt = 0

    wall_t0 = time.perf_counter()
    for q, cands in zip(queries, cands_list):
        qt = np.asarray(q["tokens"], dtype=np.float32)  # [Lq, 128]
        t0 = time.perf_counter()
        scored = []
        for did in cands:
            dt = tokens_by_doc[did]  # [Ld, 128]
            sim = qt @ dt.T
            score = float(sim.max(axis=1).sum())
            scored.append((did, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if q.get("qrels"):
            qrel_cnt += 1
            ndcg, mrr = eval_ndcg_mrr([d for d, _ in scored], q["qrels"])
            ndcg_sum += ndcg
            mrr_sum += mrr
    wall_s = time.perf_counter() - wall_t0
    arr = np.sort(np.asarray(latencies_ms))
    raw_mb = sum(v.nbytes for v in tokens_by_doc.values()) / (1024.0 * 1024.0)

    return {
        "dataset": dataset,
        "engine": "ColBERTv2 Exact FP32 (NumPy/BLAS CPU)",
        "device": "CPU (FP32 BLAS)",
        "queries": len(queries),
        "candidates_per_query": len(cands_list[0]) if cands_list else 0,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "qps": round(len(queries) / max(wall_s, 1e-9), 1),
        "ndcg_at_10": round(ndcg_sum / max(1, qrel_cnt), 4),
        "mrr_at_10": round(mrr_sum / max(1, qrel_cnt), 4),
        "index_mb": round(raw_mb, 2),
        "lut_synth_us": 0.0,
    }


def bench_colbert_pytorch(
    dataset: str,
    queries: List[Dict],
    tokens_by_doc: Dict[int, np.ndarray],
    cands_list: List[List[int]],
    device_str: str,
) -> Optional[Dict[str, object]]:
    try:
        import torch
    except ImportError:
        return None

    if device_str == "cuda" and not torch.cuda.is_available():
        return None

    device = torch.device(device_str)
    dtype = torch.float16 if device_str == "cuda" else torch.float32
    max_ld = max(v.shape[0] for v in tokens_by_doc.values())
    doc_tensor_cache: Dict[int, torch.Tensor] = {}
    for did, arr in tokens_by_doc.items():
        pad = np.zeros((max_ld, 128), dtype=np.float32)
        pad[: arr.shape[0]] = arr
        doc_tensor_cache[did] = torch.from_numpy(pad).to(device=device, dtype=dtype)

    latencies_ms = []
    ndcg_sum = 0.0
    mrr_sum = 0.0
    qrel_cnt = 0

    with torch.no_grad():
        # Warmup CUDA / CPU kernel caches before measurement
        for q, cands in zip(queries[:3], cands_list[:3]):
            qt_w = torch.from_numpy(np.asarray(q["tokens"], dtype=np.float32)).to(
                device=device, dtype=dtype
            )
            docs_w = torch.stack([doc_tensor_cache[did] for did in cands], dim=0)
            _ = torch.einsum("qd,knd->kqn", qt_w, docs_w).max(dim=-1).values.sum(dim=-1)
        if device_str == "cuda":
            torch.cuda.synchronize()

        wall_t0 = time.perf_counter()
        for q, cands in zip(queries, cands_list):
            t0 = time.perf_counter()
            qt = torch.from_numpy(np.asarray(q["tokens"], dtype=np.float32)).to(
                device=device, dtype=dtype
            )
            batch_docs = torch.stack([doc_tensor_cache[did] for did in cands], dim=0)
            sims = torch.einsum("qd,knd->kqn", qt, batch_docs)
            scores = sims.max(dim=-1).values.sum(dim=-1)
            if device_str == "cuda":
                torch.cuda.synchronize()
            score_list = scores.float().cpu().tolist()
            scored = sorted(zip(cands, score_list), key=lambda x: (-x[1], x[0]))
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if q.get("qrels"):
                qrel_cnt += 1
                ndcg, mrr = eval_ndcg_mrr([d for d, _ in scored], q["qrels"])
                ndcg_sum += ndcg
                mrr_sum += mrr
    if device_str == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_t0
    arr = np.sort(np.asarray(latencies_ms))
    raw_mb = sum(v.nbytes for v in tokens_by_doc.values()) / (1024.0 * 1024.0)
    label = "PyTorch CUDA GPU" if device_str == "cuda" else "PyTorch CPU"
    return {
        "dataset": dataset,
        "engine": f"ColBERTv2 ({label})",
        "device": device_str.upper(),
        "queries": len(queries),
        "candidates_per_query": len(cands_list[0]) if cands_list else 0,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "qps": round(len(queries) / max(wall_s, 1e-9), 1),
        "ndcg_at_10": round(ndcg_sum / max(1, qrel_cnt), 4),
        "mrr_at_10": round(mrr_sum / max(1, qrel_cnt), 4),
        "index_mb": round(raw_mb, 2),
        "lut_synth_us": 0.0,
    }


def bench_colbert_onnx(
    dataset: str,
    queries: List[Dict],
    tokens_by_doc: Dict[int, np.ndarray],
    cands_list: List[List[int]],
) -> Optional[Dict[str, object]]:
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        return None

    class ColBertMaxSimModule(torch.nn.Module):
        def forward(self, qt: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
            # qt: [Lq, D], docs: [K, Ld, D]
            sims = torch.matmul(docs, qt.transpose(0, 1))  # [K, Ld, Lq]
            return sims.max(dim=1).values.sum(dim=-1)  # [K]

    max_ld = max(v.shape[0] for v in tokens_by_doc.values())
    max_lq = max(len(q["tokens"]) for q in queries)
    k_cands = len(cands_list[0]) if cands_list else 0
    if k_cands == 0:
        return None

    onnx_path = ARTIFACTS_DIR / "lutrank" / dataset / "colbert_maxsim.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        model = ColBertMaxSimModule().eval()
        dummy_q = torch.randn(max_lq, 128, dtype=torch.float32)
        dummy_d = torch.randn(k_cands, max_ld, 128, dtype=torch.float32)
        try:
            torch.onnx.export(
                model,
                (dummy_q, dummy_d),
                str(onnx_path),
                input_names=["qt", "docs"],
                output_names=["scores"],
                dynamic_axes={"qt": {0: "lq"}, "docs": {0: "k", 1: "ld"}},
                opset_version=18,
            )
        except Exception as exc:
            print(f"[warn] ONNX export skipped ({exc})")
            return None

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    doc_pad_cache: Dict[int, np.ndarray] = {}
    for did, arr in tokens_by_doc.items():
        pad = np.zeros((max_ld, 128), dtype=np.float32)
        pad[: arr.shape[0]] = arr
        doc_pad_cache[did] = pad

    latencies_ms = []
    ndcg_sum = 0.0
    mrr_sum = 0.0
    qrel_cnt = 0

    # Warmup ORT session
    for q, cands in zip(queries[:3], cands_list[:3]):
        qt_w = np.asarray(q["tokens"], dtype=np.float32)
        docs_w = np.stack([doc_pad_cache[did] for did in cands], axis=0)
        _ = sess.run(["scores"], {"qt": qt_w, "docs": docs_w})

    wall_t0 = time.perf_counter()
    for q, cands in zip(queries, cands_list):
        t0 = time.perf_counter()
        qt = np.asarray(q["tokens"], dtype=np.float32)
        batch_docs = np.stack([doc_pad_cache[did] for did in cands], axis=0)
        scores = sess.run(["scores"], {"qt": qt, "docs": batch_docs})[0]
        scored = sorted(zip(cands, scores.tolist()), key=lambda x: (-x[1], x[0]))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if q.get("qrels"):
            qrel_cnt += 1
            ndcg, mrr = eval_ndcg_mrr([d for d, _ in scored], q["qrels"])
            ndcg_sum += ndcg
            mrr_sum += mrr
    wall_s = time.perf_counter() - wall_t0
    arr = np.sort(np.asarray(latencies_ms))
    raw_mb = sum(v.nbytes for v in tokens_by_doc.values()) / (1024.0 * 1024.0)
    return {
        "dataset": dataset,
        "engine": "ColBERTv2 (ONNX Runtime CPU)",
        "device": "CPU (ONNX)",
        "queries": len(queries),
        "candidates_per_query": k_cands,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "qps": round(len(queries) / max(wall_s, 1e-9), 1),
        "ndcg_at_10": round(ndcg_sum / max(1, qrel_cnt), 4),
        "mrr_at_10": round(mrr_sum / max(1, qrel_cnt), 4),
        "index_mb": round(raw_mb, 2),
        "lut_synth_us": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "msmarco"])
    parser.add_argument("--osprey-bin", required=True)
    parser.add_argument("--runner-bin", required=True)
    parser.add_argument("--candidates-per-query", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for ds in args.datasets:
        ds_dir = PREP_DIR / ds
        if not (ds_dir / "corpus_tokens.jsonl").exists():
            print(f"[skip] {ds}: corpus_tokens.jsonl not found")
            continue

        print(f"=== LUT-Rank vs ColBERTv2 on {ds} ===")
        rows.append(
            bench_osprey_lutrank(
                ds,
                ds_dir,
                args.osprey_bin,
                args.runner_bin,
                args.candidates_per_query,
                args.max_queries,
            )
        )

        queries = json.loads((ds_dir / "queries.json").read_text())[: args.max_queries]
        tokens_by_doc = load_corpus_tokens(ds_dir / "corpus_tokens.jsonl")
        cands_list = build_candidate_ids(queries, len(tokens_by_doc), args.candidates_per_query)

        rows.append(bench_colbert_numpy_cpu(ds, queries, tokens_by_doc, cands_list))

        for dev in ["cpu", "cuda"]:
            pt_row = bench_colbert_pytorch(ds, queries, tokens_by_doc, cands_list, dev)
            if pt_row is not None:
                rows.append(pt_row)

        onnx_row = bench_colbert_onnx(ds, queries, tokens_by_doc, cands_list)
        if onnx_row is not None:
            rows.append(onnx_row)

    out_csv = RESULTS_DIR / "lutrank_vs_colbert.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out_csv}")
        for r in rows:
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
