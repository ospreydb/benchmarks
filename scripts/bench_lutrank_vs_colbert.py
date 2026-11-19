#!/usr/bin/env python3
"""
Task 174: LUT-Rank (CPU AVX-512/AVX2) vs ColBERTv2 (PyTorch CPU / PyTorch CUDA GPU / ONNX CPU).

Measures per-query kernel and end-to-end latency (P50 / P95 / P99 / mean ms),
throughput (QPS), memory footprint, and ranking quality (NDCG@10, MRR@10)
over realistic Stage-1 retrieval candidates on MS MARCO, BEIR SciFact, and BEIR FiQA.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "datasets" / "prepared"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

CSV_FIELDNAMES = [
    "dataset",
    "engine",
    "device",
    "token_encoder",
    "candidate_source",
    "doc_token_cap",
    "query_token_cap",
    "num_threads",
    "dtype",
    "queries",
    "candidates_per_query",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "qps",
    "kernel_p50_ms",
    "kernel_p95_ms",
    "kernel_p99_ms",
    "kernel_mean_ms",
    "kernel_qps",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "e2e_mean_ms",
    "e2e_qps",
    "ndcg_at_10",
    "mrr_at_10",
    "index_mb",
    "lut_synth_us",
]


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


def load_dataset_meta(ds_dir: Path, tokens_by_doc: Dict[int, np.ndarray], queries: List[Dict]) -> Dict[str, object]:
    meta_path = ds_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    observed_doc_cap = max((v.shape[0] for v in tokens_by_doc.values()), default=0)
    observed_q_cap = max((len(q.get("tokens", [])) for q in queries), default=0)
    return {
        "token_encoder": str(meta.get("token_encoder", "unknown")),
        "model_name": str(meta.get("model_name", "unknown")),
        "doc_token_cap": int(meta.get("doc_token_cap", observed_doc_cap)),
        "query_token_cap": int(meta.get("query_token_cap", observed_q_cap)),
    }


def load_corpus_data(
    jsonl_path: Path,
) -> Tuple[List[int], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, str]]:
    doc_ids: List[int] = []
    tokens_by_doc: Dict[int, np.ndarray] = {}
    embeddings_by_doc: Dict[int, np.ndarray] = {}
    texts_by_doc: Dict[int, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            did = int(obj["doc_id"])
            doc_ids.append(did)
            tokens_by_doc[did] = np.asarray(obj["tokens"], dtype=np.float32)
            if "embedding" in obj and obj["embedding"] is not None:
                embeddings_by_doc[did] = np.asarray(obj["embedding"], dtype=np.float32)
            if "text" in obj and obj["text"] is not None:
                texts_by_doc[did] = str(obj["text"])
    return doc_ids, tokens_by_doc, embeddings_by_doc, texts_by_doc


def _load_parquet_embeddings_and_texts(
    parquet_path: Path,
    embeddings_by_doc: Dict[int, np.ndarray],
    texts_by_doc: Dict[int, str],
) -> None:
    if not parquet_path.exists():
        return
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return
    tbl = pq.read_table(parquet_path)
    dids = tbl["doc_id"].to_pylist() if "doc_id" in tbl.column_names else list(range(tbl.num_rows))
    if not embeddings_by_doc and "embedding" in tbl.column_names:
        embs = tbl["embedding"].to_pylist()
        for did, emb in zip(dids, embs):
            embeddings_by_doc[int(did)] = np.asarray(emb, dtype=np.float32)
    if not texts_by_doc and "text" in tbl.column_names:
        texts = tbl["text"].to_pylist()
        for did, txt in zip(dids, texts):
            texts_by_doc[int(did)] = str(txt or "")


def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_topk_builtin(
    doc_ids: Sequence[int],
    doc_tokens: List[List[str]],
    query_tokens: List[List[str]],
    k: int,
) -> List[List[int]]:
    """Pure-Python/NumPy Okapi BM25 fallback when rank_bm25 is unavailable."""
    n_docs = len(doc_ids)
    if n_docs == 0:
        return [[] for _ in query_tokens]
    df: Dict[str, int] = {}
    postings: Dict[str, List[Tuple[int, int]]] = {}
    doc_lens = np.zeros(n_docs, dtype=np.float32)
    for row_idx, toks in enumerate(doc_tokens):
        doc_lens[row_idx] = float(len(toks))
        tf_map = Counter(toks)
        for term, tf in tf_map.items():
            df[term] = df.get(term, 0) + 1
            postings.setdefault(term, []).append((row_idx, tf))

    avgdl = max(float(doc_lens.mean()), 1.0)
    k1 = 1.5
    b = 0.75
    idf = {
        term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for term, freq in df.items()
    }

    results: List[List[int]] = []
    k_eff = min(k, n_docs)
    for q_toks in query_tokens:
        scores = np.zeros(n_docs, dtype=np.float32)
        for term in q_toks:
            plist = postings.get(term)
            if not plist:
                continue
            w = idf[term]
            for row_idx, tf in plist:
                dl = doc_lens[row_idx]
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[row_idx] += w * ((tf * (k1 + 1.0)) / max(denom, 1e-9))
        if n_docs > k_eff:
            top_idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
        else:
            top_idx = np.argsort(-scores, kind="stable")[:k_eff]
        results.append([doc_ids[int(i)] for i in top_idx])
    return results


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (mat / norms).astype(np.float32)


def build_candidate_ids(
    queries: List[Dict],
    doc_ids: Sequence[int],
    embeddings_by_doc: Dict[int, np.ndarray],
    texts_by_doc: Dict[int, str],
    candidates_per_query: int,
    candidate_source: str,
) -> Tuple[List[List[int]], str]:
    num_docs = len(doc_ids)
    k_eff = min(candidates_per_query, num_docs)
    if num_docs == 0 or not queries:
        return [[] for _ in queries], candidate_source

    has_dense = (
        len(embeddings_by_doc) == num_docs
        and all("embedding" in q and q["embedding"] for q in queries)
    )
    has_text = len(texts_by_doc) == num_docs and any(texts_by_doc.values())

    resolved_source = candidate_source
    if resolved_source == "stage1-dense" and not has_dense:
        if has_text:
            print("[warn] Dense embeddings missing; falling back from stage1-dense to bm25")
            resolved_source = "bm25"
        else:
            print("[warn] Dense embeddings and texts missing; falling back to qrels-oracle")
            resolved_source = "qrels-oracle"
    elif resolved_source == "bm25" and not has_text:
        if has_dense:
            print("[warn] Document texts missing; falling back from bm25 to stage1-dense")
            resolved_source = "stage1-dense"
        else:
            print("[warn] Document texts and embeddings missing; falling back to qrels-oracle")
            resolved_source = "qrels-oracle"

    if resolved_source == "stage1-dense":
        doc_mat = _l2_normalize_rows(
            np.stack([embeddings_by_doc[did] for did in doc_ids], axis=0).astype(np.float32)
        )
        q_mat = _l2_normalize_rows(
            np.stack([np.asarray(q["embedding"], dtype=np.float32) for q in queries], axis=0)
        )
        all_cands: List[List[int]] = []
        chunk = 128
        for s in range(0, len(queries), chunk):
            sims = q_mat[s : s + chunk] @ doc_mat.T
            if num_docs > k_eff:
                idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
                row_sims = np.take_along_axis(sims, idx, axis=1)
                order = np.argsort(-row_sims, axis=1, kind="stable")
                top_indices = np.take_along_axis(idx, order, axis=1)
            else:
                top_indices = np.argsort(-sims, axis=1, kind="stable")[:, :k_eff]
            for row in top_indices:
                all_cands.append([doc_ids[int(i)] for i in row])
        return all_cands, "stage1-dense"

    if resolved_source == "bm25":
        tokenized_corpus = [_tokenize_bm25(texts_by_doc.get(did, "")) for did in doc_ids]
        tokenized_queries = [_tokenize_bm25(str(q.get("text", ""))) for q in queries]
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokenized_corpus)
            all_cands = []
            for q_toks in tokenized_queries:
                scores = np.asarray(bm25.get_scores(q_toks), dtype=np.float32)
                if num_docs > k_eff:
                    top_idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
                    top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
                else:
                    top_idx = np.argsort(-scores, kind="stable")[:k_eff]
                all_cands.append([doc_ids[int(i)] for i in top_idx])
            return all_cands, "bm25"
        except ImportError:
            return _bm25_topk_builtin(doc_ids, tokenized_corpus, tokenized_queries, k_eff), "bm25"

    # Legacy qrels-oracle mode (explicit opt-in only)
    doc_id_set = set(doc_ids)
    all_cands = []
    for qi, q in enumerate(queries):
        cands: List[int] = []
        seen = set()
        for d_str in q.get("qrels", {}).keys():
            did = int(d_str)
            if did in doc_id_set and did not in seen:
                seen.add(did)
                cands.append(did)
        cursor = (qi * 7919) % max(1, num_docs)
        while len(cands) < k_eff:
            did = doc_ids[cursor]
            if did not in seen:
                seen.add(did)
                cands.append(did)
            cursor = (cursor + 17) % num_docs
        all_cands.append(cands[:k_eff])
    return all_cands, "qrels-oracle"


def persist_candidates(
    dataset: str,
    queries: List[Dict],
    cands_list: List[List[int]],
    candidate_source: str,
    candidates_per_query: int,
) -> Path:
    art_dir = ARTIFACTS_DIR / "lutrank" / dataset
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "candidate_source": candidate_source,
        "candidates_per_query": candidates_per_query,
        "num_queries": len(queries),
        "queries": [
            {
                "query_id": str(q.get("query_id", qi)),
                "candidate_ids": cands,
            }
            for qi, (q, cands) in enumerate(zip(queries, cands_list))
        ],
    }
    raw = json.dumps(payload)
    cand_file = art_dir / f"candidates_{candidate_source}_k{candidates_per_query}.json"
    cand_file.write_text(raw, encoding="utf-8")
    (art_dir / "candidates.json").write_text(raw, encoding="utf-8")
    return cand_file


def _runner_supports_candidates_json(runner_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [runner_bin, "lutrank-eval", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "--candidates-json" in (proc.stdout + proc.stderr)
    except OSError:
        return False


def _build_metric_row(
    *,
    dataset: str,
    engine: str,
    device: str,
    ds_meta: Dict[str, object],
    candidate_source: str,
    num_threads: int,
    dtype_label: str,
    queries: List[Dict],
    candidates_per_query: int,
    kernel_latencies_ms: List[float],
    e2e_latencies_ms: List[float],
    e2e_wall_s: float,
    scored_all: List[Tuple[Dict, List[Tuple[int, float]]]],
    index_mb: float,
    lut_synth_us: float = 0.0,
) -> Dict[str, object]:
    ndcg_sum = 0.0
    mrr_sum = 0.0
    qrel_cnt = 0
    for q, scored in scored_all:
        if q.get("qrels"):
            qrel_cnt += 1
            ndcg, mrr = eval_ndcg_mrr([d for d, _ in scored], q["qrels"])
            ndcg_sum += ndcg
            mrr_sum += mrr

    k_arr = np.sort(np.asarray(kernel_latencies_ms, dtype=np.float64))
    e_arr = np.sort(np.asarray(e2e_latencies_ms, dtype=np.float64))
    k_wall_s = max(float(k_arr.sum()) / 1000.0, 1e-9)

    k_p50 = round(float(np.percentile(k_arr, 50)), 3) if len(k_arr) else 0.0
    k_p95 = round(float(np.percentile(k_arr, 95)), 3) if len(k_arr) else 0.0
    k_p99 = round(float(np.percentile(k_arr, 99)), 3) if len(k_arr) else 0.0
    k_mean = round(float(k_arr.mean()), 3) if len(k_arr) else 0.0
    k_qps = round(len(queries) / k_wall_s, 1)

    e_p50 = round(float(np.percentile(e_arr, 50)), 3) if len(e_arr) else 0.0
    e_p95 = round(float(np.percentile(e_arr, 95)), 3) if len(e_arr) else 0.0
    e_p99 = round(float(np.percentile(e_arr, 99)), 3) if len(e_arr) else 0.0
    e_mean = round(float(e_arr.mean()), 3) if len(e_arr) else 0.0
    e_qps = round(len(queries) / max(e2e_wall_s, 1e-9), 1)

    return {
        "dataset": dataset,
        "engine": engine,
        "device": device,
        "token_encoder": ds_meta["token_encoder"],
        "candidate_source": candidate_source,
        "doc_token_cap": ds_meta["doc_token_cap"],
        "query_token_cap": ds_meta["query_token_cap"],
        "num_threads": num_threads,
        "dtype": dtype_label,
        "queries": len(queries),
        "candidates_per_query": candidates_per_query,
        "p50_ms": k_p50,
        "p95_ms": k_p95,
        "p99_ms": k_p99,
        "mean_ms": k_mean,
        "qps": k_qps,
        "kernel_p50_ms": k_p50,
        "kernel_p95_ms": k_p95,
        "kernel_p99_ms": k_p99,
        "kernel_mean_ms": k_mean,
        "kernel_qps": k_qps,
        "e2e_p50_ms": e_p50,
        "e2e_p95_ms": e_p95,
        "e2e_p99_ms": e_p99,
        "e2e_mean_ms": e_mean,
        "e2e_qps": e_qps,
        "ndcg_at_10": round(ndcg_sum / max(1, qrel_cnt), 4),
        "mrr_at_10": round(mrr_sum / max(1, qrel_cnt), 4),
        "index_mb": round(index_mb, 2),
        "lut_synth_us": round(lut_synth_us, 2),
    }


def bench_osprey_lutrank(
    dataset: str,
    ds_dir: Path,
    ds_meta: Dict[str, object],
    osprey_bin: str,
    runner_bin: str,
    candidates_per_query: int,
    max_queries: int,
    cand_file: Path,
    candidate_source: str,
) -> Dict[str, object]:
    art_dir = ARTIFACTS_DIR / "lutrank" / dataset
    art_dir.mkdir(parents=True, exist_ok=True)
    osr_path = art_dir / "segment.osr"
    corpus_jsonl = ds_dir / "corpus_tokens.jsonl"

    needs_import = (
        not osr_path.exists()
        or corpus_jsonl.stat().st_mtime > osr_path.stat().st_mtime
    )
    if needs_import:
        subprocess.run(
            [
                osprey_bin,
                "import",
                "--input",
                str(corpus_jsonl),
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
    cmd = [
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
    ]
    if _runner_supports_candidates_json(runner_bin):
        cmd.extend(["--candidates-json", str(cand_file)])
        effective_cand_source = candidate_source
    else:
        print(
            f"[warn] {runner_bin} lutrank-eval does not support --candidates-json; "
            "using its internal qrels-oracle candidate builder for OspreyDB row."
        )
        effective_cand_source = "qrels-oracle (lutrank-eval-legacy)"

    subprocess.run(cmd, check=True)
    res = json.loads(out_json.read_text(encoding="utf-8"))
    k_p50 = round(res["p50_ms"], 3)
    k_p95 = round(res["p95_ms"], 3)
    k_p99 = round(res["p99_ms"], 3)
    k_mean = round(res["mean_ms"], 3)
    k_qps = round(res["qps"], 1)

    return {
        "dataset": dataset,
        "engine": f"OspreyDB LUT-Rank ({res['simd_tier']} CPU)",
        "device": "CPU (AVX-512 vpshufb, 1T)",
        "token_encoder": ds_meta["token_encoder"],
        "candidate_source": effective_cand_source,
        "doc_token_cap": ds_meta["doc_token_cap"],
        "query_token_cap": ds_meta["query_token_cap"],
        "num_threads": 1,
        "dtype": "INT8-LUT",
        "queries": res["queries_evaluated"],
        "candidates_per_query": res["candidates_per_query"],
        "p50_ms": k_p50,
        "p95_ms": k_p95,
        "p99_ms": k_p99,
        "mean_ms": k_mean,
        "qps": k_qps,
        "kernel_p50_ms": k_p50,
        "kernel_p95_ms": k_p95,
        "kernel_p99_ms": k_p99,
        "kernel_mean_ms": k_mean,
        "kernel_qps": k_qps,
        "e2e_p50_ms": k_p50,
        "e2e_p95_ms": k_p95,
        "e2e_p99_ms": k_p99,
        "e2e_mean_ms": k_mean,
        "e2e_qps": k_qps,
        "ndcg_at_10": round(res["ndcg_at_10"], 4),
        "mrr_at_10": round(res["mrr_at_10"], 4),
        "index_mb": round(res["osr_bytes"] / (1024.0 * 1024.0), 2),
        "lut_synth_us": round(res["lut_synth_mean_us"], 2),
    }


def bench_colbert_numpy_cpu(
    dataset: str,
    ds_meta: Dict[str, object],
    candidate_source: str,
    queries: List[Dict],
    tokens_by_doc: Dict[int, np.ndarray],
    cands_list: List[List[int]],
) -> Dict[str, object]:
    kernel_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []

    # Warmup BLAS/CPU caches
    for q, cands in zip(queries[:3], cands_list[:3]):
        qt_w = np.asarray(q["tokens"], dtype=np.float32)
        for did in cands:
            _ = qt_w @ tokens_by_doc[did].T

    scored_all: List[Tuple[Dict, List[Tuple[int, float]]]] = []
    wall_t0 = time.perf_counter()
    for q, cands in zip(queries, cands_list):
        t_e2e_0 = time.perf_counter()
        qt = np.asarray(q["tokens"], dtype=np.float32)  # [Lq, 128]
        cand_dts = [tokens_by_doc[did] for did in cands]

        t_kern_0 = time.perf_counter()
        scored = []
        for did, dt in zip(cands, cand_dts):
            sim = qt @ dt.T
            score = float(sim.max(axis=1).sum())
            scored.append((did, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        now = time.perf_counter()

        kernel_latencies_ms.append((now - t_kern_0) * 1000.0)
        e2e_latencies_ms.append((now - t_e2e_0) * 1000.0)
        scored_all.append((q, scored))
    e2e_wall_s = time.perf_counter() - wall_t0

    raw_mb = sum(v.size * 4 for v in tokens_by_doc.values()) / (1024.0 * 1024.0)
    num_threads = int(
        os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("OPENBLAS_NUM_THREADS")
        or os.cpu_count()
        or 1
    )
    return _build_metric_row(
        dataset=dataset,
        engine="ColBERTv2 Exact FP32 (NumPy/BLAS CPU)",
        device="CPU (FP32 BLAS)",
        ds_meta=ds_meta,
        candidate_source=candidate_source,
        num_threads=num_threads,
        dtype_label="FP32",
        queries=queries,
        candidates_per_query=len(cands_list[0]) if cands_list else 0,
        kernel_latencies_ms=kernel_latencies_ms,
        e2e_latencies_ms=e2e_latencies_ms,
        e2e_wall_s=e2e_wall_s,
        scored_all=scored_all,
        index_mb=raw_mb,
    )


def bench_colbert_pytorch(
    dataset: str,
    ds_meta: Dict[str, object],
    candidate_source: str,
    queries: List[Dict],
    tokens_by_doc: Dict[int, np.ndarray],
    cands_list: List[List[int]],
    device_str: str,
    num_threads: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    try:
        import torch
    except ImportError:
        return None

    if device_str == "cuda" and not torch.cuda.is_available():
        return None

    prev_threads = torch.get_num_threads()
    if device_str == "cpu" and num_threads is not None:
        torch.set_num_threads(num_threads)

    try:
        device = torch.device(device_str)
        dtype = torch.float16 if device_str == "cuda" else torch.float32
        dtype_label = "FP16" if dtype == torch.float16 else "FP32"

        ordered_dids = list(tokens_by_doc.keys())
        did_to_row = {did: idx for idx, did in enumerate(ordered_dids)}
        max_ld = max(v.shape[0] for v in tokens_by_doc.values())
        token_dim = next(iter(tokens_by_doc.values())).shape[1]

        doc_bank_np = np.zeros((len(ordered_dids), max_ld, token_dim), dtype=np.float32)
        doc_mask_np = np.full((len(ordered_dids), max_ld), -1e4, dtype=np.float32)
        for row_idx, did in enumerate(ordered_dids):
            arr = tokens_by_doc[did]
            ld = arr.shape[0]
            doc_bank_np[row_idx, :ld] = arr
            doc_mask_np[row_idx, :ld] = 0.0

        doc_bank = torch.from_numpy(doc_bank_np).to(device=device, dtype=dtype)
        doc_mask = torch.from_numpy(doc_mask_np).to(device=device, dtype=dtype)

        kernel_latencies_ms: List[float] = []
        e2e_latencies_ms: List[float] = []

        with torch.no_grad():
            # Warmup CUDA / CPU kernel caches before measurement
            for q, cands in zip(queries[:3], cands_list[:3]):
                qt_w = torch.from_numpy(np.asarray(q["tokens"], dtype=np.float32)).to(
                    device=device, dtype=dtype
                )
                idx_w = torch.tensor([did_to_row[d] for d in cands], dtype=torch.long, device=device)
                docs_w = doc_bank.index_select(0, idx_w)
                mask_w = doc_mask.index_select(0, idx_w).unsqueeze(1)
                _ = (torch.einsum("qd,knd->kqn", qt_w, docs_w) + mask_w).max(dim=-1).values.sum(dim=-1)
            if device_str == "cuda":
                torch.cuda.synchronize()

            scored_all: List[Tuple[Dict, List[Tuple[int, float]]]] = []
            wall_t0 = time.perf_counter()
            for q, cands in zip(queries, cands_list):
                t_e2e_0 = time.perf_counter()
                qt = torch.from_numpy(np.asarray(q["tokens"], dtype=np.float32)).to(
                    device=device, dtype=dtype
                )
                cands_idx = torch.tensor([did_to_row[d] for d in cands], dtype=torch.long, device=device)
                batch_docs = doc_bank.index_select(0, cands_idx)
                batch_mask = doc_mask.index_select(0, cands_idx).unsqueeze(1)
                if device_str == "cuda":
                    torch.cuda.synchronize()

                t_kern_0 = time.perf_counter()
                sims = torch.einsum("qd,knd->kqn", qt, batch_docs) + batch_mask
                scores = sims.max(dim=-1).values.sum(dim=-1)
                if device_str == "cuda":
                    torch.cuda.synchronize()
                score_list = scores.float().cpu().tolist()
                scored = sorted(zip(cands, score_list), key=lambda x: (-x[1], x[0]))
                now = time.perf_counter()

                kernel_latencies_ms.append((now - t_kern_0) * 1000.0)
                e2e_latencies_ms.append((now - t_e2e_0) * 1000.0)
                scored_all.append((q, scored))
        if device_str == "cuda":
            torch.cuda.synchronize()
        e2e_wall_s = time.perf_counter() - wall_t0

        bytes_per_elem = 2 if dtype == torch.float16 else 4
        raw_mb = sum(v.size * bytes_per_elem for v in tokens_by_doc.values()) / (1024.0 * 1024.0)
        if device_str == "cuda":
            label = "PyTorch CUDA GPU"
            dev_label = "CUDA (FP16)"
            reported_threads = 1
        elif num_threads == 1:
            label = "PyTorch CPU 1T"
            dev_label = "CPU (1T)"
            reported_threads = 1
        else:
            reported_threads = torch.get_num_threads()
            label = "PyTorch CPU"
            dev_label = f"CPU ({reported_threads}T)"

        return _build_metric_row(
            dataset=dataset,
            engine=f"ColBERTv2 ({label})",
            device=dev_label,
            ds_meta=ds_meta,
            candidate_source=candidate_source,
            num_threads=reported_threads,
            dtype_label=dtype_label,
            queries=queries,
            candidates_per_query=len(cands_list[0]) if cands_list else 0,
            kernel_latencies_ms=kernel_latencies_ms,
            e2e_latencies_ms=e2e_latencies_ms,
            e2e_wall_s=e2e_wall_s,
            scored_all=scored_all,
            index_mb=raw_mb,
        )
    finally:
        torch.set_num_threads(prev_threads)


def bench_colbert_onnx(
    dataset: str,
    ds_dir: Path,
    ds_meta: Dict[str, object],
    candidate_source: str,
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
        def forward(self, qt: torch.Tensor, docs: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
            # qt: [Lq, D], docs: [K, Ld, D], doc_mask: [K, Ld, 1]
            sims = torch.matmul(docs, qt.transpose(0, 1)) + doc_mask  # [K, Ld, Lq]
            return sims.max(dim=1).values.sum(dim=-1)  # [K]

    max_ld = max(v.shape[0] for v in tokens_by_doc.values())
    max_lq = max(len(q["tokens"]) for q in queries)
    token_dim = next(iter(tokens_by_doc.values())).shape[1]
    k_cands = len(cands_list[0]) if cands_list else 0
    if k_cands == 0:
        return None

    onnx_path = ARTIFACTS_DIR / "lutrank" / dataset / "colbert_maxsim_masked.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_jsonl = ds_dir / "corpus_tokens.jsonl"
    needs_export = (
        not onnx_path.exists()
        or (corpus_jsonl.exists() and corpus_jsonl.stat().st_mtime > onnx_path.stat().st_mtime)
    )
    if needs_export:
        model = ColBertMaxSimModule().eval()
        dummy_q = torch.randn(max_lq, token_dim, dtype=torch.float32)
        dummy_d = torch.randn(k_cands, max_ld, token_dim, dtype=torch.float32)
        dummy_m = torch.zeros(k_cands, max_ld, 1, dtype=torch.float32)
        try:
            torch.onnx.export(
                model,
                (dummy_q, dummy_d, dummy_m),
                str(onnx_path),
                input_names=["qt", "docs", "doc_mask"],
                output_names=["scores"],
                dynamic_axes={
                    "qt": {0: "lq"},
                    "docs": {0: "k", 1: "ld"},
                    "doc_mask": {0: "k", 1: "ld"},
                },
                opset_version=18,
            )
        except Exception as exc:
            print(f"[warn] ONNX export skipped ({exc})")
            return None

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    ordered_dids = list(tokens_by_doc.keys())
    did_to_row = {did: idx for idx, did in enumerate(ordered_dids)}
    doc_bank = np.zeros((len(ordered_dids), max_ld, token_dim), dtype=np.float32)
    mask_bank = np.full((len(ordered_dids), max_ld, 1), -1e4, dtype=np.float32)
    for row_idx, did in enumerate(ordered_dids):
        arr = tokens_by_doc[did]
        ld = arr.shape[0]
        doc_bank[row_idx, :ld] = arr
        mask_bank[row_idx, :ld, 0] = 0.0

    kernel_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []

    # Warmup ORT session
    for q, cands in zip(queries[:3], cands_list[:3]):
        qt_w = np.asarray(q["tokens"], dtype=np.float32)
        rows_w = [did_to_row[d] for d in cands]
        docs_w = doc_bank[rows_w]
        mask_w = mask_bank[rows_w]
        _ = sess.run(["scores"], {"qt": qt_w, "docs": docs_w, "doc_mask": mask_w})

    scored_all: List[Tuple[Dict, List[Tuple[int, float]]]] = []
    wall_t0 = time.perf_counter()
    for q, cands in zip(queries, cands_list):
        t_e2e_0 = time.perf_counter()
        qt = np.asarray(q["tokens"], dtype=np.float32)
        rows_idx = [did_to_row[d] for d in cands]
        batch_docs = doc_bank[rows_idx]
        batch_mask = mask_bank[rows_idx]

        t_kern_0 = time.perf_counter()
        scores = sess.run(["scores"], {"qt": qt, "docs": batch_docs, "doc_mask": batch_mask})[0]
        scored = sorted(zip(cands, scores.tolist()), key=lambda x: (-x[1], x[0]))
        now = time.perf_counter()

        kernel_latencies_ms.append((now - t_kern_0) * 1000.0)
        e2e_latencies_ms.append((now - t_e2e_0) * 1000.0)
        scored_all.append((q, scored))
    e2e_wall_s = time.perf_counter() - wall_t0

    raw_mb = sum(v.size * 4 for v in tokens_by_doc.values()) / (1024.0 * 1024.0)
    num_threads = sess_opts.intra_op_num_threads or (os.cpu_count() or 1)
    return _build_metric_row(
        dataset=dataset,
        engine="ColBERTv2 (ONNX Runtime CPU)",
        device="CPU (ONNX)",
        ds_meta=ds_meta,
        candidate_source=candidate_source,
        num_threads=num_threads,
        dtype_label="FP32",
        queries=queries,
        candidates_per_query=k_cands,
        kernel_latencies_ms=kernel_latencies_ms,
        e2e_latencies_ms=e2e_latencies_ms,
        e2e_wall_s=e2e_wall_s,
        scored_all=scored_all,
        index_mb=raw_mb,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "msmarco"])
    parser.add_argument("--osprey-bin", required=True)
    parser.add_argument("--runner-bin", required=True)
    parser.add_argument("--candidates-per-query", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument(
        "--candidate-source",
        choices=["stage1-dense", "bm25", "qrels-oracle"],
        default="stage1-dense",
        help="Stage-1 candidate generator for fair reranking (default: stage1-dense).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for ds in args.datasets:
        ds_dir = PREP_DIR / ds
        corpus_jsonl = ds_dir / "corpus_tokens.jsonl"
        if not corpus_jsonl.exists():
            print(f"[skip] {ds}: corpus_tokens.jsonl not found")
            continue

        print(f"=== LUT-Rank vs ColBERTv2 on {ds} ===")
        queries = json.loads((ds_dir / "queries.json").read_text(encoding="utf-8"))[: args.max_queries]
        doc_ids, tokens_by_doc, embeddings_by_doc, texts_by_doc = load_corpus_data(corpus_jsonl)
        _load_parquet_embeddings_and_texts(ds_dir / "base.parquet", embeddings_by_doc, texts_by_doc)
        ds_meta = load_dataset_meta(ds_dir, tokens_by_doc, queries)

        cands_list, actual_source = build_candidate_ids(
            queries=queries,
            doc_ids=doc_ids,
            embeddings_by_doc=embeddings_by_doc,
            texts_by_doc=texts_by_doc,
            candidates_per_query=args.candidates_per_query,
            candidate_source=args.candidate_source,
        )
        cand_file = persist_candidates(
            dataset=ds,
            queries=queries,
            cands_list=cands_list,
            candidate_source=actual_source,
            candidates_per_query=args.candidates_per_query,
        )

        rows.append(
            bench_osprey_lutrank(
                dataset=ds,
                ds_dir=ds_dir,
                ds_meta=ds_meta,
                osprey_bin=args.osprey_bin,
                runner_bin=args.runner_bin,
                candidates_per_query=args.candidates_per_query,
                max_queries=args.max_queries,
                cand_file=cand_file,
                candidate_source=actual_source,
            )
        )

        rows.append(
            bench_colbert_numpy_cpu(
                dataset=ds,
                ds_meta=ds_meta,
                candidate_source=actual_source,
                queries=queries,
                tokens_by_doc=tokens_by_doc,
                cands_list=cands_list,
            )
        )

        for dev, n_threads in [("cpu", 1), ("cpu", None), ("cuda", None)]:
            pt_row = bench_colbert_pytorch(
                dataset=ds,
                ds_meta=ds_meta,
                candidate_source=actual_source,
                queries=queries,
                tokens_by_doc=tokens_by_doc,
                cands_list=cands_list,
                device_str=dev,
                num_threads=n_threads,
            )
            if pt_row is not None:
                rows.append(pt_row)

        onnx_row = bench_colbert_onnx(
            dataset=ds,
            ds_dir=ds_dir,
            ds_meta=ds_meta,
            candidate_source=actual_source,
            queries=queries,
            tokens_by_doc=tokens_by_doc,
            cands_list=cands_list,
        )
        if onnx_row is not None:
            rows.append(onnx_row)

    out_csv = RESULTS_DIR / "lutrank_vs_colbert.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out_csv}")
        for r in rows:
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
