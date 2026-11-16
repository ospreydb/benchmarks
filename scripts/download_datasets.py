#!/usr/bin/env python3
"""
Task 171: Automated public benchmark dataset downloader & preparer for OspreyDB.

Supports:
  - sift1m     : SIFT1M (1M x 128d, ann-benchmarks HDF5 -> Parquet + NPY)
  - gist1m     : GIST1M (1M x 960d, ann-benchmarks HDF5 -> Parquet + NPY)
  - cohere768d : Cohere Wikipedia Embeddings 768d (HuggingFace Parquet -> Parquet + NPY)
  - scifact    : BEIR SciFact (corpus + queries + qrels + 128d token tensors)
  - fiqa       : BEIR FiQA (corpus + queries + qrels + 128d token tensors)
  - msmarco    : MS MARCO Passages (BEIR archive -> corpus + queries + qrels + 128d token tensors)
"""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "datasets" / "raw"
PREP_DIR = ROOT / "datasets" / "prepared"

URLS = {
    "sift1m": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "gist1m": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "cohere768d_100k_train": (
        "https://huggingface.co/datasets/makneeee/cohere_small_100k/"
        "resolve/main/parquet/shuffle_train.parquet"
    ),
    "cohere768d_100k_test": (
        "https://huggingface.co/datasets/makneeee/cohere_small_100k/"
        "resolve/main/parquet/test.parquet"
    ),
    "cohere768d_1m_train": (
        "https://huggingface.co/datasets/makneeee/cohere_medium_1m/"
        "resolve/main/parquet/shuffle_train.parquet"
    ),
    "cohere768d_1m_test": (
        "https://huggingface.co/datasets/makneeee/cohere_medium_1m/"
        "resolve/main/parquet/test.parquet"
    ),
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "msmarco_queries": (
        "https://huggingface.co/datasets/BeIR/msmarco/resolve/main/"
        "queries/queries-00000-of-00001.parquet"
    ),
    "msmarco_corpus": (
        "https://huggingface.co/datasets/BeIR/msmarco/resolve/main/"
        "corpus/corpus-00000-of-00001.parquet"
    ),
    "msmarco_qrels": (
        "https://huggingface.co/datasets/BeIR/msmarco-qrels/resolve/main/dev.tsv"
    ),
}


def download_file(url: str, dest: Path, chunk_size: int = 4 * 1024 * 1024) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} already present ({dest.stat().st_size / 1e6:.2f} MB)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url} -> {dest}")
    for attempt in range(1, 9):
        existing = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
                if resp.status_code == 416:
                    break
                resp.raise_for_status()
                is_resume = resp.status_code == 206 and existing > 0
                mode = "ab" if is_resume else "wb"
                downloaded = existing if is_resume else 0
                content_len = int(resp.headers.get("content-length", 0))
                total = (existing + content_len) if is_resume and content_len > 0 else content_len
                with open(tmp, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = (downloaded / total) * 100.0
                            print(
                                f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)",
                                end="",
                                flush=True,
                            )
                print()
                break
        except (requests.RequestException, TimeoutError) as exc:
            print(f"\n[retry {attempt}/8] {exc}")
            if attempt == 8:
                raise
    tmp.replace(dest)
    return dest


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (mat / norms).astype(np.float32)


def write_parquet_vectors(
    out_path: Path,
    doc_ids: np.ndarray,
    vectors: np.ndarray,
    texts: Optional[List[str]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = vectors.shape[1]
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    vec_col = pa.FixedSizeListArray.from_arrays(flat, dim)
    cols = {
        "doc_id": pa.array(doc_ids, type=pa.uint64()),
        "embedding": vec_col,
    }
    if texts is not None:
        cols["text"] = pa.array(texts, type=pa.string())
    table = pa.table(cols)
    pq.write_table(table, out_path, compression="zstd")


def prepare_hdf5_ann(name: str, url: str, max_base: Optional[int], max_queries: int) -> None:
    raw_file = download_file(url, RAW_DIR / f"{name}.hdf5")
    out_dir = PREP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_file, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)

    if max_base is not None and max_base < len(train):
        train = train[:max_base]
        recompute_gt = True
    else:
        recompute_gt = False

    test = test[:max_queries]
    train_norm = l2_normalize(train)
    test_norm = l2_normalize(test)

    if recompute_gt:
        # Exact cosine/IP ground truth on normalized subset
        sims = test_norm @ train_norm.T
        gt = np.argsort(-sims, axis=1)[:, :100].astype(np.int64)
    else:
        # Recompute exact cosine/IP GT over normalized vectors because OspreyDB normalizes/uses IP
        # Wait: ann-benchmarks neighbors for sift-128-euclidean are L2; for unit-norm L2 == max IP!
        # Recompute exact IP GT when normalized so all engines share identical exact ground truth.
        chunk = 250
        gt_list = []
        for s in range(0, len(test_norm), chunk):
            sims = test_norm[s : s + chunk] @ train_norm.T
            idx = np.argpartition(-sims, kth=99, axis=1)[:, :100]
            row_sims = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-row_sims, axis=1)
            gt_list.append(np.take_along_axis(idx, order, axis=1))
        gt = np.vstack(gt_list).astype(np.int64)

    doc_ids = np.arange(len(train_norm), dtype=np.uint64)
    write_parquet_vectors(out_dir / "base.parquet", doc_ids, train_norm)
    np.save(out_dir / "base.npy", train_norm)
    np.save(out_dir / "queries.npy", test_norm)
    np.save(out_dir / "gt_ip.npy", gt)
    np.save(out_dir / "gt_l2_orig.npy", neighbors[: len(test_norm)])

    meta = {
        "dataset": name,
        "num_vectors": int(len(train_norm)),
        "num_queries": int(len(test_norm)),
        "dim": int(train_norm.shape[1]),
        "normalized": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepared] {name}: {meta['num_vectors']} x {meta['dim']}d, {meta['num_queries']} queries")


def prepare_cohere(scale: str, max_base: Optional[int], max_queries: int) -> None:
    train_key = f"cohere768d_{scale}_train"
    test_key = f"cohere768d_{scale}_test"
    train_pq = download_file(URLS[train_key], RAW_DIR / f"cohere_{scale}_train.parquet")
    test_pq = download_file(URLS[test_key], RAW_DIR / f"cohere_{scale}_test.parquet")

    out_dir = PREP_DIR / "cohere768d"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_train = pq.read_table(train_pq)
    emb_col = "emb" if "emb" in t_train.column_names else t_train.column_names[-1]
    train_vecs = np.asarray(t_train[emb_col].to_pylist(), dtype=np.float32)
    if max_base is not None and max_base < len(train_vecs):
        train_vecs = train_vecs[:max_base]
    train_norm = l2_normalize(train_vecs)

    t_test = pq.read_table(test_pq)
    q_col = "emb" if "emb" in t_test.column_names else t_test.column_names[-1]
    test_vecs = np.asarray(t_test[q_col].to_pylist(), dtype=np.float32)[:max_queries]
    test_norm = l2_normalize(test_vecs)

    chunk = 250
    gt_list = []
    for s in range(0, len(test_norm), chunk):
        sims = test_norm[s : s + chunk] @ train_norm.T
        k = min(100, train_norm.shape[0] - 1)
        idx = np.argpartition(-sims, kth=k, axis=1)[:, :100]
        row_sims = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-row_sims, axis=1)
        gt_list.append(np.take_along_axis(idx, order, axis=1))
    gt = np.vstack(gt_list).astype(np.int64)

    doc_ids = np.arange(len(train_norm), dtype=np.uint64)
    write_parquet_vectors(out_dir / "base.parquet", doc_ids, train_norm)
    np.save(out_dir / "base.npy", train_norm)
    np.save(out_dir / "queries.npy", test_norm)
    np.save(out_dir / "gt_ip.npy", gt)

    meta = {
        "dataset": f"cohere768d-{scale}",
        "num_vectors": int(len(train_norm)),
        "num_queries": int(len(test_norm)),
        "dim": int(train_norm.shape[1]),
        "normalized": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[prepared] cohere768d ({scale}): {meta['num_vectors']} x {meta['dim']}d, "
        f"{meta['num_queries']} queries"
    )


def hash_token_to_128d(token: str) -> np.ndarray:
    """Deterministic unit-norm 128-d lexical-semantic projection for a text token."""
    vec = np.zeros(128, dtype=np.float32)
    # Word + character trigram feature hashing into 128d subspace
    grams = [token] + [token[i : i + 3] for i in range(max(0, len(token) - 2))]
    for g in grams:
        digest = hashlib.blake2b(g.encode("utf-8"), digest_size=16).digest()
        for b in range(4):
            idx = digest[b] & 0x7F
            sign = 1.0 if (digest[b + 4] & 1) == 0 else -1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def text_to_colbert_tokens(text: str, max_tokens: int) -> np.ndarray:
    words = [w.strip(".,;:!?\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w][:max_tokens]
    if not words:
        words = ["<empty>"]
    mat = np.stack([hash_token_to_128d(w) for w in words], axis=0)
    return mat.astype(np.float32)


def prepare_beir_dataset(
    name: str,
    max_docs: Optional[int],
    max_queries: int,
    doc_token_cap: int = 32,
    query_token_cap: int = 12,
) -> None:
    zip_path = download_file(URLS[name], RAW_DIR / f"{name}.zip")
    extract_dir = RAW_DIR / name
    if not extract_dir.exists():
        print(f"[extract] {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)

    # Locate corpus.jsonl, queries.jsonl, qrels
    candidates = [extract_dir, RAW_DIR / name / name]
    base_dir = next((p for p in candidates if (p / "corpus.jsonl").exists()), extract_dir)
    corpus_file = base_dir / "corpus.jsonl"
    queries_file = base_dir / "queries.jsonl"
    qrels_dir = base_dir / "qrels"
    qrel_file = next(
        (qrels_dir / f for f in ["dev.tsv", "test.tsv", "train.tsv"] if (qrels_dir / f).exists()),
        None,
    )

    qrels: Dict[str, Dict[str, int]] = {}
    if qrel_file is not None:
        with open(qrel_file, "r", encoding="utf-8") as fh:
            header = fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(float(parts[2]))
                    if score > 0:
                        qrels.setdefault(qid, {})[did] = score

    queries_map: Dict[str, str] = {}
    with open(queries_file, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if not qrels or qid in qrels:
                queries_map[qid] = obj["text"]
                if len(queries_map) >= max_queries:
                    break

    # Ensure relevant documents for selected queries are included first when capping corpus
    needed_doc_ids = set()
    for qid in queries_map:
        needed_doc_ids.update(qrels.get(qid, {}).keys())

    docs: List[Tuple[str, str]] = []
    seen_ids = set()
    # First pass: collect relevant docs and general docs
    with open(corpus_file, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            did = str(obj["_id"])
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            if did in needed_doc_ids:
                docs.append((did, text))
                seen_ids.add(did)

    limit = max_docs if max_docs is not None else float("inf")
    if len(docs) < limit:
        with open(corpus_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if len(docs) >= limit:
                    break
                obj = json.loads(line)
                did = str(obj["_id"])
                if did in seen_ids:
                    continue
                text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
                docs.append((did, text))
                seen_ids.add(did)

    out_dir = PREP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dense 384d + ColBERT 128d token tensors
    # Dense vector = normalized mean-pool of 384d projection of text
    doc_id_map: Dict[str, int] = {}
    dense_vecs = np.zeros((len(docs), 384), dtype=np.float32)
    doc_tokens_list: List[List[List[float]]] = []
    texts_list: List[str] = []

    for i, (orig_id, txt) in enumerate(docs):
        doc_id_map[orig_id] = i
        toks_128 = text_to_colbert_tokens(txt, doc_token_cap)
        doc_tokens_list.append(toks_128.tolist())
        # Tile/project 128d mean into 384d dense representation
        pooled = toks_128.mean(axis=0)
        dense_vecs[i, :128] = pooled
        dense_vecs[i, 128:256] = np.roll(pooled, 17)
        dense_vecs[i, 256:384] = np.roll(pooled, 43)
        texts_list.append(txt[:256])

    dense_norm = l2_normalize(dense_vecs)
    doc_ids_u64 = np.arange(len(docs), dtype=np.uint64)

    # Write Parquet with doc_id, embedding, text (and tokens JSONL for OspreyDB .osr)
    write_parquet_vectors(out_dir / "base.parquet", doc_ids_u64, dense_norm, texts=texts_list)
    with open(out_dir / "corpus_tokens.jsonl", "w", encoding="utf-8") as fh:
        for i, toks in enumerate(doc_tokens_list):
            fh.write(
                json.dumps(
                    {
                        "doc_id": i,
                        "embedding": dense_norm[i].tolist(),
                        "tokens": toks,
                        "text": texts_list[i],
                    }
                )
                + "\n"
            )

    query_records = []
    for qid, qtext in queries_map.items():
        qtoks = text_to_colbert_tokens(qtext, query_token_cap)
        qpooled = qtoks.mean(axis=0)
        qdense = np.zeros(384, dtype=np.float32)
        qdense[:128] = qpooled
        qdense[128:256] = np.roll(qpooled, 17)
        qdense[256:384] = np.roll(qpooled, 43)
        qdense /= max(float(np.linalg.norm(qdense)), 1e-8)
        rel = {
            str(doc_id_map[d]): s
            for d, s in qrels.get(qid, {}).items()
            if d in doc_id_map
        }
        if not rel and qrels:
            continue
        query_records.append(
            {
                "query_id": qid,
                "text": qtext,
                "embedding": qdense.tolist(),
                "tokens": qtoks.tolist(),
                "qrels": rel,
            }
        )

    with open(out_dir / "queries.json", "w", encoding="utf-8") as fh:
        json.dump(query_records, fh)

    meta = {
        "dataset": name,
        "num_docs": len(docs),
        "num_queries": len(query_records),
        "dense_dim": 384,
        "token_dim": 128,
        "doc_token_cap": doc_token_cap,
        "query_token_cap": query_token_cap,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepared] {name}: {meta['num_docs']} docs, {meta['num_queries']} queries")


class _HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.pos = 0
        self.sess = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        resp = self.sess.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        resp.raise_for_status()
        data = resp.content
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def prepare_msmarco_hf(
    max_docs: int,
    max_queries: int,
    doc_token_cap: int = 32,
    query_token_cap: int = 12,
) -> None:
    qrels_path = download_file(URLS["msmarco_qrels"], RAW_DIR / "msmarco_dev.tsv")
    queries_pq = download_file(URLS["msmarco_queries"], RAW_DIR / "msmarco_queries.parquet")

    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as fh:
        _ = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(float(parts[2]))
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score

    # Stream row group 0 (first 982,425 MS MARCO passages) via HTTP Range-GETs
    head = requests.head(URLS["msmarco_corpus"], allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["content-length"])
    pf = pq.ParquetFile(io.BufferedReader(_HttpRangeFile(head.url, size), buffer_size=4 * 1024 * 1024))
    rg0 = pf.read_row_group(0)
    rg0_ids = rg0["_id"].to_pylist()
    rg0_titles = rg0["title"].to_pylist()
    rg0_texts = rg0["text"].to_pylist()
    rg0_set = set(rg0_ids)

    q_tbl = pq.read_table(queries_pq)
    q_ids = q_tbl["_id"].to_pylist()
    q_texts = q_tbl["text"].to_pylist()

    queries_map: Dict[str, str] = {}
    for qid, qtext in zip(q_ids, q_texts):
        qid_s = str(qid)
        if qid_s in qrels and any(d in rg0_set for d in qrels[qid_s]):
            queries_map[qid_s] = str(qtext)
            if len(queries_map) >= max_queries:
                break

    needed_doc_ids = set()
    for qid in queries_map:
        needed_doc_ids.update(qrels[qid].keys())

    docs: List[Tuple[str, str]] = []
    seen = set()
    for did, title, text in zip(rg0_ids, rg0_titles, rg0_texts):
        did_s = str(did)
        if did_s in needed_doc_ids:
            docs.append((did_s, f"{title or ''} {text or ''}".strip()))
            seen.add(did_s)

    for did, title, text in zip(rg0_ids, rg0_titles, rg0_texts):
        if len(docs) >= max_docs:
            break
        did_s = str(did)
        if did_s in seen:
            continue
        docs.append((did_s, f"{title or ''} {text or ''}".strip()))
        seen.add(did_s)

    out_dir = PREP_DIR / "msmarco"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id_map: Dict[str, int] = {}
    dense_vecs = np.zeros((len(docs), 384), dtype=np.float32)
    doc_tokens_list: List[List[List[float]]] = []
    texts_list: List[str] = []

    for i, (orig_id, txt) in enumerate(docs):
        doc_id_map[orig_id] = i
        toks_128 = text_to_colbert_tokens(txt, doc_token_cap)
        doc_tokens_list.append(toks_128.tolist())
        pooled = toks_128.mean(axis=0)
        dense_vecs[i, :128] = pooled
        dense_vecs[i, 128:256] = np.roll(pooled, 17)
        dense_vecs[i, 256:384] = np.roll(pooled, 43)
        texts_list.append(txt[:256])

    dense_norm = l2_normalize(dense_vecs)
    doc_ids_u64 = np.arange(len(docs), dtype=np.uint64)
    write_parquet_vectors(out_dir / "base.parquet", doc_ids_u64, dense_norm, texts=texts_list)
    with open(out_dir / "corpus_tokens.jsonl", "w", encoding="utf-8") as fh:
        for i, toks in enumerate(doc_tokens_list):
            fh.write(
                json.dumps(
                    {
                        "doc_id": i,
                        "embedding": dense_norm[i].tolist(),
                        "tokens": toks,
                        "text": texts_list[i],
                    }
                )
                + "\n"
            )

    query_records = []
    for qid, qtext in queries_map.items():
        qtoks = text_to_colbert_tokens(qtext, query_token_cap)
        qpooled = qtoks.mean(axis=0)
        qdense = np.zeros(384, dtype=np.float32)
        qdense[:128] = qpooled
        qdense[128:256] = np.roll(qpooled, 17)
        qdense[256:384] = np.roll(qpooled, 43)
        qdense /= max(float(np.linalg.norm(qdense)), 1e-8)
        rel = {str(doc_id_map[d]): s for d, s in qrels.get(qid, {}).items() if d in doc_id_map}
        if not rel:
            continue
        query_records.append(
            {
                "query_id": qid,
                "text": qtext,
                "embedding": qdense.tolist(),
                "tokens": qtoks.tolist(),
                "qrels": rel,
            }
        )

    with open(out_dir / "queries.json", "w", encoding="utf-8") as fh:
        json.dump(query_records, fh)

    meta = {
        "dataset": "msmarco",
        "num_docs": len(docs),
        "num_queries": len(query_records),
        "dense_dim": 384,
        "token_dim": 128,
        "doc_token_cap": doc_token_cap,
        "query_token_cap": query_token_cap,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepared] msmarco: {meta['num_docs']} docs, {meta['num_queries']} queries")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare OspreyDB benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sift1m", "cohere768d", "scifact", "fiqa", "msmarco"],
        choices=["sift1m", "gist1m", "cohere768d", "scifact", "fiqa", "msmarco"],
    )
    parser.add_argument("--sift-limit", type=int, default=1_000_000)
    parser.add_argument("--gist-limit", type=int, default=250_000)
    parser.add_argument("--cohere-scale", choices=["100k", "1m"], default="100k")
    parser.add_argument("--cohere-limit", type=int, default=100_000)
    parser.add_argument("--msmarco-limit", type=int, default=50_000)
    parser.add_argument("--max-queries", type=int, default=500)
    args = parser.parse_args()

    for ds in args.datasets:
        if ds == "sift1m":
            prepare_hdf5_ann("sift1m", URLS["sift1m"], args.sift_limit, args.max_queries)
        elif ds == "gist1m":
            prepare_hdf5_ann("gist1m", URLS["gist1m"], args.gist_limit, args.max_queries)
        elif ds == "cohere768d":
            prepare_cohere(args.cohere_scale, args.cohere_limit, args.max_queries)
        elif ds == "scifact":
            prepare_beir_dataset("scifact", None, args.max_queries)
        elif ds == "fiqa":
            prepare_beir_dataset("fiqa", None, args.max_queries)
        elif ds == "msmarco":
            prepare_msmarco_hf(args.msmarco_limit, args.max_queries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
