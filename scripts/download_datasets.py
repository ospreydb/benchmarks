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
import string
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "datasets" / "raw"
PREP_DIR = ROOT / "datasets" / "prepared"

URLS = {
    "sift1m": [
        "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "https://huggingface.co/datasets/ann-benchmarks/sift-128-euclidean/resolve/main/sift-128-euclidean.hdf5",
    ],
    "gist1m": [
        "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    ],
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


def download_file(
    url_or_urls: Union[str, Sequence[str]],
    dest: Path,
    chunk_size: int = 4 * 1024 * 1024,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} already present ({dest.stat().st_size / 1e6:.2f} MB)")
        return dest

    urls = [url_or_urls] if isinstance(url_or_urls, str) else list(url_or_urls)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Optional[Exception] = None
    for url in urls:
        print(f"[download] {url} -> {dest}")
        for attempt in range(1, 5):
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
                    tmp.replace(dest)
                    return dest
            except (requests.RequestException, TimeoutError) as exc:
                last_exc = exc
                print(f"\n[retry {attempt}/4] {url}: {exc}")
        if tmp.exists():
            tmp.unlink()
        print(f"[fallback] primary source failed for {dest.name}, trying next mirror...")
    raise RuntimeError(f"All download mirrors failed for {dest.name}: {last_exc}")


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


def prepare_hdf5_ann(
    name: str,
    url: Union[str, Sequence[str]],
    max_base: Optional[int],
    max_queries: int,
) -> None:
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
    k_top = min(100, train_norm.shape[0])
    for s in range(0, len(test_norm), chunk):
        sims = test_norm[s : s + chunk] @ train_norm.T
        if train_norm.shape[0] > k_top:
            idx = np.argpartition(-sims, kth=k_top - 1, axis=1)[:, :k_top]
            row_sims = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-row_sims, axis=1)
            gt_list.append(np.take_along_axis(idx, order, axis=1))
        else:
            gt_list.append(np.argsort(-sims, axis=1)[:, :k_top])
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
    """Deterministic unit-norm 128-d lexical-semantic projection for a text token (synthetic baseline)."""
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


class SyntheticHashEncoder:
    """Explicit opt-in synthetic BLAKE2b trigram hash encoder (for legacy ablation only)."""

    token_encoder_name = "synthetic-hash"

    def __init__(self, doc_token_cap: int = 128, query_token_cap: int = 32) -> None:
        self.model_name = "blake2b-trigram-hash"
        self.doc_token_cap = doc_token_cap
        self.query_token_cap = query_token_cap

    def encode_docs(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [text_to_colbert_tokens(t, self.doc_token_cap) for t in texts]

    def encode_queries(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [text_to_colbert_tokens(t, self.query_token_cap) for t in texts]


class ColBERTv2Encoder:
    """Real neural ColBERTv2 encoder using BERT backbone + 128d linear projection."""

    token_encoder_name = "colbertv2"

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        doc_token_cap: int = 128,
        query_token_cap: int = 32,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real neural ColBERTv2 encoding requires torch, transformers, and huggingface-hub. "
                "Install with: pip install -r requirements-colbert.txt "
                "(or pass --token-encoder synthetic-hash for legacy synthetic tokens)."
            ) from exc

        self._torch = torch
        self._F = F
        self.model_name = model_name
        self.doc_token_cap = doc_token_cap
        self.query_token_cap = query_token_cap
        self.batch_size = max(1, batch_size)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested, but CUDA is not available.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        linear_weight = self._load_linear_weight(model_name, hf_hub_download)
        out_dim, in_dim = linear_weight.shape
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(linear_weight)

        self.bert.to(self.device).eval()
        self.linear.to(self.device).eval()

        # ColBERTv2 uses [unused0] for query marker and [unused1] for doc marker
        unk_id = self.tokenizer.unk_token_id
        q_tid = self.tokenizer.convert_tokens_to_ids("[unused0]")
        d_tid = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.q_marker = "[unused0]" if q_tid is not None and q_tid != unk_id else ""
        self.d_marker = "[unused1]" if d_tid is not None and d_tid != unk_id else ""

        # Skip punctuation and padding tokens in document representations
        self.skiplist_ids = set()
        if self.tokenizer.pad_token_id is not None:
            self.skiplist_ids.add(int(self.tokenizer.pad_token_id))
        for ch in string.punctuation:
            tid = self.tokenizer.convert_tokens_to_ids(ch)
            if tid is not None and tid != unk_id:
                self.skiplist_ids.add(int(tid))

        print(
            f"[colbertv2] loaded {model_name} on {self.device} "
            f"(doc_cap={self.doc_token_cap}, query_cap={self.query_token_cap}, "
            f"batch_size={self.batch_size}, token_dim={out_dim})"
        )

    def _load_linear_weight(self, model_name: str, hf_hub_download):
        torch = self._torch
        local_dir = Path(model_name)
        state_dict = None

        if local_dir.is_dir():
            st_path = local_dir / "model.safetensors"
            bin_path = local_dir / "pytorch_model.bin"
            if st_path.exists():
                from safetensors.torch import load_file

                state_dict = load_file(str(st_path), device="cpu")
            elif bin_path.exists():
                state_dict = self._torch_load_cpu(bin_path)
        else:
            try:
                st_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                from safetensors.torch import load_file

                state_dict = load_file(st_file, device="cpu")
            except Exception:
                bin_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                state_dict = self._torch_load_cpu(Path(bin_file))

        if state_dict is not None:
            for key in ("linear.weight", "model.linear.weight", "colbert_linear.weight"):
                if key in state_dict:
                    return state_dict[key].float()

        raise RuntimeError(
            f"Could not find ColBERT 128d projection 'linear.weight' in {model_name}."
        )

    def _torch_load_cpu(self, path: Path):
        try:
            return self._torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return self._torch.load(str(path), map_location="cpu")

    def _forward_batch(self, input_ids, attention_mask):
        torch = self._torch
        F = self._F
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        use_fp16 = self.device.type == "cuda"
        with torch.inference_mode():
            if use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    proj = self.linear(hidden)
            else:
                hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                proj = self.linear(hidden)
            normed = F.normalize(proj.float(), p=2, dim=-1)
        return normed.cpu().numpy()

    def encode_docs(self, texts: Sequence[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [f"{self.d_marker} {t}".strip() if self.d_marker else t for t in texts[start : start + self.batch_size]]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.doc_token_cap,
                return_tensors="pt",
            )
            embs = self._forward_batch(toks["input_ids"], toks["attention_mask"])
            ids_np = toks["input_ids"].numpy()
            mask_np = toks["attention_mask"].numpy()
            for i in range(len(batch)):
                keep = [
                    j
                    for j in range(ids_np.shape[1])
                    if mask_np[i, j] == 1 and int(ids_np[i, j]) not in self.skiplist_ids
                ]
                if not keep:
                    keep = [j for j in range(ids_np.shape[1]) if mask_np[i, j] == 1][:1] or [0]
                out.append(embs[i, keep].astype(np.float32))
        return out

    def encode_queries(self, texts: Sequence[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id
        for start in range(0, len(texts), self.batch_size):
            batch = [f"{self.q_marker} {t}".strip() if self.q_marker else t for t in texts[start : start + self.batch_size]]
            toks = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.query_token_cap,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]
            # ColBERTv2 query expansion: replace [PAD] with [MASK] and attend to all tokens
            if pad_id is not None and mask_id is not None:
                is_pad = input_ids == pad_id
                input_ids = input_ids.masked_fill(is_pad, mask_id)
                attention_mask = attention_mask.masked_fill(is_pad, 1)
            embs = self._forward_batch(input_ids, attention_mask)
            for i in range(len(batch)):
                out.append(embs[i].astype(np.float32))
        return out


TokenEncoder = Union[ColBERTv2Encoder, SyntheticHashEncoder]


def build_token_encoder(
    token_encoder: str,
    colbert_model: str,
    doc_token_cap: int,
    query_token_cap: int,
    batch_size: int,
    device: str,
) -> TokenEncoder:
    if token_encoder == "synthetic-hash":
        return SyntheticHashEncoder(doc_token_cap=doc_token_cap, query_token_cap=query_token_cap)
    return ColBERTv2Encoder(
        model_name=colbert_model,
        doc_token_cap=doc_token_cap,
        query_token_cap=query_token_cap,
        batch_size=batch_size,
        device=device,
    )


def _tokens_to_384d_dense(tokens: np.ndarray) -> np.ndarray:
    """Mean-pool 128d token vectors and project consistently into 384d space."""
    pooled = tokens.mean(axis=0).astype(np.float32)
    dense = np.empty(384, dtype=np.float32)
    dense[:128] = pooled
    dense[128:256] = np.roll(pooled, 17)
    dense[256:384] = np.roll(pooled, 43)
    return dense


def _encode_and_write_token_dataset(
    name: str,
    docs: List[Tuple[str, str]],
    queries_map: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    encoder: TokenEncoder,
    require_qrels: bool,
) -> None:
    out_dir = PREP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_id_map: Dict[str, int] = {orig_id: i for i, (orig_id, _) in enumerate(docs)}
    doc_texts = [txt for _, txt in docs]
    print(f"[encode] {name}: encoding {len(doc_texts)} docs with {encoder.token_encoder_name} ({encoder.model_name})...")
    doc_toks = encoder.encode_docs(doc_texts)

    dense_vecs = np.stack([_tokens_to_384d_dense(t) for t in doc_toks], axis=0) if doc_toks else np.zeros((0, 384), dtype=np.float32)
    dense_norm = l2_normalize(dense_vecs) if len(dense_vecs) > 0 else dense_vecs
    doc_ids_u64 = np.arange(len(docs), dtype=np.uint64)

    write_parquet_vectors(out_dir / "base.parquet", doc_ids_u64, dense_norm, texts=doc_texts)
    with open(out_dir / "corpus_tokens.jsonl", "w", encoding="utf-8") as fh:
        for i, toks in enumerate(doc_toks):
            fh.write(
                json.dumps(
                    {
                        "doc_id": i,
                        "embedding": dense_norm[i].tolist(),
                        "tokens": toks.tolist(),
                        "text": doc_texts[i],
                    }
                )
                + "\n"
            )

    q_items = list(queries_map.items())
    q_texts = [qt for _, qt in q_items]
    print(f"[encode] {name}: encoding {len(q_texts)} queries with {encoder.token_encoder_name} ({encoder.model_name})...")
    q_toks = encoder.encode_queries(q_texts)

    query_records = []
    for (qid, qtext), qtoks in zip(q_items, q_toks):
        qdense = _tokens_to_384d_dense(qtoks)
        qdense /= max(float(np.linalg.norm(qdense)), 1e-8)
        rel = {
            str(doc_id_map[d]): s
            for d, s in qrels.get(qid, {}).items()
            if d in doc_id_map
        }
        if not rel and (require_qrels or qrels):
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
        "token_encoder": encoder.token_encoder_name,
        "model_name": encoder.model_name,
        "doc_token_cap": encoder.doc_token_cap,
        "query_token_cap": encoder.query_token_cap,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[prepared] {name}: {meta['num_docs']} docs, {meta['num_queries']} queries "
        f"(encoder={meta['token_encoder']}, model={meta['model_name']})"
    )


def prepare_beir_dataset(
    name: str,
    max_docs: Optional[int],
    max_queries: int,
    encoder: TokenEncoder,
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
            _ = fh.readline()
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

    _encode_and_write_token_dataset(
        name=name,
        docs=docs,
        queries_map=queries_map,
        qrels=qrels,
        encoder=encoder,
        require_qrels=False,
    )


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
    encoder: TokenEncoder,
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

    _encode_and_write_token_dataset(
        name="msmarco",
        docs=docs,
        queries_map=queries_map,
        qrels=qrels,
        encoder=encoder,
        require_qrels=True,
    )


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
    parser.add_argument(
        "--cohere-limit",
        type=int,
        default=None,
        help="Optional cap on Cohere base vectors (defaults to full 100k or 1m file).",
    )
    parser.add_argument("--msmarco-limit", type=int, default=50_000)
    parser.add_argument("--max-queries", type=int, default=1500)
    parser.add_argument(
        "--token-encoder",
        choices=["colbertv2", "synthetic-hash"],
        default="colbertv2",
        help="Token embedding encoder for BEIR/MS MARCO datasets (default: colbertv2).",
    )
    parser.add_argument("--colbert-model", default="colbert-ir/colbertv2.0")
    parser.add_argument(
        "--doc-token-cap",
        type=int,
        default=128,
        help="Max tokens per document passage (default: 128, configurable up to 180).",
    )
    parser.add_argument("--query-token-cap", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.doc_token_cap < 1 or args.doc_token_cap > 180:
        parser.error("--doc-token-cap must be between 1 and 180")
    if args.query_token_cap < 1 or args.query_token_cap > 64:
        parser.error("--query-token-cap must be between 1 and 64")

    token_datasets = {"scifact", "fiqa", "msmarco"}
    encoder: Optional[TokenEncoder] = None
    if any(ds in token_datasets for ds in args.datasets):
        encoder = build_token_encoder(
            token_encoder=args.token_encoder,
            colbert_model=args.colbert_model,
            doc_token_cap=args.doc_token_cap,
            query_token_cap=args.query_token_cap,
            batch_size=args.encode_batch_size,
            device=args.device,
        )

    for ds in args.datasets:
        if ds == "sift1m":
            prepare_hdf5_ann("sift1m", URLS["sift1m"], args.sift_limit, args.max_queries)
        elif ds == "gist1m":
            prepare_hdf5_ann("gist1m", URLS["gist1m"], args.gist_limit, args.max_queries)
        elif ds == "cohere768d":
            prepare_cohere(args.cohere_scale, args.cohere_limit, args.max_queries)
        elif ds == "scifact":
            assert encoder is not None
            prepare_beir_dataset("scifact", None, args.max_queries, encoder)
        elif ds == "fiqa":
            assert encoder is not None
            prepare_beir_dataset("fiqa", None, args.max_queries, encoder)
        elif ds == "msmarco":
            assert encoder is not None
            prepare_msmarco_hf(args.msmarco_limit, args.max_queries, encoder)

    return 0


if __name__ == "__main__":
    sys.exit(main())
