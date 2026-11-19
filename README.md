# OspreyDB Benchmarks

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Reproducible benchmark suite for **[OspreyDB](https://github.com/ospreydb/ospreydb)**. Evaluates three subsystems against production baselines: recall-matched single-vector ANN search (`OspreyDB` 4-bit/1-bit RaBitQ + `f16` rescore vs. `Qdrant` HNSW FP32 / INT8 SQ8 / 1-bit BQ, `pgvector` HNSW FP32 / FP16 `halfvec`, and `FAISS` IVF-Flat), late-interaction reranking (`LUT-Rank` AVX-512 CPU vs. ColBERTv2 on CPU and CUDA GPU), and object-storage vector search (DP range coalescing with Hilbert page layout over MinIO with `tc netem 15ms` latency).

## Results

### SIFT1M (1M × 128d)

Recall-matched VLDB evaluation (`Recall@10` targets `0.95` and `0.97`, 925 unique test queries, 16 concurrent workers, L2-normalized SIFT1M vectors, OspreyDB running as its containerized production `osprey serve --role unified` server with WAL/LSM/CoW active, 4-bit RaBitQ `--bits 4 --clusters 1024`, and `--controlled-cache-mb 256`, Qdrant v1.20.2 FP32/SQ8/BQ via Rust `http-load`, `pgvector` 0.8.7 FP32/`halfvec` via Rust `tokio-postgres`, FAISS `IndexIVFFlat` (`nlist=1024`) in-process, and symmetric query-only `cgroup v2 memory.peak` after post-build page-cache eviction). Raw data: [`results/compare_engines_controlled.csv`](results/compare_engines_controlled.csv).

| Engine | Target | R@10 | P50 (ms) | P95 (ms) | QPS (16w) | RAM (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OspreyDB** | 0.95 | 0.9639 | 0.44 | 0.55 | 16,125 | **382** | **356** | 5.7 |
| **OspreyDB** | 0.97 | 0.9721 | 0.91 | 1.08 | 8,503 | **382** | **356** | 5.7 |
| Qdrant FP32 | 0.95 | 0.9524 | 0.53 | 0.65 | 7,548 | 1,483 | 1,123 | 48.4 |
| Qdrant FP32 | 0.97 | 0.9625 (miss) | 0.83 | 1.00 | 4,522 | 1,483 | 1,123 | 48.4 |
| Qdrant SQ8 | 0.95 | 0.9637 | **0.22** | **0.25** | **20,836** | 1,916 | 1,423 | 74.0 |
| Qdrant SQ8 | 0.97 | 0.9852 | **0.27** | **0.31** | **17,975** | 1,916 | 1,423 | 74.0 |
| Qdrant 1-bit BQ | 0.95 | 0.0962 (miss) | 0.49 | 0.60 | 9,378 | 1,655 | 1,205 | 127.0 |
| Qdrant 1-bit BQ | 0.97 | 0.0962 (miss) | 0.50 | 0.60 | 9,653 | 1,655 | 1,205 | 127.0 |
| pgvector FP32 | 0.95 | 0.9720 | 0.82 | 1.04 | 6,513 | 1,928 | 1,362 | 197.4 |
| pgvector FP32 | 0.97 | 0.9811 | 0.96 | 1.24 | 5,810 | 1,928 | 1,362 | 197.4 |
| pgvector halfvec | 0.95 | 0.9663 | 0.72 | 0.92 | 8,296 | 1,407 | 847 | 87.9 |
| pgvector halfvec | 0.97 | 0.9759 | 0.84 | 1.12 | 7,373 | 1,407 | 847 | 87.9 |
| FAISS IVF | 0.95 | **0.9813** | 0.35 | 0.47 | 3,710 | 1,342 | 0 | **3.0** |
| FAISS IVF | 0.97 | **0.9959** | 0.70 | 0.88 | 1,850 | 1,342 | 0 | **3.0** |

- **Footprint & build speed:** OspreyDB leads all engines in query peak memory (`382.2 MiB` cgroup peak; `372.3 MiB` process `VmHWM`) and disk (`355.72 MiB`, **2.4×–4.0× smaller**), using **5.0× less RAM** than `pgvector` FP32 (`1,927.9 MiB`) and Qdrant SQ8 (`1,916.0 MiB`), **3.9× less** than Qdrant FP32 (`1,483.2 MiB`), and **3.7× less** than `pgvector halfvec` (`1,406.9 MiB`), while building **8.4×–34.4× faster** than containerized Qdrant and `pgvector` (`5.7 s` vs `48.4–197.4 s`).
- **Latency & quantization:** At `0.95` (`40/20`, `R@10 = 0.9639`), OspreyDB reaches `0.44 ms` P50 (`0.37 ms` server) and **`16,125 QPS`** (**2×** `pgvector halfvec` and Qdrant FP32, and 77% of Qdrant SQ8's `20,836 QPS` at **1/5th the RAM**). Meanwhile, Qdrant 1-bit BQ (`always_ram=true, rescore=true, ef=384, oversampling=30.0`) collapses to `R@10 = 0.0962` on non-negative SIFT1M vectors.

### Cohere (1M × 768d)

550 unique test queries (250 calibration, 200 warmup, 550 test) over 1,000,000 768d embeddings. Raw data: [`results/compare_engines_controlled.csv`](results/compare_engines_controlled.csv).

| Engine | Target | R@10 | P50 (ms) | P95 (ms) | QPS (16w) | RAM (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OspreyDB** | 0.95 | 0.9567 | **0.69** | **0.91** | **6,743** | **2,019** | **2,004** | **30.2** |
| **OspreyDB** | 0.97 | 0.9727 | 0.94 | 1.28 | **5,123** | **2,019** | **2,004** | **30.2** |
| Qdrant FP32 | 0.95 | 0.9429 (miss) | 2.83 | 3.22 | 546 | 6,496 | 6,003 | 1,133.6 |
| Qdrant FP32 | 0.97 | 0.9651 (miss) | 3.25 | 3.67 | 457 | 6,496 | 6,003 | 1,133.6 |
| Qdrant SQ8 | 0.95 | 0.9449 (miss) | 4.70 | 5.07 | 417 | 6,451 | 5,842 | 1,080.5 |
| Qdrant SQ8 | 0.97 | 0.9678 (miss) | 4.80 | 5.30 | 406 | 6,451 | 5,842 | 1,080.5 |
| Qdrant 1-bit BQ | 0.95 | **0.9658** | 0.81 | 0.96 | 4,870 | 4,851 | 4,455 | 773.9 |
| Qdrant 1-bit BQ | 0.97 | **0.9762** | **0.85** | **1.09** | 4,212 | 4,851 | 4,455 | 773.9 |
| pgvector FP32 | 0.95 | 0.9418 (miss) | 1.75 | 2.88 | 2,351 | 6,307 | 7,935 | 1,656.0 |
| pgvector FP32 | 0.97 | 0.9593 (miss) | 2.48 | 3.61 | 1,757 | 6,307 | 7,935 | 1,656.0 |
| pgvector halfvec | 0.95 | 0.9496 (miss) | 1.33 | 2.11 | 3,039 | 4,249 | 3,537 | 580.4 |
| pgvector halfvec | 0.97 | 0.9615 (miss) | 1.74 | 2.48 | 2,505 | 4,249 | 3,537 | 580.4 |
| FAISS IVF | 0.95 | 0.9424 (miss) | 3.12 | 4.02 | 515 | 7,275 | 0 | 42.6 |
| FAISS IVF | 0.97 | 0.9718 | 6.11 | 7.47 | 257 | 7,275 | 0 | 42.6 |

- **#1 in QPS, RAM, Disk, and Build Time at 1M × 768d:** OspreyDB is the only engine besides Qdrant BQ meeting both `0.95` (`0.9567`) and `0.97` (`0.9727`) test targets, while delivering the **highest concurrent throughput at both targets (`6,743 / 5,123 QPS`)** and the fastest `0.95` sequential P50 (`0.693 ms`, server `0.578 ms`), using **`2,018.8 MiB` RAM** and **`2,003.5 MiB` disk** and building in **`30.2 s`** (faster even than in-process FAISS IVF at `42.6 s`).
- **Comparison against quantized & FP32 baselines:** OspreyDB delivers **+38% / +22% higher QPS** than Qdrant 1-bit BQ (`4,870 / 4,212 QPS`) at **2.4× less RAM**, **2.2× less disk**, and **25.7× faster build** (`30.2 s` vs `773.9 s`); **2.0×–2.2× higher QPS** than `pgvector halfvec` (`3,039 / 2,505 QPS`) at **2.1× less RAM** and **19.2× faster build**; and **11×–16× higher QPS** than Qdrant FP32/SQ8 at **3.2× less RAM** and **36×–38× faster build**.

![Controlled Peak Memory and Disk Footprint](results/figures/fig1_memory_disk_controlled.svg)

![Controlled Recall@10 vs Client P50 Latency](results/figures/fig2_recall_latency_controlled.svg)

Additional controlled plots: [Index Build Time (`fig3`)](results/figures/fig3_build_time_controlled.svg) · [Validation vs. Test Recall (`fig4`)](results/figures/fig4_validation_test_recall_controlled.svg).

### Late-Interaction Reranking: `LUT-Rank` vs. ColBERTv2

Reranking 200 candidate passages per query across 100 queries on MS MARCO, BEIR SciFact, and BEIR FiQA. `†` denotes the fastest CPU engine. Raw data: [`results/lutrank_vs_colbert.csv`](results/lutrank_vs_colbert.csv).

> **Input & Workload Disclosure:** The published numbers below use **synthetic 128d BLAKE2b trigram-hashed token vectors (`32` doc tokens × `12` query tokens) with oracle `qrels` + deterministic negatives**, NOT neural `colbert-ir/colbertv2.0` embeddings (which are anisotropic and `120–180 × 32` tokens). Re-evaluation with real neural ColBERTv2 embeddings is tracked in the harness upgrade.

| Dataset | Engine | Device | P50 (ms) | P99 (ms) | QPS | NDCG@10 | MRR@10 | Index (MiB) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **MS MARCO** | **OspreyDB LUT-Rank** | **AVX-512 CPU** | **0.329†** | 0.552 | **2,849†** | 0.7768 | 0.7389 | **97.2** |
| MS MARCO | ColBERTv2 (PyTorch) | CPU | 0.416 | **0.534†** | 2,301 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 (ONNX Runtime) | CPU | 0.476 | 0.733 | 2,005 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.801 | 0.899 | 1,217 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 (PyTorch) | CUDA GPU | 0.226 | 0.308 | 4,134 | 0.7876 | 0.7530 | 768.4 |
| **SciFact** | **OspreyDB LUT-Rank** | **AVX-512 CPU** | 0.505 | **0.551†** | 2,122 | 0.5854 | 0.5437 | **10.2** |
| SciFact | ColBERTv2 (PyTorch) | CPU | 0.345 | 1.584 | **2,398†** | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 (ONNX Runtime) | CPU | **0.338†** | 4.574 | 1,365 | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.679 | 0.849 | 1,453 | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 (PyTorch) | CUDA GPU | 0.207 | 0.703 | 2,581 | 0.6006 | 0.5638 | 81.0 |
| **FiQA** | **OspreyDB LUT-Rank** | **AVX-512 CPU** | 0.519 | 0.722 | 2,006 | 0.4257 | 0.4965 | **112.8** |
| FiQA | ColBERTv2 (PyTorch) | CPU | **0.428†** | **0.532†** | **2,220†** | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 (ONNX Runtime) | CPU | 0.469 | 4.090 | 1,050 | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.864 | 0.955 | 1,143 | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 (PyTorch) | CUDA GPU | 0.229 | 0.338 | 4,126 | 0.4389 | 0.5137 | 892.0 |

- `LUT-Rank` (`vpshufb` 4-bit lookup tables synthesized in `6.7–10.8 µs/query`) compresses `.osr` token lattices **7.9×** versus ColBERTv2 across all three datasets (`97.2 MiB` vs. `768.4 MiB` on MS MARCO), which is its primary advantage across all three datasets, while retaining `98.6%` of exact `NDCG@10` on MS MARCO. In latency/throughput, `LUT-Rank` leads CPU on MS MARCO (`2,849 QPS`, `0.329 ms` P50), whereas **PyTorch CPU was faster in P50/QPS on SciFact and FiQA** (`0.345 / 0.428 ms` P50 and `2,398 / 2,220 QPS` vs. LUT-Rank's `0.505 / 0.519 ms` and `2,122 / 2,006 QPS`).

![LUT-Rank CPU vs ColBERTv2 CPU/GPU](results/figures/fig6_lutrank_cpu_vs_gpu.svg)

### Object Storage Search: 4-Bit RaBitQ + DP Range Coalescing over MinIO (`tc netem 15ms`)

25-point `nprobe × refine_m` sweep on Cohere 768d (1M vectors: `513.2 MiB` 4-bit `.osi` + `1,378.7 MiB` Zstd-compressed `.osv`, `--clusters 1024 --bits 4`) stored in MinIO with kernel-injected `tc netem delay 15ms` (`17.46 ms` measured RTT, `200` queries, `--cache-mb 0`). Raw data: [`results/pareto_s3.csv`](results/pareto_s3.csv).

- With 4-bit progressive RaBitQ Stage-1 pruning, Stage-2 only needs **`refine_m = 20` candidates** instead of `200–300`: at **`Recall@10 = 0.9555`** (`nprobe=40, refine_m=20`), OspreyDB issues just **`19.50 S3 Range-GETs`** (`207.9 KB` fetched) in **`20.26 ms` P50**; at **`Recall@10 = 0.9685`** (`48/20`), **`19.50 S3 Range-GETs`** (`206.3 KB`) in **`20.37 ms` P50**; and at **`Recall@10 = 0.9780`** (`64/20`), **`19.51 S3 Range-GETs`** (`206.2 KB`) in **`20.54 ms` P50** (virtually a single network RTT over `17.46 ms` emulated WAN!), reaching **`0.9815` recall** at `64/40` (`38.37 GETs`, `47.21 ms` P50) while keeping local `VmHWM` at **`1,066.5–1,067.2 MiB`** with zero local `.osv` slab cache (1-bit mode `--bits 1` remains available for `321.7–362.6 MiB` `VmHWM` at `127–226 GETs`).
- **4-bit candidate reduction vs. 1-bit coalescing:** Because 4-bit RaBitQ reduces candidate minipages per query by **~10×–15×** (`19.7` raw pages vs. `192–285` raw pages in 1-bit mode), total S3 Range-GETs drop by **8.4×–11.6×** and bytes fetched drop by **19×–32×** (`~206 KB` vs. `3.95–6.69 MB`), while DP coalescing provides an additional `1.01×–1.02×` merge across the sparse 20–40 pages (and `1.33×–3.30×` when `refine_m = 150–300`).

![DP Range Coalescing S3 Range-GET Reduction](results/figures/fig5_coalesced_s3_requests.svg)

![DP Range Coalescing Pareto Frontier: Recall@10 vs S3 Latency](results/figures/fig7_pareto_recall_latency_s3.svg)

## Methodology & FAQ for Skeptics

Full protocol and per-dataset breakdown: [`results/compare_engines_controlled.md`](results/compare_engines_controlled.md).

1. **Why Qdrant 1-bit BQ collapses on SIFT1M (`R@10 = 0.0962`) vs. Cohere 768d (`0.9762`), while OspreyDB 4-bit RaBitQ reaches `0.9721` / `0.9727`:**
   SIFT1M descriptors are non-negative ($\mathbf{x} \in \mathbb{R}_{\ge 0}^{128}$). Standard 1-bit binary quantization takes $\operatorname{sign}(\mathbf{x})$ without centering, mapping almost every coordinate in the positive orthant to `111...1` and destroying Hamming distance discrimination. Cohere 768d embeddings are centered near the origin, so uncentered sign quantization works well (`0.9762` with `oversampling=30.0`). OspreyDB's `RaBitQ` quantizes the **centroid-centered residual** ($\mathbf{x} - \mathbf{c}$) after a randomized Fast Walsh-Hadamard Transform (FWHT) rotation into 4-bit progressive bitplanes (`--bits 4`) or 1-bit codes (`--bits 1`), ensuring balanced variance across all rotated coordinates regardless of input orthant bias, followed by exact `f16` Stage-2 rescoring on just `20` candidates.
2. **Containerized production serving (`osprey_docker`) and Rust client parity:**
   In `compare_engines_controlled.csv`, OspreyDB runs inside its production Docker container (`osprey serve --role unified`, with WAL, LSM, and CoW compactor active), and OspreyDB and Qdrant are driven by the exact same compiled Rust HTTP/1.1 keep-alive load client (`osprey-bench-runner http-load`) over Docker bridge networking (`server_p50_ms_median` is also exported separately). `pgvector` uses a compiled Rust `tokio-postgres` SQL client (`pg_load`) over the Docker bridge, with server execution time sampled on 100 queries via `EXPLAIN ANALYZE`, and FAISS `IndexIVFFlat` (`nlist=1024`, matching OspreyDB's `1024` clusters) runs in-process.
3. **Local NVMe memory (`cgroup v2 memory.peak` / `VmHWM`) vs. S3 zero-slab-RAM:**
   In the local NVMe controlled benchmark, OspreyDB memory-maps the 4-bit `.osi` index and uncompressed local `.osv` slab zero-copy (`382.2 MiB` cgroup peak / `372.3 MiB` `VmHWM` on SIFT1M; `2,018.8 MiB` cgroup peak / `1,922.7 MiB` `VmHWM` on Cohere 1M). In the S3 object-storage benchmark ([`results/pareto_s3.csv`](results/pareto_s3.csv), `--cache-mb 0`), the `1,378.7 MiB` Zstd-compressed `.osv` slab lives 100% in MinIO/S3, so OspreyDB only holds the `.osi` index, directory, and transient Range-GET buffers—keeping total local `VmHWM` at **`1,066.5–1,067.2 MiB`** in 4-bit mode (`19.5 GETs/query`, `20.5 ms` P50 at `0.9780` recall) or **`321.7–362.6 MiB`** in 1-bit mode.
4. **100M-vector S3 vs. RAM TCO formula (back-of-the-envelope lower bound):**
   Because `.osi` + `.osv` footprints scale linearly with vector count ($N$), the 100M-vector cloud storage footprint and request cost can be extrapolated 100× from the 1M Cohere 4-bit artifact (`513.2 MiB` `.osi` + `1,378.7 MiB` `.osv`):
   $$\text{Monthly Cost}_{\text{S3}} = \text{GB}_{\text{osi+osv}} \times \$0.023/\text{GB-mo} + \frac{Q_{\text{month}} \times \text{GETs/query}}{1000} \times \$0.0004 + \text{EC2}_{\text{compute}}$$
   Scaling 100× from 1M to 100M 768d vectors yields `~50.1 GiB` 4-bit `.osi` (`~14.2 GiB` in 1-bit mode) + `~134.6 GiB` `.osv` (in S3) = `184.7 GiB` total (`~$4.25/mo` S3 Standard), plus **`$7.80 per 1M queries`** (`$0.0078 per 1,000 queries` from `1,000,000 × 19.5 / 1,000 × $0.0004`) at `19.5 GETs/query` (`64/20`, `R@10 = 0.9780`) — compared against hosting `~650 GiB` of unquantized FP32 HNSW RAM across memory-optimized EC2 `r6i`/`r7i` instances. **Caveats:** Holding `19.5 GETs/query` constant from `1M` to `100M` vectors assumes `nprobe`/`refine_m` scale sub-linearly with cluster count (`clusters` scales up at 100M), and production competitors also deploy quantized/disk-backed modes (such as Qdrant SQ8/BQ, DiskANN, and `pgvector halfvec`), not only unquantized FP32 HNSW in RAM.
5. **Symmetric accounting & zero query repetition:**
   Every engine is run in isolation, stopped/restarted after index build, and stripped of host OS page cache before calibration (`25%` split) and measurement (`200` warmup + `925` unique SIFT1M / `550` unique Cohere test queries × `3` rounds, `16` concurrent workers).

## Reproduction

Requires Linux (`x86_64` with AVX-512 recommended), Python 3.10+, Rust toolchain, Docker Compose v2, and a sibling checkout of [`ospreydb/ospreydb`](https://github.com/ospreydb/ospreydb) at `../ospreydb`.

```bash
# 1. Create .venv and install Python dependencies
make setup

# 2. Build native osprey CLI and osprey-bench-runner (-C target-cpu=native)
make build-rust OSPREY_SRC=../ospreydb

# 3. Validate Python scripts, CSV/Markdown/SVG consistency, Rust formatting, and Docker Compose config
make check

# 4. Run the recall-matched controlled benchmark (OspreyDB, Qdrant FP32/SQ8/BQ, pgvector FP32/halfvec)
make bench-compare-controlled OSPREY_SRC=../ospreydb

# 5. Or run the full pipeline (datasets -> MinIO/Docker -> controlled + legacy ANN ->
#    LUT-Rank -> DP Coalescing S3 Pareto -> perf/ISA audit -> SVG plots)
make bench-all OSPREY_SRC=../ospreydb
```

Individual targets (`make datasets`, `make minio-up`, `make bench-lutrank`, `make bench-pareto`, `make profile-perf`, `make plots`) are documented in [`Makefile`](Makefile).

### Stage-1 Sizing & Adaptive Oversampling (`SearchOptions::resolve`)

`osprey-bench-runner` delegates Stage-1 candidate sizing to `osprey_core::SearchOptions::resolve` ([`rust/src/main.rs`](rust/src/main.rs)):

- When `candidates = 0` (the default when `--refine-m` is omitted), `SearchOptions::resolve` selects `2000` candidates for `orig_dim <= 256` and `300` for `orig_dim > 256`.
- When `refine_multiplier = 0` (adaptive default), `orig_dim <= 256` doubles `nprobe` and `candidates` (`nprobe=16, refine_m=1500` → effective `32/3000` on 128d SIFT1M, bounded by the `top_m <= 4096` Stage-1 heap limit) and enforces a 128-vector exact-`f16` rescore floor. For `orig_dim > 256` (e.g., Cohere 768d), `effective == requested`.
- Pass `--refine-multiplier 1` to disable adaptive doubling or run `make bench-compare-adaptive OSPREY_SRC=../ospreydb` / `make bench-pareto-adaptive OSPREY_SRC=../ospreydb` to reproduce the isolated A/B sweeps in `results/compare_engines_adaptive.csv` and `results/pareto_s3_adaptive.csv`.

## Published Artifacts (`results/`)

- [`results/compare_engines_controlled.csv`](results/compare_engines_controlled.csv), [`results/compare_engines_controlled.json`](results/compare_engines_controlled.json), [`results/summary.json`](results/summary.json) & [`results/compare_engines_controlled.md`](results/compare_engines_controlled.md) — Recall-matched (`0.95` / `0.97`) evaluation of OspreyDB, Qdrant FP32/SQ8/BQ, and `pgvector` FP32/`halfvec`.
- [`results/compare_engines.csv`](results/compare_engines.csv) & [`results/compare_engines_adaptive.csv`](results/compare_engines_adaptive.csv) — Uncalibrated default comparison and isolated `refine_multiplier` A/B sweep on SIFT1M.
- [`results/lutrank_vs_colbert.csv`](results/lutrank_vs_colbert.csv) — `LUT-Rank` AVX-512 CPU vs. ColBERTv2 (PyTorch CPU/CUDA, ONNX Runtime, NumPy BLAS) on MS MARCO, SciFact, and FiQA.
- [`results/pareto_s3.csv`](results/pareto_s3.csv) & [`results/pareto_s3_adaptive.csv`](results/pareto_s3_adaptive.csv) — DP range coalescing ON vs. OFF Pareto sweeps over MinIO with `tc netem 15ms`.
- [`results/sift1m_adaptive_recall_recovery.json`](results/sift1m_adaptive_recall_recovery.json) — Stage-1 adaptive oversampling recall recovery telemetry on SIFT1M.
- [`results/perf/`](results/perf/) — `perf stat` / `rusage` logs, `objdump` AVX-512 instruction counts (`prefetcht0`, `vpopcntq`, `vpshufb`, `vpternlogq`), and JSON stage breakdowns.
- [`results/figures/`](results/figures/) — Vector SVG plots (`fig1`–`fig7`) generated by [`scripts/generate_plots.py`](scripts/generate_plots.py).

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
