# Controlled recall-matched VLDB benchmark — 2026-11-19

Raw data: [`results/compare_engines_controlled.csv`](compare_engines_controlled.csv) · JSON: [`results/compare_engines_controlled.json`](compare_engines_controlled.json).
Figures: [`fig1_memory_disk_controlled.svg`](figures/fig1_memory_disk_controlled.svg),
[`fig2_recall_latency_controlled.svg`](figures/fig2_recall_latency_controlled.svg),
[`fig3_build_time_controlled.svg`](figures/fig3_build_time_controlled.svg),
[`fig4_validation_test_recall_controlled.svg`](figures/fig4_validation_test_recall_controlled.svg).

## Protocol

- Host: 12 vCPU, Linux `x86_64` (AVX-512 `avx512_vpopcntdq`). Production `osprey serve --role unified` (with WAL, LSM, CoW compactor active, and 4-bit RaBitQ `--bits 4 --clusters 1024`), Qdrant (`v1.20.2`, `FP32`, `SQ8`, `BQ`, with `indexing_threshold=20000`), and `pgvector` (`v0.8.7` on PostgreSQL `16.16`, `FP32`, `halfvec FP16`) ran in isolated Docker containers; FAISS `IndexIVFFlat` (`faiss==1.16.0`, `nlist=1024`) ran in-process.
- Datasets: SIFT1M (`1M × 128d`, 1,500 queries) and Cohere 768d (`1M × 768d`, 1,000 queries). All vectors are L2-normalized, so cosine and inner-product rankings are identical.
- Targets: `Recall@10` of `0.95` and `0.97`, calibrated on 25% of the query file (375 queries on SIFT1M, 250 on Cohere). A configuration is eligible when its validation recall reaches the target (falling back to the highest-recall swept configuration labeled `below_target` otherwise); the cheapest eligible configuration is selected.
- Measurement: 200 warmup queries plus every remaining test query, with no repetition, in 3 rounds and 16 concurrent workers. SIFT1M measured 925 unique test queries; Cohere 768d (1M) measured 550.
- Memory & cache isolation: OspreyDB used `--controlled-cache-mb 256` (`osprey_docker`). Exactly one engine was active at a time; containers were restarted and Linux host page cache was evicted after index build (`dd iflag=nocache` / `sync`) so query-stage `cgroup v2 memory.peak` is measured symmetrically for all containerized engines (capturing both process RSS and page-cache-resident index blocks); FAISS IVF reports in-process `VmHWM`.
- Serving, transport & ingestion scope: `osprey_docker` (`osprey serve`) runs over Docker bridge networking driven by the compiled Rust `http-load` keep-alive client (`osprey-bench-runner`), matching Qdrant's Rust `http-load` transport (with `qdrant_bq` evaluated up to `oversampling=30.0`). `pgvector` uses a compiled Rust `tokio-postgres` SQL driver (`pg_load`) over the Docker bridge, with server execution time sampled on 100 queries via `EXPLAIN ANALYZE`. `build_time_s` combines Parquet import (`osprey import`) with client-side batch ingestion + index build for Qdrant and `pgvector`.

## SIFT1M

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 0.95 | 40/20 -> 40/20 | 0.9568 | 0.9639 | yes | 0.443 | 0.907 | 0.370 / 0.486 | 0.546 | 0.624 | 16125.3 |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 0.97 | 96/20 -> 96/20 | 0.9709 | 0.9721 | yes | 0.906 | 1.654 | 0.828 / 1.106 | 1.075 | 1.190 | 8503.3 |
| Qdrant HNSW FP32 | 0.95 | ef=256 | 0.9531 | 0.9524 | yes | 0.532 | 1.957 | 0.413 / 1.324 | 0.646 | 0.723 | 7548.4 |
| Qdrant HNSW FP32 | 0.97 | ef=512 | 0.9672 | 0.9625 | **no** | 0.834 | 3.298 | 0.711 / 2.397 | 0.998 | 1.092 | 4521.8 |
| Qdrant int8 SQ8 + rescore | 0.95 | ef=64, os=2.0 | 0.9568 | 0.9637 | yes | 0.221 | 0.712 | 0.114 / 0.327 | 0.254 | 0.283 | 20835.7 |
| Qdrant int8 SQ8 + rescore | 0.97 | ef=16, os=10.0 | 0.9784 | 0.9852 | yes | 0.267 | 0.841 | 0.152 / 0.434 | 0.310 | 0.345 | 17974.7 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=384, os=30.0 | 0.0915 | 0.0962 | **no** | 0.495 | 1.576 | 0.382 / 0.990 | 0.598 | 0.658 | 9378.1 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=384, os=30.0 | 0.0915 | 0.0962 | **no** | 0.504 | 1.548 | 0.383 / 0.981 | 0.603 | 0.657 | 9653.4 |
| pgvector HNSW FP32 | 0.95 | ef_search=64 | 0.9605 | 0.9720 | yes | 0.817 | 1.906 | 0.808 / n/a | 1.044 | 1.861 | 6513.4 |
| pgvector HNSW FP32 | 0.97 | ef_search=80 | 0.9720 | 0.9811 | yes | 0.965 | 2.190 | 0.925 / n/a | 1.242 | 1.995 | 5809.9 |
| pgvector halfvec FP16 | 0.95 | ef_search=64 | 0.9613 | 0.9663 | yes | 0.724 | 1.604 | 0.616 / n/a | 0.923 | 1.586 | 8295.5 |
| pgvector halfvec FP16 | 0.97 | ef_search=80 | 0.9715 | 0.9759 | yes | 0.838 | 1.794 | 0.717 / n/a | 1.124 | 1.649 | 7373.1 |
| FAISS IVF-Flat | 0.95 | nlist=1024, nprobe=32 | 0.9677 | 0.9813 | yes | 0.349 | 3.381 | 0.349 / 3.381 | 0.474 | 0.517 | 3709.8 |
| FAISS IVF-Flat | 0.97 | nlist=1024, nprobe=64 | 0.9941 | 0.9959 | yes | 0.696 | 7.385 | 0.696 / 7.385 | 0.883 | 1.075 | 1850.0 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 382.2 | 372.3 | 1146.1 | 355.72 | 5.743 |
| Qdrant HNSW FP32 | 1483.2 | 1477.6 | 2656.1 | 1122.95 | 48.356 |
| Qdrant int8 SQ8 + rescore | 1916.0 | 1911.7 | 3108.9 | 1422.52 | 73.970 |
| Qdrant 1-bit BQ + rescore | 1654.9 | 1626.8 | 3670.4 | 1204.86 | 127.030 |
| pgvector HNSW FP32 | 1927.9 | 912.6 | 4877.0 | 1361.80 | 197.391 |
| pgvector halfvec FP16 | 1406.9 | 661.7 | 4176.4 | 847.09 | 87.932 |
| FAISS IVF-Flat | 1342.3 | 1342.3 | 1341.6 | 0.00 | 3.035 |

## Cohere 768d (1M)

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 0.95 | 40/20 -> 40/20 | 0.9584 | 0.9567 | yes | 0.693 | 2.161 | 0.578 / 1.130 | 0.912 | 0.989 | 6742.9 |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 0.97 | 64/20 -> 64/20 | 0.9784 | 0.9727 | yes | 0.941 | 2.766 | 0.826 / 1.852 | 1.276 | 1.803 | 5122.6 |
| Qdrant HNSW FP32 | 0.95 | ef=80 | 0.9544 | 0.9429 | **no** | 2.828 | 26.991 | 2.577 / 23.107 | 3.216 | 3.415 | 545.7 |
| Qdrant HNSW FP32 | 0.97 | ef=192 | 0.9716 | 0.9651 | **no** | 3.246 | 32.288 | 2.959 / 28.442 | 3.666 | 3.851 | 457.4 |
| Qdrant int8 SQ8 + rescore | 0.95 | ef=96, os=1.0 | 0.9548 | 0.9449 | **no** | 4.699 | 36.937 | 4.299 / 32.618 | 5.074 | 5.264 | 417.0 |
| Qdrant int8 SQ8 + rescore | 0.97 | ef=256, os=1.0 | 0.9720 | 0.9678 | **no** | 4.797 | 38.060 | 4.391 / 33.608 | 5.296 | 5.599 | 406.4 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=16, os=20.0 | 0.9692 | 0.9658 | yes | 0.805 | 3.027 | 0.641 / 2.155 | 0.963 | 1.026 | 4870.3 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=16, os=30.0 | 0.9772 | 0.9762 | yes | 0.849 | 3.496 | 0.678 / 2.446 | 1.089 | 1.137 | 4212.1 |
| pgvector HNSW FP32 | 0.95 | ef_search=48 | 0.9520 | 0.9418 | **no** | 1.750 | 5.862 | 2.063 / n/a | 2.880 | 4.074 | 2351.1 |
| pgvector HNSW FP32 | 0.97 | ef_search=80 | 0.9704 | 0.9593 | **no** | 2.485 | 7.960 | 2.684 / n/a | 3.608 | 5.572 | 1757.2 |
| pgvector halfvec FP16 | 0.95 | ef_search=64 | 0.9564 | 0.9496 | **no** | 1.330 | 4.407 | 1.369 / n/a | 2.113 | 3.365 | 3038.9 |
| pgvector halfvec FP16 | 0.97 | ef_search=96 | 0.9708 | 0.9615 | **no** | 1.741 | 5.334 | 1.695 / n/a | 2.475 | 4.506 | 2505.4 |
| FAISS IVF-Flat | 0.95 | nlist=1024, nprobe=32 | 0.9528 | 0.9424 | **no** | 3.115 | 28.663 | 3.115 / 28.663 | 4.019 | 4.427 | 515.4 |
| FAISS IVF-Flat | 0.97 | nlist=1024, nprobe=64 | 0.9788 | 0.9718 | yes | 6.110 | 55.379 | 6.110 / 55.379 | 7.471 | 8.139 | 256.8 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (Docker, 4-bit .osi + f16 .osv) | 2018.8 | 1922.7 | 7689.5 | 2003.51 | 30.163 |
| Qdrant HNSW FP32 | 6496.2 | 5826.9 | 8587.5 | 6002.80 | 1133.565 |
| Qdrant int8 SQ8 + rescore | 6450.6 | 6001.0 | 10902.5 | 5841.70 | 1080.501 |
| Qdrant 1-bit BQ + rescore | 4851.4 | 4190.2 | 7048.4 | 4455.00 | 773.940 |
| pgvector HNSW FP32 | 6306.9 | 2098.7 | 14388.2 | 7935.14 | 1656.009 |
| pgvector halfvec FP16 | 4248.8 | 2064.5 | 9422.7 | 3537.39 | 580.417 |
| FAISS IVF-Flat | 7275.1 | 7275.1 | 7276.1 | 0.00 | 42.574 |

## Analysis

### What changed with 4-bit RaBitQ + subsampled K-Means + zero-copy local tier

- **4-bit RaBitQ (`--bits 4 --clusters 1024`) + progressive bitplane / FastScan SIMD:** Stage-1 quantization error drops enough that Stage-2 `f16` refinement requires only **`refine_m = 20` vectors** on both SIFT1M and Cohere 1M (down from `300–4,000` with 1-bit RaBitQ).
- **Subsampled K-Means + SIMD assignment (`osprey import`):** Training 1,024 centroids on a deterministic Knuth Algorithm-S subsample (`65,536` vectors) builds SIFT1M in **`5.743 s`** and Cohere 1M (`768d`) in **`30.163 s`** (faster than in-process FAISS `IndexIVFFlat` at `42.574 s` and **19×–55× faster** than Qdrant and `pgvector`).
- **Zero-copy local tier (`OsiMappedView` + uncompressed local `.osv` mmap + 64-shard cache):** Eliminates per-query global lock contention, redundant `.osi` heap copies, and local Zstd decompression overhead.

### SIFT1M

- **3.5×–5.0× lower memory and 2.4×–4.0× smaller disk:** OspreyDB leads all engines in query peak memory (**382.2 MiB** cgroup peak / **372.3 MiB** `VmHWM`), disk (**355.72 MiB**), and containerized build time (**5.743 s**), using **5.0× less RAM** than `pgvector` FP32 (`1,927.9 MiB`) and Qdrant SQ8 (`1,916.0 MiB`), **3.9× less** than Qdrant FP32 (`1,483.2 MiB`), and **3.7× less** than `pgvector halfvec` (`1,406.9 MiB`).
- **Throughput & latency:** At target `0.95` (`40/20`, `R@10 = 0.9639`), OspreyDB reaches **`0.443 ms` P50** (`0.370 ms` server) and **`16,125.3 QPS`**, outperforming `pgvector halfvec` (`8,295.5 QPS`), Qdrant FP32 (`7,548.4 QPS`), `pgvector` FP32 (`6,513.4 QPS`), and FAISS IVF (`3,709.8 QPS`), and achieving 77% of Qdrant SQ8's throughput (`20,835.7 QPS`) at **1/5th the RAM**. At target `0.97` (`96/20`, `R@10 = 0.9721`), OspreyDB reaches **`8,503.3 QPS`** (`0.906 ms` P50), ahead of every engine except Qdrant SQ8.
- **Qdrant 1-bit BQ:** Collapses to `0.0962` test recall on non-negative SIFT1M descriptors ($\mathbb{R}_{\ge 0}^{128}$).

### Cohere 768d (1M)

- **#1 in QPS, RAM, Disk, and Build Time across all containerized engines:** On 1M 768d embeddings, OspreyDB is the **only engine** besides Qdrant BQ meeting both `0.95` (`0.9567`) and `0.97` (`0.9727`) test targets, while delivering the **highest concurrent throughput at both targets (`6,742.9 QPS` at `0.95` and `5,122.6 QPS` at `0.97`)** and the fastest `0.95` sequential P50 (**`0.693 ms`**, server `0.578 ms`).
- **2.1×–3.6× less RAM & 19×–55× faster build:** OspreyDB uses **2,018.8 MiB** cgroup peak and **2,003.51 MiB** disk (building in **`30.163 s`**), compared to **4,851.4 MiB** / **4,455.0 MiB** / `773.9 s` for Qdrant BQ (**+38% higher QPS at `0.95` and +22% at `0.97` with 2.4× less RAM and 25.7× faster build**), **4,248.8 MiB** / `580.4 s` for `pgvector halfvec` (**2.2× higher QPS, 2.1× less RAM, 19.2× faster build**), **6,496.2 MiB** / `1,133.6 s` for Qdrant FP32 (**12.4× higher QPS, 3.2× less RAM, 37.6× faster build**), and **6,450.6 MiB** / `1,080.5 s` for Qdrant SQ8 (**16.2× higher QPS, 3.2× less RAM, 35.8× faster build**).

## Reproduction

```bash
make bench-compare-controlled-vldb OSPREY_SRC=../ospreydb
```
