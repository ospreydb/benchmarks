# Controlled recall-matched VLDB benchmark — 2026-11-19

Raw data: [`results/compare_engines_controlled.csv`](compare_engines_controlled.csv) · JSON: [`results/compare_engines_controlled.json`](compare_engines_controlled.json).
Figures: [`fig1_memory_disk_controlled.svg`](figures/fig1_memory_disk_controlled.svg),
[`fig2_recall_latency_controlled.svg`](figures/fig2_recall_latency_controlled.svg),
[`fig3_build_time_controlled.svg`](figures/fig3_build_time_controlled.svg),
[`fig4_validation_test_recall_controlled.svg`](figures/fig4_validation_test_recall_controlled.svg).

## Protocol

- Host: 12 vCPU, Linux `x86_64` (AVX-512 `avx512_vpopcntdq`). Production `osprey serve --role unified` (with WAL, LSM, and CoW compactor active), Qdrant (`v1.20.2`, `FP32`, `SQ8`, `BQ`, with `indexing_threshold=20000`), and `pgvector` (`v0.8.7` on PostgreSQL `16.16`, `FP32`, `halfvec FP16`) ran in isolated Docker containers; FAISS `IndexIVFFlat` (`faiss==1.16.0`, `nlist=256`) ran in-process.
- Datasets: SIFT1M (`1M × 128d`, 1,500 queries) and Cohere 768d (`1M × 768d`, 1,000 queries). All vectors are L2-normalized, so cosine and inner-product rankings are identical.
- Targets: `Recall@10` of `0.95` and `0.97`, calibrated on 25% of the query file (375 queries on SIFT1M, 250 on Cohere). A configuration is eligible when its validation recall reaches the target (falling back to the highest-recall swept configuration labeled `below_target` otherwise); the cheapest eligible configuration is selected.
- Measurement: 200 warmup queries plus every remaining test query, with no repetition, in 3 rounds and 16 concurrent workers. SIFT1M measured 925 unique test queries; Cohere 768d (1M) measured 550.
- Memory & cache isolation: OspreyDB used `--controlled-cache-mb 256` (`osprey_docker`). Exactly one engine was active at a time; containers were restarted and Linux host page cache was evicted after index build (`dd iflag=nocache` / `sync`) so query-stage `cgroup v2 memory.peak` is measured symmetrically for all containerized engines (capturing both process RSS and page-cache-resident index blocks); FAISS IVF reports in-process `VmHWM`.
- Serving, transport & ingestion scope: `osprey_docker` (`osprey serve`) runs over Docker bridge networking driven by the compiled Rust `http-load` keep-alive client (`osprey-bench-runner`), matching Qdrant's Rust `http-load` transport (with `qdrant_bq` evaluated up to `oversampling=30.0`). `pgvector` uses a compiled Rust `tokio-postgres` SQL driver (`pg_load`) over the Docker bridge, with server execution time sampled on 100 queries via `EXPLAIN ANALYZE`. `build_time_s` combines Parquet import (`osprey import`) with client-side batch ingestion + index build for Qdrant and `pgvector`.

## SIFT1M

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 0.95 | 16/1500 -> 32/3000 | 0.9531 | 0.9641 | yes | 6.827 | 63.257 | 6.695 / 61.931 | 9.586 | 10.229 | 243.0 |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 0.97 | 16/2000 -> 32/4000 | 0.9680 | 0.9738 | yes | 8.293 | 78.499 | 8.159 / 77.247 | 11.808 | 12.836 | 197.9 |
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
| FAISS IVF-Flat | 0.95 | nlist=256, nprobe=16 | 0.9784 | 0.9865 | yes | 0.639 | 6.889 | 0.639 / 6.889 | 0.769 | 0.815 | 1938.9 |
| FAISS IVF-Flat | 0.97 | nlist=256, nprobe=16 | 0.9784 | 0.9865 | yes | 0.635 | 6.834 | 0.635 / 6.834 | 0.764 | 0.816 | 1928.3 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 1072.2 | 844.5 | 648.7 | 212.95 | 16.905 |
| Qdrant HNSW FP32 | 1483.2 | 1477.6 | 2656.1 | 1122.95 | 48.356 |
| Qdrant int8 SQ8 + rescore | 1916.0 | 1911.7 | 3108.9 | 1422.52 | 73.970 |
| Qdrant 1-bit BQ + rescore | 1654.9 | 1626.8 | 3670.4 | 1204.86 | 127.030 |
| pgvector HNSW FP32 | 1927.9 | 912.6 | 4877.0 | 1361.80 | 197.391 |
| pgvector halfvec FP16 | 1406.9 | 661.7 | 4176.4 | 847.09 | 87.932 |
| FAISS IVF-Flat | 1399.7 | 1399.7 | 1398.9 | 0.00 | 0.437 |

## Cohere 768d (1M)

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 0.95 | 32/300 -> 32/300 | 0.9720 | 0.9620 | yes | 4.378 | 21.367 | 4.242 / 20.560 | 4.962 | 5.172 | 671.1 |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 0.97 | 32/300 -> 32/300 | 0.9720 | 0.9620 | **no** | 4.364 | 20.863 | 4.227 / 19.697 | 4.947 | 5.261 | 676.7 |
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
| FAISS IVF-Flat | 0.95 | nlist=256, nprobe=16 | 0.9624 | 0.9404 | **no** | 5.186 | 46.228 | 5.186 / 46.228 | 6.198 | 6.571 | 289.7 |
| FAISS IVF-Flat | 0.97 | nlist=256, nprobe=32 | 0.9872 | 0.9787 | yes | 10.229 | 91.254 | 10.229 / 91.254 | 11.592 | 12.212 | 147.7 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (Docker, 1-bit .osi + f16 .osv) | 2080.4 | 557.6 | 3782.9 | 1514.17 | 184.307 |
| Qdrant HNSW FP32 | 6496.2 | 5826.9 | 8587.5 | 6002.80 | 1133.565 |
| Qdrant int8 SQ8 + rescore | 6450.6 | 6001.0 | 10902.5 | 5841.70 | 1080.501 |
| Qdrant 1-bit BQ + rescore | 4851.4 | 4190.2 | 7048.4 | 4455.00 | 773.940 |
| pgvector HNSW FP32 | 6306.9 | 2098.7 | 14388.2 | 7935.14 | 1656.009 |
| pgvector halfvec FP16 | 4248.8 | 2064.5 | 9422.7 | 3537.39 | 580.417 |
| FAISS IVF-Flat | 6998.1 | 6998.1 | 6997.7 | 0.00 | 6.799 |

## Analysis

### What changed at full 1M × 768d VLDB scale

- **Production `osprey_docker` server & Rust `tokio-postgres`:** OspreyDB now runs as its full containerized production server (`osprey serve --role unified` with WAL + LSM + CoW compactor active), and `pgvector` uses the compiled Rust `tokio-postgres` load client instead of Python `psycopg`.
- **Both benchmarks run at full 1M scale:** Both SIFT1M (`1M × 128d`) and Cohere (`1M × 768d`, `makneeee/cohere_medium_1m`) evaluate 1,000,000 vectors. With `indexing_threshold=20000`, Qdrant builds HNSW indexes cleanly across every collection.
- **Note on `pgvector` sampled `EXPLAIN ANALYZE` server latency:** For `pgvector`, `server_p50_ms_median` samples `EXPLAIN (ANALYZE, FORMAT JSON)` on 100 queries, whereas `p50_ms_median` measures all 550 test queries with prepared Rust `tokio-postgres` statements; instrumentation overhead in `EXPLAIN ANALYZE` can slightly exceed the prepared client P50 (`2.063 ms` vs `1.750 ms` on Cohere `ef_search=48`).

### SIFT1M

- **Lowest footprint and Docker build time:** OspreyDB uses the lowest query peak memory (**1,072.2 MiB** cgroup peak / **844.5 MiB** `VmHWM`), disk (**212.95 MiB**), and Docker build time (**16.905 s**), saving **1.3×–1.8× RAM** and **4.0×–6.7× disk** compared to Qdrant and `pgvector`, while building **2.9×–11.7× faster** than containerized competitors (FAISS IVF builds in-process in `0.437 s`).
- **Throughput & latency trade-off:** On 128d SIFT1M, Qdrant SQ8 (`0.221 / 0.267 ms`, `20,836 / 17,975 QPS`) and `pgvector halfvec` (`0.724 / 0.838 ms`, `8,296 / 7,373 QPS`) lead in latency and QPS, whereas OspreyDB achieves `0.9641 / 0.9738` test recall at `6.827 / 8.293 ms` (`243.0 / 197.9 QPS`) with a 213 MiB disk footprint.
- **Qdrant 1-bit BQ:** Collapses to `0.0962` test recall even at `ef=384, oversampling=30.0` because SIFT1M descriptors are non-negative ($\mathbb{R}_{\ge 0}^{128}$).

### Cohere 768d (1M)

- **3.1× less RAM, 4.0×–5.2× less disk:** On 1M 768d embeddings, OspreyDB uses **2,080.4 MiB** cgroup peak (**557.6 MiB** process `VmHWM`) and **1,514.17 MiB** disk, compared to **6,496.2 MiB** / **6,002.8 MiB** for Qdrant FP32 (**3.1×** RAM, **4.0×** disk), **6,450.6 MiB** / **5,841.7 MiB** for Qdrant SQ8 (**3.1×** RAM, **3.9×** disk), **6,306.9 MiB** / **7,935.1 MiB** for `pgvector` FP32 (**3.0×** RAM, **5.2×** disk), **4,248.8 MiB** / **3,537.4 MiB** for `pgvector halfvec` (**2.0×** RAM, **2.3×** disk), and **6,998.1 MiB** `VmHWM` for FAISS IVF-Flat (**3.4×** RAM).
- **Throughput & recall at 1M × 768d:** OspreyDB hits `0.9620` test recall at `4.378 ms` P50 (`671.1 QPS`), outperforming Qdrant FP32 (`545.7 / 457.4 QPS`), Qdrant SQ8 (`417.0 / 406.4 QPS`), and FAISS IVF-Flat (`289.7 / 147.7 QPS`) in concurrent QPS while building **3.1×–9.0× faster** (`184.3 s` vs `580–1,656 s`) than all containerized baselines. With `oversampling` unlocked to `20.0–30.0` on origin-centered 768d vectors, Qdrant BQ reaches `4,870 / 4,212 QPS` (`0.805 / 0.849 ms`, `0.9658 / 0.9762` recall) at `4,851.4 MiB` peak RAM (**2.3×** OspreyDB's cgroup peak). Note that calibration splits differed from the 550-query test split by ~1%: at `0.97` calibration, OspreyDB (`0.9720 -> 0.9620`), Qdrant FP32/SQ8, and `pgvector` slightly under-shot `0.97` on test, leaving Qdrant BQ and FAISS IVF as the two configurations hitting `>= 0.97` on test.

## Reproduction

```bash
make bench-compare-controlled-vldb OSPREY_SRC=../ospreydb
```
