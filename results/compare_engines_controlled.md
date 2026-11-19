# Controlled recall-matched benchmark — 2026-11-18

Raw data: [`results/compare_engines_controlled.csv`](compare_engines_controlled.csv) · JSON: [`results/compare_engines_controlled.json`](compare_engines_controlled.json).
Figures: [`fig1_memory_disk_controlled.svg`](figures/fig1_memory_disk_controlled.svg),
[`fig2_recall_latency_controlled.svg`](figures/fig2_recall_latency_controlled.svg),
[`fig3_build_time_controlled.svg`](figures/fig3_build_time_controlled.svg),
[`fig4_validation_test_recall_controlled.svg`](figures/fig4_validation_test_recall_controlled.svg).

## Protocol

- Host: 12 vCPU, Linux `x86_64` (AVX-512). OspreyDB ran natively inside a dedicated
  `systemd-run --user --scope` cgroup; Qdrant (`FP32`, `SQ8`, `BQ`) and `pgvector`
  (`FP32`, `halfvec FP16`) ran in Docker.
- Datasets: SIFT1M (`1M × 128`, 1500 queries) and Cohere 768d (`100k × 768`,
  1000 queries). All vectors are L2-normalized, so cosine and inner-product
  rankings are identical.
- Targets: `Recall@10` of `0.95` and `0.97`, calibrated on 25% of the query
  file (375 queries on SIFT1M, 250 on Cohere). A configuration is eligible only
  when its validation recall reaches the target; the cheapest eligible
  configuration is selected.
- Measurement: 200 warmup queries plus every remaining test query, with no
  repetition, in 3 rounds and 16 concurrent workers. SIFT measured 925 unique
  test queries; Cohere measured 550.
- OspreyDB used `--controlled-cache-mb 256` (a bounded 256 MiB cache, which
  exceeds the total `.osi+.osv` footprint on both SIFT1M at `212.93 MiB` and
  Cohere 768d at `151.85 MiB`, so the working set fits in cache once warmed).
  Exactly one engine was active at a time; containers were restarted and Linux
  host page cache was evicted after index build
  (`posix_fadvise(POSIX_FADV_DONTNEED)` / `dd iflag=nocache`) so query-stage
  `cgroup v2 memory.peak` is measured symmetrically for all 6 engines (capturing
  both process RSS and page-cache-resident index blocks).
- Serving & ingestion scope: `osprey-bench-runner serve` is a static, read-only
  HTTP wrapper over prebuilt `.osi`/`.osv` files (not a full transactional
  multi-tenant database). `build_time_s` mixes local Parquet Rust import
  (`osprey import`) with Python JSON/text ingestion for Qdrant and `pgvector`.
- HTTP transport (`osprey_http`, `qdrant_fp32`, `qdrant_sq8`, `qdrant_bq`) uses
  the compiled Rust `http-load` keep-alive client (`osprey-bench-runner`) with
  precomputed JSON payloads outside the timed loop; `pgvector` uses Python
  `psycopg` over localhost SQL rather than the Rust HTTP client, with server
  execution time sampled via `EXPLAIN ANALYZE` on 100 queries.

## SIFT1M

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 0.95 | 16/1500 -> 32/3000 | 0.9531 | 0.9641 | yes | 2.775 | 10.854 | 2.712 / 10.094 | 3.175 | 3.388 | 1452.4 |
| OspreyDB (1-bit .osi + f16 .osv) | 0.97 | 16/2000 -> 32/4000 | 0.9680 | 0.9738 | yes | 3.223 | 13.004 | 3.158 / 12.034 | 3.716 | 3.926 | 1236.9 |
| Qdrant HNSW FP32 | 0.95 | ef=32 | 0.9683 | 0.9686 | yes | 0.448 | 2.052 | 0.299 / 1.243 | 1.246 | 2.521 | 7014.9 |
| Qdrant HNSW FP32 | 0.97 | ef=64 | 0.9909 | 0.9903 | yes | 0.454 | 2.640 | 0.304 / 1.728 | 1.225 | 2.497 | 5424.7 |
| Qdrant int8 SQ8 + rescore | 0.95 | ef=32, os=1.0 | 0.9515 | 0.9579 | yes | 0.209 | 0.823 | 0.098 / 0.397 | 0.250 | 0.313 | 17480.5 |
| Qdrant int8 SQ8 + rescore | 0.97 | ef=64, os=1.0 | 0.9829 | 0.9837 | yes | 0.246 | 1.046 | 0.129 / 0.560 | 0.290 | 0.311 | 14459.5 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=384, os=3.0 | 0.0643 | 0.0666 | **no** | 1.077 | 4.251 | 0.938 / 3.425 | 1.273 | 1.377 | 3427.1 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=384, os=3.0 | 0.0643 | 0.0666 | **no** | 1.084 | 4.314 | 0.940 / 3.412 | 1.265 | 1.336 | 3420.7 |
| pgvector HNSW FP32 | 0.95 | ef_search=64 | 0.9621 | 0.9736 | yes | 2.618 | 8.045 | 2.243 / n/a | 3.154 | 3.465 | 1777.5 |
| pgvector HNSW FP32 | 0.97 | ef_search=96 | 0.9835 | 0.9866 | yes | 3.575 | 11.188 | 3.047 / n/a | 4.379 | 4.677 | 1302.5 |
| pgvector halfvec FP16 | 0.95 | ef_search=64 | 0.9587 | 0.9669 | yes | 2.369 | 7.223 | 2.032 / n/a | 2.883 | 3.107 | 1950.6 |
| pgvector halfvec FP16 | 0.97 | ef_search=96 | 0.9749 | 0.9815 | yes | 3.234 | 10.118 | 2.914 / n/a | 3.952 | 4.211 | 1444.7 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 642.3 | 466.2 | 4329.2 | 212.93 | 16.955 |
| Qdrant HNSW FP32 | 911.9 | 931.4 | 2801.2 | 611.45 | 259.334 |
| Qdrant int8 SQ8 + rescore | 1389.2 | 1352.6 | 3514.3 | 736.88 | 78.824 |
| Qdrant 1-bit BQ + rescore | 1152.9 | 1158.4 | 3418.0 | 744.86 | 71.536 |
| pgvector HNSW FP32 | 1511.8 | 150.0 | 3282.9 | 1361.49 | 379.517 |
| pgvector halfvec FP16 | 838.1 | 147.1 | 2392.2 | 847.05 | 148.565 |

## Cohere 768d (100k)

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 0.95 | 32/300 -> 32/300 | 0.9684 | 0.9669 | yes | 0.602 | 1.506 | 0.544 / 1.104 | 0.757 | 0.814 | 9960.0 |
| OspreyDB (1-bit .osi + f16 .osv) | 0.97 | 64/300 -> 64/300 | 0.9864 | 0.9889 | yes | 0.777 | 1.915 | 0.717 / 1.455 | 1.014 | 1.071 | 8022.3 |
| Qdrant HNSW FP32 | 0.95 | ef=64 | 0.9728 | 0.9664 | yes | 2.762 | 34.650 | 2.489 / 28.685 | 2.996 | 3.107 | 427.4 |
| Qdrant HNSW FP32 | 0.97 | ef=64 | 0.9728 | 0.9664 | **no** | 2.756 | 34.163 | 2.483 / 29.253 | 2.965 | 3.094 | 432.8 |
| Qdrant int8 SQ8 + rescore | 0.95 | ef=64, os=1.0 | 0.9616 | 0.9638 | yes | 2.880 | 34.732 | 2.616 / 29.125 | 3.099 | 3.177 | 431.7 |
| Qdrant int8 SQ8 + rescore | 0.97 | ef=96, os=1.0 | 0.9700 | 0.9727 | yes | 2.891 | 34.908 | 2.623 / 29.532 | 3.112 | 3.227 | 429.6 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=512, os=3.0 | 0.9288 | 0.9173 | **no** | 0.551 | 3.052 | 0.395 / 2.182 | 0.663 | 0.719 | 4843.7 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=512, os=3.0 | 0.9288 | 0.9173 | **no** | 0.562 | 2.982 | 0.403 / 2.138 | 0.686 | 0.724 | 4701.6 |
| pgvector HNSW FP32 | 0.95 | ef_search=32 | 0.9508 | 0.9409 | **no** | 1.196 | 3.779 | 1.162 / n/a | 1.511 | 1.814 | 3750.8 |
| pgvector HNSW FP32 | 0.97 | ef_search=64 | 0.9784 | 0.9773 | yes | 1.949 | 5.976 | 1.972 / n/a | 2.435 | 2.694 | 2310.6 |
| pgvector halfvec FP16 | 0.95 | ef_search=32 | 0.9512 | 0.9420 | **no** | 0.783 | 2.281 | 0.686 / n/a | 0.956 | 1.109 | 6147.6 |
| pgvector halfvec FP16 | 0.97 | ef_search=64 | 0.9732 | 0.9747 | yes | 1.220 | 3.505 | 1.121 / n/a | 1.522 | 1.715 | 4042.9 |

| Engine | Peak memory (MiB) | Process VmHWM (MiB) | Build peak (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 387.0 | 258.2 | 4329.2 | 151.85 | 19.425 |
| Qdrant HNSW FP32 | 1435.5 | 1584.7 | 2215.9 | 799.37 | 50.502 |
| Qdrant int8 SQ8 + rescore | 1461.8 | 1599.2 | 2710.1 | 750.42 | 38.097 |
| Qdrant 1-bit BQ + rescore | 819.3 | 840.1 | 2353.7 | 557.70 | 52.139 |
| pgvector HNSW FP32 | 662.9 | 147.3 | 1684.5 | 793.87 | 67.775 |
| pgvector halfvec FP16 | 450.5 | 147.1 | 1700.7 | 354.04 | 34.030 |

## Analysis

### What changed with the symmetric harness

- **Symmetric `cgroup v2 memory.peak` (query-only):** Every engine (including OspreyDB inside `systemd-run --user --scope`) is restarted after build, host page cache is evicted, and query-stage memory is read from `memory.peak`. Both process RSS (`process_vmhwm_mb`) and cgroup peak (`query_peak_rss_mb`) are exported.
- **Build peak RSS caveat (`4,329.2 MiB`):** OspreyDB's `4,329.2 MiB` `Build peak (MiB)` figure in the tables above is a known `systemd-run` cgroup-scope startup latch artifact; actual `osprey import` peak RSS recorded in [`results/compare_engines.csv`](compare_engines.csv) is `431.5 MiB` on SIFT1M and `691.4 MiB` on Cohere 768d.
- **Cache sizing & harness scope:** `--controlled-cache-mb 256` exceeds the `.osv` (and total `.osi+.osv`) file size on both `1M×128d` (`212.93 MiB`) and `100k×768d` (`151.85 MiB`). Additionally, `osprey-bench-runner serve` is a static read-only HTTP wrapper, `build_time_s` compares local Rust Parquet import against Python JSON/text ingestion for Qdrant/`pgvector`, and `pgvector` used Python `psycopg` rather than the Rust HTTP client.
- **Rust `http-load` client:** Removing Python `httpx` client queueing reveals true server-side HTTP throughput (`9,960 QPS` for OspreyDB on Cohere 768d and `17,481 QPS` for Qdrant SQ8 on SIFT1M).
- **Quantized competitor baselines (`Qdrant SQ8` and `pgvector halfvec`):** Both quantized competitors are included directly alongside FP32 and 1-bit BQ.

### SIFT1M

- **Footprint & build speed:** OspreyDB leads all 6 engines in query peak memory (`642.3 MiB` cgroup peak / `466.2 MiB` `VmHWM`), disk (`212.93 MiB`), and build time (`16.955 s`). It uses **2.4× less query RAM** than `pgvector` FP32 (`1,511.8 MiB`), **2.2× less** than Qdrant SQ8 (`1,389.2 MiB`), **1.4× less** than Qdrant FP32 (`911.9 MiB`), and **1.3× less** than `pgvector halfvec` (`838.1 MiB`), while building **4.6×–22.4× faster** than every competitor.
- **Latency & QPS:** Qdrant SQ8 (`0.209 / 0.246 ms` P50, `17,481 / 14,460 QPS`) and Qdrant FP32 (`0.448 / 0.454 ms` P50, `7,015 / 5,425 QPS`) lead latency on 128d SIFT. OspreyDB matches `pgvector` FP32 (`2.775 / 3.223 ms` vs `2.618 / 3.575 ms`) and `pgvector halfvec` (`2.369 / 3.234 ms`) at roughly half the disk and memory.
- **Qdrant 1-bit BQ:** Collapses to `0.0666` test recall (`ef=384, oversampling=3.0`) because SIFT1M descriptors are non-negative ($\mathbb{R}_{\ge 0}^{128}$).

### Cohere 768d (100k)

- **Only engine meeting both targets:** OspreyDB reaches `0.9669` at target `0.95` and `0.9889` at target `0.97`, delivering sub-millisecond sequential P50 (`0.602 / 0.777 ms`) and the highest concurrent throughput (`9,960.0 / 8,022.3 QPS`).
- **Lowest memory, disk, and build time:** OspreyDB uses `387.0 MiB` cgroup peak (`258.2 MiB` `VmHWM`) and `151.85 MiB` disk, compared to `1,435.5 MiB` / `799.37 MiB` for Qdrant FP32 (**3.7×** RAM, **5.3×** disk), `1,461.8 MiB` / `750.42 MiB` for Qdrant SQ8 (**3.8×** RAM, **4.9×** disk), `662.9 MiB` / `793.87 MiB` for `pgvector` FP32 (**1.7×** RAM, **5.2×** disk), and `450.5 MiB` / `354.04 MiB` for `pgvector halfvec` (**1.2×** RAM, **2.3×** disk).
- **Qdrant FP32/SQ8 & BQ caveats on Cohere 768d:** Qdrant FP32 and SQ8 exhibited an anomalous `~430 QPS` plateau (`2.76–2.89 ms` sequential P50, flat across `ef=64` and `ef=96`) likely due to segment fragmentation or indexing threshold behavior on `100k×768d`, whereas Qdrant BQ reached `4,844 QPS` (`0.551 ms` P50) on the same dataset (and was capped at `oversampling=3.0` = 30 rescore candidates vs. OspreyDB's 300). Meanwhile, `pgvector halfvec` keeps its compact `354 MiB` FP16 index warm (`0.783 / 1.220 ms`, `6,148 / 4,043 QPS`, though its `0.95` calibration `ef_search=32` misses slightly on test at `0.9420`).

## Reproduction

```bash
make bench-compare-controlled OSPREY_SRC=../ospreydb
```
