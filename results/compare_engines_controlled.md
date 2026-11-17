# Controlled recall-matched benchmark — 2026-11-17

Raw data: [`results/compare_engines_controlled.csv`](compare_engines_controlled.csv).

## Protocol

- Host: 12 vCPU, 29 GiB RAM, SATA `sdc`, Linux 7.0.0-42. OspreyDB ran natively;
  Qdrant and pgvector ran in Docker.
- Datasets: SIFT1M (`1M × 128`, 1500 queries) and Cohere 768d (`100k × 768`,
  1000 queries). All vectors are L2-normalized, so cosine and inner-product
  rankings are identical.
- Targets: `Recall@10` of `0.95` and `0.97`, calibrated on 25% of the query
  file (minimum 200 queries). A configuration is eligible only when its
  validation recall reaches the target; the cheapest eligible configuration is
  selected.
- Measurement: 200 warmup queries plus every remaining test query, with no
  repetition, in 3 rounds and 16 concurrent workers. SIFT measured 925 test
  queries; Cohere measured 550.
- OspreyDB used a bounded 256 MB cache. Exactly one engine was active at a
  time; Qdrant/pgvector containers were stopped between stages.
- Peak memory is `VmHWM` for the OspreyDB process and cgroup v2 `memory.peak`
  for containers. The container metric includes page-cache-resident indexes.
- Client latency is the HTTP/SQL round trip. Server latency is reported by
  OspreyDB/Qdrant; for pgvector it is `EXPLAIN ANALYZE` on a 100-query sample.

## SIFT1M

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 0.95 | 16/1500 -> 32/3000 | 0.9531 | 0.9641 | yes | 2.982 | 8.665 | 2.674 / 2.958 | 3.409 | 3.595 | 782.7 |
| OspreyDB (1-bit .osi + f16 .osv) | 0.97 | 16/2000 -> 32/4000 | 0.9680 | 0.9738 | yes | 3.434 | 9.206 | 3.126 / 3.526 | 3.910 | 4.136 | 784.2 |
| Qdrant HNSW FP32 | 0.95 | ef=32 | 0.9584 | 0.9672 | yes | 0.854 | 8.654 | 0.459 / 0.551 | 0.941 | 1.001 | 834.2 |
| Qdrant HNSW FP32 | 0.97 | ef=64 | 0.9899 | 0.9905 | yes | 0.888 | 8.244 | 0.497 / 0.566 | 0.975 | 1.047 | 827.6 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=256, os=3.0 | 0.0611 | 0.0624 | **no** | 0.726 | 10.231 | 0.344 / 0.479 | 0.846 | 0.908 | 790.1 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=256, os=3.0 | 0.0611 | 0.0624 | **no** | 0.750 | 7.973 | 0.361 / 0.482 | 0.868 | 0.913 | 848.6 |
| pgvector HNSW FP32 | 0.95 | ef_search=64 | 0.9603 | 0.9726 | yes | 2.374 | 7.521 | 2.089 / n/a | 2.887 | 3.068 | 1793.2 |
| pgvector HNSW FP32 | 0.97 | ef_search=96 | 0.9811 | 0.9863 | yes | 3.262 | 10.346 | 2.785 / n/a | 3.949 | 4.154 | 1348.5 |

| Engine | Peak memory (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 462.2 | 212.93 | 17.539 |
| Qdrant HNSW FP32 | 4194.5 | 868.21 | 34.864 |
| Qdrant 1-bit BQ + rescore | 4780.9 | 910.71 | 28.251 |
| pgvector HNSW FP32 | 2594.0 | 2143.73 | 306.482 |

## Cohere 768d (100k)

| Engine | Target | Config | Val R@10 | Test R@10 | Test >= target | P50 client seq (ms) | P50 client conc (ms) | P50 server seq/conc (ms) | P95 seq (ms) | P99 seq (ms) | QPS 16w |
| :--- | ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 0.95 | 32/300 -> 32/300 | 0.9684 | 0.9669 | yes | 0.795 | 8.268 | 0.542 / 0.663 | 0.956 | 1.035 | 907.8 |
| OspreyDB (1-bit .osi + f16 .osv) | 0.97 | 64/300 -> 64/300 | 0.9864 | 0.9889 | yes | 0.972 | 7.368 | 0.714 / 0.880 | 1.181 | 1.256 | 913.2 |
| Qdrant HNSW FP32 | 0.95 | ef=32 | 0.9632 | 0.9647 | yes | 0.703 | 8.083 | 0.281 / 0.331 | 0.776 | 0.829 | 860.1 |
| Qdrant HNSW FP32 | 0.97 | ef=64 | 0.9852 | 0.9855 | yes | 0.856 | 9.002 | 0.437 / 0.519 | 0.967 | 1.017 | 820.3 |
| Qdrant 1-bit BQ + rescore | 0.95 | ef=256, os=3.0 | 0.9424 | 0.9342 | **no** | 0.678 | 7.878 | 0.269 / 0.332 | 0.739 | 0.789 | 862.9 |
| Qdrant 1-bit BQ + rescore | 0.97 | ef=256, os=3.0 | 0.9424 | 0.9342 | **no** | 0.701 | 9.546 | 0.283 / 0.331 | 0.786 | 0.989 | 845.8 |
| pgvector HNSW FP32 | 0.95 | ef_search=64 | 0.9780 | 0.9771 | yes | 1.788 | 5.458 | 1.733 / n/a | 2.223 | 2.442 | 2290.2 |
| pgvector HNSW FP32 | 0.97 | ef_search=64 | 0.9780 | 0.9771 | yes | 1.782 | 5.512 | 1.209 / n/a | 2.227 | 2.430 | 2282.1 |

| Engine | Peak memory (MiB) | Disk (MiB) | Build (s) |
| :--- | ---: | ---: | ---: |
| OspreyDB (1-bit .osi + f16 .osv) | 234.6 | 151.85 | 19.604 |
| Qdrant HNSW FP32 | 2730.5 | 1368.16 | 16.774 |
| Qdrant 1-bit BQ + rescore | 2799.4 | 661.30 | 14.906 |
| pgvector HNSW FP32 | 1332.1 | 1184.48 | 64.016 |

## Analysis

### What improved

- The exact-target calibration removed the previous test-target misses. Of the
  non-BQ rows, only OspreyDB SIFT `0.97` is flagged `below_target` on
  validation (`0.9680`); its measured test recall is `0.9738`, so this is a
  validation-sample false negative, not a recall failure.
- Test queries are no longer repeated. SIFT uses 925 unique test queries and
  Cohere uses 550.
- Concurrent P50/P95/P99 and concurrent server P50/P95/P99 are exported.
- Container memory now includes page cache. pgvector's peak moved from the
  misleading 148 MiB process-only figure to `2594 MiB` (SIFT) and `1332 MiB`
  (Cohere).
- Request bodies are precomputed outside the timed loop; QPS is less
  Python-serialization-bound than before.

### SIFT1M

- Qdrant FP32 is the latency leader: `0.854 ms` / `0.888 ms` client P50 at
  `0.9672` / `0.9905` test recall, with `0.459 ms` / `0.497 ms` sequential
  server P50. It uses `4194.5 MiB` peak memory (`9.1x` OspreyDB) and
  `868 MiB` disk (`4.1x`).
- OspreyDB reaches `0.9641` / `0.9738` test recall at `2.982` / `3.434 ms`
  client P50, using `462.2 MiB` and `213 MiB` disk with the lowest build time
  (`17.5 s`).
- pgvector is close to OspreyDB in latency (`2.374` / `3.262 ms`) with the
  highest QPS (`1793` / `1349`), but uses `2594 MiB` memory, `2144 MiB` disk,
  and `306 s` build time. Its under-load tail is much better than the HTTP
  engines: concurrent P95/P99 `14.2/17.1 ms` and `19.2/22.5 ms`, versus
  `76.1/132.0 ms` for OspreyDB and `61.3/95.1 ms` for Qdrant FP32.
- Qdrant's server concurrent tail is excellent (`P99 1.8 ms` and `1.3 ms`), so
  the high client tail for Qdrant/OspreyDB under 16 workers is dominated by the
  Python HTTP client/connection queue, not by the engines' kernels.
- Qdrant 1-bit BQ is not a valid SIFT operating point: validation/test recall
  is `0.0611` / `0.0624` at the maximum grid point, and peak memory is higher
  than FP32 (`4780.9` vs `4194.5 MiB`). Exclude it from latency comparisons.

### Cohere 768d (100k)

- OspreyDB and Qdrant FP32 are close in latency: `0.795` / `0.972 ms` versus
  `0.703` / `0.856 ms` at the two targets. OspreyDB uses `234.6 MiB` versus
  `2730.5 MiB` peak memory (`11.6x` less) and `152 MiB` versus `1368 MiB` disk
  (`9.0x` less; see the Qdrant compaction caveat below).
- Qdrant server P50 is lower (`0.281` / `0.437 ms`), but both engines have high
  concurrent client tails under 16 workers.
- pgvector has the highest QPS (`2290` / `2282`) but `1.78 ms` client P50
  (`~2.0x` OspreyDB) and `1184 MiB` disk (`7.8x`). Its concurrent tail is the
  best of the three (`P95/P99 ~10.5/13.1 ms`); pgvector does not expose a
  concurrent server-side sample in this CSV.
- Qdrant BQ remains below target (`0.9342` test) and does not reduce peak
  memory versus FP32 (`2799.4` vs `2730.5 MiB`).

### Quantization

OspreyDB's 1-bit RaBitQ Stage-1 plus f16 rescore reaches `0.9738` on SIFT and
`0.9889` on Cohere. Qdrant's binary quantization collapses on 128-d SIFT
(`0.0624`) and remains below target on 768-d Cohere (`0.9342`). BQ also does
not reduce peak memory at these scales because `always_ram=true` plus rescore
keeps binary codes, the graph, and original vectors resident. The negative BQ
result is dimensionality-driven and is part of the published data, not a
filtered-out row.

## Caveats

- **Qdrant disk is an upper bound:** `disk_size_mb` is sampled at the end of the
  Qdrant stage, before background compaction finishes. The post-run Cohere FP32
  collection directory is `804 MiB`, versus `1368 MiB` in the CSV; the BQ data
  directory is `355 MiB`. The OspreyDB disk advantage is larger than the table
  suggests.
- **OspreyDB SIFT `0.97` status:** validation `0.9680` is below target, so the
  row is `below_target`; test recall `0.9738` meets the target. The status is
  conservative, not a failure.
- **QPS is still not server-only:** request bodies are precomputed, but HTTP
  (OspreyDB/Qdrant) and SQL (pgvector) clients differ, and the CSV does not
  export pgvector concurrent server latency. Use server P50 and concurrent
  latency alongside QPS; OspreyDB/Qdrant client tails under 16 workers are
  dominated by the Python HTTP client.
- **Cache asymmetry:** OspreyDB runs with a bounded 256 MB cache while
  Qdrant/pgvector can use the full host page cache. This favors the competitors
  on latency and is intentional for the bounded-RAM path.
- **Cohere scale:** this is the 100k subset, not 1M. The run uses 550 unique
  test queries.
- **Host noise:** no CPU pinning or cgroups around the benchmark itself; desktop
  processes were running. Three rounds showed low sequential variance, but
  concurrent tail numbers are sensitive to client scheduling.
- **Build/load semantics:** Qdrant/pgvector have no separate load phase;
  `load_time_s` is `0` for them. Only OspreyDB separates import from server
  load.

## Reproduction

```bash
cd /home/ramos/repositories/ospreydb/benchmarks
make bench-compare-controlled 2>&1 | tee /tmp/kilo/controlled_full_v2.log
```
