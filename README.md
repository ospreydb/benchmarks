# OspreyDB Reproducible Benchmarks (`ospreydb/benchmarks`)

Official scientific benchmark suite, MinIO `tc netem` S3 latency simulator, hardware profiling scripts, and publication SVG artifact generator for **[OspreyDB](https://github.com/ospreydb/ospreydb)**.

## 1. Benchmark Scope (Phase 14)

| Task | Scope | Script / Manifest |
| :--- | :--- | :--- |
| **171** | Public dataset download & preparation (MS MARCO Passages, BEIR SciFact/FiQA, SIFT1M, GIST1M, Cohere Wikipedia 768d) | `scripts/download_datasets.py` |
| **172** | Local S3 simulation with MinIO + `tc netem delay 15ms` | `docker/docker-compose.yml`, `docker/Dockerfile.netem` |
| **173** | OspreyDB vs. Qdrant (HNSW) vs. `pgvector` (HNSW) — build time, peak RSS (`VmHWM`), disk size, `Recall@10`, QPS | `scripts/bench_compare_engines.py` |
| **174** | `LUT-Rank` CPU (AVX-512 `vpshufb`) vs. ColBERTv2 (PyTorch CPU/GPU & NumPy BLAS) — latency, QPS, `NDCG@10`, `MRR@10` | `scripts/bench_lutrank_vs_colbert.py` |
| **175–176** | Hardware counter audit (`perf stat`), OS `rusage`, ISA disassembly audit (`prefetcht0`, `vpopcntq`, `vpshufb`), and Flamegraphs | `scripts/profile_perf.sh` |
| **177** | Software prefetching (`_mm_prefetch` `_MM_HINT_T0`) in `.osi` scanner & `.osr` lattice evaluator | Implemented in `ospreydb/ospreydb` (`osprey-core`, `osprey-simd`, `osprey-storage`) |
| **178** | Pareto frontier (`nprobe × refine_m`) for `Recall@10 vs Latency (ms) vs S3 Range-GETs` (`WaRP` ON vs OFF) | `scripts/bench_pareto_s3.py`, `rust/src/main.rs` |
| **179** | Academic publication vector SVG figures | `scripts/generate_plots.py` -> `results/figures/` |
| **180** | Single-command full reproduction & raw artifact publishing | `make bench-all` |

## 2. Quickstart

```bash
# 1. Create Python environment and install dependencies
make setup

# 2. Build native AVX-512 OspreyDB CLI + Rust benchmark runner
#    The runner consumes the public SearchOptions::resolve API from osprey-core
#    (commit 2e8e2f3 "feat(core): centralize adaptive low-dim oversampling policy").
make build-rust OSPREY_SRC=/home/ramos/repositories/ospreydb/ospreydb

# 3. Run the complete benchmark suite (datasets -> Docker/MinIO -> engines -> LUT-Rank -> Pareto -> perf -> SVGs)
make bench-all
```

### Adaptive low-dimensional oversampling (`SearchOptions::resolve`)

`osprey_core::SearchOptions` centralizes Stage-1 sizing for every caller:

| Field | Meaning |
| :--- | :--- |
| `nprobe` | Requested Stage-1 TSP clusters. |
| `candidates` | Requested Stage-1 RaBitQ candidate pool (`refine_m`). `0` = policy default: `2000` for `orig_dim <= 256`, `300` otherwise. |
| `top_k` | Final result count. |
| `rerank_top_m` | Stage-4 survivors (`0` = disabled). |
| `refine_multiplier` | `0` = adaptive (2x when `orig_dim <= 256`), `1` = disabled, `N > 1` = forced `Nx`. |

For `orig_dim <= 256` the adaptive rule doubles `nprobe` and `candidates` and applies a
128-vector exact-`f16` re-score floor. For `orig_dim > 256` the resolved plan is identical
to the requested plan (`effective == requested`, `low_dim_oversample == false`).

When `--refine-m` is omitted, `scripts/bench_compare_engines.py` sends `candidates = 0` and
lets `SearchOptions::resolve` choose the dimension-aware default: SIFT1M (`d=128`) becomes
`nprobe=32`, `candidates=2000` → resolved `64/4000`; Cohere 768d stays `32/300`. An explicit
`--refine-m` always bypasses the auto default.

The runner resolves one `SearchOptions` per `(nprobe, refine_m)` pair, passes
`effective_nprobe` + `stage1_limit` to `OsiIndex::search_candidates`, and reports both the
requested and resolved sizing. New flags/columns:

- `osprey-bench-runner search-eval --refine-multiplier <0|1|N>` (default `0`);
  `--refine-list` accepts `0` to request the auto candidate default.
- `SearchEvalPoint` adds `requested_nprobe`, `effective_nprobe`, `requested_candidates`,
  `effective_candidates`, `refine_limit`, `stage1_limit`, `auto_candidates`,
  `resolved refine_multiplier`, `orig_dim`, and `low_dim_oversample`.
- `scripts/bench_compare_engines.py --refine-multiplier <...> [--refine-m N] [--out-csv PATH]`
  and `scripts/bench_pareto_s3.py --refine-multiplier <...> [--out-csv PATH]` forward the
  policy to the runner. `--refine-multiplier` accepts multiple values to sweep policies in
  one run and tags each row with `refine_policy` + `requested_refine_multiplier`; the compare
  script also writes `candidates_mode=auto|explicit`.
- Core constraint: `OsiIndex::search_candidates` rejects `top_m > 4096`
  (`crates/osprey-core/src/rabitq.rs`). The low-dim auto default resolves to `4000`, which
  fits; adaptive doubling therefore only works up to `candidates <= 2048` per request.
- Measured on the prepared `SIFT1M` index (`1M × 128`, `K=256`): the official auto default
  (`32/2000` → `64/4000`) reaches `Recall@10 = 0.969`. The isolated A/B at `8/64` still
  raises recall from `0.415` to `0.5505` but is intentionally below `0.95`; there the
  candidate pool, not `nprobe`, is the limiter.

Official comparison (refreshes the published `results/compare_engines.csv` with auto defaults):
```bash
make bench-compare
```

### Controlled, recall-matched comparison (`--controlled`)

`scripts/bench_controlled.py` (invoked by `scripts/bench_compare_engines.py --controlled`)
removes the main methodological biases of the legacy comparison:

- Calibrates each engine on 200 validation queries until `Recall@10 >= target - 0.005`
  for the targets `0.95` and `0.97`, then selects the cheapest configuration and reports
  `ok` or `below_target`.
- Uses HTTP client-server for OspreyDB (`osprey-bench-runner serve`) and Qdrant, and SQL
  client-server for pgvector. No in-process Osprey timing.
- Runs 200 warmup plus 1000 measured queries in 3 independent rounds, reporting median and
  standard deviation of P50/P95/P99 and concurrent QPS with 16 workers.
- Samples Peak RSS from `VmHWM` for the active process (Osprey server or Docker container)
  and records index/artifact size.
- Starts exactly one engine at a time: Qdrant/pgvector containers are stopped before the
  next engine starts and after each dataset, so background optimization or autovacuum from
  one engine cannot contaminate another engine's stage.

Prepared data: SIFT1M (`1M × 128`, 1500 queries) and Cohere 768d (`100k × 768`, 1000
queries). All vectors are L2-normalized, so cosine and inner-product rankings are identical
for Qdrant/pgvector and OspreyDB.

Fairness caveats:

- Cohere is the 100k subset, not 1M. Its 1000 queries leave 600 unique measured queries
  after calibration and warmup, which are cycled to reach 1000 measured requests.
- OspreyDB runs with a bounded 256 MB cache (`--controlled-cache-mb 256`) while
  Qdrant/pgvector keep their full indexes in RAM. This favors the competitors on latency
  and is intentional: it measures OspreyDB's bounded-RAM path. `VmHWM` is reported so the
  memory trade-off is explicit.
- The engines use different index families (OspreyDB 1-bit RaBitQ + f16 rescore, HNSW FP32,
  and HNSW 1-bit BQ + rescore). Recall is matched, not the algorithm.
- There is no CPU pinning or cgroup isolation; run on an idle machine for publication
  numbers. Cohere 768d at 100k is the supported scale in this repository.

The full SIFT1M + Cohere run rebuilds four indexes and calibrates an on-disk HNSW index, so
it takes tens of minutes on a spinning disk. Progress is printed per calibration
configuration and per measurement round; the Python entrypoint line-buffers stdout even
when piped through `tee`. Use `--quick` or restrict `--datasets`/`--engines` for a fast
sanity check.

```bash
# Build both binaries against the local core checkout.
make build-rust OSPREY_SRC=/home/ramos/repositories/ospreydb/ospreydb

# Smoke test: small grids and splits.
.venv/bin/python3 scripts/bench_compare_engines.py --controlled --quick \
  --datasets cohere768d \
  --osprey-bin /home/ramos/repositories/ospreydb/ospreydb/target/release/osprey \
  --runner-bin rust/target/release/osprey-bench-runner \
  --out-controlled-csv /tmp/controlled_smoke.csv

# Full controlled run (writes results/compare_engines_controlled.csv).
make bench-compare-controlled

# Equivalent explicit invocation.
.venv/bin/python3 scripts/bench_compare_engines.py --controlled \
  --datasets sift1m cohere768d --recall-target 0.95 0.97 \
  --engines osprey,qdrant_fp32,qdrant_bq,pgvector \
  --workers 16 --warmup-queries 200 --measure-queries 1000 --rounds 3 \
  --controlled-cache-mb 256 \
  --osprey-bin /home/ramos/repositories/ospreydb/ospreydb/target/release/osprey \
  --runner-bin rust/target/release/osprey-bench-runner \
  --out-controlled-csv results/compare_engines_controlled.csv
```

Isolated A/B and Pareto commands (write new files, leaving the published CSVs intact):
```bash
# SIFT1M baseline (8/64) vs adaptive (16/128)
make bench-compare-adaptive
# equivalent explicit invocation
.venv/bin/python3 scripts/bench_compare_engines.py \
  --datasets sift1m --nprobe 8 --refine-m 64 --refine-multiplier 1 0 \
  --skip-qdrant --skip-pgvector \
  --osprey-bin /home/ramos/repositories/ospreydb/ospreydb/target/release/osprey \
  --runner-bin rust/target/release/osprey-bench-runner \
  --out-csv results/compare_engines_adaptive.csv

# SIFT1M MinIO Pareto sweep, both policies
make bench-pareto-adaptive
```

> **Note on `perf stat` and `cargo flamegraph`:** Hardware PMU counters (`cycles`, `instructions`, `L1-dcache-load-misses`, `LLC-load-misses`, `branch-misses`) and kernel-sampled CPU flamegraphs require `kernel.perf_event_paranoid <= 1` (or `CAP_PERFMON`). When `perf_event_paranoid > 1`, `scripts/profile_perf.sh` logs the kernel restriction, records `/usr/bin/time -v` OS telemetry + `osprey --json` stage breakdowns, and runs an `objdump` ISA audit verifying `prefetcht0`, `vpopcntq`, `vpshufb`, and `vpternlogq` emission.

## 3. Published Artifacts (`results/`)

- `results/compare_engines.csv` — OspreyDB vs. Qdrant vs. `pgvector` across SIFT & Cohere 768d, using the adaptive auto candidate default (`candidates=0`).
- `results/compare_engines_adaptive.csv` — isolated SIFT1M A/B for `refine_multiplier=1` (baseline, 8/64) vs `0` (adaptive, 16/128), plus Cohere 768d no-op rows.
- `results/pareto_s3.csv` — 39-point `nprobe × refine_m` sweep against MinIO with `tc netem 15ms` (`WaRP` ON vs OFF).
- `results/pareto_s3_adaptive.csv` — SIFT1M Pareto sweep for `refine_multiplier` 1 and 0, with requested/effective sizing per point.
- `results/lutrank_vs_colbert.csv` — `LUT-Rank` CPU vs. ColBERTv2 latency, QPS, `NDCG@10`, and `MRR@10`.
- `results/perf/` — `perf stat` / `rusage` logs, ISA disassembly counts, and flamegraph status.
- `results/figures/` — Vector SVG figures (`fig1_ram_reduction_60x.svg`, `fig2_warp_s3_requests_16x.svg`, `fig3_lutrank_cpu_vs_gpu.svg`, `fig4_pareto_recall_latency_s3.svg`).
