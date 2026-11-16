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
make build-rust

# 3. Run the complete benchmark suite (datasets -> Docker/MinIO -> engines -> LUT-Rank -> Pareto -> perf -> SVGs)
make bench-all
```

> **Note on `perf stat` and `cargo flamegraph`:** Hardware PMU counters (`cycles`, `instructions`, `L1-dcache-load-misses`, `LLC-load-misses`, `branch-misses`) and kernel-sampled CPU flamegraphs require `kernel.perf_event_paranoid <= 1` (or `CAP_PERFMON`). When `perf_event_paranoid > 1`, `scripts/profile_perf.sh` logs the kernel restriction, records `/usr/bin/time -v` OS telemetry + `osprey --json` stage breakdowns, and runs an `objdump` ISA audit verifying `prefetcht0`, `vpopcntq`, `vpshufb`, and `vpternlogq` emission.

## 3. Published Artifacts (`results/`)

- `results/compare_engines.csv` — OspreyDB vs. Qdrant vs. `pgvector` across SIFT & Cohere 768d.
- `results/pareto_s3.csv` — 39-point `nprobe × refine_m` sweep against MinIO with `tc netem 15ms` (`WaRP` ON vs OFF).
- `results/lutrank_vs_colbert.csv` — `LUT-Rank` CPU vs. ColBERTv2 latency, QPS, `NDCG@10`, and `MRR@10`.
- `results/perf/` — `perf stat` / `rusage` logs, ISA disassembly counts, and flamegraph status.
- `results/figures/` — Vector SVG figures (`fig1_ram_reduction_60x.svg`, `fig2_warp_s3_requests_16x.svg`, `fig3_lutrank_cpu_vs_gpu.svg`, `fig4_pareto_recall_latency_s3.svg`).
