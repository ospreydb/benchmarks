SHELL := /bin/bash
PYTHON ?= $(CURDIR)/.venv/bin/python3
PIP ?= $(CURDIR)/.venv/bin/pip
OSPREY_SRC ?= /home/ramos/repositories/ospreydb/ospreydb
OSPREY_BIN ?= $(OSPREY_SRC)/target/release/osprey
RUNNER_BIN := $(CURDIR)/rust/target/release/osprey-bench-runner
COMPOSE_FILE := $(CURDIR)/docker/docker-compose.yml

DATASETS ?= sift1m cohere768d scifact fiqa msmarco
COMPARE_DATASETS ?= sift1m cohere768d
LUTRANK_DATASETS ?= scifact fiqa msmarco
PARETO_DATASET ?= sift1m

.PHONY: help setup build-rust datasets minio-up minio-down bench-compare bench-compare-controlled bench-compare-adaptive bench-lutrank bench-pareto bench-pareto-adaptive profile-perf plots bench-all clean

help:
	@echo "OspreyDB Reproducible Benchmark Suite"
	@echo "  make setup          Create Python .venv and install dependencies"
	@echo "  make build-rust     Compile osprey CLI and osprey-bench-runner (--release, target-cpu=native)"
	@echo "                      Override the core checkout with OSPREY_SRC=/path/to/ospreydb/ospreydb"
	@echo "  make datasets       Download and prepare public datasets (SIFT1M, Cohere768d, SciFact, FiQA, MS MARCO)"
	@echo "  make minio-up       Start MinIO (tc netem 15ms), Qdrant, and pgvector containers"
	@echo "  make minio-down     Stop benchmark containers"
	@echo "  make bench-compare  Run OspreyDB vs Qdrant vs pgvector comparison"
	@echo "  make bench-compare-controlled  Recall-matched controlled run (0.95/0.97, isolated engines)"
	@echo "  make bench-compare-adaptive  SIFT1M A/B: refine_multiplier 1 (baseline) vs 0 (adaptive)"
	@echo "  make bench-lutrank  Run LUT-Rank CPU vs ColBERTv2 benchmark"
	@echo "  make bench-pareto   Run WaRP Pareto sweep over MinIO + tc netem"
	@echo "  make bench-pareto-adaptive  SIFT1M Pareto sweep for refine_multiplier 1 and 0"
	@echo "  make profile-perf   Run perf stat / flamegraphs / ISA audit"
	@echo "  make plots          Generate academic SVG figures in results/figures/"
	@echo "  make bench-all      Run the full end-to-end benchmark & artifact pipeline (incl. controlled)"

setup:
	python3 -m venv --system-site-packages $(CURDIR)/.venv
	$(PIP) install -r $(CURDIR)/requirements.txt

build-rust:
	@if [ -d "$(OSPREY_SRC)/crates/osprey-core" ]; then \
		RUSTFLAGS="-C target-cpu=native" cargo build --manifest-path $(OSPREY_SRC)/Cargo.toml --release --bin osprey; \
		RUSTFLAGS="-C target-cpu=native" cargo build --manifest-path $(CURDIR)/rust/Cargo.toml --release \
			--config 'patch."https://github.com/ospreydb/ospreydb.git".osprey-core.path="$(OSPREY_SRC)/crates/osprey-core"' \
			--config 'patch."https://github.com/ospreydb/ospreydb.git".osprey-simd.path="$(OSPREY_SRC)/crates/osprey-simd"' \
			--config 'patch."https://github.com/ospreydb/ospreydb.git".osprey-storage.path="$(OSPREY_SRC)/crates/osprey-storage"'; \
	else \
		echo "error: $(OSPREY_SRC) does not contain crates/osprey-core" >&2; \
		echo "       pass OSPREY_SRC=/path/to/ospreydb/ospreydb (commit 2e8e2f3 or newer)" >&2; \
		exit 1; \
	fi

datasets:
	$(PYTHON) $(CURDIR)/scripts/download_datasets.py --datasets $(DATASETS)

minio-up:
	mkdir -p $(CURDIR)/docker/minio-data $(CURDIR)/docker/qdrant-data $(CURDIR)/docker/pgvector-data
	chmod 777 $(CURDIR)/docker/minio-data
	docker compose -f $(COMPOSE_FILE) up -d --build
	@sleep 3
	@docker logs osprey-bench-minio-netem 2>&1 | tail -n 5 || true

minio-down:
	docker compose -f $(COMPOSE_FILE) down

bench-compare:
	$(PYTHON) $(CURDIR)/scripts/bench_compare_engines.py \
		--datasets $(COMPARE_DATASETS) \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN)

# Isolated SIFT1M A/B for the adaptive low-dimensional oversampling policy.
# baseline: nprobe=8, refine_m=64, refine_multiplier=1  -> 8/64,  low_dim_oversample=false
# adaptive: nprobe=8, refine_m=64, refine_multiplier=0  -> 16/128, low_dim_oversample=true
bench-compare-controlled:
	$(PYTHON) $(CURDIR)/scripts/bench_compare_engines.py \
		--controlled \
		--datasets $(COMPARE_DATASETS) \
		--recall-target 0.95 0.97 \
		--engines osprey,qdrant_fp32,qdrant_bq,pgvector \
		--workers 16 \
		--warmup-queries 200 \
		--measure-queries 1000 \
		--rounds 3 \
		--controlled-cache-mb 256 \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN) \
		--out-controlled-csv $(CURDIR)/results/compare_engines_controlled.csv

bench-compare-adaptive:
	$(PYTHON) $(CURDIR)/scripts/bench_compare_engines.py \
		--datasets sift1m \
		--nprobe 8 \
		--refine-m 64 \
		--refine-multiplier 1 0 \
		--skip-qdrant \
		--skip-pgvector \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN) \
		--out-csv $(CURDIR)/results/compare_engines_adaptive.csv

bench-lutrank:
	$(PYTHON) $(CURDIR)/scripts/bench_lutrank_vs_colbert.py \
		--datasets $(LUTRANK_DATASETS) \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN)

bench-pareto:
	$(PYTHON) $(CURDIR)/scripts/bench_pareto_s3.py \
		--dataset $(PARETO_DATASET) \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN)

# SIFT1M Pareto sweep for the published baseline (multiplier=1) and adaptive (0) policies.
bench-pareto-adaptive:
	$(PYTHON) $(CURDIR)/scripts/bench_pareto_s3.py \
		--dataset sift1m \
		--refine-multiplier 1 0 \
		--osprey-bin $(OSPREY_BIN) \
		--runner-bin $(RUNNER_BIN) \
		--out-csv $(CURDIR)/results/pareto_s3_adaptive.csv

profile-perf:
	chmod +x $(CURDIR)/scripts/profile_perf.sh
	OSPREY_BIN=$(OSPREY_BIN) OSPREY_SRC=$(OSPREY_SRC) $(CURDIR)/scripts/profile_perf.sh

plots:
	$(PYTHON) $(CURDIR)/scripts/generate_plots.py

bench-all: build-rust datasets minio-up bench-compare bench-compare-controlled bench-lutrank bench-pareto profile-perf plots
	@echo "=== All benchmarks complete! Raw CSVs, perf logs, and SVGs written to results/ ==="
	@ls -lh $(CURDIR)/results/ $(CURDIR)/results/figures/ $(CURDIR)/results/perf/

clean:
	rm -rf $(CURDIR)/artifacts/*
