#!/usr/bin/env bash
# Tasks 175 & 176: Hardware Counters (perf stat) & CPU Flamegraphs (cargo flamegraph)
# for `osprey import`, `osprey search`, and `osprey rerank`, plus assembly prefetch/SIMD audit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSPREY_SRC="${OSPREY_SRC:-$ROOT/../ospreydb}"
OSPREY_BIN="${OSPREY_BIN:-$OSPREY_SRC/target/release/osprey}"
PERF_DIR="$ROOT/results/perf"
DATASET_DIR="${DATASET_DIR:-datasets/prepared/scifact}"
ART_DIR="artifacts/perf_profile"

cd "$ROOT"
mkdir -p "$PERF_DIR" "$ART_DIR"

# Convert any paths under $ROOT to repo-relative paths for clean command lines & logs.
if [[ "$DATASET_DIR" == "$ROOT/"* ]]; then
  DATASET_DIR="${DATASET_DIR#"$ROOT/"}"
fi

sanitize_perf_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  python3 - "$file" "$ROOT" "$OSPREY_SRC" "$OSPREY_BIN" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = sys.argv[2].rstrip("/")
osprey_src = sys.argv[3].rstrip("/")
osprey_bin = sys.argv[4]

text = path.read_text(encoding="utf-8")
if osprey_bin:
    rel_bin = osprey_bin
    if osprey_src and osprey_bin.startswith(osprey_src + "/"):
        rel_bin = "./" + osprey_bin[len(osprey_src) + 1 :]
    else:
        rel_bin = "./target/release/" + Path(osprey_bin).name
    text = text.replace(osprey_bin, rel_bin)
if osprey_src:
    text = text.replace(osprey_src + "/", "./")
if root:
    text = text.replace(root + "/", "")
text = re.sub(r"/home/[^/\s\"']+/[^\s\"']*/target/release/osprey", "./target/release/osprey", text)
text = re.sub(r"/home/[^/\s\"']+/[^\s\"']*/(datasets|artifacts)/", r"\1/", text)
path.write_text(text, encoding="utf-8")
PY
}

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)"
echo "[info] kernel.perf_event_paranoid = $PARANOID" | tee "$PERF_DIR/env_info.txt"
LC_ALL=C lscpu | grep -E "Model name|CPU\(s\)|L1d|L2|L3|Flags" >> "$PERF_DIR/env_info.txt" || true

# 1. Verify software prefetching (_mm_prefetch -> prefetcht0) & AVX-512 instructions in binary
#    Report BOTH total binary counts and counts scoped to osprey_core / osprey_simd demangled functions.
DISPLAY_BIN="$OSPREY_BIN"
if [[ "$DISPLAY_BIN" == "$OSPREY_SRC/"* ]]; then
  DISPLAY_BIN="./${DISPLAY_BIN#"$OSPREY_SRC/"}"
else
  DISPLAY_BIN="./target/release/$(basename "$OSPREY_BIN")"
fi

echo "[audit] Disassembly SIMD & prefetch instruction counts for $DISPLAY_BIN:" | tee "$PERF_DIR/isa_audit.txt"
objdump -d --demangle "$OSPREY_BIN" | awk '
  /^[0-9a-fA-F]+ <.*>:/ {
    in_osprey = ($0 ~ /osprey_core|osprey_simd/) ? 1 : 0
    if (in_osprey) osp_funcs++
    next
  }
  {
    if ($0 ~ /prefetcht0/) { tot_prefetch++; if (in_osprey) osp_prefetch++ }
    if ($0 ~ /vpopcntq/)   { tot_vpopcnt++;  if (in_osprey) osp_vpopcnt++ }
    if ($0 ~ /vpshufb/)    { tot_vpshufb++;  if (in_osprey) osp_vpshufb++ }
    if ($0 ~ /vpternlogq/) { tot_vptern++;   if (in_osprey) osp_vptern++ }
  }
  END {
    printf "prefetcht0_count=%d\n", tot_prefetch + 0
    printf "vpopcntq_count=%d\n", tot_vpopcnt + 0
    printf "vpshufb_count=%d\n", tot_vpshufb + 0
    printf "vpternlogq_count=%d\n", tot_vptern + 0
    printf "osprey_core_simd_functions=%d\n", osp_funcs + 0
    printf "osprey_core_simd_prefetcht0_count=%d\n", osp_prefetch + 0
    printf "osprey_core_simd_vpopcntq_count=%d\n", osp_vpopcnt + 0
    printf "osprey_core_simd_vpshufb_count=%d\n", osp_vpshufb + 0
    printf "osprey_core_simd_vpternlogq_count=%d\n", osp_vptern + 0
    if ((osp_funcs + 0) == 0) {
      print "[note] Binary appears stripped (0 osprey_core/osprey_simd symbols in objdump --demangle); rebuild with CARGO_PROFILE_RELEASE_STRIP=none for symbol-scoped counts."
    }
  }
' | tee -a "$PERF_DIR/isa_audit.txt"
sanitize_perf_file "$PERF_DIR/isa_audit.txt"

# Prepare query JSON + query tokens + candidate IDs for `osprey search` and `osprey rerank`
python3 - "$DATASET_DIR" "$ART_DIR" <<'PY'
import json, sys
from pathlib import Path
ds_dir, art_dir = Path(sys.argv[1]), Path(sys.argv[2])
queries = json.loads((ds_dir / "queries.json").read_text())
q0 = queries[0]
(art_dir / "query_vec.json").write_text(json.dumps(q0["embedding"]))
(art_dir / "query_tokens.json").write_text(json.dumps(q0["tokens"]))
(art_dir / "candidates.txt").write_text("\n".join(str(i) for i in range(200)))
PY

run_perf_or_rusage() {
  local name="$1"
  shift
  local log_file="$PERF_DIR/perf_stat_${name}.txt"
  local out_file="$PERF_DIR/stdout_${name}.json"
  echo "=== Profiling osprey $name ===" | tee "$log_file"
  if [ "$PARANOID" -le 1 ] 2>/dev/null && command -v perf >/dev/null 2>&1; then
    perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses,task-clock \
      -o "$log_file" --append -- "$@" > "$out_file"
  else
    echo "[note] perf hardware PMU events blocked by kernel.perf_event_paranoid=$PARANOID (requires <=1 or CAP_PERFMON)." | tee -a "$log_file"
    echo "[fallback] Collecting /usr/bin/time -v OS/CPU counters + osprey JSON telemetry:" | tee -a "$log_file"
    if [ -x /usr/bin/time ]; then
      /usr/bin/time -v "$@" > "$out_file" 2>> "$log_file"
    else
      "$@" > "$out_file" 2>> "$log_file"
    fi
  fi
  sanitize_perf_file "$log_file"
  sanitize_perf_file "$out_file"
}

# 2. Profile `osprey import`
run_perf_or_rusage "import" \
  "$OSPREY_BIN" import \
    --input "$DATASET_DIR/corpus_tokens.jsonl" \
    --output "$ART_DIR" \
    --id-col doc_id \
    --vec-col embedding \
    --token-col tokens \
    --with-lutrank \
    --clusters 64 \
    --json

# 3. Profile `osprey search`
run_perf_or_rusage "search" \
  "$OSPREY_BIN" search \
    --index "$ART_DIR/segment.osi" \
    --slabs "$ART_DIR/segment.osv" \
    --query "$ART_DIR/query_vec.json" \
    --nprobe 16 \
    --candidates 200 \
    --top-k 10 \
    --json

# 4. Profile `osprey rerank`
run_perf_or_rusage "rerank" \
  "$OSPREY_BIN" rerank \
    --rerank-store "$ART_DIR/segment.osr" \
    --query-tokens "$ART_DIR/query_tokens.json" \
    --candidates-file "$ART_DIR/candidates.txt" \
    --top-k 10 \
    --json

# 5. Profile `osprey bench --with-lutrank` (20k vectors for sustained hot-loop sampling)
run_perf_or_rusage "bench_lutrank" \
  "$OSPREY_BIN" bench \
    --vectors 20000 \
    --dim 384 \
    --queries 100 \
    --clusters 128 \
    --nprobe 24 \
    --with-lutrank \
    --json

# 6. Flamegraphs (`cargo flamegraph`) when PMU access is enabled
FLAME_LOG="$PERF_DIR/flamegraphs_status.txt"
if [ "$PARANOID" -le 1 ] 2>/dev/null && cargo flamegraph --help >/dev/null 2>&1; then
  echo "[flamegraph] Generating CPU flamegraphs via cargo flamegraph..." | tee "$FLAME_LOG"
  (
    cd "$OSPREY_SRC"
    cargo flamegraph --release --bin osprey -o "$PERF_DIR/flamegraph_import.svg" -- \
      import --input "$ROOT/$DATASET_DIR/corpus_tokens.jsonl" --output "$ROOT/$ART_DIR" --id-col doc_id --vec-col embedding --token-col tokens --with-lutrank --clusters 64
    cargo flamegraph --release --bin osprey -o "$PERF_DIR/flamegraph_search.svg" -- \
      search --index "$ROOT/$ART_DIR/segment.osi" --slabs "$ROOT/$ART_DIR/segment.osv" --query "$ROOT/$ART_DIR/query_vec.json" --nprobe 16 --candidates 200 --top-k 10
    cargo flamegraph --release --bin osprey -o "$PERF_DIR/flamegraph_rerank.svg" -- \
      rerank --rerank-store "$ROOT/$ART_DIR/segment.osr" --query-tokens "$ROOT/$ART_DIR/query_tokens.json" --candidates-file "$ROOT/$ART_DIR/candidates.txt" --top-k 10
  )
else
  echo "[flamegraph] Skipped kernel sampling: perf_event_paranoid=$PARANOID (run 'sudo sysctl -w kernel.perf_event_paranoid=-1' and install cargo-flamegraph to emit hardware-sampled SVGs)." | tee "$FLAME_LOG"
fi
