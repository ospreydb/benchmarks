#!/usr/bin/env python3
"""
Verify consistency and quality invariants across benchmark CSVs, JSONs,
Markdown summaries (README.md and results/compare_engines_controlled.md),
and generated SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CONTROLLED_CSV = RESULTS_DIR / "compare_engines_controlled.csv"
CONTROLLED_JSON = RESULTS_DIR / "compare_engines_controlled.json"
CONTROLLED_MD = RESULTS_DIR / "compare_engines_controlled.md"
LUTRANK_CSV = RESULTS_DIR / "lutrank_vs_colbert.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
PARETO_CSV = RESULTS_DIR / "pareto_s3.csv"
PARETO_ADAPTIVE_CSV = RESULTS_DIR / "pareto_s3_adaptive.csv"
PERF_DIR = RESULTS_DIR / "perf"
README_MD = ROOT / "README.md"
FIG_DIR = RESULTS_DIR / "figures"

EXPECTED_FIGURES = [
    "fig1_memory_disk_controlled.svg",
    "fig2_recall_latency_controlled.svg",
    "fig3_build_time_controlled.svg",
    "fig4_validation_test_recall_controlled.svg",
    ("fig5_coalesced_s3_requests.svg", "fig5_coalesced_s3_requests_16x.svg"),
    "fig6_lutrank_cpu_vs_gpu.svg",
    "fig7_pareto_recall_latency_s3.svg",
]


def canonical_engine(name: str) -> str:
    low = name.lower().strip("* ").replace("_", " ")
    if low.startswith("ospreydb") or low.startswith("osprey "):
        return "OspreyDB"
    if "qdrant" in low and "fp32" in low:
        return "Qdrant FP32"
    if "qdrant" in low and ("sq8" in low or "int8" in low):
        return "Qdrant SQ8"
    if "qdrant" in low and ("bq" in low or "1-bit" in low or "1 bit" in low):
        return "Qdrant BQ"
    if "faiss" in low:
        return "FAISS IVF"
    if "pgvector" in low and "halfvec" in low:
        return "pgvector halfvec"
    if "pgvector" in low:
        return "pgvector FP32"
    return name.strip()


def canonical_lutrank_dataset(name: str) -> str:
    low = name.lower().strip("* ")
    if "ms marco" in low or low == "msmarco":
        return "msmarco"
    if "scifact" in low:
        return "scifact"
    if "fiqa" in low:
        return "fiqa"
    return low


def canonical_lutrank_engine(engine: str, device: str) -> str:
    eng_low = engine.lower().strip("* ")
    dev_low = device.lower().strip("* ")
    if "ospreydb" in eng_low or "lut-rank" in eng_low:
        return "OspreyDB LUT-Rank (CPU)"
    if "exact" in eng_low or "numpy" in eng_low or "blas" in dev_low:
        return "ColBERTv2 Exact FP32 (NumPy/BLAS CPU)"
    if "onnx" in eng_low or "onnx" in dev_low:
        return "ColBERTv2 (ONNX Runtime CPU)"
    if "cuda" in dev_low or "gpu" in dev_low:
        return "ColBERTv2 (PyTorch CUDA GPU)"
    if "pytorch" in eng_low:
        return "ColBERTv2 (PyTorch CPU)"
    return f"{engine.strip()} [{device.strip()}]"


def clean_cell_number(cell: str) -> float:
    text = cell.strip()
    text = text.replace("**", "")
    text = re.sub(r"[¹²†]+", "", text)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.replace(",", "").strip()
    return float(text)


def parse_markdown_tables(text: str, section_headers: Dict[str, str]) -> Dict[str, List[List[Dict[str, str]]]]:
    """Extract markdown tables grouped by dataset key based on section headers."""
    lines = text.splitlines()
    current_ds: Optional[str] = None
    tables_by_ds: Dict[str, List[List[Dict[str, str]]]] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        for prefix, ds in section_headers.items():
            if line.startswith(prefix):
                current_ds = ds
                break

        if current_ds and line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            headers = [h.strip() for h in line.strip("|").split("|")]
            i += 2
            rows: List[Dict[str, str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                vals = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(vals) == len(headers):
                    rows.append(dict(zip(headers, vals)))
                i += 1
            tables_by_ds.setdefault(current_ds, []).append(rows)
            continue
        i += 1
    return tables_by_ds


def verify_controlled_csv_and_docs(errors: List[str], warnings: List[str]) -> None:
    if not CONTROLLED_CSV.exists():
        errors.append(f"Missing {CONTROLLED_CSV}")
        return

    with open(CONTROLLED_CSV, "r", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    if not csv_rows:
        errors.append(f"{CONTROLLED_CSV} is empty")
        return

    # Index CSV rows by (dataset, canonical_engine, recall_target)
    csv_by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    csv_mem_by_ds_eng: Dict[Tuple[str, str], float] = {}
    build_rss_by_eng: Dict[str, Dict[str, float]] = {}

    for row in csv_rows:
        ds = row["dataset"]
        eng = canonical_engine(row["engine"])
        tgt = f"{float(row['recall_target']):.2f}"
        csv_by_key[(ds, eng, tgt)] = row
        csv_mem_by_ds_eng[(ds, eng)] = float(row["query_peak_rss_mb"])

        # Sanity checks (warnings unless --strict)
        vmhwm = float(row.get("process_vmhwm_mb") or 0.0)
        q_rss = float(row.get("query_peak_rss_mb") or 0.0)
        if vmhwm > 0.0 and q_rss > 0.0 and vmhwm > q_rss * 1.05:
            warnings.append(
                f"{CONTROLLED_CSV.name} {ds}/{row['variant']} target={tgt}: "
                f"process_vmhwm_mb ({vmhwm:.1f}) > query_peak_rss_mb * 1.05 ({q_rss * 1.05:.1f}) "
                f"[possible cross-cgroup page cache leakage]"
            )

        srv_p50 = float(row.get("server_p50_ms_median") or 0.0)
        cli_p50 = float(row.get("p50_ms_median") or 0.0)
        if srv_p50 > 0.0 and srv_p50 > cli_p50:
            warnings.append(
                f"{CONTROLLED_CSV.name} {ds}/{row['variant']} target={tgt}: "
                f"server_p50_ms_median ({srv_p50:.3f}) > p50_ms_median ({cli_p50:.3f})"
            )

        b_rss = float(row.get("build_peak_rss_mb") or 0.0)
        if b_rss > 0.0:
            build_rss_by_eng.setdefault(row["variant"], {})[ds] = b_rss

    for variant, ds_map in sorted(build_rss_by_eng.items()):
        if len(ds_map) > 1 and len(set(ds_map.values())) < len(ds_map):
            details = ", ".join(f"{ds}={val:.1f}" for ds, val in sorted(ds_map.items()))
            warnings.append(
                f"{CONTROLLED_CSV.name} {variant}: identical build_peak_rss_mb across datasets "
                f"({details}) [possible cgroup race]"
            )

    # 1. Check JSON files if present
    if CONTROLLED_JSON.exists():
        data = json.loads(CONTROLLED_JSON.read_text(encoding="utf-8"))
        j_rows = data.get("rows", data) if isinstance(data, dict) else data
        if len(j_rows) != len(csv_rows):
            errors.append(
                f"{CONTROLLED_JSON} row count ({len(j_rows)}) != {CONTROLLED_CSV} ({len(csv_rows)})"
            )
        else:
            for r_csv, r_json in zip(csv_rows, j_rows):
                for col in ("dataset", "variant", "test_recall", "p50_ms_median", "query_peak_rss_mb"):
                    if abs(float(r_csv[col]) - float(r_json[col])) > 1e-4 if col not in ("dataset", "variant") else str(r_csv[col]) != str(r_json[col]):
                        errors.append(f"{CONTROLLED_JSON} mismatch on {r_csv['dataset']}/{r_csv['variant']} col {col}")

    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        ctrl_summary = summary.get("controlled", {})
        for (ds, eng, tgt), r_csv in csv_by_key.items():
            ds_block = ctrl_summary.get(ds, {})
            if not ds_block:
                continue
            match = [
                item for item in ds_block.get("rows", [])
                if canonical_engine(item.get("engine", "")) == eng
                and f"{float(item.get('recall_target', -1)):.2f}" == tgt
            ]
            if not match:
                errors.append(f"{SUMMARY_JSON} missing controlled row {ds}/{eng}/{tgt}")
            else:
                item = match[0]
                for col in ("test_recall", "p50_ms_median", "query_peak_rss_mb"):
                    if abs(float(r_csv[col]) - float(item[col])) > 1e-3:
                        errors.append(f"{SUMMARY_JSON} mismatch on {ds}/{eng}/{tgt} col {col}")

    # 2. Check README.md tables
    readme_text = README_MD.read_text(encoding="utf-8")
    readme_tables = parse_markdown_tables(
        readme_text,
        {"### SIFT1M": "sift1m", "### Cohere": "cohere768d"},
    )
    for ds in ("sift1m", "cohere768d"):
        ds_tables = readme_tables.get(ds, [])
        if not ds_tables:
            errors.append(f"README.md missing controlled table for {ds}")
            continue
        table = ds_tables[0]
        seen_keys = set()
        for trow in table:
            eng = canonical_engine(trow["Engine"])
            tgt = f"{clean_cell_number(trow['Target']):.2f}"
            key = (ds, eng, tgt)
            seen_keys.add(key)
            if key not in csv_by_key:
                errors.append(f"README.md {ds} row {eng} target={tgt} not found in {CONTROLLED_CSV.name}")
                continue
            crow = csv_by_key[key]
            # Verify recall_at_10, search_p50_ms, memory_rss_mb
            md_r10 = clean_cell_number(trow["R@10"])
            csv_r10 = float(crow["test_recall"])
            if f"{md_r10:.4f}" != f"{csv_r10:.4f}":
                errors.append(f"README.md {key} R@10={md_r10:.4f} != CSV {csv_r10:.4f}")

            md_p50 = clean_cell_number(trow["P50 (ms)"])
            csv_p50 = float(crow["p50_ms_median"])
            if f"{md_p50:.2f}" != f"{csv_p50:.2f}":
                errors.append(f"README.md {key} P50={md_p50:.2f} != CSV {csv_p50:.2f}")

            md_ram = clean_cell_number(trow["RAM (MiB)"])
            csv_ram = float(crow["query_peak_rss_mb"])
            if round(md_ram) != round(csv_ram):
                errors.append(f"README.md {key} RAM={md_ram:.0f} != CSV {csv_ram:.1f} (round {round(csv_ram)})")

        expected_ds_keys = {k for k in csv_by_key if k[0] == ds}
        missing = expected_ds_keys - seen_keys
        if missing:
            errors.append(f"README.md missing rows from CSV for {ds}: {sorted(missing)}")

    # 3. Check results/compare_engines_controlled.md tables
    ctrl_md_text = CONTROLLED_MD.read_text(encoding="utf-8")
    ctrl_tables = parse_markdown_tables(
        ctrl_md_text,
        {"## SIFT1M": "sift1m", "## Cohere": "cohere768d"},
    )
    for ds in ("sift1m", "cohere768d"):
        ds_tables = ctrl_tables.get(ds, [])
        if len(ds_tables) < 2:
            errors.append(f"{CONTROLLED_MD.name} missing perf/footprint tables for {ds}")
            continue
        perf_table, mem_table = ds_tables[0], ds_tables[1]
        seen_perf = set()
        for trow in perf_table:
            eng = canonical_engine(trow["Engine"])
            tgt = f"{clean_cell_number(trow['Target']):.2f}"
            key = (ds, eng, tgt)
            seen_perf.add(key)
            if key not in csv_by_key:
                errors.append(f"{CONTROLLED_MD.name} {ds} row {eng} target={tgt} not found in CSV")
                continue
            crow = csv_by_key[key]
            md_r10 = clean_cell_number(trow["Test R@10"])
            csv_r10 = float(crow["test_recall"])
            if f"{md_r10:.4f}" != f"{csv_r10:.4f}":
                errors.append(f"{CONTROLLED_MD.name} {key} Test R@10={md_r10:.4f} != CSV {csv_r10:.4f}")

            md_p50 = clean_cell_number(trow["P50 client seq (ms)"])
            csv_p50 = float(crow["p50_ms_median"])
            if f"{md_p50:.3f}" != f"{csv_p50:.3f}":
                errors.append(f"{CONTROLLED_MD.name} {key} P50={md_p50:.3f} != CSV {csv_p50:.3f}")

        expected_ds_keys = {k for k in csv_by_key if k[0] == ds}
        missing_perf = expected_ds_keys - seen_perf
        if missing_perf:
            errors.append(f"{CONTROLLED_MD.name} missing perf rows for {ds}: {sorted(missing_perf)}")

        seen_mem = set()
        for mrow in mem_table:
            eng = canonical_engine(mrow["Engine"])
            seen_mem.add((ds, eng))
            if (ds, eng) not in csv_mem_by_ds_eng:
                errors.append(f"{CONTROLLED_MD.name} {ds} memory row {eng} not found in CSV")
                continue
            md_mem = clean_cell_number(mrow["Peak memory (MiB)"])
            csv_mem = csv_mem_by_ds_eng[(ds, eng)]
            if f"{md_mem:.1f}" != f"{csv_mem:.1f}":
                errors.append(f"{CONTROLLED_MD.name} {(ds, eng)} Peak memory={md_mem:.1f} != CSV {csv_mem:.1f}")

        expected_mem_keys = {k for k in csv_mem_by_ds_eng if k[0] == ds}
        missing_mem = expected_mem_keys - seen_mem
        if missing_mem:
            errors.append(f"{CONTROLLED_MD.name} missing footprint rows for {ds}: {sorted(missing_mem)}")


def verify_lutrank_csv_and_readme(errors: List[str]) -> None:
    if not LUTRANK_CSV.exists():
        errors.append(f"Missing {LUTRANK_CSV}")
        return

    with open(LUTRANK_CSV, "r", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    if not csv_rows:
        errors.append(f"{LUTRANK_CSV} is empty")
        return

    csv_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in csv_rows:
        ds = canonical_lutrank_dataset(row["dataset"])
        eng = canonical_lutrank_engine(row["engine"], row["device"])
        key = (ds, eng)
        if key in csv_by_key:
            errors.append(f"{LUTRANK_CSV.name} duplicate row for {key}")
        csv_by_key[key] = row

    readme_text = README_MD.read_text(encoding="utf-8")
    tables = parse_markdown_tables(readme_text, {"### Late-Interaction": "lutrank"})
    lut_tables = tables.get("lutrank", [])
    if not lut_tables:
        errors.append("README.md missing Late-Interaction Reranking table")
        return

    seen_keys = set()
    for trow in lut_tables[0]:
        ds = canonical_lutrank_dataset(trow["Dataset"])
        eng = canonical_lutrank_engine(trow["Engine"], trow["Device"])
        key = (ds, eng)
        seen_keys.add(key)
        if key not in csv_by_key:
            errors.append(f"README.md LUT-Rank row {key} not found in {LUTRANK_CSV.name}")
            continue
        crow = csv_by_key[key]

        for md_col, csv_col, fmt in (
            ("P50 (ms)", "p50_ms", ".3f"),
            ("P99 (ms)", "p99_ms", ".3f"),
            ("NDCG@10", "ndcg_at_10", ".4f"),
            ("MRR@10", "mrr_at_10", ".4f"),
            ("Index (MiB)", "index_mb", ".1f"),
        ):
            md_val = clean_cell_number(trow[md_col])
            csv_val = float(crow[csv_col])
            if f"{md_val:{fmt}}" != f"{csv_val:{fmt}}":
                errors.append(
                    f"README.md LUT-Rank {key} {md_col}={md_val:{fmt}} != CSV {csv_val:{fmt}}"
                )

        md_qps = clean_cell_number(trow["QPS"])
        csv_qps = float(crow["qps"])
        if round(md_qps) != round(csv_qps):
            errors.append(
                f"README.md LUT-Rank {key} QPS={md_qps:.0f} != CSV {csv_qps:.1f} (round {round(csv_qps)})"
            )

    missing = set(csv_by_key.keys()) - seen_keys
    if missing:
        errors.append(f"README.md LUT-Rank table missing rows from CSV: {sorted(missing)}")


def _verify_pareto_rows(
    csv_path: Path,
    required_cols: set[str],
    group_by_policy: bool,
    min_on_recall: Optional[float],
    errors: List[str],
) -> Optional[List[Dict[str, str]]]:
    if not csv_path.exists():
        errors.append(f"Missing {csv_path}")
        return None

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        errors.append(f"{csv_path} is empty")
        return None

    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        errors.append(f"{csv_path.name} missing required columns: {sorted(missing_cols)}")
        return None

    coalesce_on = [r for r in rows if r["coalesce_enabled"].lower() == "true"]
    coalesce_off = [r for r in rows if r["coalesce_enabled"].lower() == "false"]
    if not coalesce_on:
        errors.append(f"{csv_path.name} has no coalesce_enabled=True rows")
        return None
    if not coalesce_off:
        errors.append(f"{csv_path.name} has no coalesce_enabled=False rows")
        return None

    if min_on_recall is not None:
        max_recall = max(float(r["recall_at_10"]) for r in coalesce_on)
        if max_recall < min_on_recall:
            errors.append(
                f"{csv_path.name} max recall_at_10 for coalesce_enabled=True is {max_recall:.4f} (< {min_on_recall:.2f})"
            )

    for idx, r in enumerate(rows, start=2):
        raw_p = float(r["avg_raw_pages"])
        gets = float(r["avg_s3_gets"])
        factor = float(r["coalesce_factor"])
        expected = raw_p / max(gets, 1e-9)
        if abs(factor - expected) / max(expected, 1.0) > 0.02:
            errors.append(
                f"{csv_path.name} line {idx}: coalesce_factor={factor:.3f} != avg_raw_pages/avg_s3_gets={expected:.3f}"
            )
        if r["coalesce_enabled"].lower() == "true":
            if factor < 1.0 or gets > raw_p + 1e-6:
                errors.append(
                    f"{csv_path.name} line {idx}: coalesce_enabled=True has factor={factor:.3f} (< 1.0)"
                )
        else:
            if abs(factor - 1.0) > 0.02:
                errors.append(
                    f"{csv_path.name} line {idx}: coalesce_enabled=False has factor={factor:.3f} (!= 1.0)"
                )

    def pair_key(r: Dict[str, str]) -> Tuple[object, ...]:
        base: List[object] = [r.get("workers", "1"), int(r["nprobe"]), int(r["refine_m"])]
        if group_by_policy:
            base.insert(0, r["refine_policy"])
        return tuple(base)

    off_map = {pair_key(r): r for r in coalesce_off}
    paired_checked = 0
    for r in coalesce_on:
        key = pair_key(r)
        if key in off_map:
            paired_checked += 1
            on_gets = float(r["avg_s3_gets"])
            off_gets = float(off_map[key]["avg_s3_gets"])
            if on_gets >= off_gets:
                errors.append(
                    f"{csv_path.name} {key}: coalesce ON avg_s3_gets ({on_gets:.2f}) >= OFF ({off_gets:.2f})"
                )
    if paired_checked == 0:
        errors.append(f"{csv_path.name}: no matching pairs between coalesce ON and OFF")

    return rows


def verify_pareto_s3(errors: List[str]) -> None:
    required_cols = {
        "nprobe",
        "refine_m",
        "coalesce_enabled",
        "recall_at_10",
        "p50_ms",
        "avg_raw_pages",
        "avg_s3_gets",
        "coalesce_factor",
        "dataset",
    }
    rows = _verify_pareto_rows(
        PARETO_CSV,
        required_cols,
        group_by_policy=False,
        min_on_recall=0.90,
        errors=errors,
    )
    if not rows:
        return

    # Verify README.md S3 section reflects pareto_s3.csv
    readme_text = README_MD.read_text(encoding="utf-8")
    ds = rows[0]["dataset"]
    if ds == "cohere768d" and "Cohere 768d" not in readme_text:
        errors.append("README.md S3 section does not mention Cohere 768d")
    if f"{len(rows)}-point" not in readme_text:
        errors.append(f"README.md S3 section does not mention {len(rows)}-point sweep")


def verify_pareto_adaptive(errors: List[str]) -> None:
    required_cols = {
        "nprobe",
        "refine_m",
        "coalesce_enabled",
        "queries_evaluated",
        "recall_at_10",
        "p50_ms",
        "avg_raw_pages",
        "avg_s3_gets",
        "coalesce_factor",
        "dataset",
        "refine_policy",
        "requested_refine_multiplier",
    }
    rows = _verify_pareto_rows(
        PARETO_ADAPTIVE_CSV,
        required_cols,
        group_by_policy=True,
        min_on_recall=None,
        errors=errors,
    )
    if not rows:
        return

    policies = {r["refine_policy"] for r in rows}
    if not {"baseline", "adaptive"}.issubset(policies):
        errors.append(
            f"{PARETO_ADAPTIVE_CSV.name} missing expected policies (got {sorted(policies)})"
        )


def verify_perf_logs(warnings: List[str]) -> None:
    if not PERF_DIR.exists():
        return
    for path in sorted(PERF_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "/home/" in line:
                warnings.append(
                    f"{path.relative_to(ROOT)}:{lineno}: absolute /home/ path leaked in perf log"
                )


def verify_figures(errors: List[str]) -> None:
    for entry in EXPECTED_FIGURES:
        candidates: Sequence[str] = (entry,) if isinstance(entry, str) else entry
        if not any((FIG_DIR / name).exists() and (FIG_DIR / name).stat().st_size > 0 for name in candidates):
            joined = " or ".join(str(FIG_DIR / name) for name in candidates)
            errors.append(f"Missing or empty figure: {joined}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OspreyDB benchmark artifacts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote sanity warnings (e.g. cgroup/latency/path-leak checks) to fatal errors.",
    )
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    verify_controlled_csv_and_docs(errors, warnings)
    verify_lutrank_csv_and_readme(errors)
    verify_pareto_s3(errors)
    verify_pareto_adaptive(errors)
    verify_perf_logs(warnings)
    verify_figures(errors)

    if args.strict and warnings:
        errors.extend(warnings)
        warnings = []

    for warn in warnings:
        print(f"[verify_results] WARN: {warn}")

    if errors:
        print("[verify_results] FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        "[verify_results] OK: CSVs, JSONs, Markdown tables, pareto_s3.csv, "
        "pareto_s3_adaptive.csv, lutrank_vs_colbert.csv, and 7 SVGs are consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
