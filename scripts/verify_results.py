#!/usr/bin/env python3
"""
Verify consistency and quality invariants across benchmark CSVs, JSONs,
Markdown summaries (README.md and results/compare_engines_controlled.md),
and generated SVG figures.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CONTROLLED_CSV = RESULTS_DIR / "compare_engines_controlled.csv"
CONTROLLED_JSON = RESULTS_DIR / "compare_engines_controlled.json"
CONTROLLED_MD = RESULTS_DIR / "compare_engines_controlled.md"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
PARETO_CSV = RESULTS_DIR / "pareto_s3.csv"
README_MD = ROOT / "README.md"
FIG_DIR = RESULTS_DIR / "figures"

EXPECTED_FIGURES = [
    "fig1_memory_disk_controlled.svg",
    "fig2_recall_latency_controlled.svg",
    "fig3_build_time_controlled.svg",
    "fig4_validation_test_recall_controlled.svg",
    "fig5_coalesced_s3_requests_16x.svg",
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


def verify_controlled_csv_and_docs(errors: List[str]) -> None:
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
    for row in csv_rows:
        ds = row["dataset"]
        eng = canonical_engine(row["engine"])
        tgt = f"{float(row['recall_target']):.2f}"
        csv_by_key[(ds, eng, tgt)] = row
        csv_mem_by_ds_eng[(ds, eng)] = float(row["query_peak_rss_mb"])

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


def verify_pareto_s3(errors: List[str]) -> None:
    if not PARETO_CSV.exists():
        errors.append(f"Missing {PARETO_CSV}")
        return

    with open(PARETO_CSV, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        errors.append(f"{PARETO_CSV} is empty")
        return

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
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        errors.append(f"{PARETO_CSV.name} missing required columns: {sorted(missing_cols)}")
        return

    coalesce_on = [r for r in rows if r["coalesce_enabled"].lower() == "true"]
    coalesce_off = [r for r in rows if r["coalesce_enabled"].lower() == "false"]
    if not coalesce_on:
        errors.append(f"{PARETO_CSV.name} has no coalesce_enabled=True rows")
        return
    if not coalesce_off:
        errors.append(f"{PARETO_CSV.name} has no coalesce_enabled=False rows")
        return

    max_recall = max(float(r["recall_at_10"]) for r in coalesce_on)
    if max_recall < 0.90:
        errors.append(
            f"{PARETO_CSV.name} max recall_at_10 for coalesce_enabled=True is {max_recall:.4f} (< 0.90)"
        )

    for idx, r in enumerate(rows, start=2):
        raw_p = float(r["avg_raw_pages"])
        gets = float(r["avg_s3_gets"])
        factor = float(r["coalesce_factor"])
        expected = raw_p / max(gets, 1e-9)
        if abs(factor - expected) / max(expected, 1.0) > 0.02:
            errors.append(
                f"{PARETO_CSV.name} line {idx}: coalesce_factor={factor:.3f} != avg_raw_pages/avg_s3_gets={expected:.3f}"
            )
        if r["coalesce_enabled"].lower() == "true":
            if factor < 1.0 or gets > raw_p + 1e-6:
                errors.append(
                    f"{PARETO_CSV.name} line {idx}: coalesce_enabled=True has factor={factor:.3f} (< 1.0)"
                )
        else:
            if abs(factor - 1.0) > 0.02:
                errors.append(
                    f"{PARETO_CSV.name} line {idx}: coalesce_enabled=False has factor={factor:.3f} (!= 1.0)"
                )

    off_map = {(int(r["nprobe"]), int(r["refine_m"])): r for r in coalesce_off}
    paired_checked = 0
    for r in coalesce_on:
        key = (int(r["nprobe"]), int(r["refine_m"]))
        if key in off_map:
            paired_checked += 1
            on_gets = float(r["avg_s3_gets"])
            off_gets = float(off_map[key]["avg_s3_gets"])
            if on_gets >= off_gets:
                errors.append(
                    f"{PARETO_CSV.name} {key}: coalesce ON avg_s3_gets ({on_gets:.2f}) >= OFF ({off_gets:.2f})"
                )
    if paired_checked == 0:
        errors.append(f"{PARETO_CSV.name}: no matching (nprobe, refine_m) pairs between coalesce ON and OFF")

    # Verify README.md S3 section reflects pareto_s3.csv
    readme_text = README_MD.read_text(encoding="utf-8")
    ds = rows[0]["dataset"]
    if ds == "cohere768d" and "Cohere 768d" not in readme_text:
        errors.append("README.md S3 section does not mention Cohere 768d")
    if f"{len(rows)}-point" not in readme_text:
        errors.append(f"README.md S3 section does not mention {len(rows)}-point sweep")


def verify_figures(errors: List[str]) -> None:
    for name in EXPECTED_FIGURES:
        path = FIG_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"Missing or empty figure: {path}")


def main() -> int:
    errors: List[str] = []
    verify_controlled_csv_and_docs(errors)
    verify_pareto_s3(errors)
    verify_figures(errors)

    if errors:
        print("[verify_results] FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[verify_results] OK: CSVs, JSONs, Markdown tables, pareto_s3.csv, and 7 SVGs are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
