#!/usr/bin/env python3
"""
Task 179: Publication-grade Academic Vector SVG Plots from Benchmark CSVs.

Generates:
  1. fig1_memory_disk_controlled.svg        : Peak memory and disk size from the controlled run
  2. fig2_recall_latency_controlled.svg     : Test Recall@10 vs client P50 latency
  3. fig3_build_time_controlled.svg         : Index build time per engine
  4. fig4_validation_test_recall_controlled.svg : Validation vs test recall at each target
  5. fig5_coalesced_s3_requests_16x.svg     : S3 Range-GET reduction with DP range coalescing vs uncoalesced GETs
  6. fig6_lutrank_cpu_vs_gpu.svg            : LUT-Rank CPU (~3.8 ms) vs ColBERTv2 CPU & GPU latency & NDCG@10
  7. fig7_pareto_recall_latency_s3.svg      : Pareto curve Recall@10 vs Latency (ms) vs S3 Requests
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CONTROLLED_CSV = RESULTS_DIR / "compare_engines_controlled.csv"

ENGINE_ORDER = [
    "OspreyDB",
    "Qdrant FP32",
    "Qdrant SQ8",
    "Qdrant BQ",
    "FAISS IVF",
    "pgvector FP32",
    "pgvector halfvec",
]
ENGINE_COLORS = {
    "OspreyDB": "#0F766E",
    "Qdrant FP32": "#2563EB",
    "Qdrant SQ8": "#0284C7",
    "Qdrant BQ": "#7C3AED",
    "FAISS IVF": "#64748B",
    "pgvector FP32": "#D97706",
    "pgvector halfvec": "#EA580C",
}
ENGINE_MARKERS = {
    "OspreyDB": "o",
    "Qdrant FP32": "s",
    "Qdrant SQ8": "P",
    "Qdrant BQ": "^",
    "FAISS IVF": "v",
    "pgvector FP32": "D",
    "pgvector halfvec": "X",
}
DATASET_LABELS = {
    "sift1m": "SIFT1M",
    "cohere768d": "Cohere 768d (100k)",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "svg.fonttype": "path",
    }
)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def controlled_engine_label(engine: str) -> str:
    if engine.startswith("OspreyDB"):
        return "OspreyDB"
    if engine.startswith("Qdrant (HNSW FP32)"):
        return "Qdrant FP32"
    if engine.startswith("Qdrant (HNSW int8") or "SQ8" in engine:
        return "Qdrant SQ8"
    if engine.startswith("Qdrant (HNSW 1-bit"):
        return "Qdrant BQ"
    if "faiss" in engine.lower():
        return "FAISS IVF"
    if "halfvec" in engine:
        return "pgvector halfvec"
    if engine.startswith("pgvector"):
        return "pgvector FP32"
    return engine


def controlled_rows_by_dataset(rows: List[Dict[str, str]], dataset: str) -> Dict[str, Dict[str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row["dataset"] != dataset:
            continue
        label = controlled_engine_label(row["engine"])
        grouped.setdefault(label, row)
    return grouped


def controlled_datasets(rows: List[Dict[str, str]]) -> List[str]:
    present = set(r["dataset"] for r in rows)
    order = [ds for ds in ("sift1m", "cohere768d") if ds in present]
    return order or sorted(present)


def plot_fig1_controlled_memory() -> None:
    rows = read_csv_rows(CONTROLLED_CSV)
    if not rows:
        return

    datasets = controlled_datasets(rows)
    max_engines = max(
        (len(controlled_rows_by_dataset(rows, ds)) for ds in datasets), default=4
    )
    fig_w = max(6.8, 1.2 * max_engines) * max(1, len(datasets))
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(fig_w, 4.9), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        grouped = controlled_rows_by_dataset(rows, ds)
        engines = [e for e in ENGINE_ORDER if e in grouped]
        mem_vals = [float(grouped[e]["query_peak_rss_mb"]) for e in engines]
        disk_vals = [float(grouped[e]["disk_size_mb"]) for e in engines]
        colors = [ENGINE_COLORS[e] for e in engines]

        x = np.arange(len(engines))
        width = 0.36
        mem_bars = ax.bar(
            x - width / 2,
            mem_vals,
            width,
            color=colors,
            edgecolor="#0F172A",
            linewidth=0.7,
            label="Peak memory (MiB)",
        )
        disk_bars = ax.bar(
            x + width / 2,
            disk_vals,
            width,
            color="#CBD5E1",
            edgecolor="#0F172A",
            linewidth=0.7,
            hatch="//",
            label="Disk (MiB)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(engines, fontsize=8.5, rotation=20, ha="right")
        ax.set_ylabel("MiB (log scale)")
        ax.set_yscale("log")
        all_vals = mem_vals + disk_vals
        if all_vals:
            ax.set_ylim(max(1.0, min(all_vals) * 0.5), max(all_vals) * 2.4)
        ax.set_title(f"{DATASET_LABELS.get(ds, ds.upper())} — Peak memory and disk")
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

        for bar, value in zip(mem_bars, mem_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.10,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )
        for bar, value in zip(disk_bars, disk_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.10,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    fig.tight_layout()
    out = FIG_DIR / "fig1_memory_disk_controlled.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig2_controlled_recall_latency() -> None:
    rows = read_csv_rows(CONTROLLED_CSV)
    if not rows:
        return

    datasets = controlled_datasets(rows)
    fig, axes = plt.subplots(
        1, max(1, len(datasets)), figsize=(6.4 * max(1, len(datasets)), 4.6), squeeze=False
    )

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        ds_rows = [r for r in rows if r["dataset"] == ds]
        for engine in ENGINE_ORDER:
            engine_rows = sorted(
                [r for r in ds_rows if controlled_engine_label(r["engine"]) == engine],
                key=lambda r: float(r["recall_target"]),
            )
            if not engine_rows:
                continue
            xs = [float(r["p50_ms_median"]) for r in engine_rows]
            ys = [float(r["test_recall"]) * 100.0 for r in engine_rows]
            ax.errorbar(
                xs,
                ys,
                xerr=[float(r["p50_ms_std"]) for r in engine_rows],
                fmt=ENGINE_MARKERS[engine],
                color=ENGINE_COLORS[engine],
                markersize=8,
                capsize=3,
                linestyle="",
                label=engine,
                markeredgecolor="#0F172A",
                markeredgewidth=0.6,
            )
            if engine == "Qdrant BQ" and min(ys) < 50.0:
                ax.annotate(
                    "BQ below target",
                    (xs[0], ys[0]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    color=ENGINE_COLORS[engine],
                    fontweight="bold",
                )

        ax.axhline(95.0, color="#DC2626", linestyle=":", linewidth=1.1, label="target 0.95")
        ax.axhline(97.0, color="#B91C1C", linestyle="--", linewidth=1.1, label="target 0.97")
        ax.set_xlabel("P50 client latency, sequential (ms)")
        ax.set_ylabel("Test Recall@10 (%)")
        ax.set_ylim(0, 102)
        ax.set_title(f"{DATASET_LABELS.get(ds, ds.upper())} — Recall vs latency")
        ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "fig2_recall_latency_controlled.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig3_controlled_build_time() -> None:
    rows = read_csv_rows(CONTROLLED_CSV)
    if not rows:
        return

    datasets = controlled_datasets(rows)
    engines_all = sorted(
        set(controlled_engine_label(r["engine"]) for r in rows),
        key=lambda e: ENGINE_ORDER.index(e) if e in ENGINE_ORDER else len(ENGINE_ORDER),
    )
    fig, ax = plt.subplots(figsize=(max(7.8, 1.35 * len(engines_all)), 4.9))
    width = 0.78 / max(1, len(datasets))
    all_values: List[float] = []

    for di, ds in enumerate(datasets):
        grouped = controlled_rows_by_dataset(rows, ds)
        engines = [e for e in engines_all if e in grouped]
        values = [float(grouped[e]["build_time_s"]) for e in engines]
        all_values.extend(values)
        offset = (di - (len(datasets) - 1) / 2.0) * width
        x = np.array([engines_all.index(e) for e in engines], dtype=float) + offset
        bars = ax.bar(
            x,
            values,
            width=width * 0.88,
            color=[ENGINE_COLORS.get(e, "#64748B") for e in engines],
            edgecolor="#0F172A",
            linewidth=0.7,
            alpha=0.75 if di else 1.0,
            label=DATASET_LABELS.get(ds, ds.upper()),
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(all_values) * 0.015,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=35 if len(engines_all) >= 5 else 0,
            )

    ax.set_xticks(np.arange(len(engines_all)))
    ax.set_xticklabels(engines_all, fontsize=8.5, rotation=16, ha="right")
    ax.set_ylabel("Index build time (s)")
    ax.set_title("Index build time — controlled run")
    ax.set_ylim(0, max(all_values) * 1.24 if all_values else 1)
    dataset_handles = [
        Patch(facecolor="#475569", alpha=1.0 if i == 0 else 0.55, label=DATASET_LABELS.get(ds, ds.upper()))
        for i, ds in enumerate(datasets)
    ]
    ax.legend(handles=dataset_handles, frameon=True, fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "fig3_build_time_controlled.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig4_controlled_recall_targets() -> None:
    rows = read_csv_rows(CONTROLLED_CSV)
    if not rows:
        return

    datasets = controlled_datasets(rows)
    max_records = max(
        (len([r for r in rows if r["dataset"] == ds]) for ds in datasets), default=8
    )
    fig, axes = plt.subplots(
        1,
        max(1, len(datasets)),
        figsize=(max(7.2, 0.78 * max_records) * max(1, len(datasets)), 5.3),
        squeeze=False,
    )

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        ds_rows = [r for r in rows if r["dataset"] == ds]
        records = []
        labels = []
        for engine in ENGINE_ORDER:
            for row in sorted(
                [r for r in ds_rows if controlled_engine_label(r["engine"]) == engine],
                key=lambda r: float(r["recall_target"]),
            ):
                records.append(row)
                labels.append(f"{engine} ({float(row['recall_target']):.2f})")

        x = np.arange(len(records))
        width = 0.36
        val_vals = [float(r["validation_recall"]) for r in records]
        test_vals = [float(r["test_recall"]) for r in records]
        val_bars = ax.bar(
            x - width / 2,
            val_vals,
            width,
            color="#94A3B8",
            edgecolor="#0F172A",
            linewidth=0.6,
            label="Validation recall",
        )
        test_bars = ax.bar(
            x + width / 2,
            test_vals,
            width,
            color="#0F766E",
            edgecolor="#0F172A",
            linewidth=0.6,
            label="Test recall",
        )

        for pos, row in zip(x, records):
            target = float(row["recall_target"])
            ax.hlines(target, pos - 0.42, pos + 0.42, colors="#DC2626", linestyles=":", linewidth=1.6, zorder=5)
        for bar, value in zip(val_bars, val_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                rotation=90,
            )
        for bar, value in zip(test_bars, test_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                rotation=90,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=32, ha="right")
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1.14)
        ax.set_title(f"{DATASET_LABELS.get(ds, ds.upper())} — Validation vs test recall")
        ax.legend(
            frameon=True,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
        )

    fig.tight_layout()
    out = FIG_DIR / "fig4_validation_test_recall_controlled.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig5_coalesced_s3() -> None:
    rows = read_csv_rows(RESULTS_DIR / "pareto_s3.csv")
    if not rows:
        return

    coalesce_on = [r for r in rows if r["coalesce_enabled"].lower() == "true"]
    coalesce_off = [r for r in rows if r["coalesce_enabled"].lower() == "false"]
    if not coalesce_on or not coalesce_off:
        return

    ds_name = rows[0].get("dataset", "cohere768d")
    ds_label = DATASET_LABELS.get(ds_name, ds_name.upper())
    rtt_ms = float(rows[0].get("minio_netem_rtt_ms") or 15.0)
    max_sweep_factor = max(
        float(r.get("coalesce_factor") or (float(r["avg_raw_pages"]) / max(float(r["avg_s3_gets"]), 1e-6)))
        for r in coalesce_on
    )

    # Match (nprobe, refine_m) pairs present in both ON and OFF
    off_map = {(int(r["nprobe"]), int(r["refine_m"])): r for r in coalesce_off}
    pairs = []
    for r in coalesce_on:
        key = (int(r["nprobe"]), int(r["refine_m"]))
        if key in off_map:
            pairs.append((key, r, off_map[key]))
    pairs.sort(key=lambda x: x[0])

    labels = [f"np={k[0]}\nm={k[1]}" for k, _, _ in pairs]
    gets_on = [float(on["avg_s3_gets"]) for _, on, _ in pairs]
    gets_off = [float(off["avg_s3_gets"]) for _, _, off in pairs]
    lat_on = [float(on["p50_ms"]) for _, on, _ in pairs]
    lat_off = [float(off["p50_ms"]) for _, _, off in pairs]
    max_pair_factor = max(
        (off_v / max(on_v, 1e-6) for off_v, on_v in zip(gets_off, gets_on)),
        default=max_sweep_factor,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
    x = np.arange(len(labels))
    w = 0.36

    ax1.bar(x - w / 2, gets_off, width=w, label="Uncoalesced S3 GETs (Coalesce OFF)", color="#94A3B8", edgecolor="#0F172A", lw=0.6)
    ax1.bar(x + w / 2, gets_on, width=w, label="Coalesced S3 get_ranges (DP Coalescing)", color="#0F766E", edgecolor="#0F172A", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Mean S3 Range-GETs / Query")
    ax1.set_ylim(0, max(gets_off + gets_on) * 1.18 if (gets_off or gets_on) else 1.0)
    ax1.set_title(
        f"S3 Request Coalescing ({ds_label}: up to {max_pair_factor:.1f}x paired, {max_sweep_factor:.1f}x sweep max)"
    )
    ax1.legend(frameon=True, fontsize=8.5, loc="upper left")

    for i, (off_v, on_v) in enumerate(zip(gets_off, gets_on)):
        factor = off_v / max(on_v, 1e-6)
        ax1.text(x[i] + w / 2, on_v + max(gets_off) * 0.02, f"{factor:.1f}x", ha="center", va="bottom", fontsize=8, color="#0F766E", fontweight="bold")

    ax2.bar(x - w / 2, lat_off, width=w, label=f"Coalesce OFF (MinIO tc netem {rtt_ms:.1f}ms)", color="#CBD5E1", edgecolor="#0F172A", lw=0.6)
    ax2.bar(x + w / 2, lat_on, width=w, label=f"DP Coalescing (MinIO tc netem {rtt_ms:.1f}ms)", color="#2563EB", edgecolor="#0F172A", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("P50 End-to-End S3 Latency (ms)")
    ax2.set_ylim(0, max(lat_off + lat_on) * 1.18 if (lat_off or lat_on) else 1.0)
    ax2.set_title(f"S3 P50 Latency Under tc netem ({rtt_ms:.1f}ms RTT, {ds_label})")
    ax2.legend(frameon=True, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    out = FIG_DIR / "fig5_coalesced_s3_requests_16x.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig6_lutrank() -> None:
    rows = read_csv_rows(RESULTS_DIR / "lutrank_vs_colbert.csv")
    if not rows:
        return

    datasets = sorted(set(r["dataset"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    engines = []
    for r in rows:
        if r["engine"] not in engines:
            engines.append(r["engine"])

    x = np.arange(len(datasets))
    width = 0.72 / max(1, len(engines))
    colors = ["#0F766E", "#64748B", "#0284C7", "#2563EB", "#DC2626", "#D97706"]

    for ei, eng in enumerate(engines):
        p50s = []
        ndcgs = []
        for ds in datasets:
            match = [r for r in rows if r["dataset"] == ds and r["engine"] == eng]
            p50s.append(float(match[0]["p50_ms"]) if match else 0.0)
            ndcgs.append(float(match[0]["ndcg_at_10"]) if match else 0.0)
        offset = (ei - (len(engines) - 1) / 2.0) * width
        b1 = ax1.bar(x + offset, p50s, width=width * 0.9, label=eng, color=colors[ei % len(colors)], edgecolor="#0F172A", lw=0.6)
        ax2.bar(x + offset, ndcgs, width=width * 0.9, label=eng, color=colors[ei % len(colors)], edgecolor="#0F172A", lw=0.6)
        for bar, val in zip(b1, p50s):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2.0, val * 1.03, f"{val:.2f}ms", ha="center", va="bottom", fontsize=7.5)

    ax1.axhline(3.8, color="#DC2626", linestyle=":", linewidth=1.2, label="LUT-Rank 3.8 ms CPU Budget")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in datasets])
    ax1.set_ylabel("P50 Rerank Latency / Query (ms)")
    ax1.set_title("LUT-Rank CPU vs ColBERTv2 Latency (200 Docs)")
    ax1.legend(frameon=True, fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.set_ylabel("NDCG@10")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Ranking Relevance Preservation (NDCG@10)")
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "fig6_lutrank_cpu_vs_gpu.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fig7_pareto() -> None:
    rows = read_csv_rows(RESULTS_DIR / "pareto_s3.csv")
    coalesce_on = [r for r in rows if r["coalesce_enabled"].lower() == "true"]
    if not coalesce_on:
        return

    ds_name = rows[0].get("dataset", "cohere768d")
    ds_label = DATASET_LABELS.get(ds_name, ds_name.upper())

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    nprobes = sorted(set(int(r["nprobe"]) for r in coalesce_on))
    cmap = plt.get_cmap("viridis")

    all_pts = []
    for i, np_val in enumerate(nprobes):
        pts = sorted(
            [r for r in coalesce_on if int(r["nprobe"]) == np_val],
            key=lambda r: int(r["refine_m"]),
        )
        lats = [float(r["p50_ms"]) for r in pts]
        recs = [float(r["recall_at_10"]) * 100.0 for r in pts]
        gets = [float(r["avg_s3_gets"]) for r in pts]
        color = cmap(i / max(1, len(nprobes) - 1))
        ax.plot(
            lats,
            recs,
            marker="o",
            markersize=4.5,
            lw=1.25,
            alpha=0.55,
            color=color,
            label=f"nprobe={np_val}",
            zorder=2,
        )
        for x_v, y_v, g_v, r in zip(lats, recs, gets, pts):
            all_pts.append((np_val, int(r["refine_m"]), x_v, y_v, g_v, color, r))

    # Compute Pareto-optimal frontier (minimize P50 latency, maximize Recall@10)
    pareto = []
    for p in all_pts:
        dominated = any(
            q[2] <= p[2] and q[3] >= p[3] and (q[2] < p[2] or q[3] > p[3])
            for q in all_pts
        )
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda p: p[2])

    ax.plot(
        [p[2] for p in pareto],
        [p[3] for p in pareto],
        color="#0F172A",
        lw=2.0,
        linestyle="--",
        label="Pareto frontier",
        zorder=4,
    )
    for np_val, m_val, x_v, y_v, g_v, color, _ in pareto:
        ax.scatter(
            [x_v],
            [y_v],
            s=64,
            color=color,
            edgecolors="#0F172A",
            linewidths=1.3,
            zorder=5,
        )

    # Dynamic axis limits covering 100% of coalesce_enabled=True points with margin
    all_lats = [p[2] for p in all_pts]
    all_recs = [p[3] for p in all_pts]
    x_min, x_max = min(all_lats), max(all_lats)
    y_min, y_max = min(all_recs), max(all_recs)
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)
    ax.set_xlim(x_min - x_span * 0.12, x_max + x_span * 0.16)
    ax.set_ylim(max(0.0, y_min - y_span * 0.18), min(101.0, y_max + y_span * 0.12))

    # Automatically place non-overlapping labels along the Pareto frontier
    placed_xy: List[tuple] = []
    candidate_offsets = [
        (-10, 14),
        (12, -16),
        (-74, 14),
        (14, 10),
        (-74, -18),
        (14, -24),
    ]
    for idx_p, (np_val, m_val, x_v, y_v, g_v, _, _row) in enumerate(pareto):
        # Skip labeling if another labeled Pareto point sits virtually on top
        too_close = any(
            abs(x_v - px) / x_span < 0.035 and abs(y_v - py) / y_span < 0.035
            for px, py in placed_xy
        )
        if too_close:
            continue
        placed_xy.append((x_v, y_v))
        # Prefer top-left near upper frontier and right-below near lower frontier
        dx, dy = candidate_offsets[idx_p % len(candidate_offsets)]
        if x_v > x_min + 0.72 * x_span:
            dx = -abs(dx) - 52 if dx > 0 else dx
        elif x_v < x_min + 0.18 * x_span:
            dx = abs(dx)
        if y_v > y_min + 0.82 * y_span and dy > 0:
            dy = -abs(dy) - 4
        ax.annotate(
            f"{np_val}/{m_val} ({g_v:.1f} GETs)",
            (x_v, y_v),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.6,
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.92, lw=0.7),
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.8),
            zorder=6,
        )

    # Dynamic highlight callout from the highest-recall point (>= 0.95 if available, else overall max)
    high_recall = [r for r in coalesce_on if float(r["recall_at_10"]) >= 0.95]
    pool = high_recall if high_recall else coalesce_on
    best_pt = max(pool, key=lambda r: (float(r["recall_at_10"]), -float(r["p50_ms"])))
    raw_pages = float(best_pt["avg_raw_pages"])
    s3_gets = float(best_pt["avg_s3_gets"])
    factor = float(best_pt.get("coalesce_factor") or (raw_pages / max(s3_gets, 1e-6)))
    best_rec_pct = float(best_pt["recall_at_10"]) * 100.0
    best_lat = float(best_pt["p50_ms"])
    ax.text(
        0.02,
        0.96,
        (
            f"Best Recall@10 ({best_pt['nprobe']}/{best_pt['refine_m']}): "
            f"{raw_pages:.1f} pages -> {s3_gets:.1f} GETs ({factor:.1f}x coalescing)\n"
            f"Recall@10 = {best_rec_pct:.1f}% @ {best_lat:.1f} ms"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#94A3B8", alpha=0.95, lw=0.8),
        zorder=7,
    )

    ax.set_xlabel("P50 Query Latency Over MinIO tc netem (ms)")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title(f"Pareto Frontier: Recall@10 vs Latency vs S3 GETs ({ds_label}, Coalesce ON)")
    ax.legend(frameon=True, fontsize=8.2, loc="lower right", ncol=2)

    fig.tight_layout()
    out = FIG_DIR / "fig7_pareto_recall_latency_s3.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication SVG figures from benchmark CSVs."
    )
    parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1_controlled_memory()
    plot_fig2_controlled_recall_latency()
    plot_fig3_controlled_build_time()
    plot_fig4_controlled_recall_targets()
    plot_fig5_coalesced_s3()
    plot_fig6_lutrank()
    plot_fig7_pareto()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
