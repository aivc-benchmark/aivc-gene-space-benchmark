#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = ["OTF", "sciplex", "tahoe"]
TOP_NS = [2000, 3000, 5000, 10000]
HVG_LEVELS = ["dataset", "cellline"]
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300})


def apply_fraction_axis(ax, ymax):
    ymax = min(1.0, max(0.3, float(ymax)))
    ymax = np.ceil(ymax * 10) / 10
    ticks = np.linspace(0, ymax, int(round(ymax / 0.1)) + 1)
    ax.set_ylim(0, ymax)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{x:.1f}" for x in ticks])
    ax.grid(False)


def plot_fraction_boxplot(samples, value_col, dataset_order, title, out_prefix, formats):
    samples = samples[samples["dataset"].isin(dataset_order)].copy()
    samples["dataset"] = pd.Categorical(samples["dataset"], categories=dataset_order, ordered=True)
    samples = samples.sort_values("dataset")
    means = samples.groupby("dataset", observed=True)[value_col].mean().reindex(dataset_order).to_numpy()
    values = [samples.loc[samples["dataset"].astype(str).eq(d), value_col].dropna().to_numpy() for d in dataset_order]
    color_map = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(dataset_order)}
    width = max(3.8, 0.52 * len(dataset_order) + 1.0)
    fig, ax = plt.subplots(figsize=(width, 3.7))
    x = np.arange(len(dataset_order))
    bp = ax.boxplot(values, positions=x, widths=0.38, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.1},
                    whiskerprops={"color": "black", "linewidth": 1.0},
                    capprops={"color": "black", "linewidth": 1.0}, boxprops={"linewidth": 1.1})
    for patch, dataset in zip(bp["boxes"], dataset_order):
        patch.set_facecolor(color_map[dataset]); patch.set_alpha(0.2); patch.set_edgecolor(color_map[dataset])
    rng = np.random.default_rng(42)
    highs = []
    for i, (dataset, vals) in enumerate(zip(dataset_order, values)):
        highs.append(np.nanmax(vals) if len(vals) else means[i])
        if len(vals):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=6, color=color_map[dataset],
                       edgecolors="white", linewidths=0.35, alpha=0.85, zorder=3)
    ymax_data = np.nanmax([np.nanmax(v) if len(v) else 0 for v in values] + [0.3])
    apply_fraction_axis(ax, min(1.0, ymax_data + 0.08))
    for xi, mean, high in zip(x, means, highs):
        if pd.notna(mean):
            ax.text(xi, min(ax.get_ylim()[1] - 0.005, max(high, mean) + 0.012), f"{mean:.2f}",
                    ha="center", va="bottom", fontsize=9, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=30 if len(dataset_order) > 3 else 0,
                       ha="right" if len(dataset_order) > 3 else "center")
    ax.set_ylabel("Fraction"); ax.set_xlabel("Dataset"); ax.set_title(title)
    fig.tight_layout(); out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def write_long_table(counts, out_dir):
    keep = ["dataset", "cell_id", "cell_type", "perturbation_id", "hvg_level", "hvg_top_n", "deg_n"]
    long = pd.concat([
        counts[keep + ["deg_hvg_fraction_of_deg", "deg_hvg_n"]].rename(columns={"deg_hvg_fraction_of_deg": "fraction_of_deg", "deg_hvg_n": "intersection_n"}).assign(intersection_type="HVG_DEG"),
        counts[keep + ["deg_l1000_fraction_of_deg", "deg_l1000_n"]].rename(columns={"deg_l1000_fraction_of_deg": "fraction_of_deg", "deg_l1000_n": "intersection_n"}).assign(intersection_type="L1000_DEG"),
    ], ignore_index=True)
    long.to_csv(out_dir / "intersection_fraction_long_table_new.csv", index=False)
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--counts", type=Path, default=None)
    ap.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    args = ap.parse_args()
    counts_path = args.counts or (args.root / "results" / "004.hvg_l1000_deg_venn" / "all_intersection_counts.csv")
    out_dir = args.root / "results" / "005.intersection_boxplots_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(counts_path)
    counts["hvg_top_n"] = counts["hvg_top_n"].astype(int)
    counts = counts[counts["hvg_top_n"].isin(TOP_NS) & counts["hvg_level"].isin(HVG_LEVELS)].copy()
    if counts.empty:
        raise ValueError(f"No usable rows in {counts_path}")
    long = write_long_table(counts, out_dir)
    rows = []
    for hvg_level in HVG_LEVELS:
        for top_n in TOP_NS:
            sub = counts[counts["hvg_level"].eq(hvg_level) & counts["hvg_top_n"].eq(top_n)].copy()
            if sub.empty: continue
            datasets = [d for d in DATASET_ORDER if d in set(sub["dataset"])]
            level = f"top{top_n}"
            sub_dir = out_dir / hvg_level
            plot_fraction_boxplot(sub, "deg_hvg_fraction_of_deg", datasets, f"HVG-DEG intersection fraction ({hvg_level}, {level})", sub_dir / f"{level}_hvg_deg_intersection_in_deg_fraction_boxplot_new", args.formats)
            plot_fraction_boxplot(sub, "deg_l1000_fraction_of_deg", datasets, f"L1000-DEG intersection fraction ({hvg_level}, {level})", sub_dir / f"{level}_l1000_deg_intersection_in_deg_fraction_boxplot_new", args.formats)
            if hvg_level == "dataset":
                plot_fraction_boxplot(sub, "deg_hvg_fraction_of_deg", datasets, f"HVG-DEG intersection fraction ({level})", out_dir / "all" / f"{level}_intersection_in_deg_fraction_boxplot_new", args.formats)
            rows.append({"hvg_level": hvg_level, "hvg_top_n": top_n, "n_rows": int(len(sub)), "datasets": ",".join(datasets), "mean_hvg_deg_fraction": float(sub["deg_hvg_fraction_of_deg"].mean()), "mean_l1000_deg_fraction": float(sub["deg_l1000_fraction_of_deg"].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "boxplot_summary_new.csv", index=False)
    print(summary.to_string(index=False))
    long_path = out_dir / "intersection_fraction_long_table_new.csv"
    print(f"Long table: {long_path} ({len(long)} rows)")

if __name__ == "__main__":
    main()
