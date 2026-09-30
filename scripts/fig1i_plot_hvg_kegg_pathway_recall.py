#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 6

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "036.kegg_multitop_per_comparison_pathway"
OUT_DIR = IN_DIR / "plots" / "KEGG_top_axis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = IN_DIR / "kegg_multitop_per_comparison_summary.csv"

TOP_NS = [2000, 3000, 5000]
DATASETS = ["OTF", "sciplex", "tahoe"]
DATASET_LABELS = {
    "OTF": "PerturBase-ORF",
    "sciplex": "PerturBase-Chemical",
    "tahoe": "Tahoe-100M",
}
SUBSETS = ["HVG_all", "HVG_in_DEG", "DEG_not_in_HVG"]
SUBSET_LABELS = {
    "HVG_all": "Complete HVG",
    "HVG_in_DEG": "DEGs in HVG",
    "DEG_not_in_HVG": "DEGs outside HVG",
}
COLORS = {"HVG_all": "#4C78A8", "HVG_in_DEG": "#2F855A", "DEG_not_in_HVG": "#E45756"}
MARKERS = {"HVG_all": "o", "HVG_in_DEG": "s", "DEG_not_in_HVG": "^"}

df = pd.read_csv(SUMMARY)
df = df[df["hvg_top_n"].isin(TOP_NS)].copy()
df.to_csv(OUT_DIR / "kegg_multitop_per_comparison_summary_iqr_no_top10000_values.csv", index=False)


def save(fig, stem: str) -> None:
    base = OUT_DIR / stem
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def line_iqr(metric: str, ylabel: str, stem: str, percent: bool = True, ylim=(0, 1.0)) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(2.8, 1.1), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for subset in SUBSETS:
            d = df[(df.dataset == ds) & (df.comparison_subset == subset)].sort_values("hvg_top_n")
            if d.empty:
                continue
            x = d["hvg_top_n"].to_numpy(dtype=float)
            y = d[f"{metric}_median"].to_numpy(dtype=float)
            lo = d[f"{metric}_q25"].to_numpy(dtype=float)
            hi = d[f"{metric}_q75"].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color=COLORS[subset], alpha=0.16, linewidth=0)
            ax.plot(
                x,
                y,
                marker=MARKERS[subset],
                lw=0.8,
                ms=2.0,
                color=COLORS[subset],
                label=SUBSET_LABELS[subset],
            )
        ax.set_title(DATASET_LABELS[ds], fontsize=6)
        ax.set_xlabel("HVG topN", fontsize=6)
        ax.set_xticks(TOP_NS)
        ax.set_xticklabels(["2k", "3k", "5k"], fontsize=5)
        ax.tick_params(axis="both", labelsize=5, width=0.4, length=1.5)
        ax.grid(alpha=0.25, lw=0.25)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if ylim is not None:
            ax.set_ylim(*ylim)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[0].set_ylabel(ylabel, fontsize=6)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=5,
        handlelength=1.2,
        columnspacing=0.7,
    )
    fig.tight_layout(pad=0.15)
    save(fig, stem)


def heatmap_iqr(
    metric: str,
    ylabel: str,
    stem: str,
    percent: bool = True,
    vmin=0,
    vmax=1,
) -> None:
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(2.05 * len(SUBSETS), 1.55), constrained_layout=True)
    for ax, subset in zip(axes, SUBSETS):
        med = np.full((len(DATASETS), len(TOP_NS)), np.nan)
        q25 = np.full_like(med, np.nan)
        q75 = np.full_like(med, np.nan)
        for i, ds in enumerate(DATASETS):
            for j, topn in enumerate(TOP_NS):
                row = df[
                    (df.dataset == ds)
                    & (df.hvg_top_n == topn)
                    & (df.comparison_subset == subset)
                ]
                if not row.empty:
                    med[i, j] = float(row[f"{metric}_median"].iloc[0])
                    q25[i, j] = float(row[f"{metric}_q25"].iloc[0])
                    q75[i, j] = float(row[f"{metric}_q75"].iloc[0])
        im = ax.imshow(med, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_title(SUBSET_LABELS[subset], fontsize=6)
        ax.set_xticks(range(len(TOP_NS)))
        ax.set_xticklabels(["2k", "3k", "5k"], fontsize=5)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=5)
        ax.tick_params(length=0)
        for i in range(len(DATASETS)):
            for j in range(len(TOP_NS)):
                if np.isnan(med[i, j]):
                    label = "NA"
                elif percent:
                    label = f"{med[i, j] * 100:.0f}%\n({q25[i, j] * 100:.0f}-{q75[i, j] * 100:.0f})"
                else:
                    label = f"{med[i, j]:.2f}\n({q25[i, j]:.2f}-{q75[i, j]:.2f})"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    color="#111111",
                    linespacing=0.85,
                )
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
    if percent:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cbar.set_label(ylabel, fontsize=6)
    save(fig, stem)


line_iqr(
    "top10_overlap_fraction",
    "Top10 pathway recall, median (IQR) (%)",
    "kegg_top10_pathway_recall_top_axis_line_iqr_no_top10000",
    True,
    (0, 1.0),
)
heatmap_iqr(
    "top20_overlap_fraction",
    "Top20 pathway overlap, median (IQR)",
    "kegg_top20_pathway_overlap_top_axis_heatmap_iqr_no_top10000",
    True,
    0,
    1.0,
)
line_iqr(
    "sig_recall_of_deg_all",
    "Significant pathway recall, median (IQR) (%)",
    "kegg_pathway_recall_top_axis_line_iqr_no_top10000",
    True,
    (0, 1.0),
)
heatmap_iqr(
    "sig_recall_of_deg_all",
    "Significant pathway recall, median (IQR)",
    "kegg_pathway_recall_top_axis_heatmap_iqr_no_top10000",
    True,
    0,
    1.0,
)
print(f"DONE IQR no-top10000 plots: {OUT_DIR}")
