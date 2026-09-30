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
OUT_DIR = ROOT / "results" / "pathway_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["OTF", "sciplex", "tahoe"]
DATASET_LABELS = {
    "OTF": "PerturBase-ORF",
    "sciplex": "PerturBase-Chemical",
    "tahoe": "Tahoe-100M",
}

HVG_SUMMARY = (
    ROOT
    / "results"
    / "036.kegg_multitop_per_comparison_pathway"
    / "kegg_multitop_per_comparison_summary.csv"
)
L1000_SUMMARY = (
    ROOT
    / "results"
    / "037.l1000_kegg_per_comparison_pathway"
    / "l1000_kegg_per_comparison_summary.csv"
)

COLUMNS = [
    ("L1000", "all", None, "L1000_all"),
    ("L1000", "in DEG", None, "L1000_in_DEG"),
    ("HVG top2000", "all", 2000, "HVG_all"),
    ("HVG top2000", "in DEG", 2000, "HVG_in_DEG"),
    ("HVG top3000", "all", 3000, "HVG_all"),
    ("HVG top3000", "in DEG", 3000, "HVG_in_DEG"),
    ("HVG top5000", "all", 5000, "HVG_all"),
    ("HVG top5000", "in DEG", 5000, "HVG_in_DEG"),
]


def pick_row(df: pd.DataFrame, dataset: str, subset: str, topn: int | None) -> pd.Series | None:
    mask = (df["dataset"] == dataset) & (df["comparison_subset"] == subset)
    if topn is not None:
        mask &= df["hvg_top_n"] == topn
    hit = df.loc[mask]
    if hit.empty:
        return None
    if len(hit) != 1:
        raise ValueError(f"Expected one row for {dataset=} {subset=} {topn=}, found {len(hit)}")
    return hit.iloc[0]


def main() -> None:
    hvg = pd.read_csv(HVG_SUMMARY)
    l1000 = pd.read_csv(L1000_SUMMARY)

    mat = np.full((len(DATASETS), len(COLUMNS)), np.nan)
    q25 = np.full_like(mat, np.nan)
    q75 = np.full_like(mat, np.nan)
    counts = np.full_like(mat, np.nan)
    rows = []

    for i, dataset in enumerate(DATASETS):
        for j, (group, sublabel, topn, subset) in enumerate(COLUMNS):
            source = l1000 if group == "L1000" else hvg
            row = pick_row(source, dataset, subset, topn)
            if row is None:
                median = lo = hi = count = np.nan
            else:
                median = float(row["top20_overlap_fraction_median"])
                lo = float(row["top20_overlap_fraction_q25"])
                hi = float(row["top20_overlap_fraction_q75"])
                count = float(row["top20_overlap_fraction_count"])

            mat[i, j] = median
            q25[i, j] = lo
            q75[i, j] = hi
            counts[i, j] = count
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "gene_space_group": group,
                    "gene_space_subset": sublabel,
                    "hvg_top_n": topn,
                    "source_subset": subset,
                    "metric": "KEGG top20 pathway overlap fraction",
                    "unit": "per perturbation DEG comparison",
                    "n_comparisons": count,
                    "median_fraction": median,
                    "q25_fraction": lo,
                    "q75_fraction": hi,
                    "median_percent": median * 100 if np.isfinite(median) else np.nan,
                    "q25_percent": lo * 100 if np.isfinite(lo) else np.nan,
                    "q75_percent": hi * 100 if np.isfinite(hi) else np.nan,
                }
            )

    values = pd.DataFrame(rows)
    values_path = OUT_DIR / (
        "combined_l1000_hvg_kegg_top20_pathway_overlap_"
        "per_perturbation_deg_iqr_heatmap_values.csv"
    )
    values.to_csv(values_path, index=False)
    values.to_csv(
        OUT_DIR / "combined_l1000_hvg_kegg_top20_pathway_overlap_heatmap_values.csv",
        index=False,
    )

    gap_width = 0.125
    plot_widths = []
    plot_values = []
    source_to_plot_center = []
    current_x = 0.0
    for j in range(len(COLUMNS)):
        plot_widths.append(1.0)
        plot_values.append(mat[:, j])
        source_to_plot_center.append(current_x + 0.5)
        current_x += 1.0
        if j in {1, 3, 5}:
            plot_widths.append(gap_width)
            plot_values.append(np.full(len(DATASETS), np.nan))
            current_x += gap_width

    plot_mat = np.column_stack(plot_values)
    mask = np.ma.masked_invalid(plot_mat)
    x_edges = np.concatenate([[0.0], np.cumsum(plot_widths)])
    y_edges = np.arange(len(DATASETS) + 1, dtype=float)

    fig_width = 3.0
    fig_height = 1.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        mask,
        cmap=cmap,
        vmin=0,
        vmax=1.0,
        shading="flat",
        edgecolors="none",
        linewidth=0,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks(source_to_plot_center)
    ax.set_xticklabels([c[1] for c in COLUMNS], rotation=45, ha="right", va="top", fontsize=5)
    ax.set_yticks(np.arange(len(DATASETS)) + 0.5)
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=5)
    ax.tick_params(axis="both", length=0, pad=1)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                txt = f"{mat[i, j] * 100:.0f}%\n({q25[i, j] * 100:.0f}-{q75[i, j] * 100:.0f})"
            else:
                txt = "NA"
            ax.text(
                source_to_plot_center[j],
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=4.3,
                color="#111111",
                linespacing=1.15,
            )

    for j, x_center in enumerate(source_to_plot_center):
        x0 = x_center - 0.5
        for i in range(len(DATASETS)):
            rect = plt.Rectangle((x0, i), 1.0, 1.0, fill=False, edgecolor="white", linewidth=0.35)
            ax.add_patch(rect)

    group_centers = [
        float(np.mean(source_to_plot_center[0:2])),
        float(np.mean(source_to_plot_center[2:4])),
        float(np.mean(source_to_plot_center[4:6])),
        float(np.mean(source_to_plot_center[6:8])),
    ]
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(group_centers)
    secax.set_xticklabels(["L1000", "HVG top2000", "HVG top3000", "HVG top5000"], fontsize=6)
    secax.tick_params(length=0, pad=1)

    ax.set_title("Per-perturbation DEG KEGG top20 pathway overlap, median (IQR)", fontsize=6, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in secax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.00])
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(labelsize=4, width=0.25, length=1.0, pad=1)
    cbar.outline.set_linewidth(0.3)

    base_new = OUT_DIR / (
        "combined_l1000_hvg_kegg_top20_pathway_overlap_"
        "per_perturbation_deg_iqr_heatmap"
    )
    base_legacy = OUT_DIR / "combined_l1000_hvg_kegg_top20_pathway_overlap_heatmap"
    for base in [base_new, base_legacy]:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
        print(f"saved {base.with_suffix('.pdf')}")
        print(f"saved {base.with_suffix('.png')}")
    print(f"saved {values_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
