#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 6

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "037.l1000_kegg_per_comparison_pathway"
OUT_DIR = IN_DIR / "plots" / "KEGG_per_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["OTF", "sciplex", "tahoe"]
SUBSETS = ["L1000_all", "L1000_in_DEG", "DEG_not_in_L1000"]
SHORT = {
    "L1000_all": "L1000 all",
    "L1000_in_DEG": "DEGs in L1000",
    "DEG_not_in_L1000": "DEGs outside L1000",
}
COLORS = {
    "L1000_all": "#4C78A8",
    "L1000_in_DEG": "#2F855A",
    "DEG_not_in_L1000": "#E45756",
}

summary = pd.read_csv(IN_DIR / "l1000_kegg_per_comparison_summary.csv")
summary = summary[summary["comparison_subset"].isin(SUBSETS)].copy()
summary.to_csv(OUT_DIR / "l1000_kegg_significant_pathway_recall_barplot_values_iqr.csv", index=False)

metric = "sig_recall_of_deg_all"
x = np.arange(len(DATASETS))
width = 0.22
offsets = (np.arange(len(SUBSETS)) - (len(SUBSETS) - 1) / 2) * width

fig, ax = plt.subplots(figsize=(3.4, 1.2))
for off, subset in zip(offsets, SUBSETS):
    medians = []
    q25 = []
    q75 = []
    for ds in DATASETS:
        row = summary[(summary["dataset"] == ds) & (summary["comparison_subset"] == subset)]
        if row.empty:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
        else:
            medians.append(float(row[f"{metric}_median"].iloc[0]))
            q25.append(float(row[f"{metric}_q25"].iloc[0]))
            q75.append(float(row[f"{metric}_q75"].iloc[0]))
    medians = np.asarray(medians, dtype=float)
    q25 = np.asarray(q25, dtype=float)
    q75 = np.asarray(q75, dtype=float)
    yerr = np.vstack([medians - q25, q75 - medians])
    xpos = x + off
    ax.bar(
        xpos,
        medians,
        width=width,
        color=to_rgba(COLORS[subset], 0.30),
        edgecolor=COLORS[subset],
        linewidth=0.75,
        label=SHORT[subset],
    )
    ax.errorbar(
        xpos,
        medians,
        yerr=yerr,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.45,
        capsize=1.5,
        capthick=0.45,
        zorder=3,
    )

ax.set_xticks(x)
ax.set_xticklabels(DATASETS, fontsize=6)
ax.set_ylabel("Recall of significant\nDEG pathways (%)", fontsize=7)
ax.tick_params(axis="y", labelsize=6, width=0.5, length=2)
ax.tick_params(axis="x", width=0.5, length=2)
ax.grid(axis="y", alpha=0.25, lw=0.35)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylim(0, 1.0)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

ax.set_title("KEGG: significant pathway recall", fontsize=6, pad=10)
ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, borderaxespad=0.0)
fig.tight_layout(pad=0.25, rect=(0, 0, 0.80, 1))

base = OUT_DIR / "l1000_kegg_significant_pathway_recall_barplot"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"saved {base.with_suffix('.pdf')}")
