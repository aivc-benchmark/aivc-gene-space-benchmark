#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["OTF", "sciplex", "tahoe"]
METHODS = ["cv2", "dispersion", "seurat_dispersion"]
TOP_NS = [2000, 3000, 5000, 10000]
HVG_LEVELS = ["dataset", "cellline"]
DEG_FILES = [
    "OTF_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "sciplex_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "tahoe_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
]
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300})


def norm(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else None


def load_gene_set_csv(path: Path) -> set[str]:
    if not path.exists(): return set()
    df = pd.read_csv(path, usecols=["gene_id"], dtype=str)
    return set(df["gene_id"].map(norm).dropna())


def discover_hvg_sets(root: Path, methods: list[str], top_ns: list[int]):
    dataset_sets, cellline_sets, audit = {}, {}, []
    for method in methods:
        ddir = root / "data" / f"002.dataset_hvg_{method}"
        cdir = root / "data" / f"002.cellline_hvg_{method}"
        for dataset in DATASETS:
            stats_path = ddir / f"{dataset}_gene_hvg_stats.csv"
            n_obs = None
            if stats_path.exists():
                try:
                    n_obs = pd.read_csv(stats_path, usecols=["n_observations"], nrows=1)["n_observations"].iloc[0]
                except Exception:
                    pass
            for n in top_ns:
                path = ddir / f"{dataset}_top{n}_hvg.csv"
                genes = load_gene_set_csv(path)
                audit.append({"hvg_method": method, "hvg_level": "dataset", "dataset": dataset, "cell_id": "", "hvg_top_n": n, "exists": path.exists(), "n_genes": len(genes), "n_observations": n_obs, "path": str(path)})
                if genes: dataset_sets[(method, dataset, n)] = genes
        if cdir.exists():
            for path in sorted(cdir.glob("*_top*_hvg.csv")):
                stem = path.name.replace("_hvg.csv", "")
                try:
                    left, top = stem.rsplit("_top", 1)
                    dataset, cell_id = left.split("_", 1)
                    n = int(top)
                except ValueError:
                    continue
                if n not in top_ns: continue
                genes = load_gene_set_csv(path)
                stats_path = cdir / f"{dataset}_{cell_id}_gene_hvg_stats.csv"
                n_obs = None
                if stats_path.exists():
                    try:
                        n_obs = pd.read_csv(stats_path, usecols=["n_observations"], nrows=1)["n_observations"].iloc[0]
                    except Exception:
                        pass
                audit.append({"hvg_method": method, "hvg_level": "cellline", "dataset": dataset, "cell_id": cell_id, "hvg_top_n": n, "exists": True, "n_genes": len(genes), "n_observations": n_obs, "path": str(path)})
                if genes: cellline_sets[(method, dataset, cell_id, n)] = genes
    return dataset_sets, cellline_sets, pd.DataFrame(audit)


def build_deg_groups(root: Path, deg_files: list[str], chunksize: int):
    group_cols = ["dataset", "cell_id", "cell_type", "perturbation_id"]
    groups = {}
    dtype = {c: str for c in group_cols + ["gene_id"]}
    for deg_file in deg_files:
        path = root / "data" / deg_file
        print(f"Reading DEG {path}", flush=True)
        for chunk in pd.read_csv(path, usecols=list(dtype), dtype=dtype, chunksize=chunksize):
            chunk["gene_id"] = chunk["gene_id"].map(norm)
            chunk = chunk[chunk["gene_id"].notna()]
            for key, sub in chunk.groupby(group_cols, sort=False):
                groups.setdefault(tuple(key), set()).update(sub["gene_id"])
    return groups


def compute_counts(root: Path, deg_files: list[str], methods: list[str], top_ns: list[int], chunksize: int):
    dataset_sets, cellline_sets, audit = discover_hvg_sets(root, methods, top_ns)
    print(f"Loaded alternative HVG sets: dataset={len(dataset_sets)}, cellline={len(cellline_sets)}", flush=True)
    deg_groups = build_deg_groups(root, deg_files, chunksize)
    print(f"Loaded DEG groups={len(deg_groups)}", flush=True)
    rows, skipped = [], []
    for dataset, cell_id, cell_type, perturbation_id in deg_groups:
        deg = deg_groups[(dataset, cell_id, cell_type, perturbation_id)]
        deg_n = len(deg)
        for method in methods:
            for n in top_ns:
                hvg = dataset_sets.get((method, dataset, n))
                if hvg is None:
                    skipped.append({"hvg_method": method, "hvg_level": "dataset", "dataset": dataset, "cell_id": "", "hvg_top_n": n, "reason": "missing_hvg_set"})
                else:
                    inter = deg & hvg
                    rows.append({"dataset": dataset, "cell_id": cell_id, "cell_type": cell_type, "perturbation_id": perturbation_id, "hvg_method": method, "hvg_level": "dataset", "hvg_top_n": n, "deg_n": deg_n, "hvg_n": len(hvg), "hvg_deg_n": len(inter), "hvg_deg_fraction_of_deg": len(inter) / deg_n if deg_n else np.nan})
                hvg = cellline_sets.get((method, dataset, cell_id, n))
                if hvg is None:
                    skipped.append({"hvg_method": method, "hvg_level": "cellline", "dataset": dataset, "cell_id": cell_id, "hvg_top_n": n, "reason": "missing_hvg_set"})
                else:
                    inter = deg & hvg
                    rows.append({"dataset": dataset, "cell_id": cell_id, "cell_type": cell_type, "perturbation_id": perturbation_id, "hvg_method": method, "hvg_level": "cellline", "hvg_top_n": n, "deg_n": deg_n, "hvg_n": len(hvg), "hvg_deg_n": len(inter), "hvg_deg_fraction_of_deg": len(inter) / deg_n if deg_n else np.nan})
    return pd.DataFrame(rows), audit, pd.DataFrame(skipped).drop_duplicates()


def apply_fraction_axis(ax, ymax):
    ymax = min(1.0, max(0.3, float(ymax)))
    ymax = np.ceil(ymax * 10) / 10
    ticks = np.linspace(0, ymax, int(round(ymax / 0.1)) + 1)
    ax.set_ylim(0, ymax); ax.set_yticks(ticks); ax.set_yticklabels([f"{x:.1f}" for x in ticks]); ax.grid(False)


def plot_fraction_boxplot(samples: pd.DataFrame, out_prefix: Path, title: str, formats: list[str]):
    datasets = [d for d in DATASETS if d in set(samples["dataset"])]
    if not datasets: return
    values = [samples.loc[samples["dataset"].eq(d), "hvg_deg_fraction_of_deg"].dropna().to_numpy() for d in datasets]
    color_map = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(datasets)}
    fig, ax = plt.subplots(figsize=(max(3.8, 0.52 * len(datasets) + 1.0), 3.7))
    x = np.arange(len(datasets))
    bp = ax.boxplot(values, positions=x, widths=0.38, patch_artist=True, showfliers=False,
                    medianprops={"color":"black","linewidth":1.1}, whiskerprops={"color":"black","linewidth":1.0}, capprops={"color":"black","linewidth":1.0}, boxprops={"linewidth":1.1})
    for patch, dataset in zip(bp["boxes"], datasets):
        patch.set_facecolor(color_map[dataset]); patch.set_alpha(0.2); patch.set_edgecolor(color_map[dataset])
    rng = np.random.default_rng(42); means=[]; highs=[]
    for i, (dataset, vals) in enumerate(zip(datasets, values)):
        means.append(np.nanmean(vals) if len(vals) else np.nan); highs.append(np.nanmax(vals) if len(vals) else np.nan)
        if len(vals):
            ax.scatter(np.full(len(vals), x[i]) + rng.uniform(-0.12, 0.12, len(vals)), vals, s=6, color=color_map[dataset], edgecolors="white", linewidths=0.35, alpha=0.85, zorder=3)
    ymax_data = np.nanmax([np.nanmax(v) if len(v) else 0 for v in values] + [0.3])
    apply_fraction_axis(ax, min(1.0, ymax_data + 0.08)); ymax = ax.get_ylim()[1]
    for xi, mean, high in zip(x, means, highs):
        if pd.notna(mean): ax.text(xi, min(ymax - 0.005, max(high, mean) + 0.012), f"{mean:.2f}", ha="center", va="bottom", fontsize=9, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("Fraction"); ax.set_xlabel("Dataset"); ax.set_title(title)
    fig.tight_layout(); out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats: fig.savefig(out_prefix.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def plot_all(counts: pd.DataFrame, out_dir: Path, formats: list[str]) -> pd.DataFrame:
    rows = []
    for (method, level, top_n), sub in counts.groupby(["hvg_method", "hvg_level", "hvg_top_n"], sort=False):
        top_n = int(top_n)
        plot_fraction_boxplot(sub, out_dir / method / level / f"top{top_n}_{method}_{level}_hvg_deg_intersection_in_deg_fraction_boxplot", f"{method} HVG-DEG intersection fraction ({level}, top{top_n})", formats)
        rows.append({"hvg_method": method, "hvg_level": level, "hvg_top_n": top_n, "n_rows": int(len(sub)), "datasets": ",".join([d for d in DATASETS if d in set(sub["dataset"])]), "mean_hvg_deg_fraction": float(sub["hvg_deg_fraction_of_deg"].mean()), "zero_fraction_rows": int((sub["hvg_deg_n"] == 0).sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--deg-files", nargs="+", default=DEG_FILES)
    ap.add_argument("--top-n", nargs="+", type=int, default=TOP_NS)
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    args = ap.parse_args()
    out_dir = args.root / "results" / "007.alt_hvg_deg_intersection_boxplots"; out_dir.mkdir(parents=True, exist_ok=True)
    counts, audit, skipped = compute_counts(args.root, args.deg_files, args.methods, args.top_n, args.chunksize)
    audit.to_csv(out_dir / "alt_hvg_input_audit.csv", index=False)
    skipped.to_csv(out_dir / "alt_hvg_missing_sets.csv", index=False)
    if counts.empty: raise ValueError("No alternative HVG/DEG intersections were computed.")
    counts.to_csv(out_dir / "alt_hvg_deg_intersection_by_perturbation.csv", index=False)
    summary = plot_all(counts, out_dir, args.formats)
    summary.to_csv(out_dir / "alt_hvg_deg_intersection_boxplot_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"DONE rows={len(counts)} skipped={len(skipped)} out={out_dir}", flush=True)


if __name__ == "__main__": main()
