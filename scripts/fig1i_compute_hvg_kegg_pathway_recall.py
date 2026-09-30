#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HVG_DIR = DATA_DIR / "002.dataset_hvg_seurat_dispersion"
OUT_DIR = ROOT / "results" / "036.kegg_multitop_per_comparison_pathway"
PLOT_DIR = OUT_DIR / "plots" / "KEGG_top_axis"
RESOURCE_DIR = OUT_DIR / "resources"
TOP_NS = (2000, 3000, 5000, 10000)
FDR = 0.05
MIN_QUERY = 5
DATASETS = {
    "OTF": "OTF_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "sciplex": "sciplex_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "tahoe": "tahoe_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
}
SUBSETS = ("HVG_all", "HVG_in_DEG", "DEG_not_in_HVG")

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def export_orgdb_resources() -> tuple[Path, Path]:
    t2g = RESOURCE_DIR / "orgdb_kegg_term2entrez.csv"
    e2e = RESOURCE_DIR / "orgdb_ensembl2entrez.csv"
    if t2g.exists() and e2e.exists():
        return t2g, e2e
    r_code = f"""
    suppressPackageStartupMessages({{
      library(org.Hs.eg.db)
      library(AnnotationDbi)
      library(data.table)
    }})
    t2g <- AnnotationDbi::toTable(org.Hs.egPATH2EG)
    e2e <- AnnotationDbi::toTable(org.Hs.egENSEMBL2EG)
    data.table::fwrite(t2g, "{t2g}")
    data.table::fwrite(e2e, "{e2e}")
    """
    subprocess.run(
        ["/home/yangxb/miniconda3/envs/R4.2/bin/Rscript", "-e", r_code],
        check=True,
    )
    return t2g, e2e


def load_mapping() -> tuple[pd.DataFrame, dict[str, str]]:
    t2g_path, e2e_path = export_orgdb_resources()
    t2g = pd.read_csv(t2g_path, dtype=str)
    e2e = pd.read_csv(e2e_path, dtype=str)
    term_col = next(c for c in t2g.columns if "path" in c.lower())
    gene_col = next(c for c in t2g.columns if "gene" in c.lower())
    ens_col = next(c for c in e2e.columns if "ensembl" in c.lower())
    entrez_col = next(c for c in e2e.columns if "gene" in c.lower() or "entrez" in c.lower())
    t2g = t2g[[term_col, gene_col]].dropna().drop_duplicates()
    t2g.columns = ["term_id", "entrez_id"]
    t2g["term_id"] = t2g["term_id"].astype(str).map(lambda x: x if x.startswith("hsa") else f"hsa{x}")
    t2g["entrez_id"] = t2g["entrez_id"].astype(str)
    e2e = e2e[[ens_col, entrez_col]].dropna().drop_duplicates()
    e2e.columns = ["ensembl", "entrez_id"]
    ens2entrez = dict(zip(e2e["ensembl"].astype(str), e2e["entrez_id"].astype(str)))
    return t2g, ens2entrez


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    adj = np.empty_like(q)
    adj[order] = q
    out[ok] = adj
    return out


class KeggEnricher:
    def __init__(self, t2g: pd.DataFrame, universe: set[str]):
        self.universe = set(universe)
        term_sets = []
        term_ids = []
        for term_id, sub in t2g.groupby("term_id"):
            genes = set(sub["entrez_id"].astype(str)) & self.universe
            if len(genes) >= MIN_QUERY:
                term_ids.append(term_id)
                term_sets.append(genes)
        self.term_ids = np.array(term_ids, dtype=object)
        self.term_sets = term_sets
        self.term_sizes = np.array([len(s) for s in term_sets], dtype=int)
        self.N = len(self.universe)

    def enrich(self, query: set[str]) -> dict[str, object]:
        q = set(query) & self.universe
        n = len(q)
        if n < MIN_QUERY or self.N == 0:
            scores = np.zeros(len(self.term_ids), dtype=float)
            return {"n": n, "sig": set(), "top10": set(), "top20": set(), "scores": scores}
        k = np.array([len(q & genes) for genes in self.term_sets], dtype=int)
        pvals = np.ones(len(k), dtype=float)
        hit = k > 0
        pvals[hit] = hypergeom.sf(k[hit] - 1, self.N, self.term_sizes[hit], n)
        padj = bh_adjust(pvals)
        scores = -np.log10(np.clip(padj, 1e-300, 1.0))
        order = np.lexsort((-scores, padj))
        sig = set(self.term_ids[np.where(padj < FDR)[0]])
        top10 = set(self.term_ids[order[: min(10, len(order))]])
        top20 = set(self.term_ids[order[: min(20, len(order))]])
        return {"n": n, "sig": sig, "top10": top10, "top20": top20, "scores": scores}


def read_gene_set(path: Path, ens2entrez: dict[str, str]) -> set[str]:
    df = pd.read_csv(path, usecols=["gene_id"])
    return {ens2entrez[g] for g in df["gene_id"].dropna().astype(str) if g in ens2entrez}


def read_universe(dataset: str, ens2entrez: dict[str, str]) -> set[str]:
    path = HVG_DIR / f"{dataset}_gene_hvg_stats.csv"
    return read_gene_set(path, ens2entrez)


def read_hvg(dataset: str, top_n: int, ens2entrez: dict[str, str]) -> set[str]:
    path = HVG_DIR / f"{dataset}_top{top_n}_hvg.csv"
    return read_gene_set(path, ens2entrez)


def comparison_metrics(base: dict[str, object], other: dict[str, object], subset: str) -> dict[str, object]:
    base_sig = base["sig"]
    other_sig = other["sig"]
    sig_union = base_sig | other_sig
    sig_overlap = base_sig & other_sig
    top10_overlap = base["top10"] & other["top10"]
    top20_overlap = base["top20"] & other["top20"]
    base_scores = np.asarray(base["scores"], dtype=float)
    other_scores = np.asarray(other["scores"], dtype=float)
    if np.nanstd(base_scores) == 0 or np.nanstd(other_scores) == 0:
        sp = np.nan
    else:
        _sr = spearmanr(base_scores, other_scores, nan_policy="omit")
        sp = float(getattr(_sr, "statistic", getattr(_sr, "correlation", _sr[0])))
    sig_idx = [i for i, term_id in enumerate(enricher_terms_global) if term_id in base_sig]
    if sig_idx:
        delta = float(np.nanmedian(other_scores[sig_idx] - base_scores[sig_idx]))
    else:
        delta = np.nan
    return {
        "comparison_subset": subset,
        "deg_all_query_n": int(base["n"]),
        "comparison_query_n": int(other["n"]),
        "deg_all_sig_n": len(base_sig),
        "comparison_sig_n": len(other_sig),
        "sig_overlap_n": len(sig_overlap),
        "sig_recall_of_deg_all": len(sig_overlap) / len(base_sig) if base_sig else np.nan,
        "sig_jaccard": len(sig_overlap) / len(sig_union) if sig_union else np.nan,
        "top10_overlap_n": len(top10_overlap),
        "top10_overlap_fraction": len(top10_overlap) / len(base["top10"]) if base["top10"] else np.nan,
        "top20_overlap_n": len(top20_overlap),
        "top20_overlap_fraction": len(top20_overlap) / len(base["top20"]) if base["top20"] else np.nan,
        "spearman_minus_log10_fdr": sp,
        "median_delta_minus_log10_fdr": delta,
    }


enricher_terms_global: list[str] = []


def process_dataset(dataset: str, deg_file: str, t2g: pd.DataFrame, ens2entrez: dict[str, str]) -> Path:
    global enricher_terms_global
    out_path = OUT_DIR / f"{dataset}_kegg_multitop_per_comparison_metrics.csv"
    done_path = OUT_DIR / f"{dataset}.done"
    if done_path.exists() and out_path.exists():
        print(f"SKIP {dataset}: {out_path}", flush=True)
        return out_path
    universe = read_universe(dataset, ens2entrez)
    hvg_by_top = {top_n: read_hvg(dataset, top_n, ens2entrez) for top_n in TOP_NS}
    enricher = KeggEnricher(t2g, universe)
    enricher_terms_global = list(enricher.term_ids)
    hvg_all_enrichment = {top_n: enricher.enrich(hvg_by_top[top_n]) for top_n in TOP_NS}
    deg_path = DATA_DIR / deg_file
    if out_path.exists():
        out_path.unlink()
    rows_buffer = []
    current_key = None
    current_genes: set[str] = set()
    processed = 0
    start = time.time()

    def flush_one(key, genes):
        nonlocal processed, rows_buffer
        if key is None:
            return
        cell_id, perturbation_id = key
        deg_entrez = {ens2entrez[g] for g in genes if g in ens2entrez} & universe
        base = enricher.enrich(deg_entrez)
        for top_n in TOP_NS:
            hvg = hvg_by_top[top_n]
            queries = {
                "HVG_all": hvg,
                "HVG_in_DEG": deg_entrez & hvg,
                "DEG_not_in_HVG": deg_entrez - hvg,
            }
            for subset in SUBSETS:
                other = hvg_all_enrichment[top_n] if subset == "HVG_all" else enricher.enrich(queries[subset])
                row = comparison_metrics(base, other, subset)
                row.update({
                    "dataset": dataset,
                    "cell_id": cell_id,
                    "perturbation_id": perturbation_id,
                    "library": "KEGG",
                    "hvg_method": "seurat_dispersion",
                    "hvg_top_n": top_n,
                })
                rows_buffer.append(row)
        processed += 1
        if len(rows_buffer) >= 6000:
            pd.DataFrame(rows_buffer).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            rows_buffer = []
        if processed % 1000 == 0:
            elapsed = (time.time() - start) / 60
            print(f"{dataset}: processed {processed} comparisons in {elapsed:.1f} min", flush=True)

    usecols = ["cell_id", "perturbation_id", "gene_id"]
    for chunk in pd.read_csv(deg_path, usecols=usecols, chunksize=1_000_000):
        for (cell_id, perturbation_id), sub in chunk.groupby(["cell_id", "perturbation_id"], sort=False):
            key = (str(cell_id), str(perturbation_id))
            genes = set(sub["gene_id"].dropna().astype(str))
            if current_key is None:
                current_key = key
                current_genes = genes
            elif key == current_key:
                current_genes.update(genes)
            else:
                flush_one(current_key, current_genes)
                current_key = key
                current_genes = genes
    flush_one(current_key, current_genes)
    if rows_buffer:
        pd.DataFrame(rows_buffer).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    done_path.write_text(f"done {dataset} {processed} comparisons\n")
    print(f"DONE {dataset}: {processed} comparisons -> {out_path}", flush=True)
    return out_path

def summarize_and_plot(paths: list[Path]) -> None:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    all_path = OUT_DIR / "all_kegg_multitop_per_comparison_metrics.csv"
    df.to_csv(all_path, index=False)
    metrics = [
        "sig_recall_of_deg_all",
        "top10_overlap_fraction",
        "top20_overlap_fraction",
        "spearman_minus_log10_fdr",
        "median_delta_minus_log10_fdr",
    ]
    agg = {}
    for metric in metrics:
        agg[f"{metric}_count"] = (metric, "count")
        agg[f"{metric}_median"] = (metric, "median")
        agg[f"{metric}_q25"] = (metric, lambda x: x.quantile(0.25))
        agg[f"{metric}_q75"] = (metric, lambda x: x.quantile(0.75))
    summary = df.groupby(["dataset", "hvg_top_n", "comparison_subset"], as_index=False).agg(**agg)
    summary.to_csv(OUT_DIR / "kegg_multitop_per_comparison_summary.csv", index=False)

    label_map = {"HVG_all": "Complete HVG", "HVG_in_DEG": "DEGs in HVG", "DEG_not_in_HVG": "DEGs outside HVG"}
    color_map = {"HVG_all": "#4C78A8", "HVG_in_DEG": "#2F855A", "DEG_not_in_HVG": "#E45756"}
    marker_map = {"HVG_all": "o", "HVG_in_DEG": "s", "DEG_not_in_HVG": "^"}
    dataset_labels = {"OTF": "PerturBase-ORF", "sciplex": "PerturBase-Chemical", "tahoe": "Tahoe-100M"}

    def save(fig, name):
        base = PLOT_DIR / name
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    def line_plot(metric, ylabel, stem, percent=True, ylim=(0, 1.0)):
        fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.2), sharey=True)
        for ax, ds in zip(axes, DATASETS):
            for subset in SUBSETS:
                d = summary[(summary.dataset == ds) & (summary.comparison_subset == subset)].sort_values("hvg_top_n")
                if d.empty:
                    continue
                ax.plot(d.hvg_top_n, d[f"{metric}_median"], marker=marker_map[subset], lw=0.8, ms=2.0, color=color_map[subset], label=label_map[subset])
            ax.set_title(dataset_labels[ds], fontsize=6)
            ax.set_xlabel("HVG topN", fontsize=6)
            ax.set_xticks(list(TOP_NS))
            ax.set_xticklabels(["2k", "3k", "5k", "10k"], fontsize=5)
            ax.tick_params(axis="both", labelsize=5, width=0.4, length=1.5)
            ax.grid(alpha=0.25, lw=0.25)
            if percent:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
            if ylim is not None:
                ax.set_ylim(*ylim)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[0].set_ylabel(ylabel, fontsize=6)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=5, handlelength=1.2, columnspacing=0.7)
        fig.tight_layout(pad=0.15)
        save(fig, f"kegg_{stem}_top_axis_line")

    def heatmap(metric, ylabel, stem, percent=True, vmin=0, vmax=1):
        fig, axes = plt.subplots(1, len(SUBSETS), figsize=(2.05 * len(SUBSETS), 1.72), constrained_layout=True)
        for ax, subset in zip(axes, SUBSETS):
            mat = np.full((len(DATASETS), len(TOP_NS)), np.nan)
            for i, ds in enumerate(DATASETS):
                for j, topn in enumerate(TOP_NS):
                    row = summary[(summary.dataset == ds) & (summary.hvg_top_n == topn) & (summary.comparison_subset == subset)]
                    if not row.empty:
                        mat[i, j] = float(row[f"{metric}_median"].iloc[0])
            im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
            ax.set_title(label_map[subset], fontsize=6)
            ax.set_xticks(range(len(TOP_NS)))
            ax.set_xticklabels(["2k", "3k", "5k", "10k"], fontsize=5)
            ax.set_yticks(range(len(DATASETS)))
            ax.set_yticklabels([dataset_labels[d] for d in DATASETS], fontsize=5)
            ax.tick_params(length=0)
            for i in range(len(DATASETS)):
                for j in range(len(TOP_NS)):
                    val = mat[i, j]
                    lab = "NA" if np.isnan(val) else (f"{val * 100:.0f}%" if percent else f"{val:.2f}")
                    ax.text(j, i, lab, ha="center", va="center", fontsize=5, color="#111111")
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
        cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
        if percent:
            cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)
        cbar.set_label(ylabel, fontsize=6)
        save(fig, f"kegg_{stem}_top_axis_heatmap")

    line_plot("top10_overlap_fraction", "Top10 pathway recall (%)", "top10_pathway_recall", True, (0, 1.0))
    heatmap("top20_overlap_fraction", "Top20 pathway overlap", "top20_pathway_overlap", True, 0, 1.0)
    line_plot("sig_recall_of_deg_all", "Significant pathway recall (%)", "pathway_recall", True, (0, 1.0))
    heatmap("sig_recall_of_deg_all", "Significant pathway recall", "pathway_recall", True, 0, 1.0)

def main() -> None:
    ensure_dirs()
    t2g, ens2entrez = load_mapping()
    paths = []
    for dataset, deg_file in DATASETS.items():
        paths.append(process_dataset(dataset, deg_file, t2g, ens2entrez))
    summarize_and_plot(paths)
    print(f"DONE all outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
