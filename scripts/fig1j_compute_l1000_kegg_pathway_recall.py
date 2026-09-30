#!/usr/bin/env python3
from __future__ import annotations

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
L1000_ENTREZ = DATA_DIR / "003.l1000_gene_sets" / "L1000_landmark.entrez_ids.txt"
OUT_DIR = ROOT / "results" / "037.l1000_kegg_per_comparison_pathway"
PLOT_DIR = OUT_DIR / "plots" / "KEGG_per_comparison"
RESOURCE_DIR = OUT_DIR / "resources"
FDR = 0.05
MIN_QUERY = 5
DATASETS = {
    "OTF": "OTF_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "sciplex": "sciplex_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
    "tahoe": "tahoe_filtered_deg_fdr0.05_log2fc0.144_abs.csv",
}
SUBSETS = ("L1000_all", "L1000_in_DEG", "DEG_not_in_L1000")

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
    subprocess.run(["/home/yangxb/miniconda3/envs/R4.2/bin/Rscript", "-e", r_code], check=True)
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
        term_ids = []
        term_sets = []
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
    return read_gene_set(HVG_DIR / f"{dataset}_gene_hvg_stats.csv", ens2entrez)


def read_l1000() -> set[str]:
    return {line.strip() for line in L1000_ENTREZ.read_text().splitlines() if line.strip()}


enricher_terms_global: list[str] = []


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
        sr = spearmanr(base_scores, other_scores, nan_policy="omit")
        sp = float(getattr(sr, "statistic", getattr(sr, "correlation", sr[0])))
    sig_idx = [i for i, term_id in enumerate(enricher_terms_global) if term_id in base_sig]
    delta = float(np.nanmedian(other_scores[sig_idx] - base_scores[sig_idx])) if sig_idx else np.nan
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


def process_dataset(dataset: str, deg_file: str, t2g: pd.DataFrame, ens2entrez: dict[str, str], l1000_all: set[str]) -> Path:
    global enricher_terms_global
    out_path = OUT_DIR / f"{dataset}_l1000_kegg_per_comparison_metrics.csv"
    done_path = OUT_DIR / f"{dataset}.done"
    if done_path.exists() and out_path.exists():
        print(f"SKIP {dataset}: {out_path}", flush=True)
        return out_path
    universe = read_universe(dataset, ens2entrez)
    l1000 = set(l1000_all) & universe
    enricher = KeggEnricher(t2g, universe)
    enricher_terms_global = list(enricher.term_ids)
    l1000_all_enrichment = enricher.enrich(l1000)
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
        queries = {
            "L1000_all": l1000,
            "L1000_in_DEG": deg_entrez & l1000,
            "DEG_not_in_L1000": deg_entrez - l1000,
        }
        for subset in SUBSETS:
            other = l1000_all_enrichment if subset == "L1000_all" else enricher.enrich(queries[subset])
            row = comparison_metrics(base, other, subset)
            row.update({
                "dataset": dataset,
                "cell_id": cell_id,
                "perturbation_id": perturbation_id,
                "library": "KEGG",
                "panel": "L1000_landmark",
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
    df.to_csv(OUT_DIR / "all_l1000_kegg_per_comparison_metrics.csv", index=False)
    metrics = ["sig_recall_of_deg_all", "top10_overlap_fraction", "top20_overlap_fraction", "spearman_minus_log10_fdr", "median_delta_minus_log10_fdr"]
    agg = {}
    for metric in metrics:
        agg[f"{metric}_count"] = (metric, "count")
        agg[f"{metric}_median"] = (metric, "median")
        agg[f"{metric}_q25"] = (metric, lambda x: x.quantile(0.25))
        agg[f"{metric}_q75"] = (metric, lambda x: x.quantile(0.75))
    summary = df.groupby(["dataset", "comparison_subset"], as_index=False).agg(**agg)
    summary.to_csv(OUT_DIR / "l1000_kegg_per_comparison_summary.csv", index=False)

    label_map = {"L1000_all": "L1000", "L1000_in_DEG": "DEGs in L1000", "DEG_not_in_L1000": "DEGs outside L1000"}
    color_map = {"L1000_all": "#4C78A8", "L1000_in_DEG": "#2F855A", "DEG_not_in_L1000": "#E45756"}
    dataset_labels = {"OTF": "PerturBase-ORF", "sciplex": "PerturBase-Chemical", "tahoe": "Tahoe-100M"}

    def save(fig, name):
        base = PLOT_DIR / name
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    def bar_metric(metric, ylabel, stem, percent=True, ylim=None):
        fig, axes = plt.subplots(1, 3, figsize=(4.2, 1.45), sharey=True)
        for ax, ds in zip(axes, DATASETS):
            vals = []
            q25 = []
            q75 = []
            labels = []
            colors = []
            for subset in SUBSETS:
                row = summary[(summary.dataset == ds) & (summary.comparison_subset == subset)]
                if row.empty:
                    vals.append(np.nan); q25.append(np.nan); q75.append(np.nan)
                else:
                    vals.append(float(row[f"{metric}_median"].iloc[0]))
                    q25.append(float(row[f"{metric}_q25"].iloc[0]))
                    q75.append(float(row[f"{metric}_q75"].iloc[0]))
                labels.append(label_map[subset])
                colors.append(color_map[subset])
            x = np.arange(len(vals))
            yerr = np.vstack([np.array(vals) - np.array(q25), np.array(q75) - np.array(vals)])
            ax.bar(x, vals, color=colors, width=0.65, edgecolor="#222222", linewidth=0.35)
            ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=0.45, capsize=1.5)
            ax.set_title(dataset_labels[ds], fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5)
            ax.tick_params(axis="y", labelsize=5, width=0.4, length=1.5)
            ax.grid(axis="y", alpha=0.25, lw=0.25)
            if percent:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
            if ylim is not None:
                ax.set_ylim(*ylim)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[0].set_ylabel(ylabel, fontsize=6)
        fig.tight_layout(pad=0.2)
        save(fig, f"l1000_kegg_{stem}_barplot")

    def heatmap(metric, ylabel, stem, percent=True, vmin=0, vmax=1):
        mat = np.full((len(DATASETS), len(SUBSETS)), np.nan)
        for i, ds in enumerate(DATASETS):
            for j, subset in enumerate(SUBSETS):
                row = summary[(summary.dataset == ds) & (summary.comparison_subset == subset)]
                if not row.empty:
                    mat[i, j] = float(row[f"{metric}_median"].iloc[0])
        fig, ax = plt.subplots(figsize=(3.35, 1.55))
        im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SUBSETS)))
        ax.set_xticklabels([label_map[s] for s in SUBSETS], rotation=25, ha="right", fontsize=5)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([dataset_labels[d] for d in DATASETS], fontsize=5)
        ax.tick_params(length=0)
        for i in range(len(DATASETS)):
            for j in range(len(SUBSETS)):
                val = mat[i, j]
                lab = "NA" if np.isnan(val) else (f"{val*100:.0f}%" if percent else f"{val:.2f}")
                ax.text(j, i, lab, ha="center", va="center", fontsize=5, color="#111111")
        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        if percent:
            cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)
        cbar.set_label(ylabel, fontsize=6)
        fig.tight_layout(pad=0.2)
        save(fig, f"l1000_kegg_{stem}_heatmap")

    bar_metric("sig_recall_of_deg_all", "Significant pathway recall (%)", "significant_pathway_recall", True, (0, 1.0))
    heatmap("sig_recall_of_deg_all", "Significant pathway recall", "significant_pathway_recall", True, 0, 1.0)
    bar_metric("top20_overlap_fraction", "Top20 pathway overlap (%)", "top20_pathway_overlap", True, (0, 1.0))
    heatmap("top20_overlap_fraction", "Top20 pathway overlap", "top20_pathway_overlap", True, 0, 1.0)
    bar_metric("spearman_minus_log10_fdr", "Pathway score Spearman", "pathway_score_spearman", False, (-0.2, 1.0))
    heatmap("spearman_minus_log10_fdr", "Pathway score Spearman", "pathway_score_spearman", False, -0.2, 1.0)


def main() -> None:
    ensure_dirs()
    t2g, ens2entrez = load_mapping()
    l1000_all = read_l1000()
    print(f"L1000 Entrez genes: {len(l1000_all)}", flush=True)
    paths = []
    for dataset, deg_file in DATASETS.items():
        paths.append(process_dataset(dataset, deg_file, t2g, ens2entrez, l1000_all))
    summarize_and_plot(paths)
    print(f"DONE all outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
