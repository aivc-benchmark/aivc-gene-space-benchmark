#!/usr/bin/env python3
"""Shared streaming HVG utilities for alternative ranking methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from streaming import StreamingDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SC_PERTURB_ROOT = Path("data/sc_perturb")
SC_TAHOE_ROOT = Path("data/sc_tahoe")
GENE_ID_PATH = Path("data/hgnc_ensg_ids.json")
N_GENES = 19253
OTF_CIDS = {"c01", "c07", "c08"}
DEFAULT_TOP_NS = [2000, 3000, 5000]
EPS = 1e-12


METHOD_CONFIG = {
    "dispersion": {
        "score_col": "dispersion",
        "out_suffix": "dispersion",
        "description": "variance divided by mean, also called Fano factor",
    },
    "cv2": {
        "score_col": "cv2",
        "out_suffix": "cv2",
        "description": "squared coefficient of variation: variance divided by mean squared",
    },
    "seurat_dispersion": {
        "score_col": "normalized_dispersion",
        "out_suffix": "seurat_dispersion",
        "description": "log dispersion z-scored within mean-expression bins",
    },
}


def load_id2label(path: Path) -> dict[str, str]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "id2label" in data:
        return {f"c{int(k):02d}": str(v) for k, v in data["id2label"].items()}
    if isinstance(data, list):
        return {f"c{i:02d}": str(v) for i, v in enumerate(data)}
    raise ValueError(f"Unsupported cell type map: {path}")


def dataset_specs() -> dict[str, tuple[Path, set[str] | None, dict[str, str]]]:
    perturb_mds = SC_PERTURB_ROOT / "mds_classified_aligned"
    tahoe_mds = SC_TAHOE_ROOT / "mds_classified_aligned"
    perturb_labels = load_id2label(SC_PERTURB_ROOT / "metadata" / "cell_type_map_dt_perturb.json")
    tahoe_labels = load_id2label(SC_TAHOE_ROOT / "metadata" / "cell_type_map_dt_tahoe.json")
    perturb_cids = {p.name for p in perturb_mds.glob("c*") if p.is_dir()}
    return {
        "OTF": (perturb_mds, OTF_CIDS, perturb_labels),
        "sciplex": (perturb_mds, perturb_cids - OTF_CIDS, perturb_labels),
        "tahoe": (tahoe_mds, None, tahoe_labels),
    }


def iter_cell_dirs(root: Path, allowed_cids: set[str] | None) -> Iterable[Path]:
    for cell_dir in sorted(root.glob("c*")):
        if not cell_dir.is_dir():
            continue
        if allowed_cids is not None and cell_dir.name not in allowed_cids:
            continue
        yield cell_dir


def update_stats(sum_vec: np.ndarray, sumsq_vec: np.ndarray, n_obs: int, x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float32)
    sum_vec += x
    sumsq_vec += x * x
    return n_obs + 1


def rank_scores(scores: np.ndarray) -> np.ndarray:
    clean = np.asarray(scores, dtype=np.float64).copy()
    clean[~np.isfinite(clean)] = -np.inf
    order = np.argsort(-clean, kind="mergesort")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def compute_score(mean: np.ndarray, variance: np.ndarray, method: str, n_bins: int) -> tuple[pd.DataFrame, str]:
    if method == "dispersion":
        score_col = METHOD_CONFIG[method]["score_col"]
        score = variance / np.maximum(mean, EPS)
        return pd.DataFrame({score_col: score}), score_col

    if method == "cv2":
        score_col = METHOD_CONFIG[method]["score_col"]
        score = variance / np.maximum(mean * mean, EPS)
        return pd.DataFrame({score_col: score}), score_col

    if method == "seurat_dispersion":
        dispersion = variance / np.maximum(mean, EPS)
        log_mean = np.log1p(mean)
        log_dispersion = np.log1p(dispersion)
        valid = np.isfinite(log_mean) & np.isfinite(log_dispersion)
        bin_id = np.full(len(mean), -1, dtype=int)
        normalized = np.zeros(len(mean), dtype=np.float64)

        if valid.any():
            ranked = pd.Series(log_mean[valid]).rank(method="first")
            valid_bins = pd.qcut(ranked, q=min(n_bins, int(valid.sum())), labels=False, duplicates="drop")
            bin_id[valid] = np.asarray(valid_bins, dtype=int)
            for b in sorted(set(bin_id[valid])):
                idx = valid & (bin_id == b)
                vals = log_dispersion[idx]
                std = vals.std(ddof=0)
                if std > 0:
                    normalized[idx] = (vals - vals.mean()) / std

        score_col = METHOD_CONFIG[method]["score_col"]
        return pd.DataFrame({
            "dispersion": dispersion,
            "log_mean": log_mean,
            "log_dispersion": log_dispersion,
            "mean_bin": bin_id,
            score_col: normalized,
        }), score_col

    raise ValueError(f"Unsupported method: {method}")


def build_stats(
    dataset: str,
    gene_ids: list[str],
    sum_vec: np.ndarray,
    sumsq_vec: np.ndarray,
    n_obs: int,
    method: str,
    n_bins: int,
    cell_id: str | None = None,
    cell_type: str | None = None,
) -> tuple[pd.DataFrame, str]:
    mean = sum_vec / max(n_obs, 1)
    variance = (sumsq_vec / max(n_obs, 1)) - mean * mean
    variance = np.maximum(variance, 0)
    score_df, score_col = compute_score(mean, variance, method=method, n_bins=n_bins)

    data: dict[str, object] = {"dataset": dataset}
    if cell_id is not None:
        data["cell_id"] = cell_id
        data["cell_type"] = cell_type
    data.update({
        "gene_id": gene_ids,
        "mean": mean,
        "variance": variance,
    })
    stats = pd.DataFrame(data)
    stats = pd.concat([stats, score_df], axis=1)
    stats["hvg_method"] = method
    stats["hvg_score"] = stats[score_col]
    stats["hvg_rank"] = rank_scores(stats["hvg_score"].to_numpy())
    stats["n_observations"] = n_obs
    return stats, score_col


def write_top_files(out_dir: Path, stem: str, stats: pd.DataFrame, top_ns: list[int]) -> None:
    id_cols = [
        c for c in [
            "dataset",
            "cell_id",
            "cell_type",
            "gene_id",
            "mean",
            "variance",
            "hvg_method",
            "hvg_score",
            "hvg_rank",
        ]
        if c in stats.columns
    ]
    for n in top_ns:
        flag_col = f"highly_variable_top{n}"
        stats[flag_col] = stats["hvg_rank"] <= n
        stats.loc[stats[flag_col], id_cols].sort_values("hvg_rank").to_csv(
            out_dir / f"{stem}_top{n}_hvg.csv",
            index=False,
        )


def write_sample_h5ad(path: Path, rows: list[np.ndarray], obs_rows: list[dict], stats: pd.DataFrame) -> None:
    if not rows:
        return
    X = np.vstack(rows).astype(np.float32)
    obs = pd.DataFrame(obs_rows)
    var_cols = [
        c for c in [
            "mean",
            "variance",
            "dispersion",
            "cv2",
            "log_mean",
            "log_dispersion",
            "mean_bin",
            "normalized_dispersion",
            "hvg_method",
            "hvg_score",
            "hvg_rank",
        ]
        if c in stats.columns
    ] + [c for c in stats.columns if c.startswith("highly_variable_top")]
    var = stats.set_index("gene_id")[var_cols].copy()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path, compression="gzip")


def load_gene_ids() -> list[str]:
    gene_ids = json.load(open(GENE_ID_PATH))
    if len(gene_ids) != N_GENES:
        raise ValueError(f"Expected {N_GENES} genes, got {len(gene_ids)}")
    return gene_ids


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", choices=["OTF", "sciplex", "tahoe"], default=["OTF", "sciplex", "tahoe"])
    parser.add_argument("--top-n", nargs="+", type=int, default=DEFAULT_TOP_NS)
    parser.add_argument("--n-bins", type=int, default=20, help="Mean-expression bins for seurat_dispersion only.")
    parser.add_argument("--max-records-per-cell", type=int, default=0, help="Debug/testing cap on MDS records per cell line; 0 means all records.")


def main_dataset(method: str) -> None:
    parser = argparse.ArgumentParser(description=f"Compute dataset-level HVGs by {METHOD_CONFIG[method]['description']}.")
    add_common_args(parser)
    parser.add_argument("--h5ad-max-observations", type=int, default=50000, help="Sampled observations saved to h5ad per dataset; 0 disables h5ad output.")
    args = parser.parse_args()

    out_dir = ROOT / "data" / f"002.dataset_hvg_{METHOD_CONFIG[method]['out_suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_ids = load_gene_ids()
    specs = dataset_specs()
    summary_rows = []

    for dataset in args.datasets:
        root, allowed_cids, id2label = specs[dataset]
        sum_vec = np.zeros(N_GENES, dtype=np.float64)
        sumsq_vec = np.zeros(N_GENES, dtype=np.float64)
        n_obs = 0
        sampled_rows: list[np.ndarray] = []
        sampled_obs: list[dict] = []
        cell_dirs = list(iter_cell_dirs(root, allowed_cids))
        for cell_dir in tqdm(cell_dirs, desc=f"{dataset}:{method}", unit="cell"):
            cid = cell_dir.name
            cell_type = id2label.get(cid, cid)
            ds = StreamingDataset(local=str(cell_dir), batch_size=1024, shuffle=False)
            n_records = len(ds)
            if args.max_records_per_cell > 0:
                n_records = min(n_records, args.max_records_per_cell)
            for i in range(n_records):
                rec = ds[i]
                for state, key in (("ctrl", "gene_expression_value_ctrl"), ("pert", "gene_expression_value_pert")):
                    x = rec[key]
                    n_obs = update_stats(sum_vec, sumsq_vec, n_obs, x)
                    if args.h5ad_max_observations > 0 and len(sampled_rows) < args.h5ad_max_observations:
                        sampled_rows.append(np.asarray(x, dtype=np.float32))
                        sampled_obs.append({
                            "dataset": dataset,
                            "cell_id": cid,
                            "cell_type": cell_type,
                            "state": state,
                            "record_index": i,
                        })
        stats, score_col = build_stats(dataset, gene_ids, sum_vec, sumsq_vec, n_obs, method=method, n_bins=args.n_bins)
        write_top_files(out_dir, dataset, stats, args.top_n)
        stats.to_csv(out_dir / f"{dataset}_gene_hvg_stats.csv", index=False)
        write_sample_h5ad(out_dir / f"{dataset}_sampled_expression.h5ad", sampled_rows, sampled_obs, stats)
        summary_rows.append({
            "dataset": dataset,
            "hvg_method": method,
            "score_column": score_col,
            "n_cell_lines": len(cell_dirs),
            "n_observations": n_obs,
            "sampled_h5ad_observations": len(sampled_rows),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "dataset_hvg_summary.csv", index=False)
    print(summary.to_string(index=False))


def main_cellline(method: str) -> None:
    parser = argparse.ArgumentParser(description=f"Compute cell-line HVGs by {METHOD_CONFIG[method]['description']}.")
    add_common_args(parser)
    parser.add_argument("--h5ad-max-observations-per-cell", type=int, default=10000, help="Sampled observations saved to h5ad per dataset-cell; 0 disables h5ad output.")
    args = parser.parse_args()

    out_dir = ROOT / "data" / f"002.cellline_hvg_{METHOD_CONFIG[method]['out_suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_ids = load_gene_ids()
    specs = dataset_specs()
    summary_rows = []

    for dataset in args.datasets:
        root, allowed_cids, id2label = specs[dataset]
        for cell_dir in tqdm(list(iter_cell_dirs(root, allowed_cids)), desc=f"{dataset}:{method}", unit="cell"):
            cid = cell_dir.name
            cell_type = id2label.get(cid, cid)
            sum_vec = np.zeros(N_GENES, dtype=np.float64)
            sumsq_vec = np.zeros(N_GENES, dtype=np.float64)
            n_obs = 0
            sampled_rows: list[np.ndarray] = []
            sampled_obs: list[dict] = []
            ds = StreamingDataset(local=str(cell_dir), batch_size=1024, shuffle=False)
            n_records = len(ds)
            if args.max_records_per_cell > 0:
                n_records = min(n_records, args.max_records_per_cell)
            for i in range(n_records):
                rec = ds[i]
                for state, key in (("ctrl", "gene_expression_value_ctrl"), ("pert", "gene_expression_value_pert")):
                    x = rec[key]
                    n_obs = update_stats(sum_vec, sumsq_vec, n_obs, x)
                    if args.h5ad_max_observations_per_cell > 0 and len(sampled_rows) < args.h5ad_max_observations_per_cell:
                        sampled_rows.append(np.asarray(x, dtype=np.float32))
                        sampled_obs.append({
                            "dataset": dataset,
                            "cell_id": cid,
                            "cell_type": cell_type,
                            "state": state,
                            "record_index": i,
                        })
            stem = f"{dataset}_{cid}"
            stats, score_col = build_stats(dataset, gene_ids, sum_vec, sumsq_vec, n_obs, method=method, n_bins=args.n_bins, cell_id=cid, cell_type=cell_type)
            write_top_files(out_dir, stem, stats, args.top_n)
            stats.to_csv(out_dir / f"{stem}_gene_hvg_stats.csv", index=False)
            write_sample_h5ad(out_dir / f"{stem}_sampled_expression.h5ad", sampled_rows, sampled_obs, stats)
            summary_rows.append({
                "dataset": dataset,
                "cell_id": cid,
                "cell_type": cell_type,
                "hvg_method": method,
                "score_column": score_col,
                "n_observations": n_obs,
                "sampled_h5ad_observations": len(sampled_rows),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cellline_hvg_summary.csv", index=False)
    print(summary.to_string(index=False))
