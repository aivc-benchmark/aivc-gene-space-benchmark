#!/usr/bin/env python3
"""
Download and parse LINCS L1000 gene metadata into L1000/BING/AIG gene sets.

Outputs are written under:
  data/003.l1000_gene_sets

Definitions from GSE92742_Broad_LINCS_gene_info.txt.gz:
  - L1000_landmark: pr_is_lm == 1
  - BING_expanded: pr_is_bing == 1, including landmark genes
  - BING_inferred_only: pr_is_bing == 1 and pr_is_lm == 0
  - AIG_expanded: all genes in the L1000 all-inferred gene space
  - AIG_inferred_only: all genes excluding landmark genes
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_SUBDIR = Path("003.l1000_gene_sets")
GENE_INFO_NAME = "GSE92742_Broad_LINCS_gene_info.txt.gz"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_gene_info.txt.gz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"Using existing raw file: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {url}\n{exc}") from exc
    tmp.replace(dest)
    print(f"Saved raw file: {dest}")


def read_gene_info(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as handle:
        df = pd.read_csv(handle, sep="\t", dtype={"pr_gene_id": str})

    required = {"pr_gene_id", "pr_gene_symbol", "pr_gene_title", "pr_is_lm", "pr_is_bing"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df.rename(
        columns={
            "pr_gene_id": "entrez_id",
            "pr_gene_symbol": "gene_symbol",
            "pr_gene_title": "gene_title",
        }
    )
    df["is_l1000_landmark"] = pd.to_numeric(df["pr_is_lm"], errors="coerce").fillna(0).astype(int).astype(bool)
    df["is_bing"] = pd.to_numeric(df["pr_is_bing"], errors="coerce").fillna(0).astype(int).astype(bool)
    df["is_aig"] = True
    df["gene_set_membership"] = "AIG_inferred_only"
    df.loc[df["is_bing"], "gene_set_membership"] = "BING_inferred_only"
    df.loc[df["is_l1000_landmark"], "gene_set_membership"] = "L1000_landmark"

    cols = [
        "entrez_id",
        "gene_symbol",
        "gene_title",
        "is_l1000_landmark",
        "is_bing",
        "is_aig",
        "gene_set_membership",
    ]
    df = df[cols].sort_values(["gene_symbol", "entrez_id"], kind="mergesort").reset_index(drop=True)
    if df["entrez_id"].duplicated().any():
        dupes = df.loc[df["entrez_id"].duplicated(), "entrez_id"].head().tolist()
        raise ValueError(f"Duplicated Entrez IDs observed: {dupes}")
    return df


def write_gene_set(df: pd.DataFrame, out_dir: Path, name: str, mask: pd.Series) -> dict[str, object]:
    subset = df.loc[mask].sort_values(["gene_symbol", "entrez_id"], kind="mergesort").copy()
    csv_path = out_dir / f"{name}.csv"
    symbol_path = out_dir / f"{name}.symbols.txt"
    entrez_path = out_dir / f"{name}.entrez_ids.txt"

    subset.to_csv(csv_path, index=False)
    subset["gene_symbol"].dropna().drop_duplicates().to_csv(symbol_path, index=False, header=False)
    subset["entrez_id"].dropna().drop_duplicates().to_csv(entrez_path, index=False, header=False)

    return {
        "gene_set": name,
        "n_rows": int(len(subset)),
        "n_gene_symbols": int(subset["gene_symbol"].nunique()),
        "n_entrez_ids": int(subset["entrez_id"].nunique()),
        "csv": str(csv_path),
        "symbols_txt": str(symbol_path),
        "entrez_ids_txt": str(entrez_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Parse an existing raw gene_info file without network access.")
    args = parser.parse_args()

    out_dir = args.root / "data" / OUT_SUBDIR
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / GENE_INFO_NAME

    if not args.skip_download:
        download(GENE_INFO_URL, raw_path, force=args.force_download)
    elif not raw_path.exists():
        raise FileNotFoundError(f"--skip-download requested, but raw file is missing: {raw_path}")

    df = read_gene_info(raw_path)
    df.to_csv(out_dir / "l1000_gene_info_with_sets.csv", index=False)

    manifest = {
        "source_url": GENE_INFO_URL,
        "raw_file": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "all_gene_info_csv": str(out_dir / "l1000_gene_info_with_sets.csv"),
        "gene_sets": [],
    }
    manifest["gene_sets"].append(write_gene_set(df, out_dir, "L1000_landmark", df["is_l1000_landmark"]))
    manifest["gene_sets"].append(write_gene_set(df, out_dir, "BING_expanded", df["is_bing"]))
    manifest["gene_sets"].append(write_gene_set(df, out_dir, "BING_inferred_only", df["is_bing"] & ~df["is_l1000_landmark"]))
    manifest["gene_sets"].append(write_gene_set(df, out_dir, "AIG_expanded", df["is_aig"]))
    manifest["gene_sets"].append(write_gene_set(df, out_dir, "AIG_inferred_only", df["is_aig"] & ~df["is_l1000_landmark"]))

    with (out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    summary = pd.DataFrame(manifest["gene_sets"])
    summary.to_csv(out_dir / "gene_set_summary.csv", index=False)
    print(summary[["gene_set", "n_rows", "n_gene_symbols", "n_entrez_ids"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
