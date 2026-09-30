#!/usr/bin/env python3
"""Compute dataset-cell-line HVGs ranked by mean-bin normalized dispersion."""
from hvg_method_common import main_cellline


if __name__ == "__main__":
    main_cellline("seurat_dispersion")
