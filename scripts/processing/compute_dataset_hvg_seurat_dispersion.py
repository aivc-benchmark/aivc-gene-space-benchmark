#!/usr/bin/env python3
"""Compute dataset-level HVGs ranked by mean-bin normalized dispersion."""
from hvg_method_common import main_dataset


if __name__ == "__main__":
    main_dataset("seurat_dispersion")
