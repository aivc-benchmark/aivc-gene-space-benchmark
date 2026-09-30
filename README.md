# aivc-gene-space-benchmark

Code, source data and final result panels for benchmarking how truncated gene spaces retain differential-expression and pathway signals in virtual-cell modeling.

## What is included

This repository is intentionally minimal. It contains only the scripts, source tables and final panels needed for the manuscript Fig. 1f-k result panels.

- `scripts/processing/`: upstream helper scripts for Seurat-dispersion HVG calculation and L1000 gene-set preparation.
- `scripts/fig1f_average_venn.py`: average DEG/HVG/L1000 Venn analysis.
- `scripts/fig1g_hvg_deg_boxplot.py`: HVG-DEG intersection boxplots.
- `scripts/fig1h_l1000_deg_boxplot.py`: L1000-DEG intersection boxplots.
- `scripts/fig1i_compute_hvg_kegg_pathway_recall.py` and `scripts/fig1i_plot_hvg_kegg_pathway_recall.py`: HVG KEGG pathway-recall analysis and plotting.
- `scripts/fig1j_compute_l1000_kegg_pathway_recall.py` and `scripts/fig1j_plot_l1000_kegg_pathway_recall.py`: L1000 KEGG pathway-recall analysis and plotting.
- `scripts/fig1k_plot_kegg_top20_pathway_overlap_heatmap.py`: combined L1000/HVG top-20 KEGG pathway-overlap heatmap.
- `source_data/`: minimal CSV tables behind Fig. 1f-k.
- `figures/`: final PDF result panels renamed by manuscript panel.

## Data

The analyses use aligned single-cell perturbation resources and cell-line perturbation comparisons from three datasets:

1. PerturBase-ORF, shown as `OTF` in analysis files.
2. PerturBase-Chemical, shown as `sciplex` in analysis files.
3. Tahoe-100M, shown as `tahoe` in analysis files.

Differential-expression sets were defined with FDR < 0.05 and absolute log2 fold change > 0.144. HVGs were calculated from the aligned transformed expression vectors using the study-specific streaming Seurat-dispersion implementation. L1000 landmark genes were mapped to Ensembl genes and used as directly measured landmark genes only.

Large raw expression matrices, complete DEG tables and full intermediate outputs are not tracked here. The minimal source tables in `source_data/` are sufficient to audit the final plotted values for Fig. 1f-k. To rerun the full pipeline from raw aligned resources, place the original input files in the paths expected by the upstream scripts or edit the `ROOT`/input-path constants in each script.

## Final result panels

| Panel | File |
| --- | --- |
| Fig. 1f | `figures/fig1f_perturbase_orf_average_venn.pdf`, `figures/fig1f_perturbase_chemical_average_venn.pdf`, `figures/fig1f_tahoe_100m_average_venn.pdf` |
| Fig. 1g | `figures/fig1g_hvg_deg_intersection_fraction_boxplot.pdf` |
| Fig. 1h | `figures/fig1h_l1000_deg_intersection_fraction_boxplot.pdf` |
| Fig. 1i | `figures/fig1i_hvg_kegg_pathway_recall_lineplot.pdf` |
| Fig. 1j | `figures/fig1j_l1000_kegg_pathway_recall_barplot.pdf` |
| Fig. 1k | `figures/fig1k_kegg_top20_pathway_overlap_heatmap.pdf` |

## Environment

Create the analysis environment with:

```bash
conda env create -f environment.yml
conda activate aivc-gene-space-benchmark
```

The original pathway scripts call `org.Hs.eg.db` and `AnnotationDbi` to export KEGG and Ensembl-to-Entrez mappings.
