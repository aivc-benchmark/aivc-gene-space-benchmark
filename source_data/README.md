# Source data

This folder contains the minimal source tables needed for the final Fig. 1f-k panels.

- `fig1f_average_venn_summary.csv`: average mutually exclusive DEG/HVG/L1000 intersection sizes for dataset-level Seurat-dispersion top-3000 HVGs.
- `fig1g_hvg_deg_intersection_per_comparison.csv`: per-comparison HVG-DEG overlap fractions used for Fig. 1g; filtered to Seurat-dispersion, dataset-level, top-3000 HVGs.
- `fig1g_hvg_deg_intersection_summary.csv`: summary of HVG-DEG overlap boxplots.
- `fig1h_l1000_deg_intersection_per_comparison.csv`: per-comparison L1000-DEG overlap fractions used for Fig. 1h; filtered to dataset-level top-2000 rows and L1000-DEG intersections.
- `fig1h_l1000_deg_intersection_summary.csv`: summary of L1000-DEG overlap boxplots.
- `fig1i_hvg_kegg_pathway_recall_values.csv`: median and IQR values for significant KEGG pathway recall across HVG cutoffs.
- `fig1j_l1000_kegg_pathway_recall_values.csv`: median and IQR values for significant KEGG pathway recall by L1000 subsets.
- `fig1k_kegg_top20_pathway_overlap_values.csv`: values displayed in the KEGG top-20 pathway-overlap heatmap.

Raw aligned expression matrices, complete DEG tables and full intermediate outputs are intentionally not included in this repository. See the main README for data-source descriptions.
