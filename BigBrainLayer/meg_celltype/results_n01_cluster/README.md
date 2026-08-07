# Cluster-level (71 cell types) MEG + ENIGMA × cell-type-ratio results

Companion to the subclass-level results in `../results_n01/`. Same pipeline
(`meg_celltype.py`), same BN (Brainnetome, 105 L-hemi regions) atlas space, same
spin-test / multivariate machinery — but the cell-type ratios are at the finer
**cluster** level (`Astro_1`, `Pvalb_4`, `L2/3 IT_6`, … — 71 clusters) instead of
the 23 subclasses.

## How `cluster_ratio.csv` was built

The pipeline consumes a BN-space `region × cell-type` ratio table. The cluster
version was constructed with the canonical CellAlign/HomoloMap recipe (matches
`spin_test.ipynb` Cell 93 + the FGC→BN relabel used everywhere for non-FGC atlases):

1. Read the raw per-vertex ratio table `ctype_ratio_plot_FGC.csv`
   (3670 FGC vertices × 226 raw cluster IDs, e.g. `ASC.1`, `L2/3.4`).
2. Aggregate raw IDs → clusters with `ctype_ratio_agg(fgc, map_df, key='cluster')`,
   where `map_df` is `cluster_mapping_dict.csv` (`plot` → `cluster`, e.g.
   `ASC.1 → Astro_1`). Columns are summed within each cluster; 35 raw IDs with no
   cluster assignment are dropped (so per-region mass sums to ≈0.88–0.91, not 1.0).
   → 3670 × 71.
3. Relabel FGC→BN region space: `surf_relabel(src='FGC', trg='BN', method='mean')`.
   → **105 × 71**, index = BN L-hemi odd IDs `[1,3,5,…]`, row-aligned to MEG/ENIGMA.

Note: the D99 file the user first pointed at (`ctype_ratio_plot_D99.csv`) is the
**macaque** counterpart of the same raw table; the human FGC file is the correct
source for the human BN-space maps (D99→BN cross-species relabel gives ~0 spatial
correlation with the committed subclass ratios, FGC→BN reproduces them at r≈0.95).

Reproduce with `build_cluster_ratio.py` (in this dir) on a node with the
`dthbca_imgT` env + `wb_command` on PATH.

## Files

- `cluster_ratio.csv` — the 105×71 BN-space cluster ratio matrix (the input).
- `{enigma,meg}_celltype_*_{clr,raw}.csv` — univariate spin-correlation grid
  (feature × cluster: r / p / holm-adjusted p, unified over the whole grid).
- `model_r_*_{clr,raw}.csv` — per-feature multivariate adjusted R² + spin p (holm).
- `importance_*_{clr,raw}.csv` — per-cluster SHAP relative importance (auto-selected
  because 71 > 12 features; dominance is intractable at this width).
- `hcps1200_meg_fgc.csv` — the MEG band feature matrix (BN space).

## Results summary (n_spins=1000, holm FDR within each feature family)

Multivariate model R² per feature (spin-test p on adjusted R²):

- **MEG / CLR**: theta (R²=0.97, p_adj=0.012), beta (0.96, 0.015), delta (0.95, 0.020)
  survive; alpha / gamma1 near-miss (p_adj≈0.11).
- **MEG / raw**: gamma1 (R²=0.94, p_adj=0.036) survives; others p_adj≈0.16.
- **ENIGMA / CLR**: ASD is the top hit (R²=0.93) but just misses (p_adj=0.052);
  Parkinson next (p_adj=0.108).
- **ENIGMA / raw**: nothing survives (22q best, p_adj=0.169).

Univariate per-(feature × cluster) spin correlations: **0 cells survive holm**
in any branch (the correction is over 71×6=426 MEG or 71×13=923 ENIGMA cells).

### Caveats

- With **71 predictors and only 105 regions** the multivariate model is near
  saturation. Adjusted R² penalises this, and the spin-test null refits the same
  71-feature model on each rotated map, so a significant spin-p means the *observed*
  spatial alignment beats rotated nulls beyond what saturation alone explains — but
  the absolute R² values (0.9+) are not evidence of a strong effect on their own.
- MEG survives at cluster level (theta/beta/delta CLR) where it did **not** at
  subclass level. Treat this as a lead, not a confirmed result: the extra predictors
  add spatial degrees of freedom that can inflate the fitted-vs-rotated gap. The
  disappearance of every univariate hit under holm is the more conservative read.
- Model-R² FDR is applied **within** each feature family (13 disorders, or 6 bands),
  not jointly across families or across the univariate grid.
