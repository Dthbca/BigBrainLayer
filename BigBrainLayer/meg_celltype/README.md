# MEG + ENIGMA disorder × cell-type-ratio correlation

Organised from the exploratory notebook `CellAlign/spin_test.ipynb` (remote n03,
`/data100/home/dthbca/project/CellAlign/`). Tests whether the cortical
distribution of transcriptomic cell-type ratios couples with cortical feature
maps (MEG band-power or ENIGMA disorder cortical-thickness abnormalities), using
spatial-autocorrelation-preserving spin nulls.

## Feature families

Two feature families share one correlation pipeline, differing only in atlas
space and data source:

| `--feature` | atlas | features | cell-type ratios | alignment |
|-------------|-------|----------|------------------|-----------|
| `meg` (default) | BN (105 regions) | HCP-S1200 MEG band power (6 bands: delta/theta/alpha/beta/gamma1/gamma2, neuromaps) | user CSV (`--ratio-csv`) in BN space | positional (rows must match) |
| `enigma` | FGC (3670 parcels) | ENIGMA case-control cortical-thickness abnormality (13 disorders, Nat. Neurosci. 2022 s41593-022-01186-3, relabelled DK→FGC, Gaussian-smoothed) | `fetch_ctype_ratio(level='subclass')` in FGC space | index intersection |

## Pipeline

1. **Fetch feature maps** — MEG: `neuromaps.datasets.fetch_annotation(source='hcps1200',
   desc='meg<band>', space='fsLR', den='4k')`, parcellate to BN. ENIGMA:
   `fetch_enigma(atlas='DK')`, relabel DK→FGC, Gaussian-smooth (radius=8, σ=5).
2. **Load cell-type ratios** — MEG: from `--ratio-csv` (BN space, positional align).
   ENIGMA: `fetch_ctype_ratio(level='subclass')` (FGC space, index align).
3. **Outlier mask (MEG only)** — z-score per band, drop values < −2 SD (set to NaN).
4. **CLR (optional, default on)** — cell-type ratios are compositional (closure),
   so a plain correlation over raw proportions induces spurious anti-correlations.
   `--use-clr` maps each region's whole-region composition to log-ratios about its
   geometric mean. This is the *unlayered* composition, so the CLR reference is the
   geometric mean over whatever columns are in scope (all cell types, or the
   `--group` subset). Standard CLR with a pseudocount — structural zeros are **not**
   removed. `--no-clr` keeps raw proportions as a sensitivity branch.
5. **Univariate spin correlation** — for each (feature × cell type): Pearson r plus a
   spin p-value from `CellAlign.stats.nulls.SpinTest` (Alexander-Bloch rotations).
   Correction is applied **once across the entire feature × cell-type grid** (the true
   test family), not per feature — `--fdr-method` (default `holm`, FWER; matches the
   thickness pipeline's FWER control). → `<feature>_celltype_<tag>.csv`.
6. **Multivariate** — per feature, adjusted R² of the full cell-type set
   (`get_reg_r_sq` / `get_reg_r_pval`, corrected across all features) and
   per-cell-type `get_dominance_stats` relative importance. → `model_r_<tag>.csv`.

## Cell-type groups

`--group` slices the ratio table by column-name substring:

| group | cell types |
|-------|------------|
| `glia` | Astro, Oligo, OPC, Endo, VLMC, Micro-PVM |
| `in`   | Lamp5, Pvalb, Sst, Vip, Sncg, Pax6, Lamp5_Lhx6, Chandelier (inhibitory) |
| `ex`   | L2/3 IT, L4 IT, L5 IT, L6 IT, L5 ET, L5/6 NP, L6 CT, L6b, L6 IT Car3 (excitatory) |

Omit `--group` to use all cell types.

## Running (remote n03)

```bash
conda activate dthbca_imgT
export PYTHONPATH=/data100/home/dthbca/project/CellAlign:$PYTHONPATH

# MEG (BN space; primary CLR branch)
python meg_celltype.py --feature meg \
    --ratio-csv /data100/home/dthbca/project/CellAlign/tmp/subclass_ratio.csv \
    --group glia --use-clr --fdr-method holm \
    --n-spins 1000 --n-jobs -1 --out-dir ./out

# MEG (sensitivity raw branch)
python meg_celltype.py --feature meg \
    --ratio-csv /data100/home/dthbca/project/CellAlign/tmp/subclass_ratio.csv \
    --group glia --no-clr --fdr-method holm \
    --n-spins 1000 --n-jobs -1 --out-dir ./out

# ENIGMA disorders (FGC space; ratios fetched automatically)
python meg_celltype.py --feature enigma \
    --use-clr --fdr-method holm \
    --n-spins 1000 --n-jobs -1 --out-dir ./out

# ENIGMA without smoothing (if wb_command unavailable in your shell)
python meg_celltype.py --feature enigma --no-smooth \
    --use-clr --fdr-method holm \
    --n-spins 1000 --n-jobs -1 --out-dir ./out
```

## Outputs

`<tag>` = `<feature>_<group|all>_<clr|raw>`.

| file | contents |
|------|----------|
| `<feature>_celltype_<tag>.csv` | per (feature × cell type) `spin_r`, `spin_p`, `spin_p_adj` (grid-wide correction) |
| `hcps1200_meg_fgc.csv` | parcellated, outlier-masked MEG feature matrix (MEG runs only) |
| `model_r_<tag>.csv` | per-feature adjusted R² + spin p (multivariate) |

## Dependencies

`CellAlign` (transforms, stats.nulls, stats.analysis), `neuromaps`, `numpy`,
`pandas`, `scipy`, `statsmodels`, `joblib`, `tqdm`. All present in the
`dthbca_imgT` conda env.
