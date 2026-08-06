# MEG × cell-type-ratio correlation

Organised from the exploratory notebook `CellAlign/spin_test.ipynb` (remote n03,
`/data100/home/dthbca/project/CellAlign/`). Tests whether the cortical
distribution of transcriptomic cell-type ratios couples with HCP-S1200 MEG
band-power maps, using spatial-autocorrelation-preserving spin nulls.

## Pipeline

1. **Fetch MEG bands** — `neuromaps.datasets.fetch_annotation(source='hcps1200',
   desc='meg<band>', space='fsLR', den='4k')` for delta, theta, alpha, beta,
   gamma1, gamma2.
2. **Parcellate** — `CellAlign.transforms.load_data(..., trg='BN')` projects each
   surface map to the Brainnetome atlas (region × band matrix).
3. **Outlier mask** — z-score per band, drop values < −2 SD (set to NaN).
4. **Univariate spin correlation** — for each (band × cell type): Pearson r plus a
   spin p-value from `CellAlign.stats.nulls.SpinTest` (Alexander-Bloch rotations),
   FDR-corrected per band. → `meg_celltype.csv`.
5. **Multivariate** — per band, adjusted R² of the full cell-type set
   (`get_reg_r_sq` / `get_reg_r_pval`) and per-cell-type `get_dominance_stats`
   relative importance. → `model_r_<group>.csv`.

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

python meg_celltype.py \
    --ratio-csv /data100/home/dthbca/project/CellAlign/tmp/subclass_ratio.csv \
    --group glia --n-spins 1000 --n-jobs -1 --out-dir ./out
```

## Outputs

| file | contents |
|------|----------|
| `meg_celltype.csv` | per (band × cell type) `spin_r`, `spin_p`, `spin_p_adj` |
| `hcps1200_meg_fgc.csv` | parcellated, outlier-masked MEG feature matrix |
| `model_r_<group>.csv` | per-band adjusted R² + spin p (multivariate) |

## Dependencies

`CellAlign` (transforms, stats.nulls, stats.analysis), `neuromaps`, `numpy`,
`pandas`, `scipy`, `statsmodels`, `joblib`, `tqdm`. All present in the
`dthbca_imgT` conda env.
