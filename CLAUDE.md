# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this experiment does

Tests whether cortical layer cell-type composition (from spatial transcriptomics) couples with BigBrain cortical layer thickness across brain regions. The null hypothesis is that cell-type profiles and thickness co-vary by layer identity — i.e., Layer III composition predicts Layer III thickness better than it predicts any other layer's thickness.

## Running the pipeline

Single run (default: BN atlas, subclass level, external mask):

```
cd BigBrainLayer/script
python main.py --atlas BN --level subclass --mask external --n-spins 1000 --n-jobs 20 --out-dir ../tmpres
```

Sweep all 4 pipeline branches (CLR vs raw × relative vs absolute thickness):

```
python run_branches.py --atlas BN --level subclass --n-spins 1000 --n-jobs 20 --out-dir ../tmpres/branches
```

Outputs per run: `spin_test.csv`, `exact_mismatch.csv`, `whole_match.csv`, `dominance.csv`, `spin_heatmap.png`, `exact_mismatch_heatmap.png`, `dominance_heatmap.png`.

Add `--skip-contribution` to skip the dominance analysis (it's the slowest stage for quick tests):

```
python main.py --atlas BN --level subclass --n-spins 100 --n-jobs 20 --skip-contribution --out-dir ../tmpres
```

## Pipeline architecture

`dataset.py → prep.py → analysis.py → plotting.py`, plus `contribution.py` for the overall-contribution analysis, wired by `main.py`.

**dataset.py** — all I/O. Key function: `load_all()` returns a dict with `prop_mat` (region×layer×ctype proportions), `count_arr` (raw counts), `layer_CT` (BigBrain thickness), and `mask` (laminar presence mask). Atlas relabeling happens here: D99→BN for cell counts via `vol_relabel(..., cross_species=True)`, FGC→BN for thickness.

**prep.py** — compositional transforms (CLR/ALR/ILR). `clr_features()` computes per-layer CLR over the subcomposition of present cell types only (absent cell types are dropped before the geometric mean, not treated as zeros). This is the critical design choice — structural zeros must not pollute the CLR.

**analysis.py** — three independent statistical tests, all parallelised with `ProcessPoolExecutor`:
- `parallel_cross_layer_correlation`: spin-test (spatial autocorrelation null via `neuromaps.stats.compare_images`) for each (layer, ctype) pair
- `permutation_test_whole_match`: permutes layer order for the whole 6×23 composition-thickness match
- `permutation_test_exact_mismatch`: exact enumeration of all L! permutations excluding self-match, per (layer, ctype) cell

All tests apply Bonferroni FDR correction across the tested (layer, ctype) cells.

**contribution.py** — overall-contribution analysis: how much of each layer's thickness variance is explained by its cell-type composition, and which cell types drive it. Wraps CellAlign's dominance analysis (Azen & Budescu), fit per layer (never cross-layer, to respect the same structural constraint as `analysis.py`):
- `layer_dominance_analysis(X_layers, Y_layers, layers, use_adjusted_r_sq=True, method='auto', max_features=15, n_samples=10000, n_jobs=-1)`: for each layer, calls `CellAlign.stats.analysis.get_dominance_stats` on that layer's CLR features vs its thickness. Returns a long-format `DataFrame` (`layer`, `ctype`, `total_dominance`, `R2_full`) where `total_dominance` sums exactly to `R2_full` across ctypes within a layer — this is what dominance analysis guarantees (verified on n03: 100.00% match in all 6 layers).
- `method='auto'` picks exhaustive dominance (all 2^p-1 feature subsets) when a layer has ≤`max_features` present cell types, and Monte Carlo approximate dominance (`n_samples` sampled subsets) above that. In the real dataset (external mask, subclass level), per-layer feature counts are Layer I=8, II=13, III=22, IV=18, V=19, VI=17 — so Layers I-II run exhaustive and III-VI fall back to approximate.
- `plot_dominance_heatmap(data, mask=None, ...)`: renders the layer x ctype dominance matrix as a heatmap (own renderer, not `plotting.plot_layer_heatmap`).
- `layer_shap_analysis` / `get_shap_stats`-based SHAP attribution also exists in this module but is unvalidated — only the dominance path has been tested end-to-end.
- Wired into `main.py` as the `run_contribution` step (default on); `--skip-contribution` skips it. Writes `dominance.csv` and `dominance_heatmap.png` when `run_contribution=True`.

**run_branches.py** — sweeps `use_clr ∈ {True, False}` × `relative ∈ {True, False}` (4 branches), fixing `mask_kind='external'`. Produces a `branch_summary.csv` and comparison bar chart.

## Dataset layout

```
BigBrainLayer/dataset/
  Spatial/
    raw_counts_d99.npy          # np array payload: counts (n_region, n_layer, n_ctype), ctypes, regions
    cluster_mapping_dict.csv    # raw ctype → subclass/class hierarchy mapping
    mask_by_nc2025.csv          # external laminar presence mask (literature-curated)
  BigBrain/
    layer_thickness_parced.csv  # per-layer thickness in FGC atlas space (relabeled to BN at load time)
```

## Key parameters and their effects

- `--atlas`: target atlas after relabeling (default `BN`; `D99` skips relabeling for cell counts)
- `--level`: cell-type aggregation level passed to `ctype_ratio_agg` (default `subclass`)
- `--mask external|enrichment`: `external` uses the nc2025 curated laminar mask; `enrichment` uses a data-driven threshold (>5% of a ctype's total mass in that layer)
- `use_clr`: whether to CLR-transform proportions before correlation — primary analysis uses `True`
- `relative`: whether to row-normalise thickness (relative layer thickness) — primary analysis uses `True`

## Notebook

`BigBrain_layer.ipynb` is an exploratory counterpart to the script pipeline. Paths inside are hardcoded to a Linux HPC (`/data100/home/dthbca/`) and must be adjusted for local execution.

## Debugging

**Remote execution on n03:**
- Project files: `/share/user_data/dthbca/public/experiment/BigBrainLayer/`
- Dataset root: `/share/user_data/dthbca/public/experiment/BigBrainLayer/dataset`
- CellAlign: `/data100/home/dthbca/project/CellAlign/`
- Conda env: `dthbca_imgT`
- Always set `BIGBRAIN_DATASET_ROOT` when running from a different directory
- Always set `PYTHONPATH` to include CellAlign

## Feature Extraction Comparison

`compare_features.py` sweeps `prop_mode` (by_layer/by_region) × `prop_order` (before/after)  = 4 branches. Run with:

```
python compare_features.py --out-dir ../tmpres/prop_feature_compare
```

Key question: `by_region` captures both cell-type composition AND layer thickness; `by_layer` captures only composition. Compare results to see which drives the layer-specificity signal.
