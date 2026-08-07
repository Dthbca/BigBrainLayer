"""Cortical feature-map ~ cell-type-ratio correlation (MEG + ENIGMA disorders).

Organised from the exploratory notebook `CellAlign/spin_test.ipynb` (remote n03).
Tests whether the cortical distribution of cell-type ratios (from spatial
transcriptomics) couples with cortical feature maps, using spatial-
autocorrelation-preserving spin nulls. Two feature families share one pipeline
in the unified **BN (Brainnetome, 105 regions) atlas space**:

    --feature meg     HCP-S1200 MEG band power (neuromaps, fsLR→BN).
    --feature enigma  ENIGMA case-control cortical-thickness abnormality
                      (13 disorders, Nat. Neurosci. 2022 s41593-022-01186-3,
                      relabelled DK→BN, smoothed).

Both require --ratio-csv (subclass ratios in BN space, positional alignment).

Pipeline:
    load feature maps -> align to cell-type ratios in BN atlas space
      -> (MEG only) z-score outlier mask (drop < -2 SD)
      -> optional CLR transform of the (region x cell-type) composition
      -> per (feature x cell-type) spin correlation     [univariate]
      -> per-feature multivariate model R^2 + importance [multivariate]
           - dominance (exact Shapley, ≤12 features) or
           - SHAP (LinearExplainer, fast for 23 features)

Cell-type ratios are compositional, so CLR (`--use-clr`, default) is the primary
branch and raw proportions (`--no-clr`) the sensitivity branch; the two write to
distinct filenames. Multiple-comparison correction is applied ONCE across the
whole feature x cell-type grid (`--fdr-method`, default 'holm'), not per feature.

Outputs (written to --out-dir):
    <feature>_celltype_<tag>.csv  per (feature x cell-type) spin r / p / p_adj
    hcps1200_meg_fgc.csv          MEG feature matrix (MEG runs only)
    model_r_<tag>.csv             per-feature multivariate adjusted R^2 + spin p
    importance_<tag>.csv          per-cell-type relative importance (dominance/SHAP)
  where <tag> = <feature>_<group|all>_<clr|raw>.

Usage:
    # MEG
    python meg_celltype.py --feature meg --ratio-csv <subclass_ratio.csv> \
        --group glia --use-clr --fdr-method holm --n-spins 1000 --out-dir ./out
    # ENIGMA disorders (same ratio CSV, same BN space)
    python meg_celltype.py --feature enigma --ratio-csv <subclass_ratio.csv> \
        --use-clr --fdr-method holm --n-spins 1000 --out-dir ./out

Environment (remote n03):
    conda activate dthbca_imgT
    PYTHONPATH must include /data100/home/dthbca/project/CellAlign
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr, gmean
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- CellAlign / HomoloMap dependencies (package was renamed CellAlign→HomoloMap) ---
from neuromaps.datasets import fetch_annotation
try:
    from CellAlign.transforms import load_data
    from CellAlign.datasets import fetch_enigma, fetch_ctype_ratio, fetch_fslr, fetch_parc
    from CellAlign.parcellation import surf_relabel, parc_smooth
    from CellAlign.stats.nulls import SpinTest
    from CellAlign.stats import get_reg_r_sq, get_reg_r_pval
    from CellAlign.stats.analysis import get_dominance_stats
except ModuleNotFoundError:
    from HomoloMap.transforms import load_data
    from HomoloMap.datasets import fetch_enigma, fetch_ctype_ratio, fetch_fslr, fetch_parc
    from HomoloMap.parcellation import surf_relabel, parc_smooth
    from HomoloMap.stats.nulls import SpinTest
    from HomoloMap.stats import get_reg_r_sq, get_reg_r_pval
    from HomoloMap.stats.analysis import get_dominance_stats


# MEG bands published in HCP-S1200 (neuromaps `source='hcps1200'`).
MEG_BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']

# Cell-type functional groupings (column-name substrings in subclass_ratio).
CTYPE_GROUPS = {
    'glia': 'Astro|Oligo|OPC|Endo|VLMC|Micro-PVM',
    'in':   'Lamp5|Pvalb|Sst|Vip|Sncg|Pax6|Lamp5_Lhx6|Chandelier',
    'ex':   'L2/3 IT|L4 IT|L5 IT|L6 IT|L5 ET|L5/6 NP|L6 CT|L6b|L6 IT Car3',
}


# ---------------------------------------------------------------------------
# 1. Load MEG band maps and parcellate to the target atlas
# ---------------------------------------------------------------------------
def load_meg_bands(bands=MEG_BANDS, trg='BN', den='4k', mask_outliers=True,
                   z_thresh=-2.0):
    """Fetch HCP-S1200 MEG band maps, parcellate to `trg`, optionally mask.

    Returns a (n_region x n_band) DataFrame. Low outliers (z < `z_thresh`)
    are set to NaN when `mask_outliers` is True.
    """
    data_list = []
    for band in bands:
        path = fetch_annotation(source='hcps1200', desc=f'meg{band}',
                                space='fsLR', den=den)[0]
        parcelled = load_data(data=path, space='fslr', trg=trg, transform='True')
        data_list.append(parcelled.iloc[:, 0].values)

    data = pd.DataFrame(data_list).T
    data.columns = list(bands)

    if mask_outliers:
        data_z = zscore(data, axis=0, nan_policy='omit')
        data = data.mask(data_z < z_thresh)
    return data


def load_enigma_features(atlas='FGC', smooth=True, smooth_radius=8,
                         smooth_method='gaussian', smooth_sigma=5):
    """Fetch ENIGMA disorder cortical-thickness maps, relabel DK -> `atlas`, smooth.

    ENIGMA case-control cortical abnormality maps (13 disorders, DK atlas,
    Nat. Neurosci. 2022, s41593-022-01186-3). Native space is DK (34 L-hemi
    regions); we surface-relabel to the target atlas and (by default) Gaussian-
    smooth so the map sits in the same space as the cell-type ratios. Returns a
    (n_region x n_disorder) DataFrame.

    `smooth` needs Connectome Workbench (`wb_command`) on PATH for geodesic
    distances; set smooth=False (`--no-smooth`) in environments without it.
    """
    disease_parc = fetch_enigma(atlas='DK')
    disease = surf_relabel(data=disease_parc, src='DK', trg=atlas, method='mean')
    if smooth:
        disease = parc_smooth(disease, radius=smooth_radius, method=smooth_method,
                              sigma=smooth_sigma,
                              mesh=fetch_fslr(surf='inflated', return_path=True),
                              parc=fetch_parc(key=atlas))
    return disease


def load_ctype_ratio(path, group=None):
    """Load subclass cell-type ratio table; optionally slice to a functional group."""
    ratio = pd.read_csv(path, index_col=0)
    if group is not None:
        pattern = CTYPE_GROUPS[group]
        ratio = ratio.loc[:, ratio.columns.str.contains(pattern)]
    return ratio


def clr_transform(ratio, pseudocount=1e-8):
    """Standard CLR over the (region x cell-type) ratio table.

    Cell-type ratios are compositional (closure), so a plain Pearson/linear model
    over raw proportions induces spurious correlations. CLR maps each region's
    composition to log-ratios about its geometric mean, lifting the closure
    constraint. NB: this is the *unlayered* whole-region composition, so the CLR
    reference is the geometric mean over the columns passed in (all cell types, or
    a functional group when `--group` is set). Structural zeros are NOT removed;
    a `pseudocount` is added before the log (matches prep.coda_transform CLR).
    """
    data = ratio + pseudocount
    g = gmean(data, axis=1)
    return pd.DataFrame({c: np.log(data[c] / g) for c in ratio.columns},
                        index=ratio.index)


# ---------------------------------------------------------------------------
# Feature registry: bind each feature family to its atlas / loader / ratio src
# ---------------------------------------------------------------------------
def resolve_inputs(feature, atlas=None, ratio_csv=None, group=None,
                   ratio_level='subclass', smooth=True):
    """Load a feature matrix + its row-aligned cell-type ratios.

    Both feature families use the unified **BN (Brainnetome, 105 regions) atlas**
    and positional row alignment (via reset_index):

    feature='meg'    : HCP-S1200 MEG bands, parcellated fsLR→BN; ratios from
                       `ratio_csv` (must have 105 rows matching BN region order).
    feature='enigma' : ENIGMA disorder maps, relabelled DK→BN and smoothed; ratios
                       from `ratio_csv` (same BN space as MEG).

    Returns (data, ctype_ratio, atlas) with identical, positionally-aligned rows.
    """
    if feature == 'meg':
        atlas = atlas or 'BN'
        data = load_meg_bands(trg=atlas)
        if ratio_csv is None:
            raise ValueError("feature='meg' needs --ratio-csv (BN-space ratios)")
        ratio = load_ctype_ratio(ratio_csv, group=group)
        if len(data) != len(ratio):
            raise ValueError(f'MEG rows ({len(data)}) != ratio rows ({len(ratio)}); '
                             'expected same BN region order for positional align')
        return data.reset_index(drop=True), ratio.reset_index(drop=True), atlas

    if feature == 'enigma':
        atlas = atlas or 'BN'
        data = load_enigma_features(atlas=atlas, smooth=smooth)
        if ratio_csv is None:
            raise ValueError("feature='enigma' needs --ratio-csv (BN-space ratios)")
        ratio = load_ctype_ratio(ratio_csv, group=group)
        if len(data) != len(ratio):
            raise ValueError(f'ENIGMA rows ({len(data)}) != ratio rows ({len(ratio)}); '
                             'expected same BN region order for positional align')
        return data.reset_index(drop=True), ratio.reset_index(drop=True), atlas

    raise ValueError(f"unknown feature '{feature}' (use 'meg' or 'enigma')")


# ---------------------------------------------------------------------------
# 2. Univariate: per (feature x cell-type) spin correlation
# ---------------------------------------------------------------------------
def spin_correlation(data, ctype_ratio, atlas='BN', n_spins=1000,
                     method='Alexander-Bloch', n_jobs=-1, fdr_method='holm'):
    """Spin-test correlation for every (MEG band, cell type) pair.

    `data`        : (n_region x n_band) MEG feature matrix.
    `ctype_ratio` : (n_region x n_ctype) cell-type ratio matrix.

    Uses a spatial-autocorrelation-preserving null: cell-type maps are held
    fixed and the MEG map is rotated (`SpinTest.spins`). p is the fraction of
    null |r| >= observed |r|. NaNs (from CLR pseudocount edge cases or MEG
    outlier masking) are dropped pairwise before each correlation.

    Multiple-comparison correction is applied ONCE across the whole
    band x cell-type grid (not per band) via `fdr_method` -- the true test
    family is every cell tested. Returns a DataFrame indexed by cell type with
    `<band>_ratio_spin_{r,p,p_adj}` columns.
    """
    spinner = SpinTest(atlas=atlas, n_spins=n_spins, method=method)
    spins = spinner.spins

    def _one(ctype, band):
        x = ctype_ratio[ctype].values
        y = data[band].values
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3:
            return np.nan, np.nan
        r, _ = pearsonr(x[m], y[m])
        rotated = y[spins]                       # rotate full map, then mask per spin
        # A spin can rotate a NaN (MEG outlier mask) into a position the static
        # mask `m` keeps, so drop NaNs pairwise for EACH null realisation.
        r_null = np.empty(rotated.shape[1])
        for i in range(rotated.shape[1]):
            yi = rotated[:, i]
            mi = ~np.isnan(x) & ~np.isnan(yi)
            r_null[i] = pearsonr(x[mi], yi[mi])[0] if mi.sum() >= 3 else np.nan
        valid = ~np.isnan(r_null)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return r, np.nan
        p_value = (1 + np.sum(np.abs(r_null[valid]) >= np.abs(r))) / (n_valid + 1)
        return r, p_value

    bands = list(data.columns)
    ctypes = list(ctype_ratio.columns)
    r_grid, p_grid = {}, {}
    for band in tqdm(bands, desc='spin-corr'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one)(ct, band) for ct in ctypes)
        r_grid[band] = [res[0] for res in results]
        p_grid[band] = [res[1] for res in results]

    # --- unified FDR/FWER across the entire band x cell-type grid ---
    flat_p = np.array([p_grid[b][k] for b in bands for k in range(len(ctypes))])
    finite = ~np.isnan(flat_p)
    flat_adj = np.full_like(flat_p, np.nan)
    flat_adj[finite] = multipletests(flat_p[finite], method=fdr_method)[1]
    adj_grid = flat_adj.reshape(len(bands), len(ctypes))

    corr = pd.DataFrame(index=ctypes)
    for j, band in enumerate(bands):
        corr[f'{band}_ratio_spin_r'] = r_grid[band]
        corr[f'{band}_ratio_spin_p'] = p_grid[band]
        corr[f'{band}_ratio_spin_p_adj'] = adj_grid[j]
    return corr


# ---------------------------------------------------------------------------
# 3. Multivariate: per-band model R^2 + dominance
# ---------------------------------------------------------------------------
def model_r_sq_per_band(data, ctype_ratio, atlas='BN', n_spins=1000,
                        method='Alexander-Bloch', model_type='linear',
                        fdr_method='holm', importance_method='dominance'):
    """Multivariate adjusted R^2 (all cell types -> each band) with spin p.

    Model p-values are corrected across all features with `fdr_method`.

    importance_method: 'dominance' (exact, slow for >12 features) or 'shap'
                       (Shapley approximation via LinearExplainer, fast).
    """
    if importance_method == 'shap' and not SHAP_AVAILABLE:
        raise ImportError("SHAP requested but not installed. Run: pip install shap")
    if importance_method == 'dominance' and ctype_ratio.shape[1] > 12:
        print(f"[WARNING] dominance with {ctype_ratio.shape[1]} features is slow "
              f"(2^{ctype_ratio.shape[1]} subsets). Consider --importance-method shap")

    spinner = SpinTest(atlas=atlas, n_spins=n_spins, method=method)
    spins = spinner.spins
    X = zscore(ctype_ratio.values, nan_policy='omit')

    r_sq = np.full(len(data.columns), np.nan)
    pval = np.full(len(data.columns), np.nan)
    for i, band in enumerate(data.columns):
        y = zscore(data.values[:, i], nan_policy='omit')
        # `y` (and, defensively, X) can carry NaNs -- MEG outlier masking sets
        # extreme-low regions to NaN. get_reg_r_sq -> sklearn rejects NaNs, so
        # drop non-finite rows pairwise for the observed fit and, because the
        # spin rotates NaNs into new positions, per null realisation too.
        base = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if base.sum() < X.shape[1] + 2:
            continue
        r_sq[i] = get_reg_r_sq(X[base], y[base], model_type=model_type)
        null = np.empty(spins.shape[1])
        for k in range(spins.shape[1]):
            yk = y[spins[:, k]]
            mk = np.isfinite(X).all(axis=1) & np.isfinite(yk)
            null[k] = (get_reg_r_sq(X[mk], yk[mk], model_type=model_type)
                       if mk.sum() >= X.shape[1] + 2 else np.nan)
        null = null[np.isfinite(null)]
        if null.size:
            pval[i] = (1 + np.sum(null >= r_sq[i])) / (null.size + 1)

    finite = np.isfinite(pval)
    pval_adj = np.full_like(pval, np.nan)
    pval_adj[finite] = multipletests(pval[finite], method=fdr_method)[1]
    return pd.DataFrame({'feature': list(data.columns),
                         'model_r_sq': r_sq,
                         'model_pval': pval,
                         'model_pval_adj': pval_adj})


def dominance_per_band(data, ctype_ratio, band):
    """Per-cell-type total dominance for one MEG band (relative importance)."""
    X = zscore(ctype_ratio.values, nan_policy='omit')
    y = zscore(data[band].values, nan_policy='omit')
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)          # drop NaN regions
    stats = get_dominance_stats(X[m], y[m])
    return pd.Series(stats[0]['total_dominance'], index=ctype_ratio.columns)


def shap_importance_per_band(data, ctype_ratio, band):
    """Per-cell-type SHAP importance for one feature (relative contribution).

    Uses LinearExplainer (fast, exact for linear models). Returns normalized
    absolute SHAP values as relative importance (analogous to dominance analysis).
    """
    from sklearn.linear_model import LinearRegression
    X = zscore(ctype_ratio.values, nan_policy='omit')
    y = zscore(data[band].values, nan_policy='omit')

    # Remove NaN rows (from zscore nan_policy='omit' or original data)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean, y_clean = X[mask], y[mask]

    if len(X_clean) < 2:
        return pd.Series(np.nan, index=ctype_ratio.columns)

    model = LinearRegression().fit(X_clean, y_clean)
    explainer = shap.LinearExplainer(model, X_clean)
    shap_values = explainer.shap_values(X_clean)

    # Mean absolute SHAP per feature, normalized to sum=1 (relative importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    rel_importance = mean_abs_shap / mean_abs_shap.sum() if mean_abs_shap.sum() > 0 else mean_abs_shap
    return pd.Series(rel_importance, index=ctype_ratio.columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(feature='meg', ratio_csv=None, group=None, atlas=None, n_spins=1000,
        n_jobs=-1, out_dir='.', use_clr=True, fdr_method='holm', do_model=True,
        smooth=True, importance_method='auto'):
    os.makedirs(out_dir, exist_ok=True)

    data, ctype_ratio, atlas = resolve_inputs(
        feature, atlas=atlas, ratio_csv=ratio_csv, group=group, smooth=smooth)

    if feature == 'meg':                    # keep the raw feature matrix on disk
        data.to_csv(os.path.join(out_dir, 'hcps1200_meg_fgc.csv'))

    if use_clr:
        ctype_ratio = clr_transform(ctype_ratio)

    # Auto-select importance method: dominance for ≤12 features, shap for >12
    if importance_method == 'auto':
        importance_method = 'dominance' if ctype_ratio.shape[1] <= 12 else 'shap'

    # output tag keeps feature / group / raw-vs-CLR branches side by side
    tag = f"{feature}_{group or 'all'}_{'clr' if use_clr else 'raw'}"

    corr = spin_correlation(data, ctype_ratio, atlas=atlas, n_spins=n_spins,
                            n_jobs=n_jobs, fdr_method=fdr_method)
    corr.to_csv(os.path.join(out_dir, f'{feature}_celltype_{tag}.csv'))
    print(f'[meg_celltype] {tag}: spin correlations for '
          f'{ctype_ratio.shape[1]} cell types x {data.shape[1]} features '
          f'(atlas={atlas}, FDR/FWER={fdr_method}, unified over grid)')

    if do_model:
        model = model_r_sq_per_band(data, ctype_ratio, atlas=atlas,
                                    n_spins=n_spins, fdr_method=fdr_method,
                                    importance_method=importance_method)
        model.to_csv(os.path.join(out_dir, f'model_r_{tag}.csv'), index=False)
        print(f'[meg_celltype] {tag}: per-feature model R^2 -> model_r_{tag}.csv')

        # Per-cell-type importance (dominance or SHAP)
        print(f'[meg_celltype] {tag}: computing per-cell-type importance '
              f'(method={importance_method})...')
        importance_func = (dominance_per_band if importance_method == 'dominance'
                          else shap_importance_per_band)
        importance = pd.DataFrame({
            band: importance_func(data, ctype_ratio, band)
            for band in data.columns
        }).T
        importance.to_csv(os.path.join(out_dir, f'importance_{tag}.csv'))
        print(f'[meg_celltype] {tag}: per-cell-type importance ({importance_method}) '
              f'-> importance_{tag}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--feature', choices=['meg', 'enigma'], default='meg',
                    help="feature family: 'meg' (HCP-S1200 bands) or "
                         "'enigma' (disorder cortical-thickness maps); "
                         "both use BN atlas")
    ap.add_argument('--ratio-csv', required=True,
                    help='subclass cell-type ratio CSV (region x cell type) in BN space')
    ap.add_argument('--group', choices=list(CTYPE_GROUPS),
                    help='restrict to a functional group (glia/in/ex); '
                         'omit to use all cell types')
    ap.add_argument('--atlas', default=None,
                    help='target atlas; defaults per feature (meg->BN, enigma->FGC)')
    ap.add_argument('--n-spins', type=int, default=1000)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--out-dir', default='.')
    clr = ap.add_mutually_exclusive_group()
    clr.add_argument('--use-clr', dest='use_clr', action='store_true',
                     help='CLR-transform the ratios (default; primary branch)')
    clr.add_argument('--no-clr', dest='use_clr', action='store_false',
                     help='use raw proportions (sensitivity branch)')
    ap.set_defaults(use_clr=True)
    ap.add_argument('--fdr-method', default='holm',
                    help="multipletests method, applied ONCE over the whole "
                         "band x cell-type grid (default 'holm'; e.g. 'fdr_bh', "
                         "'bonferroni')")
    ap.add_argument('--no-smooth', dest='smooth', action='store_false',
                    help='skip Gaussian surface smoothing of ENIGMA maps '
                         '(needed where Connectome Workbench / wb_command is '
                         'unavailable); no effect for --feature meg')
    ap.set_defaults(smooth=True)
    ap.add_argument('--importance-method', choices=['auto', 'dominance', 'shap'],
                    default='auto',
                    help="method for per-cell-type importance in multivariate model: "
                         "'dominance' (exact Shapley, slow for >12 features), "
                         "'shap' (fast LinearExplainer approximation), "
                         "'auto' (dominance if ≤12 features else shap, default)")
    ap.add_argument('--no-model', action='store_true',
                    help='skip the multivariate model R^2 step')
    args = ap.parse_args()

    run(feature=args.feature, ratio_csv=args.ratio_csv, group=args.group,
        atlas=args.atlas, n_spins=args.n_spins, n_jobs=args.n_jobs,
        out_dir=args.out_dir, use_clr=args.use_clr, fdr_method=args.fdr_method,
        do_model=not args.no_model, smooth=args.smooth,
        importance_method=args.importance_method)


if __name__ == '__main__':
    main()
