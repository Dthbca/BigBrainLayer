"""MEG frequency-band power ~ cell-type-ratio correlation.

Organised from the exploratory notebook `CellAlign/spin_test.ipynb` (remote n03).
Tests whether the cortical distribution of cell-type ratios (from spatial
transcriptomics, projected to the Brainnetome / BN atlas) couples with HCP-S1200
MEG band-power maps, using spatial-autocorrelation-preserving spin nulls.

Pipeline:
    fetch MEG bands (neuromaps, HCP-S1200, fsLR 4k)
      -> parcellate to BN atlas
      -> z-score outlier mask (drop < -2 SD)
      -> optional CLR transform of the (region x cell-type) composition
      -> per (band x cell-type) spin correlation      [univariate]
      -> per-band multivariate model R^2 + dominance   [multivariate]

Cell-type ratios are compositional, so CLR (`--use-clr`, default) is the primary
branch and raw proportions (`--no-clr`) the sensitivity branch; the two write to
distinct filenames. Multiple-comparison correction is applied ONCE across the
whole band x cell-type grid (`--fdr-method`, default 'holm'), not per band.

Outputs (written to --out-dir):
    meg_celltype_<tag>.csv  per (band x cell-type) spin r / p / p_adj
    hcps1200_meg_fgc.csv    the parcellated, outlier-masked MEG feature matrix
    model_r_<tag>.csv       per-band multivariate adjusted R^2 + spin p
  where <tag> = <group|all>_<clr|raw>.

Usage:
    python meg_celltype.py --ratio-csv <subclass_ratio.csv> --group glia \
        --use-clr --fdr-method holm --n-spins 1000 --n-jobs -1 --out-dir ./out

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

# --- CellAlign dependencies (available in the dthbca_imgT env) ---------------
from neuromaps.datasets import fetch_annotation
from CellAlign.transforms import load_data
from CellAlign.stats.nulls import SpinTest
from CellAlign.stats import get_reg_r_sq, get_reg_r_pval
from CellAlign.stats.analysis import get_dominance_stats


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
# 2. Univariate: per (band x cell-type) spin correlation
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
        r, _ = pearsonr(x[m], y[m])
        rotated = y[spins]                       # rotate full map, then mask per spin
        r_null = np.array([pearsonr(x[m], rotated[m, i])[0]
                           for i in range(rotated.shape[1])])
        p_value = (1 + np.sum(np.abs(r_null) >= np.abs(r))) / (rotated.shape[1] + 1)
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
                        fdr_method='holm'):
    """Multivariate adjusted R^2 (all cell types -> each band) with spin p.

    Model p-values are corrected across the 6 bands with `fdr_method`.
    """
    spinner = SpinTest(atlas=atlas, n_spins=n_spins, method=method)
    X = zscore(ctype_ratio.values, nan_policy='omit')

    r_sq = np.zeros(len(data.columns))
    pval = np.zeros(len(data.columns))
    for i, band in enumerate(data.columns):
        y = zscore(data.values[:, i], nan_policy='omit')
        r_sq[i] = get_reg_r_sq(X, y, model_type=model_type)
        pval[i] = get_reg_r_pval(X, y, spinner.spins, n_spins, model_type=model_type)

    pval_adj = multipletests(pval, method=fdr_method)[1]
    return pd.DataFrame({'feature': list(data.columns),
                         'model_r_sq': r_sq,
                         'model_pval': pval,
                         'model_pval_adj': pval_adj})


def dominance_per_band(data, ctype_ratio, band):
    """Per-cell-type total dominance for one MEG band (relative importance)."""
    stats = get_dominance_stats(zscore(ctype_ratio.values),
                                zscore(data[band].values))
    return pd.Series(stats[0]['total_dominance'], index=ctype_ratio.columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(ratio_csv, group=None, atlas='BN', n_spins=1000, n_jobs=-1,
        out_dir='.', use_clr=True, fdr_method='holm', do_model=True):
    os.makedirs(out_dir, exist_ok=True)

    data = load_meg_bands(trg=atlas)
    data.to_csv(os.path.join(out_dir, 'hcps1200_meg_fgc.csv'))

    ctype_ratio = load_ctype_ratio(ratio_csv, group=group)
    if use_clr:
        ctype_ratio = clr_transform(ctype_ratio)

    # output tag keeps the raw and CLR branches side by side (no overwrite)
    tag = f"{group or 'all'}_{'clr' if use_clr else 'raw'}"

    corr = spin_correlation(data, ctype_ratio, atlas=atlas, n_spins=n_spins,
                            n_jobs=n_jobs, fdr_method=fdr_method)
    corr.to_csv(os.path.join(out_dir, f'meg_celltype_{tag}.csv'))
    print(f'[meg_celltype] {tag}: spin correlations for '
          f'{ctype_ratio.shape[1]} cell types x {data.shape[1]} bands '
          f'(FDR/FWER={fdr_method}, unified over grid)')

    if do_model:
        model = model_r_sq_per_band(data, ctype_ratio, atlas=atlas,
                                    n_spins=n_spins, fdr_method=fdr_method)
        model.to_csv(os.path.join(out_dir, f'model_r_{tag}.csv'), index=False)
        print(f'[meg_celltype] {tag}: per-band model R^2 -> model_r_{tag}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ratio-csv', required=True,
                    help='subclass cell-type ratio CSV (region x cell type)')
    ap.add_argument('--group', choices=list(CTYPE_GROUPS),
                    help='restrict to a functional group (glia/in/ex); '
                         'omit to use all cell types')
    ap.add_argument('--atlas', default='BN')
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
    ap.add_argument('--no-model', action='store_true',
                    help='skip the multivariate model R^2 step')
    args = ap.parse_args()

    run(ratio_csv=args.ratio_csv, group=args.group, atlas=args.atlas,
        n_spins=args.n_spins, n_jobs=args.n_jobs, out_dir=args.out_dir,
        use_clr=args.use_clr, fdr_method=args.fdr_method,
        do_model=not args.no_model)


if __name__ == '__main__':
    main()
