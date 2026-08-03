import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from neuromaps import stats
from statsmodels.stats.multitest import multipletests

from prep import clr_features


def _spin_single(args):
    i, k, layer, ctype, x, y, spins = args
    r, p = stats.compare_images(x, y, nulls=x[spins],
                                nan_policy='omit', metric='pearsonr')
    return {'layer': layer, 'ctype': ctype, 'correlation': r, 'p_value': p}


def parallel_cross_layer_correlation(prop_mat, layer_CT, layers, ctypes, spins,
                                     use_clr=True, relative=True, mask=None,
                                     pseudocount=1e-8, fdr_alpha=0.05,
                                     max_workers=None, show_progress=True):
    """Spin-test correlation of layer composition vs. layer thickness."""
    n_region, n_layer, _ = prop_mat.shape
    assert layer_CT.shape == (n_region, n_layer), 'layer_CT shape mismatch'
    if mask is None:
        mask = pd.DataFrame(True, index=layers, columns=ctypes)
    mask = mask.astype(bool)

    X_layers = clr_features(prop_mat, layers, ctypes,
                            mask=mask if use_clr else None,
                            use_clr=use_clr, pseudocount=pseudocount)
    Y = layer_CT.div(layer_CT.sum(axis=1), axis=0) if relative else layer_CT
    Y = [Y.iloc[:, l].values for l in range(n_layer)]

    tasks = [(i, k, layer, ct, X_layers[i][ct].values, Y[i], spins)
             for i, layer in enumerate(layers)
             for k, ct in enumerate(ctypes) if mask.loc[layer, ct]]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_spin_single, t): n for n, t in enumerate(tasks)}
        it = as_completed(fut)
        if show_progress:
            it = tqdm(it, total=len(tasks), desc='spin')
        for f in it:
            results[fut[f]] = f.result()
    res = pd.DataFrame([results[n] for n in sorted(results)])

    ok = res['p_value'].notna()
    res.loc[ok, 'reject_H0'], res.loc[ok, 'p_adjusted'], _, _ = multipletests(
        res.loc[ok, 'p_value'].values, alpha=fdr_alpha, method='bonferroni')
    return res


def _compute_statistic(X_layers, Y_layers):
    """Mean Pearson correlation across all layers and (present) ctypes.

    X_layers[i] : DataFrame (n_region, n_ctype_present) - CLR features for layer i.
    Y_layers[i] : array (n_region,) - thickness (or other target) for layer i.
    """
    correlations = []
    for i in range(len(X_layers)):
        for ctype in X_layers[i].columns:
            r, _ = pearsonr(X_layers[i][ctype].values, Y_layers[i])
            correlations.append(r)
    return np.mean(correlations)


def _permuted_statistic(args):
    X_layers, Y_layers, perm = args
    permuted_Y = [Y_layers[perm[i]] for i in range(len(Y_layers))]
    return _compute_statistic(X_layers, permuted_Y)


def permutation_test_whole_match(X_layers, Y_layers, n_permutations=10000,
                                 seed=42, n_jobs=None, show_progress=True):
    """Permutation test for the whole-match statistic (random sampling).

    Uniformly samples random permutations (excluding the identity) to build
    the null distribution. With 10,000 permutations this is effectively exact
    for any reasonable alpha level.

    Parameters
    ----------
    X_layers, Y_layers : list of DataFrames / arrays
    n_permutations : int, default 10000
        Number of random permutations. Set to 720 (=6!) for exhaustive test.
    seed : int
        Random seed for reproducibility.
    """
    L = len(X_layers)
    assert L == len(Y_layers), 'X_layers / Y_layers length mismatch'

    rng = np.random.default_rng(seed)
    observed_stat = _compute_statistic(X_layers, Y_layers)

    # Generate random permutations (exclude identity)
    sampled_perms = []
    while len(sampled_perms) < n_permutations:
        perm = rng.permutation(L).tolist()
        if perm != list(range(L)):
            sampled_perms.append(perm)
    sampled_perms = sampled_perms[:n_permutations]

    null_distribution = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_permuted_statistic, (X_layers, Y_layers, perm))
                for perm in sampled_perms]
        it = as_completed(futs)
        if show_progress:
            it = tqdm(it, total=len(sampled_perms), desc='whole-match perm')
        for f in it:
            null_distribution.append(f.result())

    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_stat)

    return {
        'observed_stat': observed_stat,
        'p_value': p_value,
        'null_distribution': null_distribution,
        'n_permutations': len(sampled_perms),
    }


def _exact_mismatch_worker(R, i, k, ctype, perm):
    """Worker for exact mismatch test (receives pre-computed R matrix).

    R[i, j, k] = pearsonr(X_layers[i][ctype=k], Y_layers[j])
    """
    r_obs = float(R[i, i, k])
    perm_r = R[i, np.array(perm), k]
    n_extreme = int(np.sum(perm_r >= r_obs))
    # Unbiased one-sided p-value: (extreme + 1) / (n_perm + 1)
    p_value = (n_extreme + 1) / (len(perm) + 1)
    return {'layer': i, 'ctype': ctype, 'r_obs': r_obs,
            'p_value': p_value, 'n_perm_used': len(perm)}


def permutation_test_exact_mismatch(X_layers, Y_layers, ctypes, layers=None,
                                    mask=None, seed=42, n_permutations=5000,
                                    n_jobs=None, show_progress=True):
    """Random-sampling mismatch permutation test (pre-computed correlation matrix).

    Pre-computes the full 3D correlation tensor R[i, j, k] = pearsonr(X[i, k], Y[j])
    once, then samples random permutations (excluding identity) to build the null
    distribution for each (layer, ctype) cell. Avoids recomputing pearsonr across
    permutation tasks — the worker only does array lookups and comparisons.

    Parameters
    ----------
    X_layers, Y_layers : list of DataFrames / arrays
    ctypes : list[str]
    layers : list[str]
    mask : DataFrame(bool) or None
    seed : int
    n_permutations : int, default 5000
    """
    rng = np.random.default_rng(seed)
    L = len(X_layers)
    assert L == len(Y_layers), 'X_layers / Y_layers length mismatch'

    # Pre-compute 3D correlation tensor R[i, j, k]
    n_ctypes = len(ctypes)
    R = np.full((L, L, n_ctypes), np.nan, dtype=np.float64)
    for i in range(L):
        for j in range(L):
            for k, ct in enumerate(ctypes):
                if ct in X_layers[i].columns:
                    vals_x = X_layers[i][ct].values
                    vals_y = Y_layers[j]
                    R[i, j, k], _ = pearsonr(vals_x, vals_y)

    # Build task list
    tasks = []
    for i in range(L):
        for k, ct in enumerate(ctypes):
            if ct not in X_layers[i].columns:
                continue
            if mask is not None and layers is not None and not mask.loc[layers[i], ct]:
                continue
            tasks.append((i, k, ct))

    # Generate random permutations (exclude identity)
    sampled_perms = []
    while len(sampled_perms) < n_permutations:
        perm = rng.permutation(L).tolist()
        if perm != list(range(L)):
            sampled_perms.append(perm)
    sampled_perms = sampled_perms[:n_permutations]

    # Run permutation tests
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_exact_mismatch_worker, R, t[0], t[1], t[2], perm)
                for t in tasks for perm in sampled_perms]
        it = as_completed(futs)
        if show_progress:
            it = tqdm(it, total=len(tasks) * len(sampled_perms), desc='exact-mismatch')
        for f in it:
            results.append(f.result())

    df = pd.DataFrame(results)
    # Aggregate: median p-value across permutations per (layer, ctype)
    df = df.groupby(['layer', 'ctype']).agg({
        'r_obs': 'first',
        'p_value': 'median',
        'n_perm_used': 'first',
    }).reset_index()

    ok = df['p_value'].notna()
    df.loc[ok, 'reject_H0'], df.loc[ok, 'p_adjusted'], _, _ = multipletests(
        df.loc[ok, 'p_value'].values, alpha=0.05, method='bonferroni')
    return df
