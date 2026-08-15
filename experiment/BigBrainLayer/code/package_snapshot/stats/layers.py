"""Spatial and layer-label permutation tests for laminar maps."""

from itertools import permutations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


def layer_spin_correlation(features, thickness, spinner, present_mask=None,
                           metric='pearsonr', correction='fdr_bh', n_jobs=1):
    """Test each layer/cell-type map against matching-layer thickness."""
    if metric not in {'pearson', 'pearsonr', 'spearman', 'spearmanr'}:
        raise ValueError("metric must be pearsonr or spearmanr")
    tasks = []
    for layer, frame in features.items():
        target = _target(thickness, layer)
        for cell_type in frame.columns:
            if present_mask is not None and not present_mask.loc[
                    _mask_layer(layer, present_mask), cell_type]:
                continue
            tasks.append((layer, cell_type, frame[cell_type], target))

    def calculate(task):
        layer, cell_type, x, y = task
        r, p = spinner.correlation(x, y, metric=metric)
        return layer, cell_type, r, p

    rows = Parallel(n_jobs=n_jobs)(delayed(calculate)(task) for task in tasks)
    result = pd.DataFrame(rows, columns=['layer', 'ctype', 'correlation', 'p_value'])
    valid = result.p_value.notna()
    result['reject_H0'] = False
    result['p_adjusted'] = np.nan
    if valid.any():
        reject, adjusted, _, _ = multipletests(
            result.loc[valid, 'p_value'], method=correction)
        result.loc[valid, 'reject_H0'] = reject
        result.loc[valid, 'p_adjusted'] = adjusted
    result.attrs['correction'] = correction
    return result


def layer_match_permutation(features, thickness, scheme='whole',
                            alternative='greater', correction='fdr_bh',
                            random_state=None, n_permutations=None):
    """Test whether cell-type layer labels specifically match thickness layers."""
    layers = list(features)
    if len(layers) != thickness.shape[1]:
        raise ValueError("features and thickness must contain the same number of layers")
    permutations_all = list(permutations(range(len(layers))))
    if n_permutations is not None and n_permutations < len(permutations_all):
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(permutations_all), n_permutations, replace=False)
        permutations_all = [permutations_all[i] for i in chosen]
    if scheme == 'whole':
        observed = _mean_match(features, thickness, tuple(range(len(layers))))
        null = np.array([_mean_match(features, thickness, p) for p in permutations_all])
        return {'observed_stat': observed, 'p_value': _empirical_p(null, observed, alternative),
                'null_distribution': null, 'n_permutations': len(null)}
    if scheme != 'mismatch':
        raise ValueError("scheme must be 'whole' or 'mismatch'")
    rows = []
    for i, layer in enumerate(layers):
        for cell_type in features[layer].columns:
            x = features[layer][cell_type]
            observed = _safe_corr(x, thickness.iloc[:, i])
            null = np.array([_safe_corr(x, thickness.iloc[:, p[i]])
                             for p in permutations_all if p[i] != i])
            rows.append((layer, cell_type, observed,
                         _empirical_p(null, observed, alternative), len(null)))
    result = pd.DataFrame(rows, columns=['layer', 'ctype', 'r_obs', 'p_value', 'n_perm_used'])
    valid = result.p_value.notna()
    result['p_adjusted'] = np.nan
    result['reject_H0'] = False
    if valid.any():
        reject, adjusted, _, _ = multipletests(result.loc[valid, 'p_value'], method=correction)
        result.loc[valid, 'reject_H0'] = reject
        result.loc[valid, 'p_adjusted'] = adjusted
    return result


def _mean_match(features, thickness, permutation):
    values = []
    for i, layer in enumerate(features):
        target = thickness.iloc[:, permutation[i]]
        values.extend(_safe_corr(features[layer][c], target) for c in features[layer].columns)
    return float(np.nanmean(values))


def _safe_corr(x, y):
    x, y = pd.Series(x).align(pd.Series(y), join='inner')
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def _empirical_p(null, observed, alternative):
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return np.nan
    if alternative == 'greater':
        extreme = null >= observed
    elif alternative == 'less':
        extreme = null <= observed
    elif alternative == 'two-sided':
        extreme = np.abs(null) >= abs(observed)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return (int(extreme.sum()) + 1) / (null.size + 1)


def _target(thickness, layer):
    if layer in thickness.columns:
        return thickness[layer]
    aliases = {'l1': 'Layer I', 'l2': 'Layer II', 'l3': 'Layer III',
               'l4': 'Layer IV', 'l5': 'Layer V', 'l6': 'Layer VI'}
    return thickness[aliases.get(layer, layer)]


def _mask_layer(layer, mask):
    if layer in mask.index:
        return layer
    aliases = {'l1': 'Layer I', 'l2': 'Layer II', 'l3': 'Layer III',
               'l4': 'Layer IV', 'l5': 'Layer V', 'l6': 'Layer VI'}
    return aliases.get(layer, layer)
