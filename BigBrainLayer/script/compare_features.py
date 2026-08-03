"""Compare normalization strategies for cell-count features.

Standalone comparison — does NOT modify the core pipeline (main.py).
Uses load_layer_data() directly with different parameters.

Strategies compared:

1. **prop_mode**: How to compute proportions
   - `by_layer`    — divide by layer total within each region.
                     Each layer's cell-type profile sums to 1 independently.
   - `by_region`   — aggregate across layers first, then divide by region total.
                     Each region sums to 1; captures both cell-type composition
                     and layer thickness.

2. **prop_order** (only meaningful for `by_region`): Order of normalize vs. relabel
   - `after` (default) — relabel first (D99->atlas), then normalize in atlas space.
                         Current pipeline behaviour.
   - `before`          — normalize within each D99 region first, then relabel.
                         The denominator is the D99 layer total, not the BN layer total.

3. **use_clr**: CLR-transform layer composition vs. raw proportions.

4. **relative**: Row-normalised (relative) thickness vs. absolute thickness.

Sweeps: prop_mode (2) x prop_order (2) x use_clr (2) x relative (2) = 16 branches.
Mask is fixed to external laminar mask (mask_kind='external').

Usage:
    python compare_features.py [--atlas BN] [--level subclass] [--n-spins 1000]
                               [--n-jobs 20] [--out-dir ../tmpres/prop_feature_compare]
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import load_layer_data, load_mask, load_bigbrain_thickness, layer_names_roman, LAYER_KEYS
from prep import clr_features
from analysis import (parallel_cross_layer_correlation,
                      permutation_test_whole_match,
                      permutation_test_exact_mismatch)
from CellAlign.spins import spin_data


STRATEGY_AXES = {
    'prop_mode': ['by_layer', 'by_region'],
    'prop_order': ['after', 'before'],
    'use_clr': [True, False],
    'relative': [True, False],
}
MASK_KIND = 'external'


def branch_name(prop_mode, prop_order, use_clr, relative):
    return (f"prop={prop_mode}_order={prop_order}"
            f"_clr{'T' if use_clr else 'F'}_rel{'T' if relative else 'F'}")


def run_strategy(atlas, level, prop_mode, prop_order, n_spins, n_jobs,
                 seed, use_clr, relative, show_progress):
    """Run full pipeline for one strategy configuration."""
    prop_mat, ctypes, regions, layers = load_layer_data(
        atlas, level, return_ratio=True, prop_mode=prop_mode, prop_order=prop_order)
    count_arr, _, _, _ = load_layer_data(
        atlas, level, return_ratio=False, prop_mode=prop_mode, prop_order=prop_order)
    layer_CT, rel_CT = load_bigbrain_thickness(regions, atlas=atlas)

    mask = load_mask(ctypes, layers=LAYER_KEYS)

    np.random.seed(seed)
    spins = spin_data(data=layer_CT, atlas=atlas, n_spins=n_spins, return_ind=True)

    spin_res = parallel_cross_layer_correlation(
        prop_mat, layer_CT, layers, ctypes, spins,
        use_clr=use_clr, relative=relative, mask=mask,
        fdr_alpha=0.05, max_workers=n_jobs, show_progress=show_progress)

    X_layers = clr_features(prop_mat, layers, ctypes, mask=mask, use_clr=use_clr)
    Y = layer_CT.div(layer_CT.sum(axis=1), axis=0) if relative else layer_CT
    Y_layers = [Y.iloc[:, l].values for l in range(len(layers))]

    whole_match_res = permutation_test_whole_match(
        X_layers, Y_layers, n_jobs=n_jobs, show_progress=show_progress)

    exact_mismatch_res = permutation_test_exact_mismatch(
        X_layers, Y_layers, ctypes, layers=layers, mask=mask,
        n_jobs=n_jobs, show_progress=show_progress)

    return {
        'spin_test': spin_res,
        'whole_match': whole_match_res,
        'exact_mismatch': exact_mismatch_res,
    }


def run_all(atlas='BN', level='subclass', n_spins=1000, n_jobs=20,
            seed=42, out_dir=None, show_progress=True):
    combos = list(itertools.product(*STRATEGY_AXES.values()))
    summary_rows = []
    all_results = {}

    for prop_mode, prop_order, use_clr, relative in combos:
        name = branch_name(prop_mode, prop_order, use_clr, relative)
        print(f'=== branch {name} ===')
        res = run_strategy(
            atlas, level, prop_mode, prop_order, n_spins, n_jobs,
            seed, use_clr, relative, show_progress)
        all_results[name] = res

        spin, mm, wm = res['spin_test'], res['exact_mismatch'], res['whole_match']
        summary_rows.append({
            'branch': name, 'prop_mode': prop_mode, 'prop_order': prop_order,
            'use_clr': use_clr, 'relative': relative, 'mask_kind': MASK_KIND,
            'n_spin_tested': len(spin),
            'n_spin_sig': int(spin['reject_H0'].sum()) if 'reject_H0' in spin else np.nan,
            'spin_mean_r': spin['correlation'].mean(),
            'n_mismatch_tested': len(mm),
            'n_mismatch_sig': int(mm['reject_H0'].sum()) if 'reject_H0' in mm else np.nan,
            'mismatch_mean_r': mm['r_obs'].mean(),
            'whole_match_stat': wm['observed_stat'], 'whole_match_p': wm['p_value'],
        })

    summary = pd.DataFrame(summary_rows)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, 'prop_feature_compare_summary.csv'), index=False)
        plot_prop_feature_compare(summary, output_file=os.path.join(out_dir, 'prop_feature_compare.png'))
    return summary, all_results


def plot_prop_feature_compare(summary, output_file=None):
    """Compare strategies across branches."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    layers = sorted(summary['branch'].unique())
    x = np.arange(len(layers))
    labels = layers

    ax = axes[0]
    for prop_mode in ['by_layer', 'by_region']:
        mask = summary['prop_mode'] == prop_mode
        ax.plot(x[mask], summary.loc[mask, 'n_spin_sig'], 'o-', label=prop_mode)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('# significant (layer, ctype) cells (FDR<0.05)')
    ax.legend(fontsize=7); ax.set_title('Significant hits per branch')

    ax = axes[1]
    for prop_mode in ['by_layer', 'by_region']:
        mask = summary['prop_mode'] == prop_mode
        ax.plot(x[mask], summary.loc[mask, 'spin_mean_r'], 'o-', label=prop_mode)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('mean correlation r'); ax.legend(fontsize=7); ax.set_title('Mean correlation per branch')

    ax = axes[2]
    for prop_mode in ['by_layer', 'by_region']:
        mask = summary['prop_mode'] == prop_mode
        colors = ['#2E7D32' if p < 0.05 else '#B0B0B0' for p in summary.loc[mask, 'whole_match_p']]
        ax.bar(x[mask], summary.loc[mask, 'whole_match_stat'], color=colors, label=prop_mode)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('whole-match observed statistic')
    ax.set_title('Whole-match statistic (green: p<0.05)')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.show()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='BN')
    parser.add_argument('--level', default='subclass')
    parser.add_argument('--n-spins', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'tmpres', 'prop_feature_compare'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_all(atlas=args.atlas, level=args.level, n_spins=args.n_spins,
            n_jobs=args.n_jobs, seed=args.seed, out_dir=args.out_dir,
            show_progress=not args.quiet)


if __name__ == '__main__':
    main()
