"""BigBrain layer-thickness ~ layer cell-type-ratio pipeline entry point.

Wires dataset.py (load) -> prep.py (CLR features) -> analysis.py (spin test +
whole-match / exact-mismatch permutation tests) -> plotting.py (heatmaps).

Usage:
    python main.py [--atlas BN] [--level subclass] [--mask external|enrichment]
                   [--n-spins 1000] [--n-jobs 20] [--out-dir ../tmpres]
"""
import argparse
import os

import numpy as np
import pandas as pd
from HomoloMap.spins import spin_data

from dataset import load_all, layer_names_roman
from prep import clr_features
from analysis import (parallel_cross_layer_correlation,
                      permutation_test_whole_match,
                      permutation_test_exact_mismatch)
from plotting import plot_layer_heatmap
from contribution import layer_dominance_analysis, plot_dominance_heatmap


def run(atlas='BN', level='subclass', mask_kind='external', n_spins=1000,
       n_jobs=20, seed=42, out_dir=None, show_progress=True,
       use_clr=True, relative=True, run_contribution=True):
    d = load_all(atlas=atlas, level=level, mask_kind=mask_kind)
    prop_mat, layer_CT, mask = d['prop_mat'], d['layer_CT'], d['mask']
    ctypes, layers = d['ctypes'], layer_names_roman

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
        X_layers, Y_layers, seed=seed, n_jobs=n_jobs, show_progress=show_progress)

    exact_mismatch_res = permutation_test_exact_mismatch(
        X_layers, Y_layers, ctypes, layers=layers, mask=mask, seed=seed,
        n_jobs=n_jobs, show_progress=show_progress)

    dominance_res = None
    if run_contribution:
        dominance_res = layer_dominance_analysis(
            X_layers, Y_layers, layers,
            use_adjusted_r_sq=True, method='auto', max_features=15,
            n_samples=10000, n_jobs=n_jobs, show_progress=show_progress)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        spin_res.to_csv(os.path.join(out_dir, 'spin_test.csv'), index=False)
        exact_mismatch_res.to_csv(os.path.join(out_dir, 'exact_mismatch.csv'), index=False)
        pd.DataFrame({'observed_stat': [whole_match_res['observed_stat']],
                     'p_value': [whole_match_res['p_value']]}
                    ).to_csv(os.path.join(out_dir, 'whole_match.csv'), index=False)

        if dominance_res is not None:
            dominance_res.to_csv(os.path.join(out_dir, 'dominance.csv'), index=False)

        r_mat = spin_res.pivot(index='layer', columns='ctype', values='correlation'
                               ).reindex(index=layers, columns=ctypes)
        p_mat = spin_res.pivot(index='layer', columns='ctype', values='p_adjusted'
                               ).reindex(index=layers, columns=ctypes)
        plot_layer_heatmap(
            r_mat, data_p=p_mat, annot=True, mask=~mask,
            vmin=-1, vmax=1, cbar_label='Pearson r (spin-test)',
            title='layer composition (CLR) ~ relative layer thickness\nspin null + FDR',
            output_file=os.path.join(out_dir, 'spin_heatmap.png'))

        r_mm = exact_mismatch_res.pivot(index='layer', columns='ctype', values='r_obs'
                                        ).reindex(index=range(len(layers)), columns=ctypes)
        r_mm.index = layers
        p_mm = exact_mismatch_res.pivot(index='layer', columns='ctype', values='p_adjusted'
                                        ).reindex(index=range(len(layers)), columns=ctypes)
        p_mm.index = layers
        plot_layer_heatmap(
            r_mm, data_p=p_mm, annot=True, mask=~mask,
            vmin=-1, vmax=1, cbar_label='Pearson r',
            title='layer specificity (layer-mismatch permutation)',
            output_file=os.path.join(out_dir, 'exact_mismatch_heatmap.png'))

        if dominance_res is not None:
            dom_mat = dominance_res.pivot(
                index='layer', columns='ctype', values='total_dominance'
            ).reindex(index=layers, columns=ctypes)
            plot_dominance_heatmap(
                dom_mat, mask=~mask,
                title='Total dominance: layer composition → thickness',
                output_file=os.path.join(out_dir, 'dominance_heatmap.png'))

    return {
        'spin_test': spin_res,
        'whole_match': whole_match_res,
        'exact_mismatch': exact_mismatch_res,
        'dominance': dominance_res,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='BN')
    parser.add_argument('--level', default='subclass')
    parser.add_argument('--mask', dest='mask_kind', default='external',
                        choices=['external'])
    parser.add_argument('--n-spins', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'tmpres'))
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--skip-contribution', dest='run_contribution',
                        action='store_false', default=True,
                        help='Skip dominance analysis (faster for quick tests)')
    args = parser.parse_args()

    run(atlas=args.atlas, level=args.level, mask_kind=args.mask_kind,
       n_spins=args.n_spins, n_jobs=args.n_jobs, seed=args.seed,
       out_dir=args.out_dir, show_progress=not args.quiet,
       run_contribution=args.run_contribution)


if __name__ == '__main__':
    main()
