"""Multi-branch runner: sweep pipeline strategy choices and compare results.

Branches swept (2 x 2 = 4 combinations), each run through the full
dataset -> prep -> analysis -> plotting pipeline (main.run):

    use_clr    : CLR-transform layer composition        vs. raw proportions
    relative   : row-normalised (relative) thickness    vs. absolute thickness

Mask is fixed to the curated external laminar mask (mask_kind='external') for
all branches.

Per-branch outputs (spin_test.csv, exact_mismatch.csv, whole_match.csv, heatmaps)
are written to <out-dir>/<branch-name>/. A cross-branch summary table and
comparison figure are written directly under <out-dir>/.

Usage:
    python run_branches.py [--atlas BN] [--level subclass] [--n-spins 1000]
                           [--n-jobs 20] [--out-dir ../tmpres/branches]
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import run

BRANCH_AXES = {
    'use_clr': [True, False],
    'relative': [True, False],
}
MASK_KIND = 'external'


def branch_name(use_clr, relative):
    return f"clr{'T' if use_clr else 'F'}_rel{'T' if relative else 'F'}_{MASK_KIND}"


def run_all_branches(atlas='BN', level='subclass', n_spins=1000, n_jobs=20,
                     seed=42, out_dir=None, show_progress=True):
    combos = list(itertools.product(*BRANCH_AXES.values()))
    summary_rows = []
    all_results = {}

    for use_clr, relative in combos:
        name = branch_name(use_clr, relative)
        print(f'=== branch {name} ===')
        branch_out = os.path.join(out_dir, name) if out_dir else None
        res = run(atlas=atlas, level=level, mask_kind=MASK_KIND,
                  n_spins=n_spins, n_jobs=n_jobs, seed=seed,
                  out_dir=branch_out, show_progress=show_progress,
                  use_clr=use_clr, relative=relative)
        all_results[name] = res

        spin, mm, wm = res['spin_test'], res['exact_mismatch'], res['whole_match']
        summary_rows.append({
            'branch': name, 'use_clr': use_clr, 'relative': relative, 'mask_kind': MASK_KIND,
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
        summary.to_csv(os.path.join(out_dir, 'branch_summary.csv'), index=False)
        plot_branch_comparison(summary, output_file=os.path.join(out_dir, 'branch_comparison.png'))
    return summary, all_results


def plot_branch_comparison(summary, output_file=None):
    """Three-panel bar comparison: significant-cell counts, mean r, whole-match statistic."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(summary))
    labels = summary['branch']

    ax = axes[0]
    ax.bar(x - 0.2, summary['n_spin_sig'], width=0.4, color='#4A7BAF', label='spin-test')
    ax.bar(x + 0.2, summary['n_mismatch_sig'], width=0.4, color='#C04535', label='exact-mismatch')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('# significant (layer, ctype) cells (FDR<0.05)')
    ax.legend(fontsize=7); ax.set_title('Significant hits per branch')

    ax = axes[1]
    ax.bar(x - 0.2, summary['spin_mean_r'], width=0.4, color='#4A7BAF', label='spin-test')
    ax.bar(x + 0.2, summary['mismatch_mean_r'], width=0.4, color='#C04535', label='exact-mismatch')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('mean correlation r'); ax.legend(fontsize=7); ax.set_title('Mean correlation per branch')

    ax = axes[2]
    colors = ['#2E7D32' if p < 0.05 else '#B0B0B0' for p in summary['whole_match_p']]
    ax.bar(x, summary['whole_match_stat'], color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
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
        os.path.dirname(os.path.abspath(__file__)), '..', 'tmpres', 'branches'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_all_branches(atlas=args.atlas, level=args.level, n_spins=args.n_spins,
                     n_jobs=args.n_jobs, seed=args.seed, out_dir=args.out_dir,
                     show_progress=not args.quiet)


if __name__ == '__main__':
    main()
