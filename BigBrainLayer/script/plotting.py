import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr, pearsonr, linregress, t as t_dist, rankdata
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
from collections import OrderedDict

layer_names_roman = ['Layer I', 'Layer II', 'Layer III',
                     'Layer IV', 'Layer V', 'Layer VI']

# cell-type class grouping (drives heatmap column order + top colour bar)
cls_def = {
    'Excitatory': ['L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'L6 IT Car3',
                   'L5 ET', 'L5/6 NP', 'L6 CT', 'L6b'],
    'Inhibitory': ['Lamp5_Lhx6', 'Lamp5', 'Pax6', 'Sncg', 'Vip',
                   'Sst', 'Pvalb', 'Chandelier'],
    'Non-neuron': ['Astro', 'Oligo', 'OPC', 'Micro-PVM', 'Endo', 'VLMC'],
}
cls_order  = ['Excitatory', 'Inhibitory', 'Non-neuron']
cls_colors = {'Excitatory': '#C04535', 'Inhibitory': '#3D9156',
              'Non-neuron': '#4A7BAF', 'Other': '#B0B0B0'}

def plot_layer_heatmap(data, data_p=None, annot=False,
                       annot_thresholds=(0.05, 0.01, 0.001),
                       annot_symbols=('*', '**', '***'),
                       figsize_scale=(0.31, 0.4), vmin=-2, vmax=2,
                       cmap_name='RdYlBu_r', mask_color='#E8E8E8',
                       cbar_label='Layer enrichment score',
                       title='Cortical layer cell type distribution',
                       add_grid=True, output_file=None, dpi=300, mask=None):
    """Nature-style layer x cell-type heatmap.

    data : DataFrame (rows = layers, cols = cell types). NaN cells shown as mask_color.
    mask : optional bool DataFrame; True cells are hidden (set to NaN).
    """
    cls_map = {ct: cls for cls in cls_order for ct in cls_def[cls]}
    base_to_cols = {}
    for col in data.columns:
        base = col if col in cls_map else (col.rsplit('_', 1)[0] if '_' in col else col)
        base_to_cols.setdefault(base, []).append(col)
    final_cols = []
    for cls in cls_order:
        for ct in cls_def[cls]:
            if ct in base_to_cols:
                final_cols.extend(base_to_cols.pop(ct))
    other = [c for cols in base_to_cols.values() for c in cols]
    if other:
        print(f'Warning: unclassified columns placed in Other: {other}')
        final_cols.extend(other)
    final_cols = [c for c in final_cols if c in data.columns]

    data = data[final_cols].copy(); data.index = layer_names_roman
    if data_p is not None:
        data_p = data_p[final_cols].copy(); data_p.index = layer_names_roman

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = pd.DataFrame(mask, columns=final_cols)
        mask = mask[final_cols].copy(); mask.index = layer_names_roman
        data = data.mask(mask.values)
        if data_p is not None:
            data_p = data_p.mask(mask.values)

    n_rows, n_cols = data.shape

    class_columns = OrderedDict()
    for col in final_cols:
        base = col if col in cls_map else (col.rsplit('_', 1)[0] if '_' in col else col)
        class_columns.setdefault(cls_map.get(base, 'Other'), []).append(col)
    cls_positions, start = {}, 0
    for cls, cols in class_columns.items():
        cls_positions[cls] = (start, start + len(cols)); start += len(cols)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad(color=mask_color)
    fig, ax = plt.subplots(figsize=(n_cols * figsize_scale[0] + 1.2,
                                    n_rows * figsize_scale[1] + 0.8))
    im = ax.imshow(data.values, cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest')
    if add_grid:
        for i in range(1, n_rows): ax.axhline(i - 0.52, color='white', lw=0.6, zorder=10)
        for j in range(1, n_cols): ax.axvline(j - 0.52, color='white', lw=0.6, zorder=10)

    if annot and data_p is not None:
        for i in range(n_rows):
            for j in range(n_cols):
                val, p = data.iloc[i, j], data_p.iloc[i, j]
                if np.isnan(val) or np.isnan(p):
                    continue
                sym = next((s for t, s in zip(annot_thresholds, annot_symbols) if p < t), None)
                if sym is None:
                    continue
                rgba = cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, sym, ha='center', va='center', zorder=20,
                        color='white' if lum < 0.5 else 'black',
                        fontsize=8, fontweight='bold')

    ax.set_xticks(np.arange(n_cols)); ax.set_xticklabels(final_cols, rotation=90, fontsize=8)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
    ax.set_yticks(np.arange(n_rows)); ax.set_yticklabels(layer_names_roman, fontsize=8, va='center')
    ax.tick_params(axis='y', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

    top = ax.inset_axes([0, 1.02, 1, 0.1], transform=ax.transAxes)
    top.set_xlim(0, n_cols); top.set_ylim(0, 1); top.axis('off')
    for cls, (s, e) in cls_positions.items():
        top.add_patch(mpatches.Rectangle((s, 0), e - s, 1,
                      facecolor=cls_colors.get(cls, '#888'), edgecolor='none'))
        if e - s > 2:
            top.text((s + e) / 2, 0.5, cls, ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.1, pad=0.08,
                        shrink=0.8, aspect=10)
    cbar.set_label(cbar_label, fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=7, length=3, width=0.6)
    ticks = np.arange(vmin, vmax + 0.01, (vmax - vmin) / 4)
    cbar.set_ticks(ticks); cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])
    ax.set_title(title, fontsize=11, pad=18, weight='normal')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    return fig, ax

def draw_scatter(ax, x, y, c, xlim=None, ylim=None, xlabel='', ylabel=''):
    ax.scatter(x, y, s=22, color=c, alpha=0.52, lw=0, zorder=3)
    xl, yl, lo, hi = _ols_ci(x, y)
    ax.plot(xl, yl, color=c, lw=1.6, zorder=4)
    ax.fill_between(xl, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
    av  = np.concatenate([x, y])
    ref = [av.min() - 0.15, av.max() + 0.15]
    rho, p = pearsonr(x, y)
    pstr = ('< 0.001' if p < 0.001
            else f'= {p:.3f}' if p < 0.01
            else f'= {p:.2f}')
    ax.set_xlabel(xlabel, fontsize=6.5, labelpad=.8,fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=6.5, labelpad=.8,fontweight='medium')
    ax.tick_params(labelsize=8, length=2.5, width=0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('medium')
    for sp in ['left', 'bottom']: ax.spines[sp].set_linewidth(0.8)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_box_aspect(1)

def _ols_ci(x, y, n_pts=200, ci=0.95):
    n  = len(x);  xl = np.linspace(x.min(), x.max(), n_pts)
    sl, ic, *_ = linregress(x, y)
    yh = sl * x + ic;  s = np.sqrt(np.sum((y - yh) ** 2) / (n - 2))
    xm = x.mean();  sx = np.sum((x - xm) ** 2)
    se = s * np.sqrt(1 / n + (xl - xm) ** 2 / sx)
    tc = t_dist.ppf((1 + ci) / 2, df=n - 2)
    yl = sl * xl + ic
    return xl, yl, yl - tc * se, yl + tc * se
