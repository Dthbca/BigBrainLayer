"""Layer-wise contribution analysis for cell-type composition → thickness coupling.

This module quantifies how much each layer's cell-type composition explains
its corresponding thickness, using dominance analysis and SHAP values.

Key metrics:
- Total dominance: overall contribution of each cell type to R²
- Conditional dominance: average marginal contribution across subset sizes
- SHAP values: additive feature attributions for individual predictions

Functions are layer-wise: we fit separate models for each layer to respect
the structural constraint that Layer III composition should predict Layer III
thickness, not Layer V thickness.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

try:
    from CellAlign.stats.analysis import (
        get_dominance_stats,
        get_shap_stats,
        get_reg_r_sq
    )
    CELLALIGN_AVAILABLE = True
except ImportError:
    CELLALIGN_AVAILABLE = False


def _layer_dominance_worker(args):
    """Worker function for parallel layer-wise dominance analysis."""
    layer_idx, layer_name, X, y, use_adjusted, method, max_features, n_samples = args

    # Handle constant target (all regions have same thickness)
    if np.std(y) < 1e-10:
        return {
            'layer': layer_name,
            'layer_idx': layer_idx,
            'warning': 'constant_target',
            'total_dominance': {},
            'conditional_dominance': {},
            'R2': 0.0
        }

    # Dominance analysis
    total_dom, cond_dom = get_dominance_stats(
        X, y,
        use_adjusted_r_sq=use_adjusted,
        method=method,
        max_features=max_features,
        n_samples=n_samples,
        verbose=False,
        n_jobs=1  # Already parallelized at layer level
    )

    # Total R² (full model)
    R2 = get_reg_r_sq(X, y, adjust=use_adjusted, model_type='linear', scale_data=True)

    return {
        'layer': layer_name,
        'layer_idx': layer_idx,
        'total_dominance': total_dom,
        'conditional_dominance': cond_dom,
        'R2': R2
    }


def layer_dominance_analysis(
    X_layers: List[pd.DataFrame],
    Y_layers: List[np.ndarray],
    layers: List[str],
    use_adjusted_r_sq: bool = True,
    method: str = 'auto',
    max_features: int = 15,
    n_samples: int = 10000,
    n_jobs: int = -1,
    show_progress: bool = True
) -> pd.DataFrame:
    """Dominance analysis for each layer's cell-type composition → thickness.

    Parameters
    ----------
    X_layers : list of pd.DataFrame
        Per-layer CLR-transformed cell-type features (n_region, n_ctype_present).
    Y_layers : list of np.ndarray
        Per-layer thickness values (n_region,).
    layers : list of str
        Layer names (e.g., ['Layer I', 'Layer II', ...]).
    use_adjusted_r_sq : bool, default=True
        Use adjusted R² (recommended for varying feature counts).
    method : str, default='auto'
        Dominance method: 'auto', 'full', 'approximate', 'incremental'.
    max_features : int, default=15
        Max features for exhaustive dominance (auto switches to approximate above this).
    n_samples : int, default=10000
        Number of Monte Carlo samples for approximate dominance.
    n_jobs : int, default=-1
        Parallel jobs (layer-level parallelism).
    show_progress : bool, default=True
        Show tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: layer, ctype, total_dominance, conditional_dominance, R2_full.
        One row per (layer, ctype) combination where ctype is present in that layer.
    """
    if not CELLALIGN_AVAILABLE:
        raise ImportError(
            "CellAlign.stats.analysis is required. "
            "Ensure CellAlign is installed and PYTHONPATH is set."
        )

    tasks = [
        (i, layer, X_layers[i].values, Y_layers[i],
         use_adjusted_r_sq, method, max_features, n_samples)
        for i, layer in enumerate(layers)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_layer_dominance_worker, t): i for i, t in enumerate(tasks)}
        it = as_completed(futs)
        if show_progress:
            it = tqdm(it, total=len(tasks), desc='dominance')

        for f in it:
            res = f.result()
            layer_name = res['layer']
            layer_idx = res['layer_idx']
            R2 = res['R2']

            if 'warning' in res:
                # Skip layers with constant thickness
                continue

            # Extract per-ctype dominance
            ctypes = X_layers[layer_idx].columns.tolist()
            for j, ctype in enumerate(ctypes):
                total_dom = res['total_dominance'].get(j, 0.0)
                cond_dom = res['conditional_dominance'].get(j, 0.0)
                results.append({
                    'layer': layer_name,
                    'ctype': ctype,
                    'total_dominance': total_dom,
                    'conditional_dominance': cond_dom,
                    'R2_full': R2
                })

    return pd.DataFrame(results)


def plot_dominance_heatmap(
    data: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    figsize_scale: Tuple[float, float] = (0.31, 0.4),
    vmin: float = 0.0,
    vmax: float = 0.3,
    cmap_name: str = 'YlOrRd',
    mask_color: str = '#E8E8E8',
    cbar_label: str = 'Total dominance',
    title: str = 'Dominance: layer composition → thickness',
    add_grid: bool = True,
    output_file: Optional[str] = None,
    dpi: int = 300
):
    """Plot dominance heatmap (layer x cell type).

    Parameters
    ----------
    data : pd.DataFrame
        Rows = layers, columns = cell types. Values = total dominance.
    mask : pd.DataFrame, optional
        Boolean mask (True = hide cell).
    figsize_scale : tuple, default=(0.31, 0.4)
        (col_width, row_height) in inches.
    vmin, vmax : float
        Color scale limits.
    cmap_name : str, default='YlOrRd'
        Matplotlib colormap.
    mask_color : str, default='#E8E8E8'
        Color for masked (absent) cells.
    cbar_label : str
        Colorbar label.
    title : str
        Plot title.
    add_grid : bool, default=True
        Add grid lines.
    output_file : str, optional
        Save path.
    dpi : int, default=300
        DPI for saved figure.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if mask is not None:
        data = data.mask(mask.values)

    n_rows, n_cols = data.shape
    figsize = (n_cols * figsize_scale[0], n_rows * figsize_scale[1])
    fig, ax = plt.subplots(figsize=figsize)

    # Colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Plot cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data.iloc[i, j]
            if pd.isna(val):
                color = mask_color
            else:
                color = cmap(norm(val))
            ax.add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', lw=0.5))

            # Annotate value
            if not pd.isna(val):
                ax.text(j + 0.5, i + 0.5, f'{val:.3f}',
                        ha='center', va='center', fontsize=7, color='black' if val < (vmax * 0.7) else 'white')

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Labels
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(data.columns, rotation=90, ha='center', va='top', fontsize=8)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(data.index, fontsize=8, va='center')
    ax.tick_params(axis='both', length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if add_grid:
        for i in range(n_rows + 1):
            ax.axhline(i, color='white', lw=1.2)
        for j in range(n_cols + 1):
            ax.axvline(j, color='white', lw=1.2)

    # Colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, orientation='vertical', fraction=0.1, pad=0.08, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=7, length=3, width=0.6)

    ax.set_title(title, fontsize=11, pad=18, weight='normal')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    return fig, ax
