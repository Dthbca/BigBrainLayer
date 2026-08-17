# HomoloMap/plotting.py
"""
Publication-quality visualization for neuroimaging and cell type analysis.

This module provides high-quality plotting functions designed for scientific
publications.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, zscore
from typing import Optional, Tuple, List, Union, Dict
import warnings

try:
    from surfplot import Plot
    SURFPLOT_AVAILABLE = True
except ImportError:
    SURFPLOT_AVAILABLE = False
    warnings.warn("surfplot not available. Surface plotting disabled.")

try:
    import nibabel as nib
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Define custom colormaps for different cell types
# These colormaps are used to visualize different cell type distributions
cmap_YlGn = LinearSegmentedColormap.from_list('Yellow-Green',                   
            ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837'], N=256) # CGE interneuron
cmap_YlBu = LinearSegmentedColormap.from_list('Yellow-Blue', 
            ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'], N=256) # MGE interneuron
cmap_YlRd = LinearSegmentedColormap.from_list('Yellow-Red', 
            ['#ffffcc', '#fed976', '#feb24c', '#fd8d3c', '#f03b20'], N=256) # IT types
cmap_YlPu = LinearSegmentedColormap.from_list('Yellow-Purple', 
            ['#ffffcc', '#d9d0e3', '#b8a4cf', '#9972af', '#762a83'], N=256) # Non Neuron
cmap_YlMg = LinearSegmentedColormap.from_list('YlMg_custom', 
            ['#ffffcc', '#fccde5', '#f768a1', '#dd1c77', '#980043'], N=256) # Deep types
cmap_YlCy = LinearSegmentedColormap.from_list('Yellow-Cyan',
            ['#ffffcc', '#b3e4cb', '#66c2a4', '#339999', '#006d6d'], N=256)

def set_style(dpi: int = 300):
    """
    Set matplotlib style for publication-quality figures.
    
    Parameters
    ----------
    dpi : int, default=300
        Resolution for raster outputs (300+ for print)
        
    """
    default_style = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 9,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
    }
    style = default_style.copy()
    style['figure.dpi'] = dpi
    style['savefig.dpi'] = dpi
    style['savefig.bbox'] = 'tight'
    style['savefig.pad_inches'] = 0.05
    
    plt.rcParams.update(style)
    sns.set_style('ticks') 

def plot_left(
    data: np.ndarray,
    species: str = 'human',
    atlas: str = 'FGC',
    surf: str = 'inflated',
    view: str = 'row',
    cmap: str = 'cividis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    outline: bool = False,
    cbar_label: str = '',
    title: str = '',
    title_fontsize: int = 16,
    size: Tuple[float, float] = (500,200),
    zoom: float = 1.5,
    dpi: int = 300,
    save_path: Optional[str] = None,
    cbar_kwargs: Optional[dict] = None
) -> plt.Figure:
    """
    brain surface visualization.
    
    Parameters
    ----------
    data : np.ndarray
        Vertex data to visualize
    species : str, default='human'
        Species ('human' or 'macaque')
    atlas : str, default='FGC'
        Atlas name
    surf : str, default='inflated'
        Surface type ('inflated', 'pial', 'midthickness')
    view : str, default='row'
        Layout view ('row' or 'column')
    cmap : str, default='cividis'
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    outline : bool, default=False
        Whether to add an outline layer
    cbar_label : str
        Colorbar label
    title : str
        Figure title
    title_fontsize : int, default=16
        Title font size
    size : tuple
        Figure size in inches
    zoom : float
        Zoom level for the surface plot
    dpi : int
        Resolution
    save_path : str, optional
        Path to save figure
    cbar_kwargs : dict, optional
        Additional keyword arguments for colorbar customization
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        
    Examples
    --------
    >>> fig = plot_brain_surface(
    ...     vertex_data,
    ...     cmap='cividis',
    ...     title='Cell Type Distribution',
    ...     save_path='figure1a.pdf'
    ... )
    """
    import os
    if not 'VTK_DEFAULT_OPENGL_WINDOW' in os.environ:
        os.environ['VTK_DEFAULT_OPENGL_WINDOW']='vtkOSOpenGLRenderWindow'

    from HomoloMap.datasets import fetch_fslr, fetch_Yerks, fetch_parc
    import nibabel as nib
    
    # Fetch the appropriate surface file based on the species
    if species == 'human':
        lh = fetch_fslr(surf=surf, return_path=True)
    elif species == 'macaque':
        lh = fetch_Yerks(surf=surf, return_path=True)
    len = nib.load(lh).darrays[0].data.shape[0]
       # Set layout and zoom based on the view type
    if size is None:
        if view == 'row':
            size = (1000, 400)
            zoom = 1.5
        elif view == 'column':
            size = (400, 600)
            zoom = 1.6

    # Create plot
    p = Plot(lh, brightness=0.7, zoom=zoom, size=size,layout=view)
    
    color_range = None
    if vmin is not None and vmax is not None:
        color_range = (vmin, vmax)
    if data.shape[0] != len:
        try:
            data = parc2vertex(data, atlas=atlas)
            if data.shape[0] != len:
                raise ValueError("Data length does not match number of vertices in the surface mesh.")
        except:
            raise ValueError("Data length does not match number of vertices in the surface mesh.")
    p.add_layer({'left': data}, cmap=cmap, color_range=color_range, cbar_label=cbar_label)
    if outline:
        p.add_layer({'left': fetch_parc(key=atlas).darrays[0].data}, cmap='gray',alpha=.5, as_outline=True, cbar=False)
    # Build with publication-quality colorbar
    kws = {
        'location': 'bottom',
        'draw_border': False,
        'decimals': 2,
        'pad': 0.02,
        'aspect': 20,
        'shrink': 0.6,
    }
    if cbar_kwargs is not None:
        kws.update(cbar_kwargs)
    fig = p.build(cbar_kws=kws,dpi=dpi)
    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.8)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def insert_surf_into_ax(surf_fig, target_ax):
    """
    Insert a surface plot into a matplotlib axis.

    Parameters:
    -----------
    surf_fig : matplotlib.figure.Figure
        Surface plot figure.
    target_ax : matplotlib.axes.Axes
        Target axis to insert the surface plot.

    Returns:
    --------
    None
    """
    # extract image data
    canvas = surf_fig.canvas
    canvas.draw()
    imflat = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    imdata = imflat.reshape(*reversed(canvas.get_width_height()), 4)
    imdata = imdata[:, :, :3]
    plt.close()
    
    # remove white space
    nonwhite_pix = (imdata != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_row[::4] = True
    nonwhite_col = nonwhite_pix.any(0)
    nonwhite_col[::4] = True
    imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
    
    # insert into plot
    target_ax.imshow(imdata_cropped)
    target_ax.axis("off")


def plot_surface_map(map, surf_path, cmap='viridis', 
                    save=False, save_path=None, view='default', clim=None):
    """
    Plot a brain map on a surface mesh using Pyvista.

    Parameters:
    -----------
    map : np.ndarray
        Brain map values to plot.
    surf_path : str
        Path to the surface mesh file.
    cmap : str
        Colormap for visualization.
    save : bool
        Whether to save the plot.
    save_path : str
        Path to save the plot.
    view : str
        View angle for the plot.
    clim : tuple
        Color limits for the plot.

    Returns:
    --------
    None
    """
    gii_mesh = nib.load(surf_path)
    points, triangles = gii_mesh.agg_data()
    mesh = pv.PolyData(
            points,
            np.c_[np.ones((triangles.shape[0],), dtype=int)*3, triangles]
            )
    pl = pv.Plotter(window_size=(1000, 1000), lighting="none", off_screen=True)
    mesh.point_data['map'] = map
    pl.add_mesh(mesh, scalars='map', cmap=cmap, clim=clim)
    pl.remove_scalar_bar()
    if view == 'yz_negative':
        pl.view_yz(negative=True)
    pl.show(auto_close=False)

    if save:
        plt.ioff()
        plt.figure()
        plt.imshow(pl.image)
        plt.axis('off')
        plt.savefig(save_path, dpi=600)
        plt.close('all')
        plt.ion()


def scatterplot(X, Y, triu=False, tight=False, figsize=None, 
                c='black', xlabel=None, ylabel=None, xscale='linear', 
                compute_r=False, compute_rho=False, 
                r_round=None, r_title="r: ", rho_title='rho: ', 
                plot_cbar=False, cbar_label='', s=9, **kwargs):
    """
    Create a scatterplot with optional correlation calculations.

    Parameters:
    -----------
    X, Y : np.ndarray
        Data for the x and y axes.
    triu : bool
        Whether to use only the upper triangular part of the data.
    tight : bool
        Whether to use tight layout.
    figsize : tuple
        Figure size.
    c : str or np.ndarray
        Color of the points.
    xlabel, ylabel : str
        Labels for the x and y axes.
    xscale : str
        Scale for the x-axis.
    compute_r, compute_rho : bool
        Whether to compute Pearson or Spearman correlation.
    r_round : int
        Rounding for correlation values.
    r_title, rho_title : str
        Titles for the correlation values.
    plot_cbar : bool
        Whether to display a colorbar.
    cbar_label : str
        Label for the colorbar.
    s : int
        Size of the points.

    Returns:
    --------
    None
    """
    # Only look at upper triangular indices
    if triu:
        X = X[np.triu_indices(len(X), 1)]
        Y = Y[np.triu_indices(len(Y), 1)]
        if isinstance(c, np.ndarray):
            c = c[np.triu_indices(len(c), 1)]

    plt.figure(figsize=figsize)
    plt.scatter(X, Y,s=s, c=c, **kwargs)

    if compute_r and compute_rho:
        r, _ = pearsonr(X, Y)
        rho, _ = pearsonr(rankdata(X), rankdata(Y))
        if r_round is None:
            plt.title(f"{r_title}{r} | {rho_title}{rho}")
        else:
            plt.title(f"{r_title}{round(r, r_round)} | "
                      f"{rho_title}{round(rho, r_round)}")
    elif compute_r:
        r, _ = pearsonr(X, Y)
        if r_round is None:
            plt.title(f"{r_title}{r}")
        else:
            plt.title(f"{r_title}{round(r, r_round)}")
    elif compute_rho:
        rho, _ = pearsonr(rankdata(X), rankdata(Y))
        if r_round is None:
            plt.title(f"{rho_title}{rho}")
        else:
            plt.title(f"{rho_title}{round(rho, r_round)}")

    # Change x/y labels if not None (if None, leave as is)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.xscale(xscale)

    if plot_cbar:
        cbar = plt.colorbar()
        cbar.set_label(cbar_label)

    if tight:
        plt.tight_layout()


def p_value_annot(ax, pvalues):
    """
    Annotate a heatmap with significance levels based on p-values.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to annotate.
    pvalues : pd.DataFrame
        DataFrame of p-values.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        Annotated axis.
    """
    for i in range(pvalues.shape[0]):
        for j in range(pvalues.shape[1]):
            pval = pvalues.iloc[i, j]
            if pval < 0.001:
                text = "***"
            elif pval < 0.01:
                text = "**"
            elif pval < 0.05:
                text = "*"
            else:
                text = ""
            ax.text(j + 0.5, i + 0.8, text, 
                    ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black')
    return ax

from HomoloMap.transforms.parcellation import parc2vertex

def plot_cumulative_effects(data, r2_col='model_r_sq', pval_col='model_pval', 
                      label_col='disease', pval_threshold=0.05,
                      figsize=(6, 8), color='#F4A460', 
                      title=None, save_path=None):
    """
    Create a horizontal bar plot with significance markers.
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Data containing disease names, R² values, and p-values
    r2_col : str
        Column name for R² values
    pval_col : str
        Column name for p-values
    label_col : str
        Column name for disease/condition labels
    pval_threshold : float
        Threshold for significance (default 0.05)
    figsize : tuple
        Figure size (width, height)
    color : str
        Bar color (default is sandy brown/orange)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Sort by R² values (optional - remove if you want original order)
    # data = data.sort_values(by=r2_col, ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bars
    y_pos = np.arange(len(data))
    bars = ax.barh(y_pos, data[r2_col],height=.7, color=color, edgecolor='none', alpha=0.8)
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data[label_col], fontsize=10)
    
    # Set x-axis
    ax.set_xlabel(f'Adjusted $R^2$', fontsize=12, fontweight='normal')
    ax.set_xlim(0, max(data[r2_col]) * 1.15)  # Add space for asterisks
    
    # Add asterisks for significant results
    for i, (idx, row) in enumerate(data.iterrows()):
        if row[pval_col] < pval_threshold:
            ax.text(1, i, '*', 
                   fontsize=14, va='center', ha='left')
    
    # Add significance threshold note
    ax.text(0.98, -0.08, f'* P_spin < {pval_threshold}', 
           transform=ax.transAxes, fontsize=10, 
           ha='right', va='top', style='italic')
    
    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, bottom=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax

def plot_spin_test_results(data,fdr = True, annot_fontsize=6, cmap='vlag', center=0, figsize=(6, 8), dpi=300):
    """
    Visualize spin test results using a heatmap of correlation coefficients and annotated p-values.

    Parameters:
    -----------
    data : pd.DataFrame
        A DataFrame containing the results of the spin test. The DataFrame should have:
        - Rows representing cell types.
        - Columns representing IDPs (Image-Derived Phenotypes), with groups of three columns:
          - Correlation coefficients (e.g., r-values).
          - Raw p-values.
          - FDR-corrected p-values (if applicable).
    fdr : bool, optional
        If True, use FDR-corrected p-values for annotations. If False, use raw p-values. Default is True.
    annot_fontsize : int, optional
        Font size for the numerical annotations on the heatmap. Default is 6.
    cmap : str, optional
        Colormap for the heatmap. Default is 'vlag'.
    center : float, optional
        Value at which to center the colormap. Default is 0.
    figsize : tuple, optional
        Dimensions of the figure in inches (width, height). Default is (6, 8).
    dpi : int, optional
        Resolution of the figure in dots per inch. Default is 300.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated heatmap figure.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    correlation_data = data.iloc[:,::3]
    # Plot the heatmap for correlation data
    sns.heatmap(
        correlation_data,
        annot=True,
        annot_kws={'fontsize': annot_fontsize},
        cmap=cmap,
        center=center,
        ax=ax
    )

    p_value_data = data.iloc[:,2::3] if fdr else data.iloc[:,1::3]
    # Overlay p-value annotations
    p_value_annot(ax, p_value_data)
    # Customize the plot
    ax.set_title('Spin Test Results', fontsize=12)
    ax.set_xlabel('IDPs', fontsize=10)
    ax.set_ylabel('Cell Types', fontsize=10)
    plt.tight_layout()
    return fig


def plot_shap_results(data, feature, plot_param=None):
    """
    Plot SHAP results as a bar plot or summary plot.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for the features.
    feature_data : np.ndarray or pd.DataFrame
        The feature data corresponding to the SHAP values.
    feature_names : list
        List of feature names (e.g., cell types).
    plot_param : dict
        Parameters for the SHAP plot, including 'plot_type', 'max_display', and 'show'...

    Returns:
    --------
    None
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required for plot_shap_results") from exc
    if plot_param is None:
        plot_param = {'plot_type': 'dot', 'max_display': 10, 'show': False}
    shap_results = data.get('explanations', data.get('shap_analysis'))
    if shap_results is None:
        raise KeyError("No SHAP results found in 'explanations' or 'shap_analysis'")
    shap_values = shap_results[feature]['shap_values']
    feature_data = data['X'].values
    feature_names = data['features']
    fig = shap.summary_plot(
        shap_values,
        feature_data,
        feature_names=feature_names,
        **plot_param
    )
    return fig


def create_multi_panel_figure(
    panels: List[Dict],
    layout: str = 'horizontal',  # 'horizontal', 'vertical', 'grid'
    panel_labels: bool = True,
    label_fontsize: int = 12,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create multi-panel figure.
    
    Parameters
    ----------
    panels : list of dict
        List of panel specifications. Each dict should contain:
        - 'plot_func': Function to generate the panel
        - 'args': Arguments for plot_func
        - 'label': Panel label (A, B, C, ...)
    layout : str
        Panel layout ('horizontal', 'vertical', 'grid')
    panel_labels : bool
        Add panel labels (A, B, C, ...)
    label_fontsize : int
        Font size for panel labels
    figsize : tuple, optional
        Overall figure size
    save_path : str, optional
        Save path
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        
    Examples
    --------
    >>> panels = [
    ...     {
    ...         'plot_func': scatter,
    ...         'args': {'x': x1, 'y': y1, 'title': 'Correlation 1'},
    ...         'label': 'A'
    ...     },
    ...     {
    ...         'plot_func': manhattan,
    ...         'args': {'results': results, 'title': 'Associations'},
    ...         'label': 'B'
    ...     }
    ... ]
    >>> fig = create_multi_panel_figure(
    ...     panels,
    ...     layout='horizontal',
    ...     save_path='figure1.pdf'
    ... )
    """
    n_panels = len(panels)
    
    # Determine layout
    if layout == 'horizontal':
        nrows, ncols = 1, n_panels
        if figsize is None:
            figsize = (4 * n_panels, 4)
    elif layout == 'vertical':
        nrows, ncols = n_panels, 1
        if figsize is None:
            figsize = (4, 4 * n_panels)
    elif layout == 'grid':
        ncols = int(np.ceil(np.sqrt(n_panels)))
        nrows = int(np.ceil(n_panels / ncols))
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
    
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                          hspace=0.3, wspace=0.3)
    
    # Create panels
    for i, panel in enumerate(panels):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot panel
        plot_func = panel['plot_func']
        args = panel.get('args', {})
        args['ax'] = ax
        plot_func(**args)
        
        # Add panel label
        if panel_labels:
            label = panel.get('label', chr(65 + i))  # A, B, C, ...
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,
                   fontsize=label_fontsize, fontweight='bold',
                   va='top', ha='right')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
