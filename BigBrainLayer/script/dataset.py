import os

import numpy as np
import pandas as pd

from CellAlign.parcellation import vol_relabel
from CellAlign.datasets.atlas import ctype_ratio_agg

LAYER_KEYS = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
layer_names_roman = ['Layer I', 'Layer II', 'Layer III',
                     'Layer IV', 'Layer V', 'Layer VI']

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT_ENV = os.environ.get('BIGBRAIN_DATASET_ROOT', None)

def _resolve_data_root():
    if _DATA_ROOT_ENV is not None:
        return _DATA_ROOT_ENV
    return os.path.normpath(os.path.join(_THIS_DIR, '..', 'dataset'))

DATA_ROOT = _resolve_data_root()


def load_ctype_map(data_root=DATA_ROOT):
    path = os.path.join(data_root, 'Spatial', 'cluster_mapping_dict.csv')
    return pd.read_csv(path, index_col=0)


def load_raw_counts(data_root=DATA_ROOT):
    """Load the precomputed D99, raw-cell-type per-layer count stack.

    Returns
    -------
    arr : np.ndarray, shape (n_region, n_layer, n_ctype)
    ctype_names : list[str]
    region_names : list[str]
    layers : list[str]
    """
    path = os.path.join(data_root, 'Spatial', 'raw_counts_d99.npy')
    payload = np.load(path, allow_pickle=True).item()
    return payload['counts'], payload['ctypes'], payload['regions'], LAYER_KEYS


def load_layer_data(atlas='BN', level='subclass', return_ratio=True,
                    data_root=DATA_ROOT, ctype_map=None, prop_mode='by_layer',
                    prop_order='after'):
    """Relabel (D99 -> atlas) and aggregate (raw cell type -> level) the raw counts.

    Parameters
    ----------
    prop_mode : str
        'by_layer'    — normalize within each layer independently.
        'by_region'   — aggregate across layers, then normalize per region.
    prop_order : str
        How to order normalize vs. relabel. Only meaningful for `by_region`.
        'after' (default) — relabel first (D99→atlas), then normalize in atlas space.
                            Current pipeline behaviour.
        'before'          — normalize within each D99 region first (denominator is the
                            per-region total summed across D99 layers), then relabel.

    Returns
    -------
    arr : np.ndarray, shape (n_region, n_layer, n_ctype)
    ctype_names : list[str]
    region_names : list[str]
    layers : list[str]
    """
    raw_arr, raw_ctypes, raw_regions, layers = load_raw_counts(data_root)
    if ctype_map is None:
        ctype_map = load_ctype_map(data_root)

    # Pre-aggregate counts across layers (needed for by_region mode).
    # Built lazily from the first layer's level-aggregated columns, since
    # ctype_ratio_agg() remaps raw_ctypes -> level-aggregated labels (e.g.
    # subclass), which don't match raw_ctypes.
    agg_counts = None
    for l in range(len(layers)):
        data = pd.DataFrame(raw_arr[:, l, :], index=raw_regions, columns=raw_ctypes)
        if atlas != 'D99':
            data = vol_relabel('D99', atlas, data, method='sum',
                               cross_species=True).round()
        data = ctype_ratio_agg(data, map_df=ctype_map, key=level)
        data = data.sort_index().fillna(0)
        agg_counts = data.copy() if agg_counts is None else agg_counts.add(data, fill_value=0)

    data_list, region_names, ctype_names = [], None, None
    for l in range(len(layers)):
        data = pd.DataFrame(raw_arr[:, l, :], index=raw_regions, columns=raw_ctypes)

        # prop_order='before': normalize in raw D99 space (by the D99 per-region
        # layer total, summed across layers) *before* relabeling/aggregating,
        # so the denominator is the D99 total, not the atlas-space total.
        pre_normalize = (return_ratio and prop_mode == 'by_region'
                        and prop_order == 'before')
        if pre_normalize:
            layer_total_d99 = pd.DataFrame(raw_arr.sum(axis=1),
                                           index=raw_regions, columns=raw_ctypes)
            data = data.div(layer_total_d99, axis=0)

        if atlas != 'D99':
            data = vol_relabel('D99', atlas, data, method='sum',
                               cross_species=True)
            if not pre_normalize:
                data = data.round()
        data = ctype_ratio_agg(data, map_df=ctype_map, key=level)

        if return_ratio and not pre_normalize:
            if prop_mode == 'by_layer':
                data = data.div(data.sum(axis=1), axis=0)
            elif prop_mode == 'by_region':
                if prop_order == 'after':
                    # Sum across layers first, then normalize per region (current)
                    layer_total = agg_counts.copy()
                    data = data.div(layer_total, axis=0)
                elif prop_order != 'before':
                    raise ValueError(f"Unknown prop_order: {prop_order}")
            else:
                raise ValueError(f"Unknown prop_mode: {prop_mode}")

        data = data.sort_index().fillna(0)
        region_names = set(data.index) if region_names is None \
            else region_names.union(set(data.index))
        if ctype_names is None:
            ctype_names = data.columns.tolist()
        elif not all(data.columns == ctype_names):
            data = data[ctype_names]
        data_list.append(data)

    region_names = sorted(region_names)
    arr = np.stack([d.reindex(index=region_names, fill_value=0).values
                    for d in data_list], axis=0)      # (layer, region, ctype)
    arr = np.transpose(arr, (1, 0, 2))                # (region, layer, ctype)
    return arr, ctype_names, region_names, layers


def load_bigbrain_thickness(regions, atlas='BN', method='mean', data_root=DATA_ROOT):
    """Load BigBrain per-layer cortical thickness, relabeled FGC -> atlas.

    Returns
    -------
    layer_CT : DataFrame (n_region, n_layer) absolute thickness, aligned to `regions`.
    rel_CT   : DataFrame, row-normalised (relative) thickness.
    """
    path = os.path.join(data_root, 'BigBrain', 'layer_thickness_parced.csv')
    layer_CT = pd.read_csv(path, index_col=0)
    if atlas != 'FGC':
        layer_CT = vol_relabel('FGC', atlas, layer_CT, method=method)
    layer_CT = layer_CT.reindex(index=regions)
    rel_CT = layer_CT.div(layer_CT.sum(axis=1), axis=0)
    return layer_CT, rel_CT


def load_mask(ctypes, layers=layer_names_roman, data_root=DATA_ROOT):
    """External laminar-assignment mask (nc2025): True = cell type present in layer.

    Handles both class-level and subclass-level ctypes by mapping subclasses
    to their class-level labels (column 'subclass' in the mapping file).
    """
    path = os.path.join(data_root, 'Spatial', 'mask_by_nc2025.csv')
    mask = pd.read_csv(path, index_col=0)
    mask = mask.iloc[:, :len(layers)]
    mask.columns = layers
    mask = mask.T  # now index=layers, columns=class-level cell types

    if not set(ctypes).issubset(set(mask.columns)):
        # ctypes are subclass-level; map to class via the mapping file
        ctype_map = load_ctype_map(data_root)
        # column 'subclass' holds the class-level labels (Astro, Pvalb, etc.)
        sub_to_class = ctype_map.set_index('cluster')['subclass']
        class_cols = [sub_to_class.get(c, c) for c in ctypes]
        mask = mask.loc[:, class_cols]
        mask.columns = ctypes
    else:
        mask = mask.loc[:, ctypes]

    return mask.astype(bool)


def enrichment_mask(count_arr, layers=layer_names_roman, ctypes=None, thr=0.05):
    """Data-driven laminar mask: True if >thr of a cell type's total mass sits in the layer.

    count_arr : (n_region, n_layer, n_ctype) raw counts, from load_layer_data(..., return_ratio=False).
    """
    lam = count_arr.sum(axis=0)                          # (layer, ctype)
    enrich = pd.DataFrame(lam / lam.sum(axis=0, keepdims=True),
                          index=layers, columns=ctypes)
    return (enrich > thr), enrich


def load_all(atlas='BN', level='subclass', mask_kind='external', prop_mode='by_layer',
             prop_order='after'):
    """Convenience loader: proportions, raw counts, thickness, and mask, all aligned."""
    prop_mat, ctypes, regions, layers = load_layer_data(
        atlas, level, return_ratio=True, prop_mode=prop_mode, prop_order=prop_order)
    count_arr, _, _, _ = load_layer_data(
        atlas, level, return_ratio=False, prop_mode=prop_mode, prop_order=prop_order)
    layer_CT, rel_CT = load_bigbrain_thickness(regions, atlas=atlas)

    if mask_kind == 'external':
        mask = load_mask(ctypes)
    elif mask_kind == 'enrichment':
        mask, _ = enrichment_mask(count_arr, ctypes=ctypes)
    else:
        raise ValueError("mask_kind must be 'external' or 'enrichment'")

    return {
        'prop_mat': prop_mat, 'count_arr': count_arr,
        'ctypes': ctypes, 'regions': regions, 'layers': layers,
        'layer_CT': layer_CT, 'rel_CT': rel_CT, 'mask': mask,
    }
