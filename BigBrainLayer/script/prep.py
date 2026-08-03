from scipy.stats import gmean
import numpy as np
import pandas as pd


def coda_transform(df, mode='CLR', ref=None, pseudocount=1e-8, basis=None):
    """ALR / CLR / ILR transform for compositional data (rows = samples).

    CLR is used for the primary analysis. ILR uses the default SBP (Helmert-type)
    orthonormal basis. Returns (transformed_df, reference_or_basis).
    """
    cols = df.columns.tolist()
    if len(cols) < 2:
        raise ValueError('Need at least two components.')
    data = df[cols] + pseudocount
    D = len(cols)

    if mode.upper() == 'ALR':
        ref_col = (data.median(axis=0).idxmax() if ref is None
                   else cols[ref] if isinstance(ref, int) else ref)
        if ref_col not in cols:
            raise ValueError(f"Reference column '{ref_col}' does not exist.")
        out = pd.DataFrame({c: np.log(data[c] / data[ref_col])
                            for c in cols if c != ref_col}, index=df.index)
        return out, ref_col

    if mode.upper() == 'CLR':
        g = gmean(data, axis=1)
        out = pd.DataFrame({c: np.log(data[c] / g) for c in cols}, index=df.index)
        return out, 'Geometric mean'

    if mode.upper() == 'ILR':
        if basis is not None:
            B = np.asarray(basis)
            if B.shape != (D, D - 1):
                raise ValueError(f'Basis must be ({D}, {D-1}), got {B.shape}')
        else:
            B = np.zeros((D, D - 1))
            for i in range(1, D):
                coef = np.sqrt(i / (i + 1))
                B[:i, i - 1] = coef / i
                B[i,  i - 1] = -coef
        g = gmean(data, axis=1)
        clr = np.log(data / g.reshape(-1, 1))
        out = pd.DataFrame(clr.values @ B, index=df.index,
                           columns=[f'ILR_{i+1}' for i in range(D - 1)])
        return out, B

    raise ValueError("mode must be 'ALR', 'CLR', or 'ILR'")

def clr_features(prop_mat, layers, ctypes, mask=None, use_clr=True, pseudocount=1e-8):
    """Per-layer feature DataFrames used by the coupling tests.

    If mask (layers x ctypes bool, True=present) is given AND use_clr, each layer's
    CLR is computed over ONLY its present cell types (structural zeros excluded so
    they do not pollute the geometric mean). Returns list[n_layer] of (region x k_valid).
    """
    X = []
    for l, layer in enumerate(layers):
        df = pd.DataFrame(prop_mat[:, l, :], columns=ctypes)
        if mask is not None:
            keep = mask.loc[layer]
            df = df.loc[:, keep[keep].index]          # subcomposition: drop absent cols
        X.append(coda_transform(df, mode='CLR', pseudocount=pseudocount)[0]
                 if use_clr else df)
    return X
