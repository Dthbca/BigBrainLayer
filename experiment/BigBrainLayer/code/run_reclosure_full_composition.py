"""Execute the branch runner with full-composition target-space reclosure.

The external laminar mask selects statistical tests only; it does not redefine
the composition denominator. This preserves spatial variation for cell types
that are structurally tested in only one layer.
"""
from pathlib import Path

runner = Path("/share/user_data/dthbca/public/experiment/BigBrainLayer/run_reclosure_layer_branches.py")
source = runner.read_text(encoding="utf-8")
start = source.index("def reclose(")
end = source.index("def branch_name", start)
replacement = '''def reclose(mapped, normalization, present):
    """Close the full target-space composition; mask is applied only to tests."""
    out = {k: v.copy().astype(float) for k, v in mapped.items()}
    zero_denominators = 0
    errors = []
    if normalization == "within_layer":
        for layer in LAYER_KEYS:
            denom = out[layer].sum(axis=1)
            good = np.isfinite(denom) & (denom > 0)
            zero_denominators += int((~good).sum())
            out[layer].loc[good] = out[layer].loc[good].div(denom[good], axis=0)
            out[layer].loc[~good] = 0.0
            errors.extend((out[layer].loc[good].sum(axis=1) - 1).abs().tolist())
    elif normalization == "within_region_cross_layer":
        for ctype in out[LAYER_KEYS[0]].columns:
            matrix = pd.concat({layer: out[layer][ctype] for layer in LAYER_KEYS}, axis=1)
            denom = matrix.sum(axis=1)
            good = np.isfinite(denom) & (denom > 0)
            zero_denominators += int((~good).sum())
            for layer in LAYER_KEYS:
                out[layer].loc[good, ctype] = matrix.loc[good, layer] / denom[good]
                out[layer].loc[~good, ctype] = 0.0
            closed = pd.concat({layer: out[layer][ctype] for layer in LAYER_KEYS}, axis=1)
            errors.extend((closed.loc[good].sum(axis=1) - 1).abs().tolist())
    else:
        raise ValueError(normalization)
    return out, {"full_composition": True, "mask_used_for_tests_only": True,
                 "zero_denominators": zero_denominators,
                 "max_abs_closure_error": float(max(errors, default=0.0)),
                 "n_closed_vectors": len(errors)}


'''
code = source[:start] + replacement + source[end:]
exec(compile(code, str(runner), "exec"), {"__name__": "__main__", "__file__": str(runner)})
