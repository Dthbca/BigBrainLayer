"""Run target-space, mask-aware reclosure as the primary eight-branch analysis."""

import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from HomoloMap.datasets.layers import (
    load_layer_counts, normalize_layer_composition, relabel_layer_counts,
    fetch_bigbrain_layer_thickness, fetch_laminar_mask, LAYER_KEYS, LAYER_LABELS,
)
from HomoloMap.transforms.layers import make_layer_subcompositions
from HomoloMap.stats import SpinTest, layer_spin_correlation, layer_match_permutation

DATA = Path("/share/user_data/dthbca/public/experiment/BigBrainLayer/dataset")
OUT = Path(os.environ["RECLOSURE_OUT"])
OLD = Path(os.environ["UNCLOSED_RESULTS"])
SOURCE = OUT / "source_data"
OUT.mkdir(parents=True, exist_ok=True); SOURCE.mkdir(exist_ok=True)
SEED, N_SPINS = 42, 1000
N_JOBS = int(os.environ.get("LAYER_N_JOBS", "8"))
ALIASES = dict(zip(LAYER_KEYS, LAYER_LABELS))


def log(x): print(time.strftime("%F %T"), x, flush=True)


def reclose(mapped, normalization, present):
    """Close over exactly the layer–ctype parts retained by the external mask."""
    out = {k: v.copy().astype(float) for k, v in mapped.items()}
    zero_denominators = 0
    if normalization == "within_layer":
        errors = []
        for layer in LAYER_KEYS:
            mask_row = present.loc[ALIASES[layer]].reindex(out[layer].columns).fillna(False).astype(bool)
            cols = list(mask_row[mask_row].index)
            out[layer].loc[:, ~mask_row] = 0.0
            denom = out[layer][cols].sum(axis=1)
            good = np.isfinite(denom) & (denom > 0)
            zero_denominators += int((~good).sum())
            out[layer].loc[good, cols] = out[layer].loc[good, cols].div(denom[good], axis=0)
            out[layer].loc[~good, cols] = 0.0
            errors.extend((out[layer].loc[good, cols].sum(axis=1) - 1).abs().tolist())
    elif normalization == "within_region_cross_layer":
        errors = []
        for ctype in out[LAYER_KEYS[0]].columns:
            active = [layer for layer in LAYER_KEYS if bool(present.loc[ALIASES[layer], ctype])]
            inactive = [layer for layer in LAYER_KEYS if layer not in active]
            for layer in inactive: out[layer].loc[:, ctype] = 0.0
            if not active: continue
            matrix = pd.concat({layer: out[layer][ctype] for layer in active}, axis=1)
            denom = matrix.sum(axis=1)
            good = np.isfinite(denom) & (denom > 0)
            zero_denominators += int((~good).sum())
            for layer in active:
                out[layer].loc[good, ctype] = matrix.loc[good, layer] / denom[good]
                out[layer].loc[~good, ctype] = 0.0
            closed = pd.concat({layer: out[layer][ctype] for layer in active}, axis=1)
            errors.extend((closed.loc[good].sum(axis=1) - 1).abs().tolist())
    else:
        raise ValueError(normalization)
    return out, {"mask_aware": True, "zero_denominators": zero_denominators,
                 "max_abs_closure_error": float(max(errors, default=0.0)),
                 "n_closed_vectors": len(errors)}


def branch_name(norm, clr, thickness):
    return f"{norm}__clr_{str(clr).lower()}__thickness_{thickness}"


try:
    params = {"primary": "post-relabel mask-aware reclosure", "seed": SEED,
              "n_spins": N_SPINS, "spin_method": "Alexander-Bloch", "n_jobs": N_JOBS,
              "metric": "pearsonr", "correction": "fdr_bh", "n_tested_per_branch": 97,
              "relabel_method": "mean", "unclosed_reference": str(OLD)}
    (OUT / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    raw, mapping = load_layer_counts(DATA, source_atlas="D99", mapping_column="subclass",
                                     unmapped="drop", return_mapping=True)
    mapped_by_norm = {}; relabel_audit = {}; closure_audit = {}
    for norm, mode in (("within_layer", "within_layer"),
                       ("within_region_cross_layer", "within_region")):
        normalized = normalize_layer_composition(raw, mode=mode, zero_policy="zero")
        mapped, audit = relabel_layer_counts(normalized, "D99", "BN", method="mean",
                                             cross_species=True, unknown_labels="drop",
                                             return_audit=True)
        regions, ctypes = mapped["l1"].index, mapped["l1"].columns
        present, mask_audit = fetch_laminar_mask("external", ctypes, data_dir=DATA)
        closed, c_audit = reclose(mapped, norm, present)
        mapped_by_norm[norm] = closed; relabel_audit[norm] = audit; closure_audit[norm] = c_audit
        log(f"RECLOSE {norm} {c_audit}")

    thick = {
        "absolute": fetch_bigbrain_layer_thickness("BN", DATA, False, regions=regions),
        "relative": fetch_bigbrain_layer_thickness("BN", DATA, True, regions=regions),
    }
    present.to_csv(SOURCE / "present_mask.csv")
    thick["absolute"].to_csv(SOURCE / "thickness_absolute.csv")
    thick["relative"].to_csv(SOURCE / "thickness_relative.csv")
    (OUT / "audit.json").write_text(json.dumps({
        "mapping_unresolved": int(mapping.get("n_unresolved_types", 0)),
        "relabel_dropped": {k: v.get("dropped_labels") for k, v in relabel_audit.items()},
        "closure": closure_audit, "mask_present_pairs": int(present.to_numpy().sum()),
    }, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")

    spinner = SpinTest(atlas="BN", n_spins=N_SPINS, method="Alexander-Bloch", seed=SEED)
    summaries=[]; all_results={}; mismatch_rows=[]
    for norm in ("within_layer", "within_region_cross_layer"):
        for clr in (False, True):
            features, _ = make_layer_subcompositions(
                mapped_by_norm[norm], present, transform="clr" if clr else "none",
                zero_method="multiplicative", invalid_rows="drop")
            for tname in ("absolute", "relative"):
                branch = branch_name(norm, clr, tname); log("START " + branch)
                res = layer_spin_correlation(features, thick[tname], spinner, present_mask=present,
                                             metric="pearsonr", correction="fdr_bh", n_jobs=N_JOBS)
                res.insert(0, "branch", branch); res.to_csv(OUT / f"{branch}.csv", index=False)
                whole = layer_match_permutation(features, thick[tname], scheme="whole",
                                                alternative="greater", random_state=SEED,
                                                n_permutations=None)
                mismatch = layer_match_permutation(features, thick[tname], scheme="mismatch",
                                                   alternative="greater", correction="fdr_bh",
                                                   random_state=SEED, n_permutations=None)
                mismatch.insert(0, "branch", branch); mismatch_rows.append(mismatch)
                mismatch.to_csv(OUT / f"{branch}__mismatch.csv", index=False)
                pd.DataFrame({"null": whole["null_distribution"]}).to_csv(
                    SOURCE / f"{branch}__whole_null.csv", index=False)
                r=res.correlation.dropna(); sig=res.p_adjusted.lt(.05)
                summaries.append({"branch":branch,"normalization":norm,"use_clr":clr,
                    "thickness":tname,"n_tested":len(res),"n_fdr_sig":int(sig.sum()),
                    "median_abs_r":r.abs().median(),"mean_abs_r":r.abs().mean(),
                    "mean_signed_r":r.mean(),"max_abs_r":r.abs().max(),
                    "whole_match_stat":whole["observed_stat"],"whole_match_p":whole["p_value"],
                    "whole_n_permutations":whole["n_permutations"],
                    "mismatch_n_fdr_sig":int(mismatch.p_adjusted.lt(.05).sum())})
                all_results[branch]=res
                log(f"DONE {branch} sig={sig.sum()} meanabs={r.abs().mean():.4f} p={whole['p_value']:.6f}")
    summary=pd.DataFrame(summaries).sort_values(["n_fdr_sig","mean_abs_r"],ascending=False)
    summary.to_csv(OUT / "branch_summary.csv", index=False)
    new_long=pd.concat(all_results.values(),ignore_index=True)
    new_long.to_csv(SOURCE / "all_branch_spin_results.csv", index=False)
    pd.concat(mismatch_rows,ignore_index=True).to_csv(SOURCE / "all_branch_mismatch_results.csv",index=False)

    old_summary=pd.read_csv(OLD / "branch_summary.csv").set_index("branch")
    old_long=pd.read_csv(OLD / "source_data" / "all_branch_spin_results.csv")
    comparisons=[]; pair_rows=[]
    for branch,new in all_results.items():
        old=old_long.loc[old_long.branch.eq(branch)].copy()
        merged=old.merge(new,on=["branch","layer","ctype"],suffixes=("_unclosed","_reclosed"))
        rho=merged.correlation_unclosed.corr(merged.correlation_reclosed,method="spearman")
        pear=merged.correlation_unclosed.corr(merged.correlation_reclosed,method="pearson")
        a=set(map(tuple,merged.loc[merged.p_adjusted_unclosed.lt(.05),["layer","ctype"]].values))
        b=set(map(tuple,merged.loc[merged.p_adjusted_reclosed.lt(.05),["layer","ctype"]].values))
        union=a|b; jac=len(a&b)/len(union) if union else 1.0
        row=summary.set_index("branch").loc[branch]; oldrow=old_summary.loc[branch]
        comparisons.append({"branch":branch,"effect_pearson_r":pear,"effect_spearman_r":rho,
            "sig_jaccard":jac,"n_sig_unclosed":len(a),"n_sig_reclosed":len(b),
            "n_sig_change":len(b)-len(a),"mean_abs_r_unclosed":oldrow.mean_abs_r,
            "mean_abs_r_reclosed":row.mean_abs_r,"whole_p_unclosed":oldrow.whole_match_p,
            "whole_p_reclosed":row.whole_match_p,"whole_stat_unclosed":oldrow.whole_match_stat,
            "whole_stat_reclosed":row.whole_match_stat})
        pair_rows.append(merged)
    pd.DataFrame(comparisons).to_csv(OUT / "reclosure_vs_unclosed_summary.csv",index=False)
    pd.concat(pair_rows,ignore_index=True).to_csv(SOURCE / "reclosure_vs_unclosed_pairs.csv",index=False)
    (OUT / "SUCCESS").write_text(time.strftime("%F %T"),encoding="utf-8")
    log("ALL PASS best=" + summary.iloc[0].branch)
except Exception:
    (OUT / "FAILED").write_text(traceback.format_exc(),encoding="utf-8")
    traceback.print_exc(); raise
