#!/usr/bin/env python3
"""Non-laminar macaque cell composition ~ human imaging pipeline.

This runner is intentionally independent from HomoloMap ``run_analysis``.  It
uses the package only for atlas relabelling, BN spin indices and dominance.
All model explanation values are computed out-of-fold on held-out regions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


RATIO = Path("/data100/home/dthbca/project/CellAlign/HomoloMap/datasets/features/SpatialTranscriptomics/ctype_ratio_plot_D99.csv")
DENSITY = Path("/data100/home/dthbca/project/CellAlign/HomoloMap/datasets/features/SpatialTranscriptomics/ctype_density_plot_D99.csv")
MAPPING = Path("/data100/home/dthbca/project/Macaque_ST/notebook/cluster_mapping_dict.csv")
MEG = Path("/data100/home/dthbca/project/CellAlign/tmp/cluster_results/hcps1200_meg_fgc.csv")
ENIGMA = Path("/data100/home/dthbca/project/CellAlign/HomoloMap/datasets/features/enigma_fgc_smoothed.csv")
BN_LABELS = np.arange(1, 210, 2, dtype=int)
MEG_OUTCOMES = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2"]
ALPHAS = np.logspace(-3, 3, 25)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--levels", nargs="+", default=["subclass", "cluster"], choices=["subclass", "cluster"])
    p.add_argument("--feature-types", nargs="+", default=["ratio", "density"], choices=["ratio", "density"])
    p.add_argument("--transforms", nargs="+", default=["none", "clr"], choices=["none", "clr"])
    p.add_argument("--phenotypes", nargs="+", default=["meg", "enigma"], choices=["meg", "enigma"])
    p.add_argument("--outcome-limit", type=int, default=None)
    p.add_argument("--n-spins", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--dominance-samples", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def close_rows(x: pd.DataFrame) -> pd.DataFrame:
    sums = x.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"Cannot close {int((sums <= 0).sum())} non-positive rows")
    return x.div(sums, axis=0)


def multiplicative_clr(x: pd.DataFrame):
    """Multiplicative zero replacement followed by CLR.

    Per row delta=min(0.65*minimum positive component, 1e-6); non-zero
    components are multiplicatively contracted to retain unit closure.
    """
    a = close_rows(x).to_numpy(float)
    replaced = np.empty_like(a)
    deltas, zeros = [], []
    for i, row in enumerate(a):
        z = row == 0
        nz = ~z
        if not nz.any():
            raise ValueError("All-zero composition")
        delta = min(0.65 * float(row[nz].min()), 1e-6)
        out = row.copy()
        out[z] = delta
        if z.sum():
            out[nz] *= (1.0 - z.sum() * delta) / out[nz].sum()
        out /= out.sum()
        replaced[i] = out
        deltas.append(delta)
        zeros.append(int(z.sum()))
    loga = np.log(replaced)
    clr = loga - loga.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=x.index, columns=x.columns), {
        "method": "multiplicative_zero_replacement_then_clr",
        "delta_rule": "min(0.65 * row minimum positive closed component, 1e-6)",
        "zero_count": int(sum(zeros)), "rows_with_zero": int(sum(v > 0 for v in zeros)),
        "delta_min": float(min(deltas)), "delta_max": float(max(deltas)),
    }


def load_mapping_audit(raw: pd.DataFrame, mapping: pd.DataFrame, source: str):
    mp = mapping.set_index("plot")
    mapped = [c for c in raw.columns if str(c) in mp.index]
    unmapped = [c for c in raw.columns if str(c) not in mp.index]
    total = float(raw.to_numpy().sum())
    unmapped_mass = float(raw[unmapped].to_numpy().sum()) if unmapped else 0.0
    by_roi = raw[unmapped].sum(axis=1) / raw.sum(axis=1) if unmapped else pd.Series(0, index=raw.index)
    audit = {
        "source": source, "raw_shape": list(raw.shape), "mapped_features": len(mapped),
        "unmapped_features": len(unmapped), "unmapped_mass_fraction": unmapped_mass / total,
        "unmapped_roi_fraction_median": float(by_roi.median()),
        "unmapped_roi_fraction_max": float(by_roi.max()), "unmapped": list(map(str, unmapped)),
    }
    return mapped, unmapped, audit


def relabel_cell_features(feature_type: str, level: str, transform: str):
    from HomoloMap.parcellation import vol_relabel  # verified legacy cross-species path
    from HomoloMap.datasets.atlas import fetch_annot
    from HomoloMap.transforms.atlas import load_volume_atlas
    path = RATIO if feature_type == "ratio" else DENSITY
    raw = pd.read_csv(path, index_col=0)
    raw.index = raw.index.astype(int)
    mapping = pd.read_csv(MAPPING, dtype=str)
    if mapping["plot"].duplicated().any():
        raise ValueError("mapping plot key is not unique")
    mapped, _, audit = load_mapping_audit(raw, mapping, str(path))
    lookup = mapping.set_index("plot")[level]
    x = raw[mapped].copy()
    x.columns = [lookup.loc[str(c)] for c in mapped]
    x = x.T.groupby(level=0, sort=True).sum().T
    if feature_type == "ratio":
        x = close_rows(x)  # D99 mapped-composition closure before relabel
    source_path, source_info = fetch_annot(atlas="D99", annot=True)
    valid_labels = set(load_volume_atlas(
        source_path, source_info, hemisphere="left")["roi_labels"])
    dropped_labels = sorted(set(x.index) - valid_labels)
    x = x.drop(index=dropped_labels, errors="ignore")
    xb = vol_relabel(src="D99", trg="BN", data=x, cross_species=True, method="mean")
    xb = xb.reindex(BN_LABELS)
    if xb.isna().all(axis=1).any():
        raise ValueError("D99->BN produced all-NA ROI rows")
    if feature_type == "ratio":
        xb = close_rows(xb.fillna(0))  # required target-space reclosure
    else:
        xb = xb.fillna(0)  # density is deliberately not closed
    clr_audit = None
    if transform == "clr":
        if feature_type != "ratio":
            raise ValueError("CLR is only valid for ratio branches")
        xb, clr_audit = multiplicative_clr(xb)
    audit.update({"level": level, "transform": transform, "aggregated_features": x.shape[1],
                  "bn_shape": list(xb.shape), "bn_labels": xb.index.tolist(),
                  "d99_labels_dropped_before_relabel": dropped_labels,
                  "ratio_reclosed_D99_and_BN": feature_type == "ratio",
                  "density_closed": False, "clr": clr_audit})
    return xb, audit, mapping


def load_phenotype(name: str):
    if name == "meg":
        y = pd.read_csv(MEG, index_col=0)
        y = y.loc[:, MEG_OUTCOMES]
        if len(y) != 105:
            raise ValueError(f"MEG expected 105 rows, got {len(y)}")
        original_index = y.index.tolist()
        y.index = BN_LABELS
        audit = {"source": str(MEG), "shape": list(y.shape), "original_index_head": original_index[:5],
                 "forced_bn_odd_labels": True, "na_total": int(y.isna().sum().sum()),
                 "na_by_outcome": y.isna().sum().astype(int).to_dict()}
        return y, audit
    from HomoloMap.parcellation import surf_relabel
    y0 = pd.read_csv(ENIGMA, index_col=0)
    y = surf_relabel(data=y0, src="FGC", trg="BN", cross_species=False, method="mean")
    y = y.reindex(BN_LABELS)
    audit = {"source": str(ENIGMA), "source_shape": list(y0.shape), "shape": list(y.shape),
             "relabel": "FGC_smoothed->BN surf_relabel mean", "na_total": int(y.isna().sum().sum())}
    return y, audit


def make_outer_splits(index, folds, seed):
    """Try BN lobe groups; deterministically fall back to shuffled KFold."""
    audit = {"requested": "BN_lobe_GroupKFold", "used": None, "reason": None}
    groups = None
    try:
        from HomoloMap.datasets import fetch_annot
        _, annot = fetch_annot(atlas="BN", annot=True)
        # Accept only an explicit lobe-like field; never infer lobes from ROI order.
        if isinstance(annot, pd.DataFrame):
            col = next((c for c in annot.columns if "lobe" in str(c).lower()), None)
            if col is not None:
                s = annot[col]
                groups = pd.Series(s.values, index=pd.Index(annot.index).astype(int)).reindex(index).values
    except Exception as exc:
        audit["reason"] = f"BN annotation unavailable: {type(exc).__name__}: {exc}"
    if groups is not None and pd.notna(groups).all() and len(np.unique(groups)) >= folds:
        audit.update({"used": "GroupKFold", "n_groups": int(len(np.unique(groups)))})
        return list(GroupKFold(folds).split(np.arange(len(index)), groups=groups)), audit
    if audit["reason"] is None:
        audit["reason"] = "No explicit BN lobe field with >= folds groups; ROI-order inference prohibited"
    audit["used"] = "KFold(shuffle=True, random_state=seed)"
    return list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(len(index)))), audit


def spin_pair(x, y, spins):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan, 0
    r = pearsonr(x[mask], y[mask])[0]
    null = []
    for k in range(spins.shape[1]):
        yp = y[spins[:, k]]
        m = np.isfinite(x) & np.isfinite(yp)
        if m.sum() >= 4 and np.nanstd(yp[m]) > 0:
            null.append(pearsonr(x[m], yp[m])[0])
    null = np.asarray(null)
    p = (1 + np.sum(np.abs(null) >= abs(r))) / (len(null) + 1) if len(null) else np.nan
    return r, p, len(null)


def outcome_analysis(X, y, spins, splits, feature_names, seed):
    good = np.isfinite(y)
    idx = np.flatnonzero(good)
    pred = np.full(len(idx), np.nan)
    shap_abs = np.zeros((len(idx), X.shape[1]))
    valid_position = {row: pos for pos, row in enumerate(idx)}
    fold_rows = []
    for fold, (tr_full, te_full) in enumerate(splits):
        tr = np.asarray([v for v in tr_full if good[v]], dtype=int)
        te = np.asarray([v for v in te_full if good[v]], dtype=int)
        if len(tr) < 6 or len(te) == 0:
            raise ValueError(f"Fold {fold} too small after outcome missingness: train={len(tr)}, test={len(te)}")
        te0 = np.asarray([valid_position[v] for v in te], dtype=int)
        sx = StandardScaler().fit(X[tr])
        Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
        model = RidgeCV(alphas=ALPHAS, cv=min(5, len(tr))).fit(Xtr, y[tr])
        pred[te0] = model.predict(Xte)
        # Exact interventional linear SHAP around the fold-training mean (zero after scaling).
        vals = Xte * model.coef_[None, :]
        shap_abs[te0] = np.abs(vals)
        fold_rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "alpha": float(model.alpha_)})
    yy = y[idx]
    perf = {"n": len(idx), "oof_r2": float(r2_score(yy, pred)),
            "oof_pearson": float(pearsonr(yy, pred)[0]), "oof_mae": float(mean_absolute_error(yy, pred))}
    ma = shap_abs.mean(axis=0); total = float(ma.sum())
    shap_rows = pd.DataFrame({"feature": feature_names, "mean_abs_shap": ma,
                              "relative_shap": ma / total if total > 0 else np.nan,
                              "total_mean_abs_shap": total})
    pred_df = pd.DataFrame({"row_position": idx, "observed": yy, "prediction": pred})
    # Total spatial significance: full-data scaled RidgeCV, fixed selected alpha under spins.
    sx = StandardScaler().fit(X[idx]); xs = sx.transform(X[idx]); ys = zscore(yy)
    full = RidgeCV(alphas=ALPHAS, cv=min(5, len(idx))).fit(xs, ys)
    obs = float(full.score(xs, ys)); null = []
    for k in range(spins.shape[1]):
        yp_full = y[spins[:, k]]; m = np.isfinite(yp_full)
        # MEG missingness rotates, so use each permutation's valid BN rows.
        sm = StandardScaler().fit(X[m]); xm = sm.transform(X[m]); yp = zscore(yp_full[m])
        null.append(Ridge(alpha=float(full.alpha_)).fit(xm, yp).score(xm, yp))
    total_model = {"full_r2": obs, "alpha": float(full.alpha_),
                   "spin_p": float((1 + np.sum(np.asarray(null) >= obs)) / (len(null) + 1)),
                   "n_spin": len(null)}
    return perf, shap_rows, pd.DataFrame(fold_rows), pred_df, total_model


def dominance_one(X, y, features, samples, jobs, smoke=False):
    from HomoloMap.stats.analysis import get_dominance_stats
    m = np.isfinite(y)
    xs = StandardScaler().fit_transform(X[m]); ys = zscore(y[m])
    p = X.shape[1]
    method = "incremental" if smoke else ("full" if p <= 15 else ("approximate" if p <= 30 else "incremental"))
    t0 = time.time()
    try:
        metrics, _ = get_dominance_stats(xs, ys, method=method,
                                         n_samples=min(samples, 50) if smoke else samples,
                                         n_jobs=jobs, verbose=False)
        val = np.asarray(metrics["total_dominance"], float)
        rows = pd.DataFrame({"feature": features, "total_dominance": val,
                             "relative_dominance": val / val.sum() if val.sum() else np.nan,
                             "full_r2": metrics.get("full_r_sq", np.nan), "method": method,
                             "runtime_sec": time.time() - t0, "status": "ok", "reason": ""})
    except Exception as exc:
        rows = pd.DataFrame({"feature": features, "total_dominance": np.nan,
                             "relative_dominance": np.nan, "full_r2": np.nan, "method": method,
                             "runtime_sec": time.time() - t0, "status": "not_run_with_reason",
                             "reason": f"{type(exc).__name__}: {exc}"})
    return rows


def main():
    a = args_parser(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    from HomoloMap.stats.nulls import SpinTest
    levels = ["subclass"] if a.smoke else a.levels
    ftypes = ["ratio"] if a.smoke else a.feature_types
    transforms = ["none"] if a.smoke else a.transforms
    phenotypes = ["meg"] if a.smoke else a.phenotypes
    metadata = {"started": pd.Timestamp.now().isoformat(), "argv": sys.argv, "python": sys.version,
                "platform": platform.platform(), "bn_labels": BN_LABELS.tolist(), "branches": [],
                "inference": {
                    "pairwise": "two-sided Pearson Alexander-Bloch spin; BH within each outcome's cell-feature family",
                    "total": "full-data Ridge R2 spatial spin; BH across outcomes within branch",
                    "prediction": "5-fold outer OOF Ridge; scaler and RidgeCV alpha selected inside training fold",
                    "shap": "held-out exact linear Ridge SHAP = standardized held-out value times fold coefficient",
                    "clr_interpretation": "CLR columns are dependent log-ratio coordinates; attribution is conditional on the full composition and is not an independent abundance effect",
                }}
    branch_summary = []
    spinner = SpinTest(atlas="BN", n_spins=a.n_spins, method="Alexander-Bloch", seed=a.seed)
    for ft in ftypes:
        for tr in transforms:
            if ft == "density" and tr == "clr":
                continue
            for level in levels:
                branch = f"{ft}_{tr}_{level}"; bd = out / branch; bd.mkdir(exist_ok=True)
                Xdf, map_audit, mapping = relabel_cell_features(ft, level, tr)
                write_json(bd / "mapping_audit.json", map_audit)
                pd.DataFrame({"unmapped_feature": map_audit["unmapped"]}).to_csv(bd / "unmapped_features.csv", index=False)
                mapping.to_csv(bd / "mapping_used.csv", index=False); Xdf.to_csv(bd / "X_bn.csv")
                split_placeholder, split_audit = make_outer_splits(Xdf.index, a.folds, a.seed)
                write_json(bd / "cv_audit.json", split_audit)
                spin_all=[]; perf_all=[]; shap_all=[]; dom_all=[]; total_all=[]; folds_all=[]; preds_all=[]
                for ph in phenotypes:
                    Y, ya = load_phenotype(ph)
                    if a.outcome_limit: Y = Y.iloc[:, :a.outcome_limit]
                    Y.to_csv(bd / f"Y_{ph}_bn.csv"); write_json(bd / f"Y_{ph}_audit.json", ya)
                    for outcome in Y.columns:
                        y=Y[outcome].to_numpy(float); X=Xdf.to_numpy(float)
                        pairs = Parallel(n_jobs=a.n_jobs)(delayed(spin_pair)(X[:,j],y,spinner.spins) for j in range(X.shape[1]))
                        pvals=np.array([q[1] for q in pairs]); adj=np.full_like(pvals,np.nan); ok=np.isfinite(pvals)
                        if ok.any(): adj[ok]=multipletests(pvals[ok],method="fdr_bh")[1]
                        for j,(r,p,nv) in enumerate(pairs): spin_all.append({"phenotype":ph,"outcome":outcome,"feature":Xdf.columns[j],"pearson_r":r,"spin_p":p,"spin_q_bh":adj[j],"n_valid_spins":nv})
                        perf,sh,fo,pr,tm=outcome_analysis(X,y,spinner.spins,split_placeholder,list(Xdf.columns),a.seed)
                        perf_all.append({"phenotype":ph,"outcome":outcome,**perf}); sh.insert(0,"outcome",outcome); sh.insert(0,"phenotype",ph); shap_all.append(sh)
                        fo.insert(0,"outcome",outcome); fo.insert(0,"phenotype",ph); folds_all.append(fo)
                        pr.insert(0,"roi",Y.index.to_numpy()[pr.pop("row_position").astype(int)]); pr.insert(0,"outcome",outcome); pr.insert(0,"phenotype",ph); preds_all.append(pr)
                        total_all.append({"phenotype":ph,"outcome":outcome,**tm})
                        dm=dominance_one(X,y,list(Xdf.columns),a.dominance_samples,a.n_jobs,a.smoke); dm.insert(0,"outcome",outcome); dm.insert(0,"phenotype",ph); dom_all.append(dm)
                total=pd.DataFrame(total_all); total["spin_q_bh_across_outcomes"]=multipletests(total.spin_p,method="fdr_bh")[1]
                pd.DataFrame(spin_all).to_csv(bd/"spin.csv",index=False); pd.DataFrame(perf_all).to_csv(bd/"performance.csv",index=False)
                pd.concat(shap_all).to_csv(bd/"oof_shap.csv",index=False); pd.concat(dom_all).to_csv(bd/"dominance.csv",index=False)
                total.to_csv(bd/"total_models.csv",index=False); pd.concat(folds_all).to_csv(bd/"folds.csv",index=False); pd.concat(preds_all).to_csv(bd/"oof_predictions.csv",index=False)
                branch_summary.append({"branch":branch,"n_roi":len(Xdf),"n_features":Xdf.shape[1],"n_outcomes":len(total),"mapping_fraction":map_audit["mapped_features"]/(map_audit["mapped_features"]+map_audit["unmapped_features"]),"mean_oof_r2":pd.DataFrame(perf_all).oof_r2.mean(),"min_total_q":total["spin_q_bh_across_outcomes"].min()})
                metadata["branches"].append({"branch":branch,"mapping_audit":map_audit,"cv":split_audit})
    pd.DataFrame(branch_summary).to_csv(out/"branch_summary.csv",index=False)
    metadata["completed"]=pd.Timestamp.now().isoformat(); metadata["status"]="success"; write_json(out/"metadata.json",metadata)
    print(json.dumps({"status":"success","output":str(out),"branches":branch_summary},indent=2))


if __name__ == "__main__":
    main()
