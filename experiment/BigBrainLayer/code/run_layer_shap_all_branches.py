"""Out-of-fold Ridge-SHAP comparison for all eight layer pipelines."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from HomoloMap.datasets.layers import (
    LAYER_KEYS, LAYER_LABELS, fetch_bigbrain_layer_thickness,
    fetch_laminar_mask, load_layer_counts, normalize_layer_composition,
    relabel_layer_counts,
)
from HomoloMap.transforms.layers import make_layer_subcompositions


CLASS_DEF = {
    "Excitatory": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3",
                   "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
    "Inhibitory": ["Lamp5_Lhx6", "Lamp5", "Pax6", "Sncg", "Vip", "Sst",
                   "Pvalb", "Chandelier"],
    "Non-neuron": ["Astro", "Oligo", "OPC", "Micro-PVM", "Endo", "VLMC"],
}


def branch_name(normalization, clr, relative):
    return (f"{normalization}__clr_{str(clr).lower()}__"
            f"thickness_{'relative' if relative else 'absolute'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package))
    args.output.mkdir(parents=True, exist_ok=True)

    raw, mapping = load_layer_counts(
        args.data, source_atlas="D99", mapping_column="subclass",
        unmapped="drop", return_mapping=True)
    class_map = {ctype: group for group, members in CLASS_DEF.items()
                 for ctype in members}
    thickness = {}
    all_layer, all_ctype, all_class, all_summary = [], [], [], []
    relabel_meta = {}
    alphas = np.logspace(-3, 3, 25)

    for normalization in ("within_layer", "within_region_cross_layer"):
        mode = "within_layer" if normalization == "within_layer" else "within_region"
        normalized = normalize_layer_composition(raw, mode=mode, zero_policy="zero")
        mapped, audit = relabel_layer_counts(
            normalized, "D99", "BN", method="mean", cross_species=True,
            unknown_labels="drop", return_audit=True)
        relabel_meta[normalization] = audit["dropped_labels"]
        regions, ctypes = mapped["l1"].index, mapped["l1"].columns
        present, _ = fetch_laminar_mask("external", ctypes, data_dir=args.data)
        for relative in (False, True):
            key = "relative" if relative else "absolute"
            if key not in thickness:
                thickness[key] = fetch_bigbrain_layer_thickness(
                    "BN", args.data, relative=relative, regions=regions)
            for clr in (False, True):
                branch = branch_name(normalization, clr, relative)
                features, _ = make_layer_subcompositions(
                    mapped, present, transform="clr" if clr else "none",
                    zero_method="multiplicative", invalid_rows="drop")
                branch_layer_rows = []
                for layer, label in zip(LAYER_KEYS, LAYER_LABELS):
                    X, y = features[layer].align(thickness[key][label], join="inner", axis=0)
                    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
                    X, y = X.loc[valid], y.loc[valid]
                    values = pd.DataFrame(0.0, index=X.index, columns=X.columns)
                    predictions = pd.Series(index=X.index, dtype=float)
                    fold_alphas = []
                    split = KFold(n_splits=5, shuffle=True, random_state=42)
                    for train, test in split.split(X):
                        scaler = StandardScaler().fit(X.iloc[train])
                        xtr, xte = scaler.transform(X.iloc[train]), scaler.transform(X.iloc[test])
                        model = RidgeCV(alphas=alphas).fit(xtr, y.iloc[train])
                        explainer = shap.LinearExplainer(model, xtr)
                        values.iloc[test] = np.asarray(explainer(xte).values)
                        predictions.iloc[test] = model.predict(xte)
                        fold_alphas.append(float(model.alpha_))
                    r2 = float(r2_score(y, predictions))
                    pearson = float(np.corrcoef(y, predictions)[0, 1])
                    all_layer.append({"branch": branch, "normalization": normalization,
                                      "clr": clr, "thickness_relative": relative,
                                      "layer": layer, "layer_label": label, "n_roi": len(X),
                                      "n_features": X.shape[1], "oof_r2": r2,
                                      "oof_pearson_r": pearson,
                                      "median_alpha": float(np.median(fold_alphas))})
                    means = values.abs().mean(axis=0)
                    for ctype, value in means.items():
                        branch_layer_rows.append({"layer": layer, "ctype": ctype,
                                                  "mean_abs_shap": float(value)})

                ctype_df = (pd.DataFrame(branch_layer_rows).groupby("ctype", as_index=False)
                            .mean(numeric_only=True).sort_values("mean_abs_shap", ascending=False))
                ctype_df["relative_contribution"] = ctype_df.mean_abs_shap / ctype_df.mean_abs_shap.sum()
                ctype_df["class"] = ctype_df.ctype.map(class_map).fillna("Other")
                for row in ctype_df.to_dict("records"):
                    all_ctype.append({"branch": branch, **row})
                class_df = ctype_df.groupby("class", as_index=False).agg(
                    relative_contribution=("relative_contribution", "sum"), n_ctype=("ctype", "size"))
                for row in class_df.to_dict("records"):
                    all_class.append({"branch": branch, **row})
                perf = [r for r in all_layer if r["branch"] == branch]
                rel = ctype_df.relative_contribution.to_numpy()
                top = ctype_df.head(5)
                all_summary.append({
                    "branch": branch, "normalization": normalization, "clr": clr,
                    "thickness_relative": relative,
                    "mean_oof_r2": float(np.mean([r["oof_r2"] for r in perf])),
                    "median_oof_r2": float(np.median([r["oof_r2"] for r in perf])),
                    "mean_oof_pearson_r": float(np.mean([r["oof_pearson_r"] for r in perf])),
                    "top1_shap_share": float(rel.max()), "top5_shap_share": float(top.relative_contribution.sum()),
                    "shap_entropy": float(-(rel * np.log(rel + 1e-15)).sum()),
                    "effective_n": float(np.exp(-(rel * np.log(rel + 1e-15)).sum())),
                    "top_ctype": str(ctype_df.iloc[0].ctype),
                    "top5_ctypes": "; ".join(top.ctype.tolist()),
                })

    pd.DataFrame(all_layer).to_csv(args.output / "shap_all_branch_layer_performance.csv", index=False)
    pd.DataFrame(all_ctype).to_csv(args.output / "shap_all_branch_ctype.csv", index=False)
    pd.DataFrame(all_class).to_csv(args.output / "shap_all_branch_class.csv", index=False)
    summary = pd.DataFrame(all_summary).sort_values("mean_oof_r2", ascending=False)
    summary.to_csv(args.output / "shap_all_branch_summary.csv", index=False)
    metadata = {
        "model": "5-fold out-of-fold RidgeCV", "explainer": "shap.LinearExplainer",
        "random_state": 42, "n_branches": 8,
        "comparison_note": "Compare OOF performance and within-branch normalized SHAP shares; raw SHAP magnitudes differ by target scale.",
        "mapping_unresolved": int(mapping.get("n_unresolved_types", 0)),
        "dropped_labels": relabel_meta,
    }
    (args.output / "shap_all_branch_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": summary.to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
