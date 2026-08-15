"""Out-of-fold linear SHAP aggregation for the selected layer pipeline."""

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
    normalized = normalize_layer_composition(
        raw, mode="within_region", zero_policy="zero")
    mapped, relabel_audit = relabel_layer_counts(
        normalized, "D99", "BN", method="mean", cross_species=True,
        unknown_labels="drop", return_audit=True)
    regions = mapped["l1"].index
    ctypes = mapped["l1"].columns
    present, mask_audit = fetch_laminar_mask(
        "external", ctypes, data_dir=args.data)
    features, _ = make_layer_subcompositions(
        mapped, present, transform="none", invalid_rows="drop")
    thickness = fetch_bigbrain_layer_thickness(
        "BN", args.data, relative=True, regions=regions)

    alphas = np.logspace(-3, 3, 25)
    split = KFold(n_splits=5, shuffle=True, random_state=42)
    layer_rows, performance = [], []
    for layer, label in zip(LAYER_KEYS, LAYER_LABELS):
        X, y = features[layer].align(thickness[label], join="inner", axis=0)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X.loc[valid], y.loc[valid]
        values = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        predictions = pd.Series(index=X.index, dtype=float)
        fold_alphas = []
        for train, test in split.split(X):
            scaler = StandardScaler().fit(X.iloc[train])
            X_train = scaler.transform(X.iloc[train])
            X_test = scaler.transform(X.iloc[test])
            model = RidgeCV(alphas=alphas).fit(X_train, y.iloc[train])
            explainer = shap.LinearExplainer(model, X_train)
            values.iloc[test] = np.asarray(explainer(X_test).values)
            predictions.iloc[test] = model.predict(X_test)
            fold_alphas.append(float(model.alpha_))
        mean_abs = values.abs().mean(axis=0)
        for ctype, contribution in mean_abs.items():
            layer_rows.append({"layer": layer, "layer_label": label,
                               "ctype": ctype, "mean_abs_shap": contribution,
                               "n_roi": len(X)})
        performance.append({
            "layer": layer, "layer_label": label, "n_roi": len(X),
            "n_features": X.shape[1], "oof_r2": r2_score(y, predictions),
            "oof_pearson_r": np.corrcoef(y, predictions)[0, 1],
            "median_alpha": float(np.median(fold_alphas)),
        })

    layer_df = pd.DataFrame(layer_rows)
    total = layer_df.groupby("ctype", as_index=False).mean(numeric_only=True)
    total = total[["ctype", "mean_abs_shap"]].sort_values(
        "mean_abs_shap", ascending=False)
    total["relative_contribution"] = (
        total.mean_abs_shap / total.mean_abs_shap.sum())
    class_map = {ctype: group for group, members in CLASS_DEF.items()
                 for ctype in members}
    total["class"] = total.ctype.map(class_map).fillna("Other")
    class_total = total.groupby("class", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "sum"),
        relative_contribution=("relative_contribution", "sum"),
        n_ctype=("ctype", "size"),
    ).sort_values("relative_contribution", ascending=False)

    layer_df.to_csv(args.output / "shap_layer_ctype_mean_abs.csv", index=False)
    total.to_csv(args.output / "shap_ctype_total.csv", index=False)
    class_total.to_csv(args.output / "shap_class_total.csv", index=False)
    pd.DataFrame(performance).to_csv(
        args.output / "shap_model_performance.csv", index=False)
    metadata = {
        "branch": "within_region_cross_layer__clr_false__thickness_relative",
        "model": "5-fold out-of-fold RidgeCV",
        "explainer": "shap.LinearExplainer",
        "aggregation": "mean absolute SHAP across held-out ROI, then mean across layers",
        "n_roi": int(len(regions)), "n_cell_types": int(len(ctypes)),
        "mapping_unresolved": int(mapping.get("n_unresolved_types", 0)),
        "dropped_labels": relabel_audit["dropped_labels"],
        "mask": mask_audit,
    }
    (args.output / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "top": total.head(5).to_dict("records"),
                      "performance": performance}, ensure_ascii=False))


if __name__ == "__main__":
    main()
