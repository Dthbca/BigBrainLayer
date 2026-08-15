"""Layer-specific OOF Ridge-SHAP for the reclosed primary pipeline."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from HomoloMap.datasets.layers import (
    LAYER_KEYS, LAYER_LABELS, fetch_bigbrain_layer_thickness,
    fetch_laminar_mask, load_layer_counts, normalize_layer_composition,
    relabel_layer_counts,
)
from HomoloMap.transforms.layers import make_layer_subcompositions

CLASS_DEF = {
    "Excitatory": ["L2/3 IT","L4 IT","L5 IT","L6 IT","L6 IT Car3","L5 ET","L5/6 NP","L6 CT","L6b"],
    "Inhibitory": ["Lamp5_Lhx6","Lamp5","Pax6","Sncg","Vip","Sst","Pvalb","Chandelier"],
    "Non-neuron": ["Astro","Oligo","OPC","Micro-PVM","Endo","VLMC"],
}

def reclose_full_composition(mapped):
    out = {k: v.copy().astype(float) for k, v in mapped.items()}
    audits = []
    for ctype in out["l1"].columns:
        matrix = pd.concat({layer: out[layer][ctype] for layer in LAYER_KEYS}, axis=1)
        denominator = matrix.sum(axis=1)
        good = np.isfinite(denominator) & (denominator > 0)
        for layer in LAYER_KEYS:
            out[layer].loc[good, ctype] = matrix.loc[good, layer] / denominator[good]
            out[layer].loc[~good, ctype] = 0.0
        closed = pd.concat({layer: out[layer][ctype] for layer in LAYER_KEYS}, axis=1).sum(axis=1)
        audits.append(float(np.max(np.abs(closed.loc[good] - 1))) if good.any() else 0.0)
    return out, {"max_abs_closure_error": max(audits, default=0.0), "denominator": "all 23 mapped subclasses"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); sys.path.insert(0, str(a.package)); a.output.mkdir(parents=True, exist_ok=True)

    raw, mapping = load_layer_counts(a.data, source_atlas="D99", mapping_column="subclass", unmapped="drop", return_mapping=True)
    normalized = normalize_layer_composition(raw, mode="within_region", zero_policy="zero")
    mapped, relabel_audit = relabel_layer_counts(normalized, "D99", "BN", method="mean", cross_species=True,
                                                 unknown_labels="drop", return_audit=True)
    mapped, closure_audit = reclose_full_composition(mapped)
    regions, ctypes = mapped["l1"].index, mapped["l1"].columns
    present, mask_audit = fetch_laminar_mask("external", ctypes, data_dir=a.data)
    features, feature_audit = make_layer_subcompositions(mapped, present, transform="none", invalid_rows="drop")
    thickness = fetch_bigbrain_layer_thickness("BN", a.data, relative=True, regions=regions)

    class_map = {c: group for group, members in CLASS_DEF.items() for c in members}
    alphas = np.logspace(-3, 3, 25)
    splitter = KFold(5, shuffle=True, random_state=42)
    contribution_rows, class_rows, perf_rows, fold_rows, prediction_rows = [], [], [], [], []

    for layer, label in zip(LAYER_KEYS, LAYER_LABELS):
        X, y = features[layer].align(thickness[label], join="inner", axis=0)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X.loc[valid], y.loc[valid]
        shap_values = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        predictions = pd.Series(index=X.index, dtype=float)
        selected_alphas = []
        for fold, (train, test) in enumerate(splitter.split(X), start=1):
            scaler = StandardScaler().fit(X.iloc[train])
            xtr, xte = scaler.transform(X.iloc[train]), scaler.transform(X.iloc[test])
            model = RidgeCV(alphas=alphas).fit(xtr, y.iloc[train])
            values = np.asarray(shap.LinearExplainer(model, xtr)(xte).values)
            shap_values.iloc[test] = values
            predictions.iloc[test] = model.predict(xte)
            selected_alphas.append(float(model.alpha_))
            fold_mean = np.abs(values).mean(axis=0)
            fold_total = fold_mean.sum()
            for ctype, value in zip(X.columns, fold_mean):
                fold_rows.append({"layer":layer,"layer_label":label,"fold":fold,"ctype":ctype,
                                  "mean_abs_shap":float(value),"relative_contribution":float(value/fold_total)})
        mean_abs = shap_values.abs().mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        total = float(mean_abs.sum())
        rows = []
        for ctype in X.columns:
            rows.append({"layer":layer,"layer_label":label,"ctype":ctype,"class":class_map.get(ctype,"Other"),
                         "mean_abs_shap":float(mean_abs[ctype]),"mean_signed_shap":float(mean_signed[ctype]),
                         "relative_contribution":float(mean_abs[ctype]/total),"n_roi":len(X)})
        rows = sorted(rows, key=lambda z: z["mean_abs_shap"], reverse=True)
        for rank, row in enumerate(rows, start=1): row["rank"] = rank
        contribution_rows.extend(rows)
        layer_table = pd.DataFrame(rows)
        for group, d in layer_table.groupby("class"):
            class_rows.append({"layer":layer,"layer_label":label,"class":group,"n_ctype":len(d),
                               "relative_contribution":float(d.relative_contribution.sum()),
                               "mean_abs_shap":float(d.mean_abs_shap.sum())})
        perf_rows.append({"layer":layer,"layer_label":label,"n_roi":len(X),"n_features":X.shape[1],
                          "oof_r2":float(r2_score(y,predictions)),
                          "oof_pearson_r":float(np.corrcoef(y,predictions)[0,1]),
                          "oof_mae":float(mean_absolute_error(y,predictions)),
                          "median_alpha":float(np.median(selected_alphas)),
                          "top_ctype":rows[0]["ctype"],"top1_share":rows[0]["relative_contribution"],
                          "top5_share":float(sum(r["relative_contribution"] for r in rows[:5]))})
        prediction_rows.extend({"layer":layer,"layer_label":label,"roi":roi,"observed":float(y.loc[roi]),
                                "predicted":float(predictions.loc[roi])} for roi in X.index)

    contributions = pd.DataFrame(contribution_rows)
    folds = pd.DataFrame(fold_rows)
    stability = (folds.groupby(["layer","layer_label","ctype"], as_index=False)
                 .agg(fold_mean_relative=("relative_contribution","mean"),
                      fold_sd_relative=("relative_contribution","std"),
                      fold_min_relative=("relative_contribution","min"),
                      fold_max_relative=("relative_contribution","max")))
    contributions.to_csv(a.output/"layer_specific_shap_contributions.csv", index=False)
    pd.DataFrame(class_rows).to_csv(a.output/"layer_specific_shap_classes.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(a.output/"layer_specific_model_performance.csv", index=False)
    folds.to_csv(a.output/"layer_specific_shap_by_fold.csv", index=False)
    stability.to_csv(a.output/"layer_specific_shap_stability.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(a.output/"layer_specific_oof_predictions.csv", index=False)
    metadata = {"pipeline":"within-region cross-layer -> BN mean relabel -> full-composition reclosure -> no CLR -> relative thickness",
                "model":"per-layer 5-fold OOF RidgeCV","explainer":"shap.LinearExplainer","random_state":42,
                "alphas":"logspace(-3,3,25)","n_roi":int(len(regions)),"n_mapped_subclasses":int(len(ctypes)),
                "mapping_unresolved":int(mapping.get("n_unresolved_types",0)),"closure_audit":closure_audit,
                "dropped_labels":relabel_audit["dropped_labels"],"mask":mask_audit,"feature_audit":str(feature_audit)}
    (a.output/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","performance":perf_rows,
                      "top3":{l:contributions[contributions.layer.eq(l)].head(3).ctype.tolist() for l in LAYER_KEYS}},ensure_ascii=False))

if __name__ == "__main__": main()
