"""Compare layer cell-type feature construction strategies on real data."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.package))
    from HomoloMap.datasets.layers import (
        LAYER_KEYS, LAYER_LABELS, fetch_laminar_mask, load_layer_counts,
        normalize_layer_composition, relabel_layer_counts,
    )

    a.output.mkdir(parents=True, exist_ok=True)
    raw, mapping = load_layer_counts(
        a.data, source_atlas="D99", mapping_column="subclass",
        unmapped="drop", return_mapping=True)
    mapped_sum, audit_sum = relabel_layer_counts(
        raw, "D99", "BN", method="sum", cross_species=True,
        unknown_labels="drop", return_audit=True)
    mapped_mean, audit_mean = relabel_layer_counts(
        raw, "D99", "BN", method="mean", cross_species=True,
        unknown_labels="drop", return_audit=True)

    all_rows, closure_rows, mask_rows = [], [], []
    pair_order = [
        ("ratio_then_mean_relabel", "sum_relabel_then_ratio"),
        ("ratio_then_mean_relabel", "mean_relabel_then_ratio"),
        ("sum_relabel_then_ratio", "mean_relabel_then_ratio"),
    ]
    for norm_label, mode in [
        ("within_layer", "within_layer"),
        ("within_region_cross_layer", "within_region"),
    ]:
        ratio_d99 = normalize_layer_composition(raw, mode=mode, zero_policy="zero")
        strategies = {
            "ratio_then_mean_relabel": relabel_layer_counts(
                ratio_d99, "D99", "BN", method="mean", cross_species=True,
                unknown_labels="drop"),
            "sum_relabel_then_ratio": normalize_layer_composition(
                mapped_sum, mode=mode, zero_policy="zero"),
            "mean_relabel_then_ratio": normalize_layer_composition(
                mapped_mean, mode=mode, zero_policy="zero"),
        }
        ctypes = strategies["ratio_then_mean_relabel"]["l1"].columns
        present, mask_audit = fetch_laminar_mask(
            "external", ctypes, data_dir=a.data)
        present = present.reindex(index=LAYER_LABELS, columns=ctypes, fill_value=False).astype(bool)
        n_present_pairs = int(present.to_numpy().sum())
        if n_present_pairs == 0:
            raise ValueError("The laminar mask contains no present layer-cell-type pairs")
        for label in LAYER_LABELS:
            for ctype in ctypes:
                mask_rows.append({"normalization": norm_label, "layer_label": label,
                                  "ctype": ctype, "included": bool(present.loc[label, ctype])})
        for strategy, maps in strategies.items():
            stack = np.stack([maps[k].to_numpy(float) for k in LAYER_KEYS])
            if mode == "within_layer":
                sums = stack.sum(axis=2)
            else:
                sums = stack.sum(axis=0)
            closure_rows.append({
                "normalization": norm_label, "strategy": strategy,
                "finite": bool(np.isfinite(stack).all()),
                "closure_median": float(np.median(sums)),
                "closure_max_abs_error": float(np.max(np.abs(sums - 1))),
            })
        for left, right in pair_order:
            pair = f"{left}__vs__{right}"
            for layer, label in zip(LAYER_KEYS, LAYER_LABELS):
                xdf, ydf = strategies[left][layer].align(
                    strategies[right][layer], join="inner", axis=0)
                xdf, ydf = xdf.align(ydf, join="inner", axis=1)
                for ctype in xdf.columns:
                    if not bool(present.loc[label, ctype]):
                        continue
                    r, n = corr(xdf[ctype], ydf[ctype])
                    all_rows.append({
                        "normalization": norm_label, "pair": pair,
                        "strategy_a": left, "strategy_b": right,
                        "layer": layer, "layer_label": label,
                        "ctype": ctype, "pearson_r": r, "n_roi": n,
                        "mask_kind": "external", "mask_included": True,
                    })

        expected = n_present_pairs * len(pair_order)
        observed = sum(row["normalization"] == norm_label for row in all_rows)
        if observed != expected:
            raise AssertionError(f"Mask-filtered comparison count mismatch for {norm_label}: expected {expected}, observed {observed}")

    result = pd.DataFrame(all_rows)
    valid = result.dropna(subset=["pearson_r"])
    summary = valid.groupby(["normalization", "pair"], as_index=False).agg(
        n_maps=("pearson_r", "size"), minimum_r=("pearson_r", "min"),
        q05_r=("pearson_r", lambda x: x.quantile(.05)),
        median_r=("pearson_r", "median"), mean_r=("pearson_r", "mean"),
        maximum_r=("pearson_r", "max"), n_below_085=("pearson_r", lambda x: int((x < .85).sum())),
    )
    lowest = valid.sort_values("pearson_r").groupby(
        ["normalization", "pair"], as_index=False).head(5)
    result.to_csv(a.output / "feature_strategy_correlations.csv", index=False)
    summary.to_csv(a.output / "feature_strategy_summary.csv", index=False)
    lowest.to_csv(a.output / "feature_strategy_lowest.csv", index=False)
    pd.DataFrame(closure_rows).to_csv(a.output / "feature_strategy_closure_audit.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(a.output / "feature_strategy_mask_audit.csv", index=False)
    metadata = {
        "source": "real macaque layer counts", "source_atlas": "D99",
        "target_atlas": "BN left hemisphere", "mapping_column": "subclass",
        "unmapped": "drop", "mask": mask_audit,
        "mapping_unresolved": int(mapping.get("n_unresolved_types", 0)),
        "dropped_labels_sum": audit_sum["dropped_labels"],
        "dropped_labels_mean": audit_mean["dropped_labels"],
        "comparison_scope": "present layer-cell-type combinations only",
        "mask_applied_before_statistics": True,
        "n_present_layer_ctype_pairs": int(n_present_pairs),
        "n_possible_layer_ctype_pairs": int(len(LAYER_KEYS) * len(ctypes)),
    }
    (a.output / "feature_strategy_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": summary.to_dict("records"),
                      "global_lowest": lowest.head(10).to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()


