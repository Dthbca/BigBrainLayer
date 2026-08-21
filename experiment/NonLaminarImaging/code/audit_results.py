from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging\results")
RUNS = [ROOT / "subclass_main_20260821", ROOT / "cluster_secondary_20260821"]
rows = []
top_rows = []

for run in RUNS:
    for branch in sorted(p for p in run.iterdir() if p.is_dir()):
        spin = pd.read_csv(branch / "spin.csv")
        total = pd.read_csv(branch / "total_models.csv")
        perf = pd.read_csv(branch / "performance.csv")
        shap = pd.read_csv(branch / "oof_shap.csv")
        dom = pd.read_csv(branch / "dominance.csv")
        audit = json.loads((branch / "mapping_audit.json").read_text())
        assert spin["spin_q_bh"].between(0, 1).all()
        assert total["spin_q_bh_across_outcomes"].between(0, 1).all()
        shap_sums = shap.groupby(["phenotype", "outcome"])["relative_shap"].sum()
        dom_ok = dom[dom["status"] == "ok"]
        dom_sums = dom_ok.groupby(["phenotype", "outcome"])["relative_dominance"].sum()
        rows.append({
            "branch": branch.name,
            "n_features": int(audit["aggregated_features"]),
            "mapping_fraction": audit["mapped_features"] /
                                (audit["mapped_features"] + audit["unmapped_features"]),
            "unmapped_mass_fraction": audit["unmapped_mass_fraction"],
            "d99_dropped_labels": ",".join(map(str, audit["d99_labels_dropped_before_relabel"])),
            "n_pairwise_fdr": int((spin["spin_q_bh"] < .05).sum()),
            "n_outcomes_pairwise_fdr": int(spin.loc[spin["spin_q_bh"] < .05, "outcome"].nunique()),
            "n_total_fdr": int((total["spin_q_bh_across_outcomes"] < .05).sum()),
            "mean_oof_r2": perf["oof_r2"].mean(),
            "median_oof_r2": perf["oof_r2"].median(),
            "n_positive_oof_r2": int((perf["oof_r2"] > 0).sum()),
            "best_oof_outcome": perf.loc[perf["oof_r2"].idxmax(), "outcome"],
            "best_oof_r2": perf["oof_r2"].max(),
            "shap_sum_max_error": float(np.abs(shap_sums - 1).max()),
            "dominance_ok_outcomes": int(len(dom_sums)),
            "dominance_sum_max_error": float(np.abs(dom_sums - 1).max()) if len(dom_sums) else np.nan,
        })
        for _, t in total.sort_values("spin_q_bh_across_outcomes").head(5).iterrows():
            key = (shap["phenotype"].eq(t["phenotype"]) & shap["outcome"].eq(t["outcome"]))
            sk = shap[key].sort_values("relative_shap", ascending=False).iloc[0]
            dk = dom[(dom["phenotype"].eq(t["phenotype"])) &
                     (dom["outcome"].eq(t["outcome"])) & (dom["status"].eq("ok"))]
            dfeature = dk.sort_values("relative_dominance", ascending=False).iloc[0]["feature"] if len(dk) else ""
            top_rows.append({"branch": branch.name, **t.to_dict(),
                             "top_shap_feature": sk["feature"],
                             "top_shap_relative": sk["relative_shap"],
                             "top_dominance_feature": dfeature})

pd.DataFrame(rows).to_csv(ROOT / "audit_summary.csv", index=False)
pd.DataFrame(top_rows).to_csv(ROOT / "top_total_models.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("\nTOP TOTAL MODELS\n", pd.DataFrame(top_rows).to_string(index=False))
