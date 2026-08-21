# HomoloMap non-laminar imaging analysis

This directory contains the reproducible analysis linking HomoloMap cortical cell-composition maps to two independent imaging families: resting-state MEG frequency maps and ENIGMA cortical-thickness effect maps.

## Analysis scope

- Primary features: mapped cell-type ratios at subclass and cluster resolution.
- Sensitivity analyses: centered log-ratio transformed ratios and cell-density features.
- Spatial inference: Pearson association with Alexander-Bloch spin nulls and Benjamini-Hochberg correction within each outcome family.
- Total contribution: regularized multivariable models tested against spatially rotated outcomes.
- Generalization: grouped out-of-fold prediction across cortical partitions.
- Cell-type interpretation: dominance analysis and SHAP values calculated for held-out regions.

MEG and ENIGMA are analyzed and reported as separate outcome families. Neurotransmitter results are literature context only and are not an input, comparator, or baseline model.

## Directory layout

- `code/`: analysis, audit, plotting, and report-building programs.
- `results/subclass_main_20260821/`: primary subclass-level ratio analysis and its sensitivity branches.
- `results/cluster_secondary_20260821/`: secondary cluster-level ratio analysis and its sensitivity branches.
- `results/hansen_style_meg_20260821/`: curated MEG summary tables.
- `results/hansen_style_enigma_20260821/`: curated ENIGMA summary tables.
- `report/nonlaminar_imaging_report.html`: integrated local HTML report.
- `report/figures/` and `report/figure_draft/`: review figures and their source data.
- `MANIFEST_SHA256.csv`: file sizes and SHA256 hashes for the synchronized release.

## Main entry points

- `code/run_nonlaminar_imaging.py`: complete MEG/ENIGMA spatial association, total-model, dominance, prediction, and held-out SHAP workflow.
- `code/run_hansen_style_meg.py`: MEG-focused analysis summary.
- `code/run_hansen_style_enigma.py`: ENIGMA-focused analysis summary.
- `code/audit_results.py`: result integrity and completeness audit.
- `code/build_evidence_led_report.py`: final evidence-led report and main figure builder.

## Data policy

Raw remote datasets, caches, temporary logs, and large TIFF exports are intentionally excluded. The synchronized release contains executable code, compact numerical results, review-quality PNG/SVG/PDF figures, audit metadata, and the HTML report.

