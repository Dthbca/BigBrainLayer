# BigBrainLayer analysis

This directory contains the reproducible Layer cell-composition–BigBrain thickness analysis used in the integrated HTML report.

## Analysis design

The primary pipeline is:

1. Normalize macaque laminar cell-type ratios across layers within each region and cell type.
2. Relabel the spatial maps to the BN atlas.
3. Reclose the relabeled compositions so the mapped components again sum to one.
4. Use relative BigBrain layer thickness.
5. Test each mask-supported layer–cell-type pair with 1,000 spatial spins and Benjamini–Hochberg correction.
6. Test the global matched-layer statistic with all 720 layer-order permutations.
7. Fit separate cross-validated models for each cortical layer and summarize cell-type contributions with SHAP.

The unreclosed continuous-feature analysis is retained as a sensitivity analysis rather than the primary result.

## Main results

- 97 mask-supported layer–cell-type pairs were tested.
- The selected reclosed branch produced 31 spin-FDR significant pairs.
- The exact whole-layer-order test gave 4 / 720, p = 0.005556.
- Layer IV–Pax6 was the strongest spatial example (r = 0.8889, spin q = 0.00745).
- Layer-specific predictive models were used for interpretation; SHAP contributions describe prediction attribution, not causality.

## Directory layout

- `code/`: analysis, correction, validation, plotting, and report-figure programs.
- `code/package_snapshot/`: the HomoloMap layer-related package modules used by this analysis.
- `results/baseline/`: original eight-branch comparison and audit files.
- `results/reclosure/`: primary reclosure analysis, layer-specific SHAP, and sensitivity outputs.
- `results/permutation_corrected/`: corrected exact permutation outputs.
- `figures/`: manuscript figure in PNG, editable SVG, and PDF.
- `report/layer_analysis_report.html`: integrated report with local assets.

## Key entry points

- `code/run_reclosure_layer_branches.py`: primary eight-branch reclosure analysis.
- `code/run_reclosure_layer_specific_shap.py`: layer-specific predictive models and SHAP.
- `code/correct_layer_permutation_results.py`: exact permutation correction and audit.
- `code/build_figure_draft_v6_wide_heatmap.py`: manuscript figure assembly.

Large TIFF exports, temporary previews, extracted PDF assets, caches, and remote raw datasets are intentionally excluded. The report and numerical tables needed for review are included.
## Source archive and project memory

- `code/HomoloMap/`: complete 21-file HomoloMap Python source snapshot used at archive time.
- `archives/HomoloMap_source_20260817.zip`: portable source archive; SHA256 `ebd8ff293094eb5a76fc8f147cc53e86030fa6c21eaf14939205e1076db3d812`.
- `PROJECT_MEMORY.md`: consolidated scientific decisions, validated results, remote-execution lessons, and report-production experience.
- `AGENT_PLAYBOOK.md`: reusable contract for remote runners, figure/report generation, validation, and GitHub publishing.
