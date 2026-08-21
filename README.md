# BigBrainLayer
explore the relationship between cell type layer distribution and layer thickness
## Integrated layer analysis

The reproducible comparison of layer cell-type composition and BigBrain layer thickness is available in [`experiment/BigBrainLayer`](experiment/BigBrainLayer). It includes the analysis code, package-module snapshot, numerical results, corrected exact permutation tests, layer-specific SHAP outputs, manuscript figures, and a self-contained HTML report.

- [Analysis overview and reproduction notes](experiment/BigBrainLayer/README.md)
- [Integrated HTML report](experiment/BigBrainLayer/report/layer_analysis_report.html)
- [Editable manuscript figure](experiment/BigBrainLayer/figures/figure_draft_v6_wide_heatmap.svg)
- [Complete HomoloMap source snapshot](experiment/BigBrainLayer/code/HomoloMap)
- [Project memory and validated decisions](experiment/BigBrainLayer/PROJECT_MEMORY.md)
- [Agent workflow playbook](experiment/BigBrainLayer/AGENT_PLAYBOOK.md)

## HomoloMap MEG and ENIGMA analysis

The non-laminar imaging analysis is available in [`experiment/NonLaminarImaging`](experiment/NonLaminarImaging). It keeps MEG frequency maps and ENIGMA cortical-thickness effect maps as separate outcome families and includes subclass/cluster ratio analyses, spatial spin tests, total-model inference, grouped out-of-fold prediction, dominance analysis, and held-out SHAP attribution.

- [Analysis overview and reproduction notes](experiment/NonLaminarImaging/README.md)
- [Integrated HTML report](experiment/NonLaminarImaging/report/nonlaminar_imaging_report.html)
- [Primary analysis program](experiment/NonLaminarImaging/code/run_nonlaminar_imaging.py)
- [Synchronized file manifest](experiment/NonLaminarImaging/MANIFEST_SHA256.csv)
