# Project memory — HomoloMap / BigBrainLayer

Last consolidated: 2026-08-17 (Asia/Shanghai)

## Durable project objective

Map macaque spatial-transcriptomic cell-type compositions to homologous human cell types, spatially relabel them to a human cortical atlas, and test their layer-matched association with BigBrain cortical layer thickness. All primary analyses use the available cortical hemisphere consistently and avoid implying bilateral validation.

## Primary analysis decisions

- Cell-type ratios are compositional data. The package supports none, CLR, and ILR transforms, but transforms are optional and restricted to ratio features.
- For the BigBrain layer analysis, the primary feature construction is: normalize across layers within each region and cell type → spatial relabel → reclose the mapped composition.
- Reclosure after relabel is the primary analysis because atlas mapping can change the component sum. The earlier unreclosed result is retained as a continuous-feature sensitivity analysis.
- Primary selected branch: cross-layer normalization, no CLR, relative layer thickness.
- Only mask-supported layer–cell-type pairs enter method comparison and multiple-testing families.
- Spatial association uses Alexander–Bloch spins with 1,000 rotations and Benjamini–Hochberg correction across the 97 eligible layer–cell-type pairs.
- Global layer-order inference uses all 720 permutations. Corrected exact result: 4/720, p = 0.005556. Mismatch pseudo-replication was removed; the corrected per-pair mismatch results have no significant pairs.
- Main branch summary: 31 spin-FDR significant pairs, mean absolute r ≈ 0.4583. Strong spatial example: Layer IV–Pax6, r = 0.8889, spin p = 0.000999, q = 0.00745.
- SHAP should be layer-specific: six independent cross-validated prediction tasks, not one pooled all-layer task. SHAP values describe predictive attribution, not causality.

## Package changes and tests

- Added robust load_data / staged analysis functions and preserved a one-stop run_analysis interface.
- Added homologous cell-type mapping policies with explicit unresolved-type audit, coverage thresholds, optional keep/drop/raise, and provenance.
- Added composition module with closure, multiplicative zero replacement, CLR, and ILR.
- Added BigBrain layer loaders, relabeling, normalization, masks, subcomposition transforms, spin correlation, exact permutation, plotting, prepare_layer_analysis, and run_layer_analysis.
- Real remote tests passed for ratio mapping, composition transforms, spin tests, cumulative models, linear/RF/SVR SHAP, Layer loaders, relabeling, thickness, masks, plotting, permutations, and the full layer workflow.
- Known atlas audit: D99 labels 106, 118, and 194 are dropped during D99→BN relabeling and must remain recorded.

## Remote execution rules learned

- The canonical CellAlign source at `/data100/home/dthbca/project/CellAlign` is treated as read-only.
- Writable experiment outputs belong under `/share/user_data/dthbca/public/experiment/BigBrainLayer`.
- `io` frequently refuses port 22; direct n03 access is the reliable fallback. Use short staged commands, explicit timeouts, and absolute Python `/data100/home/dthbca/.conda/envs/dthbca_imgT/bin/python`.
- Repeated six-layer relabeling takes roughly 50–60 seconds. Cache prepared data before plotting/spin tests and split compile, import, preparation, plotting, spin, and full-pipeline checks into separate commands.
- Sync a single verified archive, compare SHA256 and file count remotely, then extract to staging. Never overwrite an unknown dirty remote working tree.

## Agent and workflow optimization

- Delegate remote execution to a remote-runner with an explicit write boundary, selected node, Python path, test phases, and required audit outputs.
- Require stage-by-stage messages: connectivity, archive verification, compile/import, synthetic tests, real-data tests, and final manifest.
- Distinguish source failures from fixture failures. Two previous composition test failures came from an invalid test parameter and mismatched coverage indices, not package code.
- Avoid long opaque SSH commands. Commands should emit a stage marker, have a timeout, and leave a log. Cache expensive relabel outputs.
- Before GitHub publishing, clone the actual target repository separately; do not attach a remote to the untracked local research workspace. Stage only the intended experiment directory.
- For figures, define the conclusion and evidence order before plotting; preserve image aspect ratios with independent inset axes, not by changing the parent axes aspect.

## Report production lessons

- The final report should be self-contained, detailed enough for scientific review, but avoid meta labels such as “advisor version” or unnecessary statements about unavailable hemispheres.
- Preferred evidence order: analysis rationale → branch comparison → primary reclosed result → sensitivity analysis → corrected permutation → layer-specific SHAP → brain-map examples → manuscript figure draft.
- Use separate figures placed next to the corresponding result rather than dense multipanel composites inside the HTML. The manuscript draft can remain a compact multi-panel summary.
- Always package HTML with relative assets and validate every src/href after extraction. The 2026-08-17 share ZIP contained 236 files and had zero missing links.
- Export manuscript figures as PNG, editable SVG, PDF, and optional TIFF; use a centered diverging color bar for correlations and prevent legends/text from covering data.

## Canonical artifacts

- Local experiment: `D:/HomoloMap/experiment/BigBrainLayer`
- Integrated report: `report/layer_analysis_report.html`
- Full HomoloMap source: `code/HomoloMap`
- Dated source archive: `archives/HomoloMap_source_20260817.zip`
- GitHub repository: `git@github.com:Dthbca/BigBrainLayer.git`
- GitHub experiment path: `experiment/BigBrainLayer`