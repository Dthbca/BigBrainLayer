# Agent playbook — BigBrainLayer

## Remote runner contract

1. Read-only inspect the canonical CellAlign/HomoloMap source.
2. Write only under `/share/user_data/dthbca/public/experiment/BigBrainLayer`.
3. Prefer direct n03 when io refuses connection; report selected node and load.
4. Verify archive byte count, file count, and SHA256 before extraction.
5. Test in this order: compileall → imports/API existence → synthetic edge cases → real loader/mapping → cached preparation → plotting/spin/permutation → full pipeline.
6. Keep every stage independently restartable and report exact PASS/FAIL evidence.
7. Never claim success for a command stopped before output. Separate fixture mistakes from source defects.

## Figure/report contract

1. State the figure conclusion and evidence chain before editing.
2. Use Python consistently for this project’s scientific figures.
3. Keep heatmaps wide enough for cell-type labels; stack paired brain maps vertically when needed.
4. Preserve reference-art aspect ratios through inset axes.
5. Validate PNG/SVG/PDF exports and all HTML relative links.
6. Do not describe SHAP as causal or pool layers when the scientific question is layer-specific.

## Publishing contract

1. Work in an independent clone of `Dthbca/BigBrainLayer`.
2. Update `experiment/BigBrainLayer`, its README, manifest, source archive, and report links.
3. Run syntax, link, archive-hash, and size checks.
4. Commit only scoped files. Push main only when explicitly requested.