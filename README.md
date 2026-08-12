# BigBrainLayer

A student-friendly starter project to explore the relationship between:
- **BigBrain cortical layer thickness**
- **Macaque Stereo-seq spatial transcriptomics**

## Friendly file structure

```text
/home/runner/work/BigBrainLayer/BigBrainLayer
├── bigbrainlayer/
│   ├── __init__.py
│   └── pipeline.py                  # preprocess, merge, evaluate, save, plot
├── scripts/
│   └── run_analysis.py              # CLI entrypoint
├── data/
│   ├── raw/
│   │   ├── bigbrain_layer_thickness/
│   │   │   └── README.txt
│   │   └── macaque_stereo_seq_spatial/
│   │       └── README.txt
│   └── processed/
├── results/
│   ├── tmp/                         # temporary intermediate outputs
│   └── figures/                     # final figure exports
└── tests/
    └── test_pipeline.py
```

## Input format

1. `data/raw/bigbrain_layer_thickness/*.csv` or `*.tsv`
   - required columns: `region`, `thickness_mm`
2. `data/raw/macaque_stereo_seq_spatial/*.csv` or `*.tsv`
   - required columns: `region`, `expression`

## What the pipeline does

1. **Preprocess**
   - BigBrain thickness: quality filtering and z-score normalization
   - Spatial data: per-region expression aggregation and `log1p` transform
2. **Merge** by `region`
3. **Evaluate strategy options**
   - `raw_pearson`
   - `log1p_pearson`
   - `raw_spearman`
4. **Save temporary results** in `results/tmp`
   - merged CSV
   - strategy score JSON
5. **Plot** strategy comparison as SVG

## Run analysis

```bash
cd /home/runner/work/BigBrainLayer/BigBrainLayer
python scripts/run_analysis.py \
  --bigbrain data/raw/bigbrain_layer_thickness/your_bigbrain.csv \
  --spatial data/raw/macaque_stereo_seq_spatial/your_spatial.csv \
  --output-dir results/tmp/run_01
```

## Evaluate different strategies

The output JSON reports strategy scores sorted by absolute correlation. Start with the top strategy for biological interpretation, then verify robustness by comparing direction/sign and magnitude across the other two.

## Citation guidance

If you use this repository in your work, please cite the relevant data/method sources:

1. Amunts K, et al. **BigBrain: An Ultrahigh-Resolution 3D Human Brain Model**. *Science* (2013). doi:10.1126/science.1235381
2. Chen A, et al. **Large field of view-spatially resolved transcriptomics at nanoscale resolution** (Stereo-seq). *Cell* (2022). doi:10.1016/j.cell.2022.04.003
3. Virtanen P, et al. **SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python**. *Nature Methods* (2020). doi:10.1038/s41592-019-0686-2

> Replace these with the exact versions/datasets you used in your final manuscript.
