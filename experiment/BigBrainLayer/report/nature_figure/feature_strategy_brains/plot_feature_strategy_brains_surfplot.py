"""Render six standalone BN-left feature-strategy maps with HomoloMap Surfplot."""

from pathlib import Path
import argparse
import sys

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

STRATEGIES = {
    "ratio_then_mean_relabel": "Ratio → mean relabel",
    "sum_relabel_then_ratio": "Sum relabel → ratio",
    "mean_relabel_then_ratio": "Mean relabel → ratio",
}

CASES = (
    ("cross_layer_l6_endo", "Layer VI · Endo"),
    ("within_layer_l4_lamp5", "Layer IV · Lamp5"),
)


def render_table(table, case_name, case_label, output, plot_left):
    values = table[list(STRATEGIES)].to_numpy(float)
    finite = values[pd.notna(values)]
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        raise ValueError(f"Constant map for {case_name}")

    for key, strategy_label in STRATEGIES.items():
        fig = plot_left(
            table[key], atlas="BN", species="human", surf="inflated",
            view="row", cmap="cividis", vmin=vmin, vmax=vmax,
            outline=True, cbar_label="Mapped feature",
            title=f"{case_label} | {strategy_label}", title_fontsize=11,
            size=(800, 420), zoom=1.05, dpi=300,
            cbar_kwargs={"decimals": 3, "shrink": 0.62},
        )
        base = output / f"figure_brain_{case_name}_{key}"
        for ext, dpi in (("png", 300), ("tiff", 600), ("pdf", None), ("svg", None)):
            kwargs = {"bbox_inches": "tight", "facecolor": "white"}
            if dpi is not None:
                kwargs["dpi"] = dpi
            fig.savefig(str(base) + f".{ext}", **kwargs)
        plt.close(fig)

    corr = table[list(STRATEGIES)].corr()
    pd.Series({
        "ratio_sum_r": corr.iloc[0, 1],
        "ratio_mean_r": corr.iloc[0, 2],
        "sum_mean_r": corr.iloc[1, 2],
    }, name=case_name).to_csv(output / f"{case_name}_spatial_correlations.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package))
    from HomoloMap.plotting import plot_left

    args.output.mkdir(parents=True, exist_ok=True)
    for case_name, case_label in CASES:
        table = pd.read_csv(args.source / f"{case_name}_source_data.csv", index_col=0)
        if list(table.columns) != list(STRATEGIES):
            raise ValueError(f"Unexpected strategy columns for {case_name}: {list(table.columns)}")
        if table.shape[0] != 105 or not table.index.is_unique:
            raise ValueError(f"Expected 105 unique BN-left ROIs for {case_name}")
        render_table(table, case_name, case_label, args.output, plot_left)
    print({"status": "PASS", "renderer": "HomoloMap.plotting.plot_left / Surfplot", "n_maps": 6})


if __name__ == "__main__":
    main()
