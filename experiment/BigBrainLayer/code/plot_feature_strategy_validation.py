from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8,
})

PAIR_LABEL = {
    "ratio_then_mean_relabel__vs__sum_relabel_then_ratio": "Ratio→mean vs sum→ratio",
    "ratio_then_mean_relabel__vs__mean_relabel_then_ratio": "Ratio→mean vs mean→ratio",
    "sum_relabel_then_ratio__vs__mean_relabel_then_ratio": "Sum→ratio vs mean→ratio",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--correlations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.correlations)
    n_input = len(df)
    df = df[np.isfinite(df["pearson_r"])]
    n_plotted = len(df)
    if n_plotted != n_input:
        print(f"Excluded {n_input - n_plotted} non-finite correlations; plotted {n_plotted}/{n_input}.")
    order = []
    for norm in ["within_region_cross_layer", "within_layer"]:
        for pair in PAIR_LABEL:
            order.append((norm, pair))
    values = [df[(df.normalization == n) & (df.pair == p)].pearson_r.values for n, p in order]
    labels = [("Cross-layer | " if n.startswith("within_region") else "Within-layer | ") + PAIR_LABEL[p]
              for n, p in order]
    colors = ["#3C8D7B" if n.startswith("within_region") else "#D07A3B" for n, _ in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    positions = np.arange(len(values), 0, -1)
    bp = ax.boxplot(values, positions=positions, vert=False, widths=.55,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.4},
                    whiskerprops={"color": "#5E6870"}, capprops={"color": "#5E6870"})
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color); box.set_edgecolor(color); box.set_alpha(.9)
    for y, vals, color in zip(positions, values, colors):
        jitter = np.linspace(-.08, .08, num=len(vals), endpoint=True)
        ax.scatter(vals, y + jitter, s=6, color=color, alpha=.24, linewidths=0, zorder=1)
        ax.text(1.005, y, f"median {np.median(vals):.3f} | min {np.min(vals):.3f}",
                va="center", ha="left", fontsize=6.5, color="#303840")
    ax.axvline(.85, color="#A73B3B", ls="--", lw=1, label="r = 0.85 reference")
    ax.axvline(0, color="#B8BEC4", lw=.7)
    ax.set_yticks(positions, labels)
    ax.set_xlim(-.42, 1.23)
    ax.set_xlabel("Pearson correlation between spatial feature maps")
    ax.grid(axis="x", color="#E1E5E8", lw=.6, zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    base = a.out / "figure_feature_strategy_validation"
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

