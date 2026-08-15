from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


LABELS = {
    "within_layer__clr_false__thickness_absolute": "Within-layer | no CLR | absolute",
    "within_layer__clr_false__thickness_relative": "Within-layer | no CLR | relative",
    "within_layer__clr_true__thickness_absolute": "Within-layer | CLR | absolute",
    "within_layer__clr_true__thickness_relative": "Within-layer | CLR | relative",
    "within_region_cross_layer__clr_false__thickness_absolute": "Cross-layer | no CLR | absolute",
    "within_region_cross_layer__clr_false__thickness_relative": "Cross-layer | no CLR | relative",
    "within_region_cross_layer__clr_true__thickness_absolute": "Cross-layer | CLR | absolute",
    "within_region_cross_layer__clr_true__thickness_relative": "Cross-layer | CLR | relative",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.summary).sort_values("mean_oof_r2")
    colors = ["#087E8B" if "within_region" in b else "#D95F02" for b in df.branch]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    y = range(len(df))
    ax.barh(y, df.mean_oof_r2, color=colors, alpha=.88, height=.64)
    ax.set_yticks(list(y), [LABELS[b] for b in df.branch], fontsize=8)
    ax.set_xlabel("Mean out-of-fold $R^2$ across six layers")
    ax.set_xlim(0, max(.68, df.mean_oof_r2.max() + .06))
    ax.grid(axis="x", color="#D9D9D9", lw=.6, zorder=0)
    for i, row in enumerate(df.itertuples()):
        ax.text(row.mean_oof_r2 + .008, i,
                f"{row.mean_oof_r2:.3f}  |  Top: {row.top_ctype}",
                va="center", fontsize=7, color="#222222")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(a.out / "figure_shap_branch_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out / "figure_shap_branch_comparison.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out / "figure_shap_branch_comparison.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(a.out / "figure_shap_branch_comparison.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()




