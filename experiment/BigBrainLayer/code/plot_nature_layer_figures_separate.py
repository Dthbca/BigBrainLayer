"""Render standalone report figures for the BigBrain layer analysis."""

from pathlib import Path
import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


LAYERS = ["l1", "l2", "l3", "l4", "l5", "l6"]
LAYER_NAMES = ["Layer I", "Layer II", "Layer III", "Layer IV", "Layer V", "Layer VI"]
CLASS_DEF = {
    "Excitatory": ["L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"],
    "Inhibitory": ["Lamp5_Lhx6", "Lamp5", "Pax6", "Sncg", "Vip", "Sst", "Pvalb", "Chandelier"],
    "Non-neuron": ["Astro", "Oligo", "OPC", "Micro-PVM", "Endo", "VLMC"],
}
CLASS_COLORS = {"Excitatory": "#B95C50", "Inhibitory": "#4E8A68", "Non-neuron": "#557FA6", "Other": "#A9A9A9"}
FAMILY_COLORS = {"Cross · no CLR": "#245A73", "Cross · CLR": "#73A6B6", "Within · no CLR": "#A98A64", "Within · CLR": "#D0B99B"}


def family(row):
    norm = "Cross" if row.normalization == "within_region_cross_layer" else "Within"
    return f"{norm} · {'CLR' if bool(row.use_clr) else 'no CLR'}"


def label(row):
    return f"{family(row)} · {'Rel.' if row.thickness == 'relative' else 'Abs.'}"


def cell_order(columns):
    out = [c for group in CLASS_DEF.values() for c in group if c in columns]
    return out + [c for c in columns if c not in out]


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_figure(summary, output):
    data = summary.sort_values(["n_fdr_sig", "mean_abs_r"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    for i, row in data.iterrows():
        edge = "#111111" if row.whole_match_p < 0.05 else "white"
        marker = "o" if row.thickness == "relative" else "s"
        ax.scatter(row.mean_abs_r, i, s=32 + 3.2 * row.n_fdr_sig, marker=marker,
                   color=FAMILY_COLORS[family(row)], edgecolor=edge,
                   linewidth=1.3 if row.whole_match_p < 0.05 else 0.6, zorder=3)
        ax.text(row.mean_abs_r + 0.008, i, str(int(row.n_fdr_sig)), va="center", fontsize=6)
    ax.set_yticks(range(len(data)), [label(r) for _, r in data.iterrows()], fontsize=6.2)
    ax.set_xlabel("Mean |Pearson r|", fontsize=7)
    ax.set_xlim(0.24, 0.52)
    ax.grid(axis="x", color="#E5E8EA", linewidth=0.6)
    ax.text(0.99, 0.02, "Number: FDR-significant tests\nBlack edge: whole-match P < 0.05",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="#555555")
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.15, top=0.97)
    save(fig, output, "figure_pipeline_comparison")


def heatmap_figure(source, output):
    r = pd.read_csv(source / "source_data" / "best_branch_r_matrix.csv", index_col=0).reindex(LAYERS)
    p = pd.read_csv(source / "source_data" / "best_branch_p_matrix.csv", index_col=0).reindex(LAYERS)
    columns = cell_order(r.columns)
    r, p = r[columns], p[columns]
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    cmap = mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("#EEEEEE")
    image = ax.imshow(r.to_numpy(float), cmap=cmap, norm=TwoSlopeNorm(vmin=-0.9, vcenter=0, vmax=0.9), aspect="auto")
    for x in np.arange(0.5, len(columns), 1): ax.axvline(x, color="white", linewidth=0.35)
    for y in np.arange(0.5, 6, 1): ax.axhline(y, color="white", linewidth=0.5)
    for i in range(6):
        for j in range(len(columns)):
            q = p.iloc[i, j]
            if np.isfinite(q) and q < 0.05:
                ax.text(j, i, "**" if q < 0.01 else "*", ha="center", va="center",
                        fontsize=5, fontweight="bold", color="white" if abs(r.iloc[i, j]) > 0.48 else "#222222")
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=5.8)
    ax.set_yticks(range(6), LAYER_NAMES, fontsize=6.5)
    ax.tick_params(length=0, pad=2)
    class_map = {c: g for g, cells in CLASS_DEF.items() for c in cells}
    bar = ax.inset_axes([0, 1.035, 1, 0.07]); bar.set_xlim(-0.5, len(columns)-0.5); bar.set_ylim(0, 1); bar.axis("off")
    start = 0
    while start < len(columns):
        group = class_map.get(columns[start], "Other"); end = start + 1
        while end < len(columns) and class_map.get(columns[end], "Other") == group: end += 1
        bar.add_patch(Rectangle((start-0.5, 0), end-start, 1, color=CLASS_COLORS[group], linewidth=0))
        bar.text((start+end-2)/2, 0.5, group, ha="center", va="center", fontsize=6,
                 color="white", fontweight="bold")
        start = end
    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.025); cbar.set_label("Pearson r", fontsize=6.5); cbar.ax.tick_params(labelsize=5.5)
    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.30, top=0.87)
    save(fig, output, "figure_best_branch_heatmap")


def layer_figure(source, summary, output):
    data = pd.read_csv(source / "source_data" / "branch_layer_mean_abs_r.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.75)); x = np.arange(6)
    for _, row in summary.iterrows():
        selected = row.branch == "within_region_cross_layer__clr_false__thickness_relative"
        ax.plot(x, data.loc[row.branch, LAYERS], color=FAMILY_COLORS[family(row)],
                linestyle="-" if row.thickness == "relative" else "--",
                linewidth=2.3 if selected else 1.0, alpha=1 if selected else 0.72,
                marker="o" if selected else None, markersize=3.5)
    ax.set_xticks(x, ["I", "II", "III", "IV", "V", "VI"], fontsize=6.5)
    ax.set_xlabel("Cortical layer", fontsize=7); ax.set_ylabel("Mean |Pearson r|", fontsize=7)
    ax.set_ylim(0.1, 0.86); ax.grid(axis="y", color="#E5E8EA", linewidth=0.6)
    handles = [Line2D([0], [0], color=c, linewidth=2, label=k) for k, c in FAMILY_COLORS.items()]
    handles += [Line2D([0], [0], color="#555", linestyle="-", label="Relative thickness"),
                Line2D([0], [0], color="#555", linestyle="--", label="Absolute thickness")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
              fontsize=5.5, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.29, top=0.96)
    save(fig, output, "figure_layer_effect_profile")


def sensitivity_figure(source, summary, output):
    matrix = pd.read_csv(source / "source_data" / "branch_spearman_r.csv", index_col=0)
    order = summary.branch.tolist(); matrix = matrix.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(7.2, 4.25)); image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ids = [f"P{i+1}" for i in range(8)]
    ax.set_xticks(range(8), ids, fontsize=6); ax.set_yticks(range(8), ids, fontsize=6); ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04); cbar.set_label("Spearman ρ", fontsize=6.5); cbar.ax.tick_params(labelsize=5.5)
    key = "\n".join(f"P{i+1}  {label(row)}" for i, (_, row) in enumerate(summary.iterrows()))
    ax.text(1.18, 1.0, key, transform=ax.transAxes, ha="left", va="top", fontsize=5.5, linespacing=1.35)
    fig.subplots_adjust(left=0.10, right=0.66, bottom=0.12, top=0.96)
    save(fig, output, "figure_pipeline_sensitivity")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":6,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.7,"legend.frameon":False,
                         "svg.fonttype":"none","pdf.fonttype":42})
    summary = pd.read_csv(args.source / "branch_summary.csv"); summary.use_clr = summary.use_clr.astype(str).str.lower().eq("true")
    summary = summary.sort_values(["n_fdr_sig","mean_abs_r"], ascending=False).reset_index(drop=True)
    pipeline_figure(summary, args.output); heatmap_figure(args.source, args.output); layer_figure(args.source, summary, args.output); sensitivity_figure(args.source, summary, args.output)
    (args.output / "standalone_figure_contract.json").write_text(json.dumps({"backend":"python","figures":4,"data_rows":776,"titles_in_html":True,"composite":False}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
