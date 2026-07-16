from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


NumericRow = Dict[str, float | str]


def read_table(path: str | Path) -> List[NumericRow]:
    table_path = Path(path)
    delimiter = "\t" if table_path.suffix.lower() == ".tsv" else ","
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def _zscore(values: Iterable[float]) -> List[float]:
    values_list = list(values)
    mean_v = sum(values_list) / len(values_list)
    variance = sum((value - mean_v) ** 2 for value in values_list) / len(values_list)
    std = math.sqrt(variance) or 1.0
    return [(value - mean_v) / std for value in values_list]


def preprocess_bigbrain(rows: List[NumericRow]) -> List[NumericRow]:
    filtered = [row for row in rows if row.get("region") and row.get("thickness_mm")]
    thickness = [float(row["thickness_mm"]) for row in filtered]
    zscores = _zscore(thickness)
    for row, zscore in zip(filtered, zscores):
        row["thickness_mm"] = float(row["thickness_mm"])
        row["thickness_z"] = zscore
    return filtered


def preprocess_spatial(rows: List[NumericRow]) -> List[NumericRow]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        region = (row.get("region") or "").strip()
        expression = row.get("expression")
        if not region or expression in (None, ""):
            continue
        grouped.setdefault(region, []).append(float(expression))

    output: List[NumericRow] = []
    for region, values in grouped.items():
        mean_expression = sum(values) / len(values)
        output.append(
            {
                "region": region,
                "mean_expression": mean_expression,
                "log1p_expression": math.log1p(max(mean_expression, 0.0)),
            }
        )
    return output


def merge_datasets(
    bigbrain_rows: List[NumericRow], spatial_rows: List[NumericRow]
) -> List[NumericRow]:
    spatial_index = {row["region"]: row for row in spatial_rows}
    merged: List[NumericRow] = []
    for row in bigbrain_rows:
        match = spatial_index.get(row["region"])
        if not match:
            continue
        merged.append(
            {
                "region": row["region"],
                "thickness_mm": float(row["thickness_mm"]),
                "thickness_z": float(row["thickness_z"]),
                "mean_expression": float(match["mean_expression"]),
                "log1p_expression": float(match["log1p_expression"]),
            }
        )
    return merged


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
    denom = den_x * den_y
    return num / denom if denom else 0.0


def _rank(values: List[float]) -> List[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_, idx) in enumerate(ordered, start=1):
        ranks[idx] = float(rank)
    return ranks


def evaluate_strategies(merged_rows: List[NumericRow]) -> List[Dict[str, float | str]]:
    thickness = [float(row["thickness_z"]) for row in merged_rows]
    raw_expression = [float(row["mean_expression"]) for row in merged_rows]
    log_expression = [float(row["log1p_expression"]) for row in merged_rows]

    strategies = [
        ("raw_pearson", _pearson(thickness, raw_expression)),
        ("log1p_pearson", _pearson(thickness, log_expression)),
        (
            "raw_spearman",
            _pearson(_rank(thickness), _rank(raw_expression)),
        ),
    ]
    return [
        {"strategy": strategy, "score": score, "abs_score": abs(score)}
        for strategy, score in sorted(strategies, key=lambda item: abs(item[1]), reverse=True)
    ]


def save_temp_results(
    output_dir: str | Path, merged_rows: List[NumericRow], strategy_scores: List[NumericRow]
) -> Dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    merged_path = destination / f"merged_{stamp}.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()) if merged_rows else [])
        if merged_rows:
            writer.writeheader()
            writer.writerows(merged_rows)

    scores_path = destination / f"strategy_scores_{stamp}.json"
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(strategy_scores, handle, indent=2)

    return {"merged_csv": merged_path, "scores_json": scores_path}


def plot_strategy_scores_svg(scores: List[NumericRow], output_path: str | Path) -> Path:
    width = 720
    height = 80 + 80 * len(scores)
    bar_max_width = 400
    scale = bar_max_width / max((abs(float(row["score"])) for row in scores), default=1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="20" y="30" font-size="20">Strategy comparison (|correlation|)</text>',
    ]

    for idx, row in enumerate(scores):
        y = 60 + idx * 80
        score = float(row["score"])
        abs_width = abs(score) * scale
        color = "#4e79a7" if score >= 0 else "#e15759"
        lines.append(f'<text x="20" y="{y}">{row["strategy"]} ({score:.3f})</text>')
        lines.append(
            f'<rect x="250" y="{y - 15}" width="{abs_width:.2f}" height="20" fill="{color}" />'
        )

    lines.append("</svg>")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_pipeline(bigbrain_path: str, spatial_path: str, output_dir: str) -> Dict[str, str | List[Dict[str, float | str]]]:
    bigbrain = preprocess_bigbrain(read_table(bigbrain_path))
    spatial = preprocess_spatial(read_table(spatial_path))
    merged = merge_datasets(bigbrain, spatial)
    scores = evaluate_strategies(merged)
    outputs = save_temp_results(output_dir, merged, scores)
    figure = plot_strategy_scores_svg(scores, Path(output_dir) / "strategy_scores.svg")
    return {
        "n_bigbrain_regions": len(bigbrain),
        "n_spatial_regions": len(spatial),
        "n_merged_regions": len(merged),
        "strategy_ranking": scores,
        "merged_csv": str(outputs["merged_csv"]),
        "scores_json": str(outputs["scores_json"]),
        "scores_plot_svg": str(figure),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BigBrainLayer starter analysis pipeline")
    parser.add_argument("--bigbrain", required=True, help="CSV/TSV with columns: region,thickness_mm")
    parser.add_argument("--spatial", required=True, help="CSV/TSV with columns: region,expression")
    parser.add_argument("--output-dir", default="results/tmp", help="Directory for temporary outputs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_pipeline(args.bigbrain, args.spatial, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
