from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY_METRICS = [
    ("MolmoWeb QA answer accuracy", ("molmoweb", "qa", "answer_accuracy"), True),
    ("MolmoWeb QA token F1", ("molmoweb", "qa", "mean_token_f1"), True),
    ("MolmoWeb grounding click accuracy", ("molmoweb", "grounding", "click_accuracy"), True),
    (
        "MolmoWeb grounding median pixel distance",
        ("molmoweb", "grounding", "median_pixel_distance"),
        False,
    ),
    ("ScreenSpot click accuracy", ("screenspot", "grounding", "click_accuracy"), True),
    (
        "ScreenSpot median pixel distance",
        ("screenspot", "grounding", "median_pixel_distance"),
        False,
    ),
]


def _get(mapping: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def compare_metrics(baseline: dict[str, Any], sft: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path, higher_is_better in PRIMARY_METRICS:
        before = _get(baseline, path)
        after = _get(sft, path)
        delta = after - before if before is not None and after is not None else None
        improved = None if delta is None else (delta > 0 if higher_is_better else delta < 0)
        rows.append(
            {
                "metric": name,
                "baseline": before,
                "sft": after,
                "delta": delta,
                "improved": improved,
            }
        )
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Experiment 1 comparison",
        "",
        "| Metric | Baseline | SFT | Delta | Improved |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        values = [row[key] for key in ("baseline", "sft", "delta")]
        rendered = ["—" if value is None else f"{value:.4f}" for value in values]
        improved = "—" if row["improved"] is None else ("yes" if row["improved"] else "no")
        lines.append(
            f"| {row['metric']} | {rendered[0]} | {rendered[1]} | {rendered[2]} | {improved} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_files(
    baseline_path: str | Path, sft_path: str | Path, output_path: str | Path
) -> list[dict[str, Any]]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    sft = json.loads(Path(sft_path).read_text(encoding="utf-8"))
    rows = compare_metrics(baseline, sft)
    write_markdown(rows, Path(output_path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and SFT metrics")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--output", default="outputs/experiment1/comparison.md")
    args = parser.parse_args()
    rows = compare_files(args.baseline, args.sft, args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
