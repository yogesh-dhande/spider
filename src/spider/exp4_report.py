from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _px(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.1f} px"


def _action_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {int(metrics['examples'])} | {_pct(metrics['json_parse_rate'])} | "
        f"{_pct(metrics['action_name_accuracy'])} | "
        f"{_pct(metrics['action_argument_accuracy'])} | "
        f"{_pct(metrics.get('click_inside_bbox_accuracy'))} | "
        f"{_px(metrics.get('click_median_distance_px'))} |"
    )


def _action_shard_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| Shard | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("shards", []):
        rows.append(_action_row(str(item["label"]), item["metrics"]))
    return rows


def _perception_shard_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| Shard | QA N | QA exact | Grounding N | Grounding click | Median error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in summary.get("per_shard", {}).items():
        molmoweb = metrics["molmoweb"]
        qa = molmoweb["qa"]
        grounding = molmoweb["grounding"]
        rows.append(
            f"| {label} | {int(qa['examples'])} | {_pct(qa['answer_accuracy'])} | "
            f"{int(grounding['examples'])} | {_pct(grounding['click_accuracy'])} | "
            f"{_px(grounding['median_pixel_distance'])} |"
        )
    return rows


def build_exp4_report(artifact_root: Path) -> str:
    baselines = _load(artifact_root / "action_baseline/metrics.json")
    dataset_path = artifact_root / "data/dataset_summary.json"
    gate_paths = sorted((artifact_root / "validation_steps").glob("step_*/gate.json"))
    gates = [_load(path) for path in gate_paths]
    selection_path = artifact_root / "checkpoint_selection.json"
    final_path = artifact_root / "final_test/comparison.json"
    closed_loop_path = artifact_root / "closed_loop/summary.json"

    lines = [
        "# EXP004 results",
        "",
    ]
    if dataset_path.is_file():
        dataset = _load(dataset_path)
        action = dataset["action_counts"]
        perception = dataset["perception_counts"]
        lines.extend(
            [
                "## Dataset realized counts",
                "",
                "| Partition | Action | ScreenshotQA | Grounding |",
                "|---|---:|---:|---:|",
                (
                    f"| Train | {action['train']} | {perception['qa']['train']} | "
                    f"{perception['grounding']['train']} |"
                ),
                (
                    f"| Validation | {action['validation']} | "
                    f"{perception['qa']['validation']} | "
                    f"{perception['grounding']['validation']} |"
                ),
                (
                    f"| Sealed test | {action['test']} | {perception['qa']['test']} | "
                    f"{perception['grounding']['test']} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Development action baselines",
            "",
            "| Model | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _action_row("Untouched Qwen3.5-2B", baselines["base"]),
            _action_row("EXP002 perception adapter", baselines["exp002"]),
        ]
    )
    for label, title in (
        ("action-base-shard-metrics.json", "Untouched-base shard diagnostics"),
        ("action-exp002-shard-metrics.json", "EXP002-parent shard diagnostics"),
    ):
        path = artifact_root / "action_baseline" / label
        if path.is_file():
            lines.extend(["", f"### {title}", "", *_action_shard_table(_load(path))])
    lines.extend(
        [
            "",
            "## Stage validation trajectory",
            "",
            "| Step | Action name | Arguments | Click in bounds | QA exact | Grounding click | Gate |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for gate in gates:
        action = gate["action_candidate"]
        perception = gate["perception_candidate"]
        lines.append(
            f"| {int(gate['step'])} | {_pct(action['action_name_accuracy'])} | "
            f"{_pct(action['action_argument_accuracy'])} | "
            f"{_pct(action.get('click_inside_bbox_accuracy'))} | "
            f"{_pct(perception['qa_answer_accuracy'])} | "
            f"{_pct(perception['grounding_click_accuracy'])} | "
            f"{'advance' if gate.get('advance') else 'stop'} |"
        )

    if selection_path.is_file():
        selection = _load(selection_path)
        lines.extend(
            [
                "",
                "## Selected checkpoint",
                "",
                (
                    f"Step **{selection['selected_step']}**, selected only from the fixed "
                    "development probes by the preregistered lexicographic rule."
                ),
            ]
        )

    if final_path.is_file():
        final = _load(final_path)
        perception_base = final["perception_baseline"]["molmoweb"]
        perception_sft = final["perception_sft"]["molmoweb"]
        lines.extend(
            [
                "",
                "## Sealed test",
                "",
                "### Browser actions",
                "",
                "| Model | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |",
                "|---|---:|---:|---:|---:|---:|---:|",
                _action_row("EXP002 parent", final["action_baseline"]),
                _action_row(f"EXP004 step {final['selected_step']}", final["action_sft"]),
                "",
                "### Perception retention",
                "",
                "| Model | QA exact | QA token F1 | Grounding click | Median grounding error |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| EXP002 parent | {_pct(perception_base['qa']['answer_accuracy'])} | "
                    f"{perception_base['qa']['mean_token_f1']:.4f} | "
                    f"{_pct(perception_base['grounding']['click_accuracy'])} | "
                    f"{_px(perception_base['grounding']['median_pixel_distance'])} |"
                ),
                (
                    f"| EXP004 step {final['selected_step']} | "
                    f"{_pct(perception_sft['qa']['answer_accuracy'])} | "
                    f"{perception_sft['qa']['mean_token_f1']:.4f} | "
                    f"{_pct(perception_sft['grounding']['click_accuracy'])} | "
                    f"{_px(perception_sft['grounding']['median_pixel_distance'])} |"
                ),
                "",
                f"Preregistered positive-result gate: **{'PASS' if final['positive_result'] else 'FAIL'}**.",
            ]
        )
        action_shards = artifact_root / "final_test/final-action-shard-metrics.json"
        perception_shards = artifact_root / "final_test/final-perception-shard-metrics.json"
        if action_shards.is_file():
            lines.extend(
                [
                    "",
                    "### Selected-checkpoint action shard diagnostics",
                    "",
                    *_action_shard_table(_load(action_shards)),
                ]
            )
        if perception_shards.is_file():
            lines.extend(
                [
                    "",
                    "### Selected-checkpoint perception shard diagnostics",
                    "",
                    *_perception_shard_table(_load(perception_shards)),
                ]
            )

    if closed_loop_path.is_file():
        summary = _load(closed_loop_path)
        lines.extend(
            [
                "",
                "## Deterministic closed loop",
                "",
                "| Variant | Episodes | Success rate | Mean reward | Parse error rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for variant, metrics in summary["variants"].items():
            lines.append(
                f"| {variant} | {metrics['episodes']} | {_pct(metrics['success_rate'])} | "
                f"{metrics['mean_reward']:.4f} | {_pct(metrics['parse_error_rate'])} |"
            )
        comparisons = summary.get("comparisons", {})
        if comparisons:
            lines.extend(
                [
                    "",
                    "| Candidate vs control | Paired N | Success delta | 95% paired bootstrap CI | Reward delta |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for variant, comparison in comparisons.items():
                lower, upper = comparison["success_rate_delta_ci95"]
                lines.append(
                    f"| {variant} vs {summary['control_variant']} | "
                    f"{comparison['paired_episodes']} | "
                    f"{comparison['success_rate_delta']:+.2%} | "
                    f"[{lower:+.2%}, {upper:+.2%}] | "
                    f"{comparison['mean_reward_delta']:+.4f} |"
                )
    if final_path.is_file():
        lines.extend(
            [
                "",
                "## Reproducibility artifacts",
                "",
                "Full matched sealed predictions are stored under `artifacts/final_test/predictions/`.",
                (
                    "Deterministic visual and machine-readable diagnostic samples are stored under "
                    "`artifacts/final_test/failures/`; action errors are separated into output-format, "
                    "semantic-action, action-argument, and spatial-grounding buckets, while perception "
                    "errors distinguish OCR, semantic-understanding, output-format, and spatial-grounding."
                ),
                (
                    "The dashboard payload used for the baseline-versus-selected visual comparison is "
                    "archived at `artifacts/final_test/dashboard.json`."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the publication-facing EXP004 report")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build_exp4_report(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
