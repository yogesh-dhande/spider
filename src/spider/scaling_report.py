from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from spider.eval_receipt import SUITES

DISPLAY_METRICS = (
    ("qa", "answer_accuracy", "QA exact", True),
    ("qa", "mean_token_f1", "QA token F1", True),
    ("grounding", "click_accuracy", "Ground click", True),
    ("grounding", "median_pixel_distance", "Ground median", False),
    ("action", "action_name_accuracy", "Action name", True),
    ("action", "exact_action_accuracy", "Action exact", True),
    ("action", "click_inside_bbox_accuracy", "Action click", True),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(receipt: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for suite in SUITES:
        tasks = receipt["suites"][suite]["merged"]["tasks"]
        for task, metric, _, _ in DISPLAY_METRICS:
            value = tasks.get(task, {}).get(metric)
            if value is not None:
                result[f"{suite}/{task}/{metric}"] = float(value)
    return result


def _deltas(
    values: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    return {
        key: value - reference[key]
        for key, value in values.items()
        if key in reference
    }


def _validate_identity(
    receipt: dict[str, Any], model: str, model_revision: str, *, require_adapter: bool
) -> None:
    if receipt["model"] != model or receipt["model_revision"] != model_revision:
        raise ValueError(
            "Receipt model identity mismatch: "
            f"{receipt['model']}@{receipt['model_revision']} != {model}@{model_revision}"
        )
    if require_adapter and not receipt.get("adapter_sha256"):
        raise ValueError(f"Candidate {receipt['run_id']} lacks an adapter content hash")


def build_scaling_report(
    baseline: dict[str, Any],
    starting_control: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    model = str(baseline["model"])
    model_revision = str(baseline["model_revision"])
    _validate_identity(starting_control, model, model_revision, require_adapter=True)
    if baseline.get("adapter_sha256") is not None:
        raise ValueError("Untouched baseline unexpectedly has an adapter")

    baseline_metrics = _metrics(baseline)
    starting_metrics = _metrics(starting_control)
    rows: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, int, int]] = set()
    seen_adapters: set[str] = set()
    for candidate in candidates:
        receipt = candidate["receipt"]
        _validate_identity(receipt, model, model_revision, require_adapter=True)
        identity = (str(candidate["size"]), int(candidate["seed"]), int(candidate["step"]))
        if identity in seen_identities:
            raise ValueError(f"Duplicate scaling identity: {identity}")
        seen_identities.add(identity)
        adapter_hash = str(receipt["adapter_sha256"])
        if adapter_hash in seen_adapters:
            raise ValueError(f"Adapter reused across candidate runs: {adapter_hash}")
        seen_adapters.add(adapter_hash)
        metrics = _metrics(receipt)
        rows.append(
            {
                "label": str(candidate["label"]),
                "size": identity[0],
                "seed": identity[1],
                "step": identity[2],
                "run_id": receipt["run_id"],
                "adapter_sha256": adapter_hash,
                "metrics": metrics,
                "delta_vs_untouched": _deltas(metrics, baseline_metrics),
                "delta_vs_starting_control": _deltas(metrics, starting_metrics),
            }
        )

    groups: dict[str, Any] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["size"], row["step"])].append(row)
    for (size, step), members in sorted(grouped.items()):
        metric_names = sorted({key for row in members for key in row["metrics"]})
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [row["metrics"][name] for row in members if name in row["metrics"]]
            metrics[name] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        groups[f"{size}@{step}"] = {
            "size": size,
            "step": step,
            "seeds": sorted(row["seed"] for row in members),
            "runs": len(members),
            "metrics": metrics,
        }

    return {
        "schema_version": 1,
        "kind": "exp005_scaling_report",
        "model": model,
        "model_revision": model_revision,
        "controls": {
            "untouched": {
                "run_id": baseline["run_id"],
                "metrics": baseline_metrics,
            },
            "starting_adapter": {
                "run_id": starting_control["run_id"],
                "adapter_sha256": starting_control["adapter_sha256"],
                "metrics": starting_metrics,
            },
        },
        "candidates": rows,
        "groups": groups,
    }


def _display(value: float | None, percent: bool) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.1f} px"


def _group_display(value: dict[str, float] | None, percent: bool) -> str:
    if value is None:
        return "—"
    scale = 100 if percent else 1
    suffix = "%" if percent else " px"
    return f"{scale * value['mean']:.2f} ± {scale * value['sample_std']:.2f}{suffix}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EXP005 SFT scaling comparison",
        "",
        f"Model: `{report['model']}` at `{report['model_revision']}`.",
        "",
    ]
    controls = report["controls"]
    all_rows = [
        ("Untouched", controls["untouched"]["metrics"]),
        ("EXP002 starting adapter", controls["starting_adapter"]["metrics"]),
        *[(row["label"], row["metrics"]) for row in report["candidates"]],
    ]
    for suite in SUITES:
        lines.extend(
            [
                f"## {suite}",
                "",
                "| Run | " + " | ".join(item[2] for item in DISPLAY_METRICS) + " |",
                "|---|" + "---:|" * len(DISPLAY_METRICS),
            ]
        )
        for label, metrics in all_rows:
            values = []
            for task, metric, _, percent in DISPLAY_METRICS:
                values.append(_display(metrics.get(f"{suite}/{task}/{metric}"), percent))
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        lines.append("")

    if report["groups"]:
        lines.extend(["## Across-seed summaries", ""])
        for suite in SUITES:
            lines.extend(
                [
                    f"### {suite}",
                    "",
                    "| Size@step (seeds) | "
                    + " | ".join(item[2] for item in DISPLAY_METRICS)
                    + " |",
                    "|---|" + "---:|" * len(DISPLAY_METRICS),
                ]
            )
            for key, group in report["groups"].items():
                values = []
                for task, metric, _, percent in DISPLAY_METRICS:
                    name = f"{suite}/{task}/{metric}"
                    values.append(_group_display(group["metrics"].get(name), percent))
                seeds = ", ".join(str(seed) for seed in group["seeds"])
                lines.append(f"| {key} ({seeds}) | " + " | ".join(values) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a baseline/control/SFT scaling report")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    manifest = _load(args.manifest)
    parent = args.manifest.resolve().parent

    def receipt(relative: str) -> dict[str, Any]:
        path = Path(relative)
        return _load(path if path.is_absolute() else parent / path)

    candidates = [
        {**item, "receipt": receipt(item["receipt"])}
        for item in manifest["candidates"]
    ]
    report = build_scaling_report(
        receipt(manifest["baseline_receipt"]),
        receipt(manifest["starting_control_receipt"]),
        candidates,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
