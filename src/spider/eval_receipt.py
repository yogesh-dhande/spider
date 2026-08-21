from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SUITES = ("iid", "domain_balanced", "distribution_shift")
PRIMARY_METRICS = {
    "qa": ("answer_accuracy", "mean_token_f1"),
    "grounding": ("parse_rate", "click_accuracy", "median_pixel_distance"),
    "action": (
        "json_parse_rate",
        "action_name_accuracy",
        "action_argument_accuracy",
        "exact_action_accuracy",
        "click_inside_bbox_accuracy",
        "click_median_distance_px",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primary(metrics: dict[str, Any]) -> dict[str, Any]:
    tasks = metrics.get("tasks", {})
    result: dict[str, Any] = {"examples": int(metrics["examples"]), "tasks": {}}
    for task, names in PRIMARY_METRICS.items():
        if task not in tasks:
            continue
        source = tasks[task]
        result["tasks"][task] = {
            "examples": int(source["examples"]),
            **{name: source.get(name) for name in names},
        }
    return result


def _variability(shards: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tasks = sorted({task for shard in shards for task in shard["metrics"]["tasks"]})
    for task in tasks:
        names = PRIMARY_METRICS[task]
        task_result: dict[str, Any] = {}
        for name in names:
            values = [
                shard["metrics"]["tasks"][task].get(name)
                for shard in shards
                if task in shard["metrics"]["tasks"]
                and shard["metrics"]["tasks"][task].get(name) is not None
            ]
            numeric = [float(value) for value in values]
            if not numeric:
                continue
            task_result[name] = {
                "mean": statistics.fmean(numeric),
                "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
                "min": min(numeric),
                "max": max(numeric),
            }
        result[task] = task_result
    return result


def build_receipt(
    root: Path,
    *,
    run_id: str,
    control: str,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    num_shards: int = 4,
) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    model_identities: set[tuple[str, str, str | None]] = set()
    for suite in SUITES:
        merged_path = root / suite / "metrics.json"
        merged_terminal = _load(root / suite / "complete.json")
        expected_merge = {
            "run_id": run_id,
            "control": control,
            "suite": suite,
            "status": "complete",
            "exit_code": 0,
        }
        assert all(merged_terminal.get(key) == value for key, value in expected_merge.items())

        shards: list[dict[str, Any]] = []
        for shard_index in range(num_shards):
            shard_root = root / "shards" / suite / f"{shard_index:02d}"
            terminal = _load(shard_root / "complete.json")
            expected_terminal = {
                "run_id": run_id,
                "control": control,
                "suite": suite,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "status": "complete",
                "exit_code": 0,
            }
            assert all(terminal.get(key) == value for key, value in expected_terminal.items())
            metadata = _load(shard_root / "run_metadata.json")
            model_identities.add(
                (str(metadata["model"]), str(metadata["model_revision"]), metadata.get("adapter"))
            )
            metrics_path = shard_root / "metrics.json"
            shards.append(
                {
                    "shard_index": shard_index,
                    "metrics": _primary(_load(metrics_path)),
                    "signature": metadata["signature"],
                    "metrics_sha256": _sha256(metrics_path),
                    "run_metadata_sha256": _sha256(shard_root / "run_metadata.json"),
                }
            )

        merged = _primary(_load(merged_path))
        assert sum(item["metrics"]["examples"] for item in shards) == merged["examples"]
        suites[suite] = {
            "merged": merged,
            "shards": shards,
            "shard_variability": _variability(shards),
            "metrics_sha256": _sha256(merged_path),
            "run_metadata_sha256": _sha256(root / suite / "run_metadata.json"),
        }

    assert len(model_identities) == 1, model_identities
    model, model_revision, adapter = next(iter(model_identities))
    if expected_model is not None:
        assert model == expected_model, (model, expected_model)
    if expected_model_revision is not None:
        assert model_revision == expected_model_revision, (model_revision, expected_model_revision)
    if control == "base":
        assert adapter is None, adapter

    return {
        "schema_version": 1,
        "kind": "evaluation_receipt",
        "run_id": run_id,
        "control": control,
        "model": model,
        "model_revision": model_revision,
        "adapter": adapter,
        "num_shards": num_shards,
        "suites": suites,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _px(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} px"


def _summary(variability: dict[str, Any], task: str, metric: str, *, percent: bool) -> str:
    value = variability.get(task, {}).get(metric)
    if value is None:
        return "—"
    scale = 100 if percent else 1
    suffix = "%" if percent else " px"
    return f"{scale * value['mean']:.2f} ± {scale * value['sample_std']:.2f}{suffix}"


def render_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation receipt: {receipt['run_id']}",
        "",
        f"Control: `{receipt['control']}`. Model: `{receipt['model']}` at "
        f"`{receipt['model_revision']}`.",
        "",
    ]
    for suite in SUITES:
        suite_data = receipt["suites"][suite]
        merged = suite_data["merged"]["tasks"]
        variability = suite_data["shard_variability"]
        lines.extend(
            [
                f"## {suite}",
                "",
                "| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| Merged ({suite_data['merged']['examples']} examples) | "
                    f"{_pct(merged.get('qa', {}).get('answer_accuracy'))} | "
                    f"{_pct(merged.get('qa', {}).get('mean_token_f1'))} | "
                    f"{_pct(merged.get('grounding', {}).get('click_accuracy'))} | "
                    f"{_px(merged.get('grounding', {}).get('median_pixel_distance'))} | "
                    f"{_pct(merged.get('action', {}).get('action_name_accuracy'))} | "
                    f"{_pct(merged.get('action', {}).get('exact_action_accuracy'))} | "
                    f"{_pct(merged.get('action', {}).get('click_inside_bbox_accuracy'))} |"
                ),
                (
                    "| Unweighted shard mean ± sample SD | "
                    f"{_summary(variability, 'qa', 'answer_accuracy', percent=True)} | "
                    f"{_summary(variability, 'qa', 'mean_token_f1', percent=True)} | "
                    f"{_summary(variability, 'grounding', 'click_accuracy', percent=True)} | "
                    f"{_summary(variability, 'grounding', 'median_pixel_distance', percent=False)} | "
                    f"{_summary(variability, 'action', 'action_name_accuracy', percent=True)} | "
                    f"{_summary(variability, 'action', 'exact_action_accuracy', percent=True)} | "
                    f"{_summary(variability, 'action', 'click_inside_bbox_accuracy', percent=True)} |"
                ),
                "",
                "| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for shard in suite_data["shards"]:
            tasks = shard["metrics"]["tasks"]
            lines.append(
                f"| {shard['shard_index']} | {shard['metrics']['examples']} | "
                f"{_pct(tasks.get('qa', {}).get('answer_accuracy'))} | "
                f"{_pct(tasks.get('grounding', {}).get('click_accuracy'))} | "
                f"{_px(tasks.get('grounding', {}).get('median_pixel_distance'))} | "
                f"{_pct(tasks.get('action', {}).get('action_name_accuracy'))} | "
                f"{_pct(tasks.get('action', {}).get('exact_action_accuracy'))} | "
                f"{_pct(tasks.get('action', {}).get('click_inside_bbox_accuracy'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize sharded evaluations")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        args.root,
        run_id=args.run_id,
        control=args.control,
        expected_model=args.expected_model,
        expected_model_revision=args.expected_model_revision,
        num_shards=args.num_shards,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(receipt), encoding="utf-8")


if __name__ == "__main__":
    main()
