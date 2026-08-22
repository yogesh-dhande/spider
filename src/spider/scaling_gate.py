from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

PERCEPTION_METRICS = (
    "iid/qa/answer_accuracy",
    "domain_balanced/qa/answer_accuracy",
    "iid/grounding/click_accuracy",
    "domain_balanced/grounding/click_accuracy",
    "distribution_shift/grounding/click_accuracy",
)
ACTION_METRICS = tuple(
    f"{suite}/action/{metric}"
    for suite in ("iid", "domain_balanced", "distribution_shift")
    for metric in ("action_name_accuracy", "exact_action_accuracy", "click_inside_bbox_accuracy")
)
MEAN_REGRESSION_LIMIT = -0.03
SINGLE_METRIC_REGRESSION_LIMIT = -0.075
WARNING_LIMIT = -0.03


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(receipt: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for suite, suite_data in receipt["suites"].items():
        for task, values in suite_data["merged"]["tasks"].items():
            for name, value in values.items():
                if isinstance(value, (int, float)) and name != "examples":
                    result[f"{suite}/{task}/{name}"] = float(value)
    return result


def build_gate(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    untouched: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = (reference["model"], reference["model_revision"])
    if (candidate["model"], candidate["model_revision"]) != identity:
        raise ValueError("Candidate and regression reference use different base models")
    if not candidate.get("adapter_sha256"):
        raise ValueError("Candidate lacks a content-addressed adapter")
    reference_metrics = _metrics(reference)
    candidate_metrics = _metrics(candidate)
    perception_deltas = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in PERCEPTION_METRICS
        if key in candidate_metrics and key in reference_metrics
    }
    if set(perception_deltas) != set(PERCEPTION_METRICS):
        missing = set(PERCEPTION_METRICS) - set(perception_deltas)
        raise ValueError(f"Regression gate is missing perception metrics: {sorted(missing)}")
    action_deltas = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in ACTION_METRICS
        if key in candidate_metrics and key in reference_metrics
    }
    mean_perception_delta = statistics.fmean(perception_deltas.values())
    worst_key, worst_delta = min(perception_deltas.items(), key=lambda item: item[1])
    hard_regression = (
        mean_perception_delta < MEAN_REGRESSION_LIMIT
        or worst_delta < SINGLE_METRIC_REGRESSION_LIMIT
    )
    warnings = {
        key: value for key, value in perception_deltas.items() if value < WARNING_LIMIT
    }
    gate: dict[str, Any] = {
        "schema_version": 1,
        "kind": "exp005_checkpoint_regression_gate",
        "reference_run_id": reference["run_id"],
        "candidate_run_id": candidate["run_id"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "model": candidate["model"],
        "model_revision": candidate["model_revision"],
        "thresholds": {
            "mean_perception_delta": MEAN_REGRESSION_LIMIT,
            "single_perception_metric_delta": SINGLE_METRIC_REGRESSION_LIMIT,
            "warning_delta": WARNING_LIMIT,
        },
        "perception_deltas": perception_deltas,
        "mean_perception_delta": mean_perception_delta,
        "worst_perception_metric": worst_key,
        "worst_perception_delta": worst_delta,
        "perception_warnings": warnings,
        "action_deltas": action_deltas,
        "mean_action_delta": statistics.fmean(action_deltas.values()) if action_deltas else None,
        "hard_regression": hard_regression,
        "decision": "stop_regression" if hard_regression else "continue",
    }
    if untouched is not None:
        if (untouched["model"], untouched["model_revision"]) != identity:
            raise ValueError("Untouched baseline uses a different base model")
        untouched_metrics = _metrics(untouched)
        gate["delta_vs_untouched"] = {
            key: value - untouched_metrics[key]
            for key, value in candidate_metrics.items()
            if key in untouched_metrics
        }
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the registered EXP005 checkpoint gate")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--untouched", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = build_gate(
        _load(args.reference),
        _load(args.candidate),
        untouched=_load(args.untouched) if args.untouched else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if gate["hard_regression"]:
        raise SystemExit("Checkpoint failed the registered regression gate")


if __name__ == "__main__":
    main()
