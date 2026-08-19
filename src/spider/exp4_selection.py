from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("-inf")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite checkpoint-selection metric {key}: {value}")
    return result


def select_exp4_checkpoint(gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply EXP004's frozen lexicographic development-set selection rule."""
    if not gates:
        raise ValueError("No EXP004 validation gates were supplied")
    eligible = []
    seen_steps: set[int] = set()
    for gate in gates:
        step = int(gate["step"])
        if step in seen_steps:
            raise ValueError(f"Duplicate EXP004 validation step: {step}")
        seen_steps.add(step)
        if gate.get("perception_regressions") or gate.get("action_regressions"):
            continue
        metrics = gate.get("action_candidate")
        if not isinstance(metrics, dict):
            raise TypeError(f"Step {step} has no action_candidate metrics")
        score = (
            _finite_metric(metrics, "click_inside_bbox_accuracy"),
            _finite_metric(metrics, "action_name_accuracy"),
            _finite_metric(metrics, "action_argument_accuracy"),
            -step,
        )
        eligible.append((score, gate))
    if not eligible:
        raise RuntimeError("No EXP004 checkpoint passed the registered regression gate")
    _, selected = max(eligible, key=lambda item: item[0])
    return {
        "selected_step": int(selected["step"]),
        "selection_rule": [
            "click_inside_bbox_accuracy",
            "action_name_accuracy",
            "action_argument_accuracy",
            "earliest_step",
        ],
        "eligible_steps": sorted(int(gate["step"]) for _, gate in eligible),
        "selected_action_metrics": selected["action_candidate"],
        "selected_perception_metrics": selected["perception_candidate"],
    }


def select_from_directory(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    paths = sorted(root.glob("step_*/gate.json"))
    gates = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return select_exp4_checkpoint(gates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select EXP004 checkpoint from development gates")
    parser.add_argument("gate_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = select_from_directory(args.gate_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
