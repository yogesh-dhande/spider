#!/usr/bin/env python3
"""Refresh the hosted diagnostic payload from validated EXP005 evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from spider.dashboard import (
    build_probe_dashboard,
    copy_action_dashboard_images,
    copy_perception_dashboard_images,
    write_dashboard_json,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merged_predictions(root: Path, suite: str, control: str) -> Path:
    path = (
        root
        / suite
        / "benchmark_evaluation"
        / f"{control}-{suite}"
        / "predictions.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Merged predictions not found: {path}")
    return path


def verify_dashboard_metrics(
    payload: dict[str, Any], receipt: dict[str, Any], label: str, suite: str
) -> None:
    expected = receipt["suites"][suite]["merged"]["tasks"]
    checks = {
        "qa exact": (
            payload["qa"]["metrics"][label]["exact_accuracy"],
            expected["qa"]["answer_accuracy"],
        ),
        "qa token F1": (
            payload["qa"]["metrics"][label]["mean_token_f1"],
            expected["qa"]["mean_token_f1"],
        ),
        "grounding click": (
            payload["grounding"]["metrics"][label]["click_accuracy"],
            expected["grounding"]["click_accuracy"],
        ),
        "grounding median": (
            payload["grounding"]["metrics"][label]["median_pixel_distance"],
            expected["grounding"]["median_pixel_distance"],
        ),
        "action name": (
            payload["action"]["metrics"][label]["action_name_accuracy"],
            expected["action"]["action_name_accuracy"],
        ),
        "action exact": (
            payload["action"]["metrics"][label]["exact_action_accuracy"],
            expected["action"]["exact_action_accuracy"],
        ),
        "action click": (
            payload["action"]["metrics"][label]["click_inside_bbox_accuracy"],
            expected["action"]["click_inside_bbox_accuracy"],
        ),
    }
    mismatches = {
        name: {"dashboard": actual, "receipt": target}
        for name, (actual, target) in checks.items()
        if actual is None
        or target is None
        or not math.isclose(float(actual), float(target), rel_tol=0, abs_tol=1e-12)
    }
    if mismatches:
        raise ValueError(f"Dashboard/receipt metric mismatch for {label}: {mismatches}")


def refresh_dashboard(
    *,
    baseline_root: Path,
    latest_root: Path,
    baseline_receipt_path: Path,
    latest_receipt_path: Path,
    latest_name: str,
    latest_step: int,
    suite: str,
    corpus_root: Path,
    dashboard_json: Path,
    public_images: Path,
    display_limit: int,
) -> dict[str, Any]:
    prediction_paths = {
        "baseline": merged_predictions(baseline_root, suite, "base"),
        "latest": merged_predictions(latest_root, suite, "sft"),
    }
    payload = build_probe_dashboard(
        prediction_paths,
        checkpoint_labels={
            "baseline": "Untouched Qwen3.5-2B",
            "latest": latest_name,
        },
        latest_label="latest",
        latest_step=latest_step,
        action_prediction_paths=prediction_paths,
        action_display_limit=display_limit,
        perception_display_limit=display_limit,
        split=f"EXP005 {suite}",
    )
    verify_dashboard_metrics(payload, _load(baseline_receipt_path), "baseline", suite)
    verify_dashboard_metrics(payload, _load(latest_receipt_path), "latest", suite)
    copy_perception_dashboard_images(payload, corpus_root, public_images)
    copy_action_dashboard_images(payload["action"], corpus_root, public_images / "action")
    write_dashboard_json(payload, dashboard_json)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--latest-receipt", type=Path, required=True)
    parser.add_argument("--latest-name", required=True)
    parser.add_argument("--latest-step", type=int, required=True)
    parser.add_argument("--suite", choices=("iid", "domain_balanced"), default="iid")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument(
        "--dashboard-json", type=Path, default=Path("dataset-dashboard/app/qa-probe.json")
    )
    parser.add_argument(
        "--public-images", type=Path, default=Path("dataset-dashboard/public/images")
    )
    parser.add_argument("--display-limit", type=int, default=64)
    args = parser.parse_args()
    payload = refresh_dashboard(
        baseline_root=args.baseline_root,
        latest_root=args.latest_root,
        baseline_receipt_path=args.baseline_receipt,
        latest_receipt_path=args.latest_receipt,
        latest_name=args.latest_name,
        latest_step=args.latest_step,
        suite=args.suite,
        corpus_root=args.corpus_root,
        dashboard_json=args.dashboard_json,
        public_images=args.public_images,
        display_limit=args.display_limit,
    )
    print(
        json.dumps(
            {
                "event": "exp005_dashboard_refreshed",
                "latest": args.latest_name,
                "step": args.latest_step,
                "suite": args.suite,
                "qa_examples": payload["qa"]["meta"]["examples"],
                "grounding_examples": payload["grounding"]["meta"]["examples"],
                "action_examples": payload["action"]["meta"]["scored_examples"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
