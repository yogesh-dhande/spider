from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spider.config import experiment_path, load_config
from spider.evaluate import evaluate


def primary_probe_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "qa_answer_accuracy": float(metrics["molmoweb"]["qa"]["answer_accuracy"]),
        "qa_mean_token_f1": float(metrics["molmoweb"]["qa"]["mean_token_f1"]),
        "grounding_click_accuracy": float(
            metrics["molmoweb"]["grounding"]["click_accuracy"]
        ),
        "grounding_parse_rate": float(metrics["molmoweb"]["grounding"]["parse_rate"]),
        "grounding_median_pixel_distance": float(
            metrics["molmoweb"]["grounding"]["median_pixel_distance"]
        ),
    }


def run_validation_probe(
    config_path: str | Path,
    label: str,
    adapter: str | None,
    step: int,
    limit_per_task: int = 128,
) -> Path:
    if step < 0 or limit_per_task <= 0:
        raise ValueError("Probe step must be non-negative and limit must be positive")
    config = load_config(config_path)
    predictions_path, metrics = evaluate(
        config_path,
        label,
        adapter,
        ["molmoweb"],
        split="validation",
        limit=limit_per_task,
    )
    expected = limit_per_task * 2
    completed = sum(1 for _ in predictions_path.open(encoding="utf-8"))
    if completed != expected:
        raise RuntimeError(f"Validation probe expected {expected} predictions, found {completed}")
    summary = {
        "kind": "fixed_validation_probe",
        "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "label": label,
        "step": step,
        "adapter": adapter,
        "split": "validation",
        "limit_per_task": limit_per_task,
        "completed_predictions": completed,
        "primary_metrics": primary_probe_metrics(metrics),
        "metrics": metrics,
    }
    output_dir = experiment_path(config, "output_dir") / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "validation_probe_complete", **summary["primary_metrics"]}))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed MolmoWeb validation probe")
    parser.add_argument("--config", default="configs/experiment2.yaml")
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit-per-task", type=int, default=128)
    args = parser.parse_args()
    run_validation_probe(
        args.config,
        args.label,
        args.adapter,
        args.step,
        args.limit_per_task,
    )


if __name__ == "__main__":
    main()
