import json
from pathlib import Path

from spider.eval_receipt import build_receipt, render_markdown


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_receipt_validates_and_reports_shard_variability(tmp_path: Path) -> None:
    for suite in ("iid", "domain_balanced", "distribution_shift"):
        merged_tasks = {
            "grounding": {"examples": 2, "parse_rate": 1.0, "click_accuracy": 0.5, "median_pixel_distance": 20.0},
            "action": {
                "examples": 2,
                "json_parse_rate": 1.0,
                "action_name_accuracy": 0.5,
                "action_argument_accuracy": 0.0,
                "exact_action_accuracy": 0.0,
                "click_inside_bbox_accuracy": 0.25,
                "click_median_distance_px": 30.0,
            },
        }
        if suite != "distribution_shift":
            merged_tasks["qa"] = {"examples": 2, "answer_accuracy": 0.5, "mean_token_f1": 0.75}
        _write(tmp_path / suite / "metrics.json", {"examples": 4, "tasks": merged_tasks})
        _write(tmp_path / suite / "run_metadata.json", {"merged": True})
        _write(
            tmp_path / suite / "complete.json",
            {"run_id": "run-a", "control": "base", "suite": suite, "status": "complete", "exit_code": 0},
        )
        for shard in range(2):
            tasks = {
                "grounding": {"examples": 1, "parse_rate": 1.0, "click_accuracy": float(shard), "median_pixel_distance": 20.0},
                "action": {
                    "examples": 1,
                    "json_parse_rate": 1.0,
                    "action_name_accuracy": float(shard),
                    "action_argument_accuracy": 0.0,
                    "exact_action_accuracy": 0.0,
                    "click_inside_bbox_accuracy": 0.0,
                    "click_median_distance_px": 30.0,
                },
            }
            if suite != "distribution_shift":
                tasks["qa"] = {"examples": 1, "answer_accuracy": float(shard), "mean_token_f1": 0.75}
            root = tmp_path / "shards" / suite / f"{shard:02d}"
            _write(root / "metrics.json", {"examples": 2, "tasks": tasks})
            _write(
                root / "run_metadata.json",
                {"model": "model", "model_revision": "revision", "adapter": None, "signature": f"{suite}-{shard}"},
            )
            _write(
                root / "complete.json",
                {
                    "run_id": "run-a",
                    "control": "base",
                    "suite": suite,
                    "shard_index": shard,
                    "num_shards": 2,
                    "status": "complete",
                    "exit_code": 0,
                },
            )

    receipt = build_receipt(
        tmp_path,
        run_id="run-a",
        control="base",
        expected_model="model",
        expected_model_revision="revision",
        num_shards=2,
    )
    variability = receipt["suites"]["iid"]["shard_variability"]
    assert variability["grounding"]["click_accuracy"]["mean"] == 0.5
    assert variability["grounding"]["click_accuracy"]["min"] == 0.0
    assert variability["grounding"]["click_accuracy"]["max"] == 1.0
    markdown = render_markdown(receipt)
    assert "Merged (4 examples)" in markdown
    assert "50.00 ± 70.71%" in markdown
    assert "| 0 | 2 |" in markdown
