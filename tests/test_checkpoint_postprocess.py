import json
from pathlib import Path

from spider.checkpoint_postprocess import process_checkpoint


def _receipt(run_id: str, adapter: str | None, value: float, *, control: str) -> dict:
    suites = {}
    for suite in ("iid", "domain_balanced", "distribution_shift"):
        tasks = {
            "grounding": {"examples": 10, "click_accuracy": value, "median_pixel_distance": 10.0},
            "action": {
                "examples": 10,
                "action_name_accuracy": value,
                "exact_action_accuracy": value / 2,
                "click_inside_bbox_accuracy": value / 3,
            },
        }
        if suite != "distribution_shift":
            tasks["qa"] = {
                "examples": 10,
                "answer_accuracy": value,
                "mean_token_f1": value,
            }
        suites[suite] = {"merged": {"tasks": tasks}}
    return {
        "kind": "evaluation_receipt",
        "control": control,
        "run_id": run_id,
        "model": "model",
        "model_revision": "revision",
        "adapter_sha256": adapter,
        "suites": suites,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_process_checkpoint_is_idempotent_and_writes_all_outputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    starting = tmp_path / "starting.json"
    candidate = tmp_path / "candidate.json"
    manifest = tmp_path / "manifest.json"
    gate = tmp_path / "gates" / "candidate.json"
    report_json = tmp_path / "report.json"
    report_markdown = tmp_path / "report.md"
    _write(baseline, _receipt("base", None, 0.2, control="base"))
    _write(starting, _receipt("start", "start-hash", 0.3, control="exp002"))
    _write(candidate, _receipt("candidate", "candidate-hash", 0.4, control="sft"))
    _write(
        manifest,
        {
            "schema_version": 1,
            "baseline_receipt": "baseline.json",
            "starting_control_receipt": "starting.json",
            "candidates": [],
        },
    )
    arguments = {
        "reference_path": starting,
        "candidate_path": candidate,
        "untouched_path": baseline,
        "gate_path": gate,
        "manifest_path": manifest,
        "report_json_path": report_json,
        "report_markdown_path": report_markdown,
        "label": "10K seed 53 step 500",
        "size": "small",
        "seed": 53,
        "step": 500,
    }

    first = process_checkpoint(**arguments)
    second = process_checkpoint(**arguments)

    assert first == second
    assert first["decision"] == "continue"
    assert len(json.loads(manifest.read_text())["candidates"]) == 1
    assert json.loads(gate.read_text())["candidate_run_id"] == "candidate"
    assert len(json.loads(report_json.read_text())["candidates"]) == 1
    assert "10K seed 53 step 500" in report_markdown.read_text()
