from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "chain_exp002_kaggle_stages.py"
SPEC = spec_from_file_location("chain_exp002_kaggle_stages", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
probe_regressions = MODULE.probe_regressions
validate_download = MODULE.validate_download


def make_stage(root: Path, *, start: int = 250, stop: int = 500) -> Path:
    output = root / "spider" / "outputs" / "experiment2"
    checkpoint = output / "adapter" / f"checkpoint-{stop}"
    checkpoint.mkdir(parents=True)
    state = {
        "status": "complete",
        "start_step": start,
        "completed_step": stop,
        "stop_step": stop,
        "planned_epoch_steps": 1875,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "optimizer": "adamw_8bit",
        "checkpoint": f"adapter/checkpoint-{stop}",
        "resumed_from": f"/input/adapter/checkpoint-{start}",
        "stage_runtime_seconds": 10.0,
        "metrics": {"train_loss": 0.5},
    }
    (output / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    trainer = {
        "global_step": stop,
        "max_steps": 1875,
        "log_history": [
            {"step": stop, "eval_loss": 0.6, "eval_mean_token_accuracy": 0.8}
        ],
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(trainer), encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"x" * 1_000_000)
    for name in ("scheduler.pt", "rng_state_0.pth", "rng_state_1.pth"):
        (checkpoint / name).write_bytes(b"x" * 100)
    return output


def test_validate_download_accepts_exact_resumable_stage(tmp_path: Path) -> None:
    make_stage(tmp_path)
    result = validate_download(tmp_path, 1)
    assert result["completed_step"] == 500
    assert result["eval_loss"] == 0.6


def test_validate_download_rejects_wrong_start(tmp_path: Path) -> None:
    output = make_stage(tmp_path)
    state_path = output / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["start_step"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="start_step"):
        validate_download(tmp_path, 1)


def test_probe_regressions_uses_task_tolerances() -> None:
    anchor = {
        "qa_answer_accuracy": 0.36,
        "qa_mean_token_f1": 0.64,
        "grounding_click_accuracy": 0.47,
        "grounding_parse_rate": 1.0,
        "grounding_median_pixel_distance": 39.0,
    }
    within_tolerance = {**anchor, "qa_answer_accuracy": 0.34}
    assert probe_regressions(anchor, within_tolerance) == {}
    regressed = {
        **anchor,
        "grounding_click_accuracy": 0.40,
        "grounding_median_pixel_distance": 70.0,
    }
    assert set(probe_regressions(anchor, regressed)) == {
        "grounding_click_accuracy",
        "grounding_median_pixel_distance",
    }
