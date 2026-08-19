from __future__ import annotations

import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "chain_exp004_kaggle.py"
SPEC = spec_from_file_location("chain_exp004_kaggle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_status_retries_transient_api_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = iter(
        [
            subprocess.CalledProcessError(1, ["kaggle"], stderr="read timed out"),
            'job has status "KernelWorkerStatus.RUNNING"',
        ]
    )

    def fake_run(*args, **kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(MODULE, "run", fake_run)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    assert MODULE.status("job") == "RUNNING"


def test_status_or_missing_does_not_hide_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = subprocess.CalledProcessError(1, ["kaggle"], stderr="read timed out")
    monkeypatch.setattr(MODULE, "status", lambda _: (_ for _ in ()).throw(failure))
    with pytest.raises(subprocess.CalledProcessError):
        MODULE.status_or_missing("job")
    failure.stderr = "Cannot access kernel 'owner/job'"
    assert MODULE.status_or_missing("job") == "MISSING"


def test_validate_stage_requires_exact_resumable_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {
        "status": "complete",
        "start_step": 500,
        "completed_step": 750,
        "stop_step": 750,
        "planned_epoch_steps": 1875,
        "world_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
    }
    monkeypatch.setattr(MODULE, "download_json", lambda *args, **kwargs: state)
    assert MODULE.validate_stage(2)["completed_step"] == 750
    state["world_size"] = 1
    with pytest.raises(RuntimeError, match="world_size"):
        MODULE.validate_stage(2)


def test_validate_gate_checks_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "download_json",
        lambda *args, **kwargs: {"step": 250, "advance": True},
    )
    assert MODULE.validate_gate(250)["advance"] is True
    with pytest.raises(RuntimeError, match="Wrong validation step"):
        MODULE.validate_gate(500)


def test_validate_baselines_requires_full_development_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "examples": 256,
        "json_parse_rate": 0.9,
        "action_name_accuracy": 0.5,
        "action_argument_accuracy": 0.3,
    }
    monkeypatch.setattr(
        MODULE, "download_json", lambda *args, **kwargs: json.loads(json.dumps(metrics))
    )
    assert MODULE.validate_baselines()["exp002"]["examples"] == 256
