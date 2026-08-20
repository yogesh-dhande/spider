import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "archive_exp004_gcloud_training.py"
SPEC = importlib.util.spec_from_file_location("archive_exp004_gcloud_training", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_state_requires_registered_schedule() -> None:
    state = {
        "completed_step": 375,
        "effective_batch_size": 16,
        "planned_epoch_steps": 1875,
        "stage_runtime_seconds": 100.0,
    }
    MODULE.validate_state(state, 375)
    state["effective_batch_size"] = 8
    with pytest.raises(RuntimeError, match="effective batch"):
        MODULE.validate_state(state, 375)


def test_archive_training_rejects_invalid_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        MODULE.archive_training("", 0, tmp_path)
