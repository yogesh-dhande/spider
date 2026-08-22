import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_exp005_checkpoint_validation.py"
SPEC = importlib.util.spec_from_file_location("run_exp005_checkpoint_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_validate_rank_terminal_requires_exact_stage_identity() -> None:
    terminal = {
        "run_id": "run-a",
        "job_id": "job-a",
        "start_step": 0,
        "stop_step": 500,
        "node_rank": 0,
        "num_nodes": 2,
        "status": "complete",
        "exit_code": 0,
    }
    MODULE.validate_rank_terminal(
        terminal,
        run_id="run-a",
        job_id="job-a",
        start_step=0,
        stop_step=500,
        rank=0,
        num_nodes=2,
    )
    terminal["stop_step"] = 499
    with pytest.raises(ValueError, match="stop_step"):
        MODULE.validate_rank_terminal(
            terminal,
            run_id="run-a",
            job_id="job-a",
            start_step=0,
            stop_step=500,
            rank=0,
            num_nodes=2,
        )


def test_verify_adapter_identity_binds_evaluation_to_checkpoint(tmp_path: Path) -> None:
    training = tmp_path / "training.json"
    evaluation = tmp_path / "evaluation.json"
    training.write_text(json.dumps({"adapter": {"sha256": "checkpoint"}}))
    evaluation.write_text(json.dumps({"adapter_sha256": "checkpoint"}))
    MODULE.verify_adapter_identity(training, evaluation)
    evaluation.write_text(json.dumps({"adapter_sha256": "other"}))
    with pytest.raises(ValueError, match="differs from checkpoint"):
        MODULE.verify_adapter_identity(training, evaluation)
