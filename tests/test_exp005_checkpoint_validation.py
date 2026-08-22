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


def test_checkpoint_validation_cli_exposes_opt_in_warm_image() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--warm-image")' in source
    assert 'evaluation_command.extend(["--warm-image", args.warm_image])' in source


def test_checkpoint_validation_uses_short_capacity_retry_backoff() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--evaluation-retry-seconds", type=int, default=60)' in source
    assert 'str(args.evaluation_retry_seconds)' in source


def test_validate_training_receipt_allows_exact_healthy_reuse(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    receipt = {
        "kind": "exp005_training_stage_receipt",
        "run_id": "run",
        "job_id": "job",
        "status": "complete_pass",
        "start_step": 0,
        "completed_step": 500,
        "num_nodes": 2,
        "adapter": {"sha256": "hash", "health": {"status": "healthy"}},
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert MODULE.validate_training_receipt(
        path,
        run_id="run",
        job_id="job",
        start_step=0,
        stop_step=500,
        num_nodes=2,
    ) == receipt
    receipt["adapter"]["health"]["status"] = "failed"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="healthy adapter"):
        MODULE.validate_training_receipt(
            path,
            run_id="run",
            job_id="job",
            start_step=0,
            stop_step=500,
            num_nodes=2,
        )
