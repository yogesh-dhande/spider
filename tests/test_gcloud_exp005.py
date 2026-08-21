import importlib.util
import subprocess
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gcloud_exp005.py"
SPEC = importlib.util.spec_from_file_location("gcloud_exp005", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_managed_filter_is_narrowly_scoped() -> None:
    assert MODULE.MANAGED_FILTER == (
        "labels.spider-managed=true AND labels.spider-experiment=exp005"
    )


def test_materialization_shard_rejects_invalid_partition() -> None:
    try:
        MODULE.create_materialization_shard("run", "zone", "revision", 4, 4, "6h")
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("invalid materialization shard was accepted")


def test_materialization_merge_rejects_non_positive_shards() -> None:
    try:
        MODULE.create_materialization_merge("run", "zone", "revision", 0, "4h")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid materialization merge was accepted")


def test_training_stage_rejects_invalid_bounds() -> None:
    try:
        MODULE.create_training_stage("run", "zone", "revision", "small-seed53", 125, 125, "4h")
    except ValueError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("invalid training stage bounds were accepted")


def test_training_stage_rejects_unsafe_job_id() -> None:
    try:
        MODULE.create_training_stage("run", "zone", "revision", "Unsafe Job", 0, 125, "4h")
    except ValueError as error:
        assert "job_id" in str(error)
    else:
        raise AssertionError("unsafe training job ID was accepted")


def test_training_stage_uses_two_l4_shape(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    name = MODULE.create_training_stage(
        "run-a", "us-west1-b", "abc123", "small-seed53", 0, 125, "4h"
    )

    assert name == "spider-exp005-train-00125-run-a"
    assert received["machine_type"] == "g2-standard-24"
    assert received["gpu"] is True
    assert received["metadata"] == {
        "spider-job-id": "small-seed53",
        "spider-stage-start": 0,
        "spider-stage-stop": 125,
    }


def test_evaluation_rejects_unknown_control() -> None:
    try:
        MODULE.create_evaluation_shard(
            "run", "zone", "revision", "other", "iid", 0, 4, "4h"
        )
    except ValueError as error:
        assert "base, exp002, or sft" in str(error)
    else:
        raise AssertionError("unknown evaluation control was accepted")


def test_evaluation_rejects_unknown_suite() -> None:
    try:
        MODULE.create_evaluation_shard(
            "run", "zone", "revision", "base", "other", 0, 4, "4h"
        )
    except ValueError as error:
        assert "suite" in str(error)
    else:
        raise AssertionError("unknown evaluation suite was accepted")


def test_evaluation_rejects_invalid_partition() -> None:
    try:
        MODULE.create_evaluation_shard(
            "run", "zone", "revision", "base", "iid", -1, 4, "4h"
        )
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("invalid evaluation shard was accepted")


def test_sft_evaluation_requires_checkpoint_identity() -> None:
    try:
        MODULE.create_evaluation_shard(
            "run", "zone", "revision", "sft", "iid", 0, 4, "4h"
        )
    except ValueError as error:
        assert "training_job" in str(error)
    else:
        raise AssertionError("SFT evaluation without a checkpoint was accepted")


def test_sft_evaluation_passes_checkpoint_identity(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    MODULE.create_evaluation_shard(
        "run-a",
        "zone",
        "revision",
        "sft",
        "iid",
        0,
        4,
        "4h",
        "small-seed53",
        125,
    )
    assert received["metadata"]["spider-training-job"] == "small-seed53"
    assert received["metadata"]["spider-training-step"] == 125


def test_evaluation_merge_rejects_non_positive_shards() -> None:
    try:
        MODULE.create_evaluation_merge(
            "run", "zone", "revision", "base", "iid", 0, "2h"
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid evaluation merge was accepted")


def test_monitor_retries_transient_inventory_failure_without_stopping(monkeypatch) -> None:
    calls = 0
    stop_calls: list[str] = []

    def inventory(_run_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, ["gcloud"])
        return [{"name": "worker", "status": "TERMINATED"}]

    monkeypatch.setattr(MODULE, "managed_instances", inventory)
    monkeypatch.setattr(MODULE, "stop_instances", stop_calls.append)
    monkeypatch.setattr(MODULE, "append_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    MODULE.monitor("run-a", poll_seconds=1, timeout_seconds=10)

    assert calls == 2
    assert stop_calls == []


def test_create_evaluation_uses_gpu_and_scoped_metadata(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    name = MODULE.create_evaluation_shard(
        "run-a", "us-west1-b", "abc123", "base", "domain_balanced", 1, 4, "4h"
    )

    assert name == "spider-exp005-eval-base-domain-01-run-a"
    assert received["gpu"] is True
    assert received["metadata"] == {
        "spider-control": "base",
        "spider-eval-suite": "domain_balanced",
        "spider-shard-index": 1,
        "spider-num-shards": 4,
    }
