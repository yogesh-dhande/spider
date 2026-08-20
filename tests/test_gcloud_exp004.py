import importlib.util
import subprocess
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gcloud_exp004.py"
SPEC = importlib.util.spec_from_file_location("gcloud_exp004", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_managed_filter_is_narrowly_scoped() -> None:
    assert MODULE.MANAGED_FILTER == (
        "labels.spider-managed=true AND labels.spider-experiment=exp004"
    )


def test_zone_name_accepts_full_resource_url() -> None:
    assert MODULE.zone_name({"zone": "https://compute.googleapis.com/zones/us-west1-b"}) == (
        "us-west1-b"
    )


def test_training_stage_rejects_invalid_bounds() -> None:
    try:
        MODULE.create_training_stage("test", "us-west1-b", "abc", 375, 375, "6h")
    except ValueError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("invalid stage bounds were accepted")


def test_training_stage_preserves_effective_batch() -> None:
    try:
        MODULE.create_training_stage("test", "zone", "abc", 375, 500, "4h", 2, 4)
    except ValueError as error:
        assert "effective batch size 16" in str(error)
    else:
        raise AssertionError("invalid effective batch was accepted")


def test_validation_rejects_unknown_role() -> None:
    try:
        MODULE.create_validation_shard("test", "other", "zone", "abc", 500, "4h")
    except ValueError as error:
        assert "action or perception" in str(error)
    else:
        raise AssertionError("unknown validation role was accepted")


def test_speed_benchmark_preserves_effective_batch() -> None:
    try:
        MODULE.create_speed_benchmark("test", "zone", "abc", 250, 20, 2, 4, "2h")
    except ValueError as error:
        assert "effective batch size 16" in str(error)
    else:
        raise AssertionError("invalid effective batch was accepted")


def test_speed_benchmark_rejects_non_positive_parameters() -> None:
    try:
        MODULE.create_speed_benchmark("test", "zone", "abc", 250, 0, 2, 8, "2h")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive benchmark length was accepted")


def test_validation_rejects_unknown_accelerator() -> None:
    try:
        MODULE.create_validation_shard(
            "test", "perception", "zone", "abc", 500, "4h", "other"
        )
    except ValueError as error:
        assert "l4 or t4" in str(error)
    else:
        raise AssertionError("unknown validation accelerator was accepted")


def test_final_shard_rejects_invalid_partition() -> None:
    try:
        MODULE.create_final_shard("test", "zone", "abc", 500, 4, 4, "4h")
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("invalid final shard partition was accepted")


def test_final_shard_rejects_unknown_accelerator() -> None:
    try:
        MODULE.create_final_shard("test", "zone", "abc", 500, 0, 4, "4h", "other")
    except ValueError as error:
        assert "l4 or t4" in str(error)
    else:
        raise AssertionError("unknown final accelerator was accepted")


def test_closed_loop_rejects_non_positive_step() -> None:
    try:
        MODULE.create_closed_loop("test", "zone", "abc", 0, "4h")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive closed-loop checkpoint was accepted")


def test_final_merge_rejects_non_positive_shards() -> None:
    try:
        MODULE.create_final_merge("test", "zone", "abc", 500, 0, "2h")
    except ValueError as error:
        assert "num_shards" in str(error)
    else:
        raise AssertionError("non-positive final merge shard count was accepted")


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
