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
    assert MODULE.MONITOR_FAILURE_LIMIT == 20


def test_bootstrap_is_restart_safe() -> None:
    bootstrap = MODULE._bootstrap("abc123", "scripts/guest.sh")
    assert "if [[ ! -d /opt/spider/.git ]]" in bootstrap
    assert "git -C /opt/spider fetch -q origin" in bootstrap
    assert "git -C /opt/spider checkout -q abc123" in bootstrap
    assert "test ! -e /opt/spider" not in bootstrap


def test_materialization_shard_rejects_invalid_partition() -> None:
    try:
        MODULE.create_materialization_shard("run", "zone", "revision", 4, 4, "6h")
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("invalid materialization shard was accepted")


def test_materialization_shard_uses_large_standard_disk(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    MODULE.create_materialization_shard("run-a", "zone", "revision", 0, 8, "6h")
    assert received["machine_type"] == "n2-standard-8"
    assert received["boot_disk_size"] == "150GB"
    assert received["boot_disk_type"] == "pd-standard"


def test_qa_inventory_shard_uses_cpu_and_rejects_invalid_partition(monkeypatch) -> None:
    try:
        MODULE.create_qa_inventory_shard("run", "zone", "revision", 5, 5, "5h")
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("invalid QA inventory shard was accepted")

    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    MODULE.create_qa_inventory_shard("run-a", "zone", "revision", 2, 5, "5h")
    assert received["gpu"] is False
    assert received["machine_type"] == "n2-standard-8"
    assert received["metadata"] == {
        "spider-source-id": "screenshot_qa",
        "spider-shard-index": 2,
        "spider-num-shards": 5,
    }


def test_generic_source_inventory_preserves_source_identity(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    MODULE.create_source_inventory_shard(
        "run-a", "zone", "revision", "grounding_template", 1, 2, "5h"
    )
    assert received["metadata"] == {
        "spider-source-id": "grounding_template",
        "spider-shard-index": 1,
        "spider-num-shards": 2,
    }


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


def test_inventory_sync_rejects_invalid_layout(tmp_path) -> None:
    try:
        MODULE.sync_inventory_artifacts("run", "screenshot_qa", 5, tmp_path, "other")
    except ValueError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("unknown inventory layout was accepted")


def test_inventory_terminal_marker_accepts_generic_and_legacy_qa_records() -> None:
    common = {
        "run_id": "run-a",
        "shard_index": 1,
        "num_shards": 5,
        "status": "complete",
        "exit_code": 0,
    }
    MODULE.validate_inventory_terminal(
        {**common, "source_id": "screenshot_qa"},
        run_id="run-a",
        source_id="screenshot_qa",
        shard_index=1,
        num_shards=5,
    )
    MODULE.validate_inventory_terminal(
        common,
        run_id="run-a",
        source_id="screenshot_qa",
        shard_index=1,
        num_shards=5,
    )


def test_inventory_terminal_marker_rejects_failed_or_wrong_shard() -> None:
    marker = {
        "run_id": "run-a",
        "source_id": "grounding_gpt",
        "shard_index": 0,
        "num_shards": 1,
        "status": "failed",
        "exit_code": 1,
    }
    try:
        MODULE.validate_inventory_terminal(
            marker,
            run_id="run-a",
            source_id="grounding_gpt",
            shard_index=0,
            num_shards=1,
        )
    except ValueError as error:
        assert "status" in str(error)
        assert "exit_code" in str(error)
    else:
        raise AssertionError("failed inventory marker was accepted")

    marker.update({"status": "complete", "exit_code": 0, "shard_index": 1})
    try:
        MODULE.validate_inventory_terminal(
            marker,
            run_id="run-a",
            source_id="grounding_gpt",
            shard_index=0,
            num_shards=1,
        )
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("wrong inventory shard marker was accepted")


def test_training_stage_rejects_unsafe_job_id() -> None:
    try:
        MODULE.create_training_stage("run", "zone", "revision", "Unsafe Job", 0, 125, "4h")
    except ValueError as error:
        assert "job_id" in str(error)
    else:
        raise AssertionError("unsafe training job ID was accepted")


def test_training_stage_uses_available_single_l4_shape(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    name = MODULE.create_training_stage(
        "run-a", "us-west1-b", "abc123", "small-seed53", 0, 125, "4h"
    )

    assert name == "spider-exp005-train-00125-run-a"
    assert received["machine_type"] == "g2-standard-8"
    assert received["gpu"] is True
    assert received["metadata"] == {
        "spider-job-id": "small-seed53",
        "spider-stage-start": 0,
        "spider-stage-stop": 125,
        "spider-gpu-count": 1,
        "spider-gradient-accumulation": 16,
    }


def test_training_stage_uses_two_l4_shape_and_preserves_effective_batch(monkeypatch) -> None:
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    MODULE.create_training_stage(
        "run-a", "us-west1-b", "abc123", "small-seed53", 125, 250, "4h", 2
    )

    assert received["machine_type"] == "g2-standard-24"
    assert received["metadata"]["spider-gpu-count"] == 2
    assert received["metadata"]["spider-gradient-accumulation"] == 8


def test_training_stage_uses_four_l4_shape(monkeypatch) -> None:
    received = {}
    monkeypatch.setattr(MODULE, "_create", lambda **kwargs: received.update(kwargs) or kwargs["name"])

    MODULE.create_training_stage(
        "run-a", "us-west1-b", "abc123", "large-seed53", 0, 125, "4h", 4
    )

    assert received["machine_type"] == "g2-standard-48"
    assert received["metadata"]["spider-gradient-accumulation"] == 4


def test_training_stage_rejects_unsupported_gpu_count() -> None:
    try:
        MODULE.create_training_stage(
            "run-a", "zone", "revision", "small-seed53", 0, 125, "4h", 3
        )
    except ValueError as error:
        assert "gpu_count" in str(error)
    else:
        raise AssertionError("unsupported training GPU count was accepted")


def test_multinode_training_requires_supported_size_and_distinct_regions() -> None:
    try:
        MODULE.create_multinode_training_stage(
            "run-a", ["us-west1-a", "us-west1-b"], "revision", "small-seed53", 0, 125, "6h"
        )
    except ValueError as error:
        assert "distinct regions" in str(error)
    else:
        raise AssertionError("same-region multinode cluster was accepted")

    try:
        MODULE.create_multinode_training_stage(
            "run-a", ["us-west1-a", "us-west2-a", "us-west3-a"], "revision", "small-seed53", 0, 125, "6h"
        )
    except ValueError as error:
        assert "2, 4, 8, or 16" in str(error)
    else:
        raise AssertionError("unsupported multinode cluster size was accepted")


def test_multinode_training_preserves_effective_batch_and_leader_address(monkeypatch) -> None:
    created: list[dict] = []

    def create(**kwargs):
        created.append(kwargs)
        return kwargs["name"]

    monkeypatch.setattr(MODULE, "_create", create)
    monkeypatch.setattr(MODULE, "append_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "emit", lambda *args, **kwargs: None)
    names = MODULE.create_multinode_training_stage(
        "run-a",
        ["us-west1-b", "us-west2-a", "us-east1-d", "us-east4-c"],
        "abc123",
        "small-seed53",
        0,
        125,
        "6h",
    )

    assert len(names) == len(created) == 4
    assert all(row["machine_type"] == "g2-standard-8" for row in created)
    assert all(row["metadata"]["spider-gradient-accumulation"] == 4 for row in created)
    assert all(
        row["metadata"]["spider-master-address"]
        == "spider-exp005-train-mn-r00-00125-run-a.us-west1-b.c.keptune.internal"
        for row in created
    )


def test_multinode_guest_coordinates_and_only_rank_zero_uploads_adapter() -> None:
    guest = (MODULE_PATH.parent / "gcloud_exp005_train_multinode_guest.sh").read_text()

    assert '--nnodes="${NUM_NODES}"' in guest
    assert '--node_rank="${NODE_RANK}"' in guest
    assert '--master_addr="${MASTER_ADDRESS}"' in guest
    assert 'ready_count=' in guest
    assert 'if [[ "${NODE_RANK}" -eq 0 ]]; then' in guest
    assert 'assert state["world_size"] == world_size, state' in guest


def test_training_guest_uses_ddp_and_rejects_world_size_changes_between_stages() -> None:
    guest = (MODULE_PATH.parent / "gcloud_exp005_train_guest.sh").read_text()

    assert '--nproc_per_node="${GPU_COUNT}"' in guest
    assert 'assert state["world_size"] == gpu_count, state' in guest
    assert 'assert state["gradient_accumulation_steps"] == accumulation, state' in guest


def test_inventory_recovery_guest_verifies_completed_cache_before_upload() -> None:
    guest = (MODULE_PATH.parent / "gcloud_exp005_inventory_recovery_guest.sh").read_text()

    assert 'assert all(row.get("complete") for row in summaries)' in guest
    assert '"${DESTINATION}/inventory.tar.zst"' in guest
    assert "shutdown -h now" in guest


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


def test_monitor_waits_until_stopping_instance_is_terminated(monkeypatch) -> None:
    calls = 0

    def inventory(_run_id: str):
        nonlocal calls
        calls += 1
        status = "STOPPING" if calls == 1 else "TERMINATED"
        return [{"name": "worker", "status": status}]

    monkeypatch.setattr(MODULE, "managed_instances", inventory)
    monkeypatch.setattr(MODULE, "append_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    MODULE.monitor("run-a", poll_seconds=1, timeout_seconds=10)

    assert calls == 2


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
    assert received["boot_disk_type"] == "pd-standard"
    assert received["metadata"] == {
        "spider-control": "base",
        "spider-eval-suite": "domain_balanced",
        "spider-shard-index": 1,
        "spider-num-shards": 4,
    }


def test_evaluation_guest_publishes_shard_metrics_separately() -> None:
    guest = (MODULE_PATH.parent / "gcloud_exp005_eval_guest.sh").read_text()

    assert '"${RESULT_ROOT}/metrics.json" "${DESTINATION}/metrics.json"' in guest
    assert '"${RESULT_ROOT}/run_metadata.json" "${DESTINATION}/run_metadata.json"' in guest
