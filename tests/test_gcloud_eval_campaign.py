import importlib.util
import subprocess
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_exp005_evaluation_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_exp005_evaluation_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_label_is_stable() -> None:
    identity = MODULE.ShardIdentity("domain_balanced", 2)
    assert MODULE.label("sft", identity, 4) == "sft-domain_balanced-shard-02-of-04"


def test_metadata_dict_and_active_shards() -> None:
    instance = {
        "status": "RUNNING",
        "metadata": {
            "items": [
                {"key": "spider-eval-suite", "value": "iid"},
                {"key": "spider-shard-index", "value": "3"},
            ]
        },
    }
    assert MODULE.metadata_dict(instance) == {
        "spider-eval-suite": "iid",
        "spider-shard-index": "3",
    }
    assert MODULE.active_shards([instance]) == {MODULE.ShardIdentity("iid", 3)}


def test_active_gpu_regions_ignores_cpu_and_terminal_instances() -> None:
    instances = [
        {
            "status": "RUNNING",
            "zone": "zones/us-east1-b",
            "machineType": "machineTypes/g2-standard-8",
        },
        {
            "status": "RUNNING",
            "zone": "zones/us-west1-a",
            "machineType": "machineTypes/e2-standard-4",
        },
        {
            "status": "TERMINATED",
            "zone": "zones/europe-west2-a",
            "machineType": "machineTypes/g2-standard-8",
        },
    ]
    assert MODULE.active_gpu_regions(instances) == {"us-east1"}


def test_prioritize_zones_prefers_prior_gpu_capacity_stably() -> None:
    zones = ["asia-east1-c", "us-west1-a", "europe-west4-a"]
    instances = [
        {
            "status": "TERMINATED",
            "zone": "zones/us-west1-a",
            "machineType": "machineTypes/g2-standard-8",
        },
        {
            "status": "TERMINATED",
            "zone": "zones/europe-west4-a",
            "machineType": "machineTypes/g2-standard-8",
        },
        {
            "status": "TERMINATED",
            "zone": "zones/us-west1-a",
            "machineType": "machineTypes/g2-standard-8",
        },
    ]
    assert MODULE.prioritize_zones(zones, instances) == [
        "us-west1-a",
        "europe-west4-a",
        "asia-east1-c",
    ]


def test_validate_shard_terminal_rejects_wrong_identity() -> None:
    identity = MODULE.ShardIdentity("iid", 0)
    terminal = {
        "run_id": "run-a",
        "control": "exp002",
        "suite": "iid",
        "shard_index": 1,
        "num_shards": 4,
        "status": "complete",
        "exit_code": 0,
    }
    try:
        MODULE.validate_shard_terminal(
            terminal,
            run_id="run-a",
            control="exp002",
            identity=identity,
            num_shards=4,
        )
    except ValueError as error:
        assert "shard_index" in str(error)
    else:
        raise AssertionError("wrong shard terminal was accepted")


def test_launch_available_shards_falls_through_stockout_immediately() -> None:
    identity = MODULE.ShardIdentity("distribution_shift", 0)
    attempts = []
    events = []
    retry_after = {}

    def launch(shard, zone):
        attempts.append((shard, zone))
        if zone == "asia-east1-c":
            raise subprocess.CalledProcessError(1, ["gcloud", "compute"])

    def record(event, **fields):
        events.append((event, fields))

    launched = MODULE.launch_available_shards(
        missing=[identity],
        candidates=["asia-east1-c", "europe-west1-c"],
        slots=1,
        retry_after=retry_after,
        retry_seconds=600,
        now=100.0,
        launch=launch,
        record=record,
    )

    assert attempts == [
        (identity, "asia-east1-c"),
        (identity, "europe-west1-c"),
    ]
    assert launched == [(identity, "europe-west1-c")]
    assert retry_after == {"asia-east1-c": 700.0}
    assert [event for event, _ in events] == [
        "evaluation_campaign_launch_retry",
        "evaluation_campaign_shard_launched",
    ]


def test_launch_available_shards_yields_after_attempt_budget() -> None:
    identities = [
        MODULE.ShardIdentity("domain_balanced", 0),
        MODULE.ShardIdentity("domain_balanced", 1),
    ]
    attempts = []
    events = []

    def launch(shard, zone):
        attempts.append((shard, zone))
        raise subprocess.CalledProcessError(1, ["gcloud", "compute"])

    def record(event, **fields):
        events.append((event, fields))

    launched = MODULE.launch_available_shards(
        missing=identities,
        candidates=["zone-a", "zone-b", "zone-c"],
        slots=2,
        retry_after={},
        retry_seconds=600,
        now=100.0,
        launch=launch,
        record=record,
        max_attempts=2,
    )

    assert launched == []
    assert [zone for _, zone in attempts] == ["zone-a", "zone-b"]
    assert events[-1] == (
        "evaluation_campaign_launch_budget_exhausted",
        {"attempts": 2, "remaining_shards": 2},
    )


def test_launch_available_shards_uses_at_most_one_zone_per_region() -> None:
    identities = [
        MODULE.ShardIdentity("iid", 0),
        MODULE.ShardIdentity("iid", 1),
    ]
    attempts = []

    def launch(shard, zone):
        attempts.append((shard, zone))

    launched = MODULE.launch_available_shards(
        missing=identities,
        candidates=["us-east1-b", "us-east1-c", "europe-west4-a"],
        slots=2,
        retry_after={},
        retry_seconds=600,
        now=100.0,
        launch=launch,
        record=lambda *args, **kwargs: None,
    )

    assert [zone for _, zone in attempts] == ["us-east1-b", "europe-west4-a"]
    assert launched == attempts


def test_complete_shards_checks_only_requested_identities(monkeypatch) -> None:
    requested = MODULE.ShardIdentity("iid", 2)
    seen = []

    def storage_json(uri):
        seen.append(uri)
        if uri.endswith("failed.json"):
            return None
        return {
            "run_id": "run-a",
            "control": "exp002",
            "suite": "iid",
            "shard_index": 2,
            "num_shards": 4,
            "status": "complete",
            "exit_code": 0,
        }

    monkeypatch.setattr(MODULE, "storage_json", storage_json)
    completed = MODULE.complete_shards(
        run_id="run-a",
        control="exp002",
        num_shards=4,
        identities={requested},
    )

    assert completed == {requested}
    assert len(seen) == 2
    assert all("exp002-iid-shard-02-of-04" in uri for uri in seen)


def test_complete_shards_skips_missing_object_reads_with_index(monkeypatch) -> None:
    requested = MODULE.ShardIdentity("iid", 2)

    def unexpected_read(uri):
        raise AssertionError(f"unexpected object read: {uri}")

    monkeypatch.setattr(MODULE, "storage_json", unexpected_read)
    completed = MODULE.complete_shards(
        run_id="run-a",
        control="exp002",
        num_shards=4,
        identities={requested},
        objects=set(),
    )
    assert completed == set()


def test_terminal_grace_delays_relaunch_for_known_stopped_shard() -> None:
    identity = MODULE.ShardIdentity("domain_balanced", 3)
    missing_since = {}

    launchable, deferred = MODULE.terminal_grace_filter(
        missing={identity},
        known={identity},
        missing_since=missing_since,
        now=100.0,
        grace_seconds=180,
    )
    assert launchable == []
    assert deferred == [identity]
    assert missing_since == {identity: 100.0}

    launchable, deferred = MODULE.terminal_grace_filter(
        missing={identity},
        known={identity},
        missing_since=missing_since,
        now=281.0,
        grace_seconds=180,
    )
    assert launchable == [identity]
    assert deferred == []


def test_terminal_grace_does_not_delay_never_created_shard() -> None:
    identity = MODULE.ShardIdentity("distribution_shift", 0)
    launchable, deferred = MODULE.terminal_grace_filter(
        missing={identity},
        known=set(),
        missing_since={},
        now=100.0,
        grace_seconds=180,
    )
    assert launchable == [identity]
    assert deferred == []
