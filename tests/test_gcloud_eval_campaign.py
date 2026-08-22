import importlib.util
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
