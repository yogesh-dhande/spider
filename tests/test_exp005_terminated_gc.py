import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "cleanup_exp005_terminated_instances.py"
SPEC = importlib.util.spec_from_file_location("cleanup_exp005_terminated_instances", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _instance(name: str, status: str, role: str) -> dict:
    return {
        "name": name,
        "zone": "https://example/zones/us-east1-b",
        "status": status,
        "labels": {
            "spider-managed": "true",
            "spider-experiment": "exp005",
            "spider-role": role,
        },
    }


def test_terminated_targets_excludes_running_workers_and_controller() -> None:
    targets = MODULE.terminated_targets(
        [
            _instance("done-worker", "TERMINATED", "evaluation"),
            _instance("live-worker", "RUNNING", "evaluation"),
            _instance("controller", "TERMINATED", "controller"),
        ]
    )
    assert targets == [("done-worker", "us-east1-b", "evaluation")]


def test_terminated_targets_fails_closed_on_unscoped_instance() -> None:
    instance = _instance("unsafe", "TERMINATED", "evaluation")
    instance["labels"]["spider-experiment"] = "other"
    with pytest.raises(ValueError, match="unsafe target"):
        MODULE.terminated_targets([instance])

