import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_exp005_scaling_job.py"
SPEC = importlib.util.spec_from_file_location("run_exp005_scaling_job", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_only_missing_stage_requires_empty_training_regions() -> None:
    assert MODULE.training_regions_must_be_empty("missing") is True
    for state in ("running", "complete", "partial", "failed", "orphaned"):
        assert MODULE.training_regions_must_be_empty(state) is False


def test_complete_stage_reuses_pair_lock_despite_evaluation_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    events = []
    monkeypatch.setattr(
        MODULE,
        "list_instances",
        lambda: [
            {
                "zone": "https://compute/zones/alpha-a",
                "status": "RUNNING",
                "machineType": "https://compute/machineTypes/g2-standard-8",
                "guestAccelerators": [{"acceleratorCount": 1}],
            }
        ],
    )
    monkeypatch.setattr(
        MODULE,
        "emit",
        lambda event, state_log, **fields: events.append((event, fields)),
    )

    with MODULE.training_pair_slot(
        ["alpha-a", "beta-b"],
        tmp_path / "locks",
        0,
        tmp_path / "state.jsonl",
        require_empty_regions=False,
    ):
        pass

    assert events == [
        (
            "scaling_job_training_pair_acquired",
            {
                "zones": ["alpha-a", "beta-b"],
                "regions": ["alpha", "beta"],
                "required_empty_regions": False,
            },
        )
    ]
