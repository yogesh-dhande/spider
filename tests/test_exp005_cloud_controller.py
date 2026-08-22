import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "run_exp005_cloud_controller.py"
SPEC = importlib.util.spec_from_file_location("run_exp005_cloud_controller", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/ablations/experiment5_cloud_controller_v1.json").read_text()
    )


def test_controller_manifest_preserves_full_scaling_matrix() -> None:
    config = load_config()
    processes = MODULE.load_processes(config)
    assert len(processes) == 12
    assert len(config["jobs"]) == 9
    assert {item["job_id"].split("--")[1] for item in config["jobs"]} == {
        "small",
        "medium",
        "large",
    }
    assert [item.max_attempts for item in processes[:3]] == [24, 24, 24]
    assert all(item.max_attempts == 1 for item in processes[3:])


def test_large_jobs_require_explicit_release_after_medium() -> None:
    config = load_config()
    large_jobs = [item for item in config["jobs"] if "--large--" in item["job_id"]]
    assert len(large_jobs) == 3
    assert all(
        "experiments/exp005_browser_ablation_bed/artifacts/releases/large_v1.json"
        in item["prerequisites"]
        for item in large_jobs
    )


def test_seed_jobs_use_independent_training_pairs() -> None:
    config = load_config()
    for size in ("small", "medium"):
        jobs = [item for item in config["jobs"] if f"--{size}--" in item["job_id"]]
        assert len(jobs) == 3
        assert len({tuple(item["training_zones"]) for item in jobs}) == 3


def test_recovery_keeps_existing_scientific_revision_and_namespace() -> None:
    config = load_config()
    processes = MODULE.load_processes(config)
    seed59 = processes[1].command
    assert seed59[seed59.index("--evaluation-run-id") + 1] == "e-s59-4173bd-01-r2-v2"
    assert seed59[seed59.index("--repo-revision") + 1] == (
        "c1f41fb07bee936a76af2700d5ce8c7400b8f490"
    )
    assert seed59[seed59.index("--stop-step") + 1] == "500"


def test_job_command_keeps_seed53_receipt_override() -> None:
    config = load_config()
    processes = MODULE.load_processes(config)
    seed53 = next(item.command for item in processes if "small--seed-53" in item.name)
    assert "--adopt-through-step" in seed53
    assert seed53[seed53.index("--adopt-through-step") + 1] == "500"
    assert seed53[seed53.index("--receipt-override") + 1].endswith(
        "sft_small53_step0500_r2_0822a.json"
    )


def test_transient_state_upload_failure_is_nonfatal(monkeypatch, tmp_path) -> None:
    def fail(*_args, **_kwargs):
        raise MODULE.subprocess.CalledProcessError(1, ["gcloud"])

    monkeypatch.setattr(MODULE, "upload_state", fail)
    assert MODULE.upload_state_safely({}, tmp_path / "status.json") is False
