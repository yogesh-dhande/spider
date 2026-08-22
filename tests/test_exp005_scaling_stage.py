import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_exp005_scaling_stage.py"
SPEC = importlib.util.spec_from_file_location("run_exp005_scaling_stage", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_classify_stage_is_fail_closed() -> None:
    empty = [None, None]
    assert MODULE.classify_stage(terminals=empty, failures=empty, instance_states=[]) == "missing"
    assert (
        MODULE.classify_stage(terminals=empty, failures=empty, instance_states=["RUNNING"])
        == "running"
    )
    assert (
        MODULE.classify_stage(terminals=[{}, None], failures=empty, instance_states=[])
        == "partial"
    )
    assert (
        MODULE.classify_stage(terminals=[{}, {}], failures=empty, instance_states=[])
        == "complete"
    )
    assert (
        MODULE.classify_stage(terminals=empty, failures=[{"exit_code": 1}, None], instance_states=[])
        == "failed"
    )
    assert (
        MODULE.classify_stage(terminals=empty, failures=empty, instance_states=["TERMINATED"])
        == "orphaned"
    )


def test_inspect_stage_requires_exact_rank_terminals(monkeypatch) -> None:
    def reader(uri: str):
        if not uri.endswith("complete.json"):
            return None
        rank = 0 if "rank_00" in uri else 1
        return {
            "run_id": "train-a",
            "job_id": "job-a",
            "start_step": 0,
            "stop_step": 500,
            "node_rank": rank,
            "num_nodes": 2,
            "status": "complete",
            "exit_code": 0,
        }

    monkeypatch.setattr(MODULE.cloud, "managed_instances", lambda run_id: [])
    assert (
        MODULE.inspect_stage(
            run_id="train-a",
            job_id="job-a",
            start_step=0,
            stop_step=500,
            num_nodes=2,
            storage_reader=reader,
        )
        == "complete"
    )
