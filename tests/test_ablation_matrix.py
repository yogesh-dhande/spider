import hashlib
import json
from pathlib import Path

import pytest
import yaml

from spider.ablation_matrix import claim_next_job, plan_matrix, requeue_job, summarize_matrix


def _write_matrix(tmp_path: Path) -> Path:
    base = {
        "experiment": {
            "id": "test",
            "seed": 1,
            "data_dir": "data/browser",
            "output_dir": "outputs/test",
        },
        "data": {},
        "training": {"learning_rate": 0.0001},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    corpus = tmp_path / "corpus/manifests"
    corpus.mkdir(parents=True)
    manifests = {
        "small": corpus / "train_small.jsonl",
        "large": corpus / "train_large.jsonl",
    }
    for size, path in manifests.items():
        path.write_text(json.dumps({"id": size}) + "\n", encoding="utf-8")
    validation = corpus / "validation.jsonl"
    validation.write_text(json.dumps({"id": "validation"}) + "\n", encoding="utf-8")
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    ladder = {
        "tiers": {size: {"sha256": sha256(path)} for size, path in manifests.items()},
        "evaluation_suites": {"development": {"sha256": sha256(validation)}},
    }
    (tmp_path / "corpus/dataset_ladder.json").write_text(
        json.dumps(ladder), encoding="utf-8"
    )
    matrix = {
        "matrix": {
            "id": "test-matrix",
            "base_config": "base.yaml",
            "data_dir": "corpus",
            "dataset_ladder_manifest": "dataset_ladder.json",
            "validation_manifest": "manifests/validation.jsonl",
            "output_dir": "matrix-output",
            "datasets": {
                "small": {"train_manifest": "manifests/train_small.jsonl"},
                "large": {"train_manifest": "manifests/train_large.jsonl"},
            },
            "recipes": [
                {"id": "control", "overlay": {}},
                {"id": "lower-lr", "overlay": {"training": {"learning_rate": 0.00005}}},
            ],
            "budgets": {
                "pilot": {
                    "sizes": ["small", "large"],
                    "seeds": [3, 5],
                    "recipes": ["control", "lower-lr"],
                    "max_steps": 20,
                }
            },
        }
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    return path


def test_plan_matrix_materializes_isolated_immutable_jobs(tmp_path: Path) -> None:
    config = _write_matrix(tmp_path)
    root = plan_matrix(config, budget="pilot")
    plan = json.loads((root / "plan.json").read_text())
    assert plan["job_count"] == 8
    assert len({job["job_id"] for job in plan["jobs"]}) == 8
    assert len({job["output_dir"] for job in plan["jobs"]}) == 8
    for job in plan["jobs"]:
        resolved = yaml.safe_load((root / job["config_path"]).read_text())
        assert resolved["data"]["train_manifest"].startswith("manifests/train_")
        assert job["environment"]["SPIDER_OUTPUT_DIR"] == job["output_dir"]
        assert job["argv"][-2:] == ["--max-steps", "20"]
    assert plan_matrix(config, budget="pilot") == root

    source = yaml.safe_load(config.read_text())
    source["matrix"]["budgets"]["pilot"]["seeds"].append(7)
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable matrix plan"):
        plan_matrix(config, budget="pilot")


def test_claims_prevent_parallel_workers_from_taking_same_job(tmp_path: Path) -> None:
    root = plan_matrix(_write_matrix(tmp_path), budget="pilot")
    first = claim_next_job(root, worker_id="worker-a")
    second = claim_next_job(root, worker_id="worker-b")
    assert first is not None and second is not None
    assert first["job_id"] != second["job_id"]


def test_deterministic_shards_partition_jobs(tmp_path: Path) -> None:
    root = plan_matrix(_write_matrix(tmp_path), budget="pilot")
    first = claim_next_job(root, worker_id="worker-a", shard_index=0, num_shards=2)
    second = claim_next_job(root, worker_id="worker-b", shard_index=1, num_shards=2)
    plan = json.loads((root / "plan.json").read_text())
    indices = {job["job_id"]: index for index, job in enumerate(plan["jobs"])}
    assert indices[first["job_id"]] % 2 == 0
    assert indices[second["job_id"]] % 2 == 1


def test_summary_reports_pending_and_completed_jobs(tmp_path: Path) -> None:
    root = plan_matrix(_write_matrix(tmp_path), budget="pilot")
    job = claim_next_job(root, worker_id="worker")
    Path(root, job["job_root"], "result.json").write_text(
        json.dumps({"status": "complete", "exit_code": 0}), encoding="utf-8"
    )
    summary_path = summarize_matrix(root)
    summary = json.loads(summary_path.read_text())
    assert summary["status_counts"] == {"complete": 1, "pending": 7}


def test_requeue_preserves_failed_attempt_and_allows_new_claim(tmp_path: Path) -> None:
    root = plan_matrix(_write_matrix(tmp_path), budget="pilot")
    job = claim_next_job(root, worker_id="first-worker")
    job_root = root / job["job_root"]
    (job_root / "result.json").write_text(
        json.dumps({"status": "failed", "exit_code": 1}), encoding="utf-8"
    )
    (job_root / "runner.log").write_text("failure details", encoding="utf-8")

    attempt = requeue_job(root, job_id=job["job_id"], reason="worker preempted")
    assert (attempt / "claim/claim.json").exists()
    assert (attempt / "result.json").exists()
    assert (attempt / "runner.log").read_text() == "failure details"
    assert json.loads((attempt / "requeue.json").read_text())["reason"] == "worker preempted"
    assert claim_next_job(root, worker_id="second-worker")["job_id"] == job["job_id"]
