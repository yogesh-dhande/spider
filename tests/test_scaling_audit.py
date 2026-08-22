from pathlib import Path

from spider.scaling_audit import audit_scaling_completion, render_markdown


def _hash(character: str) -> str:
    return character * 64


def _write_receipts(root: Path, job_id: str, step: int, adapter_hash: str) -> None:
    import json

    job_root = root / job_id
    job_root.mkdir(parents=True)
    suites = {}
    for suite_name in ("iid", "domain_balanced", "distribution_shift"):
        tasks = {"grounding": {}, "action": {}}
        if suite_name != "distribution_shift":
            tasks["qa"] = {}
        suites[suite_name] = {
            "metrics_sha256": _hash("a"),
            "run_metadata_sha256": _hash("b"),
            "merged": {"tasks": tasks},
            "shards": [
                {
                    "shard_index": index,
                    "signature": _hash("c"),
                    "metrics_sha256": _hash("d"),
                    "run_metadata_sha256": _hash("e"),
                }
                for index in range(4)
            ],
        }
    payloads = {
        "job_result.json": {
            "kind": "exp005_scaling_job_receipt",
            "job_id": job_id,
            "status": "complete_pass",
            "final_step": step,
        },
        f"training_step_{step:05d}.json": {
            "kind": "exp005_training_stage_receipt",
            "job_id": job_id,
            "completed_step": step,
            "status": "complete_pass",
            "adapter": {"sha256": adapter_hash},
        },
        f"evaluation_step_{step:05d}.json": {
            "kind": "evaluation_receipt",
            "control": "sft",
            "run_id": f"eval-{job_id}",
            "adapter_sha256": adapter_hash,
            "num_shards": 4,
            "suites": suites,
        },
    }
    for name, payload in payloads.items():
        (job_root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_completion_audit_requires_all_seed_receipts_and_unique_adapters(tmp_path: Path) -> None:
    jobs = []
    for index, seed in enumerate((53, 59, 61), start=1):
        job_id = f"small-{seed}"
        jobs.append(
            {
                "job_id": job_id,
                "dataset_size": "small",
                "seed": seed,
                "total_optimizer_steps": 625,
            }
        )
        _write_receipts(tmp_path, job_id, 625, _hash(str(index)))

    audit = audit_scaling_completion(
        {"jobs": jobs},
        tmp_path,
        {"small@625": {"size": "small", "step": 625, "seeds": [53, 59, 61]}},
    )

    assert audit["status"] == "pass"
    assert [row["status"] for row in audit["candidates"]] == ["complete"] * 3
    assert "Status: **pass**" in render_markdown(audit)


def test_completion_audit_reports_missing_and_reused_adapter(tmp_path: Path) -> None:
    jobs = [
        {
            "job_id": f"small-{seed}",
            "dataset_size": "small",
            "seed": seed,
            "total_optimizer_steps": 625,
        }
        for seed in (53, 59, 61)
    ]
    _write_receipts(tmp_path, "small-53", 625, _hash("a"))
    _write_receipts(tmp_path, "small-59", 625, _hash("a"))

    audit = audit_scaling_completion(
        {"jobs": jobs},
        tmp_path,
        {"small@625": {"size": "small", "step": 625, "seeds": [53, 59, 61]}},
    )

    assert audit["status"] == "incomplete_or_invalid"
    messages = [item["message"] for item in audit["issues"]]
    assert any("reused" in message for message in messages)
    assert any("missing required" in message for message in messages)

