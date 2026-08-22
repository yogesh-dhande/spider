from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from spider.eval_receipt import SUITES

EXPECTED_TASKS = {
    "iid": {"qa", "grounding", "action"},
    "domain_balanced": {"qa", "grounding", "action"},
    "distribution_shift": {"grounding", "action"},
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _issue(issues: list[dict[str, str]], job_id: str, message: str) -> None:
    issues.append({"job_id": job_id, "message": message})


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def _validate_evaluation(
    receipt: dict[str, Any],
    *,
    job_id: str,
    training_adapter_sha256: str | None,
    expected_num_shards: int,
    issues: list[dict[str, str]],
) -> None:
    if receipt.get("kind") != "evaluation_receipt":
        _issue(issues, job_id, "final evaluation has the wrong receipt kind")
    if receipt.get("control") != "sft":
        _issue(issues, job_id, "final evaluation is not an SFT candidate")
    adapter_hash = receipt.get("adapter_sha256")
    if not _valid_hash(adapter_hash):
        _issue(issues, job_id, "final evaluation lacks a valid adapter SHA-256")
    elif training_adapter_sha256 and adapter_hash != training_adapter_sha256:
        _issue(issues, job_id, "training and evaluation adapter hashes differ")
    if receipt.get("num_shards") != expected_num_shards:
        _issue(issues, job_id, f"final evaluation does not declare {expected_num_shards} shards")

    suites = receipt.get("suites")
    if not isinstance(suites, dict):
        _issue(issues, job_id, "final evaluation lacks suite results")
        return
    if set(suites) != set(SUITES):
        _issue(issues, job_id, "final evaluation suite set is incomplete or unexpected")
    for suite_name in SUITES:
        suite = suites.get(suite_name)
        if not isinstance(suite, dict):
            continue
        for field in ("metrics_sha256", "run_metadata_sha256"):
            if not _valid_hash(suite.get(field)):
                _issue(issues, job_id, f"{suite_name} lacks a valid {field}")
        shards = suite.get("shards")
        if not isinstance(shards, list) or len(shards) != expected_num_shards:
            _issue(issues, job_id, f"{suite_name} does not contain {expected_num_shards} shards")
        else:
            indices = [item.get("shard_index") for item in shards if isinstance(item, dict)]
            if sorted(indices) != list(range(expected_num_shards)):
                _issue(issues, job_id, f"{suite_name} shard indices are incomplete or duplicated")
            for shard in shards:
                if not isinstance(shard, dict):
                    _issue(issues, job_id, f"{suite_name} contains a malformed shard")
                    continue
                for field in ("signature", "metrics_sha256", "run_metadata_sha256"):
                    if not _valid_hash(shard.get(field)):
                        _issue(
                            issues,
                            job_id,
                            f"{suite_name} shard {shard.get('shard_index')} lacks a valid {field}",
                        )
        tasks = suite.get("merged", {}).get("tasks", {})
        if not isinstance(tasks, dict) or not EXPECTED_TASKS[suite_name].issubset(tasks):
            _issue(issues, job_id, f"{suite_name} merged metrics lack required tasks")


def audit_scaling_completion(
    schedule: dict[str, Any],
    artifact_root: Path,
    expected_groups: dict[str, dict[str, Any]],
    *,
    expected_num_shards: int = 4,
) -> dict[str, Any]:
    jobs = {str(item["job_id"]): item for item in schedule["jobs"]}
    selected: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    seen_adapters: dict[str, str] = {}

    for group_name, group in expected_groups.items():
        size = str(group["size"])
        step = int(group["step"])
        seeds = {int(seed) for seed in group["seeds"]}
        matching = [
            job
            for job in jobs.values()
            if job.get("dataset_size") == size
            and int(job.get("total_optimizer_steps", -1)) == step
            and int(job.get("seed", -1)) in seeds
        ]
        observed_seeds = {int(job["seed"]) for job in matching}
        if observed_seeds != seeds:
            issues.append(
                {
                    "job_id": group_name,
                    "message": (
                        f"schedule seed set {sorted(observed_seeds)} does not match "
                        f"expected {sorted(seeds)}"
                    ),
                }
            )
        for job in sorted(matching, key=lambda item: int(item["seed"])):
            job_id = str(job["job_id"])
            job_root = artifact_root / job_id
            job_receipt_path = job_root / "job_result.json"
            training_path = job_root / f"training_step_{step:05d}.json"
            evaluation_path = job_root / f"evaluation_step_{step:05d}.json"
            row = {
                "group": group_name,
                "job_id": job_id,
                "size": size,
                "seed": int(job["seed"]),
                "step": step,
                "job_receipt": str(job_receipt_path),
                "training_receipt": str(training_path),
                "evaluation_receipt": str(evaluation_path),
                "status": "incomplete",
            }
            selected.append(row)

            missing = [
                str(path)
                for path in (job_receipt_path, training_path, evaluation_path)
                if not path.is_file()
            ]
            if missing:
                _issue(issues, job_id, "missing required receipt(s): " + ", ".join(missing))
                continue
            job_receipt = _load(job_receipt_path)
            training = _load(training_path)
            evaluation = _load(evaluation_path)
            if job_receipt.get("kind") != "exp005_scaling_job_receipt":
                _issue(issues, job_id, "job result has the wrong receipt kind")
            if job_receipt.get("job_id") != job_id:
                _issue(issues, job_id, "job result identity does not match the schedule")
            if job_receipt.get("status") != "complete_pass":
                _issue(issues, job_id, "job result is not complete_pass")
            if job_receipt.get("final_step") != step:
                _issue(issues, job_id, "job result final step does not match the schedule")
            if training.get("kind") != "exp005_training_stage_receipt":
                _issue(issues, job_id, "final training has the wrong receipt kind")
            if training.get("job_id") != job_id or training.get("completed_step") != step:
                _issue(issues, job_id, "final training identity or step does not match")
            if training.get("status") != "complete_pass":
                _issue(issues, job_id, "final training is not complete_pass")
            adapter_hash = training.get("adapter", {}).get("sha256")
            if not _valid_hash(adapter_hash):
                _issue(issues, job_id, "final training lacks a valid adapter SHA-256")
                adapter_hash = None
            _validate_evaluation(
                evaluation,
                job_id=job_id,
                training_adapter_sha256=adapter_hash,
                expected_num_shards=expected_num_shards,
                issues=issues,
            )
            if adapter_hash:
                previous = seen_adapters.get(adapter_hash)
                if previous:
                    _issue(issues, job_id, f"adapter hash is reused from {previous}")
                else:
                    seen_adapters[adapter_hash] = job_id
                row["adapter_sha256"] = adapter_hash
            row["evaluation_run_id"] = evaluation.get("run_id")
            row["status"] = "complete"

    issue_jobs = {item["job_id"] for item in issues}
    for row in selected:
        if row["job_id"] in issue_jobs:
            row["status"] = "invalid" if row["status"] == "complete" else row["status"]
    return {
        "schema_version": 1,
        "kind": "exp005_scaling_completion_audit",
        "status": "pass" if not issues else "incomplete_or_invalid",
        "expected_groups": expected_groups,
        "expected_num_shards": expected_num_shards,
        "candidates": selected,
        "issues": issues,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# EXP005 scaling completion audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "| Group | Seed | Step | Status | Adapter SHA-256 |",
        "|---|---:|---:|---|---|",
    ]
    for row in audit["candidates"]:
        lines.append(
            f"| {row['group']} | {row['seed']} | {row['step']} | {row['status']} | "
            f"`{row.get('adapter_sha256', '—')}` |"
        )
    if audit["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(
            f"- `{item['job_id']}`: {item['message']}" for item in audit["issues"]
        )
    lines.append("")
    return "\n".join(lines)

