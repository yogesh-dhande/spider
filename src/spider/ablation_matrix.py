"""Plan and execute isolated recipe × data-size × seed training ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from spider.config import load_config
from spider.rl.study import deep_merge


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _plan_path(plan_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else plan_root / path


def _slug(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in cleaned.split("-") if part)


def _validate_matrix(matrix: dict[str, Any], budget: str) -> dict[str, Any]:
    datasets = matrix.get("datasets")
    recipes = matrix.get("recipes")
    budgets = matrix.get("budgets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("matrix.datasets must be a non-empty mapping")
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("matrix.recipes must be a non-empty list")
    if not isinstance(budgets, dict) or budget not in budgets:
        raise ValueError(f"Unknown matrix budget: {budget}")
    selected = budgets[budget]
    if not isinstance(selected, dict):
        raise TypeError(f"Budget {budget} must be a mapping")
    sizes = list(selected.get("sizes") or [])
    seeds = [int(seed) for seed in selected.get("seeds") or []]
    recipe_ids = list(selected.get("recipes") or [row.get("id") for row in recipes])
    if not sizes or not seeds or not recipe_ids:
        raise ValueError("Every budget requires sizes, seeds, and recipes")
    missing_sizes = sorted(set(sizes) - set(datasets))
    available_recipes = {str(row.get("id")) for row in recipes if isinstance(row, dict)}
    missing_recipes = sorted(set(recipe_ids) - available_recipes)
    if missing_sizes or missing_recipes:
        raise ValueError(
            f"Budget references missing sizes={missing_sizes} or recipes={missing_recipes}"
        )
    return {"sizes": sizes, "seeds": seeds, "recipes": recipe_ids, **selected}


def plan_matrix(
    config_path: str | Path,
    *,
    budget: str,
    output_dir: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    source = load_config(config_path)
    matrix = source.get("matrix")
    if not isinstance(matrix, dict):
        raise TypeError("Config requires a matrix mapping")
    selection = _validate_matrix(matrix, budget)
    base_config_path = _resolve(config_path.parent, str(matrix["base_config"]))
    base_config = load_config(base_config_path)
    data_dir = _resolve(config_path.parent, str(matrix["data_dir"]))
    validation_manifest = str(matrix.get("validation_manifest", "manifests/validation.jsonl"))
    ladder_path = data_dir / str(matrix.get("dataset_ladder_manifest", "dataset_ladder.json"))
    validation_path = data_dir / validation_manifest
    if not ladder_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Ablation planning requires the frozen dataset ladder and validation manifest: "
            f"{ladder_path}, {validation_path}"
        )
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    ladder_sha256 = _sha256_file(ladder_path)
    validation_sha256 = _sha256_file(validation_path)
    recorded_evaluation_hashes = {
        str(row.get("sha256"))
        for row in (ladder.get("evaluation_suites") or {}).values()
        if isinstance(row, dict)
    }
    if validation_sha256 not in recorded_evaluation_hashes:
        raise ValueError("Validation manifest is not registered by the frozen dataset ladder")
    root = Path(output_dir or str(matrix.get("output_dir", "outputs/ablation-matrices")))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    plan_root = root / str(matrix["id"]) / budget

    recipes = {str(row["id"]): row for row in matrix["recipes"]}
    size_order = {size: index for index, size in enumerate(selection["sizes"])}
    jobs: list[dict[str, Any]] = []
    for recipe_id in selection["recipes"]:
        recipe = recipes[recipe_id]
        overlay = recipe.get("overlay", {})
        if not isinstance(overlay, dict):
            raise TypeError(f"Recipe {recipe_id} overlay must be a mapping")
        for size in selection["sizes"]:
            dataset = matrix["datasets"][size]
            if not isinstance(dataset, dict) or not dataset.get("train_manifest"):
                raise ValueError(f"Dataset size {size} requires train_manifest")
            train_path = data_dir / str(dataset["train_manifest"])
            if not train_path.is_file():
                raise FileNotFoundError(f"Missing frozen training manifest: {train_path}")
            train_sha256 = _sha256_file(train_path)
            tier_record = (ladder.get("tiers") or {}).get(size)
            if not isinstance(tier_record, dict) or tier_record.get("sha256") != train_sha256:
                raise ValueError(f"Training manifest for {size} does not match dataset ladder")
            for seed in selection["seeds"]:
                resolved = deep_merge(base_config, overlay)
                resolved.setdefault("data", {})["train_manifest"] = str(
                    dataset["train_manifest"]
                )
                resolved["data"]["validation_manifest"] = validation_manifest
                resolved["experiment"]["seed"] = int(seed)
                identity_payload = {
                    "matrix_id": matrix["id"],
                    "budget": budget,
                    "recipe": recipe_id,
                    "size": size,
                    "seed": seed,
                    "base_config_sha256": hashlib.sha256(base_config_path.read_bytes()).hexdigest(),
                    "dataset_ladder_sha256": ladder_sha256,
                    "train_manifest_sha256": train_sha256,
                    "validation_manifest_sha256": validation_sha256,
                    "resolved_config": resolved,
                }
                identity = _canonical_hash(identity_payload)
                resolved_config_sha256 = _canonical_hash(resolved)
                job_id = (
                    f"{_slug(recipe_id)}--{_slug(size)}--seed-{seed}--{identity[:10]}"
                )
                job_root = Path("jobs") / job_id
                config_file = job_root / "config.yaml"
                argv = ["spider-train", "--config", str(config_file), "--resume", "auto"]
                max_steps = selection.get("max_steps")
                if max_steps is not None:
                    argv.extend(["--max-steps", str(int(max_steps))])
                jobs.append(
                    {
                        "schema_version": 1,
                        "job_id": job_id,
                        "identity_sha256": identity,
                        "resolved_config_sha256": resolved_config_sha256,
                        "matrix_id": str(matrix["id"]),
                        "budget": budget,
                        "recipe": recipe_id,
                        "dataset_size": size,
                        "seed": seed,
                        "dataset_ladder_sha256": ladder_sha256,
                        "train_manifest_sha256": train_sha256,
                        "validation_manifest_sha256": validation_sha256,
                        "job_root": str(job_root),
                        "config_path": str(config_file),
                        "output_dir": str(job_root / "output"),
                        "environment": {
                            "SPIDER_DATA_DIR": str(data_dir),
                            "SPIDER_OUTPUT_DIR": str(job_root / "output"),
                        },
                        "argv": argv,
                        "resolved_config": resolved,
                    }
                )

    jobs.sort(
        key=lambda row: (row["recipe"], size_order[str(row["dataset_size"])], row["seed"])
    )
    plan_payload = {
        "schema_version": 1,
        "matrix_id": str(matrix["id"]),
        "budget": budget,
        "source_config": str(config_path),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_config": str(base_config_path),
        "base_config_sha256": hashlib.sha256(base_config_path.read_bytes()).hexdigest(),
        "dataset_ladder": str(ladder_path),
        "dataset_ladder_sha256": ladder_sha256,
        "validation_manifest_sha256": validation_sha256,
        "job_count": len(jobs),
        "jobs": [{key: value for key, value in row.items() if key != "resolved_config"} for row in jobs],
    }
    stable_plan = {
        "schema_version": 1,
        "matrix_id": str(matrix["id"]),
        "budget": budget,
        "source_config_sha256": plan_payload["source_config_sha256"],
        "base_config_sha256": plan_payload["base_config_sha256"],
        "jobs": [
            {
                key: row[key]
                for key in (
                    "job_id",
                    "identity_sha256",
                    "resolved_config_sha256",
                    "recipe",
                    "dataset_size",
                    "seed",
                    "dataset_ladder_sha256",
                    "train_manifest_sha256",
                    "validation_manifest_sha256",
                )
            }
            for row in jobs
        ],
    }
    plan_hash = _canonical_hash(stable_plan)
    plan_payload["plan_sha256"] = plan_hash
    existing_path = plan_root / "plan.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan_hash:
            raise ValueError(f"Immutable matrix plan already exists with different content: {plan_root}")
        for job in existing.get("jobs") or []:
            config_file = _plan_path(plan_root, str(job["config_path"]))
            job_file = _plan_path(plan_root, str(job["job_root"])) / "job.json"
            if not config_file.is_file() or not job_file.is_file():
                raise ValueError(f"Immutable matrix job artifacts are missing: {job['job_id']}")
            resolved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if _canonical_hash(resolved) != job.get("resolved_config_sha256"):
                raise ValueError(f"Immutable matrix config is corrupted: {config_file}")
            registered = json.loads(job_file.read_text(encoding="utf-8"))
            if registered.get("identity_sha256") != job.get("identity_sha256"):
                raise ValueError(f"Immutable matrix job record is corrupted: {job_file}")
        return plan_root

    for row in jobs:
        job_root = _plan_path(plan_root, str(row["job_root"]))
        job_root.mkdir(parents=True, exist_ok=True)
        _plan_path(plan_root, str(row["config_path"])).write_text(
            yaml.safe_dump(row["resolved_config"], sort_keys=False), encoding="utf-8"
        )
        _atomic_json(
            job_root / "job.json",
            {key: value for key, value in row.items() if key != "resolved_config"},
        )
    _atomic_json(existing_path, plan_payload)
    return plan_root


def _plan_jobs(plan_root: Path) -> list[dict[str, Any]]:
    plan = json.loads((plan_root / "plan.json").read_text(encoding="utf-8"))
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("plan.json jobs must be a list")
    return jobs


def claim_next_job(
    plan_root: str | Path,
    *,
    worker_id: str,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any] | None:
    plan_root = Path(plan_root).resolve()
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    for index, job in enumerate(_plan_jobs(plan_root)):
        if index % num_shards != shard_index:
            continue
        job_root = _plan_path(plan_root, str(job["job_root"]))
        result_path = job_root / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                continue
        claim_dir = job_root / "claim"
        try:
            claim_dir.mkdir()
        except FileExistsError:
            continue
        _atomic_json(
            claim_dir / "claim.json",
            {
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "shard_index": shard_index,
                "num_shards": num_shards,
            },
        )
        return job
    return None


def run_worker(
    plan_root: str | Path,
    *,
    worker_id: str,
    shard_index: int = 0,
    num_shards: int = 1,
    max_jobs: int | None = None,
    data_dir_override: str | Path | None = None,
) -> list[dict[str, Any]]:
    plan_root = Path(plan_root).resolve()
    completed: list[dict[str, Any]] = []
    while max_jobs is None or len(completed) < max_jobs:
        job = claim_next_job(
            plan_root,
            worker_id=worker_id,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        if job is None:
            break
        job_root = _plan_path(plan_root, str(job["job_root"]))
        started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        log_path = job_root / "runner.log"
        with log_path.open("ab") as log:
            environment = dict(os.environ)
            environment.update(
                {str(key): str(value) for key, value in (job.get("environment") or {}).items()}
            )
            if data_dir_override is not None:
                environment["SPIDER_DATA_DIR"] = str(Path(data_dir_override).resolve())
            process = subprocess.run(
                [str(value) for value in job["argv"]],
                cwd=plan_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = {
            "schema_version": 1,
            "job_id": job["job_id"],
            "identity_sha256": job["identity_sha256"],
            "status": "complete" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "worker_id": worker_id,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "log_path": str(log_path),
        }
        _atomic_json(job_root / "result.json", result)
        completed.append(result)
        if process.returncode != 0:
            break
    return completed


def requeue_job(plan_root: str | Path, *, job_id: str, reason: str) -> Path:
    """Preserve a failed/abandoned attempt and make its job claimable again."""
    if not reason.strip():
        raise ValueError("A non-empty requeue reason is required")
    plan_root = Path(plan_root).resolve()
    jobs = {str(job["job_id"]): job for job in _plan_jobs(plan_root)}
    if job_id not in jobs:
        raise ValueError(f"Unknown job ID: {job_id}")
    job_root = _plan_path(plan_root, str(jobs[job_id]["job_root"]))
    claim_dir = job_root / "claim"
    result_path = job_root / "result.json"
    if not claim_dir.exists():
        raise ValueError(f"Job {job_id} has no claim to requeue")
    result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    )
    if result is not None and result.get("status") == "complete":
        raise ValueError(f"Completed job {job_id} cannot be requeued")
    attempts_root = job_root / "attempts"
    attempt_number = len(list(attempts_root.glob("attempt-*"))) + 1
    attempt_root = attempts_root / f"attempt-{attempt_number:03d}"
    attempt_root.mkdir(parents=True)
    claim_dir.replace(attempt_root / "claim")
    if result_path.exists():
        result_path.replace(attempt_root / "result.json")
    log_path = job_root / "runner.log"
    if log_path.exists():
        log_path.replace(attempt_root / "runner.log")
    _atomic_json(
        attempt_root / "requeue.json",
        {
            "job_id": job_id,
            "reason": reason,
            "requeued_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
    )
    return attempt_root


def summarize_matrix(plan_root: str | Path) -> Path:
    plan_root = Path(plan_root).resolve()
    rows: list[dict[str, Any]] = []
    for job in _plan_jobs(plan_root):
        job_root = _plan_path(plan_root, str(job["job_root"]))
        result_path = job_root / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {"status": "pending"}
        )
        training_state_path = _plan_path(plan_root, str(job["output_dir"])) / "training_state.json"
        training_state = (
            json.loads(training_state_path.read_text(encoding="utf-8"))
            if training_state_path.exists()
            else None
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "recipe": job["recipe"],
                "dataset_size": job["dataset_size"],
                "seed": job["seed"],
                "status": result.get("status", "pending"),
                "exit_code": result.get("exit_code"),
                "training_state": training_state,
            }
        )
    statuses = Counter(str(row["status"]) for row in rows)
    summary = {
        "schema_version": 1,
        "plan_sha256": json.loads((plan_root / "plan.json").read_text())["plan_sha256"],
        "jobs": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "rows": rows,
    }
    output = plan_root / "summary.json"
    _atomic_json(output, summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run a browser-training ablation matrix")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--budget", required=True)
    plan_parser.add_argument("--output-dir", type=Path)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--plan-root", type=Path, required=True)
    worker_parser.add_argument("--worker-id", required=True)
    worker_parser.add_argument("--shard-index", type=int, default=0)
    worker_parser.add_argument("--num-shards", type=int, default=1)
    worker_parser.add_argument("--max-jobs", type=int)
    worker_parser.add_argument("--data-dir", type=Path)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--plan-root", type=Path, required=True)
    requeue_parser = subparsers.add_parser("requeue")
    requeue_parser.add_argument("--plan-root", type=Path, required=True)
    requeue_parser.add_argument("--job-id", required=True)
    requeue_parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        output = plan_matrix(args.config, budget=args.budget, output_dir=args.output_dir)
        payload = {"status": "complete", "plan_root": str(output)}
    elif args.command == "worker":
        results = run_worker(
            args.plan_root,
            worker_id=args.worker_id,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            max_jobs=args.max_jobs,
            data_dir_override=args.data_dir,
        )
        payload = {"status": "complete", "jobs_processed": len(results), "results": results}
    elif args.command == "summarize":
        output = summarize_matrix(args.plan_root)
        payload = {"status": "complete", "summary": str(output)}
    else:
        output = requeue_job(args.plan_root, job_id=args.job_id, reason=args.reason)
        payload = {"status": "complete", "archived_attempt": str(output)}
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
