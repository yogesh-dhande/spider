from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from spider.config import load_config
from spider.rl.policies import make_policy
from spider.rl.rollout import LocalArtifactStore, append_jsonl, load_jsonl, run_episode
from spider.rl.sandbox import load_task_suite


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None, None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip(), bool(dirty.stdout.strip()) if dirty.returncode == 0 else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(row["steps_taken"]) for row in rows)
    parse_errors = sum(int(row["parse_errors"]) for row in rows)
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_reward": mean(float(row["total_reward"]) for row in rows) if rows else 0.0,
        "mean_steps": mean(int(row["steps_taken"]) for row in rows) if rows else 0.0,
        "parse_error_rate": parse_errors / total_steps if total_steps else 0.0,
    }


def _paired_bootstrap_interval(
    deltas: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(deltas) for _ in deltas) for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def _compare(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        return {(str(row["task_id"]), int(row["seed"])): row for row in rows}

    control = keyed(control_rows)
    candidate = keyed(candidate_rows)
    pairs = sorted(set(control) & set(candidate))
    if len(pairs) != len(control) or len(pairs) != len(candidate):
        raise ValueError("Ablation variants must have identical paired task/seed episodes")
    success_deltas = [
        float(bool(candidate[key]["success"])) - float(bool(control[key]["success"]))
        for key in pairs
    ]
    reward_deltas = [
        float(candidate[key]["total_reward"]) - float(control[key]["total_reward"])
        for key in pairs
    ]
    lower, upper = _paired_bootstrap_interval(
        success_deltas, samples=bootstrap_samples, seed=bootstrap_seed
    )
    return {
        "paired_episodes": len(pairs),
        "success_rate_delta": mean(success_deltas) if success_deltas else 0.0,
        "success_rate_delta_ci95": [lower, upper],
        "mean_reward_delta": mean(reward_deltas) if reward_deltas else 0.0,
    }


def _comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['study_id']} — {summary['run_id']}",
        "",
        "Paired variants use identical task IDs and seeds.",
        "",
        "| Variant | Episodes | Success rate | Mean reward | Mean steps | Parse error rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant_id, metrics in summary["variants"].items():
        lines.append(
            f"| {variant_id} | {metrics['episodes']} | {metrics['success_rate']:.4f} | "
            f"{metrics['mean_reward']:.4f} | {metrics['mean_steps']:.2f} | "
            f"{metrics['parse_error_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Candidate vs control | Paired episodes | Success-rate delta | 95% paired bootstrap CI | Mean-reward delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant_id, comparison in summary["comparisons"].items():
        lower, upper = comparison["success_rate_delta_ci95"]
        lines.append(
            f"| {variant_id} vs {summary['control_variant']} | "
            f"{comparison['paired_episodes']} | {comparison['success_rate_delta']:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] | {comparison['mean_reward_delta']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_study(
    config_path: str | Path,
    *,
    run_id_override: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    study = config.get("study")
    if not isinstance(study, dict):
        raise TypeError("study config must contain a study mapping")
    study_id = study.get("id")
    run_id = run_id_override or study.get("run_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("study.id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("study.run_id or --run-id is required")
    suite_path = Path(str(study["suite_path"]))
    if not suite_path.is_absolute():
        suite_path = (config_path.parent / suite_path).resolve()
    suite_id, all_tasks = load_task_suite(suite_path)
    tasks = [
        task
        for task in all_tasks
        if int(hashlib.sha256(task.task_id.encode()).hexdigest(), 16) % num_shards == shard_index
    ]
    if not tasks:
        raise ValueError("Selected shard contains no tasks")

    base = study.get("base")
    variants = study.get("variants")
    if not isinstance(base, dict) or not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("study requires a base mapping and at least two variants")
    variant_configs: dict[str, dict[str, Any]] = {}
    for row in variants:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise TypeError("Each variant requires an id")
        variant_id = row["id"]
        if variant_id in variant_configs:
            raise ValueError(f"Duplicate variant ID: {variant_id}")
        overlay = row.get("overlay", {})
        if not isinstance(overlay, dict):
            raise TypeError(f"Variant {variant_id} overlay must be a mapping")
        variant_configs[variant_id] = deep_merge(base, overlay)
    control_variant = study.get("control_variant")
    if control_variant not in variant_configs:
        raise ValueError("study.control_variant must name one variant")

    output_root = Path(str(study.get("output_dir", "outputs/studies")))
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_root = output_root / study_id / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    effective_config = deepcopy(config)
    effective_config["study"]["run_id"] = run_id
    config_hash = _canonical_hash(effective_config)
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash:
            raise ValueError(f"Run ID {run_id} already exists with a different config")
    else:
        source_commit, source_dirty = _git_state(Path.cwd())
        manifest = {
            "schema_version": 1,
            "status": "running",
            "study_id": study_id,
            "run_id": run_id,
            "suite_id": suite_id,
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "source_git_commit": source_commit,
            "source_git_dirty": source_dirty,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "shard_index": shard_index,
            "num_shards": num_shards,
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(run_root / "config.json", effective_config)

    repeats = int(study.get("repeats", 1))
    seed = int(study.get("seed", 0))
    max_steps = int(study.get("max_steps", 8))
    if repeats <= 0 or max_steps <= 0:
        raise ValueError("study.repeats and study.max_steps must be positive")
    artifact_store = LocalArtifactStore(run_root / "artifacts")
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_id, variant_config in variant_configs.items():
        environment_config = variant_config.get("environment")
        if not isinstance(environment_config, dict):
            raise TypeError(f"Variant {variant_id} has no environment mapping")
        if environment_config.get("type") != "deterministic_browser":
            raise ValueError(
                f"Variant {variant_id} has unsupported environment type: "
                f"{environment_config.get('type')!r}"
            )
        policy_config = variant_config.get("policy")
        if not isinstance(policy_config, dict):
            raise TypeError(f"Variant {variant_id} has no policy mapping")
        policy = make_policy(policy_config, tasks)
        episode_path = run_root / "variants" / variant_id / "episodes.jsonl"
        existing = load_jsonl(episode_path)
        completed = {str(row["episode_id"]) for row in existing}
        for task in tasks:
            for repeat in range(repeats):
                episode_seed = seed + repeat
                episode_id = hashlib.sha256(
                    f"{variant_id}\0{task.task_id}\0{episode_seed}".encode()
                ).hexdigest()[:20]
                if episode_id in completed:
                    continue
                episode = run_episode(
                    variant_id=variant_id,
                    task=task,
                    seed=episode_seed,
                    policy=policy,
                    artifact_store=artifact_store,
                    max_steps=max_steps,
                )
                append_jsonl(episode_path, episode)
                existing.append(episode)
                completed.add(episode_id)
        rows_by_variant[variant_id] = existing

    bootstrap_samples = int(study.get("bootstrap_samples", 2000))
    if bootstrap_samples <= 0:
        raise ValueError("study.bootstrap_samples must be positive")
    summary = {
        "schema_version": 1,
        "study_id": study_id,
        "run_id": run_id,
        "suite_id": suite_id,
        "control_variant": control_variant,
        "paired_design": True,
        "variants": {key: _aggregate(value) for key, value in rows_by_variant.items()},
        "comparisons": {},
    }
    for variant_id, rows in rows_by_variant.items():
        if variant_id == control_variant:
            continue
        summary["comparisons"][variant_id] = _compare(
            rows_by_variant[control_variant],
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
    _atomic_json(run_root / "summary.json", summary)
    (run_root / "comparison.md").write_text(_comparison_markdown(summary), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest["summary_sha256"] = hashlib.sha256(
        (run_root / "summary.json").read_bytes()
    ).hexdigest()
    _atomic_json(manifest_path, manifest)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired browser-agent ablation study")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    output = run_study(
        args.config,
        run_id_override=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
