"""Build immutable, nested training-size manifests for browser ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from spider.config import load_config
from spider.diversity import audit_records, sampling_unit
from spider.prepare import canonical_domain, read_jsonl, stable_int, write_jsonl

SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_record_domain(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a registrable domain recovered from the URL when possible."""
    normalized = dict(record)
    domain = str(normalized.get("domain") or "").strip().lower()
    if not domain or domain == "unknown":
        domain = canonical_domain(normalized)
    normalized["domain"] = domain or "unknown"
    return normalized


def _task_spec_sizes(task_spec: dict[str, Any]) -> dict[str, int]:
    sizes = task_spec.get("sizes")
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("Every task requires a non-empty sizes mapping")
    result = {str(tier): int(count) for tier, count in sizes.items()}
    previous = 0
    for tier, count in result.items():
        if count <= previous:
            raise ValueError(
                f"Task size tiers must be strictly increasing; {tier} has {count} after {previous}"
            )
        previous = count
    return result


def validate_ladder_spec(spec: dict[str, Any]) -> list[str]:
    tasks = spec.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("dataset_ladder.tasks must be a non-empty mapping")
    tier_order: list[str] | None = None
    for task, task_spec in tasks.items():
        if not SAFE_LABEL.fullmatch(str(task)):
            raise ValueError(f"Unsafe task label: {task}")
        if not isinstance(task_spec, dict):
            raise TypeError(f"Task {task} specification must be a mapping")
        sizes = _task_spec_sizes(task_spec)
        for tier in sizes:
            if not SAFE_LABEL.fullmatch(tier):
                raise ValueError(f"Unsafe tier label: {tier}")
        if tier_order is None:
            tier_order = list(sizes)
        elif list(sizes) != tier_order:
            raise ValueError("Every task must define the same ordered size tiers")
        temperature = float(task_spec.get("temperature", 0.5))
        max_share = float(task_spec.get("max_domain_share", 0.02))
        max_per_unit = int(task_spec.get("max_per_unit", 1))
        if not 0 <= temperature <= 1:
            raise ValueError(f"Task {task} temperature must be in [0, 1]")
        if not 0 < max_share <= 1:
            raise ValueError(f"Task {task} max_domain_share must be in (0, 1]")
        if max_per_unit <= 0:
            raise ValueError(f"Task {task} max_per_unit must be positive")
    assert tier_order is not None
    return tier_order


def _candidate_pools(
    records: list[dict[str, Any]], *, seed: int, max_per_unit: int
) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        units[sampling_unit(record)].append(record)
    by_domain: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in units.values():
        domains = Counter(str(record["domain"]) for record in group)
        domain = min(domains, key=lambda value: (-domains[value], value))
        ordered = sorted(group, key=lambda record: stable_int(seed, "record", record["id"]))
        by_domain[domain].append(ordered[:max_per_unit])

    pools: dict[str, list[dict[str, Any]]] = {}
    for domain, groups in by_domain.items():
        groups.sort(key=lambda group: stable_int(seed, "unit", sampling_unit(group[0])))
        interleaved: list[dict[str, Any]] = []
        for position in range(max_per_unit):
            interleaved.extend(group[position] for group in groups if position < len(group))
        pools[domain] = interleaved
    return pools


def diversity_order(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    temperature: float = 0.5,
    max_domain_share: float = 0.02,
    max_per_unit: int = 1,
    category_weights: dict[str, float] | None = None,
    required_categories: list[str] | None = None,
    minimum_category_share: float = 0.0,
) -> list[dict[str, Any]]:
    """Return one deterministic order whose prefixes form nested diversity samples."""
    if count <= 0:
        raise ValueError("count must be positive")
    pools = _candidate_pools(records, seed=seed, max_per_unit=max_per_unit)
    if not pools:
        raise ValueError("No candidate records")
    category_weights = {
        str(category): float(weight) for category, weight in (category_weights or {}).items()
    }
    if any(weight <= 0 for weight in category_weights.values()):
        raise ValueError("category_weights must all be positive")
    if not 0 <= minimum_category_share < 1:
        raise ValueError("minimum_category_share must be in [0, 1)")
    domain_categories: dict[str, str] = {}
    for domain, pool in pools.items():
        categories = Counter(
            str(record.get("website_category") or "unclassified") for record in pool
        )
        domain_categories[domain] = min(
            categories, key=lambda value: (-categories[value], value)
        )
    available_categories = set(domain_categories.values())
    missing_categories = sorted(set(required_categories or []) - available_categories)
    if missing_categories:
        raise ValueError(f"Required website categories have no candidates: {missing_categories}")
    categories_to_cover = sorted(set(required_categories or []))
    if minimum_category_share * len(categories_to_cover) > 1:
        raise ValueError("minimum category shares exceed total sample capacity")
    domain_cap = max(1, math.ceil(count * max_domain_share))
    capacities = {domain: min(len(pool), domain_cap) for domain, pool in pools.items()}
    if sum(capacities.values()) < count:
        raise ValueError(
            f"Diversity constraints allow {sum(capacities.values())}/{count} records; "
            "increase max_domain_share, max_per_unit, or source diversity"
        )
    weights = {
        domain: (capacities[domain] ** temperature if temperature else 1.0)
        * category_weights.get(domain_categories[domain], 1.0)
        for domain in pools
    }
    selected_counts: Counter[str] = Counter()
    selected_categories: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [
            domain for domain, capacity in capacities.items() if selected_counts[domain] < capacity
        ]
        next_size = len(selected) + 1
        deficient_categories = {
            category
            for category in categories_to_cover
            if selected_categories[category] < math.floor(next_size * minimum_category_share)
        }
        if deficient_categories:
            covered = [
                domain for domain in available if domain_categories[domain] in deficient_categories
            ]
            if not covered:
                raise ValueError(
                    "Website category floor cannot be met under domain and sampling-unit caps: "
                    f"{sorted(deficient_categories)}"
                )
            available = covered
        domain = min(
            available,
            key=lambda value: (
                (selected_counts[value] + 1) / weights[value],
                stable_int(seed, "domain", value),
            ),
        )
        selected.append(pools[domain][selected_counts[domain]])
        selected_counts[domain] += 1
        selected_categories[domain_categories[domain]] += 1
    return selected


def build_nested_task_samples(
    records: list[dict[str, Any]],
    sizes: dict[str, int],
    *,
    seed: int,
    temperature: float,
    max_domain_share: float,
    max_per_unit: int,
    category_weights: dict[str, float] | None = None,
    required_categories: list[str] | None = None,
    minimum_category_share: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    largest = max(sizes.values())
    ordered = diversity_order(
        records,
        count=largest,
        seed=seed,
        temperature=temperature,
        max_domain_share=max_domain_share,
        max_per_unit=max_per_unit,
        category_weights=category_weights,
        required_categories=required_categories,
        minimum_category_share=minimum_category_share,
    )
    result = {tier: ordered[:count] for tier, count in sizes.items()}
    for tier, sample in result.items():
        audit = audit_records(sample)
        if audit["max_domain_share"] > max_domain_share + (1 / len(sample)):
            raise ValueError(
                f"Tier {tier} exceeds domain cap: {audit['max_domain_share']:.4f} > "
                f"{max_domain_share:.4f}"
            )
        category_counts = audit["website_category_counts"]
        for category in required_categories or []:
            realized = int(category_counts.get(category, 0)) / len(sample)
            if realized + (1 / len(sample)) < minimum_category_share:
                raise ValueError(
                    f"Tier {tier} category {category} is below its floor: "
                    f"{realized:.4f} < {minimum_category_share:.4f}"
                )
    return result


def _known_domains(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record.get("domain") or "").strip().lower()
        for record in records
        if str(record.get("domain") or "").strip().lower() not in {"", "unknown"}
    }


def leakage_audit(
    training: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> dict[str, Any]:
    train_ids = {str(record["id"]) for record in training}
    eval_ids = {str(record["id"]) for record in evaluation}
    train_units = {sampling_unit(record) for record in training}
    eval_units = {sampling_unit(record) for record in evaluation}
    domain_overlap = sorted(_known_domains(training) & _known_domains(evaluation))
    id_overlap = sorted(train_ids & eval_ids)
    unit_overlap = sorted(train_units & eval_units)
    return {
        "training_examples": len(training),
        "evaluation_examples": len(evaluation),
        "id_overlap_count": len(id_overlap),
        "sampling_unit_overlap_count": len(unit_overlap),
        "known_domain_overlap_count": len(domain_overlap),
        "id_overlap_preview": id_overlap[:20],
        "sampling_unit_overlap_preview": unit_overlap[:20],
        "known_domain_overlap_preview": domain_overlap[:20],
        "training_unknown_domains": sum(
            str(record.get("domain") or "").lower() == "unknown" for record in training
        ),
        "evaluation_unknown_domains": sum(
            str(record.get("domain") or "").lower() == "unknown" for record in evaluation
        ),
    }


def _resolve_paths(config_path: Path, values: list[str]) -> list[Path]:
    return [
        (Path(value) if Path(value).is_absolute() else (config_path.parent / value)).resolve()
        for value in values
    ]


def build_data_ladder(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    spec = config.get("dataset_ladder")
    if not isinstance(spec, dict):
        raise TypeError("Config requires a dataset_ladder mapping")
    tiers = validate_ladder_spec(spec)
    output = Path(str(spec["output_dir"]))
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    train_paths = _resolve_paths(config_path, list(spec.get("train_manifests") or []))
    evaluation_suites_source = spec.get("evaluation_suites")
    if evaluation_suites_source is None:
        evaluation_suites_source = {"evaluation": list(spec.get("evaluation_manifests") or [])}
    if not isinstance(evaluation_suites_source, dict) or not evaluation_suites_source:
        raise ValueError("evaluation_suites must be a non-empty mapping")
    evaluation_paths_by_suite: dict[str, list[Path]] = {}
    required_disjoint_by_suite: dict[str, list[str]] = {}
    valid_disjoint = {"id", "sampling_unit", "known_domain"}
    for suite, value in evaluation_suites_source.items():
        suite = str(suite)
        if isinstance(value, dict):
            manifests = value.get("manifests")
            required = list(value.get("required_disjoint") or sorted(valid_disjoint))
        else:
            manifests = value
            required = sorted(valid_disjoint)
        if not isinstance(manifests, list):
            raise TypeError(f"Evaluation suite {suite} manifests must be a list")
        unknown_requirements = sorted(set(required) - valid_disjoint)
        if unknown_requirements:
            raise ValueError(
                f"Evaluation suite {suite} has unknown disjoint requirements: "
                f"{unknown_requirements}"
            )
        evaluation_paths_by_suite[suite] = _resolve_paths(config_path, manifests)
        required_disjoint_by_suite[suite] = required
    for suite in evaluation_paths_by_suite:
        if not SAFE_LABEL.fullmatch(suite):
            raise ValueError(f"Unsafe evaluation suite label: {suite}")
    if not train_paths or any(not paths for paths in evaluation_paths_by_suite.values()):
        raise ValueError("Training manifests and every evaluation suite require source manifests")
    source_hashes = {str(path): sha256_file(path) for path in train_paths}
    evaluation_source_hashes = {
        suite: {str(path): sha256_file(path) for path in paths}
        for suite, paths in evaluation_paths_by_suite.items()
    }
    input_identity = {
        "dataset_id": str(spec["id"]),
        "seed": int(spec.get("seed", 0)),
        "config_sha256": sha256_file(config_path),
        "source_manifests": source_hashes,
        "evaluation_source_manifests": evaluation_source_hashes,
    }
    existing_path = output / "dataset_ladder.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_identity = {
            "dataset_id": existing.get("dataset_id"),
            "seed": existing.get("seed"),
            "config_sha256": existing.get("config_sha256"),
            "source_manifests": existing.get("source_manifests"),
            "evaluation_source_manifests": {
                suite: row.get("source_manifests")
                for suite, row in (existing.get("evaluation_suites") or {}).items()
            },
        }
        if existing_identity != input_identity:
            raise ValueError(f"Immutable dataset ladder already exists with different inputs: {output}")
        registered_files = [
            (output / row["manifest"], row["sha256"])
            for section in ("tiers", "evaluation_suites")
            for row in (existing.get(section) or {}).values()
        ]
        for path, expected_hash in registered_files:
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Frozen dataset ladder artifact is missing or corrupted: {path}")
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is non-empty without dataset_ladder.json: {output}")

    training = [
        canonicalize_record_domain(record)
        for path in train_paths
        for record in read_jsonl(path)
    ]
    evaluation_by_suite = {
        suite: [
            canonicalize_record_domain(record)
            for path in paths
            for record in read_jsonl(path)
        ]
        for suite, paths in evaluation_paths_by_suite.items()
    }
    evaluation = [record for records in evaluation_by_suite.values() for record in records]
    ids = [str(record["id"]) for record in training]
    if len(ids) != len(set(ids)):
        raise ValueError("Training candidate IDs must be globally unique")
    leakage = leakage_audit(training, evaluation)
    if leakage["id_overlap_count"] or leakage["sampling_unit_overlap_count"]:
        raise ValueError(f"Training/evaluation leakage detected: {leakage}")
    suite_leakage = {
        suite: leakage_audit(training, records)
        for suite, records in evaluation_by_suite.items()
    }
    overlap_fields = {
        "id": "id_overlap_count",
        "sampling_unit": "sampling_unit_overlap_count",
        "known_domain": "known_domain_overlap_count",
    }
    for suite, required in required_disjoint_by_suite.items():
        violations = {
            requirement: suite_leakage[suite][overlap_fields[requirement]]
            for requirement in required
            if suite_leakage[suite][overlap_fields[requirement]]
        }
        if violations:
            raise ValueError(f"Training/evaluation leakage detected in {suite}: {violations}")
    if str(spec.get("unknown_domain_policy", "reject")) == "reject" and (
        leakage["training_unknown_domains"] or leakage["evaluation_unknown_domains"]
    ):
        raise ValueError("Unknown domains remain after URL canonicalization")

    task_samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seed = int(spec.get("seed", 0))
    website_sampling = spec.get("website_sampling") or {}
    if not isinstance(website_sampling, dict):
        raise TypeError("website_sampling must be a mapping")
    for task_index, (task, task_spec) in enumerate(spec["tasks"].items()):
        candidates = [record for record in training if str(record.get("task")) == task]
        sizes = _task_spec_sizes(task_spec)
        task_samples[task] = build_nested_task_samples(
            candidates,
            sizes,
            seed=seed + task_index,
            temperature=float(task_spec.get("temperature", 0.5)),
            max_domain_share=float(task_spec.get("max_domain_share", 0.02)),
            max_per_unit=int(task_spec.get("max_per_unit", 1)),
            category_weights=dict(website_sampling.get("category_weights") or {}),
            required_categories=list(website_sampling.get("required_categories") or []),
            minimum_category_share=float(
                website_sampling.get("minimum_category_share", 0.0)
            ),
        )

    manifest_dir = output / "manifests"
    for suite, records in evaluation_by_suite.items():
        write_jsonl(manifest_dir / f"eval_{suite}.jsonl", records)
    tier_records: dict[str, list[dict[str, Any]]] = {}
    max_combined_share = float(spec.get("max_combined_domain_share", 0.02))
    if not 0 < max_combined_share <= 1:
        raise ValueError("max_combined_domain_share must be in (0, 1]")
    for tier_index, tier in enumerate(tiers):
        combined = [
            record
            for task in spec["tasks"]
            for record in task_samples[str(task)][tier]
        ]
        combined.sort(key=lambda record: stable_int(seed, tier_index, record["id"]))
        combined_audit = audit_records(combined)
        if combined_audit["max_domain_share"] > max_combined_share + (1 / len(combined)):
            raise ValueError(
                f"Combined tier {tier} exceeds domain cap: "
                f"{combined_audit['max_domain_share']:.4f} > {max_combined_share:.4f}"
            )
        write_jsonl(manifest_dir / f"train_{tier}.jsonl", combined)
        tier_records[tier] = combined

    nested_checks: dict[str, Any] = {}
    for previous, current in pairwise(tiers):
        previous_ids = {str(record["id"]) for record in tier_records[previous]}
        current_ids = {str(record["id"]) for record in tier_records[current]}
        nested_checks[f"{previous}_in_{current}"] = previous_ids <= current_ids
    if not all(nested_checks.values()):
        raise RuntimeError("Generated manifests are not strictly nested")

    provenance = {
        "schema_version": 1,
        "dataset_id": str(spec["id"]),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifests": source_hashes,
        "evaluation_suites": {
            suite: {
                "source_manifests": evaluation_source_hashes[suite],
                "manifest": f"manifests/eval_{suite}.jsonl",
                "sha256": sha256_file(manifest_dir / f"eval_{suite}.jsonl"),
                "audit": audit_records(evaluation_by_suite[suite]),
                "required_disjoint": required_disjoint_by_suite[suite],
                "leakage_audit": suite_leakage[suite],
            }
            for suite, paths in evaluation_paths_by_suite.items()
        },
        "source_audit": audit_records(training),
        "evaluation_audit": audit_records(evaluation),
        "leakage_audit": leakage,
        "tiers": {
            tier: {
                "manifest": f"manifests/train_{tier}.jsonl",
                "sha256": sha256_file(manifest_dir / f"train_{tier}.jsonl"),
                "audit": audit_records(records),
            }
            for tier, records in tier_records.items()
        },
        "nested_checks": nested_checks,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_ladder.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested browser-training data manifests")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = build_data_ladder(args.config)
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
