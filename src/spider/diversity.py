"""Deterministic website-diversity audits and group-aware manifest sampling."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spider.prepare import read_jsonl, stable_int, write_jsonl


def sampling_unit(record: dict[str, Any]) -> str:
    """Return the unit that must not dominate or be split during sampling."""
    if record.get("trajectory_id"):
        return f"trajectory:{record['trajectory_id']}"
    if record.get("image"):
        return f"image:{record['image']}"
    return f"record:{record['id']}"


def _distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record.get(field) or "unknown") for record in records)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    domains = Counter(str(record.get("domain") or "unknown") for record in records)
    action_names = Counter(
        str((record.get("target_action") or {}).get("name") or "unknown")
        for record in records
        if record.get("task") == "action"
    )
    total = len(records)
    shares = [count / total for count in domains.values()] if total else []
    effective_domains = 1 / sum(share**2 for share in shares) if shares else 0.0
    top_counts = sorted(domains.values(), reverse=True)
    return {
        "examples": total,
        "unique_domains": len(domains),
        "unique_sampling_units": len({sampling_unit(record) for record in records}),
        "unique_trajectories": len(
            {str(record["trajectory_id"]) for record in records if record.get("trajectory_id")}
        ),
        "unique_images": len(
            {str(record["image"]) for record in records if record.get("image")}
        ),
        "max_domain_share": max(shares, default=0.0),
        "top_5_domain_share": sum(top_counts[:5]) / total if total else 0.0,
        "effective_domains": effective_domains,
        "domain_counts": dict(sorted(domains.items(), key=lambda item: (-item[1], item[0]))),
        "task_counts": _distribution(records, "task"),
        "source_counts": _distribution(records, "source"),
        "action_counts": dict(
            sorted(action_names.items(), key=lambda item: (-item[1], item[0]))
        ),
        "question_type_counts": _distribution(
            [record for record in records if record.get("task") == "qa"],
            "question_type",
        ),
    }


def _domain_candidates(
    records: list[dict[str, Any]], seed: int, max_per_unit: int
) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        units[sampling_unit(record)].append(record)
    by_domain_units: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in units.values():
        domains = Counter(str(record.get("domain") or "unknown") for record in group)
        domain = min(domains, key=lambda value: (-domains[value], value))
        ordered = sorted(group, key=lambda record: stable_int(seed, "record", record["id"]))
        by_domain_units[domain].append(ordered[:max_per_unit])

    candidates: dict[str, list[dict[str, Any]]] = {}
    for domain, groups in by_domain_units.items():
        groups.sort(key=lambda group: stable_int(seed, "unit", sampling_unit(group[0])))
        interleaved: list[dict[str, Any]] = []
        for position in range(max_per_unit):
            interleaved.extend(group[position] for group in groups if position < len(group))
        candidates[domain] = interleaved
    return candidates


def balanced_group_sample(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
    temperature: float = 0.5,
    max_domain_share: float = 0.02,
    max_per_unit: int = 1,
) -> list[dict[str, Any]]:
    """Sample complete-distribution candidates without site/screenshot domination.

    Domains are scheduled proportionally to ``available_examples ** temperature``.
    A temperature of 1 preserves natural frequency; 0 approaches uniform-over-domain.
    """
    if count <= 0 or count > len(records):
        raise ValueError("count must be in [1, len(records)]")
    if not 0 <= temperature <= 1:
        raise ValueError("temperature must be in [0, 1]")
    if not 0 < max_domain_share <= 1:
        raise ValueError("max_domain_share must be in (0, 1]")
    if max_per_unit <= 0:
        raise ValueError("max_per_unit must be positive")

    pools = _domain_candidates(records, seed, max_per_unit)
    domain_cap = max(1, math.ceil(count * max_domain_share))
    capacities = {domain: min(len(pool), domain_cap) for domain, pool in pools.items()}
    if sum(capacities.values()) < count:
        raise ValueError(
            f"Diversity constraints allow {sum(capacities.values())}/{count} examples; "
            "increase max_domain_share or max_per_unit"
        )
    weights = {
        domain: capacities[domain] ** temperature if temperature else 1.0 for domain in pools
    }
    selected_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [
            domain for domain, capacity in capacities.items() if selected_counts[domain] < capacity
        ]
        domain = min(
            available,
            key=lambda value: (
                (selected_counts[value] + 1) / weights[value],
                stable_int(seed, "domain", value),
            ),
        )
        selected.append(pools[domain][selected_counts[domain]])
        selected_counts[domain] += 1
    return selected


def macro_boolean_metric(
    records: list[dict[str, Any]], field: str, group_field: str = "domain"
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get(field) is None:
            continue
        grouped[str(record.get(group_field) or "unknown")].append(float(bool(record[field])))
    values = {group: sum(items) / len(items) for group, items in sorted(grouped.items())}
    total = sum(len(items) for items in grouped.values())
    return {
        "examples": total,
        "groups": len(values),
        "micro": (
            sum(sum(items) for items in grouped.values()) / total if total else None
        ),
        "macro": sum(values.values()) / len(values) if values else None,
        "by_group": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or diversity-sample JSONL manifests")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-domain-share", type=float, default=0.02)
    parser.add_argument("--max-per-unit", type=int, default=1)
    args = parser.parse_args()
    records = [record for path in args.manifest for record in read_jsonl(path)]
    payload: dict[str, Any] = {"input": audit_records(records)}
    if (args.sample_output is None) != (args.sample_count is None):
        parser.error("--sample-output and --sample-count must be supplied together")
    if args.sample_output is not None and args.sample_count is not None:
        sampled = balanced_group_sample(
            records,
            args.sample_count,
            args.seed,
            args.temperature,
            args.max_domain_share,
            args.max_per_unit,
        )
        write_jsonl(args.sample_output, sampled)
        payload["sample"] = audit_records(sampled)
        payload["sampling"] = {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_domain_share": args.max_domain_share,
            "max_per_unit": args.max_per_unit,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
