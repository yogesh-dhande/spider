import json
from pathlib import Path

import pytest
import yaml

from spider.data_ladder import (
    build_data_ladder,
    build_nested_task_samples,
    canonicalize_record_domain,
    leakage_audit,
)
from spider.prepare import read_jsonl, write_jsonl


def _records(task: str, domains: list[str], count_per_domain: int = 2) -> list[dict]:
    records = []
    for domain in domains:
        for index in range(count_per_domain):
            identifier = f"{task}-{domain}-{index}"
            record = {
                "id": identifier,
                "task": task,
                "domain": domain,
                "url": f"https://{domain}/{index}",
                "image": f"images/{identifier}.jpg",
            }
            if task == "action":
                record["trajectory_id"] = f"trajectory-{domain}-{index}"
            records.append(record)
    return records


def test_nested_samples_are_exact_deterministic_prefixes() -> None:
    records = _records("qa", [f"site{index}.test" for index in range(8)])
    kwargs = {
        "seed": 7,
        "temperature": 0.5,
        "max_domain_share": 0.5,
        "max_per_unit": 1,
    }
    first = build_nested_task_samples(records, {"small": 4, "medium": 8, "large": 12}, **kwargs)
    second = build_nested_task_samples(records, {"small": 4, "medium": 8, "large": 12}, **kwargs)
    assert first == second
    assert [len(first[tier]) for tier in ("small", "medium", "large")] == [4, 8, 12]
    assert {row["id"] for row in first["small"]} <= {
        row["id"] for row in first["medium"]
    }
    assert {row["id"] for row in first["medium"]} <= {
        row["id"] for row in first["large"]
    }


def test_domain_is_recovered_from_url() -> None:
    record = canonicalize_record_domain(
        {"id": "x", "domain": "unknown", "url": "https://www.example.co.uk/path"}
    )
    assert record["domain"] == "example.co.uk"


def test_leakage_audit_finds_domain_and_sampling_unit_overlap() -> None:
    training = [{"id": "a", "task": "qa", "domain": "example.com", "image": "same.jpg"}]
    evaluation = [{"id": "b", "task": "qa", "domain": "example.com", "image": "same.jpg"}]
    audit = leakage_audit(training, evaluation)
    assert audit["known_domain_overlap_count"] == 1
    assert audit["sampling_unit_overlap_count"] == 1


def test_build_data_ladder_writes_nested_manifests_and_provenance(tmp_path: Path) -> None:
    tasks = ("action", "qa", "grounding")
    training = [
        row
        for task in tasks
        for row in _records(task, [f"train{index}.test" for index in range(8)])
    ]
    evaluation = [
        row
        for task in tasks
        for row in _records(task, [f"eval{index}.test" for index in range(2)], 1)
    ]
    train_path = tmp_path / "inputs/train.jsonl"
    eval_path = tmp_path / "inputs/eval.jsonl"
    write_jsonl(train_path, training)
    write_jsonl(eval_path, evaluation)
    config = {
        "dataset_ladder": {
            "id": "test-ladder",
            "seed": 11,
            "output_dir": "output",
            "train_manifests": ["inputs/train.jsonl"],
            "evaluation_manifests": ["inputs/eval.jsonl"],
            "unknown_domain_policy": "reject",
            "max_combined_domain_share": 0.5,
            "tasks": {
                task: {
                    "sizes": {"small": 4, "medium": 8, "large": 12},
                    "temperature": 0.5,
                    "max_domain_share": 0.5,
                    "max_per_unit": 1,
                }
                for task in tasks
            },
        }
    }
    config_path = tmp_path / "ladder.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = build_data_ladder(config_path)
    provenance = json.loads((output / "dataset_ladder.json").read_text())
    manifests = {
        tier: read_jsonl(output / f"manifests/train_{tier}.jsonl")
        for tier in ("small", "medium", "large")
    }
    assert [len(manifests[tier]) for tier in manifests] == [12, 24, 36]
    assert all(provenance["nested_checks"].values())
    assert provenance["leakage_audit"]["known_domain_overlap_count"] == 0
    assert provenance["tiers"]["large"]["sha256"]
    assert len(read_jsonl(output / "manifests/eval_evaluation.jsonl")) == len(evaluation)
    assert build_data_ladder(config_path) == output

    with train_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "changed"}) + "\n")
    with pytest.raises(ValueError, match="Immutable dataset ladder"):
        build_data_ladder(config_path)


def test_build_data_ladder_rejects_domain_leakage(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train_path, _records("qa", ["shared.test"]))
    write_jsonl(eval_path, _records("qa", ["shared.test"], 1))
    config = {
        "dataset_ladder": {
            "id": "bad",
            "seed": 1,
            "output_dir": "output",
            "train_manifests": ["train.jsonl"],
            "evaluation_manifests": ["eval.jsonl"],
            "max_combined_domain_share": 1.0,
            "tasks": {
                "qa": {
                    "sizes": {"small": 1},
                    "max_domain_share": 1.0,
                    "max_per_unit": 1,
                }
            },
        }
    }
    config_path = tmp_path / "ladder.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        build_data_ladder(config_path)
