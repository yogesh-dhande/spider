import json
from pathlib import Path

import huggingface_hub
from datasets import Dataset

from spider import source_inventory
from spider.prepare import read_jsonl
from spider.source_inventory import (
    action_metadata_examples,
    domain_partition,
    inventory_arrow_source_file,
    record_destination,
    screenshot_metadata_examples,
)

ACTION_SOURCE = {
    "id": "actions",
    "task": "action",
    "generator": "multi_agent",
    "role": "training",
    "dataset": "test/actions",
    "source_revision": "abc",
}


def test_action_metadata_inventory_never_requires_image_bytes() -> None:
    row = {
        "sample_id": "trajectory-1",
        "instruction": json.dumps({"goal": "Edit the project title"}),
        "trajectory": json.dumps(
            {
                "1": {
                    "screenshot": "screenshot_001.png",
                    "image_w": 1280,
                    "image_h": 720,
                    "other_obs": {
                        "page_index": 0,
                        "open_pages_titles": ["Docs"],
                        "open_pages_urls": ["https://docs.google.com/document/d/1"],
                    },
                    "action": {
                        "action_output": {
                            "thought": "Select the title",
                            "action_name": "click",
                            "action": {
                                "bbox": [100, 50, 200, 40],
                                "button": "left",
                                "click_type": "single",
                            },
                        }
                    },
                }
            }
        ),
    }
    records = action_metadata_examples(
        row,
        source=ACTION_SOURCE,
        file_path="data/actions.parquet",
        row_group=2,
        row_in_group=3,
        row_index=203,
        max_past_steps=4,
        max_steps=4,
    )
    assert len(records) == 1
    assert records[0]["question"] == "Edit the project title"
    assert records[0]["website_surface"] == "docs.google.com"
    assert records[0]["website_category"] == "work_application"
    assert records[0]["image"].startswith("locator://")
    assert records[0]["image_locator"]["screenshot"] == "screenshot_001.png"


def test_qa_and_grounding_metadata_inventory() -> None:
    common = {
        "file_path": "data/part.parquet",
        "row_group": 0,
        "row_in_group": 1,
        "row_index": 1,
        "max_messages": 3,
    }
    qa = screenshot_metadata_examples(
        {
            "metadata": {"website": "notion", "url": "https://notion.so/workspace"},
            "messages": [
                {
                    "question": "What is the project name?",
                    "answer": "Spider",
                    "question_type": "OCR",
                }
            ],
        },
        source={
            "id": "qa",
            "task": "qa",
            "generator": "qa",
            "role": "training",
            "dataset": "test/qa",
            "source_revision": "abc",
        },
        **common,
    )
    assert qa[0]["answer"] == "Spider"
    assert qa[0]["website_category"] == "work_application"
    ground = screenshot_metadata_examples(
        {
            "metadata": {
                "website": "calendar",
                "url": "https://calendar.google.com/",
                "image_w": 1000,
                "image_h": 500,
            },
            "messages": [{"question": "new event", "bbox": "[100, 50, 300, 150]"}],
        },
        source={
            "id": "ground",
            "task": "grounding",
            "generator": "gpt",
            "role": "training",
            "dataset": "test/ground",
            "source_revision": "abc",
        },
        **common,
    )
    assert ground[0]["target_point_normalized"] == [200.0, 200.0]


def test_arrow_inventory_projects_metadata_and_preserves_row_locator(
    tmp_path: Path, monkeypatch
) -> None:
    saved = tmp_path / "saved"
    Dataset.from_dict(
        {
            "image": [b"not-read"],
            "messages": [
                [
                    {
                        "question": "What is the document title?",
                        "answer": "Roadmap",
                        "question_type": "OCR",
                        "question_form": "first_person",
                    }
                ]
            ],
            "metadata": [{"website": "docs", "url": "https://docs.google.com/document/d/1"}],
        }
    ).save_to_disk(saved)
    arrow = next(saved.glob("*.arrow"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(arrow))
    source = {
        "id": "qa-full",
        "task": "qa",
        "generator": "synthetic_qa",
        "role": "training",
        "format": "arrow",
        "dataset": "test/qa",
        "source_revision": "abc",
        "data_revision": "abc",
        "file": "train/data.arrow",
        "size": arrow.stat().st_size,
        "blob_id": "blob",
    }
    summary = inventory_arrow_source_file(
        source,
        cache_root=tmp_path / "cache",
        spec={
            "arrow_batch_rows": 1,
            "max_candidates_per_unit": {"qa": 3},
            "website_rules": [],
        },
    )
    rows = read_jsonl(Path(summary["cache_dir"]) / "row-group-00000.jsonl")
    assert summary["complete"] is True
    assert rows[0]["answer"] == "Roadmap"
    assert rows[0]["image_locator"]["kind"] == "arrow_single_image"
    assert rows[0]["image_locator"]["row_index"] == 0


def test_split_assignment_is_domain_and_unit_aware() -> None:
    percentages = {"train": 80, "domain_balanced": 10, "distribution_shift": 10}
    domains = [f"site-{index}.test" for index in range(1000)]
    grouped = {name: [] for name in percentages}
    for domain in domains:
        grouped[domain_partition(domain, 53, percentages)].append(domain)
    assert all(grouped.values())
    train_domain = grouped["train"][0]
    base = {
        "id": "a",
        "task": "qa",
        "domain": train_domain,
        "image": "locator://same",
        "source_role": "training",
    }
    first = record_destination(base, seed=53, percentages=percentages, iid_percent=5)
    second = record_destination(
        {**base, "id": "b"}, seed=53, percentages=percentages, iid_percent=5
    )
    assert first == second
    shifted = {
        **base,
        "domain": grouped["distribution_shift"][0],
        "source_role": "distribution_shift",
    }
    assert (
        record_destination(shifted, seed=53, percentages=percentages, iid_percent=5)
        == "distribution_shift"
    )


def test_single_source_worker_never_attempts_global_finalization(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "inventory.yaml"
    config_path.write_text(
        "inventory:\n  id: test\n  output_dir: output\n  sources: []\n",
        encoding="utf-8",
    )
    source_file = {"id": "grounding_gpt", "file": "gpt/train.parquet"}
    monkeypatch.setattr(
        source_inventory,
        "_resolve_sources",
        lambda _spec: ([source_file], [{"source_revision": "abc"}]),
    )
    monkeypatch.setattr(
        source_inventory,
        "inventory_source_file",
        lambda *_args, **_kwargs: {"complete": True, "accepted_examples": 130370},
    )
    monkeypatch.setattr(
        source_inventory,
        "freeze_manifests",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-source worker attempted global finalization")
        ),
    )

    output = source_inventory.build_inventory(
        config_path,
        source_ids={"grounding_gpt"},
    )

    assert output.name == "scan_shard_00_of_01.json"
    assert json.loads(output.read_text())[0]["accepted_examples"] == 130370
