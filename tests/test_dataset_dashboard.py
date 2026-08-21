import json
from pathlib import Path

from spider.dataset_dashboard import build_dataset_dashboard, deterministic_stratified_sample


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_deterministic_stratified_sample_and_aggregates(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"row-{index}",
            "task": "qa" if index % 2 else "grounding",
            "domain": f"site-{index % 3}.test",
            "website_category": "work_application" if index % 3 == 0 else "general_web",
            "website_category_confidence": "manual",
            "website_surface": f"app-{index % 3}.test",
            "source": "fixture",
        }
        for index in range(30)
    ]
    path = tmp_path / "pool.jsonl"
    _write_jsonl(path, rows)
    first, audit = deterministic_stratified_sample([path], seed=7, per_task_category=2)
    second, _ = deterministic_stratified_sample([path], seed=7, per_task_category=2)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 8
    assert audit["examples"] == 30
    assert sum(audit["task_counts"].values()) == 30
    assert len(audit["websites"]) == 3


def test_build_dataset_dashboard_uses_candidate_manifests(tmp_path: Path) -> None:
    inventory_dir = tmp_path / "inventory"
    manifest = inventory_dir / "manifests" / "qa.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "id": "qa-1",
                "task": "qa",
                "domain": "docs.example",
                "website_surface": "docs.example",
                "website_category": "work_application",
                "website_category_confidence": "manual",
                "source": "fixture",
                "question": "What is selected?",
                "answer": "Budget",
            }
        ],
    )
    (inventory_dir / "inventory.json").write_text(
        json.dumps(
            {
                "identity_sha256": "fixture-id",
                "training": {"qa": {"manifest": "manifests/qa.jsonl"}},
            }
        )
    )
    output = tmp_path / "payload.json"
    payload = build_dataset_dashboard(inventory_dir=inventory_dir, output=output)
    assert payload["meta"]["provenance"] == "training candidates"
    assert payload["summary"]["examples"] == 1
    assert payload["records"][0]["question"] == "What is selected?"
    assert json.loads(output.read_text())["meta"]["inventory_identity"] == "fixture-id"
