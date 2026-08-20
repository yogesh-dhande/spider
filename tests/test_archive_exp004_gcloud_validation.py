import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "archive_exp004_gcloud_validation.py"
SPEC = importlib.util.spec_from_file_location("archive_exp004_gcloud_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retain_existing_action_records(tmp_path: Path) -> None:
    image = tmp_path / "images/action/one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    payload = {
        "meta": {"display_examples": 2, "unique_screenshots": 2},
        "records": [
            {"id": "one", "image": "/images/action/one.jpg"},
            {"id": "two", "image": "/images/action/two.jpg"},
        ],
    }

    retained = MODULE.retain_existing_action_records(payload, tmp_path, limit=1)

    assert retained == 1
    assert [record["id"] for record in payload["records"]] == ["one"]
    assert payload["meta"]["display_examples"] == 1
