import pytest

from spider.dashboard_images import required_images


def test_required_images_deduplicates_all_dashboard_tasks() -> None:
    payload = {
        "qa": {
            "records": [
                {"image": "/images/qa/a.jpg", "source_image": "images/shared/a.jpg"}
            ]
        },
        "grounding": {"records": [{"image": "images/shared/a.jpg"}]},
        "action": {"records": [{"image": "images/action/b.jpg"}]},
    }
    assert required_images(payload) == ["images/action/b.jpg", "images/shared/a.jpg"]


def test_required_images_rejects_archive_traversal() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        required_images({"qa": {"records": [{"image": "images/../secret"}]}})
