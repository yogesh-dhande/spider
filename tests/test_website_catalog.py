from pathlib import Path

from spider.website_catalog import (
    annotate_website,
    build_website_catalog,
    classify_website,
    website_surface,
    write_website_catalog,
)


def test_google_work_surfaces_remain_distinct() -> None:
    assert website_surface("https://mail.google.com/mail/u/0/") == "mail.google.com"
    assert website_surface("https://calendar.google.com/calendar/u/0/") == "calendar.google.com"
    assert website_surface("https://docs.google.com/document/d/123") == "docs.google.com"
    assert website_surface("https://docs.google.com/spreadsheets/d/123") == "sheets.google.com"
    assert website_surface("https://docs.google.com/presentation/d/123") == "slides.google.com"


def test_work_app_classification_is_auditable_and_overridable() -> None:
    inferred = annotate_website(
        {"domain": "google.com", "url": "https://calendar.google.com/calendar/u/0/"}
    )
    assert inferred["website_category"] == "work_application"
    assert inferred["website_category_confidence"] == "heuristic"
    overridden = classify_website(
        domain="example.test",
        surface="app.example.test",
        url="https://app.example.test",
        rules=[{"pattern": "app.example.test", "category": "work_application"}],
    )
    assert overridden["website_category"] == "work_application"
    assert overridden["website_category_confidence"] == "manual"


def test_catalog_summarizes_tasks_surfaces_and_actions(tmp_path: Path) -> None:
    records = [
        annotate_website(
            {
                "id": "a",
                "task": "action",
                "source": "trajectory",
                "trajectory_id": "t1",
                "domain": "google.com",
                "url": "https://docs.google.com/document/d/1",
                "target_action": {"name": "keyboard_type"},
            }
        ),
        annotate_website(
            {
                "id": "b",
                "task": "qa",
                "source": "qa",
                "image": "locator://b",
                "domain": "google.com",
                "url": "https://calendar.google.com/calendar/u/0/",
            }
        ),
    ]
    rows = build_website_catalog(records)
    assert rows[0]["domain"] == "google.com"
    assert rows[0]["tasks"] == {"action": 1, "qa": 1}
    assert set(rows[0]["surfaces"]) == {"calendar.google.com", "docs.google.com"}
    assert rows[0]["category_confidence"] == "heuristic"
    assert rows[0]["manual_review_required"] is False
    assert rows[0]["category_rules"] == {"url-token:calendar": 1, "url-token:docs": 1}
    summary = write_website_catalog(tmp_path, records)
    assert summary["websites"] == 1
    assert (tmp_path / "website_catalog.csv").is_file()
