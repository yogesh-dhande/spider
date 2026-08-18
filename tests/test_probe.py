import pytest

from spider.probe import primary_probe_metrics


def test_primary_probe_metrics_extracts_comparable_values() -> None:
    metrics = {
        "molmoweb": {
            "qa": {"answer_accuracy": 0.25, "mean_token_f1": 0.5},
            "grounding": {
                "click_accuracy": 0.6,
                "parse_rate": 0.99,
                "median_pixel_distance": 31.0,
            },
        }
    }
    assert primary_probe_metrics(metrics) == {
        "qa_answer_accuracy": 0.25,
        "qa_mean_token_f1": 0.5,
        "grounding_click_accuracy": 0.6,
        "grounding_parse_rate": 0.99,
        "grounding_median_pixel_distance": 31.0,
    }


def test_primary_probe_metrics_requires_both_tasks() -> None:
    with pytest.raises(KeyError):
        primary_probe_metrics({"molmoweb": {"qa": {}}})
