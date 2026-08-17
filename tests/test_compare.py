from spider.compare import compare_metrics


def test_compare_marks_distance_reduction_as_improvement() -> None:
    baseline = {
        "molmoweb": {
            "qa": {"answer_accuracy": 0.2, "mean_token_f1": 0.4},
            "grounding": {"click_accuracy": 0.3, "median_pixel_distance": 80},
        },
        "screenspot": {"grounding": {"click_accuracy": 0.1, "median_pixel_distance": 100}},
    }
    sft = {
        "molmoweb": {
            "qa": {"answer_accuracy": 0.5, "mean_token_f1": 0.6},
            "grounding": {"click_accuracy": 0.7, "median_pixel_distance": 30},
        },
        "screenspot": {"grounding": {"click_accuracy": 0.2, "median_pixel_distance": 90}},
    }
    rows = compare_metrics(baseline, sft)
    assert all(row["improved"] for row in rows)
