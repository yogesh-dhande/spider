from spider.metrics import normalize_answer, score_records, token_f1


def test_answer_normalization() -> None:
    assert normalize_answer("The Search Button!") == "search button"
    assert token_f1("red button", "the red search button") == 0.8


def test_scores_tasks_and_benchmarks_separately() -> None:
    records = [
        {
            "id": "qa-1",
            "benchmark": "molmoweb",
            "task": "qa",
            "answer": "The Search Button",
            "prediction": "search button",
            "question_type": "OCR",
        },
        {
            "id": "ground-1",
            "benchmark": "molmoweb",
            "task": "grounding",
            "answer": "",
            "prediction": '[{"point_2d":[500,500]}]',
            "bbox_normalized": [400, 400, 600, 600],
            "target_point_normalized": [500, 500],
            "image_width": 1280,
            "image_height": 720,
        },
        {
            "id": "screen-1",
            "benchmark": "screenspot",
            "task": "grounding",
            "answer": "",
            "prediction": "not valid",
            "bbox_normalized": [400, 400, 600, 600],
            "target_point_normalized": [500, 500],
            "image_width": 1280,
            "image_height": 720,
        },
    ]
    scored, metrics = score_records(records, [25, 50])
    assert metrics["molmoweb"]["qa"]["answer_accuracy"] == 1.0
    assert metrics["molmoweb"]["grounding"]["click_accuracy"] == 1.0
    assert metrics["screenspot"]["grounding"]["parse_rate"] == 0.0
    assert scored[2]["error_category"] == "output_format"
