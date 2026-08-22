from copy import deepcopy

import pytest

from spider.scaling_report import build_scaling_report, render_markdown


def _receipt(run_id: str, adapter: str | None, value: float) -> dict:
    suites = {}
    for suite in ("iid", "domain_balanced", "distribution_shift"):
        tasks = {
            "grounding": {"click_accuracy": value, "median_pixel_distance": 10.0},
            "action": {
                "action_name_accuracy": value,
                "exact_action_accuracy": value / 2,
                "click_inside_bbox_accuracy": value / 3,
            },
        }
        if suite != "distribution_shift":
            tasks["qa"] = {"answer_accuracy": value, "mean_token_f1": value}
        suites[suite] = {"merged": {"tasks": tasks}}
    return {
        "run_id": run_id,
        "model": "model",
        "model_revision": "revision",
        "adapter_sha256": adapter,
        "suites": suites,
    }


def test_scaling_report_computes_controls_deltas_and_seed_variability() -> None:
    baseline = _receipt("base", None, 0.2)
    starting = _receipt("start", "start-hash", 0.3)
    candidates = [
        {
            "label": "small seed 53",
            "size": "small",
            "seed": 53,
            "step": 625,
            "receipt": _receipt("small-53", "hash-53", 0.4),
        },
        {
            "label": "small seed 59",
            "size": "small",
            "seed": 59,
            "step": 625,
            "receipt": _receipt("small-59", "hash-59", 0.6),
        },
    ]

    report = build_scaling_report(baseline, starting, candidates)

    key = "iid/grounding/click_accuracy"
    assert report["candidates"][0]["delta_vs_untouched"][key] == pytest.approx(0.2)
    assert report["candidates"][0]["delta_vs_starting_control"][key] == pytest.approx(0.1)
    assert report["groups"]["small@625"]["metrics"][key]["mean"] == pytest.approx(0.5)
    assert report["groups"]["small@625"]["metrics"][key]["sample_std"] == pytest.approx(
        0.1414213562
    )
    markdown = render_markdown(report)
    assert "EXP002 starting adapter" in markdown
    assert "50.00 ± 14.14%" in markdown


def test_scaling_report_rejects_reused_candidate_adapter() -> None:
    baseline = _receipt("base", None, 0.2)
    starting = _receipt("start", "start-hash", 0.3)
    candidate = {
        "label": "candidate",
        "size": "small",
        "seed": 53,
        "step": 625,
        "receipt": _receipt("candidate", "same-hash", 0.4),
    }
    duplicate = deepcopy(candidate)
    duplicate.update(label="other", seed=59)
    with pytest.raises(ValueError, match="Adapter reused"):
        build_scaling_report(baseline, starting, [candidate, duplicate])
