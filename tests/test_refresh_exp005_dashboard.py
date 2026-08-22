import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "refresh_exp005_dashboard.py"
SPEC = importlib.util.spec_from_file_location("refresh_exp005_dashboard", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _receipt(value: float) -> dict:
    return {
        "suites": {
            "iid": {
                "merged": {
                    "tasks": {
                        "qa": {"answer_accuracy": value, "mean_token_f1": value},
                        "grounding": {
                            "click_accuracy": value,
                            "median_pixel_distance": 10.0,
                        },
                        "action": {
                            "action_name_accuracy": value,
                            "exact_action_accuracy": value,
                            "click_inside_bbox_accuracy": value,
                        },
                    }
                }
            }
        }
    }


def _payload(value: float) -> dict:
    return {
        "qa": {"metrics": {"latest": {"exact_accuracy": value, "mean_token_f1": value}}},
        "grounding": {
            "metrics": {
                "latest": {"click_accuracy": value, "median_pixel_distance": 10.0}
            }
        },
        "action": {
            "metrics": {
                "latest": {
                    "action_name_accuracy": value,
                    "exact_action_accuracy": value,
                    "click_inside_bbox_accuracy": value,
                }
            }
        },
    }


def test_verify_dashboard_metrics_accepts_receipt_exactly() -> None:
    MODULE.verify_dashboard_metrics(_payload(0.5), _receipt(0.5), "latest", "iid")


def test_verify_dashboard_metrics_rejects_stale_payload() -> None:
    with pytest.raises(ValueError, match="Dashboard/receipt metric mismatch"):
        MODULE.verify_dashboard_metrics(_payload(0.4), _receipt(0.5), "latest", "iid")
