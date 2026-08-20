import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gcloud_exp004.py"
SPEC = importlib.util.spec_from_file_location("gcloud_exp004", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_managed_filter_is_narrowly_scoped() -> None:
    assert MODULE.MANAGED_FILTER == (
        "labels.spider-managed=true AND labels.spider-experiment=exp004"
    )


def test_zone_name_accepts_full_resource_url() -> None:
    assert MODULE.zone_name({"zone": "https://compute.googleapis.com/zones/us-west1-b"}) == (
        "us-west1-b"
    )


def test_training_stage_rejects_invalid_bounds() -> None:
    try:
        MODULE.create_training_stage("test", "us-west1-b", "abc", 375, 375, "6h")
    except ValueError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("invalid stage bounds were accepted")
