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


def test_training_stage_preserves_effective_batch() -> None:
    try:
        MODULE.create_training_stage("test", "zone", "abc", 375, 500, "4h", 2, 4)
    except ValueError as error:
        assert "effective batch size 16" in str(error)
    else:
        raise AssertionError("invalid effective batch was accepted")


def test_validation_rejects_unknown_role() -> None:
    try:
        MODULE.create_validation_shard("test", "other", "zone", "abc", 500, "4h")
    except ValueError as error:
        assert "action or perception" in str(error)
    else:
        raise AssertionError("unknown validation role was accepted")


def test_speed_benchmark_preserves_effective_batch() -> None:
    try:
        MODULE.create_speed_benchmark("test", "zone", "abc", 250, 20, 2, 4, "2h")
    except ValueError as error:
        assert "effective batch size 16" in str(error)
    else:
        raise AssertionError("invalid effective batch was accepted")


def test_speed_benchmark_rejects_non_positive_parameters() -> None:
    try:
        MODULE.create_speed_benchmark("test", "zone", "abc", 250, 0, 2, 8, "2h")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive benchmark length was accepted")


def test_validation_rejects_unknown_accelerator() -> None:
    try:
        MODULE.create_validation_shard(
            "test", "perception", "zone", "abc", 500, "4h", "other"
        )
    except ValueError as error:
        assert "l4 or t4" in str(error)
    else:
        raise AssertionError("unknown validation accelerator was accepted")
