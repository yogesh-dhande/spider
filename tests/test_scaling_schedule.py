import pytest

from spider.scaling_schedule import build_schedule


def test_build_schedule_preserves_every_stage_validation_and_unique_run_ids(tmp_path) -> None:
    selection_sha = "ladder-file"
    plan = {
        "plan_sha256": "plan",
        "dataset_ladder_sha256": selection_sha,
        "jobs": [
            {
                "job_id": "small-53",
                "dataset_size": "small",
                "seed": 53,
                "identity_sha256": "a" * 64,
                "train_manifest_sha256": "small-manifest",
            },
            {
                "job_id": "large-59",
                "dataset_size": "large",
                "seed": 59,
                "identity_sha256": "b" * 64,
                "train_manifest_sha256": "large-manifest",
            },
        ],
    }
    ladder = {
        "identity_sha256": "dataset",
        "_file_sha256": selection_sha,
        "tiers": {
            "small": {"examples": 10_000, "sha256": "small-manifest"},
            "large": {"examples": 100_000, "sha256": "large-manifest"},
        },
    }

    result = build_schedule(
        plan,
        ladder,
        overrides={
            "small-53@500": {
                "training_run_id": "existing-training",
                "evaluation_run_id": "existing-evaluation",
            }
        },
    )

    assert result["job_count"] == 2
    assert result["training_stage_count"] == 15
    assert result["full_validation_campaign_count"] == 15
    assert [row["stop_step"] for row in result["jobs"][0]["stages"]] == [500, 625]
    assert result["jobs"][0]["stages"][0]["training_run_id"] == "existing-training"
    assert result["jobs"][1]["stages"][-1]["stop_step"] == 6250
    run_ids = {
        row[key]
        for job in result["jobs"]
        for row in job["stages"]
        for key in ("training_run_id", "evaluation_run_id")
    }
    assert len(run_ids) == 30
    assert all(row["full_validation_required"] for job in result["jobs"] for row in job["stages"])


def test_build_schedule_rejects_override_that_breaks_derived_instance_name(tmp_path) -> None:
    plan = {
        "plan_sha256": "plan",
        "dataset_ladder_sha256": "ladder-file",
        "jobs": [
            {
                "job_id": "small-53",
                "dataset_size": "small",
                "seed": 53,
                "identity_sha256": "a" * 64,
                "train_manifest_sha256": "small-manifest",
            }
        ],
    }
    ladder = {
        "identity_sha256": "dataset",
        "_file_sha256": "ladder-file",
        "tiers": {"small": {"examples": 10_000, "sha256": "small-manifest"}},
    }
    with pytest.raises(ValueError, match="derived GCE instance"):
        build_schedule(
            plan,
            ladder,
            overrides={
                "small-53@500": {
                    "training_run_id": "x" * 40,
                    "evaluation_run_id": "y" * 40,
                }
            },
        )
