import json
from pathlib import Path

import pytest

from spider.scaling_job import (
    active_gpu_regions,
    load_scaling_job,
    parse_receipt_overrides,
    size_label,
)


def test_load_scaling_job_validates_contiguous_full_validation(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    path.write_text(
        json.dumps(
            {
                "kind": "exp005_scaling_execution_schedule",
                "jobs": [
                    {
                        "job_id": "job",
                        "dataset_size": "small",
                        "seed": 53,
                        "total_optimizer_steps": 625,
                        "stages": [
                            {
                                "start_step": 0,
                                "stop_step": 500,
                                "training_run_id": "train-1",
                                "evaluation_run_id": "eval-1",
                                "full_validation_required": True,
                            },
                            {
                                "start_step": 500,
                                "stop_step": 625,
                                "training_run_id": "train-2",
                                "evaluation_run_id": "eval-2",
                                "full_validation_required": True,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    job = load_scaling_job(path, "job")
    assert job.total_optimizer_steps == 625
    assert [stage.stop_step for stage in job.stages] == [500, 625]


def test_receipt_overrides_and_size_labels() -> None:
    assert parse_receipt_overrides(["500=legacy.json"]) == {500: Path("legacy.json")}
    assert size_label("large") == "100K"
    with pytest.raises(ValueError, match="STEP=PATH"):
        parse_receipt_overrides(["broken"])


def test_active_gpu_regions_filters_terminal_and_cpu_instances() -> None:
    instances = [
        {
            "status": "RUNNING",
            "zone": "zones/us-east1-b",
            "machineType": "machineTypes/g2-standard-8",
        },
        {
            "status": "TERMINATED",
            "zone": "zones/europe-west4-a",
            "machineType": "machineTypes/g2-standard-8",
        },
        {
            "status": "RUNNING",
            "zone": "zones/us-central1-a",
            "machineType": "machineTypes/e2-standard-4",
        },
    ]
    assert active_gpu_regions(instances) == {"us-east1"}
