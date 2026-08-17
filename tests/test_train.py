from spider.train import training_step_plan


def test_chunk_plan_keeps_full_scheduler_horizon() -> None:
    planned, stop = training_step_plan(
        examples=30_000,
        per_device_batch=1,
        gradient_accumulation=16,
        world_size=1,
        epochs=1.0,
        current_step=500,
        additional_steps=500,
    )
    assert planned == 1875
    assert stop == 1000


def test_chunk_plan_caps_stop_at_epoch_target() -> None:
    planned, stop = training_step_plan(30_000, 1, 16, 1, 1.0, 1750, 500)
    assert planned == stop == 1875
