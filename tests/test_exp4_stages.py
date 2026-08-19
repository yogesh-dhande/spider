from spider.exp4_stages import PLANNED_STEPS, STAGE_BOUNDS, STAGE_STEPS, STEPS


def test_exp4_stage_schedule_is_complete_and_timeout_safe() -> None:
    assert STAGE_STEPS == 125
    assert len(STEPS) == 15
    assert STEPS[-1] == PLANNED_STEPS == 1875
    assert STAGE_BOUNDS[0] == (0, 125)
    assert STAGE_BOUNDS[14] == (1750, 1875)
    assert all(stop - start == 125 for start, stop in STAGE_BOUNDS.values())
