"""Shared timeout-safe stage schedule for EXP004."""

from __future__ import annotations

PLANNED_STEPS = 1875
STAGE_STEPS = 125
STEPS = tuple(range(STAGE_STEPS, PLANNED_STEPS + 1, STAGE_STEPS))
STAGE_BOUNDS = {
    stage: (0 if stage == 0 else STEPS[stage - 1], stop)
    for stage, stop in enumerate(STEPS)
}
