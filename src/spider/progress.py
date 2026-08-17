from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any


class LineProgress:
    """Sparse, one-record-per-line progress suitable for batch notebook logs."""

    def __init__(
        self,
        stage: str,
        total: int | None = None,
        every_items: int = 500,
        every_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stage = stage
        self.total = total
        self.every_items = every_items
        self.every_seconds = every_seconds
        self.clock = clock
        self.started = clock()
        self.last_log = self.started
        self.last_logged_items = 0
        self.completed = 0
        self._emit("start")

    def _emit(self, event: str, **fields: Any) -> None:
        now = self.clock()
        elapsed = max(now - self.started, 0.0)
        rate = self.completed / elapsed if elapsed > 0 else None
        eta = None
        if self.total is not None and rate and self.completed < self.total:
            eta = (self.total - self.completed) / rate
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "event": event,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "percent": (
                round(100 * self.completed / self.total, 2) if self.total else None
            ),
            "elapsed_seconds": round(elapsed, 1),
            "items_per_second": round(rate, 3) if rate is not None else None,
            "eta_seconds": round(eta, 1) if eta is not None else None,
            **fields,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        self.last_log = now
        self.last_logged_items = self.completed

    def update(self, count: int = 1, **fields: Any) -> None:
        self.completed += count
        now = self.clock()
        item_interval_reached = self.completed - self.last_logged_items >= self.every_items
        time_interval_reached = now - self.last_log >= self.every_seconds
        if item_interval_reached or time_interval_reached:
            self._emit("progress", **fields)

    def close(self, status: str = "complete", **fields: Any) -> None:
        self._emit(status, **fields)
