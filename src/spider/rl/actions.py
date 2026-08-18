from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

ACTION_FIELDS = {
    "click": {"action", "x", "y"},
    "type": {"action", "text"},
    "scroll": {"action", "direction", "amount"},
    "go_back": {"action"},
    "done": {"action", "result"},
}


class ActionParseError(ValueError):
    """Raised when a policy response is not one strict browser action."""


@dataclass(frozen=True)
class BrowserAction:
    action: str
    x: float | None = None
    y: float | None = None
    text: str | None = None
    direction: str | None = None
    amount: float | None = None
    result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ActionParseError(f"{field} must be finite")
    return number


def parse_action(raw: str) -> BrowserAction:
    """Parse one strict JSON browser action using normalized click coordinates."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ActionParseError("action must be a JSON object")
    name = payload.get("action")
    if name not in ACTION_FIELDS:
        raise ActionParseError(f"unsupported action: {name!r}")
    expected_fields = ACTION_FIELDS[name]
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ActionParseError(f"invalid fields for {name}: missing={missing}, extra={extra}")

    if name == "click":
        x = _finite_number(payload["x"], "x")
        y = _finite_number(payload["y"], "y")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ActionParseError("click coordinates must be normalized to [0, 1]")
        return BrowserAction(action=name, x=x, y=y)
    if name == "type":
        text = payload["text"]
        if not isinstance(text, str) or not text:
            raise ActionParseError("type.text must be a non-empty string")
        return BrowserAction(action=name, text=text)
    if name == "scroll":
        direction = payload["direction"]
        if direction not in {"up", "down"}:
            raise ActionParseError("scroll.direction must be 'up' or 'down'")
        amount = _finite_number(payload["amount"], "amount")
        if not 0.0 < amount <= 1.0:
            raise ActionParseError("scroll.amount must be in (0, 1]")
        return BrowserAction(action=name, direction=direction, amount=amount)
    if name == "done":
        result = payload["result"]
        if not isinstance(result, str):
            raise ActionParseError("done.result must be a string")
        return BrowserAction(action=name, result=result)
    return BrowserAction(action=name)
