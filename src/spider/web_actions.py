from __future__ import annotations

import json
import math
from typing import Any

COORDINATE_ACTIONS = {"click", "scroll_at", "mouse_drag_and_drop"}


class WebActionError(ValueError):
    """Raised when a MolmoWeb action cannot be normalized or parsed."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WebActionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WebActionError(f"{name} must be finite")
    return result


def _coordinate(value: Any, size: int, name: str) -> float:
    if size <= 0:
        raise WebActionError(f"invalid image dimension for {name}")
    return round(max(0.0, min(100.0, _number(value, name) / size * 100.0)), 1)


def _delta(value: Any, size: int, name: str) -> float:
    if size <= 0:
        raise WebActionError(f"invalid image dimension for {name}")
    return round(_number(value, name) / size * 100.0, 1)


def normalize_action_output(
    action_output: dict[str, Any], image_width: int, image_height: int
) -> tuple[dict[str, Any], list[float] | None]:
    """Convert a raw MolmoWeb action to its published 0–100 text representation.

    Clicks with a box use its deterministic center. The returned box is xyxy in original pixels
    and is retained only for evaluation.
    """
    if not isinstance(action_output, dict):
        raise WebActionError("action_output must be an object")
    name = str(action_output.get("action_name") or "").strip()
    raw = action_output.get("action")
    if not name or not isinstance(raw, dict):
        raise WebActionError("action_name and action are required")
    action: dict[str, Any] = {"name": name}
    bbox: list[float] | None = None

    if name == "click":
        if "bbox" in raw and raw["bbox"] is not None:
            values = raw["bbox"]
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise WebActionError("click bbox must be [x, y, width, height]")
            x, y, width, height = (_number(value, "bbox") for value in values)
            if width <= 0 or height <= 0:
                raise WebActionError("click bbox must have positive area")
            bbox = [x, y, x + width, y + height]
            click_x, click_y = x + width / 2.0, y + height / 2.0
        else:
            click_x = _number(raw.get("x"), "x")
            click_y = _number(raw.get("y"), "y")
        action.update(
            {
                "x": _coordinate(click_x, image_width, "x"),
                "y": _coordinate(click_y, image_height, "y"),
                "button": str(raw.get("button") or ""),
                "click_type": str(raw.get("click_type") or ""),
            }
        )
    elif name == "scroll":
        action.update(
            {
                "delta_x": _delta(raw.get("delta_x"), image_width, "delta_x"),
                "delta_y": _delta(raw.get("delta_y"), image_height, "delta_y"),
            }
        )
    elif name == "scroll_at":
        action.update(
            {
                "x": _coordinate(raw.get("x"), image_width, "x"),
                "y": _coordinate(raw.get("y"), image_height, "y"),
                "delta_x": _delta(raw.get("delta_x"), image_width, "delta_x"),
                "delta_y": _delta(raw.get("delta_y"), image_height, "delta_y"),
            }
        )
    elif name == "mouse_drag_and_drop":
        for prefix in ("from", "to"):
            action[f"{prefix}_x"] = _coordinate(
                raw.get(f"{prefix}_x"), image_width, f"{prefix}_x"
            )
            action[f"{prefix}_y"] = _coordinate(
                raw.get(f"{prefix}_y"), image_height, f"{prefix}_y"
            )
    else:
        for key, value in raw.items():
            if key not in {"bbox"}:
                action[key] = value
    return action, bbox


def action_answer(thought: str, action: dict[str, Any]) -> str:
    return json.dumps(
        {"thought": thought.strip(), "action": action},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_action_response(raw: str) -> dict[str, Any]:
    """Strictly parse one rationale-plus-action JSON response."""
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise WebActionError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict) or set(payload) != {"thought", "action"}:
        raise WebActionError("response must contain exactly thought and action")
    if not isinstance(payload["thought"], str) or not isinstance(payload["action"], dict):
        raise WebActionError("thought must be text and action must be an object")
    action = payload["action"]
    if not isinstance(action.get("name"), str) or not action["name"].strip():
        raise WebActionError("action.name is required")
    for key in (
        "x",
        "y",
        "from_x",
        "from_y",
        "to_x",
        "to_y",
    ):
        if key in action:
            value = _number(action[key], key)
            if not 0 <= value <= 100:
                raise WebActionError(f"{key} must be in [0, 100]")
    for key in ("delta_x", "delta_y"):
        if key in action:
            _number(action[key], key)
    return payload


def normalized_bbox(bbox_pixels: list[float] | None, width: int, height: int) -> list[float] | None:
    if bbox_pixels is None:
        return None
    if width <= 0 or height <= 0:
        raise WebActionError("image dimensions must be positive")
    x1, y1, x2, y2 = bbox_pixels
    return [x1 / width, y1 / height, x2 / width, y2 / height]
