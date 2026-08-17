from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

GRID_SIZE = 1000.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_bbox(bbox: Sequence[float], image_width: int, image_height: int) -> list[float]:
    if len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        raise ValueError("bbox must have four values and image dimensions must be positive")
    x1, y1, x2, y2 = map(float, bbox)
    return [
        clamp(x1 / image_width * GRID_SIZE, 0.0, GRID_SIZE),
        clamp(y1 / image_height * GRID_SIZE, 0.0, GRID_SIZE),
        clamp(x2 / image_width * GRID_SIZE, 0.0, GRID_SIZE),
        clamp(y2 / image_height * GRID_SIZE, 0.0, GRID_SIZE),
    ]


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four values")
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_to_pixels(point: Sequence[float], width: int, height: int) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("point must have two values")
    return (float(point[0]) / GRID_SIZE * width, float(point[1]) / GRID_SIZE * height)


def bbox_to_pixels(bbox: Sequence[float], width: int, height: int) -> list[float]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four values")
    return [
        float(bbox[0]) / GRID_SIZE * width,
        float(bbox[1]) / GRID_SIZE * height,
        float(bbox[2]) / GRID_SIZE * width,
        float(bbox[3]) / GRID_SIZE * height,
    ]


def point_in_bbox(point: Sequence[float], bbox: Sequence[float]) -> bool:
    x, y = map(float, point)
    x1, y1, x2, y2 = map(float, bbox)
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def pixel_distance(
    point: Sequence[float], target: Sequence[float], width: int, height: int
) -> float:
    px, py = point_to_pixels(point, width, height)
    tx, ty = point_to_pixels(target, width, height)
    return math.hypot(px - tx, py - ty)


def format_point_answer(point: Sequence[float], label: str) -> str:
    x = round(clamp(float(point[0]), 0.0, GRID_SIZE))
    y = round(clamp(float(point[1]), 0.0, GRID_SIZE))
    return json.dumps(
        [{"point_2d": [x, y], "label": label}], ensure_ascii=False, separators=(",", ":")
    )


def _walk_for_point(value: object) -> tuple[float, float] | None:
    if isinstance(value, dict):
        for key in ("point_2d", "point", "coordinates"):
            candidate = value.get(key)
            if (
                isinstance(candidate, Sequence)
                and not isinstance(candidate, (str, bytes))
                and len(candidate) >= 2
                and all(isinstance(v, (int, float)) for v in candidate[:2])
            ):
                return float(candidate[0]), float(candidate[1])
        for candidate in value.values():
            result = _walk_for_point(candidate)
            if result is not None:
                return result
    elif isinstance(value, list):
        for candidate in value:
            result = _walk_for_point(candidate)
            if result is not None:
                return result
    return None


def parse_point(text: str) -> tuple[float, float] | None:
    """Parse Qwen JSON first, with a permissive coordinate-pair fallback."""
    stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = _walk_for_point(json.loads(stripped))
        if result is not None:
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    tagged = re.search(
        r"(?:point_2d|point|coordinates)\s*[\"']?\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)"
        r"\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        stripped,
        flags=re.IGNORECASE,
    )
    if tagged:
        return float(tagged.group(1)), float(tagged.group(2))

    pair = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", stripped)
    if pair:
        return float(pair.group(1)), float(pair.group(2))
    return None
