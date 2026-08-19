from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spider.coordinates import bbox_to_pixels
from spider.prepare import stable_int, write_jsonl
from spider.web_actions import WebActionError, parse_action_response


def action_error_category(record: dict[str, Any]) -> str:
    """Map a scored browser action to a diagnostic error family."""
    if not record.get("action_parse_valid"):
        return "output_format"
    if not record.get("action_name_correct"):
        return "semantic_action"
    if (
        record.get("target_action", {}).get("name") == "click"
        and record.get("bbox_normalized")
        and not record.get("click_inside_bbox")
    ):
        return "spatial_grounding"
    if not record.get("action_arguments_correct"):
        return "action_arguments"
    return "correct"


def select_representative_action_records(
    records: list[dict[str, Any]], examples_per_bucket: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select a stable, bounded sample from each action diagnostic bucket."""
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = action_error_category(record)
        buckets[bucket].append({**record, "error_category": bucket})
    selected: list[dict[str, Any]] = []
    for bucket, items in sorted(buckets.items()):
        items.sort(key=lambda item: stable_int(str(item["id"]), bucket))
        limit = min(examples_per_bucket, 10) if bucket == "correct" else examples_per_bucket
        selected.extend(items[:limit])
    return selected, dict(sorted(Counter(action_error_category(item) for item in records).items()))


def _predicted_click(record: dict[str, Any]) -> tuple[float, float] | None:
    try:
        action = parse_action_response(str(record.get("prediction") or ""))["action"]
        if action.get("name") != "click":
            return None
        return float(action["x"]), float(action["y"])
    except (KeyError, TypeError, ValueError, WebActionError):
        return None


def _annotate(record: dict[str, Any], source: Path, target: Path) -> None:
    with Image.open(source) as image_handle:
        image = image_handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = record.get("bbox_normalized")
    if bbox:
        draw.rectangle(bbox_to_pixels(bbox, *image.size), outline=(0, 220, 80), width=4)
    target_action = record.get("target_action") or {}
    if target_action.get("name") == "click":
        tx = float(target_action["x"]) / 100.0 * image.width
        ty = float(target_action["y"]) / 100.0 * image.height
        draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), fill=(0, 220, 80))
    predicted = _predicted_click(record)
    if predicted is not None:
        px = predicted[0] / 100.0 * image.width
        py = predicted[1] / 100.0 * image.height
        draw.line((px - 10, py, px + 10, py), fill=(235, 40, 40), width=4)
        draw.line((px, py - 10, px, py + 10), fill=(235, 40, 40), width=4)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((960, 640), Image.Resampling.LANCZOS)
    image.save(target, format="JPEG", quality=88, optimize=True)


def create_action_failure_report(
    records: list[dict[str, Any]],
    data_dir: Path,
    report_dir: Path,
    examples_per_bucket: int,
) -> Path:
    """Write deterministic action examples, counts, and a visual failure gallery."""
    selected, counts = select_representative_action_records(records, examples_per_bucket)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "representative_predictions.jsonl", selected)
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "examples": len(records),
                "selected_examples": len(selected),
                "error_category_counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped[str(record["error_category"])].append(record)
    sections: list[str] = []
    assets = report_dir / "assets"
    for bucket, items in sorted(grouped.items()):
        cards: list[str] = []
        for record in items:
            asset = assets / f"{record['id']}.jpg"
            _annotate(record, data_dir / str(record["image"]), asset)
            details = {
                "instruction": record.get("question"),
                "target_action": record.get("target_action"),
                "prediction": record.get("prediction"),
                "domain": record.get("domain"),
                "click_distance_px": record.get("click_distance_px"),
                "click_inside_bbox": record.get("click_inside_bbox"),
                "error_category": bucket,
            }
            cards.append(
                '<article class="card">'
                f'<img src="assets/{html.escape(asset.name)}" loading="lazy">'
                f"<pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre>"
                "</article>"
            )
        sections.append(
            f"<h2>{html.escape(bucket)} ({counts[bucket]} total)</h2>"
            f'<section class="grid">{"".join(cards)}</section>'
        )
    output = report_dir / "failures.html"
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>EXP004 browser-action failures</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#f4f5f7;color:#17191c}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}"
        ".card{background:white;border-radius:10px;padding:12px;box-shadow:0 1px 5px #0002}"
        "img{width:100%;height:auto;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "h2{margin-top:36px}</style></head><body><h1>EXP004 action failure gallery</h1>"
        "<p>Green box/dot = target; red cross = predicted click.</p>"
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    return output
