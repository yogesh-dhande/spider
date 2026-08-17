from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spider.coordinates import bbox_to_pixels, point_to_pixels
from spider.prepare import stable_int


def _annotated_image(record: dict[str, Any], source: Path, target: Path) -> None:
    with Image.open(source) as image_handle:
        image = image_handle.convert("RGB")
    if record["task"] == "grounding":
        draw = ImageDraw.Draw(image)
        bbox = bbox_to_pixels(record["bbox_normalized"], *image.size)
        draw.rectangle(bbox, outline=(0, 220, 80), width=4)
        target_point = point_to_pixels(record["target_point_normalized"], *image.size)
        radius = 7
        draw.ellipse(
            [
                target_point[0] - radius,
                target_point[1] - radius,
                target_point[0] + radius,
                target_point[1] + radius,
            ],
            fill=(0, 220, 80),
        )
        if record.get("parsed_point") is not None:
            predicted = point_to_pixels(record["parsed_point"], *image.size)
            draw.line(
                [predicted[0] - 10, predicted[1], predicted[0] + 10, predicted[1]],
                fill=(235, 40, 40),
                width=4,
            )
            draw.line(
                [predicted[0], predicted[1] - 10, predicted[0], predicted[1] + 10],
                fill=(235, 40, 40),
                width=4,
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((960, 640), Image.Resampling.LANCZOS)
    image.save(target, format="JPEG", quality=88, optimize=True)


def create_failure_report(
    records: list[dict[str, Any]], data_dir: Path, report_dir: Path, examples_per_bucket: int
) -> Path:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result = record.get("error_category") or f"{record['task']}_correct"
        bucket = f"{record.get('benchmark', 'unknown')}_{result}"
        buckets[str(bucket)].append(record)
    selected: dict[str, list[dict[str, Any]]] = {}
    for bucket, items in buckets.items():
        items.sort(key=lambda item: stable_int(item["id"], bucket))
        limit = min(examples_per_bucket, 10) if bucket.endswith("_correct") else examples_per_bucket
        selected[bucket] = items[:limit]

    report_dir.mkdir(parents=True, exist_ok=True)
    assets = report_dir / "assets"
    sections: list[str] = []
    for bucket, items in sorted(selected.items()):
        cards: list[str] = []
        for record in items:
            asset = assets / f"{record['id']}.jpg"
            _annotated_image(record, data_dir / record["image"], asset)
            details = {
                "question": record["question"],
                "answer": record["answer"],
                "prediction": record.get("prediction", ""),
                "domain": record.get("domain", ""),
                "question_type": record.get("question_type"),
                "bbox_normalized": record.get("bbox_normalized"),
                "parsed_point": record.get("parsed_point"),
                "pixel_distance": record.get("pixel_distance"),
            }
            cards.append(
                '<article class="card">'
                f'<img src="assets/{html.escape(asset.name)}" loading="lazy">'
                f"<pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre>"
                "</article>"
            )
        sections.append(
            f"<h2>{html.escape(bucket)} ({len(buckets[bucket])} total)</h2>"
            f'<section class="grid">{"".join(cards)}</section>'
        )

    output = report_dir / "failures.html"
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Spider Experiment 1 predictions</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#f4f5f7;color:#17191c}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}"
        ".card{background:white;border-radius:10px;padding:12px;box-shadow:0 1px 5px #0002}"
        "img{width:100%;height:auto;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "h2{margin-top:36px}</style></head><body><h1>Experiment 1 prediction gallery</h1>"
        "<p>Grounding overlays: green box/dot = target; red cross = prediction.</p>"
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    return output
