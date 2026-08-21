"""Auditable website classification and cataloging for browser datasets."""

from __future__ import annotations

import csv
import fnmatch
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from spider.diversity import sampling_unit

APPLICATION_CATEGORIES = {
    "work_application",
    "transactional_application",
    "service_application",
}

WORK_APPLICATION_HINTS = {
    "airtable",
    "asana",
    "atlassian",
    "calendar",
    "canva",
    "clickup",
    "confluence",
    "docs",
    "drive",
    "dropbox",
    "figma",
    "github",
    "gitlab",
    "gmail",
    "hubspot",
    "jira",
    "linear",
    "mail",
    "monday",
    "notion",
    "office",
    "outlook",
    "salesforce",
    "sheets",
    "slack",
    "slides",
    "teams",
    "trello",
    "zoom",
}
TRANSACTION_HINTS = {
    "airbnb",
    "amazon",
    "booking",
    "doordash",
    "ebay",
    "expedia",
    "flight",
    "hotel",
    "instacart",
    "shop",
    "store",
    "travel",
    "walmart",
}
SERVICE_HINTS = {
    "bank",
    "clinic",
    "college",
    "coursera",
    "edu",
    "finance",
    "gov",
    "health",
    "insurance",
    "school",
    "university",
}
CONTENT_HINTS = {
    "blog",
    "cnn",
    "encyclopedia",
    "forbes",
    "magazine",
    "news",
    "nytimes",
    "wikipedia",
}


def website_surface(url: str, website: str = "") -> str:
    """Keep app surfaces distinct without weakening registrable-domain leakage rules."""
    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.strip("/").lower()
    if host == "docs.google.com":
        product = path.split("/", 1)[0] if path else "docs"
        aliases = {"document": "docs", "spreadsheets": "sheets", "presentation": "slides"}
        return f"{aliases.get(product, product)}.google.com"
    if host:
        return host
    return str(website or "unknown").strip().lower() or "unknown"


def classify_website(
    *,
    domain: str,
    surface: str,
    url: str,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify with transparent overrides and conservative URL heuristics."""
    haystack = f"{domain} {surface} {url}".lower()
    for index, rule in enumerate(rules or []):
        pattern = str(rule.get("pattern") or "").lower()
        if pattern and (
            fnmatch.fnmatch(domain, pattern)
            or fnmatch.fnmatch(surface, pattern)
            or pattern in haystack
        ):
            return {
                "website_category": str(rule["category"]),
                "website_category_confidence": "manual",
                "website_category_rule": f"override:{index}:{pattern}",
            }
    hints = (
        (WORK_APPLICATION_HINTS, "work_application"),
        (TRANSACTION_HINTS, "transactional_application"),
        (SERVICE_HINTS, "service_application"),
        (CONTENT_HINTS, "content_reference"),
    )
    tokens = {
        part for part in haystack.replace("/", ".").replace("-", ".").split(".") if part
    }
    for vocabulary, category in hints:
        matches = sorted(tokens & vocabulary)
        if matches:
            return {
                "website_category": category,
                "website_category_confidence": "heuristic",
                "website_category_rule": f"url-token:{matches[0]}",
            }
    return {
        "website_category": "general_web",
        "website_category_confidence": "unknown",
        "website_category_rule": "fallback",
    }


def annotate_website(
    record: dict[str, Any], rules: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    annotated = dict(record)
    url = str(annotated.get("url") or "")
    parsed = urlparse(url)
    annotated["hostname"] = (parsed.hostname or "").lower().removeprefix("www.")
    annotated["website_surface"] = website_surface(url, str(annotated.get("website") or ""))
    annotated.update(
        classify_website(
            domain=str(annotated.get("domain") or "unknown"),
            surface=annotated["website_surface"],
            url=url,
            rules=rules,
        )
    )
    annotated["application_focused"] = (
        annotated["website_category"] in APPLICATION_CATEGORIES
    )
    return annotated


class WebsiteCatalogAccumulator:
    """Incrementally collect catalog statistics without retaining all examples."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "examples": 0,
                "units": set(),
                "tasks": Counter(),
                "sources": Counter(),
                "surfaces": Counter(),
                "actions": Counter(),
                "categories": Counter(),
                "category_confidences": Counter(),
                "category_rules": Counter(),
            }
        )

    def add(self, record: dict[str, Any]) -> None:
        domain = str(record.get("domain") or "unknown")
        state = self._rows[domain]
        state["examples"] += 1
        state["units"].add(sampling_unit(record))
        state["tasks"][str(record["task"])] += 1
        state["sources"][str(record.get("source") or "unknown")] += 1
        state["surfaces"][str(record.get("website_surface") or "unknown")] += 1
        state["categories"][str(record.get("website_category") or "general_web")] += 1
        state["category_confidences"][
            str(record.get("website_category_confidence") or "unknown")
        ] += 1
        state["category_rules"][str(record.get("website_category_rule") or "unknown")] += 1
        if record.get("task") == "action":
            name = str((record.get("target_action") or {}).get("name") or "unknown")
            state["actions"][name] += 1

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for domain, state in self._rows.items():
            categories = state["categories"]
            category = min(categories, key=lambda value: (-categories[value], value))
            confidences = state["category_confidences"]
            confidence = min(confidences, key=lambda value: (-confidences[value], value))
            rows.append(
                {
                    "domain": domain,
                    "website_category": category,
                    "application_focused": category in APPLICATION_CATEGORIES,
                    "category_confidence": confidence,
                    "manual_review_required": confidence == "unknown" or len(categories) > 1,
                    "examples": state["examples"],
                    "sampling_units": len(state["units"]),
                    "tasks": dict(sorted(state["tasks"].items())),
                    "sources": dict(sorted(state["sources"].items())),
                    "surfaces": dict(
                        sorted(
                            state["surfaces"].items(), key=lambda pair: (-pair[1], pair[0])
                        )[:20]
                    ),
                    "actions": dict(
                        sorted(state["actions"].items(), key=lambda pair: (-pair[1], pair[0]))
                    ),
                    "category_votes": dict(
                        sorted(categories.items(), key=lambda pair: (-pair[1], pair[0]))
                    ),
                    "category_rules": dict(
                        sorted(
                            state["category_rules"].items(),
                            key=lambda pair: (-pair[1], pair[0]),
                        )[:20]
                    ),
                }
            )
        return sorted(rows, key=lambda row: (-row["examples"], row["domain"]))


def build_website_catalog(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    accumulator = WebsiteCatalogAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.rows()


def write_website_catalog(
    output_dir: Path,
    records: Iterable[dict[str, Any]] | None = None,
    *,
    accumulator: WebsiteCatalogAccumulator | None = None,
) -> dict[str, Any]:
    if (records is None) == (accumulator is None):
        raise ValueError("Provide exactly one of records or accumulator")
    rows = accumulator.rows() if accumulator is not None else build_website_catalog(records or [])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "website_catalog.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "website_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "website_category",
                "application_focused",
                "category_confidence",
                "manual_review_required",
                "examples",
                "sampling_units",
                "tasks",
                "sources",
                "surfaces",
                "actions",
                "category_rules",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if isinstance(row[key], dict)
                    else row[key]
                    for key in writer.fieldnames
                }
            )
    categories = Counter(row["website_category"] for row in rows)
    return {
        "websites": len(rows),
        "application_focused_websites": sum(bool(row["application_focused"]) for row in rows),
        "category_counts": dict(sorted(categories.items())),
        "json": "website_catalog.json",
        "csv": "website_catalog.csv",
    }
