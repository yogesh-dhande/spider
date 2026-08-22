#!/usr/bin/env python3
"""Verify that selected EXP005 scaling tiers are scientifically complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spider.scaling_audit import audit_scaling_completion, render_markdown


def parse_group(raw: str, seeds: list[int]) -> tuple[str, dict[str, object]]:
    try:
        size, step_text = raw.split("@", 1)
        step = int(step_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("group must have the form SIZE@STEP") from error
    return raw, {"size": size, "step": step, "seeds": seeds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    seeds = args.seed or [53, 59, 61]
    groups = dict(parse_group(raw, seeds) for raw in args.group)
    audit = audit_scaling_completion(
        json.loads(args.schedule.read_text(encoding="utf-8")),
        args.artifact_root,
        groups,
        expected_num_shards=args.num_shards,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "issues": len(audit["issues"])}))
    if args.require_complete and audit["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()

