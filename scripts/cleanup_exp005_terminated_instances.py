#!/usr/bin/env python3
"""Delete only terminated EXP005-managed worker instances."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcloud_exp005 as cloud


def terminated_targets(instances: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    targets = []
    for instance in instances:
        labels = instance.get("labels") or {}
        role = str(labels.get("spider-role", ""))
        if instance.get("status") != "TERMINATED" or role == "controller":
            continue
        if labels.get("spider-managed") != "true" or labels.get("spider-experiment") != "exp005":
            raise ValueError(f"Managed-instance query returned an unsafe target: {instance}")
        targets.append(
            (
                str(instance["name"]),
                str(instance["zone"]).rsplit("/", 1)[-1],
                role,
            )
        )
    return sorted(targets)


def delete_target(target: tuple[str, str, str]) -> dict[str, str]:
    name, zone, role = target
    subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            f"--project={cloud.PROJECT}",
            f"--zone={zone}",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"name": name, "zone": zone, "role": role}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    targets = terminated_targets(cloud.managed_instances())
    if not args.execute:
        print(json.dumps({"event": "exp005_terminated_gc_dry_run", "targets": targets}))
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        deleted = list(pool.map(delete_target, targets))
    print(
        json.dumps(
            {
                "event": "exp005_terminated_gc_complete",
                "deleted": len(deleted),
                "instances": deleted,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

