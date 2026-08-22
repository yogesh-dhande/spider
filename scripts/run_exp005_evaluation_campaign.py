#!/usr/bin/env python3
"""Run a resumable, capacity-aware EXP005 evaluation and merge campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcloud_exp005 as cloud

SUITES = ("iid", "domain_balanced", "distribution_shift")
ACTIVE_STATES = cloud.ACTIVE_STATES


@dataclass(frozen=True, order=True)
class ShardIdentity:
    suite: str
    shard_index: int


def label(control: str, identity: ShardIdentity, num_shards: int) -> str:
    return (
        f"{control}-{identity.suite}-shard-{identity.shard_index:02d}"
        f"-of-{num_shards:02d}"
    )


def metadata_dict(instance: dict[str, Any]) -> dict[str, str]:
    metadata = instance.get("metadata") or {}
    return {
        str(item["key"]): str(item.get("value", ""))
        for item in metadata.get("items", [])
    }


def zone_name(instance: dict[str, Any]) -> str:
    return str(instance["zone"]).rsplit("/", 1)[-1]


def region_for_zone(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def active_gpu_regions(instances: list[dict[str, Any]]) -> set[str]:
    regions: set[str] = set()
    for instance in instances:
        if instance.get("status") not in ACTIVE_STATES:
            continue
        machine_type = str(instance.get("machineType", "")).rsplit("/", 1)[-1]
        if machine_type.startswith("g2-") or instance.get("guestAccelerators"):
            regions.add(region_for_zone(zone_name(instance)))
    return regions


def run_json(command: list[str]) -> Any:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout or "[]")


def list_instances(run_id: str | None = None) -> list[dict[str, Any]]:
    filters = []
    if run_id:
        filters.append(f"labels.spider-run={run_id}")
    command = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={cloud.PROJECT}",
        "--format=json(name,zone,status,machineType,guestAccelerators,labels,metadata)",
    ]
    if filters:
        command.append(f"--filter={' AND '.join(filters)}")
    return run_json(command)


def storage_json(uri: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri], capture_output=True, text=True
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)


def validate_shard_terminal(
    terminal: dict[str, Any],
    *,
    run_id: str,
    control: str,
    identity: ShardIdentity,
    num_shards: int,
) -> None:
    expected = {
        "run_id": run_id,
        "control": control,
        "suite": identity.suite,
        "shard_index": identity.shard_index,
        "num_shards": num_shards,
        "status": "complete",
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": terminal.get(key)}
        for key, value in expected.items()
        if terminal.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid terminal for {identity}: {mismatches}")


def complete_shards(
    *, run_id: str, control: str, num_shards: int
) -> set[ShardIdentity]:
    completed: set[ShardIdentity] = set()
    for suite in SUITES:
        for shard_index in range(num_shards):
            identity = ShardIdentity(suite, shard_index)
            root = f"{cloud.BUCKET}/exp005/evaluation/{run_id}/{label(control, identity, num_shards)}"
            failed = storage_json(f"{root}/failed.json")
            if failed is not None:
                raise RuntimeError(f"Evaluation shard failed: {failed}")
            terminal = storage_json(f"{root}/complete.json")
            if terminal is None:
                continue
            validate_shard_terminal(
                terminal,
                run_id=run_id,
                control=control,
                identity=identity,
                num_shards=num_shards,
            )
            completed.add(identity)
    return completed


def active_shards(instances: list[dict[str, Any]]) -> set[ShardIdentity]:
    active: set[ShardIdentity] = set()
    for instance in instances:
        if instance.get("status") not in ACTIVE_STATES:
            continue
        metadata = metadata_dict(instance)
        suite = metadata.get("spider-eval-suite")
        shard = metadata.get("spider-shard-index")
        if suite in SUITES and shard is not None:
            active.add(ShardIdentity(suite, int(shard)))
    return active


def merge_complete(run_id: str, control: str, suite: str) -> bool:
    root = f"{cloud.BUCKET}/exp005/evaluation/{run_id}/merged-{control}-{suite}"
    failed = storage_json(f"{root}/failed.json")
    if failed is not None:
        raise RuntimeError(f"Evaluation merge failed: {failed}")
    terminal = storage_json(f"{root}/complete.json")
    if terminal is None:
        return False
    expected = {
        "run_id": run_id,
        "control": control,
        "suite": suite,
        "status": "complete",
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": terminal.get(key)}
        for key, value in expected.items()
        if terminal.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid merge terminal for {suite}: {mismatches}")
    return True


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", choices=("base", "exp002", "sft"), required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--zones", required=True)
    parser.add_argument("--merge-zones", required=True)
    parser.add_argument("--training-job")
    parser.add_argument("--training-step", type=int)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-seconds", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--state-log", type=Path)
    args = parser.parse_args()

    if args.control == "sft" and (not args.training_job or not args.training_step):
        parser.error("sft requires --training-job and --training-step")
    zones = [zone.strip() for zone in args.zones.split(",") if zone.strip()]
    merge_zones = [zone.strip() for zone in args.merge_zones.split(",") if zone.strip()]
    if not zones or len(merge_zones) < len(SUITES):
        parser.error("require GPU zones and at least three merge zones")
    if args.state_log:
        args.state_log.parent.mkdir(parents=True, exist_ok=True)

    def record(event: str, **fields: Any) -> None:
        payload = {"timestamp_utc": cloud.utc_now(), "event": event, **fields}
        emit(event, **fields)
        if args.state_log:
            with args.state_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    expected = {
        ShardIdentity(suite, shard_index)
        for suite in SUITES
        for shard_index in range(args.num_shards)
    }
    retry_after: dict[str, float] = {}
    deadline = time.monotonic() + args.timeout_seconds
    last_state: tuple[int, int] | None = None

    while time.monotonic() < deadline:
        completed = complete_shards(
            run_id=args.run_id, control=args.control, num_shards=args.num_shards
        )
        run_instances = list_instances(args.run_id)
        active = active_shards(run_instances)
        state = (len(completed), len(active))
        if state != last_state:
            record(
                "evaluation_campaign_progress",
                run_id=args.run_id,
                completed=len(completed),
                active=len(active),
                total=len(expected),
            )
            last_state = state

        if completed == expected:
            merge_instances = list_instances(args.run_id)
            active_merge_suites = {
                metadata_dict(instance).get("spider-eval-suite")
                for instance in merge_instances
                if instance.get("status") in ACTIVE_STATES
                and (instance.get("labels") or {}).get("spider-role") == "evaluation-merge"
            }
            for suite, zone in zip(
                SUITES, merge_zones[: len(SUITES)], strict=True
            ):
                if merge_complete(args.run_id, args.control, suite):
                    continue
                if suite in active_merge_suites:
                    continue
                cloud.create_evaluation_merge(
                    args.run_id,
                    zone,
                    args.repo_revision,
                    args.control,
                    suite,
                    args.num_shards,
                    "2h",
                )
                record("evaluation_campaign_merge_launched", suite=suite, zone=zone)
            if all(merge_complete(args.run_id, args.control, suite) for suite in SUITES):
                record("evaluation_campaign_complete", run_id=args.run_id)
                return
            time.sleep(args.poll_seconds)
            continue

        slots = max(args.max_active - len(active), 0)
        missing = sorted(expected - completed - active)
        if slots and missing:
            occupied = active_gpu_regions(list_instances())
            now = time.monotonic()
            candidates = [
                zone
                for zone in zones
                if region_for_zone(zone) not in occupied and retry_after.get(zone, 0) <= now
            ]
            for identity, zone in zip(missing[:slots], candidates, strict=False):
                try:
                    cloud.create_evaluation_shard(
                        args.run_id,
                        zone,
                        args.repo_revision,
                        args.control,
                        identity.suite,
                        identity.shard_index,
                        args.num_shards,
                        "4h",
                        args.training_job,
                        args.training_step,
                    )
                except subprocess.CalledProcessError as error:
                    retry_after[zone] = time.monotonic() + args.retry_seconds
                    record(
                        "evaluation_campaign_launch_retry",
                        suite=identity.suite,
                        shard_index=identity.shard_index,
                        zone=zone,
                        returncode=error.returncode,
                    )
                    continue
                occupied.add(region_for_zone(zone))
                record(
                    "evaluation_campaign_shard_launched",
                    suite=identity.suite,
                    shard_index=identity.shard_index,
                    zone=zone,
                )
        time.sleep(args.poll_seconds)

    cloud.stop_instances(args.run_id)
    raise TimeoutError(f"Evaluation campaign {args.run_id} exceeded its timeout")


if __name__ == "__main__":
    main()
