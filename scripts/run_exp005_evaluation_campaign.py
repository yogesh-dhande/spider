#!/usr/bin/env python3
"""Run a resumable, capacity-aware EXP005 evaluation and merge campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
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


def prioritize_zones(zones: list[str], instances: list[dict[str, Any]]) -> list[str]:
    """Prefer exact zones that have previously provisioned an EXP005 GPU."""
    success_counts: dict[str, int] = {}
    for instance in instances:
        machine_type = str(instance.get("machineType", "")).rsplit("/", 1)[-1]
        if not machine_type.startswith("g2-") and not instance.get("guestAccelerators"):
            continue
        zone = zone_name(instance)
        success_counts[zone] = success_counts.get(zone, 0) + 1
    order = {zone: index for index, zone in enumerate(zones)}
    return sorted(
        zones,
        key=lambda zone: (-success_counts.get(zone, 0), order[zone]),
    )


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
        ["gcloud", "storage", "cat", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)


def storage_objects(run_id: str) -> set[str]:
    root = f"{cloud.BUCKET}/exp005/evaluation/{run_id}/**"
    result = subprocess.run(
        ["gcloud", "storage", "ls", "-r", root],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


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
    *,
    run_id: str,
    control: str,
    num_shards: int,
    identities: set[ShardIdentity] | None = None,
    objects: set[str] | None = None,
) -> set[ShardIdentity]:
    completed: set[ShardIdentity] = set()
    targets = identities
    if targets is None:
        targets = {
            ShardIdentity(suite, shard_index)
            for suite in SUITES
            for shard_index in range(num_shards)
        }
    for identity in sorted(targets):
        root = f"{cloud.BUCKET}/exp005/evaluation/{run_id}/{label(control, identity, num_shards)}"
        failure_uri = f"{root}/failed.json"
        complete_uri = f"{root}/complete.json"
        failed = storage_json(failure_uri) if objects is None or failure_uri in objects else None
        if failed is not None:
            raise RuntimeError(f"Evaluation shard failed: {failed}")
        terminal = storage_json(complete_uri) if objects is None or complete_uri in objects else None
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


def known_shards(instances: list[dict[str, Any]]) -> set[ShardIdentity]:
    known: set[ShardIdentity] = set()
    for instance in instances:
        metadata = metadata_dict(instance)
        suite = metadata.get("spider-eval-suite")
        shard = metadata.get("spider-shard-index")
        if suite in SUITES and shard is not None:
            known.add(ShardIdentity(suite, int(shard)))
    return known


def terminal_grace_filter(
    *,
    missing: set[ShardIdentity],
    known: set[ShardIdentity],
    missing_since: dict[ShardIdentity, float],
    now: float,
    grace_seconds: int,
) -> tuple[list[ShardIdentity], list[ShardIdentity]]:
    """Delay replacement while a stopped guest may still be uploading its marker."""
    for identity in set(missing_since) - missing:
        missing_since.pop(identity)
    launchable: list[ShardIdentity] = []
    deferred: list[ShardIdentity] = []
    for identity in sorted(missing):
        if identity not in known:
            missing_since.pop(identity, None)
            launchable.append(identity)
            continue
        first_seen = missing_since.setdefault(identity, now)
        if now - first_seen < grace_seconds:
            deferred.append(identity)
        else:
            launchable.append(identity)
    return launchable, deferred


def merge_complete(
    run_id: str, control: str, suite: str, *, objects: set[str] | None = None
) -> bool:
    root = f"{cloud.BUCKET}/exp005/evaluation/{run_id}/merged-{control}-{suite}"
    failure_uri = f"{root}/failed.json"
    complete_uri = f"{root}/complete.json"
    failed = storage_json(failure_uri) if objects is None or failure_uri in objects else None
    if failed is not None:
        raise RuntimeError(f"Evaluation merge failed: {failed}")
    terminal = storage_json(complete_uri) if objects is None or complete_uri in objects else None
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


def launch_available_shards(
    *,
    missing: list[ShardIdentity],
    candidates: list[str],
    slots: int,
    retry_after: dict[str, float],
    retry_seconds: int,
    now: float,
    launch: Callable[[ShardIdentity, str], None],
    record: Callable[..., None],
) -> list[tuple[ShardIdentity, str]]:
    """Fill slots, falling through to another region immediately on stockout."""
    launched: list[tuple[ShardIdentity, str]] = []
    remaining_zones = list(candidates)
    for identity in missing:
        if slots == 0:
            break
        while remaining_zones:
            zone = remaining_zones.pop(0)
            try:
                launch(identity, zone)
            except subprocess.CalledProcessError as error:
                retry_after[zone] = now + retry_seconds
                record(
                    "evaluation_campaign_launch_retry",
                    suite=identity.suite,
                    shard_index=identity.shard_index,
                    zone=zone,
                    returncode=error.returncode,
                )
                continue
            slots -= 1
            launched.append((identity, zone))
            record(
                "evaluation_campaign_shard_launched",
                suite=identity.suite,
                shard_index=identity.shard_index,
                zone=zone,
            )
            break
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", choices=("base", "exp002", "sft"), required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--zones", required=True)
    parser.add_argument("--merge-zones", required=True)
    parser.add_argument("--training-job")
    parser.add_argument("--training-step", type=int)
    parser.add_argument("--warm-image")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-seconds", type=int, default=600)
    parser.add_argument("--terminal-grace-seconds", type=int, default=180)
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
    missing_since: dict[ShardIdentity, float] = {}
    completed: set[ShardIdentity] = set()
    completed_merges: set[str] = set()
    deadline = time.monotonic() + args.timeout_seconds
    last_state: tuple[int, int] | None = None

    while time.monotonic() < deadline:
        objects = storage_objects(args.run_id)
        completed.update(
            complete_shards(
                run_id=args.run_id,
                control=args.control,
                num_shards=args.num_shards,
                identities=expected - completed,
                objects=objects,
            )
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
                if suite in completed_merges:
                    continue
                if merge_complete(
                    args.run_id, args.control, suite, objects=objects
                ):
                    completed_merges.add(suite)
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
            if completed_merges == set(SUITES):
                record("evaluation_campaign_complete", run_id=args.run_id)
                return
            time.sleep(args.poll_seconds)
            continue

        slots = max(args.max_active - len(active), 0)
        missing = expected - completed - active
        if slots and missing:
            late_completed = complete_shards(
                run_id=args.run_id,
                control=args.control,
                num_shards=args.num_shards,
                identities=missing,
                objects=storage_objects(args.run_id),
            )
            completed.update(late_completed)
            missing -= late_completed
            missing, deferred = terminal_grace_filter(
                missing=missing,
                known=known_shards(run_instances),
                missing_since=missing_since,
                now=time.monotonic(),
                grace_seconds=args.terminal_grace_seconds,
            )
            for identity in deferred:
                record(
                    "evaluation_campaign_terminal_grace",
                    suite=identity.suite,
                    shard_index=identity.shard_index,
                    grace_seconds=args.terminal_grace_seconds,
                )
            all_instances = list_instances()
            occupied = active_gpu_regions(all_instances)
            now = time.monotonic()
            candidates = prioritize_zones(
                [
                    zone
                    for zone in zones
                    if region_for_zone(zone) not in occupied
                    and retry_after.get(zone, 0) <= now
                ],
                all_instances,
            )
            def launch(identity: ShardIdentity, zone: str) -> None:
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
                    args.warm_image,
                )

            launched = launch_available_shards(
                missing=missing,
                candidates=candidates,
                slots=slots,
                retry_after=retry_after,
                retry_seconds=args.retry_seconds,
                now=now,
                launch=launch,
                record=record,
            )
            occupied.update(region_for_zone(zone) for _, zone in launched)
        time.sleep(args.poll_seconds)

    cloud.stop_instances(args.run_id)
    raise TimeoutError(f"Evaluation campaign {args.run_id} exceeded its timeout")


if __name__ == "__main__":
    main()
