#!/usr/bin/env python3
"""Run the EXP005 scaling campaign without depending on an interactive login."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    max_attempts: int
    retry_seconds: int
    attempts: int = 0
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None
    log_path: Path | None = None
    exit_code: int | None = None
    next_start: float = 0.0


def job_command(common: dict[str, Any], job: dict[str, Any]) -> list[str]:
    command = [
        "python3",
        "scripts/run_exp005_scaling_job.py",
        "--schedule",
        common["schedule"],
        "--job-id",
        job["job_id"],
        "--training-zones",
        ",".join(job["training_zones"]),
        "--evaluation-zones",
        ",".join(common["evaluation_zones"]),
        "--merge-zones",
        ",".join(common["merge_zones"]),
        "--warm-image",
        common["warm_image"],
        "--poll-seconds",
        str(common["poll_seconds"]),
        "--timeout-seconds",
        str(common["stage_timeout_seconds"]),
        "--prerequisite-timeout-seconds",
        str(common["prerequisite_timeout_seconds"]),
        "--repo-revision",
        job["repo_revision"],
    ]
    for prerequisite in job.get("prerequisites", []):
        command.extend(["--prerequisite", prerequisite])
    if job.get("adopt_through_step"):
        command.extend(["--adopt-through-step", str(job["adopt_through_step"])])
    for override in job.get("receipt_overrides", []):
        command.extend(["--receipt-override", override])
    return command


def recovery_command(common: dict[str, Any], recovery: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "scripts/run_exp005_checkpoint_validation.py",
        "--training-run-id",
        recovery["training_run_id"],
        "--training-job",
        recovery["job_id"],
        "--start-step",
        str(recovery["start_step"]),
        "--stop-step",
        str(recovery["stop_step"]),
        "--num-nodes",
        "2",
        "--evaluation-run-id",
        recovery["evaluation_run_id"],
        "--repo-revision",
        recovery["repo_revision"],
        "--zones",
        ",".join(common["evaluation_zones"]),
        "--merge-zones",
        ",".join(common["merge_zones"]),
        "--warm-image",
        common["warm_image"],
        "--training-receipt",
        recovery["training_receipt"],
        "--evaluation-root",
        recovery["evaluation_root"],
        "--evaluation-receipt",
        recovery["evaluation_receipt"],
        "--evaluation-markdown",
        recovery["evaluation_markdown"],
        "--state-log",
        recovery["state_log"],
        "--poll-seconds",
        str(common["poll_seconds"]),
        "--training-timeout-seconds",
        str(common["stage_timeout_seconds"]),
        "--evaluation-timeout-seconds",
        str(common["stage_timeout_seconds"]),
    ]


def load_processes(config: dict[str, Any]) -> list[ManagedProcess]:
    common = config["common"]
    processes = [
        ManagedProcess(
            name=item["name"],
            command=recovery_command(common, item),
            max_attempts=int(item.get("max_attempts", 1)),
            retry_seconds=int(item.get("retry_seconds", 600)),
        )
        for item in config.get("recoveries", [])
    ]
    processes.extend(
        ManagedProcess(
            name=f"job-{item['job_id']}",
            command=job_command(common, item),
            max_attempts=1,
            retry_seconds=0,
        )
        for item in config["jobs"]
    )
    names = [item.name for item in processes]
    if len(names) != len(set(names)):
        raise ValueError("Controller process names must be unique")
    return processes


def write_status(path: Path, processes: list[ManagedProcess], terminal: bool = False) -> None:
    def last_log_line(item: ManagedProcess) -> str | None:
        if not item.log_path or not item.log_path.is_file():
            return None
        with item.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        return lines[-1][-2000:] if lines else None

    payload = {
        "schema_version": 1,
        "kind": "exp005_cloud_controller_status",
        "timestamp_utc": utc_now(),
        "terminal": terminal,
        "processes": [
            {
                "name": item.name,
                "attempts": item.attempts,
                "pid": item.process.pid if item.process and item.process.poll() is None else None,
                "exit_code": item.exit_code,
                "last_log_line": last_log_line(item),
                "next_start_seconds": max(0, round(item.next_start - time.monotonic()))
                if item.next_start
                else 0,
            }
            for item in processes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def upload_state(config: dict[str, Any], status_path: Path) -> None:
    state_uri = config["state_uri"].rstrip("/")
    with tempfile.TemporaryDirectory(prefix="spider-controller-state-") as temporary:
        archive = Path(temporary) / "latest.tar.gz"
        existing = [path for raw in config["state_paths"] if (path := Path(raw)).exists()]
        if existing:
            subprocess.run(
                ["tar", "-czf", str(archive), *[str(path) for path in existing]], check=True
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            subprocess.run(
                ["gcloud", "storage", "cp", str(archive), f"{state_uri}/latest.tar.gz"],
                check=True,
            )
            emit("controller_state_uploaded", bytes=archive.stat().st_size, sha256=digest)
    subprocess.run(
        ["gcloud", "storage", "cp", str(status_path), f"{state_uri}/status.json"], check=True
    )


def upload_state_safely(config: dict[str, Any], status_path: Path) -> bool:
    try:
        upload_state(config, status_path)
    except subprocess.CalledProcessError as error:
        emit("controller_state_upload_failed", returncode=error.returncode)
        return False
    return True


def start_process(item: ManagedProcess, log_root: Path, env: dict[str, str]) -> None:
    item.attempts += 1
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{item.name}.attempt-{item.attempts:02d}.log"
    item.log_path = log_path
    item.log_handle = log_path.open("a", encoding="utf-8")
    item.process = subprocess.Popen(
        item.command,
        stdout=item.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    item.exit_code = None
    item.next_start = 0.0
    emit(
        "controller_process_started",
        name=item.name,
        pid=item.process.pid,
        attempt=item.attempts,
        log=str(log_path),
    )


def stop_processes(processes: list[ManagedProcess]) -> None:
    for item in processes:
        if item.process and item.process.poll() is None:
            os.killpg(item.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and any(
        item.process and item.process.poll() is None for item in processes
    ):
        time.sleep(1)
    for item in processes:
        if item.process and item.process.poll() is None:
            os.killpg(item.process.pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, default=Path("outputs/experiment5/cloud-controller"))
    parser.add_argument("--status", type=Path, default=Path("outputs/experiment5/cloud-controller/status.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("kind") != "exp005_cloud_controller":
        raise ValueError("Not an EXP005 cloud controller configuration")
    processes = load_processes(config)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    shutting_down = False

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutting_down
        shutting_down = True

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    next_snapshot = 0.0
    completed_normally = False
    try:
        while not shutting_down:
            now = time.monotonic()
            for item in processes:
                if item.process is None and item.exit_code is None and now >= item.next_start:
                    start_process(item, args.log_root, env)
                    continue
                if not item.process:
                    continue
                code = item.process.poll()
                if code is None:
                    continue
                item.log_handle.close()
                item.log_handle = None
                item.process = None
                if code and item.attempts < item.max_attempts:
                    item.next_start = now + item.retry_seconds
                    emit(
                        "controller_process_retry_scheduled",
                        name=item.name,
                        exit_code=code,
                        attempt=item.attempts,
                        retry_seconds=item.retry_seconds,
                    )
                else:
                    item.exit_code = code
                    emit(
                        "controller_process_terminal",
                        name=item.name,
                        exit_code=code,
                        attempts=item.attempts,
                    )
            terminal = all(item.exit_code is not None for item in processes)
            write_status(args.status, processes, terminal=terminal)
            if now >= next_snapshot or terminal:
                upload_state_safely(config, args.status)
                next_snapshot = now + int(config.get("snapshot_seconds", 300))
            if terminal:
                failed = [item.name for item in processes if item.exit_code]
                emit("controller_terminal", status="failed" if failed else "complete", failed=failed)
                completed_normally = True
                return
            time.sleep(int(config.get("poll_seconds", 30)))
    finally:
        stop_processes(processes)
        if not completed_normally:
            write_status(args.status, processes, terminal=False)
            upload_state_safely(config, args.status)


if __name__ == "__main__":
    main()
