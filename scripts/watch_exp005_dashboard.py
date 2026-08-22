#!/usr/bin/env python3
"""Keep the EXP005 dashboard on the newest registered validation checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from spider.dashboard_watch import latest_dashboard_candidate, parse_source_overrides


CONTROLLER_SNAPSHOT_MEMBERS = (
    "experiments/exp005_browser_ablation_bed/artifacts",
    "experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json",
    "outputs/experiment5/scaling",
)


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _run(command: list[str], repo_root: Path) -> None:
    subprocess.run(command, cwd=repo_root, check=True)


def _allowed_snapshot_member(name: str) -> bool:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if any(part.startswith("._") for part in path.parts):
        return False
    return any(name == prefix or name.startswith(f"{prefix}/") for prefix in CONTROLLER_SNAPSHOT_MEMBERS)


def extract_controller_snapshot(archive: Path, mirror: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="spider-controller-mirror-") as directory:
        staging = Path(directory)
        with tarfile.open(archive, "r:gz") as source:
            members = []
            for member in source.getmembers():
                if not _allowed_snapshot_member(member.name):
                    continue
                if member.issym() or member.islnk():
                    raise ValueError(f"Controller snapshot contains a link: {member.name}")
                members.append(member)
            source.extractall(staging, members=members, filter="data")
        mirror.mkdir(parents=True, exist_ok=True)
        for top_level in ("experiments", "outputs"):
            source = staging / top_level
            if source.is_dir():
                shutil.copytree(source, mirror / top_level, dirs_exist_ok=True)


def sync_controller_snapshot(uri: str, mirror: Path, state_path: Path) -> bool:
    generation = subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=value(generation)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not generation:
        raise ValueError("Controller snapshot lacked a GCS generation")
    if _load_state(state_path).get("generation") == generation:
        return False
    with tempfile.TemporaryDirectory(prefix="spider-controller-download-") as directory:
        archive = Path(directory) / "latest.tar.gz"
        subprocess.run(
            ["gcloud", "storage", "cp", uri, str(archive)],
            check=True,
            capture_output=True,
            text=True,
        )
        extract_controller_snapshot(archive, mirror)
    _write_state(state_path, {"generation": generation, "uri": uri})
    emit("dashboard_controller_snapshot_synced", generation=generation)
    return True


def refresh(args: argparse.Namespace, candidate: Any) -> None:
    common = [
        sys.executable,
        "scripts/refresh_exp005_dashboard.py",
        "--baseline-root",
        str(args.baseline_root),
        "--latest-root",
        str(candidate.evaluation_root),
        "--baseline-receipt",
        str(args.baseline_receipt),
        "--latest-receipt",
        str(candidate.receipt),
        "--latest-name",
        candidate.label,
        "--latest-step",
        str(candidate.step),
        "--suite",
        args.suite,
        "--corpus-root",
        str(args.corpus_root),
        "--dashboard-json",
        str(args.dashboard_json),
        "--display-limit",
        str(args.display_limit),
    ]
    _run([*common, "--skip-images"], args.repo_root)
    _run(
        [
            sys.executable,
            "-m",
            "spider.dashboard_images",
            "--payload",
            str(args.dashboard_json),
            "--archive",
            str(args.corpus_archive),
            "--destination",
            str(args.corpus_root),
        ],
        args.repo_root,
    )
    _run(common, args.repo_root)
    _run(["npm", "--prefix", "dataset-dashboard", "run", "build"], args.repo_root)
    _run(
        ["node", "--test", "dataset-dashboard/tests/rendered-html.test.mjs"],
        args.repo_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-repo-root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiment5/scaling"))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/experiment5/baseline/base-all-0821a"),
    )
    parser.add_argument(
        "--baseline-receipt",
        type=Path,
        default=Path(
            "experiments/exp005_browser_ablation_bed/artifacts/baseline_base_all_0821a.json"
        ),
    )
    parser.add_argument(
        "--corpus-archive",
        type=Path,
        default=Path("outputs/experiment5/dashboard_corpus/corpus.tar.zst"),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("outputs/experiment5/dashboard_corpus_subset"),
    )
    parser.add_argument(
        "--dashboard-json", type=Path, default=Path("dataset-dashboard/app/qa-probe.json")
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/experiment5/dashboard_watch/state.json"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("outputs/experiment5/locks/dashboard-refresh.lock"),
    )
    parser.add_argument("--controller-state-uri")
    parser.add_argument(
        "--controller-mirror",
        type=Path,
        default=Path("outputs/experiment5/controller_mirror"),
    )
    parser.add_argument(
        "--controller-sync-state",
        type=Path,
        default=Path("outputs/experiment5/dashboard_watch/controller_sync.json"),
    )
    parser.add_argument("--source-override", action="append", default=[])
    parser.add_argument("--suite", choices=("iid", "domain_balanced"), default="iid")
    parser.add_argument("--display-limit", type=int, default=64)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    if args.candidate_repo_root is None:
        args.candidate_repo_root = args.repo_root
    elif not args.candidate_repo_root.is_absolute():
        args.candidate_repo_root = (args.repo_root / args.candidate_repo_root).resolve()
    for name in (
        "manifest",
        "output_root",
        "baseline_root",
        "baseline_receipt",
        "corpus_archive",
        "corpus_root",
        "dashboard_json",
        "state",
        "lock",
        "controller_mirror",
        "controller_sync_state",
    ):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, (args.repo_root / path).resolve())
    overrides = parse_source_overrides(args.source_override, args.repo_root)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    last_waiting: str | None = None
    while True:
        if args.controller_state_uri:
            try:
                sync_controller_snapshot(
                    args.controller_state_uri,
                    args.controller_mirror,
                    args.controller_sync_state,
                )
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                waiting = f"controller-sync:{type(error).__name__}:{error}"
                if waiting != last_waiting:
                    emit("dashboard_controller_sync_failed", error=str(error))
                    last_waiting = waiting
                if args.once:
                    raise
        candidate = latest_dashboard_candidate(
            args.manifest,
            repo_root=args.candidate_repo_root,
            output_root=args.output_root,
            source_overrides=overrides,
        )
        if candidate is None:
            if last_waiting != "candidate":
                emit("dashboard_waiting_for_candidate")
                last_waiting = "candidate"
        elif _load_state(args.state).get("candidate_identity") == candidate.identity:
            if args.once:
                emit("dashboard_already_current", candidate=candidate.identity)
                return
        elif not candidate.receipt.is_file() or not candidate.evaluation_root.is_dir():
            waiting = f"{candidate.receipt}:{candidate.evaluation_root}"
            if waiting != last_waiting:
                emit(
                    "dashboard_waiting_for_local_artifacts",
                    receipt=str(candidate.receipt),
                    evaluation_root=str(candidate.evaluation_root),
                )
                last_waiting = waiting
        else:
            with args.lock.open("w", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                newest = latest_dashboard_candidate(
                    args.manifest,
                    repo_root=args.candidate_repo_root,
                    output_root=args.output_root,
                    source_overrides=overrides,
                )
                if newest is not None and newest.identity == candidate.identity:
                    emit(
                        "dashboard_refresh_started",
                        candidate=candidate.identity,
                        label=candidate.label,
                    )
                    refresh(args, candidate)
                    _write_state(
                        args.state,
                        {
                            "candidate_identity": candidate.identity,
                            "label": candidate.label,
                            "receipt": str(candidate.receipt),
                            "evaluation_root": str(candidate.evaluation_root),
                        },
                    )
                    emit(
                        "dashboard_refresh_completed",
                        candidate=candidate.identity,
                        label=candidate.label,
                    )
                    last_waiting = None
            if args.once:
                return
        if args.once:
            raise SystemExit("No refreshable registered checkpoint is available")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
