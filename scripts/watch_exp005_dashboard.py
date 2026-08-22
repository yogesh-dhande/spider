#!/usr/bin/env python3
"""Keep the EXP005 dashboard on the newest registered validation checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from spider.dashboard_watch import latest_dashboard_candidate, parse_source_overrides


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
    parser.add_argument("--source-override", action="append", default=[])
    parser.add_argument("--suite", choices=("iid", "domain_balanced"), default="iid")
    parser.add_argument("--display-limit", type=int, default=64)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
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
    ):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, (args.repo_root / path).resolve())
    overrides = parse_source_overrides(args.source_override, args.repo_root)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    last_waiting: str | None = None
    while True:
        candidate = latest_dashboard_candidate(
            args.manifest,
            repo_root=args.repo_root,
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
                    repo_root=args.repo_root,
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
