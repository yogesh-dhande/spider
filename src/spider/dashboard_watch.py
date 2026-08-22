from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spider.candidate_registry import SIZE_ORDER


@dataclass(frozen=True)
class DashboardCandidate:
    label: str
    size: str
    seed: int
    step: int
    run_id: str
    adapter_sha256: str
    receipt: Path
    evaluation_root: Path

    @property
    def identity(self) -> str:
        return f"{self.size}:{self.seed}:{self.step}:{self.adapter_sha256}"


def parse_source_overrides(values: list[str], repo_root: Path) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for value in values:
        receipt_text, separator, root_text = value.partition("=")
        if not separator or not receipt_text or not root_text:
            raise ValueError("Source overrides must use RECEIPT=EVALUATION_ROOT")
        receipt = _repo_path(repo_root, receipt_text)
        if receipt in result:
            raise ValueError(f"Duplicate dashboard source override: {receipt}")
        result[receipt] = _repo_path(repo_root, root_text)
    return result


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _candidate_receipt(manifest_path: Path, item: dict[str, Any]) -> Path:
    return _repo_path(manifest_path.parent, str(item["receipt"]))


def standard_evaluation_root(receipt: Path, *, repo_root: Path, output_root: Path) -> Path:
    try:
        relative = receipt.relative_to(
            repo_root / "experiments" / "exp005_browser_ablation_bed" / "artifacts" / "scaling"
        )
    except ValueError as error:
        raise ValueError(
            f"No standard dashboard source mapping for receipt {receipt}; "
            "register a --source-override"
        ) from error
    if len(relative.parts) != 2 or not relative.name.startswith("evaluation_step_"):
        raise ValueError(f"Unexpected scaling receipt layout: {receipt}")
    step_text = relative.stem.removeprefix("evaluation_step_")
    if not step_text.isdigit():
        raise ValueError(f"Cannot recover optimizer step from {receipt}")
    return output_root / relative.parent.name / f"step_{int(step_text):05d}" / "evaluation"


def latest_dashboard_candidate(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_root: Path,
    source_overrides: dict[Path, Path] | None = None,
) -> DashboardCandidate | None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", [])
    if not candidates:
        return None
    item = max(
        candidates,
        key=lambda candidate: (
            SIZE_ORDER[str(candidate["size"])],
            int(candidate["step"]),
            int(candidate["seed"]),
        ),
    )
    receipt = _candidate_receipt(manifest_path.resolve(), item)
    overrides = source_overrides or {}
    evaluation_root = overrides.get(receipt)
    if evaluation_root is None:
        evaluation_root = standard_evaluation_root(
            receipt, repo_root=repo_root.resolve(), output_root=output_root.resolve()
        )
    return DashboardCandidate(
        label=str(item["label"]),
        size=str(item["size"]),
        seed=int(item["seed"]),
        step=int(item["step"]),
        run_id=str(item["run_id"]),
        adapter_sha256=str(item["adapter_sha256"]),
        receipt=receipt,
        evaluation_root=evaluation_root,
    )
