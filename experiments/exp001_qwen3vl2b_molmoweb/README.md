# EXP001 — Qwen3-VL-2B + MolmoWeb perception foundation

Status: superseded without an official run by EXP002 (Qwen3.5-2B).

Initial implementation commit: `4b608ee`.

No baseline, training, or post-SFT results were produced. The design and implementation are
retained as provenance for the backbone decision.

## Question

Can `Qwen/Qwen3-VL-2B-Instruct` acquire strong browser screenshot understanding and precise
GUI grounding from a relatively small amount of MolmoWeb-style training data?

## Hypothesis

A balanced 30K-example QLoRA SFT run will materially improve domain-held-out ScreenshotQA
answer accuracy and click-inside-element accuracy, with transfer to evaluation-only ScreenSpot.

The result will be interpreted descriptively, with no preregistered pass/fail threshold. The
report will present baseline and post-SFT estimates, absolute changes, task/category breakdowns,
and representative failures without selecting a success cutoff after observing the results.

## Frozen design

- 15K MolmoWeb-SyntheticQA + 15K GPT-authored MolmoWeb-SyntheticGround training examples.
- Per task: 1K validation and 2K test examples.
- Registrable-domain-disjoint splits shared across both tasks.
- Aspect-preserving screenshots fit within 1280 x 720, without padding or upscaling.
- Grounding target: annotated element center, serialized as Qwen JSON `point_2d` on 0–1000.
- One multitask QLoRA adapter; no action tokens, trajectories, planning, or RL.
- Greedy, identically quantized baseline and post-SFT inference.
- ScreenSpot is external evaluation only.
- Primary metrics: normalized exact ScreenshotQA accuracy and click-inside-element accuracy.
- Diagnostics: token F1, parse/on-grid rate, mean/median pixel distance, and pixel thresholds.

Exact model and dataset revisions are pinned in
[`configs/experiment1.yaml`](../../configs/experiment1.yaml).

## Results

Pending. Each completed run will be stored under `results/<run-id>/` by `spider-archive` and
committed. Raw predictions and failure galleries are retained as Kaggle artifacts.

## Deviations and notes

Record any change made after the official baseline begins here, including its reason and the
first affected run ID. Never silently replace a completed run's configuration or result files.
