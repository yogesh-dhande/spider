# EXP002 — Qwen3.5-2B + MolmoWeb perception foundation

Status: Kaggle GPU compatibility smoke passed; official baseline not started.

Parent: EXP001, superseded before any official baseline or training run.

Initial implementation commit: `df765eb`.

## Question

Can `Qwen/Qwen3.5-2B` acquire strong browser screenshot understanding and precise GUI
grounding from a relatively small amount of MolmoWeb-style training data?

## Why the backbone changed

Before EXP001 consumed GPU quota, the official Qwen3.5-2B release showed a stronger starting
point for the long-term browser-use objective. Its reported non-thinking scores exceed
Qwen3-VL-2B on OCRBench (85.4 vs 79.2) and ScreenSpot Pro (54.5 vs 48.5), while RefCOCO
grounding is approximately tied (84.3 vs 84.8). EXP001 remains recorded but unrun.

## Hypothesis and interpretation

A balanced 30K-example QLoRA SFT run will materially improve domain-held-out ScreenshotQA
answer accuracy and click-inside-element accuracy, with transfer to evaluation-only ScreenSpot.

Results will be interpreted descriptively with no preregistered pass/fail threshold. The report
will present baseline and post-SFT estimates, absolute changes, task/category breakdowns, and
representative failures.

## Frozen design

- 15K MolmoWeb-SyntheticQA + 15K GPT-authored MolmoWeb-SyntheticGround training examples.
- Per task: 1K validation and 2K test examples.
- Registrable-domain-disjoint splits shared across both tasks.
- Aspect-preserving screenshots fit within 1280 x 720, without padding or upscaling.
- Grounding target: annotated element center, serialized as JSON `point_2d` on 0–1000.
- Qwen3.5 non-thinking mode for training and evaluation.
- Language-model QLoRA targets cover full attention, MLP, and Gated DeltaNet projections;
  trainable adapter parameters remain FP32 while the frozen base uses 4-bit/FP16 on T4.
- The Kaggle compatibility stack is pinned in `requirements/experiment2-kaggle.txt`; a
  two-step GPU training smoke test is required before the official baseline or SFT run.
- No action tokens, trajectories, planning, or RL.
- Greedy, identically quantized baseline and post-SFT inference.
- ScreenSpot is external evaluation only.
- Primary metrics: normalized exact ScreenshotQA accuracy and click-inside-element accuracy.
- Diagnostics: token F1, parse/on-grid rate, mean/median pixel distance, and pixel thresholds.

Exact model and dataset revisions are pinned in
[`configs/experiment2.yaml`](../../configs/experiment2.yaml).

## Results

Pending. Completed runs will be archived under `results/<run-id>/` and committed; raw
predictions and failure galleries will be retained as Kaggle artifacts.

## Compatibility validation

Kaggle smoke version 7 passed on 2026-08-17 using two T4 GPUs and source commit `6cc023e`.
It covered base inference for both tasks, two QLoRA optimizer steps, adapter save/reload, and
post-adapter inference. This synthetic check is not an experimental result. Its machine-readable
record is in `compatibility/20260817_kaggle_v7.json`.

## Deviations and notes

Record every change made after the official baseline begins, including its reason and first
affected run ID. Never silently replace a completed run's configuration or result files.
