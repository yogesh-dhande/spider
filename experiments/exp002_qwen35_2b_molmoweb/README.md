# EXP002 — Qwen3.5-2B + MolmoWeb perception foundation

Status: pre-SFT baseline complete, independently validated, and immutably archived; two-T4 QLoRA
compatibility passed and official timeout-safe stage 0 version 2 (steps 0–250) completed with a
validated resumable checkpoint. Held-out validation probes and the cross-kernel resume gate are
next; the frozen test sets remain untouched until the final SFT checkpoint.

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

The pre-SFT baseline over 5,272 frozen examples is complete: 22.40% MolmoWeb ScreenshotQA exact
accuracy, 56.65% MolmoWeb grounding click accuracy, and 47.17% ScreenSpot grounding click
accuracy. See [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md) for all eight shard scores,
variability, diagnostics, validation, and failure-category counts. Raw/scored predictions and the
representative failure gallery are retained in Kaggle merge version 2. The compact immutable
archive is `results/20260818_baseline_v1/`.

## Compatibility validation

Kaggle smoke version 7 passed on 2026-08-17 using two T4 GPUs and source commit `6cc023e`.
It covered base inference for both tasks, two QLoRA optimizer steps, adapter save/reload, and
post-adapter inference. This synthetic check is not an experimental result. Its machine-readable
record is in `compatibility/20260817_kaggle_v7.json`.

## Prepared-data reuse

Finalizer version 3 preserves `data/molmoweb_30k_domain17` as an immutable private Kaggle
notebook artifact containing the resized images, frozen manifests, summary, config, and checksums.
Training and evaluation attach that exact version instead of repeating preparation. A separate
Kaggle Dataset can be created later through a server-side workflow; it is not required for runs.
Do not make derived images public without first verifying the upstream redistribution terms.

## Timeout-safe SFT execution

The official one-epoch QLoRA horizon is 1,875 optimizer steps with per-device batch 1 and
effective batch 16. The official run uses two T4 processes with gradient accumulation
8 after a full-resolution compatibility gate passed distributed QLoRA, checkpointing, adapter
reload, and inference. Training is split at steps 250, 500, 750, 1000, 1250, 1500, 1750, and
1875; measured throughput projects roughly 2.2–2.7 hours per regular stage, leaving a large margin
below Kaggle's 12-hour session limit. Every successful stage saves the
adapter plus Trainer optimizer, scheduler, scaler, RNG, and data-skip state. The next stage must
restore exactly one completed predecessor and continue against the original 1,875-step scheduler
horizon. Failed or partial stages are never eligible resume sources. Stage boundaries therefore
change execution packaging, not the frozen dataset, batch size, prompts, loss, learning rate,
scheduler, seed, or model configuration.

## Deviations and notes

Record every change made after the official baseline begins, including its reason and first
affected run ID. Never silently replace a completed run's configuration or result files.

- 2026-08-17: termination of Kaggle baseline version 1 was requested after about 1 hour 40
  minutes because it combined data preparation and the complete evaluation in one timeout-bound
  session. It is an aborted infrastructure run and will not be scored or treated as experimental
  evidence. The first affected replacement run will be baseline version 2. The frozen model,
  examples, prompts, decoding, and metrics are unchanged; execution is split into a CPU-only
  prepared-data artifact and eight deterministic GPU evaluation shards with a validated merge.
  QLoRA sessions start at 100 optimizer steps rather than 500 until throughput is measured.
- 2026-08-17: future preparation, evaluation, and training runners replace animated progress
  bars with sparse line-oriented logs. This observability-only change was prompted by Kaggle's
  CLI not exposing live `tqdm` output and the web log being dominated by carriage-return control
  sequences. It does not change example selection, model computation, or metrics. The currently
  running QA and grounding preparation version 1 jobs remain pinned to the prior implementation.
- 2026-08-18: baseline shard 3 and 4 version-1 notebooks failed before model loading because
  Kaggle did not expose the attached finalizer artifact at its previously observed hard-coded
  notebook path. They wrote zero predictions and are not experimental results. Version 2 searches
  the full `/kaggle/input` tree for the uniquely named prepared-data directory; this changes only
  artifact discovery. The remaining runners also disable pip's animated download progress output.
- 2026-08-18: CPU merge version 1 restored all eight completed shards, then rejected them because
  shard versions recorded two equivalent Kaggle mount prefixes for the same three manifests. It
  produced no merged predictions. Version 2 compares the logical manifest filenames and still
  enforces exact deterministic shard IDs, non-overlap, full coverage, and matching model,
  revision, adapter, and split. This is merge plumbing only; no inference is repeated.
- 2026-08-18: the first one-T4 SFT stage was cancelled without a checkpoint after its 10-step
  progress report measured 71.1 seconds per optimizer step, projecting about 37 hours for the
  epoch. It is an infrastructure throughput probe, not an experimental result, and will not be
  resumed. A separate two-T4 gate measured 32.0 seconds per step while preserving effective batch
  16 and the 1,875-step horizon, then passed checkpoint and adapter-reload inference. Official SFT
  restarts from the untouched base model on two T4 processes and uses 250-step resumable stages.
- 2026-08-18: official two-T4 stage 0 version 2 completed steps 0–250 in 6,674.6 seconds and
  produced a validated checkpoint with optimizer, scheduler, two-rank RNG, Trainer, and adapter
  state. Its final held-out language-model validation loss was 0.6321 with token accuracy 0.8309.
  Task-level ScreenshotQA and grounding comparisons require the fixed held-out validation probe;
  no official test examples were evaluated at this stage.
