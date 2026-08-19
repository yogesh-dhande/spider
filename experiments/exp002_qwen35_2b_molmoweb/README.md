# EXP002 — Qwen3.5-2B + MolmoWeb perception foundation

Status: pre-SFT baseline complete, independently validated, and immutably archived. Two-T4 QLoRA
stages have produced validated resumable checkpoints through step 1,750, and the shortened final
stage is running toward step 1,875. Corrected fixed validation probes run after every remaining
stage. Step 1,750 improved QA and grounding relative to step 1,500 and passed the regression gate;
all checkpoints remain eligible for final selection. The frozen test sets remain untouched until
final checkpoint selection.

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
- ScreenSpot is external evaluation only; its Web subset is the relevant browser comparison.
- ScreenSpot-Pro and other desktop/professional benchmarks are out of scope and must not influence
  training or checkpoint selection.
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
- 2026-08-18: after the step-250 regression gate, task-level validation probes become on-demand
  rather than automatic at every 250-step boundary. Training continues automatically while stage
  loss/eval-loss health checks remain sound; the user will request the next task-metric round.
  This reduces probe compute and repeated adaptation decisions against the same 256 examples.
- 2026-08-18: stage 1 version 1 restored checkpoint 250 and constructed both trainers, then failed
  before its first optimizer step with bitsandbytes `invalid argument` while restoring the paged
  AdamW8bit optimizer state. The failure is consistent with upstream reports for paged optimizer
  resume under multi-GPU training. It produced no scientific output and is not resumable. Future
  stages use non-paged `adamw_8bit`, which preserves the AdamW8bit algorithm and saved state while
  changing only state allocation; a one-step resume from the full checkpoint must pass first.
- 2026-08-18: fixed 256-example task probes are scheduled after stage 3 (step 1,000) and stage 7
  (step 1,875). The step-1,000 probe is a regression gate before stage 4 launches. A drop greater
  than three percentage points in QA exact, QA token F1, grounding click accuracy, or parse rate,
  or a median grounding-distance increase greater than 25 pixels relative to step 250, pauses the
  chain for review. The step-1,875 probe runs before the one-time final test evaluation.
- 2026-08-18: stage 3 completed through step 1,000 with eval loss 0.5897 and token accuracy 0.8373.
  Its first task probe generated almost every response to the 96-token limit because evaluation
  inherited model EOS `<|endoftext|>` instead of also stopping on chat end-of-turn `<|im_end|>`.
  The raw QA result is invalid. Truncating saved predictions at the decoded next-turn boundary
  recovers 0.3672 exact and 0.6874 token F1, both above step 250, but this is diagnostic only.
  Training remains paused until the same checkpoint is rerun with explicit chat-EOS stopping.
- 2026-08-18: the corrected step-1,000 probe reproduced the diagnostic values and passed the
  regression gate. Because QA exact and grounding improvements are flattening while train loss
  continues to fall faster than validation loss, task probes will run after every remaining stage
  at steps 1,250, 1,500, 1,750, and 1,875. Each probe gates the next stage; the frozen test set is
  still evaluated only once on the final selected checkpoint.
- 2026-08-18: stage 5 completed through step 1,500. QA exact held at 0.3750 and token F1 improved
  from 0.7000 to 0.7059, while grounding click accuracy declined from 0.5781 to 0.5469 and median
  distance rose from 23.9 to 32.3 pixels on the 128-example grounding probe. The checkpoint still
  passed the predeclared step-250 regression anchor, and the mixed movement on a small probe does
  not establish broad regression. Stage 6 continues; final checkpoint selection will retain step
  1,250 as a candidate rather than assuming the last checkpoint is best.
