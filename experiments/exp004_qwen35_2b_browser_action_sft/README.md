# EXP004 — Qwen3.5-2B browser action SFT

Status: running; preparation, baselines, and two-GPU compatibility complete; staged SFT in progress.

Parent: EXP002 checkpoint 1875, selected on its validation split before EXP002 final-test
evaluation.

## Question

Can the EXP002 Qwen3.5-2B perception checkpoint learn reliable browser actions from a small,
contamination-safe set of successful MolmoWeb synthetic trajectories while retaining its browser
OCR and grounding ability?

## Frozen design

- Continue from the EXP002 LoRA weights with a fresh optimizer and cosine schedule.
- Train on 20,000 action steps and 10,000 perception-replay examples for one epoch.
- Use only `from_template`, `multi_agent`, `node_traversal`, and synthetic skills for actions.
- Exclude `task_seeded_wv` and `task_seeded_om2w` because those configs are seeded from external
  benchmark tasks. Never use WebVoyager, Online-Mind2Web, or BrowserGym benchmark test tasks for
  training or model selection.
- Do not use human trajectories. This isolates the cleaner synthetic-action hypothesis and follows
  the published MolmoWeb ablation in which adding human trajectories reduced overall performance.
- Preserve ordinary rationale-plus-JSON text targets. Do not add custom action tokens.
- Normalize point and scroll values to `[0, 100]`, use a deterministic bounding-box center for
  click targets, retain at most 10 prior actions, and resize screenshots within 1280×720 without
  changing aspect ratio.
- Split whole trajectories by registrable domain where available, then by trajectory ID. No
  trajectory may cross train, validation, or test.
- Use two Kaggle T4 GPUs after a compatibility smoke. Run resumable 125-optimizer-step stages, each
  comfortably shorter than the failed first run, validate after every stage, save an optimizer
  checkpoint every 25 steps, and emit sparse JSON progress.

### Runtime-safety amendment

The first stage attempt used the initially registered 250-step boundary and stayed healthy through
step 190, but both distributed workers received `SIGKILL` after 7,392 seconds, before the first
planned checkpoint. Kaggle did not report the external cause. No CUDA error, Python exception,
metric divergence, quota warning, or recoverable checkpoint was present; plausible causes include
host-memory enforcement or worker eviction, not a demonstrated session-time limit. This occurred
before any EXP004 checkpoint validation or model selection. To limit exposure to another external
kill, the unchanged 1,875-step training schedule is now split into fifteen 125-step stages, with
external validation after each stage and rolling optimizer checkpoints every 25 steps. The data,
optimizer, effective batch size, learning-rate schedule, selection rule, and sealed tests are
unchanged.

The fixed training mixture is:

| Source | Train examples |
|---|---:|
| Synthetic trajectories: `from_template` | 6,000 |
| Synthetic trajectories: `multi_agent` | 9,000 |
| Synthetic trajectories: `node_traversal` | 4,000 |
| Synthetic skills | 1,000 |
| EXP002 ScreenshotQA replay | 5,000 |
| EXP002 grounding replay | 5,000 |
| **Total** | **30,000** |

Action validation contains 512 examples; its fixed first 256 examples form the development subset
used for stage selection. The sealed action test contains 1,024. The development perception probe
contains 256 QA and 256 grounding examples. Final perception retention uses the already sealed
EXP002 test manifests.

Checkpoint selection is fixed before observing any EXP004 stage metrics. Eligible stages must pass
the registered action and perception regression gate. Among them, select lexicographically by
development click-in-bounds accuracy, action-name accuracy, then action-argument accuracy; an exact
tie selects the earlier optimizer step. The sealed test is not consulted during selection.

## Metrics and decision rule

Offline action metrics are strict JSON parse rate, action-name accuracy, action-argument accuracy,
click-in-element-bounds accuracy, click pixel distance, and per-action counts. Rationales are saved
for diagnosis but are not scored. Closed-loop metrics are verified task success, invalid-action
rate, steps to completion, and reward components on deterministic browser tasks.

Before any action training, measure both the untouched Qwen3.5-2B base and the EXP002 adapter on the
same fixed action validation subset. Model selection compares each staged checkpoint with the
EXP002 adapter baseline. A stage advances when action metrics improve and neither QA exact accuracy
nor grounding click accuracy regresses by more than 3 absolute percentage points on the fixed
development probe. The preregistered positive result requires, on the sealed action test, at least
+5 points action-name accuracy or +10 points click-in-bounds accuracy over the EXP002 adapter,
without a greater-than-3-point regression on either sealed EXP002 perception task. All results are
reported even if the gate is missed.

The sealed action test is opened once, after checkpoint selection. External live-browser benchmark
tasks remain untouched for a later experiment; EXP004's closed-loop test uses only deterministic
local tasks with programmatic verifiers.

After development-stage selection, the sealed action and perception tests run in four independent
GPU shards. A CPU merge job verifies completeness, aggregates both tasks, and applies the frozen
positive-result rule. A separate paired closed-loop job compares the EXP002 parent adapter with the
selected EXP004 adapter on identical deterministic tasks and seeds. This keeps checkpoint selection,
sealed evaluation, and agent-level evaluation as distinct reproducible phases.

## Source audit

| Dataset | Pinned revision | Rows / size | License | Use |
|---|---|---:|---|---|
| `allenai/MolmoWeb-SyntheticTrajs` | `9b80ce0…` | 108,254 / 284 GB | ODC-BY-1.0 | Selected uncontaminated configs |
| `allenai/MolmoWeb-SyntheticSkills` | `34f7869…` | 5,545 / 16.6 GB | ODC-BY-1.0 | 1,000 action steps |

The audit was made against the official Hugging Face dataset cards and the official MolmoWeb
training implementation. The published full training mixture is not reproduced: it includes both
benchmark-seeded configs and human trajectories, which conflict with this experiment's narrower
causal question and benchmark-hygiene requirement.

Preparation streams from one pinned Parquet shard per source and stops at the registered quota. A
schema smoke initially used the 156 MB skills preview, but its 10 trajectories yielded only 302
usable train examples; the registered skills preparation therefore streams from
`data/train-00000.parquet` and stops after 1,000 accepted train examples.

### Pre-training feasibility amendment

The first full preparation attempt showed that per-source proportional held-out quotas were not
feasible under the frozen domain hash: `from_template-00001` contains only `espn.com`, which hashes
to train, and correctly ended with 6,000 train examples but no validation/test examples. This was
observed before any baseline inference or training. The replacement keeps the 20K training mixture
unchanged and keeps domain-disjoint assignment, but allocates held-out examples only to sources and
shards whose audited domains map to validation/test. The global totals remain exactly 20,000 train,
512 validation, and 1,024 sealed test examples. A metadata-only Parquet audit selected
`multi_agent-00008` for the 180-validation/563-test supplement; it is never used for training.

## Results

Pending.
