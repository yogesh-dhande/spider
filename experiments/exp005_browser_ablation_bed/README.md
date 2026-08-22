# EXP005 — trustworthy browser-training ablation bed

Status: leakage-safe 10K/30K/100K manifests and the corrected shared screenshot corpus are frozen.
The zero-missing gate rejected and stopped the first materialization attempt before merge when it
exposed a trajectory image-path decoder bug; the immutable incident receipt is
`artifacts/materialization_attempt_v1.json`. Corrected run `corpus-v2-0821a` then completed all
eight shards with zero missing screenshots and passed the guarded merge; its receipt is
`artifacts/materialization_attempt_v2.json`. Untouched-model evaluation `base-all-0821a` completed
all twelve shards and three guarded merges. Its machine-readable receipt and shard tables are
`artifacts/baseline_base_all_0821a.json` and `artifacts/baseline_base_all_0821a.md`. Multinode
compatibility testing selected two nearby single-L4 nodes over a geographically dispersed
four-node cluster; the immutable receipt is `artifacts/multinode_smoke_v1.json`. The first exact
10K/seed-53 reference stage completed step 500 on the selected topology with exact two-rank
terminal markers and a healthy adapter. Its matched validation is now running across all three
frozen suites. The matched EXP002 starting-adapter control is complete.

Protocol amendments are immutable, numbered records in this directory. Amendment 001 excludes the
alphabetically truncated Hugging Face partial Parquet conversion of ScreenshotQA after its domain
skew was measured, and replaces it with all five original pinned Arrow shards before any EXP005
dataset freeze, inference, or training.

## Objective

Create a contamination-safe, website-diverse, reproducible experiment bed that can distinguish
changes to browser-model architecture, data, and training methods. It must support cheap screening,
parallel execution, confirmation across seeds, and later data-scaling studies without changing the
evaluation suite underneath a comparison.

The first reference recipe is the EXP004 QLoRA/SFT recipe starting from the frozen EXP002
perception adapter. EXP005 v1 remains MolmoWeb-only so the first comparison isolates the effect of
better inventory and sampling. Additional training sources can be registered later without changing
the manifest, run-identity, or evaluation contracts. External browser benchmarks remain sealed.

## Nested training ladder

| Tier | Action | ScreenshotQA | Grounding | Total |
|---|---:|---:|---:|---:|
| Small | 5,400 | 3,300 | 1,300 | 10,000 |
| Medium | 16,200 | 9,900 | 3,900 | 30,000 |
| Large | 54,000 | 33,000 | 13,000 | 100,000 |

The manifests must satisfy `small ⊂ medium ⊂ large`. They reference one shared image store; the
dataset is not copied three times. A deterministic diversity order is sampled from metadata
inventoried across every eligible pinned shard. Every tier is audited for source, task, action,
website, trajectory, screenshot, effective-domain count, and website concentration. The realized
maximum combined contribution from one registrable domain is 4.02%, 5.13%, and 5.73% for the
small, medium, and large tiers, respectively, below the registered 6.5% ceiling.

Unknown domains are rejected. Complete trajectories/screenshots remain within one split. Training
allows at most four action steps per trajectory, three QA examples per screenshot, and one grounding
example per screenshot during ladder selection. The QA cap prevents repeated questions from one
screenshot from displacing screenshot and website diversity. The exact realized manifests,
configuration, source revisions, and SHA-256 checksums are immutable publication artifacts.

Browser-internal error documents such as `chromewebdata` and localhost pages are explicitly
excluded and counted in the inventory record; they are not treated as website domains.

## Website coverage and application focus

The metadata inventory emits JSON and CSV catalogs of every eligible registrable website, including
task/source counts, screenshot or trajectory counts, app surfaces, and action types. App surfaces
such as Gmail, Google Calendar, Docs, Sheets, and Slides remain distinct in the catalog, while
leakage isolation deliberately groups them under `google.com`.

Every nested training tier must cover all available website families: work applications,
transactional applications, service applications, content/reference sites, and the general web.
Task-specific feasible floors are 1% for action, 5% for QA, and 0.1% for grounding. Remaining
capacity is weighted toward work applications (4×), then transactional and service applications
(2×), general web (1×), and passive content/reference sites (0.5×). Per-task registrable-domain
caps are 3% for action, 2% for QA, and 29% for grounding. The larger grounding allowance is an
explicit source limitation: its examples are highly concentrated in Semantic Scholar, Amazon, and
Cambridge. URL-based categories record their rule and confidence, and the highest-volume surfaces
remain manually reviewable.

Unseen-domain evaluation retains all website families and the same application-oriented weighting,
with category-level metrics reported separately. Distribution-shift evaluation follows the natural
composition of its held-out generator so it measures a real shift rather than a curated one.

## Fixed evaluation stack

1. Distribution-weighted IID development set for expected in-distribution behavior.
2. Domain-balanced, unseen-domain development set with micro and macro-over-domain metrics.
3. Distribution-shift development set with held-out trajectory generators or task templates.
4. Larger verified closed-loop browser suite used for model selection after the harness is frozen.
5. Sealed external browser-only benchmarks used only after all choices are frozen.

Training/evaluation ID and sampling-unit overlap must always be zero. Known-domain overlap must also
be zero for the domain-balanced and distribution-shift suites; it is intentionally allowed for the
IID suite. Evaluation uncertainty is clustered by domain, trajectory, or screenshot rather than
treating repeated steps and questions as independent examples.

## Ablation execution contract

An ablation matrix is `recipe × dataset size × seed`. Each resolved job receives a content-derived
identity, its own configuration, claim directory, logs, checkpoints, and output directory. Atomic
claims protect workers sharing a filesystem. Independent cloud workers can instead use deterministic
matrix shards; their shard index is part of the launch record, so they never target the same jobs.
All job paths are relative to the portable plan root, and cloud workers explicitly remap the shared
corpus mount. Failed or abandoned attempts require an explicit requeue reason; their prior claim,
result, and log are archived rather than overwritten.

The budget ladder is:

1. `smoke`: 10K manifest, one seed, 20 optimizer steps.
2. `pilot`: normally the 30K manifest, one seed, bounded training and development evaluation.
3. `confirm`: 30K, three seeds, full development evaluation.
4. `scaling`: only for a promising frozen recipe, 10K/30K/100K across three seeds.
5. `final`: sealed external evaluation, launched separately and never used for selection.

The matrix planner refuses to overwrite an existing plan with different content. Training also
hashes its config and train/validation manifests and refuses to resume an output directory belonging
to another job.

`scripts/run_exp005_scaling_stage.py` is the reusable cloud stage boundary. It inspects an exact
two-rank terminal state before launch, refuses partial, failed, or orphaned attempts, preserves the
effective batch-16 topology, waits for a checksumed training receipt, runs all three frozen
evaluation suites, binds the evaluated adapter hash to the trained checkpoint, writes isolated
receipts, and stops any remaining training or evaluation VMs in a `finally` path. The command runs
one stage only so metrics can be reviewed for regression before the next registered stage begins;
independent size/seed jobs may use separate invocations in parallel.

The exact one-epoch schedule is frozen in `artifacts/scaling_schedule_v2.json` with schedule hash
`eec4b382e410c9cfc6b20e8c51b298d71f83dfd85bc046811eae55ef3f2c1679`. At effective batch 16,
the 10K, 30K, and 100K tiers require 625, 1,875, and 6,250 optimizer steps per seed. Stages are at
most 500 steps and every stage receives all three frozen validation suites, yielding 57 training
stages and 57 full validation campaigns across nine jobs. The single override file preserves the
run IDs of the already-running first stage instead of relabelling that scientific attempt.
Schedule v1 is retained but rejected: its first two replication launches were refused by GCE before
resource creation because the prefixed instance names exceeded 63 characters. The incident receipt
is `artifacts/scaling_schedule_launch_incident_v1.json`; v2 uses compact IDs and validates every
derived training, evaluation-shard, and evaluation-merge name before it can be written.

Capacity failures are infrastructure attempts, not scientific runs. The first parallel replication
launches and their zero-step boundaries are preserved in `artifacts/scaling_launch_attempts_v1.json`.
The seed-59 stage found capacity on an alternate two-region pair without changing world size,
microbatch, accumulation, or effective batch. Seed 61 is now running as
`t-s61-527140-01b-v2` on the released nearby pair after its one-rank stockout attempt was stopped
with no ready marker or stage object.

Protocol amendment 005 pre-registers the checkpoint continuation gate before the first SFT metrics
exist. Relative to the prior valid checkpoint (the EXP002 starting adapter for step 500), a hard
perception regression means either the mean of two QA-exact and three grounding-click deltas is
below -3 percentage points or any one delta is below -7.5 points. Individual drops beyond 3 points
are warnings. Action-name, exact-action, and click-in-bounds recovery are reported separately and
cannot hide a hard perception regression. Training continues on warnings and stops only on a hard
regression, as implemented in `src/spider/scaling_gate.py`.

Validated SFT receipts enter `control_comparison_manifest_v1.json` through the lock-protected,
idempotent `spider-register-candidate` command. It rejects duplicate scaling identities and reused
adapter hashes, then the scaling report regenerates baseline, starting-adapter, per-checkpoint, and
across-seed tables from the registry. This lets parallel jobs finish in any order without silently
overwriting a comparison.
`spider-process-checkpoint` performs the registered regression gate, immutable gate write,
idempotent candidate registration, and atomic live-report refresh under one process lock. Every
stage controller must pass its validated evaluation receipt through this command before that job is
advanced.
`scripts/run_exp005_scaling_job.py` is the job-level supervisor: it adopts already-running stages,
processes every validated receipt, chains each later gate to that job's immediately preceding valid
checkpoint, and runs the remaining frozen stages only after a continue decision. Pair-specific
filesystem locks plus live regional GPU checks prevent two local supervisors from racing for the
same two-region training topology.

Cloud training stages support one, two, or four L4 GPUs (`g2-standard-8`, `g2-standard-24`, and
`g2-standard-48`). Gradient accumulation is adjusted to keep the effective batch size fixed at 16
(16, 8, or 4 accumulation steps respectively). A job cannot change its GPU world size between
resumable stages: the restored training state must match the requested GPU count and accumulation
before another optimizer step can run. This lets runtime scale without silently changing the SFT
recipe.

## Frozen data receipt

The authoritative receipt is `artifacts/dataset_freeze_v1.json`. It registers the full inventory,
all six manifests, nesting checks, leakage results, realized website composition, and hashes.
Amendment 003 records the pre-freeze whole-trajectory correction and the fixed 54/33/13 task mix.

## Materialized corpus

Only screenshots referenced by the frozen large tier and evaluation manifests were materialized
into one shared image store. The corrected run resolved 91,976 unique images across 1,654 source
groups, with zero missing images in every shard. The smaller tiers reference the same files rather
than copying images.

The evaluation controls are (1) untouched `Qwen/Qwen3.5-2B`, (2) the frozen EXP002 perception
adapter, and (3) each EXP005 reference SFT run. All controls use identical manifests, image
preprocessing, prompts, decoding, and scoring.

## Untouched-model baseline

The pinned untouched model is `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Its merged primary metrics are:

| Suite | Examples | QA exact | Grounding click | Action name | Exact action |
|---|---:|---:|---:|---:|---:|
| IID | 1,536 | 21.09% | 46.68% | 34.96% | 4.88% |
| Domain-balanced unseen websites | 1,536 | 26.95% | 61.72% | 31.45% | 2.54% |
| Distribution shift | 1,024 | — | 54.10% | 37.50% | 1.76% |

These are matched development controls, not sealed external benchmark claims. Individual shard
metrics, unweighted shard means, sample standard deviations, signatures, and artifact hashes are
preserved in the baseline receipt. Two infrastructure-only launch attempts checked out a mistyped
commit hash and stopped before inference; `artifacts/baseline_launch_incident_v1.json` records the
incident and the launcher guard added before the successful reruns.

## Matched starting-adapter control

The frozen EXP002 perception adapter was evaluated with the identical model revision, manifests,
prompts, decoding, and scoring. All twelve shards and three guarded merges completed in
`exp002-all-0822a`; the content-addressed adapter and all shard variability are preserved in
`artifacts/exp002_control_all_0822a.json` and `.md`.

| Suite | QA exact | QA token F1 | Grounding click | Ground median | Action name | Exact action |
|---|---:|---:|---:|---:|---:|---:|
| IID | 28.12% | 66.46% | 66.21% | 14.5 px | 0.20% | 0.00% |
| Domain-balanced unseen websites | 31.05% | 66.83% | 78.32% | 4.0 px | 0.00% | 0.00% |
| Distribution shift | — | — | 71.09% | 11.2 px | 0.00% | 0.00% |

Relative to the untouched model, the perception adapter improves QA exact by 7.03 and 4.10
percentage points on IID and unseen websites, and grounding click accuracy by 19.53, 16.60, and
16.99 points across the three suites. It also nearly eliminates valid browser-action generation,
which is expected for a perception-only adapter but makes this an essential matched starting-point
control: EXP005 SFT must retain the perception gains while recovering action behavior. The compact
control comparison is `artifacts/control_comparison_v1.{json,md}`.

## Multinode training gate

The nearby two-node L4 smoke completed at roughly 33 seconds per optimizer step with effective
batch size 16. A dispersed four-node run took roughly 63 seconds per step at the same effective
batch, so the scaling runs use two nodes and resumable stages of at most 500 steps. The initial
two-step compatibility run produced finite, changed adapter weights, but both steps were still at
zero learning rate during warmup. The subsequent 20-step smoke recovered from initial dynamic-loss-
scaling overflows, ended with ten consecutive finite gradient norms, and emitted a healthy adapter;
its receipt is `artifacts/numerical_smoke_v1.json`. This passes the final numerical-stability gate.
Every subsequent stage rejects non-finite adapter tensors and publishes an `adapter_health.json`
receipt before uploading the checkpoint.

The first 500-step launch was intentionally stopped before its first optimizer step when review
found that the multinode launcher had not exposed EXP004's previously validated L4 microbatch-2
execution setting. It produced no checkpoint or scientific result. The incident receipt is
`artifacts/training_launch_incident_v1.json`; a matched two-node benchmark must pass before the
stage is relaunched.

Protocol amendment 004 adopts per-device microbatch 2 with gradient accumulation 4 for the
two-node topology, preserving effective batch size 16. Its matched 20-step benchmark reduced stage
runtime from 648.2 to 383.4 seconds and ended with finite loss, finite final gradient norm, two
complete rank markers, and a healthy adapter. A separately materialized pinned-model cache avoids
repeated anonymous Hub downloads; its file-level hashes and immutable GCS object identity are in
`artifacts/model_cache_v1.json`.
The identical snapshot is also stored as 13 directly syncable objects so workers can avoid archive
decompression; its receipt is `artifacts/model_files_v1.json`.

## Warm evaluation image gate

The opt-in image `spider-exp005-eval-warm-qwen35-0822a` caches the pinned model, frozen corpus, and
exact evaluation environment. It is approved for later EXP005 validations only after a matched
EXP002/IID/shard-0 rerun produced byte-identical raw predictions, scored predictions, metrics, and
run metadata for all 384 examples. Evaluation setup fell from 1,345 to 182 seconds (7.39× faster),
saving 1,163 seconds per shard; inference time remained comparable. The immutable image identity,
GCS object generations, file hashes, timing, invalidation conditions, and the earlier fail-closed
sentinel incident are recorded in `artifacts/warm_eval_image_v1.json`,
`artifacts/warm_eval_image_gate_v1.json`, and `artifacts/warm_eval_image_attempt_v1.json`.
