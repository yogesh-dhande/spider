# Active experiment coordination

Last verified: 2026-08-22T03:45:20Z

This file is the mutable handoff for concurrent experiment work. Update it whenever a run starts,
stops, changes phase, or changes ownership. Durable working conventions belong in `AGENTS.md`, and
scientific history belongs in the relevant experiment record.

## EXP005

Status: active.

Corrected shared-corpus run `corpus-v2-0821a` completed and merged with zero missing screenshots.
The untouched `Qwen/Qwen3.5-2B` baseline `base-all-0821a` completed all twelve evaluation shards
and three guarded suite merges. Two- and four-node compatibility tests completed; the nearby
two-node topology was selected because the dispersed four-node topology was slower. All associated
VMs are terminated and deleted. The registered 20-step numerical smoke
`smoke20-small53-0822a` completed successfully: dynamic FP16 loss scaling recovered, the last ten
reported steps had finite gradient norms, both ranks emitted exact complete markers, and the final
adapter contained zero non-finite values. The first long reference attempt
`small53-s00000-e00500-0822a` was stopped before its first optimizer step after review found that
the launcher had not carried forward EXP004's validated microbatch-2 L4 setting. No checkpoint or
stage artifact was produced. Fixed-effective-batch two-node microbatch-2 benchmark
`smoke20b2-small53-0822a` completed successfully and reduced 20-step runtime from 648.2 to 383.4
seconds while preserving effective batch size 16 and a healthy adapter. Protocol amendment 004
adopts microbatch 2 for subsequent two-node stages. CPU run `model-cache-0822a` completed a
checksumed copy of the pinned base model in EXP005 GCS inputs so later workers avoid repeated
anonymous Hugging Face downloads.
The exact 10K/seed-53 stage was relaunched as `small53-s00000-e00500-0822b`, steps 0 through 500,
on the selected two-node topology using microbatch 2, gradient accumulation 4, and the pinned local
model snapshot.
All nine immutable scaling-job configurations are now staged under their content-derived GCS job
prefixes. The eight newly staged configurations were checked byte-for-byte against the local
matrix plan, and each parsed configuration matches its registered canonical configuration hash.
No additional training jobs were launched by this staging step.
All twelve EXP002-control shards and all three guarded suite merges completed with exact terminal
markers; every associated VM is terminated. The validated receipt is
`artifacts/exp002_control_all_0822a.{json,md}` and the baseline/control comparison is
`artifacts/control_comparison_v1.{json,md}`. CPU run
`model-files-0822a` completed the same frozen model as 13 directly syncable files, with the source
manifest hash preserved, to remove archive decompression from later worker setup.
The authoritative research status and receipts are under `experiments/exp005_browser_ablation_bed/`.

The local monitor remains active for `small53-s00000-e00500-0822b`; do not stop its VMs or switch
this worktree's branch. Require exact rank markers, training state, and adapter health before
launching matched step-500 validation. Checkpoint handoff controller
`run_exp005_checkpoint_validation.py` is also active for this stage: after both exact rank markers
arrive it will archive the training receipt, launch `sft-small53-step0500-0822a` across all three
frozen suites, archive its receipt and predictions, and require the evaluated adapter hash to match
the trained checkpoint. It does not authorize or launch the next training stage. The completed
control campaign's sparse state log is
`outputs/experiment5/controllers/exp002-all-0822a.jsonl`. The verified scaling plan hash is
`5a79bbb31c53970506dac1b9831d66b24852caa5df32cca4d331d6859417f1a6`. Preserve world size across
any resumed training stages.

Opt-in warm evaluation image `spider-exp005-eval-warm-qwen35-0822a` was created from the boot disk
of a successfully completed control shard. The first matched check failed closed before inference
because its model-file sentinel used the wrong pinned-snapshot filename; incident receipt
`artifacts/warm_eval_image_attempt_v1.json` records the CPU-only inspection and fix. Replacement
`warmcheck2-exp002-iid00-0822a` completed and its VM is terminated. All 384 raw predictions, scored
predictions, metrics, and run metadata are byte-identical to the frozen control shard. Setup fell
from 1,345 to 182 seconds (7.39×, saving 1,163 seconds per shard), so the image is approved for
future opt-in EXP005 evaluations subject to the invalidation conditions in
`artifacts/warm_eval_image_v1.json`. The already-running first step-500 controller remains on its
registered cold-image path.
Reusable runner `scripts/run_exp005_scaling_stage.py` now packages one exact resumable training
stage and its full matched validation, refuses ambiguous prior stage state, and guarantees a final
VM-stop sweep. Use separate invocations for parallel size/seed jobs; do not use it to replace the
already-running step-500 controller.
The full registered one-epoch schedule is now frozen as `artifacts/scaling_schedule_v2.json`
(SHA-256 identity `eec4b382e410c9cfc6b20e8c51b298d71f83dfd85bc046811eae55ef3f2c1679`): nine jobs,
57 at-most-500-step training stages, and full validation after every stage. The override file only
preserves the existing run IDs for the active small/seed-53/step-500 stage.
Schedule v1 is rejected: GCE refused the first seed-59 and seed-61 rank names before creating any
resource because launcher prefixes pushed them past the instance-name limit. Incident receipt
`artifacts/scaling_schedule_launch_incident_v1.json` records the zero-resource boundary. V2 uses
compact content-derived IDs and validates all derived instance names before launch.
Small/seed-59/step-500 replication `t-s59-4173bd-01-v2` is running on
`asia-south1-b` plus `asia-southeast1-c` after four alternative pairs reported L4 stockout. Its
stage controller will run all frozen suites with the approved warm image and stop both training and
evaluation resources. The first seed-61 attempt created one US rank before the partner zone reported
stockout; the fail-safe stopped it with zero ready markers, stage objects, or optimizer steps. Four
further European pairs reported stockout before creating resources. Retry seed 61 with a new
attempt-specific run ID after the currently healthy nearby northamerica pair is released. Exact
capacity history is in `artifacts/scaling_launch_attempts_v1.json`.
The seed-59 training workers are pinned to revision
`81b90a9301bc0d22c985d945288e02cf0ceae121`. Its initial local waiting controller was replaced
before any checkpoint or evaluation because it forwarded symbolic `HEAD`; the replacement pins the
same resolved revision for evaluation and did not interrupt either GPU worker. The stage runner now
resolves a revision once before launching or constructing any downstream command.
Seed-61 replacement controller is waiting for the current small/seed-53 training pair to leave all
active states, then will launch `t-s61-527140-01b-v2` on the same registered nearby regions at
revision `b8ffcd604c0c09cd9b2eab1f5299a5c269915826`. It will not touch the active seed-53
workers and will use its own warm validation and shutdown sweep.

Protected EXP005 scope:

- `configs/experiment5.yaml`
- `configs/datasets/exp005_*`
- `configs/ablations/experiment5_matrix.yaml`
- `experiments/exp005_browser_ablation_bed/`
- `data/exp005_browser_ablation_v1/`
- `outputs/experiment5/`
- `scripts/gcloud_exp005*`
- `gs://keptune-spider-experiments-1088401257609/exp005/`
- Cloud resources labelled `spider-experiment=exp005`

Do not modify these paths, switch the primary worktree's branch, stop its monitor, or reuse its
cloud/storage namespaces while unrelated work proceeds.

## EXP006

Status: planned; implementation and data generation have not started.

Objective: generate screenshot-only browser trajectories from green Playwright tests, freeze an
off-the-shelf-model baseline, and only then consider training. The initial target is a bounded form
workflow from the Ministry of Justice OPG Modernising LPA repository. The model must not receive
selectors, DOM, or accessibility-tree observations; selectors may be used only as privileged
teacher information during data generation.

EXP006 may proceed concurrently with EXP005 subject to these isolation requirements:

- Create a separate worktree, recommended at `/Users/yogesh/projects/spider-exp006`.
- Use a dedicated branch, recommended `exp006-playwright-baseline`.
- Use `data/exp006_playwright_portal_v0/` for generated data.
- Use `outputs/experiment6/` for evaluations and reports.
- Use `experiments/exp006_playwright_portal_baseline/` for the experiment record.
- Use EXP006-specific Docker Compose project names, host ports, temporary directories, run IDs,
  cloud labels, and storage prefixes.
- Keep initial data generation CPU/browser-only so it does not contend with EXP005 GPU work.
- Do not register, merge, or modify EXP005 artifacts as part of EXP006.

The proposed explicit `wait` action is adaptive: poll screenshots every 250 ms, return after a
visual change stabilizes for two frames, and time out after five seconds. Ordinary actions receive
a shorter automatic settle. Limit policies to three consecutive explicit waits before marking an
episode stalled.

## Preflight for a new session

Before touching either experiment, run read-only checks equivalent to:

```bash
git status --short
git worktree list
ps -axo pid,etime,command | rg 'spider|exp005|exp006|gcloud_exp005|gcloud_exp006'
```

Then verify that the paths and resource namespaces intended for the new work do not overlap an
active experiment. Update this document when the observed state differs from the entries above.
