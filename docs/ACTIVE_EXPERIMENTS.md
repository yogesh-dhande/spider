# Active experiment coordination

Last verified: 2026-08-22T01:47:00Z

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
Five additional EXP002-control shards were launched in recycled regions, bringing eleven of twelve
shard identities to completed or active. Only distribution-shift shard 0 remains uncreated after
repeated regional L4 stockouts. CPU run `model-files-0822a` completed the same frozen model as 13
directly syncable files, with the source manifest hash preserved, to remove archive decompression
from later worker setup.
The authoritative research status and receipts are under `experiments/exp005_browser_ablation_bed/`.

Local monitors are active for `small53-s00000-e00500-0822b` and `exp002-all-0822a`; do not stop
their VMs or switch this worktree's branch. Require exact rank markers, training state, and adapter
health before launching matched step-500 validation. A capacity-aware controller is active for
`exp002-all-0822a`; it validates exact shard markers, retries the final uncreated shard across free
regions, and will launch all three guarded merges after full coverage. Its sparse state log is
`outputs/experiment5/controllers/exp002-all-0822a.jsonl`. The verified scaling plan hash is
`5a79bbb31c53970506dac1b9831d66b24852caa5df32cca4d331d6859417f1a6`. Preserve world size across
any resumed training stages.

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
