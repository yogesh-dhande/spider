# Active experiment coordination

Last verified: 2026-08-21T23:39:31Z

This file is the mutable handoff for concurrent experiment work. Update it whenever a run starts,
stops, changes phase, or changes ownership. Durable working conventions belong in `AGENTS.md`, and
scientific history belongs in the relevant experiment record.

## EXP005

Status: active.

Corrected shared-corpus run `corpus-v2-0821a` completed and merged with zero missing screenshots.
The untouched `Qwen/Qwen3.5-2B` baseline `base-all-0821a` completed all twelve evaluation shards
and three guarded suite merges. All associated VMs are terminated. The next authorized phase is a
two-node, two-optimizer-step compatibility/speed smoke before any long SFT stage. No EXP005
training has started. The authoritative research status and receipts are under
`experiments/exp005_browser_ablation_bed/`.

The baseline monitor exited after verifying shutdown. There is no expected local EXP005 monitor at
this handoff. Before starting SFT, verify the scaling plan hash
`5a79bbb31c53970506dac1b9831d66b24852caa5df32cca4d331d6859417f1a6`, upload the selected job
configuration, and run the two-node compatibility smoke. Preserve world size across any resumed
training stages.

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
