# EXP003 — portable browser-action and RL pipeline

Status: deterministic infrastructure smoke implemented while EXP002 finishes. No learned browser
action policy has been trained, and no agent benchmark result is claimed yet.

Parent: EXP002 perception checkpoint.

## Objective

Build a fast idea-to-ablation pipeline for browser-agent architecture and training research. The
first learned experiment will turn the best EXP002 checkpoint into a browser action executor that
can operate independently on simple tasks or under concrete subgoals from a stronger planner.

## Decisions

- Browser use only; desktop and professional-application environments remain out of scope.
- Begin with deterministic sandbox tasks and exact programmatic verifiers.
- Keep the core pipeline independent of Kaggle, local paths, and future Google Cloud launchers.
- Use strict JSON actions made from the model's existing text tokens; do not add custom action
  tokens yet.
- Express ablations as overlays on one base config and compare them on paired task IDs and seeds.
- Store content-addressed screenshots and resumable JSONL replay suitable for local files or GCS.
- Use thin notebooks and keep environments, policies, rewards, training, and evaluation in tested
  Python modules.
- Advance ideas through unit, smoke, pilot, confirmation, and sealed-final budget tiers.

## Infrastructure smoke

The first study intentionally compares an oracle action policy with centered clicks against the
same policy with a 120-pixel horizontal bias. This is not a model-quality experiment. It verifies
that the pipeline detects an expected grounding regression while exercising screenshot capture,
strict action parsing, deterministic transitions, decomposed rewards, replay persistence, resume,
paired aggregation, bootstrap intervals, and immutable config checks.

The clean-tree `smoke_v2` run at source commit `a1e6530` passed end to end. The centered control
completed 12/12 episodes and the offset variant completed 3/12, producing the expected -0.75 paired
success-rate delta with a 95% paired bootstrap interval of [-1.00, -0.50]. Its compact immutable
record is under `runs/20260818_smoke_v2/`; full replay and content-addressed screenshots remain in
the ignored local output tree.

Configuration: [`sandbox_coordinate_bias.yaml`](../../configs/studies/sandbox_coordinate_bias.yaml)

Design details: [`RESEARCH_PIPELINE.md`](../../docs/RESEARCH_PIPELINE.md)

## Next implementation increment

Add a Qwen policy adapter and a BrowserGym/Playwright environment adapter behind the proven
interfaces. Run a small frozen-checkpoint action baseline before any trajectory fine-tuning, then
train a balanced MolmoWeb trajectory pilot with perception replay.
