# Repository guidance

- Keep this file limited to durable working conventions. Record changing experiment state in
  `docs/ACTIVE_EXPERIMENTS.md`, never here.
- Before starting, resuming, or modifying an experiment, read `docs/ACTIVE_EXPERIMENTS.md` and the
  experiment's record under `experiments/`.
- Treat the paths, processes, cloud resources, and namespaces listed for an active experiment as
  protected unless the task explicitly targets that experiment.
- Run concurrent experiments from separate Git worktrees and branches. Give each experiment its
  own data directory, output directory, experiment record, container project/ports, cloud labels,
  storage prefix, and run identifiers.
- Do not switch branches in a worktree that has an active monitor, launcher, materializer, trainer,
  evaluator, or other long-running process. Do not stop or replace such a process unless the task
  explicitly requires it.
- Before launching parallel work, check `git status`, `git worktree list`, relevant local
  processes, and the active-experiment status document. Resolve any namespace or resource collision
  before proceeding.
- When experiment state changes, update `docs/ACTIVE_EXPERIMENTS.md` with the UTC timestamp,
  evidence, paths, and safe handoff instructions. Keep historical scientific decisions and
  immutable receipts in the corresponding `experiments/<id>/` record.
