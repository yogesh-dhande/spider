# EXP005 — trustworthy browser-training ablation bed

Status: infrastructure setup. No EXP005 training or model-quality claim has started.

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
| Small | 6,000 | 2,000 | 2,000 | 10,000 |
| Medium | 18,000 | 6,000 | 6,000 | 30,000 |
| Large | 60,000 | 20,000 | 20,000 | 100,000 |

The manifests must satisfy `small ⊂ medium ⊂ large`. They reference one shared image store; the
dataset is not copied three times. A deterministic diversity order is sampled from metadata
inventoried across every eligible pinned shard. Every tier is audited for source, task, action,
website, trajectory, screenshot, effective-domain count, and website concentration. The maximum
combined contribution from one registrable domain is 2%.

Unknown domains are rejected. Complete trajectories/screenshots remain within one split. Training
allows at most four action steps per trajectory, three QA examples per screenshot, and one grounding
example per screenshot during ladder selection. The QA cap is required because the pinned source has
roughly 10,000 screenshots but many questions per screenshot. The exact realized manifests,
configuration, source revisions, and
SHA-256 checksums are immutable publication artifacts.

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

## Next data-build phase

Inventory metadata across every eligible pinned MolmoWeb shard/config without downloading images.
Record generator/task-template fields needed for the distribution-shift split, audit capacity under
the 2% website cap, freeze the candidate/evaluation manifests, and only then download the selected
shared screenshots.
