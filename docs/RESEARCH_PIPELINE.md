# Browser-agent research pipeline

## Objective

Move from an architecture, data, reward, or training idea to a paired ablation result with the
smallest possible amount of one-off code. The pipeline begins with deterministic browser sandboxes
and must transfer from local machines and Kaggle to Google Cloud without changing scientific logic.

## Portability boundary

Core Python owns task definitions, policy adapters, action parsing, rollouts, rewards, replay
records, aggregation, and evaluation. Compute platforms only launch the same CLI inside a pinned
container and provide paths or object-store credentials.

- Kaggle notebooks remain thin installation and command launchers.
- Local runs use the filesystem artifact store.
- Google Cloud will add a GCS artifact-store adapter and job launcher, not a separate trainer or
  evaluator.
- Every task/episode has a stable ID and deterministic seed, so rollout jobs can be sharded across
  machines and compared pairwise.
- Replay uses JSONL metadata and content-addressed screenshots. These files map directly to GCS
  objects and do not rely on notebook mount paths.

## Idea-to-ablation contract

An ablation is one base configuration plus small variant overlays. All variants receive the same
task IDs and seeds. The runner emits per-episode replay, aggregate metrics, paired deltas, and a
paired bootstrap interval.

```yaml
base:
  policy:
    type: oracle
    coordinate_bias_px: [0, 0]
variants:
  - id: control
    overlay: {}
  - id: changed
    overlay:
      policy:
        coordinate_bias_px: [120, 0]
```

The intended budget ladder is:

1. `unit`: parsers, reward functions, config resolution, and environment transitions on CPU.
2. `smoke`: 20–50 deterministic episodes and at most a tiny model update.
3. `pilot`: a fixed development task suite and bounded, resumable single-stage training.
4. `confirm`: multiple paired seeds on the full development suite.
5. `final`: one run on sealed external browser benchmarks after all choices are frozen.

Only ideas that pass a cheaper tier advance. Final benchmarks never select configurations.

## Training-size and parallel matrix contract

Training-data scaling uses nested manifests, not independent samples. The default EXP005 ladder is
10K, 30K, and 100K examples with `small ⊂ medium ⊂ large`; all tiers reference one shared image
store and use identical evaluation manifests. This keeps a size ablation from also changing the
websites or examples available at smaller scales.

`spider-ablation-matrix` resolves a base config plus recipe overlays into a deterministic
`recipe × dataset_size × seed` plan. Each job has a content-derived identity and isolated config,
logs, checkpoints, and output directory. A plan is immutable once materialized. Atomic job claims
support workers on a shared filesystem; independent cloud workers use deterministic shard indices.
Training hashes the resolved config and train/validation manifests and refuses to reuse an output
directory with a different identity. Matrix planning itself refuses to start until every selected
training manifest and the fixed validation manifest match their hashes in `dataset_ladder.json`.
Failed or abandoned claims are never stolen automatically; an explicit `requeue` archives the old
claim, result, and log before making the job available again.

Most ideas screen on one seed and one size. Only promising frozen recipes advance to the full
three-size, three-seed scaling matrix. Fixed-epoch scaling measures the practical data-plus-compute
curve; a separately labeled fixed-update-budget study is required to isolate data diversity from
additional optimization.

Build and inspect the nested manifests:

```bash
spider-build-data-ladder --config configs/datasets/exp005_browser_ablation_v1.yaml
```

Materialize a matrix without launching compute:

```bash
spider-ablation-matrix plan \
  --config configs/ablations/experiment5_matrix.yaml \
  --budget smoke
```

Parallel workers may then claim jobs from the same plan root, or receive disjoint
`--shard-index/--num-shards` assignments. Matrix planning and data construction remain CPU-only;
GPU jobs do not start implicitly. Job paths are relative to the plan root so a plan archive is
portable; a cloud worker supplies its mounted corpus with `worker --data-dir /mounted/corpus`.

## Current vertical slice

`spider-study` currently provides:

- strict JSON browser actions using normalized coordinates;
- a deterministic screenshot-producing browser sandbox;
- a policy adapter boundary with an oracle test implementation;
- an HTTP policy adapter that can target a local model server or managed GPU endpoint;
- raw environment signals composed by configurable reward weights;
- content-addressed screenshot artifacts and resumable JSONL episodes;
- paired task/seed variants with automatic comparison output;
- stable task sharding for future parallel workers;
- immutable run IDs protected by a full-config checksum.

Run the smoke ablation:

```bash
spider-study --config configs/studies/sandbox_coordinate_bias.yaml
```

Run the identical CLI in the portable research container:

```bash
docker build \
  --build-arg SPIDER_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -f containers/research.Dockerfile \
  -t spider-study:dev .
docker run --rm \
  -v "$PWD/outputs/container:/artifacts" \
  spider-study:dev \
  --config configs/studies/sandbox_coordinate_bias.yaml \
  --run-id container-smoke \
  --output-dir /artifacts
```

The mounted output directory is the only platform-specific part. A Google Cloud job will mount or
upload the same artifact tree to GCS and invoke the same entrypoint.

The real-browser integration image pins Playwright and its Chromium runtime to the same version:

```bash
docker build \
  --build-arg SPIDER_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -f containers/browser.Dockerfile \
  -t spider-browser-study:dev .
docker run --rm --init --ipc=host \
  -v "$PWD/outputs/browser-container:/artifacts" \
  spider-browser-study:dev \
  --config configs/studies/playwright_coordinate_bias.yaml \
  --run-id container-smoke \
  --output-dir /artifacts
```

The oracle is only an infrastructure probe. The next adapter will call the frozen Qwen policy using
the same observation and action contract. BrowserGym/Playwright environments and model training
algorithms will be added behind their respective interfaces after the deterministic loop is proven.

Remote policies are selected through config without embedding service addresses or credentials:

```yaml
policy:
  type: http
  id: qwen35-2b-exp002-step-1250
  endpoint_env: SPIDER_POLICY_ENDPOINT
  bearer_token_env: SPIDER_POLICY_TOKEN
  timeout_seconds: 120
  max_history: 4
```

The request includes the task, current URL, image dimensions, base64 screenshot, deterministic seed,
and bounded previous outputs. The service returns either a raw text `output` or parsed `action`
object. The policy ID must include immutable model/checkpoint identity.

Parallel workers write isolated deterministic shard directories. Merge only after every shard is
complete and validated:

```bash
spider-study --config study.yaml --run-id pilot --shard-index 0 --num-shards 8
# Run indices 1–7 on other workers using the same command/config.
spider-study-merge --run-root outputs/studies/<study-id>/pilot
```

## Planned RL loop

Under Kaggle's session limit, start with an asynchronous cycle:

1. Roll out a versioned policy in deterministic sandboxes.
2. Score outcomes and reward components.
3. Persist compact replay shards.
4. Train one bounded, resumable SFT, rejection-finetuning, preference, or GRPO stage.
5. Evaluate against paired seeds and promote only if gates pass.

On Google Cloud, rollout workers, GPU training jobs, and evaluation workers can scale independently
while using the same task manifests, policy checkpoints, replay schema, and study configs.
