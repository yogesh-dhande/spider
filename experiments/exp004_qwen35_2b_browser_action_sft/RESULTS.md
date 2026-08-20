# EXP004 results

## Dataset realized counts

| Partition | Action | ScreenshotQA | Grounding |
|---|---:|---:|---:|
| Train | 20000 | 5000 | 5000 |
| Validation | 512 | 256 | 256 |
| Sealed test | 1024 | 2000 | 2000 |

## Development action baselines

| Model | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| Untouched Qwen3.5-2B | 256 | 74.22% | 41.02% | 9.38% | 4.55% | 175.6 px |
| EXP002 perception adapter | 256 | 0.39% | 0.00% | 0.00% | 0.00% | — |

### Untouched-base shard diagnostics

| Shard | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| action-base-shard-00-of-02 | 128 | 73.44% | 42.19% | 9.38% | 2.17% | 208.1 px |
| action-base-shard-01-of-02 | 128 | 75.00% | 39.84% | 9.38% | 7.14% | 162.3 px |

### EXP002-parent shard diagnostics

| Shard | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| action-exp002-shard-00-of-02 | 128 | 0.00% | 0.00% | 0.00% | 0.00% | — |
| action-exp002-shard-01-of-02 | 128 | 0.78% | 0.00% | 0.00% | 0.00% | — |

## Stage validation trajectory

| Step | Action name | Arguments | Click in bounds | QA exact | Grounding click | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 125 | 50.39% | 32.42% | 29.55% | 39.06% | 61.72% | advance |
| 250 | 71.09% | 50.78% | 29.55% | 37.50% | 54.69% | advance |
| 375 | 69.14% | 48.05% | 38.64% | 36.72% | 56.25% | stop |
| 500 | 71.48% | 48.83% | 38.64% | 38.28% | 62.50% | stop |

## Selected checkpoint

Step **500**, selected only from the fixed development probes by the preregistered lexicographic rule.

## Sealed test

### Browser actions

| Model | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| EXP002 parent | 1024 | 2.64% | 1.66% | 0.00% | 0.00% | 908.9 px |
| EXP004 step 500 | 1024 | 100.00% | 64.26% | 42.97% | 30.95% | 61.8 px |

### Perception retention

| Model | QA exact | QA token F1 | Grounding click | Median grounding error |
|---|---:|---:|---:|---:|
| EXP002 parent | 26.20% | 0.6103 | 73.40% | 8.5 px |
| EXP004 step 500 | 25.55% | 0.6035 | 72.05% | 9.6 px |

Preregistered positive-result gate: **PASS**.

### Matched EXP002-parent action shard diagnostics

| Shard | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| final-action-exp002-shard-00-of-04 | 256 | 1.95% | 0.78% | 0.00% | 0.00% | — |
| final-action-exp002-shard-01-of-04 | 256 | 3.52% | 2.34% | 0.00% | 0.00% | — |
| final-action-exp002-shard-02-of-04 | 256 | 1.17% | 0.78% | 0.00% | 0.00% | — |
| final-action-exp002-shard-03-of-04 | 256 | 3.91% | 2.73% | 0.00% | 0.00% | 908.9 px |

### Selected-checkpoint action shard diagnostics

| Shard | N | JSON parse | Action name | Arguments | Click in bounds | Median click error |
|---|---:|---:|---:|---:|---:|---:|
| final-action-step-0500-shard-00-of-04 | 256 | 100.00% | 62.11% | 45.70% | 37.25% | 25.7 px |
| final-action-step-0500-shard-01-of-04 | 256 | 100.00% | 63.28% | 41.41% | 30.48% | 116.3 px |
| final-action-step-0500-shard-02-of-04 | 256 | 100.00% | 66.80% | 46.09% | 27.84% | 56.9 px |
| final-action-step-0500-shard-03-of-04 | 256 | 100.00% | 64.84% | 38.67% | 28.45% | 93.8 px |

### Selected-checkpoint perception shard diagnostics

| Shard | QA N | QA exact | Grounding N | Grounding click | Median error |
|---|---:|---:|---:|---:|---:|
| final-perception-step-0500-shard-00-of-04 | 500 | 27.60% | 500 | 72.60% | 7.8 px |
| final-perception-step-0500-shard-01-of-04 | 500 | 24.20% | 500 | 72.00% | 9.7 px |
| final-perception-step-0500-shard-02-of-04 | 500 | 27.00% | 500 | 71.40% | 11.2 px |
| final-perception-step-0500-shard-03-of-04 | 500 | 23.40% | 500 | 72.20% | 9.7 px |

## Deterministic closed loop

| Variant | Episodes | Success rate | Mean reward | Mean steps | Parse error rate |
|---|---:|---:|---:|---:|---:|
| exp002_parent | 12 | 0.00% | -0.6000 | 6.00 | 100.00% |
| exp004_selected | 12 | 0.00% | -0.2250 | 6.00 | 0.00% |

| Candidate vs control | Paired N | Success delta | 95% paired bootstrap CI | Reward delta |
|---|---:|---:|---:|---:|
| exp004_selected vs exp002_parent | 12 | +0.00% | [+0.00%, +0.00%] | +0.3750 |

## Reproducibility artifacts

Full matched sealed predictions are stored under `artifacts/final_test/predictions/`.
The realized dataset configuration and SHA-256 manifest are stored under `artifacts/data/`; per-stage runtime versions, optimizer state history, and Trainer logs are stored under `artifacts/training_stages/`.
Deterministic visual and machine-readable diagnostic samples are stored under `artifacts/final_test/failures/`; action errors are separated into output-format, semantic-action, action-argument, and spatial-grounding buckets, while perception errors distinguish OCR, semantic-understanding, output-format, and spatial-grounding.
The dashboard payload used for the baseline-versus-selected visual comparison is archived at `artifacts/final_test/dashboard.json`.
