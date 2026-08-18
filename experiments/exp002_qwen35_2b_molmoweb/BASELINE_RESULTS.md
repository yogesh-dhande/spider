# EXP002 pre-SFT baseline results

Model: `Qwen/Qwen3.5-2B` at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, evaluated without an adapter using greedy,
non-thinking, 4-bit inference.

Evaluation artifact: [Kaggle baseline merge version 2](https://www.kaggle.com/code/yogeshkd/spider-exp002-baseline-merge)

## Primary merged results

| Evaluation task | Examples | Primary accuracy |
|---|---:|---:|
| MolmoWeb ScreenshotQA exact answer | 2,000 | 22.40% |
| MolmoWeb GUI grounding click inside bounds | 2,000 | 56.65% |
| ScreenSpot GUI grounding click inside bounds | 1,272 | 47.17% |

These merged metrics over all 5,272 examples are authoritative. The shard statistics below are
deterministic partition diagnostics, not independent training seeds or confidence intervals.

## Individual shard results and variability

Each shard contains 250 MolmoWeb QA, 250 MolmoWeb grounding, and 159 ScreenSpot examples.

| Shard | QA answer accuracy | MolmoWeb click accuracy | ScreenSpot click accuracy |
|---:|---:|---:|---:|
| 0 | 23.60% | 60.80% | 48.43% |
| 1 | 19.20% | 57.20% | 46.54% |
| 2 | 23.60% | 56.80% | 48.43% |
| 3 | 20.80% | 57.20% | 44.65% |
| 4 | 26.00% | 55.20% | 48.43% |
| 5 | 23.20% | 56.40% | 54.09% |
| 6 | 23.20% | 52.00% | 46.54% |
| 7 | 19.60% | 57.60% | 40.25% |
| Mean | 22.40% | 56.65% | 47.17% |
| Population standard deviation | 2.17 pp | 2.31 pp | 3.67 pp |
| Minimum–maximum | 19.20–26.00% | 52.00–60.80% | 40.25–54.09% |

## Diagnostics

- ScreenshotQA mean token F1: 49.64%. Exact accuracy by question type: OCR 42.62%
  (1,044 examples), affordance 0.52% (573), and summarization 0.00% (383).
- MolmoWeb grounding: 99.95% parse rate, 99.90% on-grid rate, 31.15 px median distance,
  and 47.10% / 57.00% / 68.15% within 25 / 50 / 100 px.
- ScreenSpot grounding: 99.61% parse/on-grid rate, 50.99 px median distance, and 32.78% /
  48.98% / 69.65% within 25 / 50 / 100 px. Click accuracy is 39.13% for icons and 53.80%
  for text elements.
- MolmoWeb mean pixel distance is 1,416,844.84 px because one otherwise parseable model output
  emitted `[100, 4000000000]`, producing a 2.832-billion-pixel error. Median and fixed-threshold
  distances are the meaningful robust diagnostics; click accuracy is unaffected.

## Validation and saved failures

The merge independently enforced exact per-shard expected IDs, no duplicate or overlapping IDs,
complete equality with the frozen 5,272-ID test set, matching model/revision/adapter/split, and
matching prediction signatures. A second local audit repeated the full ID-set and count checks.

The scored predictions automatically categorize 599 QA failures as OCR, 953 as semantic
understanding, 866 MolmoWeb grounding failures and 667 ScreenSpot failures as spatial grounding,
and 1 MolmoWeb plus 5 ScreenSpot failures as output-format errors. These labels are diagnostic
heuristics, not human-adjudicated causes. The Kaggle artifact retains all raw/scored predictions,
the HTML failure report, and 136 representative failure images.

This baseline does not answer the experiment's causal question by itself; it is the fixed pre-SFT
anchor for the later small-data QLoRA comparison. No fine-tuning was started during this stage.
