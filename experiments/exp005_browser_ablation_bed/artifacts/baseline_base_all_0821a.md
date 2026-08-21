# Evaluation receipt: base-all-0821a

Control: `base`. Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`.

## iid

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1536 examples) | 21.09% | 53.84% | 46.68% | 59.6 px | 34.96% | 4.88% | 7.55% |
| Unweighted shard mean ± sample SD | 21.14 ± 3.75% | 53.79 ± 1.44% | 46.76 ± 4.11% | 62.62 ± 8.69 px | 34.88 ± 4.30% | 4.84 ± 2.29% | 7.58 ± 2.80% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 384 | 26.32% | 44.09% | 68.3 px | 35.48% | 1.61% | 7.55% |
| 1 | 384 | 18.57% | 52.85% | 53.2 px | 28.93% | 6.61% | 5.45% |
| 2 | 384 | 18.18% | 45.67% | 57.5 px | 39.20% | 4.80% | 5.77% |
| 3 | 384 | 21.50% | 44.44% | 71.5 px | 35.92% | 6.34% | 11.54% |

## domain_balanced

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1536 examples) | 26.95% | 54.76% | 61.72% | 17.0 px | 31.45% | 2.54% | 1.09% |
| Unweighted shard mean ± sample SD | 27.06 ± 4.06% | 54.83 ± 4.55% | 61.59 ± 5.13% | 19.07 ± 4.53 px | 31.49 ± 2.62% | 2.53 ± 0.93% | 1.14 ± 1.31% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 384 | 29.91% | 62.33% | 21.9 px | 32.06% | 3.82% | 2.27% |
| 1 | 384 | 29.71% | 56.91% | 22.7 px | 34.96% | 1.63% | 2.27% |
| 2 | 384 | 21.21% | 58.62% | 19.1 px | 29.41% | 2.21% | 0.00% |
| 3 | 384 | 27.41% | 68.50% | 12.7 px | 29.51% | 2.46% | 0.00% |

## distribution_shift

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1024 examples) | — | — | 54.10% | 39.7 px | 37.50% | 1.76% | 1.12% |
| Unweighted shard mean ± sample SD | — | — | 54.19 ± 4.42% | 40.30 ± 9.33 px | 37.49 ± 4.24% | 1.74 ± 1.46% | 1.16 ± 1.48% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256 | — | 56.20% | 30.7 px | 35.56% | 1.48% | 0.00% |
| 1 | 256 | — | 48.85% | 44.8 px | 33.60% | 0.80% | 3.08% |
| 2 | 256 | — | 52.63% | 51.1 px | 37.40% | 0.81% | 0.00% |
| 3 | 256 | — | 59.06% | 34.7 px | 43.41% | 3.88% | 1.56% |
