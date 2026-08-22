# Evaluation receipt: exp002-all-0822a

Control: `exp002`. Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`.

Adapter SHA-256: `f0039802ff9ca298474628aa906988d6d07f98d557803ac160559e1c53fc83e0`.

## iid

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1536 examples) | 28.12% | 66.46% | 66.21% | 14.5 px | 0.20% | 0.00% | 0.00% |
| Unweighted shard mean ± sample SD | 28.04 ± 4.32% | 66.39 ± 1.71% | 66.17 ± 5.60% | 13.38 ± 4.54 px | 0.20 ± 0.40% | 0.00 ± 0.00% | 0.00 ± 0.00% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 384 | 33.83% | 66.14% | 17.4 px | 0.00% | 0.00% | 0.00% |
| 1 | 384 | 24.29% | 69.92% | 10.7 px | 0.00% | 0.00% | 0.00% |
| 2 | 384 | 28.79% | 58.27% | 17.1 px | 0.80% | 0.00% | 0.00% |
| 3 | 384 | 25.23% | 70.37% | 8.4 px | 0.00% | 0.00% | 0.00% |

## domain_balanced

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1536 examples) | 31.05% | 66.83% | 78.32% | 4.0 px | 0.00% | 0.00% | 0.00% |
| Unweighted shard mean ± sample SD | 31.21 ± 3.38% | 66.92 ± 3.97% | 78.09 ± 3.76% | 4.03 ± 0.41 px | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 384 | 34.58% | 81.51% | 3.7 px | 0.00% | 0.00% | 0.00% |
| 1 | 384 | 31.88% | 75.61% | 3.9 px | 0.00% | 0.00% | 0.00% |
| 2 | 384 | 26.52% | 74.14% | 4.6 px | 0.00% | 0.00% | 0.00% |
| 3 | 384 | 31.85% | 81.10% | 3.9 px | 0.00% | 0.00% | 0.00% |

## distribution_shift

| Aggregate | QA exact | QA token F1 | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged (1024 examples) | — | — | 71.09% | 11.2 px | 0.00% | 0.00% | 0.00% |
| Unweighted shard mean ± sample SD | — | — | 71.12 ± 1.35% | 10.76 ± 2.64 px | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% |

| Shard | N | QA exact | Grounding click | Ground median | Action name | Action exact | Action click in bounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256 | — | 72.73% | 11.8 px | 0.00% | 0.00% | 0.00% |
| 1 | 256 | — | 69.47% | 13.5 px | 0.00% | 0.00% | 0.00% |
| 2 | 256 | — | 71.43% | 10.5 px | 0.00% | 0.00% | 0.00% |
| 3 | 256 | — | 70.87% | 7.3 px | 0.00% | 0.00% | 0.00% |
