# EXP005 SFT scaling comparison

Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`.

## iid

| Run | QA exact | QA token F1 | Ground click | Ground median | Action name | Action exact | Action click |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched | 21.09% | 53.84% | 46.68% | 59.6 px | 34.96% | 4.88% | 7.55% |
| EXP002 starting adapter | 28.12% | 66.46% | 66.21% | 14.5 px | 0.20% | 0.00% | 0.00% |

## domain_balanced

| Run | QA exact | QA token F1 | Ground click | Ground median | Action name | Action exact | Action click |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched | 26.95% | 54.76% | 61.72% | 17.0 px | 31.45% | 2.54% | 1.09% |
| EXP002 starting adapter | 31.05% | 66.83% | 78.32% | 4.0 px | 0.00% | 0.00% | 0.00% |

## distribution_shift

| Run | QA exact | QA token F1 | Ground click | Ground median | Action name | Action exact | Action click |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched | — | — | 54.10% | 39.7 px | 37.50% | 1.76% | 1.12% |
| EXP002 starting adapter | — | — | 71.09% | 11.2 px | 0.00% | 0.00% | 0.00% |
