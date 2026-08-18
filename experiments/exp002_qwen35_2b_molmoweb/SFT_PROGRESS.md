# EXP002 SFT progress

Task metrics use the same fixed held-out validation probe: 128 ScreenshotQA examples and 128
grounding examples. Step 250 is the first post-SFT regression gate. After it passes, later task
probes run only when requested; per-stage language-model validation loss still provides a training
health signal. This avoids repeatedly tuning against the small probe and does not touch the frozen
5,272-example test suite or ScreenSpot until step 1,875.

| Step | Runtime (h) | Train loss | Eval loss | Eval token acc. | QA exact | QA token F1 | Ground click acc. | Ground parse rate | Median distance | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | pending | pending | pending | pending | pending | baseline probe pending |
| 250 | 1.854 | 0.6481 | 0.6321 | 0.8309 | pending | pending | pending | pending | pending | checkpoint validated |

`Train loss` is the stage aggregate reported by Trainer. Task metrics, rather than language-model
loss alone, determine improvement relative to the baseline. Runtime is training-stage wall time;
notebook startup and artifact publication add a few minutes.
