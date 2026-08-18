# EXP002 SFT progress

Task metrics use the same fixed held-out validation probe: 128 ScreenshotQA examples and 128
grounding examples. Step 250 is the first post-SFT regression gate. After it passes, later task
probes are scheduled after stage 3 (step 1,000) and stage 7 (step 1,875); per-stage language-model
validation loss still provides a training health signal. This avoids tuning at every small stage
against the same probe and does not touch the frozen 5,272-example test suite or ScreenSpot until
step 1,875.

| Step | Runtime (h) | Train loss | Eval loss | Eval token acc. | QA exact | QA token F1 | Ground click acc. | Ground parse rate | Median distance | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | 0.3594 | 0.6047 | 0.1953 | 1.0000 | 222.8 px | baseline anchor |
| 250 | 1.854 | 0.6481 | 0.6321 | 0.8309 | 0.3594 | 0.6453 | 0.4688 | 1.0000 | 39.3 px | regression gate passed |
| 500 | 1.901 | 0.2853 | 0.6048 | 0.8308 | — | — | — | — | — | checkpoint validated; task probe deferred |
| 750 | 1.950 | 0.1780 | 0.5966 | 0.8331 | — | — | — | — | — | checkpoint validated; task probe deferred |
| 1000 | 1.754 | 0.1330 | 0.5897 | 0.8373 | invalid | invalid | 0.5156* | 1.0000* | 28.8 px* | checkpoint validated; probe rerun required |
| 1875 | scheduled | pending | pending | pending | scheduled | scheduled | scheduled | scheduled | scheduled | final validation probe |

`Train loss` is the stage aggregate reported by Trainer. Task metrics, rather than language-model
loss alone, determine improvement relative to the baseline. Runtime is training-stage wall time;
notebook startup and artifact publication add a few minutes.

At step 250, QA exact match was unchanged, QA token F1 increased 4.06 percentage points,
grounding click accuracy increased 27.34 points, and median click error decreased 183.5 pixels.
The 256-example probe is a development signal rather than a final estimate; the frozen test suite
remains reserved for step 1,875.

## Step-1,000 probe stop-token diagnosis

The first step-1,000 task probe is not an official QA result. Its raw QA exact accuracy of 0.0078
and token F1 of 0.2130 were caused by an evaluation stop-token mismatch, not evidence that the
checkpoint lost ScreenshotQA ability. The Qwen3.5 chat template terminates assistant turns with
token `<|im_end|>` (ID 248046), while the model text config exposes `<|endoftext|>` (ID 248044) as
its default EOS. Training correctly supervised `<|im_end|>`, but `generate()` received no explicit
chat end-of-turn stop ID. It therefore crossed the learned boundary into synthetic `user` and
`assistant` turns until reaching the 96-token cap. Decoding removed the special boundary tokens,
leaving apparently continuous role text in 126 of 128 QA outputs and all 128 grounding outputs.

As a diagnostic only, truncating the saved output before the first decoded next-turn marker
restores QA exact accuracy to 0.3672 and token F1 to 0.6874, compared with 0.3594 and 0.6453 at
step 250. This also explains why the probe ran unusually slowly. Grounding's permissive point
parser had already hidden the same issue in 71 of 128 step-250 outputs by accepting the first
coordinate and ignoring trailing turns. The starred step-1,000 grounding values above remain
useful first-coordinate diagnostics, but the corrected generation probe must be run before the
stage-3 regression gate is considered passed or stage 4 begins.
