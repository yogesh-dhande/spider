# EXP002 SFT progress

Task metrics use the same fixed held-out validation probe: 128 ScreenshotQA examples and 128
grounding examples. Step 250 is the first post-SFT regression gate. After the corrected step-1,000
probe showed diminishing gains, task probes are scheduled after every remaining stage (steps
1,250, 1,500, 1,750, and 1,875); per-stage language-model validation loss remains a training
health signal. The frozen 5,272-example test suite and ScreenSpot were evaluated exactly once on
the selected step-1,875 checkpoint after validation-based selection.

| Step | Runtime (h) | Train loss | Eval loss | Eval token acc. | QA exact | QA token F1 | Ground click acc. | Ground parse rate | Median distance | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | 0.3594 | 0.6047 | 0.1953 | 1.0000 | 222.8 px | baseline anchor |
| 250 | 1.854 | 0.6481 | 0.6321 | 0.8309 | 0.3594 | 0.6453 | 0.4688 | 1.0000 | 39.3 px | regression gate passed |
| 500 | 1.901 | 0.2853 | 0.6048 | 0.8308 | — | — | — | — | — | checkpoint validated; task probe deferred |
| 750 | 1.950 | 0.1780 | 0.5966 | 0.8331 | — | — | — | — | — | checkpoint validated; task probe deferred |
| 1000 | 1.754 | 0.1330 | 0.5897 | 0.8373 | 0.3672 | 0.6874 | 0.5156 | 1.0000 | 28.8 px | corrected regression gate passed |
| 1250 | 1.854 | 0.0962 | 0.5637 | 0.8396 | 0.3750 | 0.7000 | 0.5781 | 1.0000 | 23.9 px | regression gate passed |
| 1500 | 1.741 | 0.0760 | 0.5701 | 0.8427 | 0.3750 | 0.7059 | 0.5469 | 1.0000 | 32.3 px | mixed plateau; gate passed |
| 1750 | 1.777 | 0.0625 | 0.5667 | 0.8449 | 0.3828 | 0.7197 | 0.5625 | 1.0000 | 24.8 px | regression gate passed |
| 1875 | 0.906 | 0.0289 | 0.5663 | 0.8451 | 0.3828 | 0.7226 | 0.5703 | 1.0000 | 24.8 px | final probe passed; selected for test |

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

The corrected step-1,000 rerun stopped on both model EOS and chat end-of-turn and exactly matched
the saved-output diagnostic: QA exact 0.3672, QA token F1 0.6874, grounding click accuracy 0.5156,
parse rate 1.0000, and median grounding distance 28.8 pixels. This passes the step-250 regression
gate, but the small QA exact and later grounding gains indicate a plateau. At the user's request,
automatic continuation now pauses for the fixed task probe after every remaining stage.

At step 1,250, QA exact increased another 0.78 percentage points, token F1 increased 1.26
points, and grounding click accuracy increased 6.25 points relative to step 1,000. Median click
distance fell by 4.9 pixels. The checkpoint passed the regression gate and stage 5 began.

At step 1,500, QA exact was unchanged and token F1 increased 0.59 percentage points relative to
step 1,250. Grounding click accuracy decreased 3.13 points and median distance increased 8.4
pixels on the 128-example probe. This is a mixed plateau signal rather than a broad regression:
all metrics remain substantially above the step-250 gate anchor, parse rate remains perfect, and
the probe is small. Stage 6 therefore continues as planned, while step 1,250 remains the best
grounding checkpoint observed so far and all checkpoints remain eligible for final selection.

At step 1,750, QA exact increased 0.78 percentage points and token F1 increased 1.39 points
relative to step 1,500. Grounding click accuracy recovered 1.56 points and median distance fell
7.5 pixels. This checkpoint has the best QA exact and token F1 observed so far; step 1,250 still
has the highest click-in-bounds accuracy by 1.56 points and the smallest median distance by 0.9
pixels. The regression gate passed and the shortened final stage began toward step 1,875.

At step 1,875, QA exact remained unchanged, token F1 increased 0.28 percentage points, and
grounding click accuracy increased 0.78 points relative to step 1,750. Median distance remained
24.8 pixels. Step 1,875 has the best QA token F1, ties step 1,750 for best QA exact, and is one
grounding example below step 1,250's peak click-in-bounds result. Because the final checkpoint
improved the combined task signal without a primary-metric regression, it is selected for the
one-time frozen full-test evaluation.
