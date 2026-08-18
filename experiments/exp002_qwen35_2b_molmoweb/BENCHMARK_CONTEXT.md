# Benchmark context for EXP002

Research checked on 2026-08-18. Scores from different datasets or evaluation protocols are
context only and must not be presented as directly comparable to the EXP002 domain-held-out
MolmoWeb probe.

## Exact MolmoWeb synthetic datasets

- `MolmoWeb-SyntheticQA` publishes a training split partitioned by website, not a fixed public
  benchmark test split or leaderboard. The paper describes 2,237,252 QA pairs across 395 sites.
- `MolmoWeb-SyntheticGround` is likewise released as training data. The paper describes more than
  7M synthetic grounding pairs, plus 1.1M repurposed PixMoPoints examples in its training mixture.
- The MolmoWeb paper does not report ScreenshotQA answer accuracy or synthetic-ground click
  accuracy on a held-out split. Consequently, there is no defensible published "best score" for
  our exact two data sources.

The strongest apples-to-apples control is therefore to evaluate released `MolmoWeb-4B` and
`MolmoWeb-8B` checkpoints on our immutable validation/test manifests with our evaluator. This is a
recommended follow-up after EXP002, not a replacement for the frozen Qwen baseline.

## GUI grounding reference: ScreenSpot-v2

ScreenSpot-v2 is the closest widely used external benchmark for element-description-to-click
grounding.

| Reference | Parameters | Click accuracy |
|---|---:|---:|
| UI-Venus-72B, current public leaderboard leader | 72B | 95.3% |
| Best explicitly 2B entry on the current leaderboard, ShowUI | 2B | 77.3% |
| MolmoWeb-Ground-8B, MolmoWeb paper | 8B | 91.8% |
| MolmoWeb-4B general browser agent, MolmoWeb paper | 4B | 89.5% |

The current leaderboard and MolmoWeb paper tables are not identical snapshots of the benchmark,
so each comparison must name its source and version. EXP002's existing 47.2% external result is on
the original ScreenSpot evaluation, not ScreenSpot-v2, and should only be compared with original
ScreenSpot results. In the MolmoWeb paper, MolmoWeb-Ground-8B scores 88.7% on original ScreenSpot.

Sources:

- https://gui-agent.github.io/grounding-leaderboard/screenspot
- https://arxiv.org/abs/2604.08516 (Table 8)

## Screenshot-QA references

### ScreenQA Short

ScreenQA Short is a strong nearby benchmark: short-answer QA over mobile UI screenshots with
SQuAD-normalized exact match and token F1. Unlike MolmoWeb-SyntheticQA, its screenshots and answers
are human annotated and mobile-oriented.

| Reference | Setting | Exact match | Token F1 |
|---|---|---:|---:|
| Gemini 1.5 Pro | zero-shot | 81.4% | 87.2% |
| ScreenAI 5B | fine-tuned | 90.7% | 94.6% |
| Gemini 1.5 Flash | fine-tuned | 90.5% | 94.9% |
| PaliGemma 3B, 896 px | fine-tuned | 89.4% | 93.2% |

Source: https://aclanthology.org/2025.naacl-long.477/ (Tables 5 and 6).

### WebUIBench OCR in Webpage

WebUIBench is browser-specific and includes OCR of red-boxed regions in web screenshots. Its OCR
score is character-level text similarity, not exact-match QA, so it is supplementary context only.
The paper reports 83.4 for Qwen2-VL-72B, 79.1 for GPT-4o, and 49.6 for Qwen2-VL-2B.

Source: https://aclanthology.org/2025.findings-acl.815/ (Table 3).

## Recommended publication framing

Report three tiers separately:

1. EXP002 domain-held-out MolmoWeb probe: baseline versus every Qwen checkpoint.
2. Released MolmoWeb-4B/8B controls on exactly the same manifests and evaluator.
3. Standard external benchmarks: ScreenSpot-v2 for grounding and ScreenQA Short (or a
   browser-specific QA benchmark with a stable public split) for screenshot understanding.

This avoids claiming a state-of-the-art gap from incompatible splits or scoring rules.
