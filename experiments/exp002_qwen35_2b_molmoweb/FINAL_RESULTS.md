# EXP002 final results

The selected step-1,875 QLoRA adapter was evaluated once on the frozen test manifests after all
checkpoint decisions were complete. The authoritative merge contains exactly 5,272 predictions:
2,000 MolmoWeb ScreenshotQA, 2,000 MolmoWeb grounding, and 1,272 untouched ScreenSpot examples.
MolmoWeb is the in-scope browser result; ScreenSpot is retained only as an out-of-domain grounding
diagnostic.

## Baseline versus SFT

| Metric | Untouched baseline | Step 1,875 SFT | Absolute delta |
|---|---:|---:|---:|
| MolmoWeb QA exact accuracy | 0.2240 | **0.2620** | **+0.0380** |
| MolmoWeb QA mean token F1 | 0.4964 | **0.6103** | **+0.1139** |
| MolmoWeb grounding click accuracy | 0.5665 | **0.7340** | **+0.1675** |
| MolmoWeb grounding parse rate | 0.9995 | **1.0000** | **+0.0005** |
| MolmoWeb grounding median distance | 31.15 px | **8.52 px** | **−22.63 px** |
| ScreenSpot click accuracy | 0.4717 | **0.5896** | **+0.1179** |
| ScreenSpot median distance | 50.99 px | **26.35 px** | **−24.64 px** |

The experiment answers its initial question positively: a 2B Qwen3.5-VL model acquired materially
stronger browser screenshot understanding and precise GUI grounding from 30,000 MolmoWeb-style
examples. The largest gain is browser grounding, while QA improved meaningfully but remains the
clearer bottleneck for later browser-agent work.

## ScreenshotQA by question type

| Type | Examples | Baseline exact | SFT exact | Baseline F1 | SFT F1 |
|---|---:|---:|---:|---:|---:|
| OCR | 1,044 | 0.4262 | **0.4770** | 0.6070 | **0.6792** |
| Affordance | 573 | 0.0052 | **0.0384** | 0.3998 | **0.5830** |
| Summarization | 383 | 0.0000 | **0.0104** | 0.3393 | **0.4633** |

Exact match is intentionally strict for free-form affordance and summarization answers, so token F1
is the more informative metric for those categories. OCR remains much stronger than the two
semantic categories.

## Final SFT shard diagnostics

Shards are deterministic partitions, not independent experimental replicates. The merged values
above are authoritative; the table below exposes partition variability.

| Shard | QA exact | QA token F1 | MolmoWeb click | ScreenSpot click |
|---:|---:|---:|---:|---:|
| 0 | 0.280 | 0.6272 | 0.736 | 0.6101 |
| 1 | 0.228 | 0.5838 | 0.764 | 0.6101 |
| 2 | 0.256 | 0.6093 | 0.716 | 0.5283 |
| 3 | 0.260 | 0.6088 | 0.712 | 0.5849 |
| 4 | 0.280 | 0.6247 | 0.752 | 0.6667 |
| 5 | 0.264 | 0.6168 | 0.712 | 0.5849 |
| 6 | 0.304 | 0.6284 | 0.756 | 0.5975 |
| 7 | 0.224 | 0.5833 | 0.724 | 0.5346 |
| Mean | 0.262 | 0.6103 | 0.734 | 0.5896 |
| Population standard deviation | 0.0251 | — | 0.0197 | 0.0413 |

## Execution note

All eight GPU inference shards completed successfully. CPU merge version 1 rejected the shards
because Kaggle mounted the same step-1,875 adapter under two equivalent path prefixes. No model
inference was repeated. Merge version 2 canonicalized the mount prefix while retaining the source
kernel and adapter-relative path as integrity invariants, then validated exact, non-overlapping
coverage of all 5,272 frozen IDs.
