+# Spider EXP002 prepared browser-perception data

Private experiment artifact for EXP002. It contains the frozen, domain-disjoint data used to
measure and fine-tune Qwen3.5-2B for browser ScreenshotQA and GUI grounding.

## Contents

- MolmoWeb: 15,000 train, 1,000 validation, and 2,000 test examples per task.
- Tasks remain separate: ScreenshotQA text answers and normalized 0–1000 grounding points.
- ScreenSpot: 1,272 evaluation-only grounding examples.
- 5,231 unique JPEG screenshots, fit within 1280×720 without upscaling or padding.
- Combined train/validation/test manifests, exact experiment config, summary, and SHA-256
  checksums for every artifact file.

## Provenance

- MolmoWeb-SyntheticQA revision: `8b6199086b04057106c576504c84e44333b49fea`
- MolmoWeb-SyntheticGround revision: `3cce6f5c446e43a050b16658638fcbf49c815e9a`
- ScreenSpot revision: `0be08781e2e188582f6131625ae1598d443b4d5d`
- Preparation seed: `17`
- Preparation source commit: `ba42ce6`
- Final Kaggle source: `yogeshkd/spider-exp002-finalize-prepared-data/3`

The artifact is private. Do not make it public until upstream redistribution terms have been
reviewed. Kaggle metadata uses the generic `other` license category rather than asserting rights
not established by this derived artifact.

