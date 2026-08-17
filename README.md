# Spider: Experiment 1

This repository tests whether `Qwen/Qwen3-VL-2B-Instruct` can learn browser screenshot
understanding and precise GUI grounding from a small, balanced slice of MolmoWeb data.
It is deliberately limited to visual perception. Browser actions, trajectories, custom action
tokens, planning, and reinforcement learning are later increments.

## Experiment design

- 30,000 training examples: 15,000 ScreenshotQA and 15,000 GPT-authored grounding examples.
- Per task: 1,000 validation and 2,000 test examples.
- Domains are hash-assigned to exactly one split across both tasks. The preparation command
  fails if it detects domain overlap.
- Screenshots are downscaled to fit within 1280 x 720 without upscaling, padding, or changing
  aspect ratio.
- Grounding uses the center of the annotated element bounds as its target. Responses retain
  Qwen3-VL's existing JSON `point_2d` format on a normalized 0-1000 grid.
- A single multitask QLoRA adapter is trained with equal numbers of QA and grounding examples.
- The base model and adapter are evaluated with identical greedy decoding.
- ScreenSpot is evaluation-only and never participates in training or model selection.

The source datasets are
[MolmoWeb-SyntheticQA](https://huggingface.co/datasets/allenai/MolmoWeb-SyntheticQA),
[MolmoWeb-SyntheticGround](https://huggingface.co/datasets/allenai/MolmoWeb-SyntheticGround),
and [ScreenSpot](https://huggingface.co/datasets/bevaya/ScreenSpot). Review their licenses and
responsible-use terms before redistributing prepared artifacts.

## Kaggle workflow

The primary interface is [the Kaggle notebook](notebooks/experiment1_kaggle.ipynb). Enable
Internet access and a GPU accelerator. The default configuration targets one 16 GB T4/P100;
QLoRA uses NF4 weights, FP16 computation on T4/P100, batch size 1, and gradient accumulation.

From a terminal or notebook cell:

```bash
pip install -e ".[train]"
spider-prepare --config configs/experiment1.yaml

# Measure the untouched base model first. This resumes at the prediction level.
spider-evaluate --config configs/experiment1.yaml --label baseline

# Train in a bounded Kaggle session. Re-running adds another chunk and resumes checkpoints.
spider-train --config configs/experiment1.yaml --additional-steps 500

# After the configured one-epoch target has completed:
spider-evaluate \
  --config configs/experiment1.yaml \
  --label sft \
  --adapter outputs/experiment1/adapter/final

spider-compare \
  --baseline outputs/experiment1/evaluation/baseline/metrics.json \
  --sft outputs/experiment1/evaluation/sft/metrics.json \
  --output outputs/experiment1/comparison.md

# Create a small immutable archive suitable for Git/publication provenance.
spider-archive --config configs/experiment1.yaml
```

Save each completed Kaggle notebook version with `data/experiment1` and
`outputs/experiment1` as outputs. In the next session, attach that version's output and copy
both directories back into the repository before resuming. The notebook contains a cell for
this. A 500-step chunk is intentionally conservative; adjust it once measured throughput is
known.

For two T4s, launch training through Accelerate:

```bash
accelerate launch --num_processes 2 -m spider.train \
  --config configs/experiment1.yaml --additional-steps 500
```

The effective batch size scales with GPU count, and the chunk logic caps training at the
configured one-epoch target.

## Outputs

Preparation creates compact JSONL manifests and deduplicated resized JPEGs:

```text
data/experiment1/
  dataset_summary.json
  experiment_config.json
  images/{qa,grounding,screenspot}/
  manifests/{qa,grounding,combined}_{train,validation,test}.jsonl
  manifests/screenspot_test.jsonl
```

Each evaluation label produces:

```text
outputs/experiment1/evaluation/<label>/
  predictions.raw.jsonl   # append-only resume state
  predictions.jsonl       # predictions plus per-example scores/error category
  metrics.json
  report/failures.html
  report/assets/*.jpg
```

The primary metrics are normalized exact answer accuracy for ScreenshotQA and click accuracy
(predicted point lies inside the annotated element) for grounding. Token F1, parse rate,
on-grid rate, mean/median pixel distance, and 25/50/100-pixel accuracy provide diagnostics.
MolmoWeb and ScreenSpot metrics remain separate.

Failure labels are diagnostic heuristics, not ground-truth error annotations:

- wrong OCR-tagged ScreenshotQA examples -> `ocr`
- other wrong ScreenshotQA examples -> `semantic_understanding`
- unparseable grounding responses -> `output_format`
- parsed clicks outside the element -> `spatial_grounding`

The HTML gallery includes representative successes and failures. Grounding images display the
target bounds/center in green and the predicted point as a red cross.

The tracked [`experiments/registry.yaml`](experiments/registry.yaml) indexes experiments. Each
official run is archived under its experiment record with config/result checksums, dependency
versions, model and dataset revisions, and the exact source Git state. Commit each archive;
retain large raw predictions and galleries in the corresponding Kaggle output version.

## Development checks

Lightweight tests do not download models or datasets:

```bash
pip install -e ".[dev]"
pytest -q
```

Before paying for a full run, use `--limit 8` on evaluation and `--max-steps 2` on training to
verify the current Kaggle image and dependency versions.
