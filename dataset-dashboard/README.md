# MolmoWeb QA Lens

Small static explorer for the fixed EXP002 validation probe. It contains 128 held-out
ScreenshotQA examples and 128 GUI-grounding examples over 31 resized browser screenshots. QA
compares the untouched Qwen3.5-2B baseline, step 250, raw step 1,000, and the step-1,000 first
answer recovered before the leaked next chat turn. Grounding overlays the annotated element box
and center with each checkpoint's predicted click and pixel error.

The checked-in payload is generated from immutable Kaggle prediction artifacts by reusable
Python code:

```bash
python scripts/build_dataset_dashboard_data.py \
  --baseline <baseline-predictions.raw.jsonl> \
  --step-250 <step-250-predictions.raw.jsonl> \
  --step-1000 <step-1000-predictions.raw.jsonl> \
  --output dataset-dashboard/app/qa-probe.json
```

Run the explorer locally:

```bash
cd dataset-dashboard
npm install
npm run dev
```

The screenshots come from `allenai/MolmoWeb-SyntheticQA` and
`allenai/MolmoWeb-SyntheticGround`, licensed ODC-BY 1.0. The dashboard is a diagnostic
development view; recovered metrics do not replace a corrected model-generation probe.
