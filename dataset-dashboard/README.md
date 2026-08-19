# MolmoWeb QA Lens

Small static explorer for the fixed EXP002 validation probe and EXP004 browser-action
development probe. It contains 128 held-out
ScreenshotQA examples and 128 GUI-grounding examples over 31 resized browser screenshots. It
always compares the untouched Qwen3.5-2B baseline with the latest completed validation probe.
Grounding and browser-action views overlay the annotated element box and center with both
checkpoint click predictions. The action view appears automatically after EXP004's first completed
stage validation and emphasizes misses for diagnosis.

The checked-in payload is generated from immutable Kaggle prediction artifacts by reusable
Python code:

```bash
python scripts/build_dataset_dashboard_data.py \
  --baseline <baseline-predictions.jsonl> \
  --latest <latest-predictions.jsonl> \
  --latest-step <optimizer-step> \
  --output dataset-dashboard/app/qa-probe.json
```

Run the explorer locally:

```bash
cd dataset-dashboard
npm install
npm run dev -- --port 4173
```

The screenshots come from `allenai/MolmoWeb-SyntheticQA` and
`allenai/MolmoWeb-SyntheticGround`, licensed ODC-BY 1.0. The dashboard payload is regenerated
by the Kaggle stage monitor after every completed probe.
