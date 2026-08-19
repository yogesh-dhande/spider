# Spider browser-model diagnostics

Small static explorer for ScreenshotQA, GUI grounding, and browser-action diagnostics. During
EXP004 training it compares the EXP002 parent checkpoint with the latest completed stage on the
fixed development probes: 128 ScreenshotQA examples, 128 GUI-grounding examples, and 256 browser
actions. After checkpoint selection and the sealed evaluation, the metric cards cover the complete
sealed sets while each task view retains a deterministic 64-example diagnostic sample.

Grounding and browser-action views overlay the annotated element box and target center with both
checkpoint click predictions. Failure-first ordering makes OCR, semantic, output-format, and
spatial errors easier to inspect. The browser-action view appears automatically after EXP004's
first completed validation.

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

The screenshots come from AllenAI's MolmoWeb synthetic datasets under ODC-BY 1.0. The Kaggle
orchestrator regenerates the payload after every completed development probe and once more from
the matched sealed predictions at the end of EXP004.
