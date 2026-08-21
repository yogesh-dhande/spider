# Spider browser-model diagnostics

Small static explorer for training-data composition, ScreenshotQA, GUI grounding, and
browser-action diagnostics. The default EXP005 data-audit view includes the website/category
breakdown for the training candidate pool, a searchable website inventory, and a deterministic
cross-task sample. Action and grounding cards include selectively materialized screenshots and
ground-truth click overlays; QA prompt/answer cards remain browsable without downloading the
multi-gigabyte Arrow shards just for the dashboard.

During
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

Regenerate the EXP005 data-audit payload from the local inventory. If the final nested ladder
exists, the builder automatically uses its largest frozen tier; otherwise it clearly labels the
view as training candidates. Existing cached previews are reused:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_exp005_dataset_dashboard.py \
  --inventory-dir data/exp005_browser_inventory \
  --selection-dir data/exp005_browser_selection_v1 \
  --output dataset-dashboard/app/exp005-data.json \
  --public-dir dataset-dashboard/public
```

Add `--materialize-images` to fetch action and grounding previews. ScreenshotQA images are
deliberately skipped because they require downloading full Arrow shards.

Run the explorer locally:

```bash
cd dataset-dashboard
npm install
npm run dev -- --port 4173
```

The screenshots come from AllenAI's MolmoWeb synthetic datasets under ODC-BY 1.0. The Kaggle
orchestrator regenerates the payload after every completed development probe and once more from
the matched sealed predictions at the end of EXP004.
