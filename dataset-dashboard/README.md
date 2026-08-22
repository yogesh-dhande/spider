# Spider browser-model diagnostics

Small static explorer for training-data composition, ScreenshotQA, GUI grounding, and
browser-action diagnostics. The default EXP005 data-audit view includes the website/category
breakdown for the training candidate pool, a searchable website inventory, and a deterministic
cross-task sample. Action and grounding cards include selectively materialized screenshots and
ground-truth click overlays; QA prompt/answer cards remain browsable without downloading the
multi-gigabyte Arrow shards just for the dashboard.

For EXP005 it compares untouched Qwen3.5-2B with the latest validated scaling checkpoint on the
frozen IID development suite. Metric cards use the complete merged suite; each task view retains a
deterministic diagnostic sample with both model outputs and click overlays. The candidate registry
and regression gate update before the hosted payload, so the dashboard cannot display an
unreceipted checkpoint.

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

For EXP005, first build the verified payload without copying images, extract only its selected
images from the shared GCS corpus archive, then rerun with image copying enabled:

```bash
PYTHONPATH=src .venv/bin/python scripts/refresh_exp005_dashboard.py \
  --baseline-root outputs/experiment5/baseline/base-all-0821a \
  --latest-root <archived-evaluation-root> \
  --baseline-receipt experiments/exp005_browser_ablation_bed/artifacts/baseline_base_all_0821a.json \
  --latest-receipt <validated-evaluation-receipt> \
  --latest-name '<size/seed/step label>' --latest-step <step> \
  --corpus-root outputs/experiment5/dashboard_corpus_subset --skip-images

PYTHONPATH=src .venv/bin/python -m spider.dashboard_images \
  --payload dataset-dashboard/app/qa-probe.json \
  --archive outputs/experiment5/dashboard_corpus.tar.zst \
  --destination outputs/experiment5/dashboard_corpus_subset
```

During a scaling campaign, a quiet watcher can keep the checked-in payload on the highest-scale,
latest-step registered checkpoint. Standard scaling receipt paths are resolved automatically;
the first recovered seed-53 checkpoint needs the explicit historical source mapping shown here:

```bash
PYTHONPATH=src .venv/bin/python scripts/watch_exp005_dashboard.py \
  --controller-state-uri \
  gs://keptune-spider-experiments-1088401257609/exp005/controller/v1/latest.tar.gz \
  --controller-mirror outputs/experiment5/controller_mirror \
  --candidate-repo-root outputs/experiment5/controller_mirror \
  --manifest \
  outputs/experiment5/controller_mirror/experiments/exp005_browser_ablation_bed/control_comparison_manifest_v1.json \
  --output-root outputs/experiment5/controller_mirror/outputs/experiment5/scaling \
  --source-override \
  outputs/experiment5/controller_mirror/experiments/exp005_browser_ablation_bed/artifacts/sft_small53_step0500_r2_0822a.json=outputs/experiment5/sft/small53-step0500-r2-0822a
```

The watcher mirrors only the registry, receipts, and evaluation outputs from the keyless cloud
controller. It updates only after candidate registration, re-verifies dashboard metrics against
both receipts, materializes only selected screenshots, and requires the production build/render
test to pass before recording the dashboard checkpoint as current.

Run the dashboard on any available local port, for example:

```bash
npm --prefix dataset-dashboard run dev -- --port 4174
```
