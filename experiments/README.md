# Experiment registry

Every experiment receives a stable ID, a frozen YAML configuration, a design record, and one
or more immutable result archives. `registry.yaml` is the cross-experiment index.

After an official baseline/SFT run, archive the small publication artifacts with:

```bash
spider-archive --config configs/experiment1.yaml
```

The archive records model/dataset revisions, exact metrics, package versions, config and result
checksums, source Git commit, dirty-worktree state, and the baseline-to-SFT table. Raw
predictions and image galleries remain under `outputs/` because they are too large for Git;
preserve those as Kaggle notebook-version outputs or another artifact store.

Result archives are immutable: reusing an existing run ID is rejected. Commit each new archive
and update the experiment status/interpretation in its design record.

