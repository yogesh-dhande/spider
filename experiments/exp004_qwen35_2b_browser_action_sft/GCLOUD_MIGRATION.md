# EXP004 GCloud migration

The Kaggle run remains the source of checkpoints through step 375. Later stages may run in the
`keptune` Google Cloud project using the same pinned repository revision, prepared-data checksums,
optimizer state, scheduler state, and effective batch size.

## Billing safety

Only instances with both labels below are controlled by `scripts/gcloud_exp004.py`:

- `spider-managed=true`
- `spider-experiment=exp004`

Every created VM must also specify `--max-run-duration` and
`--instance-termination-action=STOP`. Guest jobs install an EXIT trap that powers off the VM, and
the external monitor stops every still-active VM in a `finally` block. Existing Keptune VMs without
the two labels are never modified.

The durable artifact root is:

`gs://keptune-spider-experiments-1088401257609/exp004/`

VM creation and stop events are appended to `artifacts/gcloud/vm_registry.jsonl` for the experiment
record.

## Compatibility gate

Kaggle used two T4 GPUs with gradient accumulation 8. The initial GCloud candidate is one L4 with
gradient accumulation 16, preserving an effective batch size of 16 and the registered 1,875-step
schedule. Before a GCloud stage is accepted, a disposable two-step resume must verify:

1. checkpoint, optimizer, scheduler, and trainer state restore successfully;
2. effective batch size remains 16 and the planned schedule remains 1,875 steps;
3. loss is finite and a saved adapter reloads for inference; and
4. the disposable checkpoint is not used as a scientific result.

Training stages remain sequential. Action, QA, and grounding evaluation shards may use separate
VMs in parallel, followed by the existing deterministic merge and regression gate.

Stage validation runs as two independent VMs: action inference on one L4 and the combined QA plus
grounding probe on one T4. Each uploads metrics and resumable raw predictions before stopping. The
frozen gate is implemented once in `spider.exp4_gate` so Kaggle and GCloud use identical logic.

The guest uses `torch==2.10.0+cu128`, matching the observed Kaggle runtime, together with the exact
versions in `requirements/experiment2-kaggle.txt`. A stage checkpoint is uploaded only after the
compatibility assertions and registered stage assertions pass.

Compute dtype is explicitly FP16. T4 selected FP16 automatically, whereas L4 would otherwise select
BF16 and could not restore the registered FP16 gradient-scaler state. The first disposable cloud
attempt caught this before taking a training step and shut itself down; amendment 002 records it.

The VM image is pinned to
`common-cu129-ubuntu-2404-nvidia-580-v20260819` rather than a moving image family.
