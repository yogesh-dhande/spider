# EXP005 keyless cloud controller

The scaling campaign is supervised by a CPU-only GCE instance rather than a workstation process
authenticated as a human user. This prevents Google Cloud session-control expiry, terminal-pipe
backpressure, and workstation sleep from interrupting orchestration.

## Identities

- `spider-exp005-controller@keptune.iam.gserviceaccount.com` is attached only to the controller VM.
  It can manage Compute Engine instances, use Google Cloud services, and read/write the EXP005
  artifact bucket.
- `spider-exp005-worker@keptune.iam.gserviceaccount.com` is attached to every newly created EXP005
  worker. It can read/write the experiment bucket but cannot create or manage VMs.
- The controller can attach only the worker identity. No user-managed service-account key exists.

The controller is intentionally excluded from the `spider-managed=true` worker selector, so a
worker cleanup sweep cannot stop its own supervisor.

## Durability and shutdown

`scripts/run_exp005_cloud_controller.py` redirects every child process to a named log, publishes a
sparse status document, and snapshots scientific receipts plus controller state every five minutes
to `gs://keptune-spider-experiments-1088401257609/exp005/controller/v1/`. A restarted controller
restores that snapshot before adopting the campaign. Only the two explicitly registered step-500
recovery validations are retried; ordinary scaling jobs remain fail-closed.

The runtime exits when every registered process is terminal, and the startup wrapper then shuts
down the controller. Compute Engine independently stops it after 14 days if the runtime never
reaches a terminal state. GPU guests retain their own shorter maximum-run and shutdown guards.

## Reproduction

The exact campaign topology, revisions, recovery namespaces, and state paths are frozen in
`configs/ablations/experiment5_cloud_controller_v1.json`. Provisioning is idempotent except for VM
creation, which refuses to replace an existing controller:

```bash
python3 scripts/provision_exp005_cloud_controller.py iam
python3 scripts/provision_exp005_cloud_controller.py snapshot \
  --config configs/ablations/experiment5_cloud_controller_v1.json
python3 scripts/provision_exp005_cloud_controller.py create \
  --revision COMMIT_SHA \
  --config configs/ablations/experiment5_cloud_controller_v1.json
```

Read the latest sparse status without opening an SSH session:

```bash
gcloud storage cat \
  gs://keptune-spider-experiments-1088401257609/exp005/controller/v1/status.json
```
