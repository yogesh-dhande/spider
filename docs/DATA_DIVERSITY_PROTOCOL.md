# Website-diversity protocol

EXP002 and EXP004 demonstrate that domain-disjoint splitting alone is insufficient. EXP002's
training manifests cover hundreds of domains, but its fixed development prefixes are concentrated;
the EXP004 grounding stage probe can be entirely Wikipedia. EXP004 action preparation also reads
one shard per source and stops when its quota fills, so it is a causal pilot rather than evidence of
broad browser generalization.

## EXP004 interpretation

Do not change EXP004's primary development set or checkpoint-selection rule after observing its
metrics. After checkpoint selection, run a separately labeled post-hoc diagnostic drawn from unused
shards and domains. It must not influence the selected checkpoint or the confirmatory claim.

## Required design for the next data build

1. Inventory metadata across every eligible pinned shard before downloading screenshots.
2. Split at the domain level, and keep complete trajectories/screenshots within one split.
3. Select sampling units rather than individual steps or questions. Default caps are one example
   per unit for evaluation and a preregistered small cap for training.
4. Use deterministic temperature sampling over domains (`n_domain ** 0.5` by default), with an
   explicit maximum website share. Never use a dataset prefix as an evaluation sample.
5. Balance and report source configuration, action type, QA type, trajectory count, screenshot
   count, and website count.
6. Maintain two evaluations: a distribution-weighted IID set and a domain-balanced unseen-domain
   set. Report micro accuracy on the former and both micro and macro-over-domain accuracy on the
   latter.
7. Bootstrap uncertainty by domain or screenshot/trajectory—not by individual example—so repeated
   questions or steps are not treated as independent.
8. Keep external benchmark test sets sealed and out of all training and checkpoint selection.

`spider-audit-diversity` records concentration, effective-domain count, and the complete domain,
task, and source distributions. With `--sample-output`, it produces a deterministic group-aware
sample and records the before/after audit.
