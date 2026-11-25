# Recovery-v2 replacement detector protocol

2026-11-21. This is the explicitly authorized replacement of the interrupted v1
attempt. It changes validation completeness and execution durability, with both
replacement detectors trained afresh. Original v1 protocol/receipt/source/overlay,
completed historical A, incomplete C and unknown C exit status remain preserved.
No final-test use, reference training, application, publication or submission.

## Why a replacement pair is necessary

Pinned Ultralytics non_max_suppression can break its image loop on a wall-clock
budget, returning untouched empty outputs for later images. Finite outputs do not
prove processing completeness. An affected unselected epoch can still undermine
best-checkpoint selection: complete evaluation might have changed its ranking.
The synthetic three-positive-image delayed-clock regression reproduces counts
[1,0,0] with the default limit versus[1,1,1] with the limit disabled.
Historical A therefore remains historical; its complete new45-source preflight
only tests the recovery adapter and does not certify its old50-epoch selection.

## Narrow NMS amendment

`complete_validation.complete_nms` calls the original installed pinned NMS function
with only max_time_img changed to float('inf'). It does not edit packages, silence
warnings, alter box math or permanently monkeypatch global functions. Confidence,
NMS IoU, class filtering, multi-label behavior, max_det/max_nms/max_wh, precision
and matching metrics remain those of the pinned callers. Validator multi-label
isTrue; predictor retains its original defaultFalse. Ordinary-case equivalence,
class filtering, limits, class-agnostic behavior and returned indices are tested.
The protocol JSON records an unlimited-wall-clock string, not an Infinity literal.

CompleteValidator applies this policy during every training clean-validation pass
and final best-checkpoint validation. Expected membership is anchored to the
physical authenticated45-image validation view, not inferred from a potentially
short loader. It records source paths/batch IDs, NMS output counts and metric
coverage; missing/duplicate processing fails before metrics/fitness can return.
The training checkpoint writer additionally requires that epoch's complete pass.
Empty detections are legitimate if the source was processed. CompletePredictor
uses the same policy for validation galleries. The future shared evaluator must
use this declared policy; no held-out conditions/predictions have been generated.

## Unchanged scientific training contract

Parent config `configs/main_ac_frozen_v1.json` remains unchanged. New config:
`configs/main_ac_recovery_v2.json`. The training dictionary, data/initial hashes,
source order, exposure manifest and degradation implementation are identical.
50epochs,416px,batch4,MPS,workers0,cachefalse,AMPfalse,seed42,strict determinism;
AdamW and all nbs/warmup/LR/weight-decay/EMA conventions remain pinned. No generic
augmentation, early stopping, resume, extra epochs or automatic retries. A is
clean; C alone uses the frozen clean1/2 and each of six degradations1/12 schedule.
Each model has2650microbatches,10500presentations,185updates,10458optimizer-consumed
presentations and42unstepped tail images. Unstepped gradients are not flushed.
Both six-class initialized tensor states must match before updates. Latest exact
clean-validation mAP50-95 fitness tie wins, with the same EMA convention.

## Durable observed evidence and consistent epoch commits

A SHA256-chained append-only JSONL journal fsyncs actual sample deliveries, batch
bindings, optimizer/EMA steps with consumed presentation IDs, epoch boundaries,
validation batch/source completeness and checkpoint writes as observed. Journal
reading validates sequence/hash chaining and reports a truncated final record,
retaining its raw tail; missing events are never reconstructed from expectations.
The standard tests cover truncation and reordered/corrupt evidence.

Root status files use temporary writes, fsync and atomic replacement. Each epoch
has a separately committed immutable provenance/accounting/exposure snapshot,
checkpoint hashes/content-addressed checkpoint copies, a manifest and journal
boundary. A final commit follows full training/checkpoint validation. Staged
checkpoint writes/final stripping protect the previous current checkpoint on
failure; the checkpoint object store uses separate copies, not mutable hardlinks.
Commit objects are audit evidence, not authorization or a guarantee to resume:
pending live gradients are not serialized as a supported continuation protocol.
Interrupted work remains incomplete; any replacement must be separately documented.
The root provenance may gain final timing/memory fields after the final scientific
snapshot; the commit identifies its exact snapshot boundary.

## Process supervision and resources

`supervise_recovery.py` is a foreground finite-job supervisor in the inherited
process group, with no detach, automation or automatic restart. It records child/
supervisor PIDs, command/environment, separate stdout/stderr, resource heartbeat
and real returncode only after child wait and native cleanup. Explicit stop signals
are forwarded to that child. An absent exit record remains unknown. C waits for
A's verified training and successful real exit, and starts original weights.

Child sampling records RSS/system available/global swap and MPS tensor/driver
allocations every second; supervisor records RSS/swap/disk through cleanup every
2seconds. Samples are observed maxima, not guaranteed peaks; unified counters are
not additive, global swap cannot be solely attributed to training, and native
process footprint is not resident RAM. Both monotonic elapsed measurements and
wall-clock process timestamps are retained, with discontinuities disclosed.
Before optimization require MPS/strict controls,512MiB available memory and6GB disk
headroom. No unrelated applications are terminated. A watchdog may fail a stalled
job, but this implementation never returns partial validation as successful metrics.

## Reproduction and verification

The versioned receipt/overlay under `docs/main_ac_recovery_v2` pins parent artifacts,
new source snapshots and additional installed NMS/validator/predictor source hashes.
The pre-training gate records20focused tests,137regressions and the all45-source
validation-only preflight. Pre-training candidate snapshots are retained separately;
only the tested final overlay is used by the replacement runs. There is no invented
Git commit ID. Exact sequential supervisor commands are recorded in the receipt.

The read-only recovery verifier requires real exit0, all50epochs/51complete full
validation passes, full exposure/order/condition/optimizer/EMA/tail evidence, journal
and every epoch commit, hashes and loadable checkpoints. It checks final selected
weights against the committed selected EMA tensor state and requires readable
complete validation galleries. The original incomplete C must still fail its
original verifier. Final acceptance and W2/G05 reference clearance remain separate.
