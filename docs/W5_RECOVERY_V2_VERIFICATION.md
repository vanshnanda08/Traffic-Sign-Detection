# W5 recovery-v2 executed verification

2026-11-21. **Replacement W5 passed.** Both fresh A/C models completed the frozen
training contract, every validation pass was complete, both processes exited0
through native cleanup, and the read-only recovery pair verifier exited0.
No final-test degradation, model inference or metric evaluation, reference
training, application, publication or submission were performed. Frozen-file
integrity hashing is separate from model development/evaluation. W2/G05 remains blocked separately.

## Protocol lineage and source preservation

The original v1 attempt is retained exactly: A's completed historical training;
C's42 complete persisted epochs plus unflushed raw epoch43 activity; unknown C
exit/termination mechanism. No OOM is inferred, no old provenance was edited and
no partial checkpoint was deleted/resumed. Its original verifier was actually
rerun and still exits1 (`Run did not pass every main training epoch`). The new
pair does not silently reuse historical A after changing validation selection.

- Parent protocol SHA: `802144f7501490b35cf371395c88a1ce1d0206fcc2a9b64aef33fbb54b305626`.
- Replacement protocol SHA: `ffa15a791c5591d19fea1e459cb92b890be42e608d709fca4ef6fbac93e1e8b8`.
- Replacement execution overlay SHA: `63ff0c3d084426ea69a2a0f810a8378138345914bfd53e82d00b8f64b5ad3b5f`.
- Immutable replacement receipt SHA: `27d4787701031af2c292f603c4622b3ca809ed8c4845960c51e1bd18cdee9226`.
- Exposure manifest SHA: `ea7cb07bd1284d0a3c9a7fd187f182f2a88254f7fef5908fb4a914c470acc565`.
- Dataset lock SHA: `ec2b3d0acef8bc03717a606313aae6b6b10ad91d37ebb6e6eecdda72c1fded51`.
- Original initialization SHA: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- Canonical six-class initial tensor state, both processes before updates:
  `fa0a46dc06278fc877f99d04d05a21ccdd6e075434cf5816f6b5856bd01daa76`.

All **16,307 protected historical files** retain SHA256 and mtimes, including
both Indian dataset revisions, old sources/pins/receipts/overlays, original weights,
old A/C checkpoints, feasibility artifacts and historical evidence. No old source,
requirements or run_all.sh bytes were edited. The new overlay snapshots
12 new source/configuration files and records parent library/NMS
hashes. No Git commit ID is invented. [Preservation](main_ac_recovery_v2/preservation_after.json).

The receipt's candidate label reflects creation before tests; the immutable bytes
were not rewritten after training. [Pre-training gate](main_ac_recovery_v2/pretraining_gate.json)
records actual promotion after tests and45-source preflight; the separate
[verification receipt](main_ac_recovery_v2/verification_receipt.json) records the
completed pair without retroactively repinning it. Pre-training source candidates
were retained; neither model trained under an untested earlier candidate.

## Changes and implementation

New config `configs/main_ac_recovery_v2.json`; protocol/receipt/overlay and source
snapshots under `docs/main_ac_recovery_v2` and ignored
`results/main_ac_recovery_v2/execution_snapshots`. New modules:
`complete_validation.py`, `durability.py`, `recovery_protocol.py`,
`recovery_training.py`, `recovery_verification.py`. New scripts:
`create_recovery_contract.py`, `train_recovery.py`, `supervise_recovery.py`,
`recovery_preflight.py`, `check_recovery.py`. Added `tests/test_recovery.py` and
[protocol rationale](RECOVERY_V2_PROTOCOL.md); current status/roadmap/checklist updated.

The narrow NMS wrapper calls the pinned installed computation with only
max_time_img=float('inf'). No package edit, warning suppression or permanent global
monkeypatch. All thresholds, class/multi-label/max_det/max_nms/box/precision/AP
computation conventions remain pinned. Standard JSON uses a string time policy.
The slow-clock positive fixture returns[1,0,0] under default timeout and[1,1,1]
under complete policy. Ten ordinary-case equivalence variants also pass.

Expected validation source membership is authenticated against the frozen45-image
view. Source/batch IDs, NMS and metric coverage are checked before stats/fitness
return; incomplete or duplicate processing raises. Empty detections remain valid.
The same complete policy runs during all50epoch validations, final best validation
and six-image galleries for each model. The future evaluator must inherit this
NMS policy and add authorized frozen test-source coverage; current training and
preflight paths accept only development inputs. No final-test evaluator was run.

Durable, SHA-chained append-only journals fsync actual deliveries, batch bindings,
optimizer/EMA/consumed-source events, epochs, validation coverage and checkpoint
writes. Atomic epoch commits retain consistent provenance/accounting/exposure and
content-addressed checkpoint objects with journal boundaries. Truncated final
records are reported/preserved rather than completed from expected data. Checkpoint
staging/atomic replacement and failure preservation are tested. These audit commits
are not a supported resume protocol; live pending gradients must not be silently
restored or flushed. A foreground supervisor records exact PID/commands, separate
stdout/stderr/resources/heartbeat and actual exit after native cleanup, without
detachment, automatic restart or monitoring automation. Explicit stop is forwarded.

## Executed tests and validation-only preflight

**20 focused recovery tests passed** (1.24s), **137 full smoke/regressions passed**
(89.40s), no skips, exit0, before replacement optimization. Tests cover timeout
reproduction/equivalence/known positives, legitimate empties, missing/duplicate
coverage and forbidden metrics, atomic failure preservation, journal integrity/
truncation, checkpoint copy consistency, frozen initialization/config guards,
existing directories and continued original-C rejection. Existing source-order,
adapter, optimizer-budget and lock regressions are included. No extra plumbing
training experiment was run. [Exact commands/exits](main_ac_recovery_v2/executions.json),
portable XML and local logs are retained.

Before replacement training, retained historical A was used only for a separately
saved all45 clean-validation preflight. It completed six batches/45 sources with
45 metric inputs, exit0. It neither tuned hyperparameters nor retrospectively
certified old epoch45 selection. [Preflight](main_ac_recovery_v2/historical_A_all45_preflight.json).
Resource/device/source/lock gates passed before launching A, then C. No unrelated
process was terminated. Supervisor commands and actual child argument vectors are
recorded in the frozen receipt and per-run process records.

## Actual replacement training and verification

Both runs: `results/main_ac_recovery_v2/A` and `results/main_ac_recovery_v2/C`.
Original pretrained bytes and equal canonical six-class initialization verified.
No resume, automatic retry, early stopping, replay, extra epochs or device/batch
change. The original optimizer/LR/EMA/degradation settings and source order remain
fixed. Each model's actual contract passed:

| Item | A | C |
| --- | ---: | ---: |
| Epochs | 50 | 50 |
| Microbatches | 2650 | 2650 |
| Presentations | 10500 | 10500 |
| Optimizer updates / EMA updates | 185 /185 | 185 /185 |
| Optimizer-consumed presentations | 10458 | 10458 |
| Unstepped tail images | 42 | 42 |
| Complete full validation passes | 51 | 51 |
| Atomic commits (50epochs +final) | 51 | 51 |
| Actual child process exit | 0 | 0 |

Complete step dictionaries, microbatch sizes and per-epoch pending tails match the
frozen trace, not just totals. A/C source order/membership matches for all10500
presentations; A exactly reproduces historical A's source order. Every original
appears once per epoch. A is entirely clean; C actual counts are clean5290,
F1=859,F2=858,F3=908,O1=884,O2=806,O3=895, matching the frozen schedule. Actual pending
source IDs/conditions and consumed-condition counts are retained, distinguishing
presentations from optimizer-consumed gradients.

The verifier checks every journal chain/event, epoch commit and checkpoint object,
all51passes per model with45unique authenticated sources each, matching validation
source/batch order, all actual exposures/steps/EMA/tails, scientific configs and
initialization. It checks selected epoch/fitness/latest-exact-tie against CSV and
checkpoint-write events. Final selected model tensors match the selected committed
raw EMA checkpoint. Best/last load and hash checks pass, plus finite correct-class/
box validation inference and readable six-image galleries. One57001 image from
each new gallery was visually inspected; raw complete outputs for all six are saved.
Gallery coverage uses the shared output-stage counter, not a separate accuracy
measurement. No final-test predictions were generated.

## Checkpoints and clean-validation-only measurements

| Model | Selected epoch | mAP50 | Precision | Recall | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 26 | 0.961969 | 0.950131 | 0.926467 | 0.834265 |
| C | 15 | 0.957435 | 0.903760 | 0.938841 | 0.828851 |

These are post-training best-checkpoint **clean-validation development results**,
not unbiased held-out performance or robustness results. P/R use pinned best-F1
aggregation, not fixed-.25; .25 is only display confidence. Both models trained the
full50epochs although the identical validation rule selected different epochs.
No improvement is assumed: C's displayed clean mAP/precision are lower and recall
higher here. Fading/occlusion robustness and the final E0/E1 conclusion remain
unmeasured. Epoch fitness and reloaded serialized-EMA metrics retain their actual
small numerical differences.

A best SHA `15ed8470368fc535eed861d89dbee09b55977f1f935b40d41b24e052a6d6e4da`;
A last SHA `67b64b76e0ec75843b25c2ff7a6618ff4b00bab23787dee5185b2cb405ffa8bc`.
C best SHA `cb918cad486d0fc6a2eb4c4886cd446bb4aef430d2f6e3581ccc2abff84c8091`;
C last SHA `6645249e9769c6c78cb035775a3f8369ce9eeef77eacb176f091b706b309486e`.
Per-run configurations/source/artifact/journal hashes and all exact metrics are in
the verification receipt; portable provenance, results CSV, accounting, exposure
CSV and gallery prediction JSON are retained alongside it. All model binaries and
raw journals/object stores remain ignored. No external model URL is invented.

## Runtime, native cleanup and resource limits

| Model | Child instrumented seconds | Supervisor monotonic seconds including cleanup | Recorded wall timestamp span seconds |
| --- | ---: | ---: | ---: |
| A | 1889.145 | 2012.405 | 4670.935 |
| C | 1529.639 | 1650.780 | 1650.779 |

Supervisor observations continue after the child training timer ends. Actual
returncodes are0 after cleanup, not inferred from checkpoints. A's wall timestamp
span differs from monotonic elapsed by about2658.53s across the task pause; raw
start/end timestamps and both elapsed measures are retained. The cause of that
discontinuity is not established and is not silently called compute time. C's
wall/monotonic observations agree closely. These real measurements supersede the
old25–50minute planning estimate; do not claim a continuous wall runtime from it.

A/C child one-second observed RSS maxima:1,248,919,552 /1,248,624,640 bytes;
MPS driver maxima:1,429,946,368 /1,434,746,880; tensor maxima:507,308,288 /507,256,832.
Minimum available system bytes:536,100,864 /599,425,024. Maximum global swap in child
samples:14,625,865,728 /14,769,651,712. Full-process2second RSS/resource maxima and
minimum free disk bytes are in the verification receipt. Observations are samples,
not guaranteed peaks; global swap is not solely attributable to training, and
unified counters cannot be added. A read-only native sample during C cleanup
reported footprint21.8G, peak22.1G; this is not resident RAM. Native graph-cache
cleanup and measurement overhead are retained without asserting an unverified
library defect or equating footprint with hardware RAM. No unrelated apps killed.

Neither new training stream contains an NMS time-limit warning. The pinned
unbounded policy and verified source processing remove that specific truncation
path; no assertion about model accuracy follows from completeness alone.

## Preservation and next boundary

The original pair still fails, original C's exit remains unknown, and all16307
historical protected files are unchanged. The new source/config/library and snapshot
hashes verify after training. Large audit checkpoint-object stores are retained
locally (~1.03GB per run); they are not the final below200MB submission bundle.
Future packaging must select verified final deliverables or real external links;
no history is deleted here to meet a packaging limit. The pinned historical
run_all.sh stays unchanged; its full-study reproduction remains incomplete, even
if its static old pending-stage message predates these completed training runs.

**Replacement W5 passed; W6 shared held-out evaluation is the next bounded task.**
Wait for that instruction before generating conditions or predictions. W2/G05 is
still separately blocked on reference terms/use-basis and a defensible executed
reference. G02/G03/G04/G06/G07 final measurement gates, app/reproduction/report/
presentation/second-machine/public-release/submission requirements remain pending.
All20 final acceptance checkboxes stay unchecked. No publication or submission.
