# W6 detector evaluation verification

Date: 2026-11-21, Asia/Kolkata. **Verdict: passed for detector evaluation,
figures and failure analysis only.** Fourteen real sequential runs exited 0
including cleanup: two replacement detectors × seven conditions × 45 sources =
630 model-source evaluations. No training, checkpoint reselection, reference
training, application development, publication or packaging occurred. There were
no technical reruns of the held-out evaluations and no missing cells.

## Authenticated inputs and freeze

- Model A best: `results/main_ac_recovery_v2/A/fit/weights/best.pt`; SHA-256
  `15ed8470368fc535eed861d89dbee09b55977f1f935b40d41b24e052a6d6e4da`.
- Model C best: `results/main_ac_recovery_v2/C/fit/weights/best.pt`; SHA-256
  `cb918cad486d0fc6a2eb4c4886cd446bb4aef430d2f6e3581ccc2abff84c8091`.
- Replacement protocol: `ffa15a791c5591d19fea1e459cb92b890be42e608d709fca4ef6fbac93e1e8b8`;
  core-v2 lock: `ec2b3d0acef8bc03717a606313aae6b6b10ad91d37ebb6e6eecdda72c1fded51`.
- Evaluation config SHA-256: `b47f309933a1f0c9f9253519dc90f8b95517e9f572db57b8917484a7f02ef685`.
- Evaluator receipt SHA-256: `1259b36c461d8285a3f6ce4d6097ea10d7d6b9457e68d6b76a620b2d8617f4a5`.
- Paired manifest SHA-256: `dfdf205f17343749f5e6756045d359039203e046d85615aa4b011cc9ae5fb71a`.
- [Code/library/checkpoint receipt](evaluation_v1/evaluator_receipt.json),
  [post-run audit](evaluation_v1/post_run_verification.json),
  [actual commands, child processes, logs and exits](evaluation_v1/execution_ledger.json),
  [full method/limitations](evaluation_v1/README.md).

The initial full replacement pair verification passed again before this handoff;
`results/w6_verification_v1/starting_pair_verification.json` retains that verdict.
Checkpoint SHA checks and original passed model lineage/actual exits were enforced
before every worker. Historical, trial and last weights were never substituted.
W5's training validator remains development-only. The new narrow validator accepts
only authenticated sources from the separate paired held-out view.

Before freeze, both A/C were evaluated on development validation only with the
existing complete-validation path and new evaluator at identical settings. Each
path completed all 45 sources/46 boxes, with exactly zero differences in the four
aggregate metrics and captured network boxes/classes/confidences. Tolerances were
predeclared as metric absolute 1e-10, prediction absolute 1e-6; neither was widened.
Preflight measured monotonic runtime: 11.795566s. Its raw results,
coverage and memory observations are in `results/w6_verification_v1/preflight/`.
Freeze occurred before creation of every derived test image and prediction file;
file ordering and frozen source/snapshot/library hashes were reverified afterward.

## Conditions and actual coverage

The original 45 clean test sources and label bytes were preserved. Seven views
produce315 PNG/label pairs, exactly 45 sources and 45 ground-truth boxes per condition.
The test set contains no multi-box source; the development set contains 46 boxes
and synthetic overlapping multi-box tests verify geometry preservation. The
six test class counts are8,8,8,7,7,7 in frozen class-ID order: COMPULSARY_KEEP_RIGHT,
NO_ENTRY, NO_STOPPING_OR_STANDING, ROUNDABOUT, SPEED_LIMIT_30, SPEED_LIMIT_80.

Every view, including clean, uses lossless PNG. Clean decoded pixels equal the
original decoded pixels; all 315views were recomputed using the pinned degradation
implementation and compared exactly in the post-run audit. Original JPEGs were
never overwritten. The JSON manifest records file, decoded-pixel, label and
configuration/source hashes, mask details and true box counts. The portable
[JSON](evaluation_v1/paired_manifest.json) and [CSV](evaluation_v1/paired_manifest.csv)
retain source pairing. Source IDs and all 12batch memberships/order match across
all 14 runs. Each run records45 NMS completions and 45 metric updates before returning
metrics. Empty detections are allowed only for actually processed sources.

Frozen settings: nominal imgsz416, MPS FP32, half/AMP false, batch4, rect true,
workers0, confidence floor .001, NMS IoU .7, max_det300, class-aware NMS,
validator multi_label true, no augmentation/class filter. The existing loader's
stride32/default pad.5 is retained; **all measured input tensors were448×448**.
This is transparent preservation of the preflight-matched preprocessing, not a
claim that the network received416×416 tensors. Absolute YAMLs contain null
train/val and explicit test paths, with unchanged six-class metadata; downloads,
redirects and symlinked YAML ancestors are rejected. No installed package/global
monkeypatch or separate predictor stream was used. Missing execution flags fail
before held-out work. A shared lock prevented concurrent training/evaluation jobs.

## Measured results

| Condition | A mAP50 % | C mAP50 % | C-A pp | A P % | C P % | A R % | C R % |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 98.026 | 96.282 | -1.745 | 97.662 | 92.080 | 97.107 | 93.306 |
| F1 | 97.944 | 97.130 | -0.814 | 97.686 | 92.649 | 97.550 | 95.288 |
| F2 | 97.585 | 96.244 | -1.341 | 94.386 | 91.140 | 94.231 | 95.238 |
| F3 | 96.656 | 92.900 | -3.756 | 91.326 | 82.619 | 90.329 | 89.729 |
| O1 | 97.485 | 97.110 | -0.375 | 92.009 | 91.315 | 97.227 | 94.011 |
| O2 | 93.619 | 95.705 | +2.086 | 84.296 | 89.896 | 90.396 | 94.572 |
| O3 | 57.773 | 91.209 | +33.436 | 56.840 | 80.176 | 50.214 | 91.729 |

P/R: evaluator best-F1 convention, not fixed0.25. Machine metrics are fractions.


[Fourteen-row metrics JSON, including per-class results](evaluation_v1/measured/metrics_14.json),
[metrics CSV](evaluation_v1/measured/metrics_14.csv),
[paired fraction-unit table](evaluation_v1/measured/paired_table.json),
[fading curves PNG](evaluation_v1/measured/fading_robustness.png),
[occlusion curves PNG](evaluation_v1/measured/occlusion_robustness.png).
PDF versions and exact fraction-unit curve data are in the same measured folder.
Raw metrics/predictions and logs remain under `results/detector_eval_v1/`.
Each canonical row traces its checkpoint, config, condition manifest, metrics and
actual evaluator prediction file by SHA-256. Aggregate and all six per-class
metrics were independently recomputed after JSON serialization using the pinned
AP calculation and stored exact float32 metric-space IoUs/TP masks; all differences
are within the declared absolute1e-10 tolerance. CSV/JSON tables, signed C−A
fractions, percentage-point display and plotted values were checked for consistency.

C's O3 mAP50 is higher by 33.436 percentage points and O2 by 2.086 points, while clean
and all fading levels favor A; O1 also favors A. Degradation-aware training therefore
has a measured strong-occlusion benefit on this benchmark, not universal superiority.
No thresholds, severities, classes, preprocessing or checkpoints were changed in
response to test scores. These numbers do not demonstrate statistical significance
or robustness on independent real-road scenes. The 45-source test is small and
many source samples are crops/illustrations with unknown upstream physical/track/
geographic provenance. Controlled rectangular box-region masks are not all real
occlusions; fading is not every natural weathering process. See method limitations.

## Deterministic failure analysis and visual verification

All 630stored records were classified for explanation at confidence >=.25,
class-aware metric-space IoU.5. This is separate from best-F1 evaluator P/R.
The inventory contains 123 FP and 68 FN across repeated model/condition views; these
are explanatory counts, not counts of independent road scenes. Wrong-class
predictions can produce both FP and FN. Ten paired panels were selected under
rules declared before test predictions: three distinct-source FP, three FN,
strictly negative/positive C cases, a hardest equal-error neutral case and a
hardest combined-error case. No requested category was absent. Selection metadata
pins both actual evaluator prediction files. All raw detections are retained;
panels draw at most8 top-confidence predictions with GT/class/confidence legends.

- [FP: 11094_jpg, F3](evaluation_v1/measured/failures/00_FP_F3_11094_jpg.png)
- [FP: 11099_jpg, O3](evaluation_v1/measured/failures/01_FP_O3_11099_jpg.png)
- [FP: 44125_jpg, F3](evaluation_v1/measured/failures/02_FP_F3_44125_jpg.png)
- [FN: 11008_jpg, F3](evaluation_v1/measured/failures/03_FN_F3_11008_jpg.png)
- [FN: 11029_jpg, O3](evaluation_v1/measured/failures/04_FN_O3_11029_jpg.png)
- [FN: 11033_jpg, O3](evaluation_v1/measured/failures/05_FN_O3_11033_jpg.png)
- [negative_for_C: 11094_jpg, F3](evaluation_v1/measured/failures/06_negative_for_C_F3_11094_jpg.png)
- [positive_for_C: 11029_jpg, O3](evaluation_v1/measured/failures/07_positive_for_C_O3_11029_jpg.png)
- [neutral: 44125_jpg, F3](evaluation_v1/measured/failures/08_neutral_F3_44125_jpg.png)
- [difficult: 44125_jpg, F3](evaluation_v1/measured/failures/09_difficult_F3_44125_jpg.png)

Visual inspection checked both full robustness plots, the fixed first-source
seven-condition contact sheet, and the FN/negative-C/positive-C/neutral panels.
The negative C example11094/F3 has A0 versus C3 FP/FN errors;11029/O3 has A2 versus
C0 errors;44125/F3 is neutral by total errors (A2,C2) but differs in error type.
These examples show mixed outcomes rather than cherry-picking only improvements.
Canonical artifact hashes and byte-identical portable copies are recorded in
[evaluation copy receipt](evaluation_v1/measured_copy_receipt.json).
The [final integrity receipt](evaluation_v1/final_integrity.json) additionally
recomputes the deterministic selections, checks all23portable copies and315CSV
rows, confirms test class box counts and zero NMS timeout-warning logs, rechecks
all protected hashes/mtimes, and validates128local documentation links. The
[source/policy receipt](evaluation_v1/final_source_checks.json) records syntax
checks and the supplemental read-only auditor hash without changing the frozen
evaluator receipt.

## Executed tests and runtime records

- Focused development suite:17 passed initially, then 18after preprocessing parity,
  then 19after YAML-ancestor hardening. Logs/XML are preserved in
  `results/w6_verification_v1/focused_tests*`. All occurred before preflight/freeze.
- Full smoke/regression suite initially 155 passed/98.72s, then final156 passed/90.75s,
  exit0, no skips. Final command and XML:
  `P57_RUN_DIR=results/w6_verification_v1/final_regression_checks bash run_all.sh --smoke --config configs/dataset_v2.json`;
  `results/w6_verification_v1/final_regression_checks/smoke/pytest.xml`.
  The 156include the 19 focused cases; these are not175distinct tests.
- Post-run focused repeat:19 passed/1.88s, exit0. It performs no real test inference.
- Development preflight passed for both models; freeze,315-pair generation,
  all 14evaluations, artifacts and read-only post-run audit each exited0.
- Tests cover split/checkpoint drift, source coverage (including legitimate empty
  outputs and missing/duplicate/foreign rejection), deterministic overlapping
  multi-box transforms, PNG integrity, class IDs, pinned matching/metrics,
  fraction/percentage units, missing-cell rejection, delayed-clock NMS completeness,
  ordinary-case NMS equivalence and old/new development preprocessing equality.

Executed machine: Apple M3,8GiB unified memory; Python3.12.16, torch2.8.0,
Ultralytics8.3.228, actual MPS FP32. Per-run package versions, original/input shapes,
configuration, checkpoint SHA and measured evaluator runtime are in metrics JSON.
Supervisor one-second RSS/available/global-swap/disk observations continue through
native cleanup, with actual child exit records and raw stdout/stderr. Child
one-second observations include MPS tensor/driver allocations before native cleanup.

| Run | Supervisor monotonic seconds incl. cleanup | Wall span seconds | Exit |
|---|---:|---:|---:|
| A_clean | 18.102717 | 18.102734 | 0 |
| C_clean | 20.283865 | 20.283881 | 0 |
| A_F1 | 20.336690 | 20.336703 | 0 |
| C_F1 | 18.276854 | 18.276870 | 0 |
| A_F2 | 17.109732 | 17.109744 | 0 |
| C_F2 | 18.151356 | 18.151373 | 0 |
| A_F3 | 17.114292 | 17.114304 | 0 |
| C_F3 | 19.133219 | 19.133244 | 0 |
| A_O1 | 21.300851 | 21.300868 | 0 |
| C_O1 | 19.259735 | 19.259752 | 0 |
| A_O2 | 20.363927 | 20.363943 | 0 |
| C_O2 | 19.274852 | 19.274865 | 0 |
| A_O3 | 18.260840 | 18.260857 | 0 |
| C_O3 | 18.177651 | 18.177665 | 0 |

Total sequential supervised runtime including each child cleanup:
265.146580s.
Largest absolute wall-span vs monotonic-span difference:
0.000024996s.
Both raw clock measurements are retained; no cause is inferred. Historical W5
wall-clock discrepancies also remain unchanged. Maximum observed child RSS:
737,247,232bytes; minimum observed system
available memory:952,434,688bytes;
maximum observed global swap:6,182,338,560bytes.
Maximum sampled child MPS tensor/driver allocations:
68,372,480/1,162,772,480bytes.
Sampling can miss peaks; RSS/MPS/unified-memory measurements overlap and are not
additive. Global swap can predate or belong to other processes; it is not this
job's exclusive memory consumption. No memory-failure explanation is invented.

## Preservation, changed files and acceptance boundary

The read-only audit rehashed all 16,970 protected files and compared mtimes; every
entry matches. This includes old source/configs/snapshots/receipts, raw and both
core dataset revisions, selected/last/initial checkpoints and historical runs.
Preserved data locks include:

- `data/interim/phase1_draft_v0/p57_core300/dataset.lock.json`: `135e17cb792b3c6e8254f60972609f52cab352abf00f90b0b96e7c8118649ecf`
- `data/processed/p57_core300/dataset.lock.json`: `5fcd5289506b17f471bf0cd878a37aadeb29503552e4a916e64a563f70c371fe`
- `data/processed/p57_core300_v2/dataset.lock.json`: `ec2b3d0acef8bc03717a606313aae6b6b10ad91d37ebb6e6eecdda72c1fded51`

Original interrupted C remains historical/incomplete and its exit remains UNKNOWN;
no new old-C exit was synthesized. Both replacement best hashes still match the
explicitly authorized inputs. The old runner, requirements, environment and W4/W5
pinned code were not edited. Git exclusions retain data/checkpoints/raw predictions
outside the source inventory. No commit, push, publication or external links were
created in this handoff.

New implementation: `configs/evaluation_v1.json`;
`src/p57/evaluation_data.py`, `evaluation_engine.py`, `evaluation_protocol.py`,
`evaluation_artifacts.py`; scripts `evaluation_preflight.py`, `freeze_evaluator.py`,
`evaluate_heldout.py`, `supervise_evaluations.py`, `evaluation_artifacts.py`,
`verify_evaluation.py`; `tests/test_evaluation.py`; this verification and
`docs/evaluation_v1/` receipts/manifests/measured artifacts. Current status,
roadmap, checklist, workflow, proposal-change record and README are updated;
prior evidence and historical dated entries are preserved.

**Detector gates G02/G03/G04 now pass.** G06/G07 have verified detector metrics,
curves/tables/FP/FN evidence but remain partial/in progress because their GTSRB
reference requirements are unexecuted. W2/G05 stays separately blocked. G01 final
scope acceptance, application, public release/checkpoint distribution, main-runner
reproduction, second-machine check, report/slides/team sign-off/viva/submission
remain pending as documented. No new independent completion score is assigned;
the 59/100 snapshot predates W6.

The next independent bounded engineering task can be W7 offline application work
using these frozen A/C checkpoints, with no training or test-driven retuning.
Separately resolve W2/G05's terms/use-basis, class/track independence and task
protocol before any reference training. Neither task starts in this handoff.
Stop after W6 detector evaluation, figures and failure analysis.
