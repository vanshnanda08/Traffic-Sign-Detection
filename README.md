# P57 Indian traffic-sign detection under degradation

P57 compares matched clean-trained and degradation-aware YOLOv8n models on six
classes from the [SDI Indian Traffic Sign v1 dataset](https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1).
The frozen selection has 300 source images (train 210, validation 45, held-out
test 45). Seven conditions reuse the same held-out source set. Original track
and geographic provenance are unknown, and crops and illustrations limit
generalization.

## Results

The clean AP50 values are 98.026% for clean-trained A and 96.282% for
degradation-aware C. Strong occlusion gives C its largest observed gain,
33.436 percentage points. C loses on clean and all three fading conditions.
The stored-prediction paired bootstrap gives O3 a per-condition 95% interval
of +20.587 to +40.954 points for the frozen sources and weights.

- [All W6 results](docs/evaluation_v1/measured/main_table.md)
- [Paired bootstrap](docs/evaluation_v3_stats/paired_ci.csv)
- [Number-to-log map](release_evidence/w8a_v1/number_to_log_ledger.json) and [extension](release_evidence/w8b_v1/number_to_log_ledger.json)

### GTSRB crop-recognition reference

The recorded four-class crop-recognition accuracy was **91.375% (2,193/2,400)**
on the official GTSRB final test crops for speed limit 30, speed limit 80,
no entry and keep right. This is a separate classification result. W6 reports
detection mAP on SDI. The historical training/validation split grouped by
frame index and shared all 242 training tracks across the two partitions, so
its validation score overstates the strength of the model selection evidence.
[Protocol, confusion matrix and run evidence](docs/gtsrb_reference_v1/README.md)

## Setup and reproduction

Use Python 3.12.16 on Apple Silicon macOS, the pinned [requirements](requirements.txt)
and a system-provided Arial font. To reproduce the W6 table, obtain the exact
authorized SDI v1 YOLOv8 export and canonical A/C checkpoints. The evaluator
requires these inputs in addition to the source checkout. Follow the
[authorized-input procedure](docs/AUTHORIZED_INPUTS.md) to acquire, install and
verify their SHA-256 hashes.

```bash
uv venv --python 3.12.16 .venv
uv pip sync --python .venv/bin/python requirements.txt
# Install the exact authorized inputs using the procedure linked above.
.venv/bin/python scripts/release_assets.py verify
bash run_all.sh
```

The default starts fresh CPU evaluation without training. A run on a second
physical Mac remains pending. The local [application](app/README.md) supports
paired inference and local export.

## Hardware and measured runtime

All runs below used one Apple M3 Mac with 8 GiB of unified memory, PyTorch 2.8.0
and Ultralytics 8.3.228.

| Workload | Device | Time |
|---|---|---|
| Original training of A, 50 epochs ([W5 record](docs/W5_RECOVERY_V2_VERIFICATION.md)) | MPS | 2012.405 s |
| Original training of C, 50 epochs (same record) | MPS | 1650.780 s |
| Fresh evaluation of the W6 table: 14 runs, 630 model-source evaluations, all exit codes 0 ([status](release_evidence/w8a_v1/status.json)) | CPU | 214.557 s |

The training times are supervisor monotonic seconds including native cleanup. A's
wall-clock timestamps span 4670.935 s because of a task pause, so that figure is not
continuous compute. The 214.557 s is evaluation only, not training. It ran on the
same Mac with the same installed environment, so it is not a second-machine
reproduction, which remains pending.

## Changes from the Proposal

- **Data.** The proposal requested 300 self-collected, self-annotated street images.
  We use 300 images and labels from the public SDI v1 export and claim no collection
  or annotation of them. Faculty approval is student-reported
  ([details](docs/DATASET_PROTOCOL.md)). Many images are crops or illustrations.
- **GTSRB reference.** The proposal planned a GTSRB reference. GTSRB is cropped-sign
  classification, so the delivered reference is a separate four-class recognizer
  with **91.375% (2,193/2,400)** test accuracy. It is not detection mAP and is not
  comparable to the W6 table. The training split grouped crops by frame index, so
  train and validation shared real tracks and the intended track exclusions were
  ineffective. The validation score is leaky; the test figure is a historical single
  run with that limitation ([audit](docs/gtsrb_reference_v1/evidence_audit.json)).
  Faculty acceptance of this substitute is student-reported; written acceptance was
  unavailable for verification.
- **Outcome.** C is higher than A at O2 (+2.086 points) and O3 (+33.436) and lower
  on clean, F1-F3 and O1. Only O3's paired 95% interval excludes zero
  ([paired bootstrap](docs/evaluation_v3_stats/paired_ci.csv)).

## Credits and scope

SDI/Roboflow v1 supplies the images and labels. Project work includes subset
selection, remapping, degradation scheduling, instrumentation, paired evaluation,
application and reproducibility. The detector uses Ultralytics YOLOv8n, its
pretrained weights and evaluator; other pinned software is credited in the
[scientific](docs/release_v1/scientific_dependency_credits.json) and
[UI](docs/release_v1/ui_dependency_credits.json) inventories.
The referenced
[asset notices](docs/release_v1/ASSET_NOTICES.md) and pinned dependency license
describe the relevant upstream terms. [Project AGPL-3.0 license](LICENSE) ·
[Notices](NOTICE.md) · [Corresponding source](docs/CORRESPONDING_SOURCE.md).
