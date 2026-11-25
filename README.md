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
