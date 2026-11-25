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

## Setup and reproduction

Use Python 3.12.16 on Apple Silicon macOS, the pinned [requirements](requirements.txt)
and a system-provided Arial font. The local evaluator authenticates all required
files before running. The expected local reproduction archive has SHA-256
`ba3b8cd05cd1ad0ea4c278b132abf3fa8e2f7ebc065e3ceccceb6461682532c4`.
Public assets await distribution clearance. Use an authorized copy.

```bash
uv venv --python 3.12.16 .venv
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python scripts/release_assets.py install --archive /path/to/p57_reproduction_assets_v1.zip
.venv/bin/python scripts/release_assets.py verify
bash run_all.sh
```

The default starts fresh CPU evaluation without training. The local
[application](app/README.md) supports paired inference and local export.

## Credits and scope

SDI/Roboflow v1 supplies the images and labels. Project work includes subset
selection, remapping, degradation scheduling, instrumentation, paired evaluation,
application and reproducibility. The detector uses Ultralytics YOLOv8n, its
pretrained weights and evaluator; other pinned software is credited in the
[scientific](docs/release_v1/scientific_dependency_credits.json) and
[UI](docs/release_v1/ui_dependency_credits.json) inventories.
The referenced
[asset notices](docs/release_v1/ASSET_NOTICES.md) and pinned dependency license
describe the relevant upstream terms. The project distribution basis is
pending approval. GTSRB was unexecuted.
