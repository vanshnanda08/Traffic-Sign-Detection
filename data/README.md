# Data storage

All image/label/archive content is excluded from Git.

- `raw/Indian Traffic Sign.v1i.yolov8.zip`: unchanged supplied source.
- `interim/indian_traffic_sign_v1/`: safely extracted and hash-verified source mirror.
- `processed/p57_core300/`: frozen train/val/test YOLO structure and lock.
- `interim/phase1_draft_v0/`: superseded hash-only draft, retained for audit;
  excluded from the final manifest and never used for model development.

See `docs/DATASET_PROTOCOL.md`, `docs/dataset/` and `configs/dataset.json` for
attribution, faculty-approved proposal change, source/version/checksums, exact
class mapping, selection and split seed 42. No degradation or training exists.
