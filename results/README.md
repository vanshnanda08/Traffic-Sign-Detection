# Local generated evidence

`environment/` and `smoke/` retain Phase 0 validation reports.
`dataset/` contains Phase 1 full-source audit, class distribution, label issues,
duplicate/scene links, source inventory, selection manifest, extraction report,
validation report, reproduction log, review contact sheets and pytest JUnit XML.
These are infrastructure/data records, not model performance results. All outputs
here are ignored by Git; small frozen dataset records are copied to `docs/dataset/`.
A failed command exits nonzero and can leave earlier output; inspect timestamps
and exit status. No model accuracy metrics or trained checkpoints exist.
