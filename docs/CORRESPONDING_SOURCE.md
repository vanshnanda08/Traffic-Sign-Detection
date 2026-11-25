# P57 corresponding-source inventory

This tree includes the project source used to prepare inputs, train the
replacement pair, evaluate the frozen conditions, and run the local app.

| Area | Source and required inputs |
|---|---|
| Installation | `.python-version`, `requirements.txt`, `app/requirements-ui.txt`, `pyproject.toml` |
| Dataset selection, split and transformations | `scripts/prepare_dataset.py`, `src/p57/dataset.py`, `src/p57/relationships.py`, `src/p57/degradation.py`, `configs/dataset_v2.json`, `docs/dataset_v2/manifest.csv` and the relationship inputs |
| Replacement A/C model training | `scripts/train_recovery.py`, `scripts/supervise_recovery.py`, `scripts/create_recovery_contract.py`, `src/p57/recovery_*.py`, `src/p57/main_training.py`, `src/p57/protocol.py`, `src/p57/online.py`, `src/p57/complete_validation.py`, the frozen configurations, execution overlays, budget and schedule |
| W6 evaluation and reproduction | `run_all.sh`, `scripts/reproduce.py`, `scripts/verify_reproduction_run.py`, `src/p57/reproduction.py`, `src/p57/evaluation_*.py`, `configs/reproduction_v1.json`, the frozen evaluator/asset manifest, canonical W6 results and number-to-log evidence |
| Local browser app | `launch_app.py`, `app/streamlit_app.py`, `app/worker.py`, `app/p57_ui/`, `app/models.json` |
| Input installation | `scripts/release_assets.py`, `src/p57/release_assets.py`, `scripts/install_authorized_inputs.py` |

Ultralytics source stays identified by the exact
[v8.3.228 tag](https://github.com/ultralytics/ultralytics/tree/v8.3.228).
The pinned dependency and project notices apply. [Attribution](../NOTICE.md)

The release path starts with lawful access to the exact SDI export and
canonical model files, then verifies hashes and runs evaluation without
training. The older training commands describe the original research
environment. Reconstructing that environment would additionally require
the authorized full source dataset, development serving inputs, pretrained
weights and historical execution sidecars. The release does not provide
them or certify a new training run. The old W4 receipt pins the pre-W8A
launcher and runner test, which are retained under `scripts/historical/`.
Execution snapshots of the recorded source files can be reconstructed from
the overlay path and digest lists. The internal v1 interruption receipt
remains separate from the release and is not required by `bash run_all.sh`.
