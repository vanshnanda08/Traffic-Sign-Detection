#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:---smoke}"
if [[ $# -gt 0 ]]; then shift; fi
usage() { echo "Usage: $0 [--validate|--smoke|--dataset|--dataset-validate|--protocol|--development|--gtsrb-audit|--feasibility|--full] [--config DATASET_PATH] [--trial-config PATH] [--run-dir PATH]" >&2; }
TRIAL_CONFIG="configs/feasibility_v1.json"
TRIAL_RUN=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--config" && $# -ge 2 ]]; then
    export P57_DATASET_CONFIG="$2"
    shift 2
  elif [[ "$1" == "--trial-config" && $# -ge 2 ]]; then
    TRIAL_CONFIG="$2"; shift 2
  elif [[ "$1" == "--run-dir" && $# -ge 2 ]]; then
    TRIAL_RUN="$2"; shift 2
  else
    usage; exit 2
  fi
done
case "$MODE" in
  --validate|--smoke|--dataset|--dataset-validate|--protocol|--development|--gtsrb-audit|--feasibility|--full) ;;
  *) usage; exit 2 ;;
esac
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Follow README.md setup instructions." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=57
export PYTORCH_ENABLE_MPS_FALLBACK=0
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export YOLO_AUTOINSTALL=false
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
REPORT_ROOT="${P57_RUN_DIR:-$("$PYTHON" -c 'from p57.dataset import ROOT, config; c=config(); print(ROOT / c.get("validation_dir", "results/phase1_repair/legacy_validation").rsplit("/",1)[0] / "runs")')}/${MODE#--}"
mkdir -p "$REPORT_ROOT"
"$PYTHON" -c 'from p57.dataset import config; c=config(); print("Dataset revision:", c.get("revision", "historical"), "| output:", c["output_dir"])'
"$PYTHON" scripts/validate_environment.py --report-dir "$REPORT_ROOT"
if [[ "$MODE" == "--full" ]]; then
  cat >&2 <<'MESSAGE'
INCOMPLETE: Main experiment reproduction is not implemented.
Development examples and three-epoch feasibility artifacts are not completed baseline results.
Pending: main baseline training; paired final-test degradation/evaluation; GTSRB
reference; degradation-aware training; ablation; failure analysis and report.
No experiment metrics for the final main study are produced by this mode. Exit code: 2.
MESSAGE
  exit 2
fi
if [[ "$MODE" == "--protocol" ]]; then
  "$PYTHON" scripts/verify_protocol.py --report-dir "$REPORT_ROOT"
  "$PYTHON" -m pytest tests/test_protocol.py tests/test_development.py tests/test_gtsrb_audit.py --junitxml="$REPORT_ROOT/pytest.xml"
fi
if [[ "$MODE" == "--development" ]]; then
  "$PYTHON" scripts/development_examples.py
  "$PYTHON" -m pytest tests/test_development.py tests/test_gtsrb_audit.py --junitxml="$REPORT_ROOT/pytest.xml"
fi
if [[ "$MODE" == "--gtsrb-audit" ]]; then
  "$PYTHON" scripts/audit_gtsrb.py
  "$PYTHON" scripts/complete_gtsrb_inventory.py
fi
if [[ "$MODE" == "--feasibility" ]]; then
  if [[ -z "$TRIAL_RUN" ]]; then echo "--feasibility requires a new --run-dir; only three epochs authorized" >&2; exit 2; fi
  export PYTHONHASHSEED=42
  "$PYTHON" scripts/run_feasibility.py train --config "$TRIAL_CONFIG" --run-dir "$TRIAL_RUN"
fi
if [[ "$MODE" == "--dataset" ]]; then
  "$PYTHON" scripts/prepare_dataset.py all
  "$PYTHON" -m pytest tests/test_dataset.py tests/test_relationships.py --junitxml="$REPORT_ROOT/pytest.xml"
  "$PYTHON" scripts/check_repository_files.py
fi
if [[ "$MODE" == "--dataset-validate" ]]; then
  "$PYTHON" scripts/prepare_dataset.py validate --report-dir "$REPORT_ROOT"
fi
if [[ "$MODE" == "--smoke" ]]; then
  "$PYTHON" scripts/validate_environment.py --smoke --report-dir "$REPORT_ROOT"
  "$PYTHON" -m pytest --junitxml="$REPORT_ROOT/pytest.xml"
fi
