#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing scientific .venv. Install pinned requirements.txt; no automatic install/download." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export PYTORCH_ENABLE_MPS_FALLBACK=0
export YOLO_AUTOINSTALL=false
MODE="${1:---reproduce}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  --reproduce|--full)
    exec "$PYTHON" "$ROOT/scripts/reproduce.py" "$@"
    ;;
  --smoke|--validate|--dataset-validate|--protocol)
    mkdir -p "$ROOT/.cache"
    exec "$PYTHON" "$ROOT/scripts/historical_checks.py" "$MODE" "$@"
    ;;
  *)
    echo "Usage: bash run_all.sh [--reproduce [--device cpu|mps] [--assets PATH] [--output NEW_DIR] | --smoke | --validate | --dataset-validate | --protocol]" >&2
    echo "Default: fresh evaluation-only main-table reproduction. Training is never dispatched here." >&2
    exit 2
    ;;
esac
