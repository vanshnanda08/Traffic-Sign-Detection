#!/usr/bin/env python3
"""Phase 1 CLI. Use run_all.sh --dataset, or PYTHONPATH=src with .venv Python."""

import argparse
from pathlib import Path

from p57.dataset import ROOT, Source, audit, config, prepare, safe_extract, save_json, validate_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["extract", "audit", "prepare", "validate", "all"])
    parser.add_argument("--config")
    parser.add_argument("--report-dir", type=Path, help="Separate validation evidence directory (never changes frozen data)")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.mode in {"extract", "all"}:
        path = ROOT / cfg["source"]
        # Source provides actionable missing-download instructions. Already extracted
        # directories are supported without attempting to treat them as ZIPs.
        source = Source(path)
        source.close()
        report = (safe_extract(path, ROOT / "data/interim/indian_traffic_sign_v1") if path.is_file() else
                  {"status": "existing_directory", "destination": str(path), "extraction": "not needed"})
        save_json(ROOT / cfg["audit_dir"] / "extraction.json", report)
        print(report)
    if args.mode in {"audit", "all"}:
        audit(cfg)
    if args.mode in {"prepare", "all"}:
        prepare(cfg)
    if args.mode == "validate":
        validate_prepared(cfg, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
