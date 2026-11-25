"""CLI entry point; run through run_all.sh for consistent process settings."""

import argparse
import json
from pathlib import Path

from p57.environment import ROOT, validate_environment, write_report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_environment(args.device)
    write_report((args.report_dir / "environment.json" if args.report_dir else ROOT / "results/environment/environment.json"), report)
    print(f"Environment passed: Python {report['python']}; device={report['device']}")
    if args.smoke:
        from p57.smoke import run_smoke
        smoke = run_smoke(report["device"])
        write_report((args.report_dir / "smoke.json" if args.report_dir else ROOT / "results/smoke/smoke.json"), smoke)
        print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
