"""Fetch only the explicitly named official assets and record provenance."""

import argparse
import json

from p57.assets import fetch_asset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(fetch_asset(name), indent=2), flush=True)
