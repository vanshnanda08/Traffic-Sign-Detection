"""Validate the entire separate UI lock without installing or importing torch."""
import importlib.metadata
import json
from pathlib import Path

versions={}
for line in (Path(__file__).resolve().parent/'requirements-ui.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    package,expected=line.split('==')
    actual=importlib.metadata.version(package)
    if actual!=expected:raise SystemExit(f'UI dependency mismatch: {package}: {actual} != {expected}. Sync the UI lock explicitly.')
    versions[package]=actual
print(json.dumps({'status':'passed','packages':versions},indent=2))
