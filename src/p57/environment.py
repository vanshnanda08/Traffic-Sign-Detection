"""Environment validation, device selection and reproducibility controls."""

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    return json.loads((ROOT / "configs/environment.json").read_text())


def select_device(requested="auto"):
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable")
    return "mps" if available and requested != "cpu" else "cpu"


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Fail on unsupported deterministic operations rather than silently weakening it.
    torch.use_deterministic_algorithms(deterministic)


def validate_pins():
    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Dependency mismatch: {name}: {actual} != {expected}")
        versions[name] = actual
    return versions


def validate_environment(requested=None):
    if sys.version_info[:3] != (3, 12, 16):
        raise RuntimeError("Use the pinned Python 3.12.16 environment")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the project virtual environment")
    cfg = load_config()
    device = select_device(requested or cfg["device"])
    versions = validate_pins()
    seed_everything(cfg["seed"], cfg["deterministic_algorithms"])
    torch.set_num_threads(cfg["torch_threads"])
    # Exercise real compute and synchronization, rather than only checking a flag.
    x = torch.tensor([[1., 2.], [3., 4.]], device=device)
    actual = (x @ x).cpu()
    torch.testing.assert_close(actual, torch.tensor([[7., 10.], [15., 22.]]))
    memory_bytes = None
    chip = platform.processor()
    if sys.platform == "darwin":
        memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {
        "status": "passed", "scope": "environment validation only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "os": platform.platform(), "architecture": platform.machine(),
        "chip": chip, "memory_bytes": memory_bytes,
        "device": device, "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "tensor_check": "passed", "config": cfg, "packages": versions,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "determinism_note": "Seeds and deterministic algorithms are configured; bitwise equivalence across devices/releases is not guaranteed."
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
