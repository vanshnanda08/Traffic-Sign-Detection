"""Check tracked/eligible files for research binaries, secrets and >100 MiB files."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode().split("\0")
    problems = []
    checked = 0
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if not path.is_file():
            continue
        checked += 1
        if path.stat().st_size > 100_000_000:
            problems.append(f"File above 100 MB: {name}")
        if (name.startswith((".venv/", "checkpoints/", "weights/", "data/raw/", "data/interim/", "data/processed/"))
                and path.name != ".gitkeep"):
            problems.append(f"Research binary/environment path in Git inventory: {name}")
        if path.suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".zip"}:
            problems.append(f"Binary checkpoint/archive in Git inventory: {name}")
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            problems.append(f"Environment secret file in Git inventory: {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Repository artifact policy passed: {checked} tracked/eligible files; no disallowed paths or files above 100 MB.")


if __name__ == "__main__":
    main()
