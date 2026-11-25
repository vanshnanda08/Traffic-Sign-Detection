"""Install exact evaluation inputs from an authorized SDI export and separate A/C files."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p57.dataset import Source
from p57.release_assets import authority, no_links, safe_name, sha

SOURCE_SHA256 = "d4ce92dfd534681f4f0a70de3a217a4a79e7146382450c3c030888244306a762"
SELECTION_SHA256 = "62f2a195f6e06436841f7a8708f7dbb05cb325070009a5867f24c476de593118"


def install(dataset, weights):
    dataset = no_links(dataset)
    if not dataset.is_file() or sha(dataset) != SOURCE_SHA256:
        raise ValueError("Publisher export must match the frozen SDI v1 YOLOv8 ZIP SHA-256.")
    manifest = authority(ROOT)
    settings = json.loads((ROOT / "configs/reproduction_assets.json").read_text())
    selection_path = ROOT / "docs/dataset_v2/manifest.csv"
    if sha(selection_path) != SELECTION_SHA256:
        raise ValueError("Frozen dataset selection manifest drift.")
    with selection_path.open(newline="") as stream:
        selected = {row["image_path"]: row for row in csv.DictReader(stream)}
    lock = json.loads((ROOT / "docs/dataset_v2/dataset.lock.json").read_text())
    mapping = {int(k): v["id"] for k, v in lock["class_mapping"].items()}

    cache = no_links(ROOT / ".cache")
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p57-inputs-", dir=cache) as directory:
        stage = Path(directory)
        copies = {}
        def stage_file(relative):
            safe_name(relative)
            target = no_links(ROOT / relative)
            if target.exists() or target in copies:
                raise FileExistsError(f"Refusing occupied asset target: {relative}")
            temp = stage / relative
            temp.parent.mkdir(parents=True, exist_ok=True)
            copies[target] = temp
            return temp

        model_hashes = {}
        for model, path in weights.items():
            source_path = no_links(path)
            expected = manifest["checkpoints"][model]["sha256"]
            if not source_path.is_file() or sha(source_path) != expected:
                raise ValueError(f"Model {model} must match the frozen fine-tuned checkpoint.")
            target = stage_file(settings["checkpoints"][model])
            shutil.copyfile(source_path, target)
            if sha(target) != expected:
                raise ValueError(f"Staged Model {model} changed during copy.")
            model_hashes[model] = expected

        source = Source(dataset)
        try:
            for split, rows in manifest["splits"].items():
                for row in rows:
                    key = row["image_path"]
                    record = selected[key]
                    if record["split"] != split or record["label_path"] != row["label_path"]:
                        raise ValueError(f"Frozen selection mismatch: {key}")
                    image = source.read(source.names[record["source_path"]])
                    labels = source.read(source.names[record["source_label_path"]])
                    if (hashlib.sha256(image).hexdigest() != row["image_sha256"] or
                            hashlib.sha256(labels).hexdigest() != record["source_label_sha256"]):
                        raise ValueError(f"Publisher member mismatch: {key}")
                    remapped = []
                    for line in labels.decode().splitlines():
                        if line.strip():
                            tokens = line.split()
                            tokens[0] = str(mapping[int(float(tokens[0]))])
                            remapped.append(" ".join(tokens))
                    label = "\n".join(remapped) + "\n"
                    if (label != row["label_text"] or
                            hashlib.sha256(label.encode()).hexdigest() != row["label_sha256"]):
                        raise ValueError(f"Remapped label mismatch: {key}")
                    stage_file(f'{settings["core"]}/{key}').write_bytes(image)
                    stage_file(f'{settings["core"]}/{row["label_path"]}').write_bytes(label.encode())
        finally:
            source.close()

        installed = []
        try:
            for target, temporary in copies.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(temporary, target)
                installed.append(target)
        except OSError:
            for target in reversed(installed):
                if target.is_file() and target.samefile(copies[target]):
                    target.unlink()
            raise

    return {"status": "passed", "mode": "publisher_export_and_separate_exact_weights",
            "publisher_export_sha256": SOURCE_SHA256, "all_sha256_match": True,
            "installed_payload_files": len(copies), "model_hashes": model_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Authorized exact SDI v1 YOLOv8 ZIP")
    parser.add_argument("--model-a", required=True, type=Path, help="Separately authorized exact A checkpoint")
    parser.add_argument("--model-c", required=True, type=Path, help="Separately authorized exact C checkpoint")
    args = parser.parse_args()
    print(json.dumps(install(args.dataset, {"A": args.model_a, "C": args.model_c}), indent=2))


if __name__ == "__main__":
    main()
