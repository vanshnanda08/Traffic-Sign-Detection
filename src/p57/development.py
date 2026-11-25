"""A physically separate train/validation view; final test is never a model input."""

import csv
import json
import shutil
from pathlib import Path

import yaml

from p57.assets import file_sha256
from p57.dataset import ROOT, config, parse_labels, save_json

EXPECTED_LOCK = "ec2b3d0acef8bc03717a606313aae6b6b10ad91d37ebb6e6eecdda72c1fded51"
DEFAULT_VIEW = "data/processed/p57_development_v1"


def verify_frozen_core(cfg=None):
    cfg = cfg or config(ROOT / "configs/dataset_v2.json")
    root = ROOT / cfg["output_dir"]
    lock_path = root / "dataset.lock.json"
    if file_sha256(lock_path) != EXPECTED_LOCK:
        raise ValueError("Unexpected v2 dataset lock; no development/training permitted")
    lock = json.loads(lock_path.read_text())
    for name, expected in lock["files"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return cfg, root, lock


def build_development_view(target=DEFAULT_VIEW):
    cfg, core, lock = verify_frozen_core()
    target = ROOT / target
    files = {name: sha for name, sha in lock["files"].items() if name.startswith(("train/", "val/"))}
    metadata = yaml.safe_load((core / "dataset.yaml").read_text())
    metadata = {key: metadata[key] for key in ("train", "val", "nc", "names")}
    yaml_bytes = yaml.safe_dump(metadata, sort_keys=False).encode()
    if (target / "view.lock.json").exists():
        expected = json.loads((target / "view.lock.json").read_text())
        if expected["core_lock_sha256"] != EXPECTED_LOCK or expected["files"] != files:
            raise ValueError("Development view differs; create a separate version")
    elif target.exists() and any(target.iterdir()):
        raise ValueError("Unrecognized existing development directory; refusing overwrite")
    else:
        for name in files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(core / name, dest)
        (target / "dataset.yaml").write_bytes(yaml_bytes)
        save_json(target / "view.lock.json", {"core_lock_sha256": EXPECTED_LOCK, "files": files})
    if (target / "dataset.yaml").read_bytes() != yaml_bytes:
        raise ValueError("Development YAML changed")
    validate_development_view(target / "dataset.yaml")
    return target / "dataset.yaml"


def validate_development_view(yaml_path):
    yaml_path = Path(yaml_path).absolute()
    # Resolve only after rejecting links, including links on ancestor directories.
    if any(p.is_symlink() for p in (yaml_path, *yaml_path.parents)):
        raise ValueError("External development image/label link")
    yaml_path = yaml_path.resolve()
    data = yaml.safe_load(yaml_path.read_text())
    if set(data) != {"train", "val", "nc", "names"}:
        raise ValueError("Development input must contain train/val/nc/names only; test/download/path overrides forbidden")
    core_lock=ROOT/'data/processed/p57_core300_v2/dataset.lock.json'
    if file_sha256(core_lock)!=EXPECTED_LOCK:
        raise ValueError("Frozen core lock changed")
    frozen=json.loads(core_lock.read_text())
    mapping=frozen['class_mapping']
    names={v['id']:v['name'] for v in mapping.values()}
    if data['nc']!=len(names) or data['names']!=names:
        raise ValueError("Class metadata differs from the frozen v2 mapping")
    if data["train"] != "train/images" or data["val"] != "val/images" or (yaml_path.parent / "test").exists():
        raise ValueError("Unsafe development paths or exposed test directory")
    lock = json.loads((yaml_path.parent / "view.lock.json").read_text())
    if lock["core_lock_sha256"] != EXPECTED_LOCK:
        raise ValueError("Development provenance mismatch")
    expected = {name: sha for name, sha in frozen['files'].items() if name.startswith(('train/', 'val/'))}
    if lock['files'] != expected:
        raise ValueError("Development membership/hashes differ from authenticated frozen core")
    # Historical Ultralytics caches are allowed as derived metadata, never input
    # authority. W4's online dataset parses authenticated labels without caches.
    allowed = set(expected) | {'dataset.yaml', 'view.lock.json', 'train/labels.cache', 'val/labels.cache'}
    for path in yaml_path.parent.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(yaml_path.parent):
            raise ValueError("External development image/label link")
        if path.is_file() and path.relative_to(yaml_path.parent).as_posix() not in allowed:
            raise ValueError("Unexpected development file")
    for split, count in (("train", 210), ("val", 45)):
        images = sorted((yaml_path.parent / split / "images").glob("*"))
        if len(images) != count:
            raise ValueError("Development cardinality mismatch")
        for image in images:
            if image.is_symlink() or not image.resolve().is_relative_to(yaml_path.parent):
                raise ValueError("External development image path")
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            for path in (image, label):
                relative = path.relative_to(yaml_path.parent).as_posix()
                if file_sha256(path) != lock["files"][relative]:
                    raise ValueError("Development bytes changed")
            parse_labels(label.read_text(), data["nc"])
    return data


def development_records():
    cfg = config(ROOT / "configs/dataset_v2.json")
    with (ROOT / cfg["audit_dir"] / "manifest.csv").open() as stream:
        return [r for r in csv.DictReader(stream) if r["split"] in {"train", "val"}]


def require_development_row(row):
    if row["split"] not in {"train", "val"} or not row["image_path"].startswith(row["split"] + "/images/"):
        raise ValueError("Final-test model/degradation access is forbidden during development")
