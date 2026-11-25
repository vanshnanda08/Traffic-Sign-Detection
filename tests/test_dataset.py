"""Phase 1: invalid-label, grouping, mapping, immutability and real-data checks."""

import copy
import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from p57.dataset import (
    ROOT, Source, candidate_pools, choose_classes, config, group_records,
    parse_labels, prepared_payload, safe_extract, stratified_sample, validate_prepared,
)


@pytest.mark.parametrize("text", [
    "0 .5 .5 0 .2", "0 .5 .5 -.1 .2", "0 nan .5 .1 .2", "0 inf .5 .1 .2",
    "-1 .5 .5 .1 .2", "2 .5 .5 .1 .2", "0.5 .5 .5 .1 .2",
    "0 .95 .5 .2 .2", "0 .5 1.1 .2 .2", "0 .5 .5 1.1 .2",
    "0 .5 .5 .2", "0 .5 .5 .2 .2 .1", "cat .5 .5 .2 .2",
    "0 .5 .5 .2 .2\n0 .5 .5 .2 .2",
])
def test_bad_detection_annotations_rejected(text):
    with pytest.raises(ValueError):
        parse_labels(text, 2)


def test_valid_detection_labels_and_explicit_negative():
    assert parse_labels("1 .5 .5 1 1\n", 2) == [[1, .5, .5, 1., 1.]]
    assert parse_labels("", 2) == []


def make_record(path, split, sid, phash, dhash, pixel):
    return {"path": path, "source_id": sid, "split": split, "phash": phash, "dhash": dhash,
            "image_sha256": path, "pixel_sha256": pixel, "boxes": [[0, .5, .5, .2, .2]],
            "issues": [], "augmentation_marker": False}


def test_transitive_related_group_never_promotes_test_relatives_to_train():
    records = [make_record("train/a", "train", "same", 0, 0, "a"),
               make_record("val/b", "val", "same", 65535, 65535, "b"),
               make_record("test/c", "test", "other", 65534, 65534, "c")]
    groups, edges = group_records(records, 1, 1)
    assert len(groups) == 1
    assert {e["reason"] for e in edges} == {"source_id", "perceptual"}
    pools = candidate_pools(groups)
    assert not pools[0]["train"] and not pools[0]["val"]
    assert [r["path"] for r in pools[0]["test"]] == ["test/c"]


def test_conflicting_exact_class_annotations_excluded():
    a = make_record("train/a", "train", "a", 0, 0, "same-pixels")
    b = make_record("test/b", "test", "b", 999, 999, "same-pixels")
    b["boxes"][0][0] = 1
    groups, _ = group_records([a, b], 0, 0)
    assert not candidate_pools(groups)


def test_insufficient_real_class_support_fails_instead_of_duplicating():
    with pytest.raises(ValueError, match="Insufficient"):
        choose_classes({}, [], config())


def test_stratified_selection_repeats_without_duplicate_sampling():
    candidates = [{"path": str(i), "max_box_area": i/100, "sharpness": 100-i} for i in range(80)]
    a = stratified_sample(candidates, 35, random.Random(42))
    b = stratified_sample(candidates, 35, random.Random(42))
    assert a == b and len({r["path"] for r in a}) == 35


@pytest.fixture
def actual_dataset():
    cfg = config()
    if not (ROOT / cfg["output_dir"] / "dataset.lock.json").is_file():
        pytest.skip("Real Phase 1 artifacts absent; run ./run_all.sh --dataset")
    return cfg


def test_actual_all_annotations_mapping_splits_and_frozen_hashes(actual_dataset,tmp_path):
    result = validate_prepared(actual_dataset,report_dir=tmp_path)
    assert result["selected_images"] == 300
    # Single-class images can contain multiple genuine boxes of that class.
    assert result["selected_boxes"] >= result["selected_images"]
    assert result["split_counts"] == {"train": 210, "val": 45, "test": 45}
    assert len(result["per_class_images"]) == 6
    assert sum(result["per_class_images"].values()) == 300
    assert result["exhaustive_pairs_checked"] == 44850
    assert result["relationship_statement"] == "No relationships detected under this protocol"
    assert result["physical_sign_independence_guaranteed"] is False


def test_actual_selection_is_deterministic_and_uses_only_eligible_groups(actual_dataset):
    cfg = actual_dataset
    out = ROOT / cfg["audit_dir"]
    records = json.loads((out / "source_records.json").read_text())
    summary = json.loads((out / "source_audit.json").read_text())
    a, mapping, _ = prepared_payload(cfg, records, summary)
    b, mapping2, _ = prepared_payload(cfg, list(reversed(records)), summary)
    assert mapping == mapping2
    assert [(r["path"], r["assigned_split"]) for r in a] == [(r["path"], r["assigned_split"]) for r in b]
    group_splits = {}
    for r in records:
        group_splits.setdefault(r["group_id"], set()).add(r["split"])
    for r in a:
        if r["assigned_split"] in {"train", "val"}:
            assert "test" not in group_splits[r["group_id"]]
        else:
            assert r["split"] == "test"


def test_actual_selected_pairs_have_no_geometric_or_perceptual_relationships(actual_dataset):
    from p57.relationships import exhaustive_relationship_report
    cfg = actual_dataset
    out = ROOT / cfg["audit_dir"]
    records = json.loads((out / "source_records.json").read_text())
    summary = json.loads((out / "source_audit.json").read_text())
    selected, _, _ = prepared_payload(cfg, records, summary)
    source = Source(ROOT / cfg["source"])
    try:
        report = exhaustive_relationship_report(source, list(reversed(selected)), cfg)
    finally:
        source.close()
    assert report["pairs_checked"] == 44850
    assert report["accepted_count"] == 0
    assert report["approximate_candidate_pruning"] is False
    assert report == json.loads((out / "preparation_relationships.json").read_text())


def test_ultralytics_resolves_real_dataset_yaml_from_outside_repo(actual_dataset, monkeypatch, tmp_path):
    # Dataset parser only. No YOLO model construction, inference or training.
    from ultralytics.data.utils import check_det_dataset
    monkeypatch.chdir(tmp_path)
    root = ROOT / actual_dataset["output_dir"]
    result = check_det_dataset(str(root / "dataset.yaml"), autodownload=False)
    for split in ("train", "val", "test"):
        assert Path(result[split]).resolve() == root / split / "images"


def test_missing_source_gives_precise_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="version 1.*YOLOv8 detection ZIP"):
        Source(tmp_path / "missing.zip")


@pytest.mark.parametrize("kind", ["image", "label", "manifest"])
def test_frozen_files_tampering_detected_without_repair(actual_dataset, tmp_path, monkeypatch, kind):
    import p57.dataset as dataset
    cfg = copy.deepcopy(actual_dataset)
    original = ROOT / cfg["output_dir"]
    target = tmp_path / cfg["output_dir"]
    shutil.copytree(original, target)
    audit_dir = tmp_path / cfg["audit_dir"]
    audit_dir.mkdir(parents=True)
    for name in ("manifest.csv", "selection.json", "source_records.json", "preparation_relationships.json"):
        shutil.copyfile(ROOT / cfg["audit_dir"] / name, audit_dir / name)
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    path = (next((target / "test/images").glob("*")) if kind == "image" else
            next((target / "test/labels").glob("*")) if kind == "label" else audit_dir / "manifest.csv")
    path.write_bytes(b"corrupt fixture, never production data")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Frozen file changed|Manifest changed"):
        validate_prepared(cfg)
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "x/../../escape.txt", "C:\\escape.txt"])
def test_extraction_rejects_traversal_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("safe.txt", "safe fixture")
        z.writestr(name, "unsafe fixture")
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_extract(archive, destination)
    assert not destination.exists()


def test_extraction_rejects_symlinks(tmp_path):
    archive = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as z:
        z.writestr(info, "../target")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_extract(archive, tmp_path / "output")


def test_extraction_is_idempotent_and_detects_changed_files(tmp_path):
    archive = tmp_path / "safe.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("nested/a.txt", "original fixture")
    destination = tmp_path / "output"
    assert safe_extract(archive, destination) == safe_extract(archive, destination)
    (destination / "nested/a.txt").write_text("changed fixture")
    with pytest.raises(ValueError, match="changed"):
        safe_extract(archive, destination)
    assert (destination / "nested/a.txt").read_text() == "changed fixture"


def test_cli_missing_source_explains_exact_download(tmp_path):
    cfg = config()
    cfg["source"] = str(tmp_path / "missing.zip")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    run = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dataset.py"), "all", "--config", str(config_file)],
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert run.returncode != 0
    assert "version 1, YOLOv8 detection ZIP" in run.stderr


def test_cli_accepts_an_already_extracted_source(tmp_path):
    # Tiny read-only export fixture, not research data.
    export = tmp_path / "export"
    export.mkdir()
    (export / "data.yaml").write_text("nc: 1\nnames: [fixture]\n")
    cfg = config()
    cfg.update(source=str(export), audit_dir=str(tmp_path / "audit"))
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    run = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dataset.py"), "extract", "--config", str(config_file)],
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "existing_directory" in run.stdout
