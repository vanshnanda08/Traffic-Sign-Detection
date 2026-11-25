"""Regressions for selected-core geometry missed by the old approximate audit."""

import csv
import hashlib
import json
import os
import subprocess

import cv2
import numpy as np
import pytest

import p57.dataset as dataset
import p57.relationships as relationships

ROOT = dataset.ROOT
FLAGGED = [
    ("test/images/70093_jpg.rf.a1f8a7ffd1a7e0e3c8adb0a414ea926f.jpg",
     "train/images/70023_jpg.rf.9b3bbb955a22dea8d3b7e1ea2856b79b.jpg"),
    ("train/images/65015_jpg.rf.766904021dd9712d5e0700327edd94c7.jpg",
     "valid/images/65017_jpg.rf.1128da4c408768b05dcb62d8cb7217ff.jpg"),
]


@pytest.fixture(scope="module")
def flagged_features():
    cfg = dataset.config(ROOT / "configs/dataset.json")
    records = {r["path"]: r for r in json.loads((ROOT / "results/dataset/source_records.json").read_text())}
    selected = [records[p] for pair in FLAGGED for p in pair]
    source = dataset.Source(ROOT / cfg["source"])
    try:
        features = relationships.extract_features(source, selected, cfg)
    finally:
        source.close()
    return cfg, records, features


@pytest.mark.parametrize("pair_index", [0, 1])
def test_reported_pairs_accepted_and_ransac_independent_of_call_order(flagged_features, pair_index):
    cfg, records, features = flagged_features
    a, b = (records[p] for p in FLAGGED[pair_index])
    first = relationships.geometric_pair(a, b, features, cfg)
    assert first is not None
    assert first["inliers"] >= 10 and first["inlier_fraction"] >= .65 and first["coverage"] >= .06
    cv2.setRNGSeed(98765)
    noise = np.empty((100, 100), np.float32)
    cv2.randu(noise, 0, 1)
    assert relationships.geometric_pair(b, a, features, cfg) == first


def test_exhaustive_checker_visits_every_pair_without_group_or_candidate_pruning(monkeypatch):
    records = [{"path": str(i), "source_id": str(i), "image_sha256": str(i), "pixel_sha256": str(i),
                "phash": i*100, "dhash": i*13, "split": "train" if i else "test", "group_id": str(i)}
               for i in range(4)]
    visited = []
    monkeypatch.setattr(relationships, "extract_features", lambda *args: {})

    def pair(a, b, *args):
        visited.append((a["path"], b["path"]))
        return {"a": a["path"], "b": b["path"], "inliers": 10, "inlier_fraction": .7, "coverage": .07} if a["path"] == "0" and b["path"] == "3" else None

    monkeypatch.setattr(relationships, "geometric_pair", pair)
    report = relationships.exhaustive_relationship_report(None, list(reversed(records)), dataset.config())
    assert len(visited) == len(set(visited)) == 6
    assert ("0", "3") in visited
    assert report["cross_split_count"] >= 1
    assert report["status"] == "failed" and report["physical_sign_independence_guaranteed"] is False


def test_validator_rejects_accepted_geometry_even_when_stored_groups_are_unique(monkeypatch, tmp_path):
    # Original core satisfies every old validator condition; group IDs alone hide
    # the findings. Inject only the detector result to prove the mandatory call.
    cfg = dataset.config(ROOT / "configs/dataset.json")
    calls = []

    def detected(source, selected, cfg):
        calls.append(len(selected))
        return {"accepted_count": 1, "cross_split_count": 1, "status": "failed"}

    monkeypatch.setattr(dataset, "exhaustive_relationship_report", detected)
    with pytest.raises(ValueError, match="Unresolved accepted selected-core relationships"):
        dataset.validate_prepared(cfg, report_dir=tmp_path)
    assert calls == [300]
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "failed"


def test_corrected_graph_contains_both_flagged_pairs_and_quarantines_test_component():
    cfg = dataset.config()
    assert cfg.get("revision") == "p57_core300_v2"
    records = {r["path"]: r for r in json.loads((ROOT / cfg["audit_dir"] / "source_records.json").read_text())}
    rows = list(csv.DictReader((ROOT / cfg["audit_dir"] / "manifest.csv").open()))
    for a, b in FLAGGED:
        assert records[a]["group_id"] == records[b]["group_id"]
        assert not ({a, b} <= {r["source_path"] for r in rows})
    gid = records[FLAGGED[0][0]]["group_id"]
    assert all(r["split"] == "test" for r in rows if r["group_id"] == gid)


def test_original_freeze_and_historical_evidence_unchanged():
    snapshot = ROOT / "results/phase1_repair/preserved_before.json"
    if not snapshot.exists():
        pytest.skip("Local repair preservation snapshot absent")
    for name, before in json.loads(snapshot.read_text()).items():
        path = ROOT / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == before["sha256"], name
        assert path.stat().st_mtime_ns == before["mtime_ns"], name


def test_cross_class_target_keypoint_count_is_not_a_pruning_rule(monkeypatch):
    # Ten good query matches can map to nine outside target descriptors. The
    # documented rule counts inlier correspondences, not unique target indices.
    cfg = dataset.config()
    outer = np.float32([[10,10],[128,10],[245,10],[10,128],[245,128],
                        [10,245],[128,245],[245,245],[64,30],[200,30]])
    a = {"path": "a", "boxes": [[0,.5,.5,.1,.1]]}
    b = {"path": "b", "boxes": [[1,.5,.5,.1,.1]]}
    target = np.vstack([outer[:9], np.float32([[128,128]])])
    descriptors = np.zeros((10,128), np.float32)
    features = {"a": (outer, descriptors), "b": (target, descriptors)}

    class Matcher:
        def knnMatch(self, *args, **kwargs):
            return [(cv2.DMatch(i,i % 9,1.), cv2.DMatch(i,9,10.)) for i in range(10)]

    monkeypatch.setattr(relationships.cv2, "BFMatcher", Matcher)
    monkeypatch.setattr(relationships.cv2, "findHomography", lambda *args, **kwargs: (np.eye(3),np.ones((10,1),np.uint8)))
    result = relationships.measure_direction(a,b,features,cfg)
    assert result["accepted"] and result["outside_inliers"] == 10


def test_runner_explicit_config_overrides_environment(tmp_path):
    run = subprocess.run([str(ROOT / "run_all.sh"), "--validate", "--config", "configs/dataset_v2.json"],
                         cwd=tmp_path, env={**os.environ, "P57_DATASET_CONFIG": str(ROOT / "configs/dataset.json"),
                                           "P57_RUN_DIR": str(tmp_path / "evidence")},
                         text=True,capture_output=True,timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Dataset revision: p57_core300_v2" in run.stdout
