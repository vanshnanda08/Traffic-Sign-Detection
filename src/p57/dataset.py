"""Auditable YOLO detection subset preparation. No model inference or training."""

import csv
import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import yaml

from p57.relationships import extract_features, geometric_pair, exhaustive_relationship_report, POLICY_VERSION

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config(path=None):
    if path is None:
        path = os.environ.get("P57_DATASET_CONFIG")
        active = ROOT / "configs/dataset_active.json"
        if path is None and active.exists():
            path = json.loads(active.read_text())["config"]
    return json.loads((ROOT / path if path else ROOT / "configs/dataset.json").read_text())


def assert_writable_revision(cfg):
    if ((ROOT / cfg["audit_dir"]).resolve() == (ROOT / "results/dataset").resolve() or
            (ROOT / cfg["output_dir"]).resolve() == (ROOT / "data/processed/p57_core300").resolve()):
        raise ValueError("Historical revision is read-only. Use configs/dataset_v2.json; validate legacy data with a separate report directory.")


def safe_extract(archive, destination):
    """Validate all paths/size/CRC, stage extraction and never overwrite source files."""
    archive, destination = Path(archive), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(archive) as z:
        entries = z.infolist()
        names = [info.filename for info in entries]
        if len(names) != len(set(names)) or len(names) > 200000:
            raise ValueError("Duplicate ZIP members or excessive entry count")
        if sum(info.file_size for info in entries) > 4*1024**3:
            raise ValueError("Archive exceeds 4 GiB extraction limit")
        for info in entries:
            name = info.filename
            if (PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
                    "\\" in name or ":" in name or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {name}")
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
        expected = {i.filename: digest(z.read(i)) for i in entries if not i.is_dir()}
        marker = {"archive_sha256": archive_hash, "files": expected}
        if destination.exists():
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
            if actual != set(expected) | {".source_extraction.json"}:
                raise ValueError("Existing extraction inventory differs; refusing to overwrite")
            if json.loads((destination / ".source_extraction.json").read_text()) != marker:
                raise ValueError("Existing extraction archive identity differs")
            for name, h in expected.items():
                p = destination / name
                if p.is_symlink() or digest(p.read_bytes()) != h:
                    raise ValueError(f"Existing extracted file changed: {name}")
        else:
            with tempfile.TemporaryDirectory(prefix="p57-extract-", dir=destination.parent) as tmp:
                stage = Path(tmp) / "dataset"
                stage.mkdir()
                for info in entries:
                    path = stage / info.filename
                    if info.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(z.read(info))
                save_json(stage / ".source_extraction.json", marker)
                stage.rename(destination)
    return {"status": "passed", "archive_sha256": archive_hash,
            "files": len(expected), "destination": str(destination), "crc_check": "passed"}


class Source:
    """Read ZIPs without extraction, or an existing exported directory."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download SDI/indian-traffic-sign version 1, "
                "YOLOv8 detection ZIP, and place it in data/raw/. No API key is required in this repository."
            )
        self.zip = ZipFile(self.path) if self.path.is_file() else None
        self.members = sorted(self.zip.namelist() if self.zip else
                              [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()])
        if len(self.members) != len(set(self.members)):
            raise ValueError("Ambiguous duplicate ZIP member paths")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in self.members):
            raise ValueError("Unsafe dataset member path")
        yamls = [n for n in self.members if PurePosixPath(n).name in {"data.yaml", "dataset.yaml"}]
        if len(yamls) != 1:
            raise ValueError("Expected exactly one source data.yaml or dataset.yaml")
        self.yaml_member = yamls[0]
        self.prefix = str(PurePosixPath(yamls[0]).parent)
        self.prefix = "" if self.prefix == "." else self.prefix + "/"
        self.names = {n[len(self.prefix):]: n for n in self.members if n.startswith(self.prefix)}
        self.metadata = yaml.safe_load(self.read(yamls[0]))

    def read(self, member):
        return self.zip.read(member) if self.zip else (self.path / member).read_bytes()

    def identity(self):
        if self.zip:
            with self.path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        # Stable directory fingerprint covers names and bytes, not mtimes.
        h = hashlib.sha256()
        for name in self.members:
            h.update(name.encode() + b"\0" + bytes.fromhex(digest(self.read(name))))
        return h.hexdigest()

    def close(self):
        if self.zip:
            self.zip.close()


def parse_labels(text, nc):
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {number}: expected class xc yc width height, not polygon/classification")
        values = list(map(float, parts))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"line {number}: nonfinite coordinate")
        c, x, y, w, h = values
        if c != int(c) or not 0 <= c < nc:
            raise ValueError(f"line {number}: invalid class ID")
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"line {number}: coordinates outside normalized bounds")
        # Accommodate only decimal serialization rounding, never repair boxes.
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"line {number}: box extends outside image")
        boxes.append([int(c), x, y, w, h])
    if len({tuple(b) for b in boxes}) != len(boxes):
        raise ValueError("duplicate annotation rows")
    return boxes


def image_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()
    median = np.median(dct[1:])
    phash = sum(int(v > median) << i for i, v in enumerate(dct))
    small = cv2.resize(gray, (9, 8))
    dhash = sum(int(v) << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).flatten()))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return phash, dhash, sharpness


def source_id(filename):
    stem = Path(filename).stem.split(".rf.")[0]
    # Strip only explicit augmentation suffixes, not arbitrary trailing digits.
    return re.sub(r"[_-](?:aug(?:mented)?|flip|rotate|rot|brightness)[_-]?\d*$", "", stem, flags=re.I)


def group_records(records, phash_distance, dhash_distance, scene_edges=()):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    edges = []
    keys = {}
    for i, r in enumerate(records):
        for kind in ("source_id", "pixel_sha256", "image_sha256"):
            key = kind, r[kind]
            if key in keys:
                union(i, keys[key])
                edges.append({"a": records[keys[key]]["path"], "b": r["path"], "reason": kind,
                              "phash_distance": "", "dhash_distance": ""})
            else:
                keys[key] = i
    # Exhaustive comparisons; no approximate index can omit an eligible pair.
    for i, r in enumerate(records):
        for j in range(i):
            q = records[j]
            pd = (r["phash"] ^ q["phash"]).bit_count()
            if pd <= phash_distance:
                dd = (r["dhash"] ^ q["dhash"]).bit_count()
                if dd <= dhash_distance:
                    union(i, j)
                    edges.append({"a": q["path"], "b": r["path"], "reason": "perceptual",
                                  "phash_distance": pd, "dhash_distance": dd})
    by_path = {r["path"]: i for i, r in enumerate(records)}
    for edge in scene_edges:
        union(by_path[edge["a"]], by_path[edge["b"]])
        edges.append({"a": edge["a"], "b": edge["b"], "reason": "scene_geometry",
                      "phash_distance": "", "dhash_distance": ""})
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[find(i)].append(r)
    for members in groups.values():
        gid = digest("\n".join(sorted(r["path"] for r in members)).encode())[:20]
        for r in members:
            r["group_id"] = gid
    return list(groups.values()), edges


def candidate_pools(groups):
    pools = defaultdict(lambda: defaultdict(list))
    for members in groups:
        # Quarantine related training copies of source validation/test images.
        owner = "test" if any(r["split"] == "test" for r in members) else "development"
        # Conflicting exact-image classes are excluded, not silently reconciled.
        pixels = defaultdict(set)
        for r in members:
            pixels[r["pixel_sha256"]].add(tuple(sorted({b[0] for b in r["boxes"]})))
        if any(len(v) > 1 for v in pixels.values()):
            continue
        candidates = [r for r in members if (r["split"] == "test" if owner == "test" else r["split"] != "test") and not r["issues"]
                      and r["boxes"] and len({b[0] for b in r["boxes"]}) == 1 and not r["augmentation_marker"]]
        if candidates:
            # A single representative overall; no hidden class-specific double use.
            r = min(candidates, key=lambda r: r["path"])
            pools[r["boxes"][0][0]][owner].append(r)
    return pools


def scene_relationships(source, records, cfg):
    """Approximate full-source retrieval, followed by the SAME deterministic pair rules."""
    params = cfg["scene_matching"]
    unique = {}
    for r in sorted(records, key=lambda r: r["path"]):
        unique.setdefault(r["pixel_sha256"], r)
    unique = list(unique.values())
    features = extract_features(source, unique, cfg)
    all_descriptors, owners = [], []
    for i, r in enumerate(unique):
        desc = features[r["path"]][1]
        all_descriptors.extend(desc)
        owners.extend([i]*len(desc))
    if len(all_descriptors) < params["neighbors"]:
        return []
    all_descriptors = np.array(all_descriptors, np.float32)
    owners = np.array(owners)
    cv2.setRNGSeed(cfg["seed"])
    index = cv2.flann_Index(all_descriptors, dict(algorithm=1, trees=4))
    candidates = set()
    for i, r in enumerate(unique):
        desc = features[r["path"]][1]
        if len(desc) < params["min_inliers"]:
            continue
        idx, _ = index.knnSearch(desc, params["neighbors"], params={"checks": 32})
        votes = Counter(int(j) for j in owners[idx].flatten() if j != i)
        for j, count in votes.most_common(params["candidates_per_image"]):
            if count >= 8:
                candidates.add(tuple(sorted((i, j))))
    edges = []
    for i, j in sorted(candidates):
        edge = geometric_pair(unique[i], unique[j], features, cfg)
        if edge:
            edges.append(edge)
    return edges


def refine_selected_core(source, records, scene_edges, names, cfg):
    """Merge whole full-source components and reselect until exhaustive closure."""
    out = ROOT / cfg["audit_dir"]
    keyed = {tuple(sorted((e["a"], e["b"]))): e for e in scene_edges}
    rounds = []
    summary = {"classes_distribution": [{"class_name": n} for n in names]}
    for iteration in range(cfg.get("max_refinement_rounds", 50)):
        groups, _ = group_records(records, cfg["phash_distance"], cfg["dhash_distance"], list(keyed.values()))
        try:
            selected, _, _ = prepared_payload(cfg, records, summary)
        except ValueError as error:
            save_json(out / "refinement/blocker.json", {"status": "blocked", "round": iteration, "reason": str(error)})
            raise
        report = exhaustive_relationship_report(source, selected, cfg)
        save_json(out / f"refinement/round_{iteration:03d}.json", report)
        rounds.append({"round": iteration, "accepted_count": report["accepted_count"],
                       "cross_split_count": report["cross_split_count"], "pairs_checked": report["pairs_checked"],
                       "selected_inputs_sha256": report["selected_inputs_sha256"]})
        print(f"Exhaustive refinement {iteration}: {report['pairs_checked']} pairs; {report['accepted_count']} relationships", flush=True)
        if not report["accepted_count"]:
            save_json(out / "refinement/summary.json", {"status": "passed", "policy": POLICY_VERSION, "rounds": rounds})
            return list(keyed.values())
        for e in report["accepted_relationships"]:
            keyed[tuple(sorted((e["a"], e["b"])))] = e
    save_json(out / "refinement/blocker.json", {"status": "blocked", "reason": "Refinement limit reached", "rounds": rounds})
    raise ValueError("Unresolved selected-core relationships after refinement limit; no benchmark frozen")


def audit(cfg):
    assert_writable_revision(cfg)
    source = Source(ROOT / cfg["source"])
    try:
        meta = source.metadata
        rf = meta.get("roboflow", {})
        for field in ("workspace", "project", "version", "license"):
            if rf.get(field) != cfg["expected_" + field]:
                raise ValueError(f"Unexpected source {field}: {rf.get(field)}")
        names = meta["names"]
        if isinstance(names, dict):
            if sorted(names) != list(range(len(names))):
                raise ValueError("Source class IDs are not contiguous")
            names = [names[i] for i in range(len(names))]
        if len(names) != meta["nc"] or len(set(names)) != len(names):
            raise ValueError("Invalid source class names")
        records, issues = [], []
        image_members = [n for n in source.names if "/images/" in n and Path(n).suffix.lower() in IMAGE_SUFFIXES]
        matched_labels = set()
        for rel in sorted(image_members):
            member = source.names[rel]
            split = rel.split("/")[0]
            split = {"valid": "val", "validation": "val"}.get(split, split)
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown source split: {rel}")
            label_rel = str(PurePosixPath(rel.replace("/images/", "/labels/")).with_suffix(".txt"))
            b = source.read(member)
            record_issues = []
            image = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                issues.append({"path": member, "problem": "corrupt_image"})
                continue
            boxes, label_bytes = [], b""
            if label_rel not in source.names:
                record_issues.append("missing_label")
            else:
                matched_labels.add(label_rel)
                label_bytes = source.read(source.names[label_rel])
                try:
                    boxes = parse_labels(label_bytes.decode(), len(names))
                except (ValueError, UnicodeDecodeError) as error:
                    record_issues.append(f"invalid_label: {error}")
            ph, dh, sharpness = image_features(image)
            for problem in record_issues:
                issues.append({"path": member, "problem": problem})
            records.append({
                "path": member, "filename": Path(member).name, "label_path": source.names.get(label_rel),
                "source_id": source_id(member), "split": split, "boxes": boxes, "issues": record_issues,
                "image_sha256": digest(b), "label_sha256": digest(label_bytes),
                "pixel_sha256": digest(image.tobytes()), "phash": ph, "dhash": dh,
                "width": image.shape[1], "height": image.shape[0], "sharpness": sharpness,
                "max_box_area": max((box[3]*box[4] for box in boxes), default=0),
                "augmentation_marker": bool(re.search(r"(?:_|-)(aug|flip|rotat|brightness)", Path(member).stem, re.I))
            })
        orphan_labels = sorted(n for n in source.names if "/labels/" in n and n.endswith(".txt") and n not in matched_labels)
        issues.extend({"path": n, "problem": "orphan_label"} for n in orphan_labels)
        scene_edges = scene_relationships(source, records, cfg)
        supplemental = cfg.get("supplemental_relationships")
        if supplemental:
            report = json.loads((ROOT / supplemental).read_text())
            scene_edges.extend(report["accepted_relationships"])
        scene_edges = refine_selected_core(source, records, scene_edges, names, cfg)
        groups, edges = group_records(records, cfg["phash_distance"], cfg["dhash_distance"], scene_edges)
        pools = candidate_pools(groups)
        counts = Counter(b[0] for r in records for b in r["boxes"])
        image_counts = Counter(c for r in records for c in {b[0] for b in r["boxes"]})
        pixel_counts = Counter(r["pixel_sha256"] for r in records)
        class_rows = [{"source_class_id": c, "class_name": name, "images": image_counts[c], "boxes": counts[c],
                       **{f"eligible_{s}_groups": len(pools[c][s]) for s in ("development", "test")}}
                      for c, name in enumerate(names)]
        by_pixel = defaultdict(set)
        for r in records:
            by_pixel[r["pixel_sha256"]].add(tuple(sorted({b[0] for b in r["boxes"]})))
        readmes = {Path(n).name: source.read(n).decode() for n in source.members if "README" in Path(n).name}
        summary = {
            "relationship_policy": POLICY_VERSION,
            "supplemental_relationships_sha256": digest((ROOT / cfg["supplemental_relationships"]).read_bytes()) if cfg.get("supplemental_relationships") else None,
            "source_images": len(image_members), "decoded_images": len(records),
            "source_label_files": len(matched_labels) + len(orphan_labels),
            "boxes": sum(counts.values()), "classes": len(names),
            "source_split_counts": dict(Counter(r["split"] for r in records)),
            "issues": len(issues), "empty_labels": sum(not r["boxes"] and not r["issues"] for r in records),
            "exact_duplicate_groups": sum(n > 1 for n in pixel_counts.values()),
            "redundant_exact_images": sum(n-1 for n in pixel_counts.values()),
            "unique_decoded_images": len(pixel_counts), "related_groups": len(groups),
            "cross_source_split_groups": sum(len({r["split"] for r in g}) > 1 for g in groups),
            "perceptual_edges": sum(e["reason"] == "perceptual" for e in edges),
            "scene_geometry_edges": len(scene_edges),
            "conflicting_exact_class_groups": sum(len(v) > 1 for v in by_pixel.values()),
            "augmentation_filename_markers": sum(r["augmentation_marker"] for r in records),
            "largest_related_group": max(map(len, groups), default=0),
            "source_metadata": rf, "source_identity_sha256": source.identity(),
            "source_yaml_sha256": digest(source.read(source.yaml_member)), "source_readmes": readmes,
            "classes_distribution": class_rows,
            "limitations": ["Roboflow export is preprocessed, not camera-original files",
                            "Perceptual grouping is conservative and cannot certify unknown scene/video provenance",
                            "Mixed sign crops, illustrations and road scenes; no full-scene detection claim"]
        }
        out = ROOT / cfg["audit_dir"]
        save_json(out / "source_audit.json", summary)
        save_json(out / "source_records.json", records)
        save_json(out / "scene_relationships.json", scene_edges)
        save_csv(out / "class_distribution.csv", class_rows, list(class_rows[0]))
        save_csv(out / "source_issues.csv", issues, ["path", "problem"])
        save_csv(out / "duplicate_pairs.csv", edges, ["a", "b", "reason", "phash_distance", "dhash_distance"])
        print(json.dumps({k: v for k, v in summary.items() if k not in {"source_readmes", "classes_distribution"}}, indent=2))
        return records, summary
    finally:
        source.close()


def choose_classes(pools, names, cfg):
    k = cfg["class_count"]
    # Minimum support for balanced 210/45/45 allocation. Rank independently of model scores.
    needs = {"development": (cfg["split_counts"]["train"]+cfg["split_counts"]["val"])/k,
             "test": cfg["split_counts"]["test"]/k}
    ranking = sorted(pools, key=lambda c: (-min(len(pools[c][s])/needs[s] for s in needs), c))
    chosen = []
    for c in ranking:
        if len(pools[c]["development"]) < 30 or len(pools[c]["test"]) < math.ceil(needs["test"]):
            continue
        if names[c].startswith("SPEED_LIMIT_") and sum(names[i].startswith("SPEED_LIMIT_") for i in chosen) >= cfg["max_speed_limit_classes"]:
            continue
        chosen.append(c)
        if len(chosen) == k:
            return sorted(chosen)
    raise ValueError("Insufficient independent groups for requested size/classes; review audit, do not invent samples")


def balanced_allocation(total, capacities):
    counts = {c: 0 for c in capacities}
    while sum(counts.values()) < total:
        available = [c for c in capacities if counts[c] < capacities[c]]
        if not available:
            raise ValueError("Insufficient non-overlapping groups for split")
        c = min(available, key=lambda c: (counts[c], c))
        counts[c] += 1
    return counts


def stratified_sample(candidates, count, rng):
    # Rank-based strata retain relative framing and sharpness variation per class/split.
    buckets = defaultdict(list)
    n = len(candidates)
    area_rank = {r["path"]: i for i, r in enumerate(sorted(candidates, key=lambda r: (r["max_box_area"], r["path"])))}
    sharp_rank = {r["path"]: i for i, r in enumerate(sorted(candidates, key=lambda r: (r["sharpness"], r["path"])))}
    for r in sorted(candidates, key=lambda r: r["path"]):
        buckets[(min(3, 4*area_rank[r["path"]]//n), min(2, 3*sharp_rank[r["path"]]//n))].append(r)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(buckets)
    while len(selected) < count:
        rng.shuffle(keys)
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    return selected


def prepared_payload(cfg, records, summary):
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    pools = candidate_pools(list(groups.values()))
    names = [r["class_name"] for r in summary["classes_distribution"]]
    chosen = choose_classes(pools, names, cfg)
    mapping = {c: i for i, c in enumerate(chosen)}
    rng = random.Random(cfg["seed"])
    selected = []
    dev_counts = balanced_allocation(cfg["split_counts"]["train"]+cfg["split_counts"]["val"],
                                    {c: len(pools[c]["development"]) for c in chosen})
    test_counts = balanced_allocation(cfg["split_counts"]["test"], {c: len(pools[c]["test"]) for c in chosen})
    val_counts = balanced_allocation(cfg["split_counts"]["val"], dev_counts)
    for c in chosen:
        dev = stratified_sample(pools[c]["development"], dev_counts[c], rng)
        rng.shuffle(dev)
        for i, r in enumerate(dev):
            selected.append({**r, "assigned_split": "val" if i < val_counts[c] else "train"})
        for r in stratified_sample(pools[c]["test"], test_counts[c], rng):
            selected.append({**r, "assigned_split": "test"})
    return selected, mapping, names


def prepare(cfg):
    assert_writable_revision(cfg)
    out = ROOT / cfg["audit_dir"]
    records = json.loads((out / "source_records.json").read_text())
    summary = json.loads((out / "source_audit.json").read_text())
    selected, mapping, names = prepared_payload(cfg, records, summary)
    target = ROOT / cfg["output_dir"]
    source = Source(ROOT / cfg["source"])
    try:
        if source.identity() != summary["source_identity_sha256"]:
            raise ValueError("Raw source changed since audit")
        relationship_report = exhaustive_relationship_report(source, selected, cfg)
        save_json(out / "preparation_relationships.json", relationship_report)
        if relationship_report["accepted_count"]:
            raise ValueError("Unresolved accepted selected-core relationships; rerun the full versioned audit before freezing")
        manifest = []
        payload = {}
        for r in sorted(selected, key=lambda r: (r["assigned_split"], r["path"])):
            split = r["assigned_split"]
            name = r["source_id"] + "_" + r["image_sha256"][:12] + Path(r["path"]).suffix.lower()
            image_path = f"{split}/images/{name}"
            label_path = f"{split}/labels/{Path(name).stem}.txt"
            image = source.read(r["path"])
            if digest(image) != r["image_sha256"]:
                raise ValueError("Source image hash mismatch")
            # Preserve source coordinate tokens verbatim; change only the class token.
            lines = []
            for line in source.read(r["label_path"]).decode().splitlines():
                if line.strip():
                    tokens = line.split()
                    tokens[0] = str(mapping[int(float(tokens[0]))])
                    lines.append(" ".join(tokens))
            label = ("\n".join(lines) + "\n").encode()
            parse_labels(label.decode(), len(mapping))
            if image_path in payload:
                raise ValueError("Output filename collision")
            payload[image_path], payload[label_path] = image, label
            manifest.append({
                "original_filename": r["filename"], "source_path": r["path"], "source_label_path": r["label_path"],
                "source_id": r["source_id"], "group_id": r["group_id"], "source_split": r["split"], "split": split,
                "source_class_ids": ";".join(str(c) for c in sorted({b[0] for b in r["boxes"]})),
                "selected_classes": ";".join(names[c] for c in sorted({b[0] for b in r["boxes"]})),
                "remapped_class_ids": ";".join(str(mapping[c]) for c in sorted({b[0] for b in r["boxes"]})),
                "image_path": image_path, "label_path": label_path, "boxes": len(r["boxes"]),
                "image_sha256": r["image_sha256"], "source_label_sha256": r["label_sha256"],
                "label_sha256": digest(label), "max_box_area": r["max_box_area"], "sharpness": r["sharpness"]
            })
        # Omit 'path': with an absolute YAML filename Ultralytics resolves the
        # image paths relative to this YAML, independent of its global datasets_dir.
        dataset_yaml = {"train": "train/images", "val": "val/images", "test": "test/images",
                        "nc": len(mapping), "names": {v: names[k] for k, v in mapping.items()}}
        payload["dataset.yaml"] = yaml.safe_dump(dataset_yaml, sort_keys=False).encode()
        frozen = {
            "source_identity_sha256": summary["source_identity_sha256"],
            "config_sha256": digest(json.dumps(cfg, sort_keys=True).encode()),
            "class_mapping": {str(k): {"id": v, "name": names[k]} for k, v in mapping.items()},
            "files": {p: digest(b) for p, b in sorted(payload.items())}
        }
        lock_bytes = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
        if target.exists() and any(target.iterdir()):
            # Idempotency verifies, never overwrites an existing benchmark or test image.
            if not (target / "dataset.lock.json").exists() or (target / "dataset.lock.json").read_bytes() != lock_bytes:
                raise ValueError("Existing dataset differs. Preserve it; use a new configured output directory for a new protocol.")
            for p, h in frozen["files"].items():
                if not (target / p).is_file() or digest((target / p).read_bytes()) != h:
                    raise ValueError(f"Frozen file missing/changed: {p}; refusing to overwrite")
        else:
            for p, b in payload.items():
                path = target / p
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b)
            (target / "dataset.lock.json").write_bytes(lock_bytes)
        save_csv(out / "manifest.csv", manifest, list(manifest[0]))
        save_json(out / "selection.json", {"seed": cfg["seed"], "class_mapping": frozen["class_mapping"],
                  "selected_images": len(manifest), "split_counts": dict(Counter(r["split"] for r in manifest)),
                  "source_identity_sha256": summary["source_identity_sha256"],
                  "dataset_lock_sha256": digest(lock_bytes),
                  "manifest_sha256": digest((out / "manifest.csv").read_bytes()),
                  "preparation_relationships_sha256": digest((out / "preparation_relationships.json").read_bytes()),
                  "source_records_sha256": digest((out / "source_records.json").read_bytes()), "config": cfg})
        return validate_prepared(cfg)
    finally:
        source.close()


def validate_prepared(cfg, report_dir=None):
    target = ROOT / cfg["output_dir"]
    lock_bytes = (target / "dataset.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    out = ROOT / cfg["audit_dir"]
    selection = json.loads((out / "selection.json").read_text())
    if digest(lock_bytes) != selection["dataset_lock_sha256"]:
        raise ValueError("Dataset lock changed since selection")
    if digest((out / "manifest.csv").read_bytes()) != selection["manifest_sha256"]:
        raise ValueError("Manifest changed since selection")
    if digest((out / "source_records.json").read_bytes()) != selection["source_records_sha256"]:
        raise ValueError("Source relationship inventory changed since selection")
    if "preparation_relationships_sha256" in selection and digest((out / "preparation_relationships.json").read_bytes()) != selection["preparation_relationships_sha256"]:
        raise ValueError("Preparation relationship evidence changed since selection")
    if lock["config_sha256"] != digest(json.dumps(cfg, sort_keys=True).encode()):
        raise ValueError("Dataset config changed since freeze")
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    records = {r["path"]: r for r in json.loads((out / "source_records.json").read_text())}
    group_splits = defaultdict(set)
    for record in records.values():
        group_splits[record["group_id"]].add(record["split"])
    meta = yaml.safe_load((target / "dataset.yaml").read_text())
    mapping = {int(k): v["id"] for k, v in lock["class_mapping"].items()}
    if sorted(meta["names"]) != list(range(len(mapping))):
        raise ValueError("Noncontiguous remapped classes")
    if meta["names"] != {v["id"]: v["name"] for v in lock["class_mapping"].values()}:
        raise ValueError("Class names differ from frozen mapping")
    if dict(Counter(r["split"] for r in rows)) != cfg["split_counts"] or len(rows) != 300:
        raise ValueError("Incorrect split cardinalities")
    for key in ("group_id", "source_id", "image_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Repeated {key} in benchmark")
    files = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    if files != set(lock["files"]) | {"dataset.lock.json"}:
        raise ValueError("Unexpected/missing prepared files")
    for p, h in lock["files"].items():
        if digest((target / p).read_bytes()) != h:
            raise ValueError(f"Frozen file changed: {p}")
    classes = Counter()
    per_split = defaultdict(Counter)
    for row in rows:
        r = records[row["source_path"]]
        for field in ("group_id", "source_id", "image_sha256"):
            if row[field] != r[field]:
                raise ValueError(f"Manifest provenance mismatch: {field}")
        if (row["split"] == "test") != (r["split"] == "test"):
            raise ValueError("Source test boundary violated")
        if row["split"] != "test" and "test" in group_splits[r["group_id"]]:
            raise ValueError("A source-test relative entered development")
        if digest((target / row["image_path"]).read_bytes()) != r["image_sha256"]:
            raise ValueError("Image bytes differ from source")
        image = cv2.imdecode(np.frombuffer((target / row["image_path"]).read_bytes(), np.uint8), 1)
        if image is None:
            raise ValueError("Corrupt prepared image")
        boxes = parse_labels((target / row["label_path"]).read_text(), len(mapping))
        expected = [[mapping[b[0]], *b[1:]] for b in r["boxes"]]
        if boxes != expected:
            raise ValueError("Annotation coordinates changed or labels dropped")
        for c in {b[0] for b in boxes}:
            classes[meta["names"][c]] += 1
            per_split[row["split"]][meta["names"][c]] += 1
    class PreparedReader:
        def read(self, member):
            return (target / image_paths[member]).read_bytes()

    image_paths = {row["source_path"]: row["image_path"] for row in rows}
    selected = [{**records[row["source_path"]], "assigned_split": row["split"]} for row in rows]
    # Actual prepared image bytes, no stored candidate list or group-ID trust.
    relationship_report = exhaustive_relationship_report(PreparedReader(), selected, cfg)
    destination = (Path(report_dir) if report_dir is not None else
                   ROOT / cfg.get("validation_dir", "results/phase1_repair/legacy_validation"))
    save_json(destination / "selected_relationships.json", relationship_report)
    if relationship_report["accepted_count"]:
        save_json(destination / "validation.json", {"status": "failed", "reason": "Unresolved accepted relationships",
                  "accepted_count": relationship_report["accepted_count"], "cross_split_count": relationship_report["cross_split_count"]})
        raise ValueError("Unresolved accepted selected-core relationships; frozen validation failed")
    result = {"status": "passed", "selected_images": len(rows), "split_counts": cfg["split_counts"],
              "per_class_images": dict(classes), "per_split_class_images": dict(per_split),
              "selected_boxes": sum(int(r["boxes"]) for r in rows),
              "duplicate_or_related_group_leakage": 0, "unchanged_source_image_bytes": True,
              "relationship_policy": POLICY_VERSION, "exhaustive_pairs_checked": relationship_report["pairs_checked"],
              "relationship_statement": relationship_report["statement"], "physical_sign_independence_guaranteed": False,
              "label_geometry_unchanged": True, "dataset_lock_sha256": digest(lock_bytes),
              "test_set_policy": "frozen source-test subset; hashes checked; no model development/evaluation performed",
              "training_executed": False, "degradations_generated": False}
    save_json(destination / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result
