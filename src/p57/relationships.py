"""Deterministic image relationship checks; matches do not establish sign identity."""

import hashlib
import json

import cv2
import numpy as np

POLICY_VERSION = "selected-core-exhaustive-bidirectional-v2"


def extract_features(source, records, cfg):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(cfg["seed"])
    params = cfg["scene_matching"]
    size = params["image_size"]
    sift = cv2.SIFT_create(nfeatures=params["features"], contrastThreshold=.025)
    features = {}
    for record in sorted(records, key=lambda r: r["path"]):
        image = cv2.imdecode(np.frombuffer(source.read(record["path"]), np.uint8), 0)
        if image is None:
            raise ValueError(f"Corrupt relationship input: {record['path']}")
        keypoints, desc = sift.detectAndCompute(cv2.resize(image, (size, size)), None)
        if desc is None:
            desc = np.empty((0, 128), np.float32)
        points = np.array([kp.pt for kp in keypoints], np.float32).reshape(-1, 2)
        features[record["path"]] = points, desc
    return features


def _outside(points, record, size):
    outside = np.ones(len(points), dtype=bool)
    for _, x, y, w, h in record["boxes"]:
        outside &= ~((points[:, 0] >= (x-w/2)*size) & (points[:, 0] <= (x+w/2)*size) &
                     (points[:, 1] >= (y-h/2)*size) & (points[:, 1] <= (y+h/2)*size))
    return outside


def _coverage(points, mask, size):
    if int(mask.sum()) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points[mask]))/(size*size))


def measure_direction(a, b, features, cfg):
    """Same numerical acceptance rules as v1, with a reset seed for EACH RANSAC."""
    params = cfg["scene_matching"]
    size = params["image_size"]
    pa, da = features[a["path"]]
    pb, db = features[b["path"]]
    result = {"query": a["path"], "target": b["path"], "accepted": False,
              "ratio_matches": 0, "inliers": 0, "inlier_fraction": 0.0, "coverage": 0.0,
              "ransac_seed": cfg["seed"]}
    if min(len(da), len(db)) < params["min_inliers"]:
        return {**result, "rejection": "insufficient_descriptors"}
    different_classes = {x[0] for x in a["boxes"]} != {x[0] for x in b["boxes"]}
    # This is a necessary acceptance condition, not approximate candidate pruning.
    # Each query descriptor contributes at most one ratio-test match, so fewer
    # than ten outside QUERY keypoints cannot supply ten outside inliers. Do not
    # apply this bound to target keypoints: many queries can match one target.
    # Every selected pair and both directions still enter this function.
    if different_classes and int(_outside(pa, a, size).sum()) < params["min_inliers"]:
        return {**result, "rejection": "insufficient_outside_box_keypoints"}
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < params["ratio"]*n.distance]
    result["ratio_matches"] = len(good)
    if len(good) < params["min_inliers"]:
        return {**result, "rejection": "insufficient_ratio_matches"}
    qa = np.float32([pa[m.queryIdx] for m in good])
    qb = np.float32([pb[m.trainIdx] for m in good])
    cv2.setRNGSeed(cfg["seed"])
    _, mask = cv2.findHomography(qa, qb, cv2.RANSAC, 3.0)
    if mask is None:
        return {**result, "rejection": "no_homography"}
    use = mask.ravel().astype(bool)
    result.update(inliers=int(use.sum()), inlier_fraction=float(use.sum()/len(good)),
                  coverage=min(_coverage(p, use, size) for p in (qa, qb)))
    if result["inliers"] < params["min_inliers"] or result["inlier_fraction"] < params["min_inlier_fraction"] or result["coverage"] < params["min_coverage"]:
        return {**result, "rejection": "inlier_or_coverage_threshold"}
    if different_classes:
        outside = use & _outside(qa, a, size) & _outside(qb, b, size)
        result["outside_inliers"] = int(outside.sum())
        result["outside_coverage"] = min(_coverage(p, outside, size) for p in (qa, qb))
        if result["outside_inliers"] < params["min_inliers"] or result["outside_coverage"] < params["min_coverage"]:
            return {**result, "rejection": "outside_box_threshold"}
    return {**result, "accepted": True}


def geometric_pair(a, b, features, cfg):
    """Accept either directed check, each using unchanged thresholds, in stable order."""
    a, b = sorted((a, b), key=lambda r: r["path"])
    measurements = [measure_direction(a, b, features, cfg), measure_direction(b, a, features, cfg)]
    accepted = [m for m in measurements if m["accepted"]]
    if not accepted:
        return None
    first = accepted[0]
    return {"a": a["path"], "b": b["path"],
            "inliers": first["inliers"], "inlier_fraction": first["inlier_fraction"],
            "coverage": first.get("outside_coverage", first["coverage"]),
            "directions": measurements}


def exhaustive_relationship_report(source, records, cfg):
    """Visit ALL n(n-1)/2 pairs. Group IDs and approximate candidate lists are unused."""
    records = sorted(records, key=lambda r: r["path"])
    features = extract_features(source, records, cfg)
    accepted = []
    pairs_checked = 0
    for i, a in enumerate(records):
        for b in records[i+1:]:
            pairs_checked += 1
            reasons = [key for key in ("source_id", "image_sha256", "pixel_sha256") if a[key] == b[key]]
            pd = (a["phash"] ^ b["phash"]).bit_count()
            dd = (a["dhash"] ^ b["dhash"]).bit_count()
            if pd <= cfg["phash_distance"] and dd <= cfg["dhash_distance"]:
                reasons.append("perceptual")
            geometry = geometric_pair(a, b, features, cfg)
            if geometry:
                reasons.append("scene_geometry")
            if reasons:
                sa, sb = a.get("assigned_split", a["split"]), b.get("assigned_split", b["split"])
                accepted.append({**(geometry or {"a": a["path"], "b": b["path"]}),
                                 "source_id_a": a["source_id"], "source_id_b": b["source_id"],
                                 "split_a": sa, "split_b": sb, "cross_split": sa != sb,
                                 "reasons": reasons, "phash_distance": pd, "dhash_distance": dd,
                                 "disposition": "unresolved_requires_component_merge"})
    cross = sum(e["cross_split"] for e in accepted)
    return {"policy": POLICY_VERSION, "seed": cfg["seed"], "thresholds": cfg["scene_matching"],
            "selected_images": len(records), "pairs_checked": pairs_checked,
            "geometry_directions_checked": pairs_checked*2,
            "accepted_relationships": accepted, "accepted_count": len(accepted),
            "cross_split_count": cross,
            "status": "failed" if accepted else "passed",
            "statement": "No relationships detected under this protocol" if not accepted else "Unresolved accepted relationships detected",
            "physical_sign_independence_guaranteed": False,
            "approximate_candidate_pruning": False,
            "selected_inputs_sha256": hashlib.sha256(json.dumps(
                [(r["path"], r["image_sha256"], r.get("assigned_split", r["split"])) for r in records]
            ).encode()).hexdigest()}
