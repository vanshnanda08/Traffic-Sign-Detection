"""Calculate paired uncertainty from stored W6 records, with zero model inference."""
import csv
import hashlib

import numpy as np

from verify_v3_preservation import ROOT, read, sha, verify_anchors
from p57.durability import atomic_json
from p57.evaluation_engine import recompute
from p57.evaluation_protocol import library_sources
from p57.stats_v3 import SourceAP50, bootstrap_deltas, box_area_fraction


def main():
    verify_anchors()
    # W8A replaced the root launcher/test interface. Its archived originals still
    # satisfy all W4 receipt pins; the release-mode evaluator is unchanged.
    aliases = {"run_all.sh": "scripts/historical/run_all_pre_w8a.sh",
               "tests/test_runner.py": "scripts/historical/test_runner_pre_w8a.py"}
    parent = read(ROOT / "docs/protocol_w4_v1/freeze_receipt.json")
    for name, expected in parent["pinned_files"].items():
        if sha(ROOT / aliases.get(name, name)) != expected:
            raise ValueError(f"Historical W4 receipt pin changed: {name}")
    receipt = read(ROOT / "docs/evaluation_v1/evaluator_receipt.json")
    library = {key: hashlib.sha256(source.encode()).hexdigest()
               for key, source in library_sources().items()}
    if library != receipt["library_hashes"]:
        raise ValueError("Pinned W6 evaluator library changed")
    config = read(ROOT / "configs/main_ac_v3.json")["bootstrap"]
    measured = ROOT / "docs/evaluation_v1/measured"
    canonical = {(r["condition"], r["model"]): r for r in read(measured / "metrics_14.json")}
    paired = {r["condition"]: r for r in read(measured / "paired_table.json")}
    output = ROOT / "docs/evaluation_v3_stats"
    names = {row["id"]: row["name"] for row in canonical["clean", "A"]["per_class"]}
    classes = sorted(names)
    draws = np.random.default_rng(config["seed"]).multinomial(
        45, [1 / 45] * 45, size=config["resamples"]
    )
    results, per_class, buckets, input_hashes = [], [], [], {}
    original_order = None
    all_bootstraps = []
    for condition, historical in paired.items():
        matrices = {}
        for model in ("A", "C"):
            summary = canonical[condition, model]
            path = ROOT / f"results/detector_eval_v1/runs/{model}_{condition}/predictions.json"
            actual = sha(path)
            input_hashes[str(path.relative_to(ROOT))] = actual
            prediction = read(path)
            if (actual != summary["predictions_sha256"] or not prediction["complete"]
                    or prediction["scope"] != "heldout" or prediction["condition"] != condition
                    or prediction["checkpoint_sha256"] != summary["checkpoint_sha256"]):
                raise ValueError(f"W6 prediction identity check failed: {model}/{condition}")
            rows = prediction["records"]
            order = [r["source_id"] for r in rows]
            if original_order is None:
                original_order = order
            if len(set(order)) != 45 or order != original_order:
                raise ValueError("Expected all 45 sources in original evaluator order")
            detail = recompute(rows, names, 1e-10, detailed=True)
            score = detail["aggregate"]["mAP50"]
            if score != summary["mAP50"]:
                raise ValueError(f"W6 point estimate difference: {score!r} versus {summary['mAP50']!r}")
            matrix = SourceAP50(rows, classes)
            matrices[model] = matrix
            class_scores = matrix.class_scores(np.ones(45, dtype=int))
            if score != matrix.mean(np.ones(45, dtype=int)):
                raise ValueError("Bootstrap equally weighted point differs from pinned AP50")
            for row in detail["per_class"]:
                if row["mAP50"] != class_scores[row["id"]]:
                    raise ValueError("Per-class AP50 differs from pinned metric")
                per_class.append({"condition": condition, "model": model,
                                  "class_id": row["id"], "class_name": names[row["id"]],
                                  **{k: row[k] for k in ("mAP50", "precision", "recall", "mAP50_95")},
                                  "support": int(matrix.targets[:, classes.index(row["id"])].sum())})
        a, c = matrices["A"], matrices["C"]
        for left, right in zip(a.records, c.records):
            if left["ground_truth"] != right["ground_truth"] or left["image_sha256"] != right["image_sha256"]:
                raise ValueError("Stored paired ground truth or pixels differ")
        delta = canonical[condition, "C"]["mAP50"] - canonical[condition, "A"]["mAP50"]
        if delta != historical["delta_mAP50_fraction"]:
            raise ValueError("W6 delta differs")
        distribution = bootstrap_deltas(a, c, draws)
        all_bootstraps.append(distribution)
        low, high = np.quantile(distribution, config["interval"])
        results.append({"condition": condition, "A_mAP50": historical["A_mAP50"],
                        "C_mAP50": historical["C_mAP50"], "delta": delta,
                        "ci_low": float(low), "ci_high": float(high)})
        areas = np.array([box_area_fraction(r) for r in a.records])
        lower, upper = config["area_bucket_edges"]
        for bucket, mask in (("small", areas < lower),
                             ("medium", (areas >= lower) & (areas < upper)),
                             ("large", areas >= upper)):
            weights = mask.astype(int)
            scores = {model: m.mean(weights) for model, m in matrices.items()}
            buckets.append({"condition": condition, "bucket": bucket, "sources": int(mask.sum()),
                            "classes_present": int(np.count_nonzero(weights @ a.targets)), **scores,
                            "delta": scores["C"] - scores["A"] if mask.any() else None})
        print(f"{condition}: point exact; delta {delta*100:+.3f} pp, 95% CI [{low*100:+.3f}, {high*100:+.3f}] pp", flush=True)

    atomic_json(output / "stats.json", {
        "config": config, "runtime_scope": "Stored predictions only; no inference",
        "config_sha256": sha(ROOT / "configs/main_ac_v3.json"),
        "source_order": original_order, "predictions_sha256": input_hashes,
        "w6_metrics_sha256": sha(measured / "metrics_14.json"),
        "w6_points_exact": True, "paired": results, "per_class": per_class, "area_buckets": buckets,
        "area_profile": {"median": float(np.median(areas)),
                         "at_least_50_percent": int(np.count_nonzero(areas >= 0.50)),
                         "at_least_80_percent": int(np.count_nonzero(areas >= 0.80))},
        "bootstrap_missing_class_replicates": int(np.count_nonzero(np.any(draws @ a.targets == 0, axis=1))),
    })
    np.savez_compressed(output / "bootstrap_draws.npz", samples=draws,
                        deltas=np.array(all_bootstraps), conditions=list(paired))
    for name, rows in (("per_class", per_class), ("area_buckets", buckets), ("paired_ci", results)):
        with (output / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
