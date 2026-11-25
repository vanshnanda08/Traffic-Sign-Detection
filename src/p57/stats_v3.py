"""Source-paired bootstrap of W6 AP50, using frozen per-image match records."""
import numpy as np
from ultralytics.utils.metrics import compute_ap


class SourceAP50:
    def __init__(self, records, class_ids):
        self.records = records
        self.class_ids = tuple(class_ids)
        self.targets = np.array([
            [row["ground_truth"]["classes"].count(class_id) for class_id in class_ids]
            for row in records
        ])
        confidence = np.concatenate([row["confidence"] for row in records]).astype(np.float32)
        predictions = np.concatenate([row["classes"] for row in records])
        sources = np.repeat(np.arange(len(records)), [len(row["confidence"]) for row in records])
        matching = np.concatenate([
            np.asarray(row["matching"]["stats"]["tp"], dtype=bool).reshape(-1, 10)[:, 0]
            for row in records
        ])
        order = np.argsort(-confidence)
        self.by_class = {
            class_id: (sources[indices], matching[indices].astype(np.int64))
            for class_id in class_ids
            for indices in [order[predictions[order] == class_id]]
        }

    def class_scores(self, weights):
        scores = {}
        for class_id, count in zip(self.class_ids, weights @ self.targets):
            if not count:
                continue
            sources, tp = self.by_class[class_id]
            counts = weights[sources]
            repeated = np.repeat(tp, counts)
            if not repeated.size:
                scores[class_id] = 0.0
                continue
            true = repeated.cumsum()
            scores[class_id] = float(compute_ap(
                true / count, true / np.arange(1, repeated.size + 1)
            )[0])
        return scores

    def mean(self, weights):
        scores = self.class_scores(weights)
        return float(np.mean(list(scores.values()))) if scores else None


def bootstrap_deltas(baseline, degraded, samples):
    """Reuse each source multiplicity for both arms and every treatment."""
    if [r["source_id"] for r in baseline.records] != [r["source_id"] for r in degraded.records]:
        raise ValueError("Bootstrap requires identical source order")
    if baseline.class_ids != degraded.class_ids or not np.array_equal(baseline.targets, degraded.targets):
        raise ValueError("Bootstrap requires identical paired ground-truth membership")
    return np.array([degraded.mean(weights) - baseline.mean(weights) for weights in samples])


def box_area_fraction(record):
    box, = record["ground_truth"]["xyxy"]
    height, width = record["shape"]
    return (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
