#!/usr/bin/env python3
"""Historical four-class GTSRB crop-recognition experiment.

The retained run measured crop classification for GTSRB IDs 1, 5, 17
and 38. See docs/gtsrb_reference_v1/README.md for the recorded result
and the frame-index grouping defect in this historical implementation.
The published metrics file makes this script refuse to rerun in this checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/gtsrb_official"
OUT = ROOT / "docs/gtsrb_reference_v1"
IDS = [1, 5, 17, 38]
NAMES = {1: "SPEED_LIMIT_30", 5: "SPEED_LIMIT_80", 17: "NO_ENTRY", 38: "COMPULSORY_KEEP_RIGHT"}
EXCLUDED = {"00001/00013", "00001/00066"}  # Intended exclusions. See the retained run protocol.
SIZE = 48


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(buf: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(buf)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_train():
    items = {}  # track -> (class_idx, [arrays])
    with zipfile.ZipFile(RAW / "GTSRB_Final_Training_Images.zip") as z:
        for n in sorted(z.namelist()):
            if not n.endswith(".ppm"):
                continue
            parts = n.split("/")
            cls, fname = int(parts[-2]), parts[-1]
            if cls not in IDS:
                continue
            track = f"{parts[-2]}/{fname.split('_')[1]}"
            if track in EXCLUDED:
                continue
            items.setdefault(track, (IDS.index(cls), []))[1].append(load(z.read(n)))
    return items


def load_test():
    with zipfile.ZipFile(RAW / "GTSRB_Final_Test_GT.zip") as z:
        rows = list(csv.DictReader(io.StringIO(z.read("GT-final_test.csv").decode()), delimiter=";"))
    keep = [(r["Filename"], IDS.index(int(r["ClassId"]))) for r in rows if int(r["ClassId"]) in IDS]
    X, y = [], []
    with zipfile.ZipFile(RAW / "GTSRB_Final_Test_Images.zip") as z:
        for fn, c in keep:
            X.append(load(z.read(f"GTSRB/Final_Test/Images/{fn}")))
            y.append(c)
    return np.stack(X), np.array(y)


def split_tracks(items, seed):
    rng = random.Random(seed)
    tr, va = [], []
    for c in range(len(IDS)):
        tracks = sorted(t for t, (ci, _) in items.items() if ci == c)
        rng.shuffle(tracks)
        k = max(1, round(0.2 * len(tracks)))
        va += tracks[:k]
        tr += tracks[k:]
    assert not set(tr) & set(va)
    return sorted(tr), sorted(va)


def stack(items, tracks):
    X = np.concatenate([np.stack(items[t][1]) for t in tracks])
    y = np.concatenate([[items[t][0]] * len(items[t][1]) for t in tracks])
    return X, y


def net():
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, len(IDS)))


def tens(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().float().div(255.0)


@torch.no_grad()
def predict(m, X, dev):
    m.eval()
    out = []
    for i in range(0, len(X), 512):
        out.append(m(tens(X[i:i + 512]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def confusion(y, p):
    cm = np.zeros((len(IDS), len(IDS)), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    receipt = OUT / "freeze_receipt.json"
    if (OUT / "test_metrics.json").exists():
        sys.exit("test already evaluated once; refusing to rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    items = load_train()
    tr_t, va_t = split_tracks(items, a.seed)
    Xtr, ytr = stack(items, tr_t)
    Xva, yva = stack(items, va_t)
    print(f"train {len(Xtr)} imgs/{len(tr_t)} tracks, val {len(Xva)} imgs/{len(va_t)} tracks, device {dev}", flush=True)

    m = net().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(a.seed)
    best, best_state, log = -1.0, None, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g).numpy()
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = tens(Xtr[idx]).to(dev)
            yb = torch.from_numpy(ytr[idx]).long().to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        acc = float((predict(m, Xva, dev) == yva).mean())
        log.append({"epoch": ep, "val_acc": acc})
        print(f"epoch {ep} val_acc {acc:.4f}", flush=True)
        if acc > best:
            best, best_state, best_ep = acc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep
    m.load_state_dict(best_state)
    wpath = OUT / "reference_cnn.pt"
    torch.save(best_state, wpath)
    receipt.write_text(json.dumps({
        "frozen_before_test": True, "selected_epoch": best_ep, "val_acc": best, "seed": a.seed,
        "weights_sha256": sha(wpath), "train_tracks": len(tr_t), "val_tracks": len(va_t),
        "train_images": int(len(Xtr)), "val_images": int(len(Xva)),
        "excluded_tracks": sorted(EXCLUDED), "epoch_log": log, "train_seconds": round(time.time() - t0, 1),
        "source_sha256": {p.name: sha(p) for p in sorted(RAW.glob("*.zip"))},
    }, indent=2))

    Xte, yte = load_test()
    pte = predict(m, Xte, dev)
    cm = confusion(yte, pte)
    (OUT / "test_metrics.json").write_text(json.dumps({
        "task": "four-class GTSRB crop recognition (not detection; not mAP)",
        "classes": [NAMES[i] for i in IDS], "gtsrb_ids": IDS, "test_images": int(len(yte)),
        "accuracy": float((pte == yte).mean()),
        "per_class_recall": {NAMES[IDS[i]]: float(cm[i, i] / cm[i].sum()) for i in range(len(IDS))},
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "note": "Official test set has no track IDs; independence from Indian data is not certified.",
    }, indent=2))
    print(f"TEST accuracy {float((pte == yte).mean()):.4f}  n={len(yte)}\n{cm}")


if __name__ == "__main__":
    main()
