"""Generate labeled review sheets only; never change benchmark image/label bytes."""

import csv
from pathlib import Path

import cv2
import numpy as np

from p57.dataset import ROOT, config


def main():
    cfg = config()
    out = ROOT / cfg["audit_dir"] / "review"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / cfg["audit_dir"] / "manifest.csv").open()))
    classes = sorted({r["selected_classes"] for r in rows})
    for c in classes:
        selected = [r for r in rows if r["selected_classes"] == c]
        thumbs = []
        for i, row in enumerate(selected):
            image = cv2.imread(str(ROOT / cfg["output_dir"] / row["image_path"]))
            h, w = image.shape[:2]
            text = (ROOT / cfg["output_dir"] / row["label_path"]).read_text()
            for line in text.splitlines():
                _, x, y, bw, bh = map(float, line.split())
                cv2.rectangle(image, (int((x-bw/2)*w), int((y-bh/2)*h)),
                              (int((x+bw/2)*w), int((y+bh/2)*h)), (0, 255, 0), 3)
            thumb = np.full((162, 180, 3), 255, np.uint8)
            thumb[:140] = cv2.resize(image, (180, 140))
            cv2.putText(thumb, f"{i+1:02d} {row['split']} {row['source_id'][:16]}", (2, 155), 0, .36, (0, 0, 0), 1)
            thumbs.append(thumb)
        while len(thumbs) % 5:
            thumbs.append(np.full((162, 180, 3), 255, np.uint8))
        canvas = np.vstack([np.hstack(thumbs[i:i+5]) for i in range(0, len(thumbs), 5)])
        cv2.imwrite(str(out / f"{c}.jpg"), canvas)
    print(f"Wrote {len(classes)} labeled contact sheets to {out}")


if __name__ == "__main__":
    main()
