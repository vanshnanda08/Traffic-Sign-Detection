"""Generate development-only examples without modifying frozen source files."""

import json
from pathlib import Path

import cv2
import numpy as np

from p57.assets import file_sha256
from p57.dataset import ROOT, parse_labels, save_json
from p57.development import build_development_view, development_records, require_development_row
from p57.degradation import degrade


def main():
    view = build_development_view().parent
    cfg = json.loads((ROOT / "configs/degradation_dev_v1.json").read_text())
    records = development_records()
    chosen = []
    for cls in range(6):
        chosen.append(next(r for r in records if int(r["remapped_class_ids"]) == cls))
    multi = next(r for r in records if int(r["boxes"]) > 1)
    if multi not in chosen:
        chosen.append(multi)
    for split in ("train", "val"):
        if not any(r["split"] == split for r in chosen):
            chosen.append(next(r for r in records if r["split"] == split))
    out = ROOT / "results/development_w2_v1/examples"
    if out.exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['config']!=cfg:
            raise ValueError("Existing development protocol differs; create a new version")
        by_id={r['source_id']:r for r in records}
        for entry in manifest['examples']:
            row=by_id[entry['source_id']]
            require_development_row(row)
            if file_sha256(view/row['image_path'])!=entry['source_sha256'] or file_sha256(out/f"{entry['source_id']}_{entry['condition']}.png")!=entry['output_sha256']:
                raise ValueError("Existing source/example bytes changed")
        print(f"Validated {len(manifest['examples'])} existing development examples; no overwrite")
        return
    out.mkdir(parents=True)
    sheets, evidence = [], []
    for row in chosen:
        require_development_row(row)
        source = view / row["image_path"]
        image = cv2.imread(str(source))
        boxes = parse_labels((view / row["label_path"]).read_text(),6)
        cells = []
        for condition in ("clean","F1","F2","F3","O1","O2","O3"):
            derived, _, record = (image.copy(), None, {"condition":"clean"}) if condition == "clean" else degrade(image,boxes,row["image_sha256"],condition,cfg)
            path = out / f"{row['source_id']}_{condition}.png"
            cv2.imwrite(str(path),derived)
            evidence.append({"source_id":row["source_id"],"split":row["split"],"boxes":boxes,"source_sha256":file_sha256(source),"output_sha256":file_sha256(path),**record})
            cell = np.full((205,180,3),255,np.uint8)
            cell[:180] = cv2.resize(derived,(180,180))
            cv2.putText(cell,condition,(5,198),0,.5,(0,0,0),1)
            cells.append(cell)
        sheet = np.vstack([np.full((25,1260,3),255,np.uint8),np.hstack(cells)])
        cv2.putText(sheet,f"{row['split']} {row['source_id']} {row['selected_classes']} ({len(boxes)} boxes)",(5,18),0,.42,(0,0,0),1)
        sheets.append(sheet)
    cv2.imwrite(str(out/'development_contact_sheet.jpg'),np.vstack(sheets))
    save_json(out/'manifest.json',{"scope":"development only; no final-test variants","config":cfg,"examples":evidence})
    print(f"Generated {len(evidence)} development examples, including multi-box input; no test data")


if __name__ == "__main__":
    main()
