"""Box-based fading/occlusion with stable seeds and explicit overlap behavior."""

import hashlib
import json
import math

import numpy as np


def stable_seed(seed, image_id, condition):
    value = json.dumps([seed, image_id, condition], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def box_regions(boxes, width, height):
    regions = []
    for _, x, y, w, h in boxes:
        x0, y0 = max(0, math.floor((x-w/2)*width)), max(0, math.floor((y-h/2)*height))
        x1, y1 = min(width, math.ceil((x+w/2)*width)), min(height, math.ceil((y+h/2)*height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Empty box region")
        regions.append((x0, y0, x1, y1))
    return sorted(set(regions))


def degrade(image, boxes, image_id, condition, cfg):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image")
    if condition not in {*cfg["fading_alpha"], *cfg["occlusion_fraction"]}:
        raise ValueError("Unknown degradation condition")
    height, width = image.shape[:2]
    regions = box_regions(boxes, width, height)
    union = np.zeros((height, width), bool)
    mask = np.zeros_like(union)
    for x0, y0, x1, y1 in regions:
        union[y0:y1, x0:x1] = True
        if condition.startswith("F"):
            mask[y0:y1, x0:x1] = True
        else:
            fraction = cfg["occlusion_fraction"][condition]
            if not 0 < fraction < 1:
                raise ValueError("Occlusion fraction outside (0,1)")
            rw, rh = min(x1-x0, math.ceil((x1-x0)*math.sqrt(fraction))), min(y1-y0, math.ceil((y1-y0)*math.sqrt(fraction)))
            anchor = stable_seed(cfg["seed"], image_id, ["occlusion_anchor", x0, y0, x1, y1]) % 4
            left = x0 if anchor % 2 == 0 else x1-rw
            top = y0 if anchor < 2 else y1-rh
            mask[top:top+rh, left:left+rw] = True
    result = image.copy()
    if condition.startswith("F"):
        alpha = cfg["fading_alpha"][condition]
        if not 0 < alpha < 1:
            raise ValueError("Fading alpha outside (0,1)")
        target = np.array(cfg["fading_target_bgr"], np.float32)
        result[mask] = np.rint((1-alpha)*image[mask].astype(np.float32)+alpha*target).clip(0,255).astype(np.uint8)
    else:
        result[mask] = np.array(cfg["occlusion_fill_bgr"], np.uint8)
    record = {"condition": condition, "condition_seed": stable_seed(cfg["seed"], image_id, condition),
              "bbox_union_pixels": int(union.sum()), "affected_region_pixels": int(mask.sum()),
              "affected_union_fraction": float(mask.sum()/max(1,union.sum())),
              "per_box_coverage": [float(mask[y0:y1,x0:x1].mean()) for x0,y0,x1,y1 in regions],
              "overlap_policy": "union; transform each pixel once", "segmentation_claim": False}
    return result, mask, record
