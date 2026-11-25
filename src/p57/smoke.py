"""Offline integration checks on synthetic inputs; no training or accuracy claims."""

from io import BytesIO
from datetime import datetime, timezone

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from p57.environment import load_config, seed_everything


def run_smoke(device):
    cfg = load_config()
    seed_everything(cfg["seed"], cfg["deterministic_algorithms"])
    size = cfg["smoke_image_size"]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), (0, 0, 255), -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image):
        raise RuntimeError("OpenCV PNG roundtrip failed")
    frame = pd.DataFrame({"value": [1, 2, 3]})
    if frame["value"].sum() != 6:
        raise RuntimeError("Pandas smoke check failed")
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    if not buffer.getvalue().startswith(b"\x89PNG"):
        raise RuntimeError("Matplotlib rendering failed")
    # Packaged architecture only: no .pt weights, dataset, or network download.
    model = YOLO(cfg["model_architecture"], task="detect").model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, size, size), device=device))
    predictions = output[0] if isinstance(output, tuple) else output
    if predictions.ndim != 3 or predictions.shape[0] != 1 or not torch.isfinite(predictions).all().item():
        raise RuntimeError("YOLOv8 forward check failed")
    return {
        "status": "passed", "scope": "synthetic infrastructure smoke check only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device, "seed": cfg["seed"],
        "checks": ["opencv_png_roundtrip", "pandas_sum", "matplotlib_png", "yolov8n_forward"],
        "model": cfg["model_architecture"], "weights": "random initialization",
        "prediction_shape": list(predictions.shape),
        "training_executed": False, "dataset_used": False,
        "accuracy_metrics": None
    }
