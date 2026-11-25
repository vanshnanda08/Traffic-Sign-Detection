"""Bounded, instrumented YOLOv8 feasibility. No final-test or main-run interface."""

import fcntl
import json
import math
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from p57.assets import file_sha256
from p57.dataset import ROOT, save_json
from p57.development import build_development_view, validate_development_view, verify_frozen_core, development_records
from p57.environment import seed_everything, validate_pins


def check_trial_config(config):
    cfg = config["training"]
    required = {"epochs":3,"imgsz":416,"device":"mps","workers":0,"cache":False,"amp":False,
                "seed":42,"deterministic":True,"patience":0,"resume":False,"fraction":1.0}
    if any(cfg.get(k) != v for k,v in required.items()) or cfg["batch"] not in {4,2,1}:
        raise ValueError("Trial configuration exceeds the authorized feasibility scope")
    for key in ("mosaic","mixup","copy_paste","hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective","flipud","fliplr"):
        if cfg[key] != 0:
            raise ValueError("Clean feasibility trial must retain the documented zero augmentation settings")
    return cfg


def memory_sample():
    record = {"rss_bytes":psutil.Process().memory_info().rss,"system_available_bytes":psutil.virtual_memory().available,
              "system_swap_used_bytes":psutil.swap_memory().used}
    if torch.backends.mps.is_available():
        record.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
    return record


def validate_detections(boxes, classes, confidences, shape, nc):
    if not (np.isfinite(boxes).all() and np.isfinite(classes).all() and np.isfinite(confidences).all()):
        raise ValueError("Nonfinite detection")
    if len(boxes):
        h,w = shape[:2]
        if (np.any(classes != classes.astype(int)) or np.any(classes < 0) or np.any(classes >= nc)
                or np.any(confidences < 0) or np.any(confidences > 1)
                or np.any(boxes[:,2:] < boxes[:,:2]) or np.any(boxes < -1e-3)
                or np.any(boxes[:,[0,2]] > w+1e-3) or np.any(boxes[:,[1,3]] > h+1e-3)):
            raise ValueError("Invalid class/confidence/box geometry")


def validation_gallery(run_dir, checkpoint, target_name="validation_gallery"):
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    summary = json.loads((run_dir/"provenance.json").read_text())
    if not checkpoint.is_file() or file_sha256(checkpoint) != summary["checkpoint_hashes"].get(checkpoint.name):
        raise ValueError("Missing, changed or unrecorded feasibility checkpoint")
    view = ROOT / "data/processed/p57_development_v1"
    data = validate_development_view(view/"dataset.yaml")
    rows = development_records()
    selected = [next(r for r in rows if r["split"] == "val" and int(r["remapped_class_ids"]) == c) for c in range(6)]
    out = run_dir / target_name
    if out.exists():
        raise ValueError("Gallery exists; use a new output name")
    out.mkdir()
    model = YOLO(str(checkpoint),task="detect")
    reports = []
    for row in selected:
        image_path = (view/row["image_path"]).resolve()
        if not image_path.is_relative_to((view/"val/images").resolve()) or image_path.is_symlink():
            raise ValueError("Only validation-gallery inputs are permitted")
        image = cv2.imread(str(image_path))
        prediction = model.predict(source=image,imgsz=416,device="mps",conf=.001,iou=.7,verbose=False,save=False)[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy()
        confidences = prediction.boxes.conf.cpu().numpy()
        validate_detections(boxes,classes,confidences,image.shape,data["nc"])
        # Raw smoke detections stay complete; a separate display threshold avoids
        # unreadable low-confidence overlays on tiny source crops.
        scale = 640 / max(image.shape[:2])
        shown = cv2.resize(image, None, fx=scale, fy=scale)
        visible = [v for v in sorted(zip(boxes,classes,confidences),key=lambda item:-float(item[2])) if float(v[2]) >= .25][:10]
        drawn = cv2.copyMakeBorder(shown,0,max(60,24*(len(visible)+1)),0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        for index,(box,c,score) in enumerate(visible):
            x0,y0,x1,y1 = (box*scale).astype(int)
            cv2.rectangle(drawn,(x0,y0),(x1,y1),(0,180,0),2)
            cv2.putText(drawn,str(index+1),(max(0,x0),max(18,y0+18)),0,.6,(0,150,0),2)
            cv2.putText(drawn,f"{index+1}: {data['names'][int(c)]} {score:.3f}",(8,shown.shape[0]+24*(index+1)),0,.5,(0,0,0),1)
        if not visible:
            cv2.putText(drawn,"No detection at display confidence >= 0.25",(8,shown.shape[0]+24),0,.5,(0,0,0),1)
        cv2.imwrite(str(out/f"{row['source_id']}.jpg"),drawn)
        reports.append({"source_id":row["source_id"],"split":"val","source_sha256":file_sha256(image_path),
                        "xyxy":boxes.tolist(),"classes":classes.tolist(),"confidence":confidences.tolist()})
    result = {"scope":"three-epoch feasibility validation inference only; not final test",
              "confidence_for_smoke":.001,"display_confidence":.25,"gallery_top_k":10,"checkpoint_sha256":file_sha256(checkpoint),
              "images":reports,"detections":sum(len(r["classes"]) for r in reports)}
    save_json(out/"predictions.json",result)
    return result


def run_trial(config_path, run_dir):
    start = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=False)
    cfg_doc = json.loads(Path(config_path).read_text())
    cfg = dict(check_trial_config(cfg_doc))
    state = {"status":"in_progress","scope":cfg_doc["scope"],"started_utc":datetime.now(timezone.utc).isoformat(),
             "requested_config":cfg_doc,"epochs":[],"optimizer_steps":[],"validation_passes":[],"memory_samples":[],
             "memory_method":"psutil process RSS/system available+swap and torch.mps tensor/driver allocation, sampled every second; observed maxima, not guaranteed peaks; unified-memory counters must not be summed"}
    save_json(run_dir/"provenance.json",state)
    stop = threading.Event()
    lock = None
    try:
        (ROOT/".cache").mkdir(exist_ok=True)
        lock = (ROOT/".cache/p57_training.lock").open("w")
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0") != "0":
            raise RuntimeError("Explicit MPS required; CPU fallback is forbidden")
        state["packages"] = validate_pins()
        seed_everything(42,True)
        torch.set_num_threads(2)
        yaml_path = build_development_view()
        if (ROOT/cfg["data"]).resolve() != yaml_path.resolve():
            raise ValueError("Trial must use the protected development-only dataset")
        validate_development_view(yaml_path)
        weights = ROOT/cfg["model"]
        receipt = json.loads(weights.with_suffix(".pt.source.json").read_text())
        if file_sha256(weights) != receipt["sha256"]:
            raise ValueError("Initial pretrained weights changed")
        state.update(initial_weights=receipt,development_yaml_sha256=file_sha256(yaml_path),
                     frozen_core_lock_sha256=file_sha256(ROOT/"data/processed/p57_core300_v2/dataset.lock.json"),
                     hardware={"machine":os.uname().machine,"cpu":"Apple M3 (environment inspection)","ram_bytes":psutil.virtual_memory().total},
                     code_fingerprints={str(p.relative_to(ROOT)):file_sha256(p) for d in (ROOT/"src/p57",ROOT/"scripts") for p in d.glob("*.py")},
                     environment={k:os.environ.get(k) for k in ("PYTHONHASHSEED","PYTORCH_ENABLE_MPS_FALLBACK")})
        samples_since_step = 0
        microbatch_sizes = []

        class TrialTrainer(DetectionTrainer):
            def get_dataset(self):
                data = super().get_dataset()
                if data.get("test") or Path(data["train"]).resolve()!=yaml_path.parent/"train/images" or Path(data["val"]).resolve()!=yaml_path.parent/"val/images":
                    raise ValueError("Non-development dataset exposed to trainer")
                return data

            def preprocess_batch(self,batch):
                nonlocal samples_since_step
                for p in batch["im_file"]:
                    if not Path(p).resolve().is_relative_to(yaml_path.parent/"train/images"):
                        raise ValueError("Non-training image entered optimization")
                result = super().preprocess_batch(batch)
                count = int(result["img"].shape[0])
                samples_since_step += count
                microbatch_sizes.append(count)
                return result

            def optimizer_step(self):
                nonlocal samples_since_step
                if not torch.isfinite(self.loss).all():
                    raise RuntimeError("Nonfinite training loss")
                super().optimizer_step()
                state["optimizer_steps"].append({"epoch":self.epoch+1,"microbatches_seen":len(microbatch_sizes),
                      "accumulation_setting":self.accumulate,"actual_images_since_previous_step":samples_since_step})
                samples_since_step = 0

        def sample_loop():
            while not stop.is_set():
                state["memory_samples"].append({"elapsed":time.monotonic()-start,**memory_sample()})
                stop.wait(1)

        threading.Thread(target=sample_loop,daemon=True).start()
        model = YOLO(str(weights),task="detect")
        def train_start(trainer):
            seed_everything(42,True)  # Ultralytics init_seeds uses warn_only=True; restore strict mode.
            state["setup_seconds"] = time.monotonic()-start
            state["actual_device"] = str(next(trainer.model.parameters()).device)
            if not state["actual_device"].startswith("mps"):
                raise RuntimeError("Trainer silently changed device")
            state["strict_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            if not state["strict_deterministic_algorithms"]:
                raise RuntimeError("Strict determinism not enabled")
            state["resolved_config"] = vars(trainer.args)
            state["optimizer"] = {"type":type(trainer.optimizer).__name__,"groups":[{k:v for k,v in g.items() if k!='params'} for g in trainer.optimizer.param_groups]}
            save_json(run_dir/"provenance.json",state)

        epoch_clock = {}
        def epoch_start(trainer):
            epoch_clock["start"] = time.monotonic()
        def epoch_train_end(trainer):
            epoch_clock["train_end"] = time.monotonic()
        def epoch_end(trainer):
            index = trainer.epoch+1
            if any(e["epoch"]==index for e in state["epochs"]):
                return
            metrics = {k:float(v) for k,v in trainer.metrics.items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError("Nonfinite feasibility validation metrics")
            state["epochs"].append({"epoch":index,"train_seconds":epoch_clock["train_end"]-epoch_clock["start"],
              "train_validation_save_seconds":time.monotonic()-epoch_clock["start"],"metrics_feasibility_validation_only":metrics,
              "optimizer_steps_cumulative":len(state["optimizer_steps"]),**memory_sample()})
            save_json(run_dir/"provenance.json",state)
        validation_clock = {}
        def val_start(validator):
            validation_clock["start"] = time.monotonic()
        def val_end(validator):
            state["validation_passes"].append({"seconds":time.monotonic()-validation_clock["start"],"scope":"development val only"})

        for event,callback in [("on_train_start",train_start),("on_train_epoch_start",epoch_start),("on_train_epoch_end",epoch_train_end),
                               ("on_fit_epoch_end",epoch_end),("on_val_start",val_start),("on_val_end",val_end)]:
            model.add_callback(event,callback)
        cfg.update(model=str(weights),data=str(yaml_path),project=str(run_dir.resolve()),name="fit",exist_ok=False)
        model.train(trainer=TrialTrainer,**cfg)
        if len(state["epochs"]) != 3:
            raise RuntimeError("Trial did not complete all three authorized epochs")
        state["checkpoint_hashes"] = {p.name:file_sha256(p) for p in (run_dir/"fit/weights").glob("*.pt")}
        if not {"best.pt","last.pt"} <= state["checkpoint_hashes"].keys():
            raise RuntimeError("Expected trial checkpoints missing")
        state["microbatch_sizes"] = microbatch_sizes
        state["optimizer_update_count"] = len(state["optimizer_steps"])
        state["pending_unstepped_images"] = samples_since_step
        save_json(run_dir/"provenance.json",state)
        gallery = validation_gallery(run_dir,run_dir/"fit/weights/best.pt")
        state.update(status="passed",gallery_images=len(gallery["images"]),gallery_detections=gallery["detections"],
                     result_label="Three-epoch feasibility validation measurements, not completed baseline or final research results")
        return state
    except Exception as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        state["total_elapsed_seconds"] = time.monotonic()-start
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if state["memory_samples"]:
            state["observed_memory_maxima"] = {k:max(s[k] for s in state["memory_samples"]) for k in state["memory_samples"][0] if k.endswith("bytes")}
            state["minimum_system_available_bytes"] = min(s["system_available_bytes"] for s in state["memory_samples"])
        save_json(run_dir/"provenance.json",state)
        verify_frozen_core()
        if lock:
            lock.close()
