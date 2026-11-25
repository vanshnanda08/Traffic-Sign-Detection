"""Single JSON request/reply, frozen-runtime inference; no dataset/run receipts needed."""
import base64
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'app')]
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
from p57_ui.core import (CLASSES, checkpoint, model_manifest, normalize, png,
                         pixel_sha, regions_valid, select_device, sha, validate_predictions)


def transformed(request):
    import numpy as np
    from p57.degradation import degrade, box_regions
    data = base64.b64decode(request['image_png'], validate=True)
    image, _ = normalize(data, max_bytes=64*1024*1024)
    regions = regions_valid(request.get('regions', []), *image.size)
    condition = request.get('condition', 'original')
    if condition not in {'original', 'F1', 'F2', 'F3', 'O1', 'O2', 'O3'}:
        raise ValueError('Unknown simulation condition.')
    identity = pixel_sha(image)
    rgb = np.array(image, dtype=np.uint8)
    detail = {'condition': condition, 'manual_simulation_regions': regions, 'ground_truth': False}
    if condition != 'original':
        if not regions:
            raise ValueError('Add at least one user-marked simulation region before fading/occlusion.')
        # Inset by a millionth of one pixel avoids floating-point floor/ceil drift.
        # Frozen box_regions must reconstruct exactly the user's integer rectangles.
        boxes = [[0, (x0+x1)/2/image.width, (y0+y1)/2/image.height,
                  (x1-x0-2e-6)/image.width, (y1-y0-2e-6)/image.height] for x0,y0,x1,y1 in regions]
        if box_regions(boxes, *image.size) != [tuple(r) for r in regions]:
            raise ValueError('Simulation rasterization did not match manual regions.')
        path = ROOT / 'configs/degradation_dev_v1.json'
        out, _, record = degrade(rgb[:, :, ::-1].copy(), boxes, 'upload:'+identity, condition, json.loads(path.read_text()))
        rgb = out[:, :, ::-1].copy()
        detail.update(record)
        detail['frozen_degradation_config_sha256'] = sha(path.read_bytes())
    from PIL import Image
    shared = Image.fromarray(rgb)
    detail.update(upload_fingerprint=identity, shared_pixel_sha256=pixel_sha(shared),
                  normalized_original_preserved=True, simulation='User-selected regions; not automatic sign segmentation or benchmark annotation.')
    return shared, png(shared), detail


def run(request):
    if request.get('op') not in {'capabilities', 'transform', 'infer'}:
        raise ValueError('Only capabilities, transform or infer operations are supported.')
    manifest, manifest_sha = model_manifest(ROOT, request.get('manifest_path'))
    for path, expected in manifest['frozen_files'].items():
        if sha((ROOT/path).read_bytes()) != expected:
            raise ValueError(f'Frozen inference dependency changed: {path}')
    if request['op'] == 'capabilities':
        import torch
        return {'mps_available': torch.backends.mps.is_available(), 'default_device': 'cpu'}
    shared, shared_bytes, simulation = transformed(request)
    base = {'shared_png': base64.b64encode(shared_bytes).decode(), 'simulation': simulation,
            'shared_file_sha256': sha(shared_bytes), 'shared_pixel_sha256': pixel_sha(shared)}
    if request['op'] == 'transform':
        return base
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from p57.complete_validation import CompletePredictor
    from p57.environment import seed_everything, validate_pins
    versions = validate_pins()
    device = select_device(request.get('device', 'cpu'), torch.backends.mps.is_available())
    models = request.get('models', ['A', 'C'])
    if models not in [['A'], ['C'], ['A', 'C']]:
        raise ValueError('Choose A, C or A then C.')
    paths = {m: checkpoint(ROOT, manifest, m) for m in models}  # all bytes checked before loading any model
    seed_everything(42, True)
    torch.set_num_threads(2)
    class AppPredictor(CompletePredictor):
        def pre_transform(self, images):
            self.original_shapes = [list(im.shape) for im in images]
            return [LetterBox(new_shape=(448,448), auto=False, scale_fill=False,
                              scaleup=True, center=True, stride=32)(image=im) for im in images]

        def preprocess(self, images):
            output = super().preprocess(images)
            self.actual_shape = list(output.shape)
            return output

    source = np.array(shared)[:, :, ::-1].copy()
    result = {k:v for k,v in base.items() if k != 'shared_png'}
    result.update(models={}, device=device, seed=42, manifest_sha256=manifest_sha,
                  app_protocol_revision='w7.1-drawable-geometry',
                  runtime={'python':__import__('platform').python_version(),
                           'platform':__import__('platform').platform(),
                           'architecture':__import__('platform').machine()},
                  software_versions=versions,
                  inference={'confidence_floor': .001, 'nms_iou': .7, 'max_det': 300,
                             'class_aware': True, 'multi_label': False, 'half': False,
                             'augmentation': False, 'class_filter': None,
                             'preprocessing': 'Fixed448x448 centered aspect-preserving LetterBox, scaleupTrue, stride32, fill114; BGR to RGB /255 FP32. No stock rect auto-padding.',
                             'coordinates': 'xyxy pixels relative to EXIF-normalized uploaded image; no ground truth.',
                             'coordinate_postprocessing': 'Clip native boxes to normalized image. Retain zero-area padding detections flagged in raw output; display positive-area boxes only.',
                             'NMS': 'Frozen CompletePredictor with unlimited wall-clock NMS',
                             'versus_W6': 'Upload predictor uses multi_labelFalse and fixed448 letterbox. W6 evaluator used multi_labelTrue and nominal416/defaultpad.5 loading with actual448 tensors. Upload predictions are not benchmark metric reproduction.'})
    for model in models:
        started = time.monotonic()
        network = YOLO(str(paths[model]), task='detect')
        names = [network.names[i] for i in range(len(network.names))]
        if names != CLASSES:
            raise ValueError('Checkpoint class metadata differs from the canonical six classes.')
        actual = network.predict(source=source, predictor=AppPredictor, device=device,
                                 imgsz=448, batch=1, rect=False, half=False, conf=.001,
                                 iou=.7, max_det=300, agnostic_nms=False, augment=False,
                                 classes=None, save=False, save_txt=False, save_crop=False,
                                 show=False, verbose=False, plots=False, stream=False)
        predictor = network.predictor
        if len(actual) != 1 or not predictor.last_completion['complete']:
            raise ValueError('Upload inference did not complete.')
        if str(predictor.device).split(':')[0] != device or predictor.model.fp16:
            raise ValueError('Inference device/precision differs from the explicit selection.')
        rows = [{'class_id':int(c), 'class_name':CLASSES[int(c)], 'confidence':float(s), 'xyxy':[float(x) for x in box]}
                for box,c,s in zip(actual[0].boxes.xyxy.cpu().tolist(), actual[0].boxes.cls.cpu().tolist(), actual[0].boxes.conf.cpu().tolist())]
        for row in rows:
            box=row['xyxy'];row['drawable']=bool(box[2]>box[0] and box[3]>box[1])
            row['geometry_status']='positive_area' if row['drawable'] else 'clipped_to_zero_area'
        validate_predictions(rows, shared.width, shared.height)
        result['models'][model] = {'name':manifest['models'][model]['name'],
            'checkpoint_sha256':manifest['models'][model]['sha256'], 'raw_predictions':rows,
            'shared_pixel_sha256':pixel_sha(shared), 'device':str(predictor.device),
            'actual_tensor_shape':predictor.actual_shape, 'elapsed_model_seconds':time.monotonic()-started,
            'raw_post_NMS_count':len(rows),'non_drawable_clipped_count':sum(not r['drawable'] for r in rows),
            'NMS_complete':predictor.last_completion['complete']}
        del actual, predictor, network
        import gc
        gc.collect()
        if device == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
    return result


if __name__ == '__main__':
    started = time.monotonic()
    try:
        raw = sys.stdin.buffer.read(90*1024*1024+1)
        if len(raw)>90*1024*1024:
            raise ValueError('Request exceeds the bounded protocol size.')
        request = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            answer = run(request)
        response = {'ok':True, 'result':answer, 'worker_seconds_before_cleanup':time.monotonic()-started}
    except Exception as error:
        response = {'ok':False, 'error':str(error), 'worker_seconds_before_cleanup':time.monotonic()-started}
    sys.stdout.write(json.dumps(response, allow_nan=False))
    sys.stdout.flush()
    raise SystemExit(0 if response['ok'] else 1)
