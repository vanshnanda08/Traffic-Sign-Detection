"""Image validation, state identity, bounded rendering and portable exports."""
import base64
import csv
import hashlib
import io
import json
import math
import warnings
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
CLASSES = ['COMPULSARY_KEEP_RIGHT', 'NO_ENTRY', 'NO_STOPPING_OR_STANDING',
           'ROUNDABOUT', 'SPEED_LIMIT_30', 'SPEED_LIMIT_80']
HASHES = {'A': '15ed8470368fc535eed861d89dbee09b55977f1f935b40d41b24e052a6d6e4da',
          'C': 'cb918cad486d0fc6a2eb4c4886cd446bb4aef430d2f6e3581ccc2abff84c8091'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png(image):
    out = io.BytesIO()
    image.save(out, format='PNG', compress_level=3)
    return out.getvalue()


def pixel_sha(image):
    image = image.convert('RGB')
    header = json.dumps([image.width, image.height, 'RGB', 'uint8'], separators=(',', ':')).encode()
    return sha(header + image.tobytes())


def normalize(data, max_bytes=MAX_BYTES):
    if not data or len(data) > max_bytes:
        raise ValueError('Upload must be nonempty and at most 10 MiB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in {'JPEG', 'PNG'}:
                    raise ValueError('Decoded content must be JPEG or PNG.')
                if getattr(im, 'n_frames', 1) != 1:
                    raise ValueError('Animated images are not supported.')
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError('Image exceeds the 12 megapixel limit.')
                fmt, mode, original_size = im.format, im.mode, list(im.size)
                orientation = im.getexif().get(274, 1)
                im.load()
                oriented = ImageOps.exif_transpose(im)
                if oriented.mode in {'RGBA', 'LA', 'P'} or 'transparency' in oriented.info:
                    rgba = oriented.convert('RGBA')
                    rgb = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                    rgb.alpha_composite(rgba)
                    rgb = rgb.convert('RGB')
                else:
                    rgb = oriented.convert('RGB')
                rgb.info.clear()  # Do not export original EXIF/GPS/ICC metadata.
                return rgb, {'source_file_sha256': sha(data), 'decoded_format': fmt,
                             'source_mode': mode, 'source_size': original_size,
                             'exif_orientation': orientation, 'normalized_size': list(rgb.size),
                             'normalized_pixel_sha256': pixel_sha(rgb),
                             'normalization': 'EXIF transpose; RGB; alpha on white; grayscale/CMYK via Pillow RGB; strip metadata. No color-profile calibration claim.'}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Image is corrupt, unsafe or unreadable.') from error


def regions_valid(regions, width, height):
    if not isinstance(regions, list) or len(regions) > 32:
        raise ValueError('Use at most 32 simulation regions.')
    out = []
    for r in regions:
        if not isinstance(r, (list, tuple)) or len(r) != 4 or any(type(v) is not int for v in r):
            raise ValueError('Simulation region coordinates must be four integers.')
        x0, y0, x1, y1 = r
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError('Regions must have positive area and stay inside the image.')
        if list(r) not in out:
            out.append(list(r))
    return sorted(out)


def state_key(image_sha, regions, condition, models, device, manifest_sha):
    return sha(json.dumps([image_sha, regions, condition, models, device, manifest_sha,
                           {'shape': 448, 'conf': .001, 'iou': .7, 'max_det': 300}], sort_keys=True).encode())


def select_device(requested, mps_available):
    if requested not in {'cpu', 'mps'}:
        raise ValueError('Choose CPU or Apple GPU/MPS explicitly.')
    if requested == 'mps' and not mps_available:
        raise ValueError('Apple GPU/MPS was requested but is unavailable. Select CPU explicitly.')
    return requested


def model_manifest(root, path=None):
    root = Path(root)
    path = Path(path) if path else root / 'app/models.json'
    if not path.is_absolute():
        path = root / path
    data = json.loads(path.read_text())
    if data.get('classes') != CLASSES or set(data.get('models', {})) != {'A', 'C'}:
        raise ValueError('Model manifest has incorrect canonical classes/models.')
    for key, expected in HASHES.items():
        if data['models'][key]['sha256'] != expected:
            raise ValueError('Model manifest does not identify the canonical replacement checkpoint.')
    return data, sha(path.read_bytes())


def checkpoint(root, manifest, model):
    if model not in HASHES:
        raise ValueError('Unknown model selection.')
    p = Path(manifest['models'][model]['path']).expanduser()
    if not p.is_absolute():
        p = Path(root) / p
    if not p.is_file():
        raise ValueError(f'Model {model} checkpoint is missing. Configure a local path in app/models.json; no download is attempted.')
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != HASHES[model]:
        raise ValueError(f'Model {model} checkpoint bytes are incorrect or corrupt; inference rejected.')
    return p.resolve()


def validate_predictions(rows, width, height):
    if not isinstance(rows, list) or len(rows) > 300:
        raise ValueError('Invalid prediction count.')
    for r in rows:
        if type(r['class_id']) is not int or not 0 <= r['class_id'] < 6 or r['class_name'] != CLASSES[r['class_id']]:
            raise ValueError('Invalid prediction class.')
        x0, y0, x1, y1 = r['xyxy']
        if not all(math.isfinite(v) for v in [x0, y0, x1, y1, r['confidence']]):
            raise ValueError('Nonfinite prediction.')
        if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height and 0 <= r['confidence'] <= 1):
            raise ValueError('Invalid prediction geometry/confidence.')
        positive = x1 > x0 and y1 > y0
        if r.get('drawable', positive) != positive:
            raise ValueError('Drawable flag disagrees with clipped geometry.')
        if not positive and not (r.get('drawable') is False and r.get('geometry_status') == 'clipped_to_zero_area'):
            raise ValueError('Zero-area prediction must be explicitly flagged as non-drawable.')
    return rows


def displayed(rows, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Display confidence must be in [0,1].')
    return [r for r in rows if r['confidence'] >= threshold
            and r['xyxy'][2] > r['xyxy'][0] and r['xyxy'][3] > r['xyxy'][1]]


def annotated(image, rows, title):
    """Numbered boxes and separate readable legend; never calls inference."""
    preview = image.copy()
    preview.thumbnail((900, 680))
    scale_x, scale_y = preview.width / image.width, preview.height / image.height
    shown = rows[:50]
    font = ImageFont.load_default(size=19)
    canvas = Image.new('RGB', (max(760, preview.width), preview.height + 90 + 27 * len(shown)), 'white')
    canvas.paste(preview, (0, 45))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f'{title} | {len(rows)} displayed detections', font=font, fill='#16324f')
    for i, row in enumerate(shown, 1):
        box = [row['xyxy'][0]*scale_x, row['xyxy'][1]*scale_y+45,
               row['xyxy'][2]*scale_x, row['xyxy'][3]*scale_y+45]
        color = ['#0057b8', '#b24100', '#008050', '#8946a6'][row['class_id'] % 4]
        draw.rectangle(box, outline=color, width=3)
        label = str(i)
        draw.rectangle((box[0], box[1], box[0]+32, box[1]+27), fill='white')
        draw.text((box[0]+3, box[1]+2), label, fill=color, font=font)
        name = row['class_name'].replace('_', ' ')
        draw.text((10, preview.height+55+(i-1)*27), f'{i}. {name}   {row["confidence"]:.3f}', fill=color, font=font)
    if not rows:
        draw.text((10, preview.height+55), 'No detections at this display confidence.', fill='#404040', font=font)
    elif len(rows) > 50:
        draw.text((10, canvas.height-28), 'First 50 shown; JSON/CSV contain every displayed detection.', fill='#404040', font=font)
    return png(canvas)


def region_preview(image, regions):
    copy = image.copy()
    draw = ImageDraw.Draw(copy)
    for i, box in enumerate(regions, 1):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1-1, y1-1), outline='#dc7200', width=max(2, image.width//250))
        draw.text((x0+3, y0+3), f'R{i}', fill='#dc7200', font=ImageFont.load_default(size=20))
    return copy


def csv_predictions(result, threshold):
    out = io.StringIO(newline='')
    keys = ['model', 'kind', 'index', 'class_id', 'class_name', 'confidence', 'x0', 'y0', 'x1', 'y1', 'drawable', 'geometry_status']
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader()
    for model, value in result['models'].items():
        for kind, rows in [('raw', value['raw_predictions']), ('displayed', displayed(value['raw_predictions'], threshold))]:
            for i, row in enumerate(rows, 1):
                positive = row['xyxy'][2] > row['xyxy'][0] and row['xyxy'][3] > row['xyxy'][1]
                w.writerow(dict(zip(keys, [model, kind, i, row['class_id'], row['class_name'], row['confidence'], *row['xyxy'], positive, row.get('geometry_status', 'positive_area' if positive else 'clipped_to_zero_area') ])))
    return out.getvalue().encode()


def exports(normalized_bytes, shared_bytes, result, threshold, normalization):
    image = Image.open(io.BytesIO(shared_bytes)).convert('RGB')
    report = json.loads(json.dumps(result))
    report['display_confidence'] = threshold
    report['display_geometry'] = 'Positive-area clipped image boxes only; padding detections clipped to zero area remain flagged in raw JSON/CSV.'
    report['normalization'] = normalization
    report['upload_metrics'] = 'Not computed: no ground truth. Manual simulation regions are not ground truth.'
    report['rendering'] = 'Numbered boxes/legend, first50 only if crowded; JSON/CSV include all predictions.'
    report['models'] = {key: {**value, 'displayed_predictions': displayed(value['raw_predictions'], threshold)} for key, value in report['models'].items()}
    files = {'normalized_input.png': normalized_bytes, 'shared_input.png': shared_bytes}
    for model, value in report['models'].items():
        files[f'annotated_{model}.png'] = annotated(image, value['displayed_predictions'], value['name'])
    files['predictions.csv'] = csv_predictions(result, threshold)
    report['file_sha256'] = {k: sha(v) for k, v in files.items()}
    files['predictions.json'] = json.dumps(report, indent=2, allow_nan=False).encode()
    files['checksums.json'] = json.dumps({k: sha(v) for k, v in files.items()}, indent=2).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data)
    return files, buf.getvalue()
