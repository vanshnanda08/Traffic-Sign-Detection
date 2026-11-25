"""Focused upload/application tests. No benchmark inference or dataset changes."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from p57_ui.core import (normalize, png, pixel_sha, MAX_BYTES, regions_valid, state_key,
                         select_device, model_manifest, checkpoint, CLASSES, displayed,
                         validate_predictions, exports, sha)
from worker import transformed


def encoded(image, fmt='PNG', **kwargs):
    out=io.BytesIO();image.save(out,format=fmt,**kwargs);return out.getvalue()


@pytest.mark.parametrize('mode', ['L','RGBA','CMYK','P'])
def test_normalize_modes_and_strip_metadata(mode):
    image=Image.new(mode,(12,8));raw=encoded(image,'JPEG' if mode=='CMYK' else 'PNG')
    actual,meta=normalize(raw)
    assert actual.mode=='RGB' and actual.size==(12,8) and not actual.info
    assert meta['source_mode']==mode and meta['normalized_pixel_sha256']==pixel_sha(actual)
    if mode=='RGBA':assert actual.getpixel((0,0))==(255,255,255)


def test_exif_orientation_before_coordinate_system():
    image=Image.new('RGB',(40,20),'blue');image.paste('red',(0,0,10,20))
    exif=Image.Exif();exif[274]=6
    raw=encoded(image,'JPEG',exif=exif)
    actual,meta=normalize(raw)
    expected=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    assert actual.size==(20,40) and actual.tobytes()==expected.tobytes() and meta['exif_orientation']==6
    assert 'exif' not in Image.open(io.BytesIO(png(actual))).info


@pytest.mark.parametrize('data', [b'bad png', b'', b'x'*(MAX_BYTES+1)])
def test_corrupt_empty_or_oversized(data):
    with pytest.raises(ValueError):normalize(data)


def test_reject_animated_and_decoded_format_not_extension():
    first=Image.new('RGB',(4,4),'red');second=Image.new('RGB',(4,4),'blue')
    animation=encoded(first,save_all=True,append_images=[second],duration=30)
    with pytest.raises(ValueError,match='Animated'):normalize(animation)
    with pytest.raises(ValueError,match='JPEG or PNG'):normalize(encoded(first,'BMP'))
    with pytest.raises(ValueError,match='megapixel'):normalize(encoded(Image.new('L',(4001,3000))))


@pytest.mark.parametrize('regions', [[[0,0,0,1]], [[-1,0,1,1]], [[0,0,11,1]], [[0,0,1.1,1]], [[True,0,1,1]]])
def test_region_geometry_rejects_invalid(regions):
    with pytest.raises(ValueError):regions_valid(regions,10,10)


def test_region_dedup_and_state_invalidation():
    assert regions_valid([[0,0,2,3],[0,0,2,3]],5,5)==[[0,0,2,3]]
    args=['image',[],'original',['A','C'],'cpu','manifest']
    initial=state_key(*args)
    for i,value in enumerate(['newimage',[[0,0,1,1]],'F1',['A'],'mps','newmanifest']):
        changed=copy.deepcopy(args);changed[i]=value;assert state_key(*changed)!=initial


@pytest.mark.parametrize('condition', ['original','F1','F2','F3','O1','O2','O3'])
def test_deterministic_shared_multiregion_transform(condition):
    image=Image.new('RGB',(113,79),'blue');raw=png(image)
    request={'image_png':base64.b64encode(raw).decode(),'regions':[[1,2,70,60],[30,20,100,70]],'condition':condition}
    a,ab,am=transformed(request);b,bb,bm=transformed({**request,'regions':list(reversed(request['regions']))})
    assert ab==bb and am==bm and raw==png(image) and am['shared_pixel_sha256']==pixel_sha(a)==pixel_sha(b)
    assert am['ground_truth'] is False
    if condition=='original':assert a.tobytes()==image.tobytes()


def test_fading_overlaps_processed_once_and_requires_regions():
    image=Image.new('RGB',(40,30),'black');raw=base64.b64encode(png(image)).decode()
    a,_,_=transformed({'image_png':raw,'regions':[[0,0,30,30],[10,0,40,30]],'condition':'F1'})
    assert a.getpixel((5,5))==a.getpixel((15,5))==a.getpixel((35,5))==(26,26,26)
    with pytest.raises(ValueError,match='at least one'):transformed({'image_png':raw,'regions':[],'condition':'O1'})


def test_models_missing_wrong_bytes_and_class_manifest(tmp_path):
    manifest,_=model_manifest(ROOT)
    bad=copy.deepcopy(manifest);bad['models']['A']['path']=str(tmp_path/'missing.pt')
    with pytest.raises(ValueError,match='missing'):checkpoint(ROOT,bad,'A')
    p=tmp_path/'wrong.pt';p.write_bytes(b'not a checkpoint');bad['models']['A']['path']=str(p)
    with pytest.raises(ValueError,match='incorrect or corrupt'):checkpoint(ROOT,bad,'A')
    bad['classes']=['wrong'];p=tmp_path/'manifest.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='classes'):model_manifest(ROOT,p)
    bad=copy.deepcopy(manifest);bad['models']['A']['sha256']='0'*64;p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='canonical'):model_manifest(ROOT,p)


def test_device_and_valid_empty_predictions():
    assert select_device('cpu',False)=='cpu' and select_device('mps',True)=='mps'
    with pytest.raises(ValueError):select_device('mps',False)
    with pytest.raises(ValueError):select_device('auto',True)
    assert validate_predictions([],10,10)==[]
    good={'class_id':0,'class_name':CLASSES[0],'confidence':.9,'xyxy':[0,0,10,10]}
    validate_predictions([good],10,10)
    for change in [{'class_id':6},{'confidence':float('nan')},{'xyxy':[-1,0,10,10]}]:
        with pytest.raises(ValueError):validate_predictions([{**good,**change}],10,10)


def test_exports_fixed_safe_paths_hashes_raw_displayed_and_empty():
    image=png(Image.new('RGB',(40,20),'gray'))
    raw=[{'class_id':0,'class_name':CLASSES[0],'confidence':.9,'xyxy':[0,0,30,20]},
         {'class_id':1,'class_name':CLASSES[1],'confidence':.1,'xyxy':[10,0,20,10]}]
    result={'models':{m:{'name':m,'raw_predictions':raw,'shared_pixel_sha256':'shared'} for m in ['A','C']},'device':'cpu'}
    files,archive=exports(image,image,result,.25,{})
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.testzip() is None and set(z.namelist())==set(files)
        assert all('/' not in p and not p.endswith('.pt') for p in z.namelist())
        checks=json.loads(z.read('checksums.json'))
        assert all(sha(z.read(p))==h for p,h in checks.items())
        report=json.loads(z.read('predictions.json'))
        assert all(len(v['raw_predictions'])==2 and len(v['displayed_predictions'])==1 for v in report['models'].values())
        assert z.read('predictions.csv').decode().count('\n')==7
    files,empty=exports(image,image,result,1,{})
    assert not json.loads(files['predictions.json'])['models']['A']['displayed_predictions']
    assert len(result['models']['A']['raw_predictions'])==2  # export/filter never mutates raw data


def test_non_square_inverse_letterbox_geometry():
    import torch
    from ultralytics.utils.ops import scale_boxes
    # Known box after fixed448 letterbox for a 960x540 normalized source.
    original=torch.tensor([[120.,60.,720.,480.]])
    ratio=448/960;pad_y=(448-540*ratio)/2
    network=original*ratio;network[:,[1,3]]+=pad_y
    restored=scale_boxes((448,448),network,(540,960))
    torch.testing.assert_close(restored,original,atol=1e-4,rtol=0)


def test_real_worker_failure_captures_exit_and_cleanup():
    from p57_ui.client import call_worker, WorkerFailure
    with pytest.raises(WorkerFailure) as failed:
        call_worker(ROOT,{'op':'transform','image_png':base64.b64encode(png(Image.new('RGB',(20,10)))).decode(),
                          'condition':'F1','regions':[]})
    receipt=failed.value.receipt
    assert receipt['actual_exit']==1 and receipt['elapsed_seconds_including_cleanup']>0
    assert 'at least one' in str(failed.value) and receipt['owned_temporary_directory_removed_on_return']


def test_padding_predictions_retained_raw_but_never_drawn():
    row={'class_id':0,'class_name':CLASSES[0],'confidence':.9,'xyxy':[0,0,10,0]}
    with pytest.raises(ValueError,match='Zero-area'):validate_predictions([row],10,10)
    flagged={**row,'drawable':False,'geometry_status':'clipped_to_zero_area'}
    validate_predictions([flagged],10,10)
    assert displayed([flagged],0)==[] and len([flagged])==1
    with pytest.raises(ValueError,match='Drawable'):validate_predictions([{**flagged,'drawable':True}],10,10)
