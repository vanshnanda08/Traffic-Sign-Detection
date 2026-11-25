"""Exact-byte/pixel supplement and full available-track exclusion manifest."""

import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

from p57.dataset import ROOT, config, parse_labels, save_csv, save_json
from p57.assets import file_sha256
from p57.degradation import box_regions
from p57.gtsrb_audit import official_rows, variants


def pixel_key(image):
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def main():
    out=ROOT/'results/development_w2_v1'
    cfg=config(ROOT/'configs/dataset_v2.json');core=ROOT/cfg['output_dir']
    indian=list(csv.DictReader((ROOT/cfg['audit_dir']/'manifest.csv').open()))
    raw_hashes=defaultdict(list);pixel_hashes=defaultdict(list);normalized_hashes=defaultdict(list)
    for r in indian:
        raw_hashes[r['image_sha256']].append(r['source_id'])
        im=cv2.imread(str(core/r['image_path']));h,w=im.shape[:2]
        labels=parse_labels((core/r['label_path']).read_text(),6)
        for variant,image in variants(im,box_regions(labels,w,h)):
            pixel_hashes[pixel_key(image)].append((r['source_id'],variant))
            normalized_hashes[pixel_key(cv2.resize(image,(64,64)))].append((r['source_id'],variant))
    rows,train,test,gt=official_rows();matches=[]
    for r in rows:
        data=(train if r['split']=='train' else test).read(r['member'])
        sha=hashlib.sha256(data).hexdigest()
        for core_id in raw_hashes.get(sha,[]):matches.append({**r,'core_source_id':core_id,'kind':'exact_file_bytes'})
        image=cv2.imdecode(np.frombuffer(data,np.uint8),1);x0,y0,x1,y1=r['roi']
        for variant,im in variants(image,[(x0,y0,min(image.shape[1],x1+1),min(image.shape[0],y1+1))]):
            for key,pool in [('exact_decoded_pixels',pixel_hashes),('exact_normalized_pixels',normalized_hashes)]:
                candidate=im if key=='exact_decoded_pixels' else cv2.resize(im,(64,64))
                for core_id,core_variant in pool.get(pixel_key(candidate),[]):matches.append({**r,'core_source_id':core_id,'core_variant':core_variant,'official_variant':variant,'kind':key})
    near=list(csv.DictReader((out/'gtsrb_audit/accepted_overlaps.csv').open()))
    excluded={r['official_track'] for r in near if r['official_split']=='train'}|{r['track'] for r in matches if r['split']=='train'}
    manifest=[{'member':r['member'],'class_id':r['class_id'],'track':r['track'],
               'excluded_for_potential_overlap':r['track'] in excluded,
               'compatible_transfer_class':r['class_id'] in {1,5,17,38},
               'roundabout_requires_design_review':r['class_id']==40} for r in rows if r['split']=='train']
    save_csv(out/'reference_training_inventory.csv',manifest,list(manifest[0]))
    counts=[]
    for c in [1,5,17,38,40]:
        eligible=[r for r in manifest if r['class_id']==c and not r['excluded_for_potential_overlap']]
        counts.append({'gtsrb_class_id':c,'remaining_images':len(eligible),'remaining_tracks':len({r['track'] for r in eligible}),'transfer_status':'conditional only; not blanket mapping' if c==40 else 'compatible candidate verified from official ID and appearance'})
    save_json(out/'gtsrb_exact_and_inventory.json',{'official_images_checked':len(rows),'exact_match_records':matches,
        'exact_match_counts':dict(Counter(r['kind'] for r in matches)),
        'excluded_training_tracks':sorted(excluded),'excluded_training_images':sum(r['excluded_for_potential_overlap'] for r in manifest),
        'compatible_support':counts,'classification_is_not_detection':True,'test_track_ids_available':False,
        'indian_freeze_changed':False,'reference_training_executed':False})
    for z in (train,test,gt):z.close()
    print('Exact audit complete:',len(rows),'official images;',len(matches),'exact records;',len(excluded),'excluded tracks')


if __name__=='__main__':main()
