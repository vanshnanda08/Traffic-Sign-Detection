"""Official GTSRB structural/provenance and conservative overlap-risk audit."""

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from p57.assets import file_sha256
from p57.dataset import ROOT, config, image_features, parse_labels, save_json, save_csv


def official_rows():
    root=ROOT/"data/raw/gtsrb_official"
    train=ZipFile(root/"GTSRB_Final_Training_Images.zip")
    test=ZipFile(root/"GTSRB_Final_Test_Images.zip")
    gt=ZipFile(root/"GTSRB_Final_Test_GT.zip")
    rows=[]
    for member in sorted(n for n in train.namelist() if n.endswith('.csv')):
        for row in csv.DictReader(io.StringIO(train.read(member).decode('utf-8-sig')),delimiter=';'):
            name=str(Path(member).parent/row['Filename'])
            rows.append({"split":"train","member":name,"class_id":int(row['ClassId']),
                         "track":f"{int(row['ClassId']):05d}/{row['Filename'].split('_')[0]}",
                         "width":int(row['Width']),"height":int(row['Height']),
                         "roi":[int(row[k]) for k in ('Roi.X1','Roi.Y1','Roi.X2','Roi.Y2')]})
    for row in csv.DictReader(io.StringIO(gt.read('GT-final_test.csv').decode('utf-8-sig')),delimiter=';'):
        rows.append({"split":"test","member":"GTSRB/Final_Test/Images/"+row['Filename'],"class_id":int(row['ClassId']),
                     "track":None,"width":int(row['Width']),"height":int(row['Height']),
                     "roi":[int(row[k]) for k in ('Roi.X1','Roi.Y1','Roi.X2','Roi.Y2')]})
    return rows,train,test,gt


def variants(image,boxes):
    result=[("full",image)]
    for i,(x0,y0,x1,y1) in enumerate(boxes):
        if x1>x0 and y1>y0:
            result.append((f"roi_{i}",image[y0:y1,x0:x1]))
    return result


def normalized(image,size=64):
    color=cv2.resize(image,(size,size),interpolation=cv2.INTER_LINEAR)
    ph,dh,_=image_features(color)
    return color,ph,dh


def geometry(a,b,sift,params):
    features=[]
    for image in (a,b):
        gray=cv2.cvtColor(cv2.resize(image,(params['geometry_size'],)*2),cv2.COLOR_BGR2GRAY)
        kp,d=sift.detectAndCompute(gray,None)
        features.append((np.float32([k.pt for k in kp]),d))
    accepted=[]
    for (pa,da),(pb,db) in (features,list(reversed(features))):
        if da is None or db is None or min(len(da),len(db))<10:continue
        good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<params['geometry_ratio']*n.distance]
        if len(good)<10:continue
        qa=np.float32([pa[m.queryIdx] for m in good]);qb=np.float32([pb[m.trainIdx] for m in good])
        cv2.setRNGSeed(params['seed'])
        _,mask=cv2.findHomography(qa,qb,cv2.RANSAC,3.0)
        if mask is None:continue
        use=mask.ravel().astype(bool);n=int(use.sum())
        if n<params['geometry_min_inliers'] or n/len(good)<params['geometry_min_inlier_fraction']:continue
        coverage=min(cv2.contourArea(cv2.convexHull(p[use]))/params['geometry_size']**2 for p in (qa,qb))
        if coverage>=params['geometry_min_coverage']:accepted.append({"inliers":n,"inlier_fraction":n/len(good),"coverage":coverage})
    return max(accepted,key=lambda x:x['inliers']) if accepted else None


def audit_gtsrb():
    params=json.loads((ROOT/'configs/gtsrb_audit_v1.json').read_text())
    out=ROOT/'results/development_w2_v1/gtsrb_audit'
    if out.exists():raise ValueError('Audit directory exists; use a separately versioned audit')
    out.mkdir(parents=True)
    cfg=config(ROOT/'configs/dataset_v2.json'); core=ROOT/cfg['output_dir']
    rows_core=list(csv.DictReader((ROOT/cfg['audit_dir']/'manifest.csv').open()))
    queries=[]
    for row in rows_core:
        image=cv2.imread(str(core/row['image_path']));h,w=image.shape[:2]
        boxes=parse_labels((core/row['label_path']).read_text(),6)
        crops=[]
        for _,x,y,bw,bh in boxes:
            crops.append((max(0,int(np.floor((x-bw/2)*w))),max(0,int(np.floor((y-bh/2)*h))),min(w,int(np.ceil((x+bw/2)*w))),min(h,int(np.ceil((y+bh/2)*h)))))
        for variant,im in variants(image,crops):
            rgb,ph,dh=normalized(im)
            queries.append({"row":row,"variant":variant,"color":rgb,"ph":ph,"dh":dh})
    qph=np.array([q['ph'] for q in queries],dtype=np.uint64);qdh=np.array([q['dh'] for q in queries],dtype=np.uint64)
    lut=np.array([i.bit_count() for i in range(256)],np.uint8)
    rows,train,test,gt=official_rows()
    count_by_class=defaultdict(Counter);track_counts=Counter();accepted=[];issues=[];excluded=set();candidate_count=0;variants_count=0
    readmes={}
    for z in (train,test):
        for n in z.namelist():
            if 'readme' in n.lower():readmes[n]=z.read(n).decode('cp1252')
    save_json(out/'official_readmes.json',readmes)
    sift=cv2.SIFT_create(nfeatures=120,contrastThreshold=.025);cv2.setNumThreads(2)
    fields=['core_source_id','core_split','core_variant','official_split','official_member','official_class_id','official_track','official_variant','phash_distance','dhash_distance','normalized_mae','gray_correlation','disposition','inliers','inlier_fraction','coverage']
    with (out/'candidates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for index,r in enumerate(rows):
            z=train if r['split']=='train' else test
            image=cv2.imdecode(np.frombuffer(z.read(r['member']),np.uint8),1)
            if image is None or image.shape[:2]!=(r['height'],r['width']):
                issues.append({"member":r['member'],"issue":"decode/dimension mismatch"});continue
            count_by_class[r['class_id']][r['split']]+=1
            if r['track']:track_counts[r['track']]+=1
            x0,y0,x1,y1=r['roi']
            if not (0<=x0<=x1<r['width'] and 0<=y0<=y1<r['height']):
                issues.append({"member":r['member'],"issue":"invalid official ROI"});reference_variants=[('full',image)]
            else:reference_variants=variants(image,[(x0,y0,x1+1,y1+1)])
            for variant,im in reference_variants:
                color,ph,dh=normalized(im);variants_count+=1
                pd=lut[np.bitwise_xor(qph,np.uint64(ph)).view(np.uint8).reshape(-1,8)].sum(axis=1)
                dd=lut[np.bitwise_xor(qdh,np.uint64(dh)).view(np.uint8).reshape(-1,8)].sum(axis=1)
                for qi in np.flatnonzero((pd<=params['phash_distance'])&(dd<=params['dhash_distance'])):
                    candidate_count+=1;q=queries[int(qi)];cr=q['row'];mae=float(np.abs(q['color'].astype(np.float32)-color).mean())
                    ga=cv2.cvtColor(q['color'],cv2.COLOR_BGR2GRAY).astype(np.float32);gb=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY).astype(np.float32)
                    informative=min(float(ga.std()),float(gb.std()))>=params['minimum_gray_std']
                    correlation=float(np.corrcoef(ga.ravel(),gb.ravel())[0,1]) if informative else 0.0
                    decision='candidate_not_accepted';measurement={}
                    if informative and mae<=params['pixel_mae_max'] and correlation>=params['pixel_correlation_min']:
                        decision='accepted_potential_pixel_overlap'
                    elif informative and params['candidate_class_ids'].get(cr['remapped_class_ids'])==r['class_id'] and mae<=params['geometry_mae_max'] and correlation>=params['geometry_correlation_min']:
                        measurement=geometry(q['color'],color,sift,params) or {}
                        if measurement:decision='accepted_potential_geometric_overlap'
                    if not informative:decision='uninformative_no_independence_conclusion'
                    result=dict(zip(fields,[cr['source_id'],cr['split'],q['variant'],r['split'],r['member'],r['class_id'],r['track'] or '',variant,int(pd[qi]),int(dd[qi]),mae,correlation,decision,measurement.get('inliers',''),measurement.get('inlier_fraction',''),measurement.get('coverage','')]))
                    writer.writerow(result)
                    if decision.startswith('accepted'):
                        accepted.append(result)
                        if r['track']:excluded.add(r['track'])
            if index%5000==0:print(f'GTSRB screened {index}/{len(rows)}; candidates={candidate_count}; accepted={len(accepted)}',flush=True)
    for z in (train,test,gt):z.close()
    save_csv(out/'accepted_overlaps.csv',accepted,fields)
    exclusions=[{'track':track,'available_training_images':track_counts[track],'disposition':'exclude_complete_available_training_track'} for track in sorted(excluded)]
    save_csv(out/'excluded_training_tracks.csv',exclusions,['track','available_training_images','disposition'])
    compatible=[]
    for c in [1,5,17,38,40]:
        all_tracks={t for t in track_counts if t.startswith(f'{c:05d}/')};remaining=all_tracks-excluded
        compatible.append({'gtsrb_class_id':c,'training_images':count_by_class[c]['train'],'training_tracks':len(all_tracks),'remaining_training_tracks':len(remaining),'remaining_training_images':sum(track_counts[t] for t in remaining),'status':'conditional designs only; not whole ROUNDABOUT class' if c==40 else 'ID verified from official CSV and visual representative; future reference protocol pending'})
    summary={'status':'blocked','scope':'data-integrity overlap risk audit; no model training/inference',
      'blockers':['No explicit dataset license grant located in the supplied official archive READMEs or public archive metadata; source-declared license unresolved', 'Accepted potential matches require review before asserting independence; generic sign designs can match; official final-test files lack available acquisition-track IDs'],
      'official_images':sum(sum(v.values()) for v in count_by_class.values()),'official_split_counts':dict(Counter(r['split'] for r in rows)),
      'official_classes':len(count_by_class),'training_tracks':len(track_counts),'indian_images':len(rows_core),'indian_query_variants':len(queries),'official_variants':variants_count,
      'candidate_variant_pairs':candidate_count,'accepted_variant_pairs':len(accepted),'indian_images_with_accepted_matches':len({r['core_source_id'] for r in accepted}),
      'excluded_training_tracks':len(excluded),'excluded_training_images':sum(track_counts[t] for t in excluded),'compatible_support_after_exclusion':compatible,
      'issues':issues,'policy':params,'physical_identity_or_independence_guaranteed':False,
      'source_receipts':[json.loads(p.read_text()) for p in (ROOT/'data/raw/gtsrb_official').glob('*.source.json')]}
    save_json(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['source_receipts','policy','issues']},indent=2))
    return summary
