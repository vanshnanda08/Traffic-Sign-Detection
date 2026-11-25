"""Fixed-threshold DEVELOPMENT diagnostic matching. Not research mAP or inference."""
from p57_ui.core import CLASSES,displayed


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0


def match_image(raw,ground_truth,confidence=.25,iou_threshold=.5):
    # Original indices survive filtering; class/box/index are explicit deterministic ties.
    selected=[(i,r) for i,r in enumerate(raw) if displayed([r],confidence)]
    selected.sort(key=lambda v:(-v[1]['confidence'],v[1]['class_id'],*v[1]['xyxy'],v[0]))
    unused=set(range(len(ground_truth)));decisions=[]
    counts=[{'class_id':i,'class_name':name,'TP':0,'FP':0,'FN':0,'GT':sum(g['class_id']==i for g in ground_truth)} for i,name in enumerate(CLASSES)]
    for index,pred in selected:
        candidates=[(iou(pred['xyxy'],ground_truth[j]['xyxy']),j) for j in sorted(unused) if ground_truth[j]['class_id']==pred['class_id']]
        candidates.sort(key=lambda v:(-v[0],v[1]))
        best=candidates[0] if candidates else None
        if best and best[0]>=iou_threshold:
            overlap,j=best;unused.remove(j);kind='TP';counts[pred['class_id']]['TP']+=1
        else:overlap=None;j=None;kind='FP';counts[pred['class_id']]['FP']+=1
        decisions.append({'prediction_index':index,'kind':kind,'ground_truth_index':j,'matched_iou':overlap})
    for j in sorted(unused):counts[ground_truth[j]['class_id']]['FN']+=1
    return {'per_class':counts,'decisions':decisions,'unmatched_ground_truth_indices':sorted(unused),
            **{key:sum(r[key] for r in counts) for key in ['TP','FP','FN','GT']},'displayed_count':len(selected)}


def aggregate(records):
    totals=[{'class_id':i,'class_name':name,'TP':0,'FP':0,'FN':0,'GT':0} for i,name in enumerate(CLASSES)]
    for record in records:
        for row in record['per_class']:
            for key in ['TP','FP','FN','GT']:totals[row['class_id']][key]+=row[key]
    def rates(row):
        row['precision']=row['TP']/(row['TP']+row['FP']) if row['TP']+row['FP'] else None
        row['recall']=row['TP']/row['GT'] if row['GT'] else None
        return row
    overall={key:sum(r[key] for r in totals) for key in ['TP','FP','FN','GT']}
    return {**rates(overall),'per_class':[rates(r) for r in totals],
            'scope':'Validation-only fixed .25 confidence/.5 IoU diagnostic; micro aggregate, not mAP/best-F1/W6/test evidence.'}
