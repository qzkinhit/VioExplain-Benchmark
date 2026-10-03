"""Descriptive audit only; reads stored predictions, performs no inference or tuning."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
p=Path(__file__).resolve().parent
frame=pd.read_csv(p/'predictions_scored.csv');frame['candidates']=frame.candidate_events.map(json.loads)
diagnostics=[];classification=[]
for name,g in frame.groupby('method'):
    fault=g[g['class']!=0];present=np.array([int(c) in cs for c,cs in zip(fault['class'],fault.candidates)])
    pr,re,f1,support=precision_recall_fscore_support(g['class'],g.predicted_class,labels=list(range(22)),zero_division=0)
    for c in range(22):classification.append({'method':name,'class':c,'precision':pr[c],'recall':re[c],'f1':f1[c],'support':support[c]})
    diagnostics.append({'method':name,'fault_candidate_recall':float(present.mean()),'top1_among_candidate_present_fault_windows':float((fault.loc[present,'predicted_class']==fault.loc[present,'class']).mean()) if present.any() else None,'zero_candidate_windows':int((g.n_candidates==0).sum()),'normal_windows':int((g['class']==0).sum())})
pd.DataFrame(diagnostics).to_csv(p/'candidate_diagnostics.csv',index=False);pd.DataFrame(classification).to_csv(p/'per_class_classification.csv',index=False)
a=frame[frame.method=='minexplain_frozen_chronos2'];b=frame[frame.method=='minexplain_raw_temporal'];pairs=a.merge(b,on=['item','start','class'],suffixes=('_fm','_raw'));deltas=[]
for c,g in pairs.groupby('class'):
    fm=g.predicted_class_fm==c;raw=g.predicted_class_raw==c
    deltas.append({'class':c,'n':len(g),'fm_top1':float(fm.mean()),'raw_top1':float(raw.mean()),'difference':float(fm.mean()-raw.mean()),'fm_only_correct':int((fm&~raw).sum()),'raw_only_correct':int((raw&~fm).sum())})
pd.DataFrame(deltas).to_csv(p/'paired_class_differences.csv',index=False)
knowledge=json.loads((p/'knowledge.json').read_text());report={'normal_calibration_windows':int((pd.read_csv(p/'training_split_windows.csv').query('split == "calibration"')['class']==0).sum()),'knowledge':{key:{'mandatory':sum(v['mandatory'] for v in vals),'possible':sum(not v['mandatory'] for v in vals)} for key,vals in knowledge['events'].items()},'fm_improved_classes':sum(r['difference']>0 for r in deltas),'fm_harmed_classes':sum(r['difference']<0 for r in deltas),'fm_tied_classes':sum(r['difference']==0 for r in deltas),'fm_only_correct_windows':sum(r['fm_only_correct'] for r in deltas),'raw_only_correct_windows':sum(r['raw_only_correct'] for r in deltas),'postprocess_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print(pd.DataFrame(diagnostics).to_string(index=False))
