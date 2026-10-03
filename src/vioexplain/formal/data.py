"""Load official data without inventing causal or event annotations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

@dataclass
class Record:
    dataset: str
    item: str
    train: np.ndarray
    test: np.ndarray
    label: np.ndarray
    events: list
    columns: list
    sources: list

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def intervals(y):
    padded=np.r_[False,np.asarray(y,dtype=bool),False]
    return list(zip(np.flatnonzero(np.diff(padded.astype(int))==1),np.flatnonzero(np.diff(padded.astype(int))==-1)))

def smd_records(root):
    root=Path(root)
    for p in sorted((root/'test').glob('machine-*.txt')):
        tr=root/'train'/p.name; lab=root/'test_label'/p.name; ip=root/'interpretation_label'/p.name
        x=np.loadtxt(p,delimiter=','); train=np.loadtxt(tr,delimiter=','); y=np.loadtxt(lab,delimiter=',').astype(int)
        ev=[]
        for line in ip.read_text().splitlines():
            span,dims=line.split(':'); start,end=map(int,span.split('-'))
            ev.append({'start':start,'end':end,'dims':[int(d)-1 for d in dims.split(',')]})
        yield Record('smd',p.stem,train,x,y,ev,[f'd{j+1}' for j in range(x.shape[1])],[tr,p,lab,ip])

def skab_records(root):
    root=Path(root); normal=root/'anomaly-free/anomaly-free.csv'
    frame=pd.read_csv(normal,sep=';'); cols=[c for c in frame if c not in ['datetime','anomaly','changepoint']]
    train=frame[cols].to_numpy(float)
    for p in sorted(root.glob('*/*.csv')):
        if p==normal: continue
        df=pd.read_csv(p,sep=';'); x=df[cols].to_numpy(float); y=df.anomaly.to_numpy(int)
        yield Record('skab',str(p.relative_to(root)),train,x,y,[],cols,[normal,p])

def tep_array(path):
    x=np.loadtxt(path)
    if x.shape[0]==52 and x.shape[1]!=52:x=x.T
    if x.shape[1]!=52:raise ValueError(f'{path}: expected 52 columns, got {x.shape}')
    return x

def tep_records(root):
    root=Path(root)
    if (root/'TE_process').exists():root=root/'TE_process'
    tr=root/'d00.dat'; train=tep_array(tr)
    cols=[f'xmeas_{j}' for j in range(1,42)]+[f'xmv_{j}' for j in range(1,12)]
    for f in range(22):
        p=root/f'd{f:02d}_te.dat'; x=tep_array(p);y=np.zeros(len(x),int)
        if f:y[160:]=1
        yield Record('tep',f'd{f:02d}',train,x,y,[],cols,[tr,p])

def records(dataset,root):
    return {'smd':smd_records,'skab':skab_records,'tep':tep_records}[dataset](root)
