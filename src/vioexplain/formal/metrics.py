"""No point adjustment, oracle cardinality, or label-selected threshold."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support
from .data import intervals

def detect_metrics(label,score,threshold):
    valid=np.isfinite(score); y=np.asarray(label)[valid];s=np.asarray(score)[valid];p=s>threshold
    pr,re,f,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    ev=intervals(y);delay=[int(np.flatnonzero(p[a:b])[0]) for a,b in ev if p[a:b].any()]
    return {'n_scored':len(y),'n_anomaly':int(y.sum()),'precision':pr,'recall':re,'point_f1':f,
        'auprc':average_precision_score(y,s) if y.any() else np.nan,'auroc':roc_auc_score(y,s) if 0<y.sum()<len(y) else np.nan,
        'normal_fpr':float(p[y==0].mean()) if (y==0).any() else np.nan,
        'event_recall':len(delay)/len(ev) if ev else np.nan,'n_events':len(ev),'n_hit_events':len(delay),
        'detected_event_median_delay':float(np.median(delay)) if delay else np.nan,'threshold':float(threshold)}

def rank_metrics(rank,truth,k):
    truth=set(truth);chosen=set(rank[:k]);hit=len(chosen&truth)
    p=hit/len(chosen) if chosen else 0.;r=hit/len(truth) if truth else 0.
    gains=np.array([int(i in truth) for i in rank[:k]]);discount=1/np.log2(np.arange(len(gains))+2)
    ideal=(1/np.log2(np.arange(min(k,len(truth)))+2)).sum()
    return {'k':k,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'ndcg':float((gains*discount).sum()/ideal) if ideal else 0.,'hit':int(hit>0),'n_true':len(truth),'n_pred':len(chosen),'inter':hit}

def normal_threshold(score,alpha=.01):
    score=np.asarray(score);score=score[np.isfinite(score)]
    if not len(score):raise ValueError('Empty calibration scores')
    # Finite-sample upper order statistic; temporal dependence precludes an IID guarantee.
    k=min(len(score)-1,int(np.ceil((len(score)+1)*(1-alpha)))-1)
    return float(np.sort(score)[k])
