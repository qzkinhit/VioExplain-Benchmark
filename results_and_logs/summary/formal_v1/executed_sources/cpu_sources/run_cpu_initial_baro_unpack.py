"""Full official splits, statistical detection and event-conditional dimension ranking."""
from pathlib import Path
import argparse,json,sys,time,platform,importlib.metadata,traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import records,sha256
from vioexplain.formal.metrics import detect_metrics,rank_metrics,normal_threshold

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['smd','skab','tep'],required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--baro-source');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Refusing to overwrite completed run')
    baro=None
    if a.baro_source:
        # Import the untouched official module. It imports baro.utility, so vendor root is explicit.
        sys.path.insert(0,a.baro_source)
        from baro.root_cause_analysis import robust_scorer
        baro=robust_scorer
    manifest={'dataset':a.dataset,'normal_split':[.6,.2,.2],'threshold_alpha':.01,'seeds':[0,1,2],
        'python':sys.version,'platform':platform.platform(),'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),*list((ROOT/'src/vioexplain/formal').glob('*.py'))]},'data':{},'source':'official releases','label_access':'evaluation only; SMD event boundaries are given for conditional diagnosis',
        'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn']}}
    rows=[];erows=[];fail=[];t0=time.time()
    cached=None
    for rec in records(a.dataset,a.data):
        tag=rec.item.replace('/','__').replace('.csv',''); n=len(rec.train);fit_end=int(.6*n);scale_end=int(.8*n)
        manifest['data'][rec.item]={'files':{str(p):sha256(p) for p in rec.sources},'train_shape':list(rec.train.shape),'test_shape':list(rec.test.shape),'fit_end':fit_end,'scale_end':scale_end}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        train=rec.train; x=rec.test
        med=np.median(train[:fit_end],axis=0);iqr=np.quantile(train[:fit_end],.75,axis=0)-np.quantile(train[:fit_end],.25,axis=0);iqr=np.maximum(iqr,1e-6)
        tr=(train-med)/iqr;xt=(x-med)/iqr
        score_arrays={};ranks={};thresholds={};timings={}
        start=time.time()
        pca=PCA(n_components=.95,svd_solver='full').fit(tr[:fit_end])
        for name,cal,test in [('robust_z',np.abs(tr),np.abs(xt)),('pca_q',(tr-pca.inverse_transform(pca.transform(tr)))**2,(xt-pca.inverse_transform(pca.transform(xt)))**2)]:
            score_cal=cal.max(axis=1) if name=='robust_z' else cal.sum(axis=1)
            score=test.max(axis=1) if name=='robust_z' else test.sum(axis=1)
            thresholds[name]=normal_threshold(score_cal[scale_end:]);score_arrays[name]=score;ranks[name]=test;timings[name]=time.time()-start
        for seed in [0,1,2]:
            start=time.time();name=f'iforest_seed{seed}';model=IsolationForest(n_estimators=200,random_state=seed,n_jobs=4).fit(tr[:fit_end])
            thresholds[name]=normal_threshold(-model.score_samples(tr[scale_end:]));score_arrays[name]=-model.score_samples(xt);timings[name]=time.time()-start
        np.savez_compressed(out/f'{tag}_scores.npz',**score_arrays)
        (out/f'{tag}_thresholds.json').write_text(json.dumps(thresholds,indent=2))
        for name,score in score_arrays.items():
            seed=int(name[-1]) if name.startswith('iforest') else -1;method='iforest' if name.startswith('iforest') else name
            rows.append({'dataset':a.dataset,'item':rec.item,'method':method,'seed':seed,'seconds':timings[name],**detect_metrics(rec.label,score,thresholds[name])})
        if a.dataset=='smd':
            for ei,ev in enumerate(rec.events):
                s,e=ev['start'],ev['end']; event=x[s:e]
                ranking={name:np.argsort(-mat[s:e].mean(axis=0),kind='stable').tolist() for name,mat in ranks.items()}
                if baro is not None:
                    width=e-s;ctx=x[max(0,s-width):s]
                    if len(ctx)==0:ctx=train[max(0,fit_end-width):fit_end]
                    frame=pd.DataFrame(np.vstack([ctx,event]),columns=rec.columns);frame.insert(0,'time',np.arange(len(frame)))
                    try:
                        result=baro(frame,anomalies=[len(ctx)])
                        ranking['baro_robust_scorer']=[rec.columns.index(c) for c,v in result['ranks']]
                    except Exception as exc:
                        fail.append({'item':rec.item,'event':ei,'method':'baro_robust_scorer','error':repr(exc)})
                for name,rank in ranking.items():
                    for k in [1,3,5]:
                        erows.append({'dataset':'smd','split':'group1_development' if rec.item.startswith('machine-1-') else 'group23_evaluation','item':rec.item,'event_id':ei,'start':s,'end':e,'method':name,'seed':-1,'rank':json.dumps(rank),'truth':json.dumps(ev['dims']),**rank_metrics(rank,ev['dims'],k)})
        pd.DataFrame(rows).to_csv(out/'detection_per_file.csv',index=False);pd.DataFrame(erows).to_csv(out/'conditional_explanation_per_event.csv',index=False)
        (out/'failures.json').write_text(json.dumps(fail,indent=2));print(rec.item,'done',round(time.time()-t0,2),flush=True)
    df=pd.DataFrame(rows);df.groupby(['dataset','method','seed'])[['point_f1','auprc','auroc','normal_fpr','event_recall']].mean().reset_index().to_csv(out/'detection_summary.csv',index=False)
    if erows:pd.DataFrame(erows).groupby(['split','method','k'])[['precision','recall','f1','ndcg','hit']].mean().reset_index().to_csv(out/'explanation_summary.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps({'seconds':time.time()-t0,'n_records':len(manifest['data']),'failed_events':len(fail)},indent=2))
if __name__=='__main__':main()
