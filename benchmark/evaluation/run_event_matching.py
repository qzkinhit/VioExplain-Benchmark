"""Frozen Chronos-2 plus event representations. Run only on remote research servers."""
from pathlib import Path
import argparse,hashlib,json,os,pickle,sys,time,traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import tep_array,sha256
from vioexplain.formal.event_matching import EventKnowledge,TemporalMetric,raw_temporal,statistics_features,solve_event
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'

def save_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def source_hashes():
    paths=[Path(__file__),ROOT/'src/vioexplain/formal/event_matching.py',ROOT/'src/vioexplain/formal/data.py',*sorted((ROOT/'src/vioexplain/setcover').glob('*.py')),*sorted((ROOT/'src/vioexplain/mincost').glob('*.py'))]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}

def load(data,split):
    windows=[];rows=[]
    for c in range(22):
        path=data/(f'd{c:02d}_te.dat' if split=='test' else f'd{c:02d}.dat');x=tep_array(path);n=len(x)
        ranges={'fit':(20 if c else 0,int(.6*n)),'validation':(int(.6*n),int(.8*n)),'calibration':(int(.8*n),n),'test':(160 if c else 0,n)}
        start,end=ranges[split]
        for s in range(start,end-63,64):
            windows.append(x[s:s+64]);rows.append({'class':c,'item':f'd{c:02d}','start':s,'end':s+64,'split':split})
    return np.asarray(windows),rows

def embed(x,model,output,split):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4);torch.manual_seed(0)
    pipeline=Chronos2Pipeline.from_pretrained(str(model),device_map='cuda',torch_dtype=torch.float32)
    pipeline.model.eval()
    for p in pipeline.model.parameters():p.requires_grad_(False)
    t=time.time();reps=[]
    for begin in range(0,len(x),8):
        features,_=pipeline.embed(x[begin:begin+8].astype(np.float32).transpose(0,2,1),batch_size=416,context_length=64)
        reps.extend(v.float().mean(1).numpy() for v in features)
        if begin%64==0:print('embedding',split,begin,len(x),flush=True)
    arr=np.asarray(reps,np.float32);np.savez_compressed(output/f'embeddings_{split}.npz',features=arr)
    save_json(output/f'embeddings_{split}_manifest.json',{'shape':list(arr.shape),'seconds':time.time()-t,'pooling':'mean of all encoder tokens per channel including special tokens','cross_variates':True,'model_revision':REVISION,'model_sha256':sha256(model/'model.safetensors'),'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'cache_sha256':sha256(output/f'embeddings_{split}.npz')})
    del pipeline
    torch.cuda.empty_cache()
    return arr

def metric_truth(rows):return np.array([r['class'] for r in rows])

def predict_event(k,x,rows,config,dist=None,omit=()):
    result=[]
    for i,(window,row) in enumerate(zip(x,rows)):
        pred=solve_event(k,window,None if dist is None else dist[i],config['weight'],config['solver'],config['strict'],omit)
        result.append({**row,**pred,'predicted_class':pred['rank'][0] if pred['rank'] else -1,'method':config['name']})
    return result

def scores(rows):
    y=np.array([r['class'] for r in rows]);p=np.array([r['predicted_class'] for r in rows]);tp=fp=fn=0;exact=[]
    for r in rows:
        expected={r['class']} if r['class'] else set(); chosen=set(r['selected']);tp+=len(expected&chosen);fp+=len(chosen-expected);fn+=len(expected-chosen);exact.append(expected==chosen)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {'n':len(rows),'top1':accuracy_score(y,p),'top3':float(np.mean([r['class'] in r['rank'][:3] for r in rows])),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=list(range(22)),average='macro',zero_division=0),'explanation_precision':precision,'explanation_recall':recall,'explanation_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'explanation_exact_match':float(np.mean(exact)),'normal_misexplanation':float(np.mean([bool(r['selected']) for r in rows if r['class']==0])) if 0 in y else None,'mean_explanations':float(np.mean([len(r['selected']) for r in rows])),'mean_unexplained_fraction':float(np.mean([r['n_unexplained']/max(1,r['n_evidence']) for r in rows]))}

def classifier_rows(model,x,rows,name):
    decision=model.decision_function(x);classes=model.classes_;order=np.argsort(-decision,axis=1);out=[]
    for r,ranking in zip(rows,order):
        rank=classes[ranking].astype(int).tolist();p=rank[0];out.append({**r,'method':name,'predicted_class':p,'rank':rank,'selected':[p] if p else [],'n_evidence':0,'n_unexplained':0,'n_candidates':22,'cost':0.,'missing_mean':0.,'candidate_events':classes.astype(int).tolist(),'edge_costs':{}})
    return out

def trajectory_rows(rows):
    out=[]
    for c in range(22):
        group=[r for r in rows if r['class']==c];votes={p:sum(r['predicted_class']==p for r in group) for p in range(-1,22)}
        rank=sorted(votes,key=lambda p:(-votes[p],p));p=rank[0]
        selected=[h for h in range(1,22) if sum(h in r['selected'] for r in group)>=len(group)/2]
        out.append({'class':c,'method':group[0]['method'],'item':f'd{c:02d}','start':0,'end':0,'split':'test_trajectory','predicted_class':p,'rank':rank,'selected':selected,'n_evidence':sum(r['n_evidence'] for r in group),'n_unexplained':sum(r['n_unexplained'] for r in group),'window_top1':float(np.mean([r['predicted_class']==c for r in group]))})
    return out

def csv_rows(path,rows):
    clean=[]
    for row in rows:
        clean.append({k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items() if k!='edge_costs'})
    pd.DataFrame(clean).to_csv(path,index=False)

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);data=Path(a.data);model=Path(a.model)
    if a.stage=='develop':
        if (out/'LOCK.json').exists():raise RuntimeError('Existing lock; cannot repeat development')
        X,tr=load(data,'fit');V,vr=load(data,'validation');C,cr=load(data,'calibration');y=metric_truth(tr)
        pd.DataFrame(tr+vr+cr).to_csv(out/'training_split_windows.csv',index=False)
        k=EventKnowledge().fit(X,y);save_json(out/'knowledge.json',k.serializable())
        E=embed(np.concatenate([X,V,C]),model,out,'development');n=len(X);nv=len(V)
        fm=TemporalMetric().fit(E[:n],y);raw=TemporalMetric().fit(raw_temporal(X,k),y)
        distances={'raw_temporal':raw.distances(raw_temporal(V,k)),'frozen_chronos2':fm.distances(E[n:n+nv]),'interval':None}
        calibration_distances={'raw_temporal':raw.distances(raw_temporal(C,k)),'frozen_chronos2':fm.distances(E[n+nv:]),'interval':None}
        configs=[{'name':'aec_prototype_strict','solver':'prototype','strict':True,'weight':0.,'metric':'interval'},{'name':'aec_prototype_relaxed','solver':'prototype','strict':False,'weight':0.,'metric':'interval'}]
        development=[]
        for metric in ['interval','raw_temporal','frozen_chronos2']:
            for solver in ['aec_adapted','minexplain']:
                best=None
                for weight in ([0.] if metric=='interval' else [0.,.25,.5]):
                    cfg={'name':f'{solver}_{metric}','solver':solver,'strict':False,'weight':weight,'metric':metric}
                    s=scores(predict_event(k,V,vr,cfg,distances[metric]));development.append({**cfg,**s})
                    key=(s['macro_f1'],s['explanation_f1'],-weight)
                    if best is None or key>best[0]:best=(key,cfg)
                configs.append(best[1])
        for cfg in configs[:2]:development.append({**cfg,**scores(predict_event(k,V,vr,cfg))})
        pd.DataFrame(development).to_csv(out/'validation_grid.csv',index=False)
        baselines={};svm_dev=[]
        for kind in ['statistics','window_pca']:
            x=statistics_features(X) if kind=='statistics' else X.reshape(len(X),-1)
            v=statistics_features(V) if kind=='statistics' else V.reshape(len(V),-1)
            best=None
            for c in [1.,10.]:
                steps=[StandardScaler()]
                if kind=='window_pca':steps.append(PCA(n_components=min(64,len(X)-1),svd_solver='full'))
                steps.append(SVC(C=c,kernel='rbf',gamma='scale',decision_function_shape='ovr'))
                estimator=make_pipeline(*steps).fit(x,y);s=balanced_accuracy_score(metric_truth(vr),estimator.predict(v));svm_dev.append({'kind':kind,'C':c,'balanced_accuracy':s})
                if best is None or s>best[0]:best=(s,c,estimator)
            baselines[kind]=best[2]
        pd.DataFrame(svm_dev).to_csv(out/'svm_validation_grid.csv',index=False)
        # Fixed omitted class F21. Threshold from known calibration only, never from test.
        reject={}
        for cfg in configs:
            known=[i for i,r in enumerate(cr) if r['class']!=21];dist=calibration_distances[cfg['metric']]
            pred=predict_event(k,C[known],[cr[i] for i in known],cfg,None if dist is None else dist[known],omit=(21,))
            u=[r['n_unexplained']/max(1,r['n_evidence']) for r in pred];reject[cfg['name']]=float(np.quantile(u,.95,method='higher'))
        lock={'configs':configs,'unknown_protocol':{'omitted_class':21,'score':'unexplained evidence fraction','threshold_quantile':.95,'thresholds':reject,'threshold_source':'known calibration classes 0..20'},'source_hashes':source_hashes(),'model_revision':REVISION,'model_sha256':sha256(model/'model.safetensors'),'data_sha256':{p.name:sha256(p) for p in sorted(data.glob('*.dat'))},'protocol_sha256':sha256(Path(a.protocol)),'fit_windows':len(X),'validation_windows':len(V),'calibration_windows':len(C),'test_reads_before_lock':False,'randomness':'deterministic full SVD and frozen encoder; no random fitting seeds','locked_at_unix':time.time()}
        with open(out/'fitted.pkl','wb') as f:pickle.dump({'knowledge':k,'fm':fm,'raw':raw,'baselines':baselines},f)
        lock['fitted_sha256']=sha256(out/'fitted.pkl');save_json(out/'LOCK.json',lock)
        print('DEVELOPMENT LOCKED',json.dumps(configs),flush=True)
    else:
        if (out/'TEST_STARTED.json').exists():raise RuntimeError('Test already started; no repeat test allowed')
        lock=json.loads((out/'LOCK.json').read_text())
        if source_hashes()!=lock['source_hashes']:raise RuntimeError('Source changed after lock')
        if sha256(model/'model.safetensors')!=lock['model_sha256']:raise RuntimeError('Model changed')
        if sha256(out/'fitted.pkl')!=lock['fitted_sha256']:raise RuntimeError('Fit changed')
        if {p.name:sha256(p) for p in sorted(data.glob('*.dat'))}!=lock['data_sha256']:raise RuntimeError('Data changed')
        save_json(out/'TEST_STARTED.json',{'time':time.time(),'lock_sha256':sha256(out/'LOCK.json')})
        with open(out/'fitted.pkl','rb') as f:fit=pickle.load(f)
        k=fit['knowledge'];T,rows=load(data,'test');pd.DataFrame(rows).to_csv(out/'test_split_windows.csv',index=False)
        E=embed(T,model,out,'test');distances={'interval':None,'raw_temporal':fit['raw'].distances(raw_temporal(T,k)),'frozen_chronos2':fit['fm'].distances(E)}
        predictions=[];unknown=[]
        for cfg in lock['configs']:
            pred=predict_event(k,T,rows,cfg,distances[cfg['metric']]);predictions.extend(pred)
            unknown_pred=predict_event(k,T,rows,cfg,distances[cfg['metric']],omit=(21,));thr=lock['unknown_protocol']['thresholds'][cfg['name']]
            for r in unknown_pred:
                score=r['n_unexplained']/max(1,r['n_evidence']);unknown.append({'method':cfg['name'],'item':r['item'],'class':r['class'],'start':r['start'],'score':score,'threshold':thr,'reject':score>thr,'unknown_truth':r['class']==21})
            print('predicted',cfg['name'],flush=True)
        for kind,estimator in fit['baselines'].items():
            x=statistics_features(T) if kind=='statistics' else T.reshape(len(T),-1);predictions.extend(classifier_rows(estimator,x,rows,f'svm_{kind}'))
        # Save prediction artifacts before scoring labels. Class labels are joined below for metrics.
        save_json(out/'predictions_unscored.json',[{kk:vv for kk,vv in r.items() if kk not in ['class','edge_costs']} for r in predictions])
        csv_rows(out/'predictions_scored.csv',predictions);summary=[];trajectories=[];per_class=[]
        for name in dict.fromkeys(r['method'] for r in predictions):
            pred=[r for r in predictions if r['method']==name];traj=trajectory_rows(pred);trajectories.extend(traj)
            for level,rs in [('window',pred),('trajectory',traj)]:summary.append({'method':name,'level':level,**scores(rs)})
            for c in range(22):per_class.append({'method':name,'class':c,**scores([r for r in pred if r['class']==c])})
        pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(per_class).to_csv(out/'per_class.csv',index=False);csv_rows(out/'per_trajectory.csv',trajectories)
        uf=pd.DataFrame(unknown);uf.to_csv(out/'missing_knowledge_predictions.csv',index=False);us=[]
        for name,g in uf.groupby('method'):
            us.append({'method':name,'known_false_rejection':float(g.loc[~g.unknown_truth,'reject'].mean()),'unknown_recall':float(g.loc[g.unknown_truth,'reject'].mean()),'unknown_auroc':roc_auc_score(g.unknown_truth,g.score),'threshold':float(g.threshold.iloc[0])})
        pd.DataFrame(us).to_csv(out/'missing_knowledge_summary.csv',index=False)
        save_json(out/'COMPLETE.json',{'time':time.time(),'lock_sha256':sha256(out/'LOCK.json'),'test_windows':len(T),'independent_trajectories':22,'methods':len(summary)//2,'limitations':['one train and test trajectory per class','conditional fault interval classification','empirical learned signatures, no physical causality assertion','unknown only held out F21','one normal calibration window does not imply calibrated normal FPR']})
        print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['develop','test'],required=True);ap.add_argument('--data',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);ap.add_argument('--protocol',default=str(ROOT/'benchmark/protocol/event_matching_v1.md'));args=ap.parse_args()
    try:main(args)
    except Exception:
        Path(args.output).mkdir(parents=True,exist_ok=True)
        with open(Path(args.output)/'failures.log','a') as f:traceback.print_exc(file=f)
        raise
