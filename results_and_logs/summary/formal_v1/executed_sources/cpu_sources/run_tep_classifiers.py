"""TEP conditional event classification with official fault IDs and locked windows."""
from pathlib import Path
import argparse,json,sys,time,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import tep_array,sha256

def features(x):
    grid=np.arange(len(x),dtype=float);grid-=grid.mean()
    slope=(grid[:,None]*x).sum(axis=0)/(grid*grid).sum()
    return np.concatenate([x.mean(axis=0),x.std(axis=0),slope])

def samples(x,c,start,end,split):
    return [{'class':c,'start':s,'end':s+64,'split':split,'x':features(x[s:s+64])} for s in range(start,end-63,64)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();data=Path(a.data);data=data/'TE_process' if (data/'TE_process').exists() else data;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite')
    train=[];val=[];cal=[];test=[];manifest={'source_sha256':sha256(__file__),'window':64,'stride':64,'train_warmup_exclusion':20,'test_conditional_onset':160,'split':[.6,.2,.2],'normal_class':0,'fault_classes':list(range(1,22)),'features':'64-point per-channel mean, standard deviation(ddof0), least-squares slope; StandardScaler fit only labeled training windows','identity':'supervised conditional fault classification; not end-to-end fault onset detection','files':{},'sklearn':importlib.metadata.version('scikit-learn'),'config':{'rf':{'trees':500,'min_samples_leaf':1,'seeds':[0,1,2]},'svm':{'C':[.1,1.,10.,100.],'gamma':['scale',.001,.01],'selection':'validation balanced_accuracy then smaller C and gamma order'},'lda':{'solver':'lsqr','shrinkage':'auto'}}}
    t0=time.time()
    for c in range(22):
        p=data/f'd{c:02d}.dat';q=data/f'd{c:02d}_te.dat';tr=tep_array(p);te=tep_array(q);n=len(tr);b=int(.6*n);e=int(.8*n)
        train.extend(samples(tr,c,20 if c else 0,b,'fit'));val.extend(samples(tr,c,b,e,'development'));cal.extend(samples(tr,c,e,n,'calibration'));test.extend(samples(te,c,160 if c else 0,len(te),'test'))
        manifest['files'][p.name]=sha256(p);manifest['files'][q.name]=sha256(q)
    X=np.stack([r['x'] for r in train]);y=np.array([r['class'] for r in train]);V=np.stack([r['x'] for r in val]);vy=np.array([r['class'] for r in val]);T=np.stack([r['x'] for r in test])
    manifest['counts']={name:len(rows) for name,rows in [('fit',train),('development',val),('calibration',cal),('test',test)]}
    pd.DataFrame([{k:v for k,v in row.items() if k!='x'} for row in [*train,*val,*cal,*test]]).to_csv(out/'split_windows.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    models=[];dev=[];best=None
    for C in [.1,1.,10.,100.]:
        for gamma in ['scale',.001,.01]:
            model=make_pipeline(StandardScaler(),SVC(C=C,gamma=gamma,kernel='rbf'))
            model.fit(X,y);score=balanced_accuracy_score(vy,model.predict(V));dev.append({'C':C,'gamma':gamma,'validation_balanced_accuracy':score})
            if best is None or score>best[0]:best=(score,model,C,gamma)
    pd.DataFrame(dev).to_csv(out/'svm_development.csv',index=False);models.append(('rbf_svm',-1,best[1],{'C':best[2],'gamma':best[3]}))
    lda=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(X,y);models.append(('shrinkage_lda',-1,lda,{}))
    for seed in [0,1,2]:
        rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=1,random_state=seed,n_jobs=4).fit(X,y);models.append(('random_forest',seed,rf,{}))
    predictions=[]
    for name,seed,model,config in models:
        pred=model.predict(T)
        for row,p in zip(test,pred):predictions.append({'method':name,'seed':seed,'item':f"d{row['class']:02d}",'start':row['start'],'end':row['end'],'predicted_class':int(p)})
        print(name,seed,'predicted',flush=True)
    # Prediction artifacts exist before test labels are joined into the score table.
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    truth={(f"d{r['class']:02d}",r['start']):r['class'] for r in test};frame=pd.DataFrame(predictions);frame['true_class']=[truth[(r.item,r.start)] for r in frame.itertuples()];frame.to_csv(out/'predictions_scored.csv',index=False)
    summary=[];trajectory=[]
    for (method,seed),group in frame.groupby(['method','seed']):
        ytrue=group.true_class.to_numpy();ypred=group.predicted_class.to_numpy();summary.append({'method':method,'seed':seed,'level':'window','n':len(group),'accuracy':accuracy_score(ytrue,ypred),'balanced_accuracy':balanced_accuracy_score(ytrue,ypred),'macro_f1':f1_score(ytrue,ypred,average='macro',zero_division=0)})
        traj=[]
        for item,g in group.groupby('item'):
            counts=np.bincount(g.predicted_class,minlength=22);p=int(np.argmax(counts));target=int(g.true_class.iloc[0]);traj.append((target,p));trajectory.append({'method':method,'seed':seed,'item':item,'true_class':target,'predicted_class':p,'window_accuracy':float((g.true_class==g.predicted_class).mean()),'n_windows':len(g)})
        ty,tp=np.array(traj).T;summary.append({'method':method,'seed':seed,'level':'trajectory','n':len(traj),'accuracy':accuracy_score(ty,tp),'balanced_accuracy':balanced_accuracy_score(ty,tp),'macro_f1':f1_score(ty,tp,average='macro',zero_division=0)})
        np.savetxt(out/f'{method}_seed{seed}_confusion.csv',confusion_matrix(ytrue,ypred,labels=list(range(22))),fmt='%d',delimiter=',')
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(trajectory).to_csv(out/'per_trajectory.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps({'seconds':time.time()-t0,'runs':5,'n_trajectories':22,'n_windows':len(test)},indent=2))
if __name__=='__main__':main()
