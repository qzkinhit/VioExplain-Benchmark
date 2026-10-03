"""Official TreeSHAP attribution of a fitted Isolation Forest; conditional ranking."""
from pathlib import Path
import argparse,sys,json,time,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import shap
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import smd_records,sha256
from vioexplain.formal.metrics import rank_metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite')
    manifest={'method':'TreeSHAP-IsolationForest','official_software':'https://github.com/shap/shap','shap_version':shap.__version__,'seeds':[0,1,2],'fit_fraction':.6,'n_trees':200,'attribution':'TreeExplainer(tree_path_dependent), mean absolute contributions of <=30 evenly spaced event points','candidate_dimensions':'all38','source_sha256':sha256(__file__),'target_explained':'Isolation Forest expected path length; not a causal model','data':{}}
    rows=[];t0=time.time()
    for rec in smd_records(a.data):
        end=int(.6*len(rec.train));train=rec.train[:end];manifest['data'][rec.item]={str(p):sha256(p) for p in rec.sources}
        for seed in [0,1,2]:
            model=IsolationForest(n_estimators=200,random_state=seed,n_jobs=4).fit(train);explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent')
            for ei,ev in enumerate(rec.events):
                event=rec.test[ev['start']:ev['end']];idx=np.linspace(0,len(event)-1,min(len(event),30),dtype=int);sv=np.asarray(explainer.shap_values(event[idx],check_additivity=True));rank=np.argsort(-np.abs(sv).mean(axis=0),kind='stable').tolist()
                for k in [1,3,5]:rows.append({'dataset':'smd','split':'group1_development' if rec.item.startswith('machine-1-') else 'group23_evaluation','item':rec.item,'method':'treeshap_iforest','seed':seed,'event_id':ei,'start':ev['start'],'end':ev['end'],'rank':json.dumps(rank),'truth':json.dumps(ev['dims']),**rank_metrics(rank,ev['dims'],k)})
            print(rec.item,seed,'done',flush=True)
        pd.DataFrame(rows).to_csv(out/'conditional_explanation_per_event.csv',index=False);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(rows).groupby(['split','method','seed','k'])[['precision','recall','f1','ndcg','hit']].mean().reset_index().to_csv(out/'explanation_summary.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps({'n_records':28,'rows':len(rows),'seconds':time.time()-t0},indent=2))
if __name__=='__main__':main()
