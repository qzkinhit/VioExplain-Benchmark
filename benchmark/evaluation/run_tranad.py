"""Untouched official TranAD network and objective with explicit protocol adaptation.

Only imports are isolated with AST extraction; architecture class source is unchanged.
Training targets and test scores are aligned to the reconstructed current observation.
"""
from pathlib import Path
import argparse,ast,json,sys,time,math,platform
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import records,sha256
from vioexplain.formal.metrics import detect_metrics,rank_metrics,normal_threshold

class LegacyEncoder(nn.Module):
    """PyTorch-1-compatible container for the unchanged custom encoder layer."""
    def __init__(self,layer,num_layers):
        super().__init__();import copy
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    def forward(self,src):
        for layer in self.layers:src=layer(src)
        return src

class LegacyDecoder(nn.Module):
    """No new causal-mask kwargs are passed to the official custom layer."""
    def __init__(self,layer,num_layers):
        super().__init__();import copy
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    def forward(self,tgt,memory):
        for layer in self.layers:tgt=layer(tgt,memory)
        return tgt

def official_class(vendor):
    expected={'src/models.py':'43a30e7caa7c8b28e91b5e875c724219c9c15789da1f6009bb2015fb1f81d84b','src/dlutils.py':'35265d47f3c1c0f999590ab25cf47774cfd2293c99ca3f842419f31dfef750fe'}
    for filename,digest in expected.items():
        if sha256(Path(vendor)/filename)!=digest:
            raise ValueError('TranAD vendor differs from locked 7ffb98d revision: '+filename)
    namespace={'torch':torch,'nn':nn,'F':torch.nn.functional,'math':math,'np':np,
        'TransformerEncoder':LegacyEncoder,'TransformerDecoder':LegacyDecoder,'lr':.0001}
    requests={'src/dlutils.py':['PositionalEncoding','TransformerEncoderLayer','TransformerDecoderLayer'],'src/models.py':['TranAD']}
    for file,names in requests.items():
        tree=ast.parse((Path(vendor)/file).read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
        if len(nodes)!=len(names):raise ValueError('Official code classes changed')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(Path(vendor)/file),'exec'),namespace)
    return namespace['TranAD']

def windows(x):
    # Equivalent to official convert_to_windows at i=t+1: data[i-10:i].
    padded=np.pad(x,((9,0),(0,0)),mode='edge')
    return torch.as_tensor(np.stack([padded[i:i+10] for i in range(len(x))]),dtype=torch.float64)

def predict(model,x,device,batch=1024):
    model.eval();w=windows(x);ans=[]
    with torch.no_grad():
        for d in w.split(batch):
            win=d.to(device).permute(1,0,2);target=win[-1:].clone();z=model(win,target)[1]
            ans.append(((z-target)**2)[0].cpu().numpy())
    return np.concatenate(ans)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['smd','skab','tep']);ap.add_argument('--data',required=True);ap.add_argument('--vendor',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--epochs',type=int,default=5);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite')
    torch.set_num_threads(4);device='cuda';klass=official_class(a.vendor);rows=[];erows=[];t0=time.time()
    manifest={'official_repository':'https://github.com/imperial-qore/TranAD','official_commit':'7ffb98d0c18189cc3d9ab732b4cb0278200a0af0','license':'BSD-3-Clause','torch':torch.__version__,'dtype':'float64','gpu':torch.cuda.get_device_name(0),'normal_split':[.6,.2,.2],'epochs':a.epochs,'seeds':a.seeds,'lr':.0001,'batch_size':128,'optimizer':'AdamW','weight_decay':1e-5,'scheduler':'StepLR(5,0.9)','threshold_alpha':.01,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),*list((ROOT/'src/vioexplain/formal').glob('*.py'))]},'official_files_sha256':{str(p):sha256(p) for p in [Path(a.vendor)/'src/models.py',Path(a.vendor)/'src/dlutils.py',Path(a.vendor)/'main.py']},'data':{},'adaptations':['AST-isolated unchanged network class removes unused DGL and global argparse imports; legacy one-layer containers omit PyTorch2 causal kwargs without changing computations','Train-minmax normalization applied to all partitions without clipping; no separate test fit','Scores timestamped at the observation reconstructed, correcting official one-sample plot roll','Chronological fit/calibration split; fixed held-out-normal 0.99 quantile replaces test-aware POT/PA','Bounded inference batches do not alter eval-mode network outputs','All seeds and all machines, rather than official hardcoded machine-1-1']}
    cache={}
    for rec in records(a.dataset,a.data):
        tag=rec.item.replace('/','__').replace('.csv','');n=len(rec.train);end=int(.6*n);cal=int(.8*n)
        manifest['data'][rec.item]={'files':{str(p):sha256(p) for p in rec.sources},'fit_end':end,'calibration_start':cal,'train_shape':list(rec.train.shape),'test_shape':list(rec.test.shape)}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        lo=rec.train[:end].min(axis=0);hi=rec.train[:end].max(axis=0);scale=np.maximum(hi-lo,1e-8)
        train=(rec.train-lo)/scale;x=(rec.test-lo)/scale
        for seed in a.seeds:
            marker=out/f'{tag}_seed{seed}.json'
            if marker.exists():
                saved=json.loads(marker.read_text());rows.extend(saved['detection']);erows.extend(saved['explanation']);continue
            start=time.time();torch.manual_seed(seed);np.random.seed(seed)
            key=(seed,sha256(rec.sources[0]))
            if key not in cache:
                model=klass(x.shape[1]).double().to(device);opt=torch.optim.AdamW(model.parameters(),lr=model.lr,weight_decay=1e-5);sched=torch.optim.lr_scheduler.StepLR(opt,5,.9)
                w=windows(train[:end]);losses=[]
                for epoch in range(a.epochs):
                    model.train();loss=[]
                    for d in w.split(model.batch):
                        win=d.to(device).permute(1,0,2);target=win[-1:];z1,z2=model(win,target)
                        objective=(((z1-target)**2)/(epoch+1)+(1-1/(epoch+1))*(z2-target)**2).mean()
                        opt.zero_grad();objective.backward();opt.step();loss.append(objective.item())
                    sched.step();losses.append(float(np.mean(loss)))
                    print(rec.item,seed,epoch,'loss',losses[-1],flush=True)
                normal=predict(model,train,device);threshold=normal_threshold(normal[cal:].mean(axis=1))
                checkpoint=out/f'{tag}_seed{seed}.pt';torch.save({'model':model.state_dict(),'lo':lo,'scale':scale,'losses':losses},checkpoint)
                cache[key]=(model,threshold,losses,str(checkpoint),sha256(checkpoint))
                if a.dataset=='smd':cache={key:cache[key]}
            model,threshold,losses,checkpoint,chash=cache[key]
            residual=predict(model,x,device);score=residual.mean(axis=1)
            np.savez_compressed(out/f'{tag}_seed{seed}_scores.npz',score=score,residual=residual)
            row={'dataset':a.dataset,'item':rec.item,'method':'tranad_official','seed':seed,'seconds':time.time()-start,**detect_metrics(rec.label,score,threshold)};ers=[]
            for ei,ev in enumerate(rec.events):
                rank=np.argsort(-residual[ev['start']:ev['end']].mean(axis=0),kind='stable').tolist()
                for k in [1,3,5]:ers.append({'dataset':a.dataset,'split':'group1_development' if rec.item.startswith('machine-1-') else 'group23_evaluation','item':rec.item,'method':'tranad_official','seed':seed,'event_id':ei,'start':ev['start'],'end':ev['end'],'rank':json.dumps(rank),'truth':json.dumps(ev['dims']),**rank_metrics(rank,ev['dims'],k)})
            marker.write_text(json.dumps({'detection':[row],'explanation':ers,'losses':losses,'checkpoint':checkpoint,'checkpoint_sha256':chash},indent=2));rows.append(row);erows.extend(ers)
            pd.DataFrame(rows).to_csv(out/'detection_per_file.csv',index=False);pd.DataFrame(erows).to_csv(out/'conditional_explanation_per_event.csv',index=False)
            print('DONE',rec.item,seed,round(time.time()-start,2),flush=True)
    expected={'smd':28,'skab':34,'tep':22}[a.dataset]
    if len(manifest['data'])!=expected:raise ValueError('Incomplete official dataset')
    pd.DataFrame(rows).groupby(['dataset','method','seed'])[['point_f1','auprc','auroc','normal_fpr','event_recall']].mean().reset_index().to_csv(out/'detection_summary.csv',index=False)
    if erows:pd.DataFrame(erows).groupby(['split','method','seed','k'])[['precision','recall','f1','ndcg','hit']].mean().reset_index().to_csv(out/'explanation_summary.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps({'seconds':time.time()-t0,'n_records':len(manifest['data']),'n_runs':len(rows),'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20},indent=2))
if __name__=='__main__':main()
