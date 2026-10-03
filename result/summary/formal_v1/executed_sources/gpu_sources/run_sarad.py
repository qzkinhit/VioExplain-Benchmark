"""Official SARAD network/loss with chronological, machine-bounded evaluation."""
from pathlib import Path
from typing import Optional
import argparse,ast,copy,json,sys,time,math,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Subset
import einops
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from vioexplain.formal.data import smd_records,sha256
from vioexplain.formal.metrics import detect_metrics,rank_metrics,normal_threshold
EXPECTED={'src/models/components/sar73.py':'d3278e4e326dd487ee525ae63ed0748322d01473a2ed6975df598ed81e439e53','src/models/components/sar76.py':'c49cce068c7ae3871fd33350f6579a39250d0264541d6be920dd0bcd9c98e622','src/models/sar76_module.py':'89548dc79dbdca4b3cd3035023248245ee3d82cde76ae02c54c2a12564bf1d59','src/data/smd_datamodule.py':'71dc2cc4bebd597c4e3dcd6cc50f971cc7d7687af7ef77498046a743f0da66dc'}

def official_class(vendor):
    namespace={'torch':torch,'nn':nn,'Tensor':torch.Tensor,'math':math,'np':np,'einops':einops,'Optional':Optional}
    for filename,digest in EXPECTED.items():
        if sha256(Path(vendor)/filename)!=digest:raise ValueError('SARAD vendor does not match locked revision '+filename)
    for file in ['src/models/components/sar73.py','src/models/components/sar76.py']:
        tree=ast.parse((Path(vendor)/file).read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(Path(vendor)/file),'exec'),namespace)
    return namespace['SAR76']

class Windows(Dataset):
    def __init__(self,arrays,segments,window=100):
        self.arrays=arrays;self.segments=segments;self.window=window
        self.ends=np.cumsum([b-a for a,b in segments]);self.starts=np.r_[0,self.ends[:-1]]
    def __len__(self):return int(self.ends[-1])
    def __getitem__(self,i):
        j=int(np.searchsorted(self.ends,i,side='right'));pos=int(self.segments[j][0]+i-self.starts[j]);x=self.arrays[j];start=max(0,pos-self.window+1);window=x[start:pos+1]
        if len(window)<self.window:window=np.concatenate([np.repeat(x[:1],self.window-len(window),axis=0),window])
        return torch.from_numpy(window)

def raw_scores(model,dataset,batch=128):
    model.eval();recon=[];det=[]
    with torch.no_grad():
        for x in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
            x=x.cuda();xh,s,q,qh=model(x)
            recon.append(((xh-x)**2).mean(dim=1).cpu().numpy())
            det.append(((q-qh)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(recon),np.concatenate(det)

def standardized(raw,scales):
    r,d=raw
    score=(r.mean(1)-scales['recon_avg'])/scales['recon_std']+(d.mean(1)-scales['detec_avg'])/scales['detec_std']
    diagnosis=.5*(r-scales['rdiag_avg'])/scales['rdiag_std']+.5*(d-scales['ddiag_avg'])/scales['ddiag_std']
    return score,diagnosis

def get_scales(r,d):
    s={'recon_avg':r.mean(1).mean(),'recon_std':r.mean(1).std(),'detec_avg':d.mean(1).mean(),'detec_std':d.mean(1).std(),'rdiag_avg':r.mean(0),'rdiag_std':r.std(0),'ddiag_avg':d.mean(0),'ddiag_std':d.std(0)}
    if any(np.any(~np.isfinite(v)) for v in s.values()) or any(np.any(s[k]<=0) for k in ['recon_std','detec_std','rdiag_std','ddiag_std']):raise ValueError('Undefined official SARAD score normalization')
    return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--vendor',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium');klass=official_class(a.vendor);recs=list(smd_records(a.data));fit=[(0,int(.6*len(r.train))) for r in recs];scalep=[(int(.6*len(r.train)),int(.8*len(r.train))) for r in recs];cal=[(int(.8*len(r.train)),len(r.train)) for r in recs]
    scaler=StandardScaler().fit(np.concatenate([r.train[:end] for r,(start,end) in zip(recs,fit)]));tr=[scaler.transform(r.train).astype(np.float32) for r in recs]
    trainset=Windows(tr,fit);scaleset=Windows(tr,scalep);rows=[];erows=[];t0=time.time()
    manifest={'official_repo':'https://github.com/daidahao/SARAD','official_commit':'24854d9723b4eed31b547344061671c08fbfb3e2','license':'MIT','model':'SAR76 official default','epochs':a.epochs,'seed_list':a.seeds,'shared_model_across_28_machines':True,'sampling_fraction':.1,'window':100,'batch':128,'model_size':512,'layers':3,'heads':8,'patches':2,'detector_size':64,'dropout':.1,'detector_loss_weight':100.,'lr':.0007,'scheduler':'StepLR(1,0.5)','normal_split':[.6,.2,.2],'threshold_alpha':.01,'dtype':'float32','matmul_precision':'medium','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'source_sha256':sha256(__file__),'official_files_sha256':EXPECTED,'n_fit_positions':len(trainset),'data':{r.item:{str(p):sha256(p) for p in r.sources} for r in recs},'adaptations':['Official unchanged SAR73/SAR76 class AST extraction removes Lightning/Hydra/logging import requirements only','Official 10 percent seeded random training subset of pooled data preserved','Per-machine chronological first60 fit, middle20 score scale, final20 threshold replace original80/20 and val_dataloader erroneously selecting data_train','Windows stop at current timestamp and never cross machine boundaries; original concatenation can cross a machine boundary','Global StandardScaler fit exclusively on first60 segments, all test timestamps evaluated without point adjustment','Official summed standardized detection score and equal-weight standardized dimension diagnosis unchanged; normal-only per-machine0.99 thresholds','3 independent shared-model seeds, final epoch as official save_last config; no test model selection']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for seed in a.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=klass(38,100,512,3,8,2,64,.1,False).cuda();opt=torch.optim.Adam(model.parameters(),lr=.0007);sched=torch.optim.lr_scheduler.StepLR(opt,1,.5)
        selected=torch.randperm(len(trainset))[:int(np.ceil(.1*len(trainset)))].tolist();np.save(out/f'seed{seed}_training_indices.npy',np.asarray(selected));loader=DataLoader(Subset(trainset,selected),batch_size=128,shuffle=False,num_workers=0)
        losses=[];start=time.time()
        for epoch in range(a.epochs):
            model.train();sums=[]
            for number,x in enumerate(loader):
                x=x.cuda();xh,assoc,q,qh=model(x);recon=((xh-x)**2).mean();detec=((q-qh)**2).mean();loss=recon+100*detec
                opt.zero_grad();loss.backward();opt.step();sums.append([loss.item(),recon.item(),detec.item()])
                if number%100==0:print('TRAIN',seed,epoch,number,len(loader),'elapsed',round(time.time()-start,2),flush=True)
            sched.step();losses.append(np.mean(sums,axis=0).tolist());print('EPOCH',seed,epoch,losses[-1],flush=True)
        print('NORMALIZATION',seed,flush=True);r,d=raw_scores(model,scaleset);scales=get_scales(r,d);np.savez_compressed(out/f'seed{seed}_normal_scales.npz',**scales)
        checkpoint=out/f'seed{seed}.pt';torch.save({'model':model.state_dict(),'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'losses':losses,'training_indices':selected},checkpoint)
        for j,rec in enumerate(recs):
            cp=Windows([tr[j]],[cal[j]]);normal,_=standardized(raw_scores(model,cp),scales);threshold=normal_threshold(normal)
            test=scaler.transform(rec.test).astype(np.float32);raw=raw_scores(model,Windows([test],[(0,len(test))]));score,diag=standardized(raw,scales)
            np.savez_compressed(out/f'{rec.item}_seed{seed}_scores.npz',score=score,diagnosis=diag)
            row={'dataset':'smd','item':rec.item,'method':'sarad_official','seed':seed,**detect_metrics(rec.label,score,threshold)};rows.append(row)
            for ei,ev in enumerate(rec.events):
                rank=np.argsort(-diag[ev['start']:ev['end']].mean(0),kind='stable').tolist()
                for k in [1,3,5]:erows.append({'dataset':'smd','split':'group1_development' if rec.item.startswith('machine-1-') else 'group23_evaluation','item':rec.item,'method':'sarad_official','seed':seed,'event_id':ei,'start':ev['start'],'end':ev['end'],'rank':json.dumps(rank),'truth':json.dumps(ev['dims']),**rank_metrics(rank,ev['dims'],k)})
            pd.DataFrame(rows).to_csv(out/'detection_per_file.csv',index=False);pd.DataFrame(erows).to_csv(out/'conditional_explanation_per_event.csv',index=False)
            print('TEST',seed,j+1,rec.item,round(time.time()-start,2),flush=True)
        (out/f'seed{seed}_COMPLETE.json').write_text(json.dumps({'seed':seed,'checkpoint_sha256':sha256(checkpoint),'seconds':time.time()-start,'losses':losses},indent=2))
    pd.DataFrame(rows).groupby(['method','seed'])[['point_f1','auprc','auroc','normal_fpr','event_recall']].mean().reset_index().to_csv(out/'detection_summary.csv',index=False)
    pd.DataFrame(erows).groupby(['split','method','seed','k'])[['precision','recall','f1','ndcg','hit']].mean().reset_index().to_csv(out/'explanation_summary.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps({'n_records':28,'n_shared_model_seeds':3,'n_runs':len(rows),'seconds':time.time()-t0,'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20},indent=2))
if __name__=='__main__':main()
