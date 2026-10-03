"""TEP事件知识与违反匹配。所有估计量仅通过fit调用学习。"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from vioexplain.setcover.models import Constraint, Reason
from vioexplain.setcover.set_covering import MyCover
from vioexplain.mincost import build_instance, greedy_density


def summary_features(windows):
    x=np.asarray(windows,float); t=np.arange(x.shape[1],dtype=float);t-=t.mean()
    slope=np.einsum('ntd,t->nd',x,t)/(t@t)
    return np.stack([x.mean(1),x.std(1),slope],axis=-1)


def statistics_features(windows):
    x=np.asarray(windows,float); centered=x-x.mean(1,keepdims=True)
    lag=(centered[:,:-1]*centered[:,1:]).mean(1)/(centered.var(1)+1e-12)
    return np.concatenate([summary_features(x).reshape(len(x),-1),np.quantile(x,.1,axis=1),np.quantile(x,.9,axis=1),np.diff(x,axis=1).std(1),lag],axis=1)


def interval_iou(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    union=max(a[1],b[1])-min(a[0],b[0])
    return 1.-inter/union if union>0 else 0.


class EventKnowledge:
    def fit(self,windows,labels):
        self.labels=np.asarray(labels); x=np.asarray(windows); f=summary_features(x)
        normal=f[self.labels==0]; self.center=normal.mean(0)
        # Floors reflect variation of the normal time points, not fault values.
        pointscale=np.maximum(x[self.labels==0].reshape(-1,x.shape[-1]).std(0),1e-8)
        floor=np.stack([pointscale/np.sqrt(64),pointscale/np.sqrt(128),pointscale/np.sqrt((np.arange(64)-31.5)@ (np.arange(64)-31.5))],axis=-1)
        self.scale=np.maximum(normal.std(0),floor)
        z=(f-self.center)/self.scale; self.fit_z=z
        self.knowledge={}
        for c in range(1,22):
            rows=z[self.labels==c]; items={}
            for ch in range(x.shape[-1]):
                for kind in range(3):
                    for sign in [-1,1]:
                        a=sign*rows[:,ch,kind]; active=a>3.; freq=float(active.mean())
                        if freq>=.25:
                            vals=a[active];items[(ch,kind,sign)]={'freq':freq,'mandatory':freq>=.75,'interval':(float(max(0,vals.min()-.5)),float(vals.max()+.5))}
            self.knowledge[c]=items
        return self

    def evidence(self,window):
        z=(summary_features(np.asarray(window)[None])[0]-self.center)/self.scale
        return {(ch,kind,sign):(max(0.,float(sign*z[ch,kind]-.5)),float(sign*z[ch,kind]+.5))
                for ch in range(z.shape[0]) for kind in range(3) for sign in [-1,1] if sign*z[ch,kind]>3.}

    def graph(self,evidence,strict=False,omit=()):
        result={}; missing={}
        for c,items in self.knowledge.items():
            if c in omit:continue
            mandatory={k for k,v in items.items() if v['mandatory']}
            # A class without empirical mandatory evidence is not fabricated into a candidate.
            if not mandatory:continue
            missed=len(mandatory-set(evidence))/len(mandatory)
            matches=set(evidence)&set(items)
            if matches and missed<=(0. if strict else .5):
                result[c]=sorted(matches);missing[c]=missed
        return result,missing

    def serializable(self):
        return {'center':self.center.tolist(),'scale':self.scale.tolist(),'events':{str(c):[{'channel':k[0],'kind':['mean','std','slope'][k[1]],'sign':k[2],**v} for k,v in items.items()] for c,items in self.knowledge.items()}}


class TemporalMetric:
    """Per-channel nearest training prototype distance; PCA is shared over channels."""
    def fit(self,features,labels):
        f=np.asarray(features,float);self.labels=np.asarray(labels);self.shape=f.shape[1:]
        # Standardize input coordinates with fit values, retaining zero-variance protection.
        flat=f.reshape(-1,f.shape[-1]);self.scaler=StandardScaler().fit(flat)
        self.pca=PCA(n_components=min(16,flat.shape[1],flat.shape[0]-1),svd_solver='full')
        self.train=self.pca.fit_transform(self.scaler.transform(flat)).reshape(len(f),f.shape[1],-1)
        pair=[]
        for c in range(22):
            a=self.train[self.labels==c]
            if len(a)>1:pair.extend(np.mean((a[1:]-a[:-1])**2,axis=-1).ravel())
        self.distance_scale=max(float(np.median(pair)),1e-8)
        return self

    def distances(self,features):
        f=np.asarray(features,float);shape=f.shape
        p=self.pca.transform(self.scaler.transform(f.reshape(-1,shape[-1]))).reshape(shape[0],shape[1],-1)
        result=np.zeros((len(f),22,shape[1]))
        for c in range(22):
            proto=self.train[self.labels==c]
            d=((p[:,None]-proto[None])**2).mean(-1).min(1)/self.distance_scale
            result[:,c]=d/(1.+d)
        return result


def raw_temporal(windows,knowledge):
    # Absolute normal-scale channel trajectories are available to the no-FM temporal control.
    scale=np.maximum(knowledge.scale[:,0]*np.sqrt(64),1e-8)
    return ((np.asarray(windows)-knowledge.center[:,0])/scale).transpose(0,2,1)


def solve_event(knowledge,window,temporal_dist=None,weight=0.,solver='minexplain',strict=False,omit=()):
    evidence=knowledge.evidence(window);graph,missing=knowledge.graph(evidence,strict,omit)
    names=sorted(evidence);matched=sorted(set().union(*map(set,graph.values()))) if graph else []
    uncovered=sorted(set(names)-set(matched)); candidates=sorted(graph)
    if not matched:
        return {'selected':[],'rank':[0] if not names else [-1],'n_evidence':len(names),'n_unexplained':len(names),'n_candidates':0,'cost':0.,'missing_mean':0.,'edge_costs':{},'candidate_events':[]}
    edge={}
    for c,keys in graph.items():
        for k in keys:
            raw=interval_iou(evidence[k],knowledge.knowledge[c][k]['interval'])
            td=0. if temporal_dist is None else float(temporal_dist[c,k[0]])
            # Frequency and missing mandatory evidence are explicitly retained in every adapted match.
            reliability=1.-knowledge.knowledge[c][k]['freq']
            edge[k,c]=min(.999999, .8*((1-weight)*raw+weight*td)+.1*missing[c]+.1*reliability)
    if solver=='prototype':
        observations=[Constraint(str(k),*evidence[k]) for k in matched]
        reasons=[Reason(str(c),[Constraint(str(k),*knowledge.knowledge[c][k]['interval']) for k in graph[c]]) for c in candidates]
        cost,selected=MyCover(reasons,observations.copy(),observations.copy(),[])
        selected=[int(r.name) for r in selected]
    elif solver=='aec_adapted':
        # Cost-matrix adaptation, not the original interval solver. In this protocol
        # all constraints are univariate and association rules are empty. The
        # original Select therefore chooses the cheapest candidate among its
        # initial top-two comparisons. HasCovered only stores already-selected
        # events, so its add-if-not-in-H branch cannot add another event here.
        alive=set(matched);available=candidates.copy();selected=[];cost=0.
        initial={key:min(sum(edge[k,c] for k in graph[c]) for c in available if key in graph[c]) for key in matched}
        for key in sorted(matched,key=lambda key:initial[key]):
            if key not in alive:continue
            choices=[c for c in available if key in graph[c]]
            if not choices:continue
            chosen=min(choices,key=lambda c:sum(edge[k,c] for k in graph[c] if k in alive))
            cost+=sum(edge[k,chosen] for k in graph[chosen] if k in alive)
            selected.append(chosen);alive.difference_update(graph[chosen]);available.remove(chosen)
    else:
        indexed={(i,j):edge[k,c] for i,k in enumerate(matched) for j,c in enumerate(candidates) if (k,c) in edge}
        # Fixed opening=1; missing penalty is per edge, not double-counted in opening.
        inst=build_instance(matched,candidates,[1.]*len(candidates),indexed,1.)
        result=greedy_density(inst);cost=result.cost;selected=[candidates[j] for j in result.E]
    selected=list(dict.fromkeys(selected))
    # Solver selection order is the event ranking. Remaining candidates use identical edge mean.
    remaining=sorted(set(candidates)-set(selected),key=lambda c:(np.mean([edge[k,c] for k in graph[c]]),c))
    rank=selected+remaining
    return {'selected':selected,'rank':rank,'n_evidence':len(names),'n_unexplained':len(uncovered),'n_candidates':len(candidates),'cost':float(cost),'missing_mean':float(np.mean(list(missing.values()))),'edge_costs':{str((k,c)):v for (k,c),v in edge.items()},'candidate_events':candidates}
