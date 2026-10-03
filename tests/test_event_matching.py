"""Meaningful invariant tests, executed remotely before any formal test prediction."""
import numpy as np
from vioexplain.formal.event_matching import EventKnowledge,interval_iou,solve_event,TemporalMetric

def toy_knowledge():
    k=EventKnowledge();k.center=np.zeros((2,3));k.scale=np.ones((2,3));k.knowledge={1:{(0,0,1):{'freq':1.,'mandatory':True,'interval':(3.5,4.5)}},2:{(1,0,1):{'freq':1.,'mandatory':True,'interval':(3.5,4.5)}}};return k

def test_interval_distance_valid():
    assert interval_iou((1,2),(1,2))==0
    assert interval_iou((1,2),(3,4))==1
    assert 0<interval_iou((1,3),(2,4))<1

def test_same_graph_temporal_changes_cost_not_candidates():
    k=toy_knowledge();x=np.ones((64,2))*4
    for solver in ['prototype','aec_adapted','minexplain']:
        a=solve_event(k,x,np.zeros((22,2)),0,solver)
        b=solve_event(k,x,np.ones((22,2)),.5,solver)
        assert a['candidate_events']==b['candidate_events']==[1,2]
        assert a['n_unexplained']==b['n_unexplained']==0
        assert set(a['selected'])=={1,2}

def test_missing_knowledge_evidence_is_retained():
    k=toy_knowledge();x=np.ones((64,2))*4
    for solver in ['prototype','aec_adapted','minexplain']:
        a=solve_event(k,x,solver=solver,omit=(2,))
        assert a['selected']==[1]
        assert a['n_evidence']==2 and a['n_unexplained']==1

def test_normal_empty_set_and_no_fabricated_mandatory():
    k=toy_knowledge();assert solve_event(k,np.zeros((64,2)))['selected']==[]
    k.knowledge[1][(0,0,1)]['mandatory']=False
    assert 1 not in k.graph({(0,0,1):(3.5,4.5)})[0]

def test_temporal_fit_does_not_change_at_inference():
    rng=np.random.default_rng(0);x=rng.normal(size=(44,2,4));labels=np.repeat(np.arange(22),2)
    metric=TemporalMetric().fit(x,labels);before=metric.pca.components_.copy()
    d=metric.distances(x[:1]*100)
    assert d.shape==(1,22,2) and np.all((d>=0)&(d<=1))
    assert np.array_equal(before,metric.pca.components_)
