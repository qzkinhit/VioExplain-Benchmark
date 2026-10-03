import numpy as np
from vioexplain.formal.metrics import normal_threshold,rank_metrics,detect_metrics
from vioexplain.formal.data import intervals,tep_array

def test_fixed_k_is_not_oracle_cardinality():
    one=rank_metrics([0,1,2,3],{0},3);three=rank_metrics([0,1,2,3],{0,1,2},3)
    assert one['n_pred']==three['n_pred']==3

def test_no_point_adjustment():
    row=detect_metrics(np.array([0,1,1,1,0]),np.array([0.,2.,0.,0.,0.]),1.)
    assert row['recall']==1/3 and row['event_recall']==1

def test_threshold_depends_only_on_normal_scores():
    normal=np.arange(100.)
    assert normal_threshold(normal)==99.
    assert normal_threshold(normal,.1)==90.

def test_constant_and_empty_event_sets():
    assert intervals([0,0,0])==[]
    assert intervals([1,1,0,1])==[(0,2),(3,4)]

def test_transposed_tep(tmp_path):
    path=tmp_path/'d00.dat';np.savetxt(path,np.arange(52*60).reshape(52,60))
    assert tep_array(path).shape==(60,52)
