import numpy as np
from vioexplain.tii.residuals import causal_windows, baseline_forecasts, residual_scale, calibration_threshold

def test_forecast_block_cannot_see_its_targets():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(400,3)); changed=x.copy(); changed[288:320]+=10000
    a=baseline_forecasts(x,np.zeros(3)); b=baseline_forecasts(changed,np.zeros(3))
    for name in a: np.testing.assert_equal(a[name][288:320],b[name][288:320])
    starts,histories=causal_windows(x)
    assert histories.shape==(5,3,256)
    np.testing.assert_equal(histories[1],x[32:288].T.astype(np.float32))

def test_calibration_only_depends_on_designated_normal_rows():
    x=np.arange(100,dtype=float)[:,None]; changed=x.copy(); changed[80:]=1e9
    mask=np.arange(100)<80
    np.testing.assert_equal(residual_scale(x,mask),residual_scale(changed,mask))
    assert calibration_threshold(x[:,0],mask)==calibration_threshold(changed[:,0],mask)
