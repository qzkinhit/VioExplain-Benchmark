"""Causal blocked forecasts and empirical normal-only calibration."""
from __future__ import annotations
import numpy as np


def causal_windows(values, context=256, horizon=32):
    values=np.asarray(values,dtype=np.float32)
    starts=np.arange(context,len(values),horizon,dtype=int)
    histories=np.stack([values[s-context:s].T for s in starts])
    return starts,histories


def baseline_forecasts(values, fit_median, context=256, horizon=32):
    values=np.asarray(values,dtype=np.float64)
    starts,histories=causal_windows(values,context,horizon)
    forecasts={name:np.full_like(values,np.nan) for name in
               ['train_median','context_median','persistence','seasonal_naive_32']}
    for s,hist in zip(starts,histories):
        width=min(horizon,len(values)-s)
        forecasts['train_median'][s:s+width]=fit_median
        forecasts['context_median'][s:s+width]=np.median(hist,axis=1)
        forecasts['persistence'][s:s+width]=hist[:,-1]
        forecasts['seasonal_naive_32'][s:s+width]=hist[:,-horizon:].T[:width]
    return forecasts


def residual_scale(residual, scale_mask):
    selected=np.asarray(residual)[scale_mask]
    # Absolute residuals may be exactly zero on constant channels.
    scale=np.nanquantile(selected,.99,axis=0)
    return np.maximum(scale,1e-8)


def score_residual(residual, scale):
    return np.max(np.asarray(residual)/np.asarray(scale),axis=1)


def calibration_threshold(scores, threshold_mask):
    selected=np.asarray(scores)[threshold_mask]
    if not np.all(np.isfinite(selected)):
        raise ValueError('Calibration scores must be finite')
    return float(np.quantile(selected,.99,method='higher'))
