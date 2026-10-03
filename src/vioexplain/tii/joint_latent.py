"""Joint latent-state inference with group-sparse measurement corrections.

The caller supplies normal-only calibrated precisions and regularization.
"""
from __future__ import annotations
import numpy as np


def system_matrices(observed,shadow,A,b,wf,wc,lam=1.0):
    observed=np.asarray(observed,dtype=float);shadow=np.asarray(shadow,dtype=float)
    A=np.asarray(A,dtype=float);b=np.asarray(b,dtype=float)
    wf2=np.asarray(wf,dtype=float)**2;wc2=np.asarray(wc,dtype=float)**2
    H=np.diag(wf2)+lam*(A.T*wc2)@A
    G=(observed-shadow)*wf2+lam*((observed@A.T-b)*wc2)@A
    return H,G


def group_soft_threshold(values,threshold):
    norm=np.linalg.norm(values,axis=-2,keepdims=True)
    return values*np.maximum(0.0,1.0-threshold/np.maximum(norm,1e-300))


def kkt_residual(delta,H,G,tau):
    gradient=delta@H-G
    norm=np.linalg.norm(delta,axis=-2,keepdims=True)
    direction=np.divide(delta,np.maximum(norm,1e-300))
    active=np.squeeze(norm,axis=-2)>1e-9
    active_error=np.linalg.norm(gradient+tau*direction,axis=-2)
    zero_error=np.maximum(np.linalg.norm(gradient,axis=-2)-tau,0)
    return np.where(active,active_error,zero_error)


def solve_group_correction(H,G,tau,max_iter=2000,tolerance=1e-5):
    """FISTA for .5 delta H delta - <G,delta> + tau sum column norms."""
    H=np.asarray(H,float);G=np.asarray(G,float)
    if tau<0:raise ValueError('tau must be nonnegative')
    lipschitz=float(np.linalg.eigvalsh(H).max())
    if lipschitz<=0:return np.zeros_like(G),{'iterations':0,'normalized_kkt':0.,'converged':True}
    delta=np.zeros_like(G);accelerated=delta.copy();momentum=1.
    normalized_kkt=np.inf
    for iteration in range(1,max_iter+1):
        nxt=group_soft_threshold(accelerated-(accelerated@H-G)/lipschitz,tau/lipschitz)
        next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
        accelerated=nxt+(momentum-1)/next_momentum*(nxt-delta)
        delta=nxt;momentum=next_momentum
        if iteration%20==0 or iteration==max_iter:
            normalized_kkt=float(np.max(kkt_residual(delta,H,G,tau))/max(1.,tau))
            if normalized_kkt<=tolerance:break
    return delta,{'iterations':iteration,'normalized_kkt':normalized_kkt,
                  'converged':bool(normalized_kkt<=tolerance),'lipschitz':lipschitz}


def calibrate_tau(normal_gradients,quantile=.95):
    gradients=np.asarray(normal_gradients)
    per_block=np.max(np.linalg.norm(gradients,axis=-2),axis=-1)
    return float(np.quantile(per_block,quantile,method='higher'))
