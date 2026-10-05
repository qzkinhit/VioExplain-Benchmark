"""Single-thread CPU time per window of VioExplain and the baselines (raw window in, explanation set out).

Usage:  V3_MODE=f0 V3_LOG=log_timing.txt NJ=2 OMP_NUM_THREADS=2 python final_timing.py CNN_NPZ
Models are trained as in final_fuse.py (VioExplain, VioExplain-T, BR-LGBM), final_run.py (RF, XGB, MC-LGBM top-1),
final_base.py (CC-LGBM, FDA, PCA-RBC, AEC, MinExplain) and final_rocket.py (MiniRocket, fitted on 300 windows per
class because the transform and the ridge head have the same size for any training set). The 1D-CNN forward pass uses
an FCN (64, 128, 64 filters, kernels 8, 5, 3) with untrained weights, which has the same cost.
Timing runs under threadpoolctl limits of one thread with n_jobs = 1 for every model, on 2000 random TEV windows and
2000 random pair windows. Each method starts from the raw window (window statistics Fn included where used).
Reported: process CPU time and wall time per window for one batched call, and per-call CPU time for 200 windows
processed one at a time (streaming).
Output: results/final_v1/timing.json
"""
import os, sys, json
SCRIPTS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SCRIPTS)
_src = open(os.path.join(SCRIPTS, 'final_fuse.py')).read()
_key = "evaluate('VioExplain-T', explain_t)"
exec(compile(_src.split(_key)[0], os.path.join(SCRIPTS, 'final_fuse.py'), 'exec'))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import ClassifierChain
from threadpoolctl import threadpool_limits
NOTES = {}

# ---------------- baselines (same code as the final runners) ----------------
rf = RandomForestClassifier(500, n_jobs=NJ, random_state=0).fit(FLAT(XB), YB); log('timing: RF trained')
mc = mk_lgb(); mc.fit(FLAT(XB), YB); log('timing: MC-LGBM trained')
import xgboost as xgb
xm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ,
                       tree_method='hist').fit(FLAT(XB), YB); log('timing: XGB trained')
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(FLAT(XB), YB)
Xml_ = [XB]; Yml_ = [np.zeros((len(YB), K), int)]
Yml_[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = 1
for ev, Zs, Zrest, idx in AUG:
    Y_ = np.zeros((len(Zs), K), int); Y_[:, np.array(ev) - 1] = 1; Xml_.append(Zs); Yml_.append(Y_)
cc = ClassifierChain(mk_lgb(300), order='random', random_state=0).fit(FLAT(np.concatenate(Xml_)), np.concatenate(Yml_))
log('timing: CC-LGBM trained')

# PCA-RBC
XN = FLAT(Z['tr'][0][ok_tr[0]]).astype(np.float64); mu_n = XN.mean(0); Xc_ = XN - mu_n
S_ = np.cov(Xc_.T); ev_, V_ = np.linalg.eigh(S_); o = np.argsort(ev_)[::-1]; ev_ = ev_[o]; V_ = V_[:, o]
l_ = int(np.searchsorted(np.cumsum(ev_) / ev_.sum(), 0.90)) + 1; Pm = V_[:, :l_]; lam = ev_[:l_]
T2m = Pm @ np.diag(1 / lam) @ Pm.T; Qm = np.eye(len(mu_n)) - Pm @ Pm.T
t2 = np.einsum('ij,jk,ik->i', Xc_, T2m, Xc_); spe = np.einsum('ij,jk,ik->i', Xc_, Qm, Xc_)
PHI = T2m / t2.mean() + Qm / spe.mean()
XI = {}
for h in range(1, C):
    g = ok_tr[h]; D = (FLAT(Z['tr'][h][g]) - FLAT(Z['tr'][0][g])).astype(np.float64)
    U, s, _ = np.linalg.svd(D.T, full_matrices=False); r = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.8)) + 1
    XI[h] = U[:, :min(r, 3)]


def phi(X): return np.einsum('ij,jk,ik->i', X, PHI, X)


def rec_index(X, Hs):
    Xi = np.concatenate([XI[h] for h in Hs], 1); A_ = Xi.T @ PHI @ Xi
    f = np.linalg.lstsq(A_, Xi.T @ PHI @ X.T, rcond=None)[0]; return phi(X - (Xi @ f).T)


zs_ = Z['tcal'][0]; per_ = len(zs_) // len(TCAL); pick_ = np.arange(len(TCAL)) * per_ + rng.integers(per_, size=len(TCAL))
s0_ = np.sort(phi(FLAT(zs_[pick_]) - mu_n)); LIM = s0_[min(len(s0_) - 1, int(np.ceil(0.95 * (len(s0_) + 1))) - 1)]


def make_rbc(kappa):
    def rbc(Zw, maxk=5):
        X = FLAT(Zw).astype(np.float64) - mu_n; cur = phi(X); S = [[] for _ in range(len(X))]
        act = np.where(cur > LIM)[0]
        for step in range(maxk):
            if len(act) == 0: break
            keys = {}
            for i in act: keys.setdefault(tuple(S[i]), []).append(i)
            nxt = []
            for Hs, rows in keys.items():
                rows = np.array(rows); bst = np.full(len(rows), np.inf); arg = np.zeros(len(rows), int)
                for h in range(1, C):
                    if h in Hs: continue
                    v = rec_index(X[rows], list(Hs) + [h]); b = v < bst; bst[b] = v[b]; arg[b] = h
                for n_, i in enumerate(rows):
                    S[i].append(int(arg[n_]))
                    if bst[n_] > kappa * LIM: nxt.append(i)
            act = np.array(nxt, int)
        return S
    return rbc


# AEC and MinExplain
V0_ = tviol_z(Z['tr'][0]); NK = V0_.shape[1]; PR_ = np.zeros((C, NK))
for h in range(1, C):
    g = ok_tr[h]; PR_[h] = (tviol_z(Z['tr'][h][g]) & ~V0_[g]).mean(0)
P0_ = np.clip(V0_.mean(0), 1e-4, None)


def make_aec(gp):
    th_e, th_p, lam_, tau = gp
    EX = PR_[1:] >= th_e; PO = PR_[1:] >= th_p; Wt = np.where(PO, PR_[1:], 0.0)

    def aec(Zw, maxk=5):
        V = tviol_z(Zw); out = []
        miss = (EX[None] & ~V[:, None, :]).sum(2)
        for i in range(len(V)):
            unc = V[i].copy(); S = []
            for _ in range(maxk):
                sc = Wt[:, unc].sum(1) - lam_ * miss[i]
                for e in S: sc[e - 1] = -np.inf
                j = int(np.argmax(sc))
                if sc[j] < tau: break
                S.append(j + 1); unc &= ~PO[j]
                if not unc.any(): break
            out.append(S)
        return out
    return aec


def make_minexp(w):
    cost = -np.log(np.clip(PR_[1:], 1e-4, 1.0)); c0 = -np.log(P0_)

    def minexp(Zw, maxk=5):
        V = tviol_z(Zw); out = []
        for i in range(len(V)):
            ks = np.where(V[i])[0]; cur = c0[ks].copy(); S = []
            for _ in range(maxk):
                if len(ks) == 0: break
                gain = np.maximum(cur[None] - cost[:, ks], 0).sum(1) - w
                for e in S: gain[e - 1] = -np.inf
                j = int(np.argmax(gain))
                if gain[j] <= 0: break
                S.append(j + 1); cur = np.minimum(cur, cost[j, ks])
            out.append(S)
        return out
    return minexp


# tuned parameters of the final base run when already logged, otherwise the middle of the tuning grid
tuned = []
try:
    for line in open(OUT + 'log_base.txt'):
        if line.startswith('tuned '):
            tuned.append(eval(line[6:].rsplit('  [', 1)[0], {'np': np})[0])
except Exception as e:
    log('timing: could not parse tuned values', e)
order_ = [n_ for n_ in ('PCA-RBC', 'AEC', 'MinExplain')]
PAR = {'PCA-RBC': 4, 'AEC': (0.9, 0.2, 0.5, 1.5), 'MinExplain': 8}
SRC = {k_: 'grid middle (tuned value not yet logged)' for k_ in PAR}
for k_, v_ in zip(order_, tuned): PAR[k_] = v_; SRC[k_] = 'tuned value from log_base.txt'
NOTES['params'] = {k_: [str(PAR[k_]), SRC[k_]] for k_ in PAR}
log('timing: params', NOTES['params'])

# MiniRocket
from aeon.classification.convolution_based import MiniRocketClassifier
nm_ = RAW['tr'][0][ok_tr[0]].reshape(-1, M); PM_ = nm_.mean(0); PSD_ = nm_.std(0) + 1e-8


def prep(X): return ((X - PM_) / PSD_).astype(np.float32).transpose(0, 2, 1).copy()


rs_ = np.random.default_rng(0)
XTR_ = np.concatenate([RAW['tr'][c][rs_.choice(ok_tr[c], 300, replace=False)] for c in range(C)])
YTR_ = np.repeat(np.arange(C), 300)
mrk = MiniRocketClassifier(n_jobs=1, random_state=0).fit(prep(XTR_), YTR_); log('timing: MiniRocket trained')

# 1D-CNN (FCN) on CPU torch
import torch
from torch import nn


class FCN(nn.Module):
    def __init__(s, n_ch=M, n_cls=C, widths=(64, 128, 64), kernels=(8, 5, 3)):
        super().__init__(); L = []; prev = n_ch
        for w_, k_ in zip(widths, kernels): L += [nn.Conv1d(prev, w_, k_, padding='same'), nn.BatchNorm1d(w_), nn.ReLU()]; prev = w_
        s.blocks = nn.Sequential(*L); s.head = nn.Linear(prev, n_cls)

    def forward(s, x): return s.head(s.blocks(x.transpose(1, 2)).mean(dim=2))


torch.manual_seed(0); cnn = FCN().eval()


def cnn_probs(X):
    with torch.no_grad():
        xs = torch.from_numpy(((X - PM_) / PSD_).astype(np.float32))
        return torch.cat([torch.softmax(cnn(xs[s:s + 256]), 1) for s in range(0, len(xs), 256)]).numpy()


# ---------------- gates and thresholds ----------------
P_rf = lambda Zw: rf.predict_proba(FLAT(Zw)); q_rf = conformal_empty(P_rf)
P_mc = lambda Zw: mc.predict_proba(FLAT(Zw)); q_mc = conformal_empty(P_mc)
P_xg = lambda Zw: xm.predict_proba(FLAT(Zw)); q_xg = conformal_empty(P_xg)
P_fd = lambda Zw: lda.predict_proba(FLAT(Zw)); q_fd = conformal_empty(P_fd)
n0_ = len(Z['tcal'][0]); per_ = n0_ // len(TCAL); pick_ = np.arange(len(TCAL)) * per_ + rng.integers(per_, size=len(TCAL))
s_ = np.sort(1 - mrk.predict_proba(prep(TC_RAW[:n0_][pick_]))[:, 0]); q_mr = 1 - s_[min(len(s_) - 1, int(np.ceil(0.95 * (len(s_) + 1))) - 1)]
TH_CC = 0.6
rbc = make_rbc(PAR['PCA-RBC']); aec = make_aec(PAR['AEC']); minexp = make_minexp(PAR['MinExplain'])
gate = lambda P, q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P]

# ---------------- single-thread settings ----------------
for m_ in [SC, DEC, mc] + brs + list(cc.estimators_): m_.set_params(n_jobs=1)
rf.set_params(n_jobs=1); xm.set_params(n_jobs=1); xm.get_booster().set_param('nthread', 1)
torch.set_num_threads(1)
try:
    import numba; numba.set_num_threads(1)
except Exception as e:
    log('numba threads', e)

# ---------------- samples ----------------
rs = np.random.default_rng(123)
PAIR_Z = np.concatenate([g[1] for g in COMP]); PAIR_RAW = np.concatenate([GRAW['pair_%d+%d' % g[0]] for g in COMP])
PAIR_PT = np.concatenate([PT['pair_%d+%d' % g[0]] for g in COMP])
iev = rs.choice(len(EV_Z), 2000, replace=False); ipr = rs.choice(len(PAIR_Z), 2000, replace=False)
SETS = {'TEV': (EV_RAW[iev], PT['ev'][iev]), 'pairs': (PAIR_RAW[ipr], PAIR_PT[ipr])}
assert np.allclose(Fn(EV_RAW[iev[:50]]), EV_Z[iev[:50]]) and np.allclose(Fn(PAIR_RAW[ipr[:50]]), PAIR_Z[ipr[:50]])


def ours_t(X, pt):
    Zw = Fn(X); key = 'timing_%d' % id(Zw); KEYOF[id(Zw)] = key; PT[key] = pt
    out = explain_t(Zw); del PT[key]; del KEYOF[id(Zw)]
    return out


METHODS = {
    'VioExplain': lambda X, pt: explain(Fn(X)),
    'VioExplain-T (excl. CNN forward)': ours_t,
    '1D-CNN forward (CPU torch)': lambda X, pt: cnn_probs(X),
    'RF-top1': lambda X, pt: gate(P_rf(Fn(X)), q_rf),
    'XGB-top1': lambda X, pt: gate(P_xg(Fn(X)), q_xg),
    'MC-LGBM-top1': lambda X, pt: gate(P_mc(Fn(X)), q_mc),
    'BR-LGBM': lambda X, pt: [[k + 1 for k in np.where(p > TH_BR)[0]] for p in br_probs(Fn(X))],
    'CC-LGBM': lambda X, pt: [[k + 1 for k in np.where(p > TH_CC)[0]] for p in cc.predict_proba(FLAT(Fn(X)))],
    'FDA': lambda X, pt: gate(P_fd(Fn(X)), q_fd),
    'PCA-RBC': lambda X, pt: rbc(Fn(X)),
    'AEC': lambda X, pt: aec(Fn(X)),
    'MinExplain': lambda X, pt: minexp(Fn(X)),
    'MiniRocket': lambda X, pt: gate(mrk.predict_proba(prep(X)), q_mr),
}
RESULT = dict(mode=MODE, n_windows={'TEV': 2000, 'pairs': 2000}, n_stream=200, notes=dict(
    cpu='process CPU time (time.process_time) under threadpoolctl limit 1 and n_jobs=1; the host was shared with other jobs, so wall time is inflated',
    input='raw 64x52 window; window statistics Fn included for every method that uses them',
    cnn='FCN (64,128,64 filters, kernels 8,5,3, untrained weights, same cost as trained), batch 256 in batched mode',
    minirocket='aeon MiniRocketClassifier fitted on 300 windows per class, inference cost independent of training size',
    vioexplain_t='explain_t with the deep diagnoser posterior given, the CNN forward pass is the separate row', **NOTES),
    methods={})
with threadpool_limits(limits=1):
    for name, fn in METHODS.items():
        res = {}
        for sn, (X, pt) in SETS.items():
            fn(X[:20], pt[:20])
            c0 = time.process_time(); w0 = time.perf_counter(); fn(X, pt); c1 = time.process_time(); w1 = time.perf_counter()
            cs = []
            for i in range(200):
                a = time.process_time(); fn(X[i:i + 1], pt[i:i + 1]); cs.append(time.process_time() - a)
            res[sn] = dict(batch_cpu_ms_per_window=1e3 * (c1 - c0) / len(X), batch_wall_ms_per_window=1e3 * (w1 - w0) / len(X),
                           stream_cpu_ms_mean=1e3 * float(np.mean(cs)), stream_cpu_ms_median=1e3 * float(np.median(cs)),
                           stream_cpu_ms_p95=1e3 * float(np.quantile(cs, 0.95)))
        RESULT['methods'][name] = res; log('timing', name, json.dumps(res))
        json.dump(RESULT, open('/path/to/vioexplain/results/final_v1/timing.json', 'w'), indent=1)
log('timing DONE')
