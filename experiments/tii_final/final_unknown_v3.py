"""TEP-U with the final VioExplain pipeline (final_fuse5.py, V3_FUNC=1, V3_MODE=f100t, V3_ABL=full, V3_TFEAT=0) and
fault h held out of every component.

Usage:  V3_HELD=h V3_MODE=f100t V3_FUNC=1 V3_RES=final_v3/unknown/h<h> NJ=2 OMP_NUM_THREADS=2 python final_unknown_v3.py
Needs results/final_v3/unknown/cnn_h<h>_{model.pt,probs.npz} from gpu_unknown.py (waits for them before the fusion step).

Method (identical to final_fuse5.py except that h never appears): knowledge Supp/FOOT, deduction operators, the
deduction-consistent scorer (classes 0 + 19 known), the BR response detectors (19 known), the counterfactual composition
(known pairs and random known triples), the addition decision fitted on calibration-run compositions of at most three
known effective faults, the temporal CNN retrained without h (posterior over 0 + 19 known) mixed into the empty-set
decision and first pick with the weight chosen on calibration, split-conformal tau0 (fused) and tau1 on known
test-calibration windows, one window per run. Arrays keep 21 (or 20) columns indexed by fault id, column h is zero.

Unknown scores (higher = more unknown), thresholds split-conformal at 95 percent on known test-calibration windows, one
window per (class, run):
  Ours    squared Mahalanobis distance of the description from mu_0 + sum_{e in S} (mu_e - mu_0), S = explained set
  MDS     squared Mahalanobis distance to the nearest known class mean
  MSP     minus the maximum softmax probability of a LightGBM multiclass classifier on known single faults
  Energy  minus logsumexp of its raw scores
Pooled within-class covariance of known classes on fit windows, shrunk 0.9 cov + 0.1 diag + 1e-3 I.
Evaluation (evaluation runs only, frozen windows removed): K1 known single faults and normal (official test set, seeded
25 percent subsample, identical across h), U1 fault h alone (all windows), K2 simulator pairs of two known faults (3000
windows sampled), U2 simulator pairs that contain h (all windows).
Output: results/final_v3/unknown/h<h>.json
"""
import sys
from common_u import *
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
UD = '/path/to/vioexplain/results/final_v3/unknown/'
K1FRAC = float(os.environ.get('V3_K1FRAC', '0.25')); NK2 = int(os.environ.get('V3_NK2', '3000'))
TIM = {}
SMOKE = os.environ.get('V3_SMOKE') == '1'   # tiny models and sets, only to test the code path end to end
if SMOKE:
    _mk_full = mk_lgb
    def mk_lgb(n=400): return _mk_full(5)
    K1FRAC = 0.01; NK2 = 200
log('held-out fault', HELD, 'known', KN, 'composition sets', len(AUG), 'triples', len(TRI_TRAIN))
assert all(HELD not in ev for ev, *_ in AUG) and HELD not in OP and HELD not in Supp and HELD not in set(YB.tolist())


def to21(p, cols):
    out = np.zeros((len(p), C), np.float32); out[:, np.asarray(cols)] = p; return out


def mix(ps, pt, a):
    l = (1 - a) * np.log(np.clip(ps, 1e-9, 1)) + a * np.log(np.clip(pt, 1e-9, 1)); l -= l.max(1, keepdims=True)
    e = np.exp(l); return e / e.sum(1, keepdims=True)


# ---------------- BR response detectors (known faults, same data as final_fuse5) ----------------
Xml = [XB]; Yml = [np.zeros((len(YB), K), bool)]
Yml[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = True
for ev, Zs, Zrest, idx in AUG:
    Y = np.zeros((len(Zs), K), bool); Y[:, np.array(ev) - 1] = True; Xml.append(Zs); Yml.append(Y)
Xml = np.concatenate(Xml); Yml = np.concatenate(Yml)
brs = {}
for e in KN:
    m = mk_lgb(300); m.fit(FLAT(Xml), Yml[:, e - 1].astype(int)); brs[e] = m
del Xml, Yml
log('BR detectors trained'); TIM['br'] = time.time() - t0


def br_probs(Zw):
    Xf = FLAT(Zw); out = np.zeros((len(Zw), K), np.float32)
    for e, m in brs.items(): out[:, e - 1] = m.predict_proba(Xf)[:, 1]
    return out


# ---------------- deduction-consistent scorer ----------------
Xa = [XB]; ya = [YB]
for ev, Zs, Zrest, idx in AUG:
    if Zrest is None: continue
    Xa.append(deduct(Zs, ev[1])); ya.append(np.full(len(Zs), ev[0]))
for tri, idx, Zs in TRI_TRAIN:
    for keep in tri:
        R = Zs.copy()
        for g in tri:
            if g != keep: R = deduct(R, g)
        Xa.append(R); ya.append(np.full(len(R), keep))
for h in KN:
    if OP[h] is not None:
        i2 = rng.choice(ok_tr[h], len(ok_tr[h]) // 3, replace=False); Xa.append(deduct(Z['tr'][h][i2], h)); ya.append(np.zeros(len(i2), int))
Xa = np.concatenate(Xa); ya = np.concatenate(ya)
SC = mk_lgb(); SC.fit(FLAT(Xa), ya); del Xa, ya
SCC = [int(c) for c in SC.classes_]; assert SCC == LABS, SCC
PS = lambda Zw: to21(SC.predict_proba(FLAT(Zw)), SCC)
tk = np.where(TC_Y != HELD)[0]; TCk_Z = TC_Z[tk]; TCk_Y = TC_Y[tk]; TCk_T = [TC_T[i] for i in tk]; TCk_RUN = TC_RUN[tk]
log('scorer trained')
TIM['scorer'] = time.time() - t0


# ---------------- addition decision ----------------
def cand_matrix(Zw, R, S_list, BRW):
    n = len(Zw); pr = PS(R); brr = br_probs(R); rn = (np.clip(R, -50, 50) ** 2).sum(axis=(1, 2))
    vch = tviol_z(Zw).reshape(n, M, 6).any(2)
    expl = np.zeros((n, M), bool)
    for i, S in enumerate(S_list):
        for g in S: expl[i] |= CHS[g]
    un = vch & ~expl; nun = un.sum(1)
    rank = np.argsort(np.argsort(-pr[:, 1:], 1), 1)
    F = np.zeros((n, K, 8), np.float32)
    for e in KN:
        red = (rn - (np.clip(deduct(R, e), -50, 50) ** 2).sum(axis=(1, 2))) / (rn + 1.0)
        cov = np.where(nun > 0, (un & CHS[e][None]).sum(1) / np.maximum(nun, 1), 0.0)
        F[:, e - 1] = np.stack([pr[:, e], rank[:, e - 1], pr[:, 0], BRW[:, e - 1], brr[:, e - 1], cov, red,
                                np.array([len(S) for S in S_list])], 1)
    return F


crng = np.random.default_rng(5)


def cal_window(events, i):
    x = RAW['cal'][0][i].copy()
    for e in events: x = x + RAW['cal'][e][i] - RAW['cal'][0][i]
    return x


ok_cal = {c: ~FR['cal'][c] for c in range(C)}
rows = []; lab = []; batch = []; tries_ = 0
while len(batch) < (300 if SMOKE else 2700) and tries_ < 200000:
    tries_ += 1
    k = int(crng.choice([1, 2, 2, 3, 3])); ev = [int(x) for x in crng.choice(KEFF, k, replace=False)]; i = int(crng.integers(ncal))
    if not all(ok_cal[e][i] for e in ev): continue
    if not all(tuple(sorted(p_)) in COVERED for p_ in itertools.combinations(ev, 2)): continue
    batch.append((ev, i))
log('addition-decision compositions', len(batch), 'tries', tries_)
Xc = np.stack([cal_window(ev, i) for ev, i in batch]); Zc = Fn(Xc)
truth = []
for (ev, i), x in zip(batch, Xc):
    V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
    for e in ev:
        Vr = tviol_raw(cal_window([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cal_window([e], i)[None])[0]
        if (caused & (~Vr | Ve)).any(): t.append(e)
    truth.append(set(t))


class CNN(nn.Module):
    def __init__(s, out=C):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(128, out)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))


MPATH = UD + 'cnn_h%d_model.pt' % HELD; PPATH = UD + 'cnn_h%d_probs.npz' % HELD
tw = time.time()
while not (os.path.exists(MPATH) and os.path.exists(PPATH)):
    if time.time() - tw > 7200: raise SystemExit('temporal CNN for h=%d not found' % HELD)
    time.sleep(15)
log('temporal CNN found after waiting %.0fs' % (time.time() - tw))
_ck = torch.load(MPATH, map_location='cpu', weights_only=False); TNET = CNN(len(LABS)); TNET.load_state_dict(_ck['state']); TNET.eval()
PT = {k_: to21(v_, LABS) for k_, v_ in np.load(PPATH).items()}
assert len(PT['ev']) == len(EV_Z) and len(PT['tcal']) == len(TC_Z)


def pt_raw(X):
    out = []
    with torch.no_grad():
        for s0 in range(0, len(X), 4096):
            x = torch.as_tensor(((X[s0:s0 + 4096] - _ck['PM']) / _ck['PSD']).astype(np.float32).transpose(0, 2, 1))
            out.append(torch.softmax(TNET(x), 1).numpy())
    return to21(np.concatenate(out), LABS)


PTc = pt_raw(Xc); ps_c = PS(Zc); ps_tc = PS(TCk_Z); PTk_tcal = PT['tcal'][tk]; best = None
for a in (0.0, 0.25, 0.5, 0.75, 0.9):
    f_single = f1_score(TCk_Y, mix(ps_tc, PTk_tcal, a).argmax(1), average='macro')
    pk = mix(ps_c, PTc, a)[:, 1:].argmax(1) + 1; f_comp = np.mean([int(f_) in t for f_, t in zip(pk, truth) if t])
    log('weight', a, 'tcal macroF1', round(f_single, 4), 'composed first-pick acc', round(f_comp, 4))
    if best is None or f_single + f_comp > best[1]: best = (a, f_single + f_comp)
A = best[0]; log('fusion weight (single + composed calibration)', best)
p = mix(ps_c, PTc, A); first = p[:, 1:].argmax(1) + 1
keep = np.array([f in t for f, t in zip(first, truth)])
Zc = Zc[keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]
S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = br_probs(Zc)
active = np.arange(len(Zc))
for step in range(4):
    if len(active) == 0: break
    F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active])
    nxt_active = []
    for n_, i in enumerate(active):
        for e in KN:
            if e in S_list[i]: continue
            rows.append(F[n_, e - 1]); lab.append(int(e in truth[i]))
        rem = [e for e in truth[i] if e not in S_list[i]]
        if rem:
            nxt = max(rem, key=lambda e: F[n_, e - 1, 0]); S_list[i].append(nxt); nxt_active.append(i)
    if nxt_active:
        idx = np.array(nxt_active); R[idx] = deduct_groups(R[idx], [S_list[i][-1] for i in idx])
    active = np.array(nxt_active, int)
DEC = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
log('cooperative decision rows', len(lab), 'positives', int(np.sum(lab)))
single_idx = np.where((TCk_Y > 0) & np.array([len(t) > 0 for t in TCk_T]))[0]
ps_ = mix(PS(TCk_Z[single_idx]), PTk_tcal[single_idx], A); fp = ps_[:, 1:].argmax(1) + 1; good = single_idx[fp == TCk_Y[single_idx]]
Rg = deduct_groups(TCk_Z[good], TCk_Y[good])
Fg = cand_matrix(TCk_Z[good], Rg, [[int(y)] for y in TCk_Y[good]], br_probs(TCk_Z[good]))
pp = DEC.predict_proba(Fg.reshape(-1, 8))[:, 1].reshape(len(good), K)
pp[:, HELD - 1] = -1; pp[np.arange(len(good)), TCk_Y[good] - 1] = -1
TAU_C = conformal_upper(pp.max(1), TCk_RUN[good])
TAU0 = conformal_empty(PS)
n0 = len(Z['tcal'][0]); per = n0 // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
s = np.sort(1 - mix(PS(Z['tcal'][0][pick]), PT['tcal'][:n0][pick], A)[:, 0])
TAU0F = 1 - s[min(len(s) - 1, int(np.ceil(0.95 * (len(s) + 1))) - 1)]
log('tau0', TAU0, 'tau0 fused', TAU0F, 'tau_c', TAU_C); TIM['decision'] = time.time() - t0


def explain_t(Zw, ptw, maxk=5):
    out = []
    for s0 in range(0, len(Zw), 20000):
        Zb = Zw[s0:s0 + 20000]; pb = ptw[s0:s0 + 20000]; p = mix(PS(Zb), pb, A); n = len(Zb)
        S_list = [[] if p[i, 0] >= TAU0F else [int(np.argmax(p[i, 1:])) + 1] for i in range(n)]
        act = np.array([i for i in range(n) if S_list[i]], int)
        if len(act):
            R = Zb.copy(); R[act] = deduct_groups(Zb[act], [S_list[i][0] for i in act]); BRW = br_probs(Zb)
            for step in range(maxk - 1):
                if len(act) == 0: break
                F = cand_matrix(Zb[act], R[act], [S_list[i] for i in act], BRW[act])
                pp = DEC.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K); pp[:, HELD - 1] = -1
                for n_, i in enumerate(act):
                    for g in S_list[i]: pp[n_, g - 1] = -1
                j = pp.argmax(1); add = pp[np.arange(len(act)), j] > TAU_C
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        out += S_list
    return out


# ---------------- unknown scores ----------------
XBf = FLAT(XB).astype(np.float64)
MEANS = np.stack([XBf[YB == c].mean(0) for c in LABS]); L2I = {c: i for i, c in enumerate(LABS)}
D_ = XBf - MEANS[np.array([L2I[int(y)] for y in YB])]
cov = np.cov(D_.T); cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov)) + 1e-3 * np.eye(cov.shape[0]); del D_, XBf
icov = np.linalg.inv(cov); icov = (icov + icov.T) / 2; LW = np.linalg.cholesky(icov)   # d' icov d = |LW' d|^2
MW = MEANS @ LW; FOOTW = {c: MW[L2I[c]] - MW[0] for c in KN}


def ours_score(Zw, sets):
    Xw = FLAT(Zw).astype(np.float64) @ LW; T = np.repeat(MW[:1], len(Zw), 0)
    for i, S in enumerate(sets):
        for e in S: T[i] += FOOTW[e]
    return ((Xw - T) ** 2).sum(1)


def mds_score(Zw):
    Xw = FLAT(Zw).astype(np.float64) @ LW
    d = (Xw ** 2).sum(1)[:, None] - 2 * Xw @ MW.T + (MW ** 2).sum(1)[None]
    return d.min(1), np.array(LABS)[d.argmin(1)]


BASE = mk_lgb(); BASE.fit(FLAT(XB), YB); assert [int(c) for c in BASE.classes_] == LABS
log('baseline multiclass classifier trained'); TIM['baseline'] = time.time() - t0


def all_scores(Zw, ptw):
    sets = explain_t(Zw, ptw); Xf = FLAT(Zw)
    pb = BASE.predict_proba(Xf); raw = BASE.predict(Xf, raw_score=True); md, mcls = mds_score(Zw)
    return dict(Ours=ours_score(Zw, sets), MDS=md, MSP=-pb.max(1), Energy=-logsumexp(raw, 1)), \
        dict(sets=sets, gbm_cls=np.array(LABS)[pb.argmax(1)], mds_cls=mcls)


METH = ['Ours', 'MDS', 'MSP', 'Energy']
# calibration: one known test-calibration window per (class, run), the same pick rule as conformal_upper
r_ = np.random.default_rng(17); cpick = []
for g in np.unique(TCk_RUN):
    ii = np.where(TCk_RUN == g)[0]; cpick.append(ii[r_.integers(len(ii))])
cpick = np.array(cpick)
sc_cal, _ = all_scores(TCk_Z[cpick], PTk_tcal[cpick])
THR = {}
for k_ in METH:
    v = np.sort(sc_cal[k_]); n = len(v); THR[k_] = float(v[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)])
log('calibration windows', len(cpick), 'thresholds', THR)
# simulator-domain diagnostics: simulator normal runs and single known-fault runs of test_pairs (valid windows).
# THR_SIM: split-conformal 95 percent on one window per simulator run (normal + known singles), a domain-matched
# alternative to THR for the simulator pair groups K2/U2 (these single runs are not part of K2/U2).
SIMX = []; SIMC = []; SIMG = []
for e in [0] + KN:
    key_ = (e,) if e else ()
    Wd, valid = sim_windows('test_pairs', MP, key_); X_ = Wd[key_]
    nrun = len(MP['runs']['normal' if e == 0 else str(e)]); per_ = len(X_) // nrun; ii = np.where(valid)[0]
    SIMX.append(X_[ii]); SIMC.append(np.full(len(ii), e)); SIMG.append(1000 * e + ii // per_)
SIMX = np.concatenate(SIMX); SIMC = np.concatenate(SIMC); SIMG = np.concatenate(SIMG)
sc_sim, aux_sim = all_scores(Fn(SIMX), pt_raw(SIMX)); del SIMX
r_ = np.random.default_rng(17); spick = []
for g in np.unique(SIMG):
    ii = np.where(SIMG == g)[0]; spick.append(ii[r_.integers(len(ii))])
spick = np.array(spick); THR_SIM = {}
for k_ in METH:
    v = np.sort(sc_sim[k_][spick]); n = len(v); THR_SIM[k_] = float(v[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)])
log('simulator single windows', len(SIMC), 'runs', len(spick), 'thresholds', THR_SIM)
# evaluation sets
sel_rng = np.random.default_rng(23); k1mask = sel_rng.random(len(EV_Z)) < K1FRAC
k1 = np.where(k1mask & (EV_Y != HELD))[0]; u1 = np.where(EV_Y == HELD)[0]
pool = [(g, w) for g, (ev, Zw, *_) in enumerate(COMP) if HELD not in ev for w in range(len(Zw))]
k2 = [pool[i] for i in sorted(np.random.default_rng(29 + HELD).choice(len(pool), min(NK2, len(pool)), replace=False))]
u2 = [(g, w) for g, (ev, Zw, *_) in enumerate(COMP) if HELD in ev for w in range(len(Zw))]


def gather(lst):
    if not lst: return np.zeros((0,) + Z['tr'][0].shape[1:], np.float32), np.zeros((0, C), np.float32)
    Zs = np.stack([COMP[g][1][w] for g, w in lst]); Ps = np.stack([PT['pair_%d+%d' % COMP[g][0]][w] for g, w in lst])
    return Zs, Ps


SETZ = {'K1': (EV_Z[k1], PT['ev'][k1]), 'U1': (EV_Z[u1], PT['ev'][u1]), 'K2': gather(k2), 'U2': gather(u2)}
SCO = {}; AUX = {}
for nm_, (Zs, Ps) in SETZ.items():
    if len(Zs) == 0: SCO[nm_] = {k_: np.zeros(0) for k_ in METH}; AUX[nm_] = dict(sets=[], gbm_cls=np.zeros(0, int), mds_cls=np.zeros(0, int)); continue
    SCO[nm_], AUX[nm_] = all_scores(Zs, Ps); log('scored', nm_, len(Zs))
TIM['scored'] = time.time() - t0


def auroc(neg, pos):
    if len(neg) == 0 or len(pos) == 0: return float('nan')
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


res = dict(held=HELD, n=dict(K1=len(k1), U1=len(u1), K2=len(k2), U2=len(u2), cal=len(cpick)),
           fusion_weight=float(A), tau0_fused=float(TAU0F), tau_c=float(TAU_C), thresholds=THR, methods={})
for k_ in METH:
    s_ = {nm_: SCO[nm_][k_] for nm_ in SCO}
    res['methods'][k_] = dict(
        auroc_all=auroc(np.r_[s_['K1'], s_['K2']], np.r_[s_['U1'], s_['U2']]), auroc_single=auroc(s_['K1'], s_['U1']),
        recall_U2=float((s_['U2'] > THR[k_]).mean()) if len(s_['U2']) else float('nan'),
        recall_U1=float((s_['U1'] > THR[k_]).mean()) if len(s_['U1']) else float('nan'),
        falseflag_K2=float((s_['K2'] > THR[k_]).mean()), falseflag_K1=float((s_['K1'] > THR[k_]).mean()))
res['n']['sim_single'] = int(len(SIMC)); res['n']['sim_cal_runs'] = int(len(spick)); res['thresholds_sim'] = THR_SIM
for k_ in METH:
    s_ = {nm_: SCO[nm_][k_] for nm_ in SCO}
    res['methods'][k_].update(
        auroc_pairs=auroc(s_['K2'], s_['U2']),
        recall_U2_simthr=float((s_['U2'] > THR_SIM[k_]).mean()) if len(s_['U2']) else float('nan'),
        falseflag_K2_simthr=float((s_['K2'] > THR_SIM[k_]).mean()),
        flag_simNormal_riethThr=float((sc_sim[k_][SIMC == 0] > THR[k_]).mean()),
        flag_simKnownSingle_riethThr=float((sc_sim[k_][SIMC > 0] > THR[k_]).mean()))
np.savez_compressed(UD + 'h%d%s_scores.npz' % (HELD, '_smoke' if SMOKE else ''),
                    **{'%s_%s' % (nm_, k_): SCO[nm_][k_] for nm_ in SCO for k_ in METH},
                    **{'cal_%s' % k_: sc_cal[k_] for k_ in METH}, **{'sim_%s' % k_: sc_sim[k_] for k_ in METH},
                    sim_class=SIMC, sim_group=SIMG, K1_y=EV_Y[k1], K2_pairs=np.array([COMP[g][0] for g, w in k2]),
                    U2_pairs=np.array([COMP[g][0] for g, w in u2]).reshape(-1, 2),
                    **{'%s_nsel' % nm_: np.array([len(S) for S in AUX[nm_]['sets']]) for nm_ in AUX})
# sanity of the explanation on known data
lab_k1 = [S[0] if S else 0 for S in AUX['K1']['sets']]
res['sanity'] = dict(ours_K1_label_macroF1=float(f1_score(EV_Y[k1], lab_k1, average='macro')),
                     ours_K1_setF1=float(np.mean([setf1(S, EV_T[i]) for S, i in zip(AUX['K1']['sets'], k1)])),
                     ours_K2_setF1=float(np.mean([setf1(S, COMP[g][2][w]) for S, (g, w) in zip(AUX['K2']['sets'], k2)])) if k2 else float('nan'),
                     gbm_K1_acc=float(np.mean(AUX['K1']['gbm_cls'] == EV_Y[k1])))
# violations caused by h on U2: share assigned to a known event
#   ours: largest-footprint rule among selected events whose support contains the unit (any selected event qualifies)
#   baseline: predicted known class c (GBM argmax, or nearest mean for MDS); assigned if the unit is in Supp[c]
#   *_only: violations caused by h and not by the known partner; gated: windows above the method's threshold send nothing
cnt = {k_: np.zeros(4) for k_ in ('Ours', 'GBM', 'MDS')}   # sent_all, sent_only, sent_all_gated, sent_only_gated
cnt_gbm_gate = {'MSP': np.zeros(2), 'Energy': np.zeros(2)}
tot = np.zeros(2)
for n_, (g, w) in enumerate(u2):
    ev, Zw, tru, V, causes = COMP[g]; b = [e for e in ev if e != HELD][0]
    ch = causes[HELD][w]; keys = np.where(ch)[0]
    if len(keys) == 0: continue
    only = ~causes[b][w][keys]; units = keys // 6; tot += [len(keys), only.sum()]
    S = AUX['U2']['sets'][n_]
    sent = np.array([any(u in SUPPSET[e] for e in S) for u in units])
    flag = SCO['U2']['Ours'][n_] > THR['Ours']
    cnt['Ours'] += [sent.sum(), (sent & only).sum(), 0 if flag else sent.sum(), 0 if flag else (sent & only).sum()]
    for key_, cls_, sc_ in (('GBM', AUX['U2']['gbm_cls'][n_], 'MSP'), ('MDS', AUX['U2']['mds_cls'][n_], 'MDS')):
        sent = np.array([cls_ != 0 and u in SUPPSET[int(cls_)] for u in units]); flag = SCO['U2'][sc_][n_] > THR[sc_]
        cnt[key_] += [sent.sum(), (sent & only).sum(), 0 if flag else sent.sum(), 0 if flag else (sent & only).sum()]
        if key_ == 'GBM':
            fe = SCO['U2']['Energy'][n_] > THR['Energy']
            cnt_gbm_gate['Energy'] += [0 if fe else sent.sum(), 0 if fe else (sent & only).sum()]
            cnt_gbm_gate['MSP'] += [0 if flag else sent.sum(), 0 if flag else (sent & only).sum()]
dv = lambda a, b: float(a / b) if b > 0 else float('nan')
res['h_violations_U2'] = dict(total=int(tot[0]), h_only=int(tot[1]))
res['methods']['Ours'].update(misassign=dv(cnt['Ours'][0], tot[0]), misassign_only=dv(cnt['Ours'][1], tot[1]),
                              misassign_gated=dv(cnt['Ours'][2], tot[0]), misassign_only_gated=dv(cnt['Ours'][3], tot[1]))
res['methods']['MDS'].update(misassign=dv(cnt['MDS'][0], tot[0]), misassign_only=dv(cnt['MDS'][1], tot[1]),
                             misassign_gated=dv(cnt['MDS'][2], tot[0]), misassign_only_gated=dv(cnt['MDS'][3], tot[1]))
for k_ in ('MSP', 'Energy'):
    res['methods'][k_].update(misassign=dv(cnt['GBM'][0], tot[0]), misassign_only=dv(cnt['GBM'][1], tot[1]),
                              misassign_gated=dv(cnt_gbm_gate[k_][0], tot[0]), misassign_only_gated=dv(cnt_gbm_gate[k_][1], tot[1]))
res['timing_s'] = {k_: round(v_) for k_, v_ in TIM.items()}; res['timing_s']['total'] = round(time.time() - t0)
json.dump(res, open(UD + 'h%d%s.json' % (HELD, '_smoke' if SMOKE else ''), 'w'), indent=1)
log('RESULT', json.dumps(res))
log('DONE')
