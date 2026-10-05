"""VioExplain final runner (version 5). Same as version 4 below, plus:
  * V3_ABL in {full, noCoop, noDeductTrain, noOp, noTemp}; V3_NOTWIN=1 and V3_FUNC handled by common.py.
  * No composition at all when MODE is f0 or V3_ABL=noCoop: the addition rule is the residual posterior with a
    split-conformal threshold, and the fusion weight is chosen on single-fault test-calibration windows only.
  * Compositions used to fit the addition decision have at most three events and, in coverage modes, only covered
    pairs (every pair inside a composed set is covered), so four-fault windows and never-composed pairs are unseen.
  * tau_1 (and tau_1 of noCoop) is split-conformal with one window per calibration run.
  * Ties in the fusion-weight choice go to the smaller weight. V3_BR=1 also evaluates BR-LGBM.
VioExplain with temporal evidence, version 4 = version 3 plus a convolutional response detector.
V3_DMODEL=.../mlcnn_<mode>_model.pt and a second argument .../mlcnn_<mode>_probs_v2.npz add the detector probability of
each candidate on the window (multi-label CNN over raw windows, trained on the same data as the BR detectors) as an
addition feature. Version 3 description follows.
VioExplain with temporal evidence, version 3 (weight chosen on single and composed calibration windows).

Usage:  V3_MODE=f0|f100t V3_TMODEL=.../cnn_model.pt V3_TNAME=TC3 [V3_TFEAT=1] V3_LOG=... V3_METRICS=... \
        python final_fuse3.py .../cnn_probs_v2.npz
Same as final_fuse.py except: (1) the temporal diagnoser (saved weights, CPU inference) also scores the composed
calibration windows used to fit the addition decision; (2) the mixture weight a is chosen to maximize the sum of the
label macro F1 on test-calibration windows and the first-pick accuracy on composed calibration windows (both
calibration data, never evaluation data); the first pick of the addition-decision training uses the same mixture;
(3) with V3_TFEAT=1 the temporal posterior of each candidate on the window is a ninth addition feature.
"""
import sys
from common import *
_fs = [dict(np.load(p_)) for p_ in sys.argv[1].split(',')]
PT = {k_: np.mean([f_[k_] for f_ in _fs], 0) for k_ in _fs[0]}
TNAME = os.environ.get('V3_TNAME', 'T')
ABLS = set(os.environ.get('V3_ABL', 'full').split(',')); ABL = os.environ.get('V3_ABL', 'full')
if PCT == 0: ABLS.add('noCoop')
NOCOMP = PCT == 0 or 'noCoop' in ABLS
NOSTEP = os.environ.get('V3_NOSTEP', '0') == '1'   # addition decision without the number of selected events as a feature
TFEAT = os.environ.get('V3_TFEAT', '0') == '1'; DFEAT = 'V3_DMODEL' in os.environ; NF = 8 + int(TFEAT) + int(DFEAT)
DET = dict(np.load(sys.argv[2])) if DFEAT else None
import torch, torch.nn as nn
if 'noOp' in ABLS:
    def deduct(Zw, h):
        out = Zw.copy(); out[:, Supp[h], :] = 0; return out

    def deduct_groups(R, events):
        out = R.copy(); events = np.asarray(events)
        for h in np.unique(events):
            m = events == h; out[m] = deduct(R[m], int(h))
        return out
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))


class CNN(nn.Module):
    def __init__(s, out=C):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(128, out)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))


class ResBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.Conv1d(o, o, 5, padding=2),
                            nn.BatchNorm1d(o), nn.ReLU(), nn.Conv1d(o, o, 3, padding=1), nn.BatchNorm1d(o))
        s.s = nn.Sequential(nn.Conv1d(i, o, 1), nn.BatchNorm1d(o))
    def forward(s, x): return torch.relu(s.c(x) + s.s(x))


class ResNet(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(ResBlock(M, 64), ResBlock(64, 128), ResBlock(128, 128)); s.h = nn.Linear(128, C)
    def forward(s, x): return s.h(s.f(x).mean(-1))


_ck = torch.load(os.environ['V3_TMODEL'], map_location='cpu', weights_only=False)
if DFEAT:
    _dk = torch.load(os.environ['V3_DMODEL'], map_location='cpu', weights_only=False); DNET = CNN(K); DNET.load_state_dict(_dk['state']); DNET.eval()


def det_raw(X):
    out = []
    with torch.no_grad():
        for s0 in range(0, len(X), 4096):
            x = torch.as_tensor(((X[s0:s0 + 4096] - _dk['PM']) / _dk['PSD']).astype(np.float32).transpose(0, 2, 1))
            out.append(torch.sigmoid(DNET(x)).numpy())
    return np.concatenate(out)
TNET = CNN() if 'cnn' in os.path.basename(os.environ['V3_TMODEL']) else ResNet(); TNET.load_state_dict(_ck['state']); TNET.eval()


def pt_raw(X):
    out = []
    with torch.no_grad():
        for s0 in range(0, len(X), 4096):
            x = torch.as_tensor(((X[s0:s0 + 4096] - _ck['PM']) / _ck['PSD']).astype(np.float32).transpose(0, 2, 1))
            out.append(torch.softmax(TNET(x), 1).numpy())
    return np.concatenate(out)


def mix(ps, pt, a):
    l = (1 - a) * np.log(np.clip(ps, 1e-9, 1)) + a * np.log(np.clip(pt, 1e-9, 1)); l -= l.max(1, keepdims=True)
    e = np.exp(l); return e / e.sum(1, keepdims=True)

# ---------------- multi-label baseline with the same data ----------------
Xml = [XB]; Yml = [np.zeros((len(YB), K), bool)]
Yml[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = True
for ev, Zs, Zrest, idx in AUG:
    Y = np.zeros((len(Zs), K), bool); Y[:, np.array(ev) - 1] = True; Xml.append(Zs); Yml.append(Y)
Xml = np.concatenate(Xml); Yml = np.concatenate(Yml)
brs = []
for k in range(K):
    m = mk_lgb(300); m.fit(FLAT(Xml), Yml[:, k].astype(int)); brs.append(m)


def br_probs(Zw):
    Xf = FLAT(Zw); return np.stack([m.predict_proba(Xf)[:, 1] for m in brs], 1)


pml = br_probs(TC_Z); best = None
for ta in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    s = [[k + 1 for k in np.where(p > ta)[0]] for p in pml]; f = np.mean([setf1(a, b) for a, b in zip(s, TC_T)])
    if best is None or f > best[1]: best = (ta, f)
TH_BR = best[0]
if os.environ.get('V3_BR') == '1': evaluate('BR-LGBM', lambda Zw: [[k + 1 for k in np.where(p > TH_BR)[0]] for p in br_probs(Zw)])


# ---------------- VioExplain scorer (deduction-consistent) ----------------
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
for h in range(1, C):
    if OP[h] is not None:
        i2 = rng.choice(ok_tr[h], len(ok_tr[h]) // 3, replace=False); Xa.append(deduct(Z['tr'][h][i2], h)); ya.append(np.zeros(len(i2), int))
Xa = np.concatenate(Xa); ya = np.concatenate(ya)
if 'noDeductTrain' in ABLS: Xa, ya = XB, YB
SC = mk_lgb(); SC.fit(FLAT(Xa), ya)
PS = lambda Zw: SC.predict_proba(FLAT(Zw))
RES['Ours_label_macroF1'] = float(f1_score(EV_Y, PS(EV_Z).argmax(1), average='macro'))
log('ours label macro F1', RES['Ours_label_macroF1'])


# ---------------- cooperative add decision ----------------
def cand_matrix(Zw, R, S_list, BRW, PTW=None, DTW=None):
    """Feature rows (n, 20, 8) for every candidate event at the current step, batched over windows."""
    n = len(Zw); pr = PS(R); brr = br_probs(R); rn = (np.clip(R, -50, 50) ** 2).sum(axis=(1, 2))
    vch = tviol_z(Zw).reshape(n, M, 6).any(2)
    expl = np.zeros((n, M), bool)
    for i, S in enumerate(S_list):
        for g in S: expl[i] |= CHS[g]
    un = vch & ~expl; nun = un.sum(1)
    rank = np.argsort(np.argsort(-pr[:, 1:], 1), 1)
    F = np.zeros((n, K, NF), np.float32)
    for e in range(1, C):
        red = (rn - (np.clip(deduct(R, e), -50, 50) ** 2).sum(axis=(1, 2))) / (rn + 1.0)
        cov = np.where(nun > 0, (un & CHS[e][None]).sum(1) / np.maximum(nun, 1), 0.0)
        cols = [pr[:, e], rank[:, e - 1], pr[:, 0], BRW[:, e - 1], brr[:, e - 1], cov, red, np.zeros(n) if NOSTEP else np.array([len(S) for S in S_list])]
        if TFEAT: cols.append(PTW[:, e])
        if DFEAT: cols.append(DTW[:, e - 1])
        F[:, e - 1] = np.stack(cols, 1)
    return F


crng = np.random.default_rng(5)


def cal_window(events, i):
    x = RAW['cal'][0][i].copy()
    for e in events: x = x + RAW['cal'][e][i] - RAW['cal'][0][i]
    return x


ok_cal = {c: ~FR['cal'][c] for c in range(C)}
rows = []; lab = []
batch = []
tries_ = 0
while not NOCOMP and len(batch) < 2700 and tries_ < 200000:
    tries_ += 1
    k = int(crng.choice([1, 2, 2, 3, 3])); ev = list(crng.choice(EFF, k, replace=False)); i = int(crng.integers(ncal))
    if not all(ok_cal[e][i] for e in ev): continue
    if not all(tuple(sorted(p_)) in COVERED for p_ in itertools.combinations([int(e) for e in ev], 2)): continue
    batch.append((ev, i))
log('addition-decision compositions', len(batch), 'tries', tries_, 'nocomp', NOCOMP)
if NOCOMP: batch = [([int(EFF[0])], 0)]   # placeholder, unused
Xc = np.stack([cal_window(ev, i) for ev, i in batch]); Zc = Fn(Xc); Vc = tviol_raw(Xc)
truth = []
for (ev, i), x in zip(batch, Xc):
    V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
    for e in ev:
        Vr = tviol_raw(cal_window([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cal_window([e], i)[None])[0]
        if (caused & (~Vr | Ve)).any(): t.append(e)
    truth.append(set(t))
PTc = pt_raw(Xc); DTc = det_raw(Xc) if DFEAT else np.zeros((len(Xc), K)); ps_c = PS(Zc); ps_tc = PS(TC_Z); best = None
for a in ((0.0,) if 'noTemp' in ABLS else (0.0, 0.25, 0.5, 0.75, 0.9)):
    f_single = f1_score(TC_Y, mix(ps_tc, PT['tcal'], a).argmax(1), average='macro')
    pk = mix(ps_c, PTc, a)[:, 1:].argmax(1) + 1; f_comp = 0.0 if NOCOMP else np.mean([int(f_) in t for f_, t in zip(pk, truth) if t])
    log('weight', a, 'tcal macroF1', round(f_single, 4), 'composed first-pick acc', round(f_comp, 4))
    if best is None or f_single + f_comp > best[1]: best = (a, f_single + f_comp)
A = best[0]; log('fusion weight (single + composed calibration)', best)
p = mix(ps_c, PTc, A); first = p[:, 1:].argmax(1) + 1
keep = np.array([f in t for f, t in zip(first, truth)])
Zc = Zc[keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]; PTc = PTc[keep]; DTc = DTc[keep]
S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = br_probs(Zc)
active = np.arange(0 if NOCOMP else len(Zc))
for step in range(4):
    if len(active) == 0: break
    F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active], PTc[active], DTc[active])
    nxt_active = []
    for n_, i in enumerate(active):
        for e in range(1, C):
            if e in S_list[i]: continue
            rows.append(F[n_, e - 1]); lab.append(int(e in truth[i]))
        rem = [e for e in truth[i] if e not in S_list[i]]
        if rem:
            nxt = max(rem, key=lambda e: F[n_, e - 1, 0]); S_list[i].append(nxt); nxt_active.append(i)
    if nxt_active:
        idx = np.array(nxt_active); R[idx] = deduct_groups(R[idx], [S_list[i][-1] for i in idx])
    active = np.array(nxt_active, int)
DEC = None if NOCOMP else lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
log('cooperative decision rows', len(lab), 'positives', int(np.sum(lab)))
# stop threshold: conformal on single-fault test-calibration windows whose first pick is correct
single_idx = np.where((TC_Y > 0) & np.array([len(t) > 0 for t in TC_T]))[0]
ps_ = mix(PS(TC_Z[single_idx]), PT['tcal'][single_idx], A); fp = ps_[:, 1:].argmax(1) + 1; good = single_idx[fp == TC_Y[single_idx]]
Rg = deduct_groups(TC_Z[good], TC_Y[good])
if NOCOMP:
    pp = PS(Rg)[:, 1:].copy()
else:
    Fg = cand_matrix(TC_Z[good], Rg, [[int(y)] for y in TC_Y[good]], br_probs(TC_Z[good]), PT['tcal'][good], DET['tcal'][good] if DFEAT else None)
    pp = DEC.predict_proba(Fg.reshape(-1, NF))[:, 1].reshape(len(good), K)
pp[np.arange(len(good)), TC_Y[good] - 1] = -1
TAU_C = conformal_upper(pp.max(1), TC_RUN[good])
TAU0 = conformal_empty(PS)
log('tau0', TAU0, 'tau_c', TAU_C)


def explain(Zw, maxk=5, chunk=20000):
    out = []
    for s0 in range(0, len(Zw), chunk):
        Zb = Zw[s0:s0 + chunk]; p = PS(Zb); n = len(Zb)
        S_list = [[] if p[i, 0] >= TAU0 else [int(np.argmax(p[i, 1:])) + 1] for i in range(n)]
        act = np.array([i for i in range(n) if S_list[i]], int)
        if len(act):
            R = Zb.copy(); R[act] = deduct_groups(Zb[act], [S_list[i][0] for i in act]); BRW = br_probs(Zb)
            for step in range(maxk - 1):
                if len(act) == 0: break
                F = cand_matrix(Zb[act], R[act], [S_list[i] for i in act], BRW[act])
                pp = DEC.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K)
                for n_, i in enumerate(act):
                    for g in S_list[i]: pp[n_, g - 1] = -1
                j = pp.argmax(1); add = pp[np.arange(len(act)), j] >= TAU_C
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        out += S_list
    return out

n0 = len(Z['tcal'][0]); per = n0 // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
s = np.sort(1 - mix(PS(Z['tcal'][0][pick]), PT['tcal'][:n0][pick], A)[:, 0])
TAU0F = 1 - s[min(len(s) - 1, int(np.ceil(0.95 * (len(s) + 1))) - 1)]
log('tau0 fused', TAU0F)
RES['Ours' + TNAME + '_label_macroF1'] = float(f1_score(EV_Y, mix(PS(EV_Z), PT['ev'], A).argmax(1), average='macro'))


def explain_t(Zw, maxk=5):
    pt_all = PT[KEYOF[id(Zw)]]; dt_all = DET[KEYOF[id(Zw)]] if DFEAT else np.zeros((len(Zw), K)); out = []
    for s0 in range(0, len(Zw), 20000):
        Zb = Zw[s0:s0 + 20000]; p = mix(PS(Zb), pt_all[s0:s0 + 20000], A); n = len(Zb)
        S_list = [[] if p[i, 0] >= TAU0F else [int(np.argmax(p[i, 1:])) + 1] for i in range(n)]
        act = np.array([i for i in range(n) if S_list[i]], int)
        if len(act):
            R = Zb.copy(); R[act] = deduct_groups(Zb[act], [S_list[i][0] for i in act]); BRW = br_probs(Zb)
            for step in range(maxk - 1):
                if len(act) == 0: break
                if NOCOMP:
                    pp = PS(R[act])[:, 1:].copy()
                else:
                    F = cand_matrix(Zb[act], R[act], [S_list[i] for i in act], BRW[act], pt_all[s0:s0 + 20000][act], dt_all[s0:s0 + 20000][act])
                    pp = DEC.predict_proba(F.reshape(-1, NF))[:, 1].reshape(len(act), K)
                for n_, i in enumerate(act):
                    for g in S_list[i]: pp[n_, g - 1] = -1
                j = pp.argmax(1); add = pp[np.arange(len(act)), j] > TAU_C
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        out += S_list
    return out


evaluate('VioExplain' + ('' if ABL == 'full' else '-' + ABL.replace(',', '+')) + ('-nostep' if NOSTEP else '') + ('-noTwin' if NOTWIN else '') + ('' if FUNC else '-noFunc') + '-' + TNAME, explain_t)
log('DONE')
