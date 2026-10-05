"""Shared data, knowledge and evaluation for the VioExplain v3 final runs (imported by every runner).
Identical to the first part of final_run.py, plus raw windows of every evaluation group (GRAW) and a lookup KEYOF
from an evaluation array to its group key, so that probability files from other machines can be scored.

Usage:  python final_run.py MODE   MODE in {f0, f100t}
  f0     knowledge and scorers from single-fault data only
  f100t  plus counterfactual composition (all fault pairs and random triples superposed from paired runs)

Data
  fit / cal : Rieth training runs from the frozen hash partition (fit 300 runs, cal 100 runs), onset index 20
  test      : Rieth official test set, runs split by hash into tcal (100 runs, thresholds only) and eval (400 runs),
              onset index 160, 12 windows of 64 samples per run
  concurrent: simulator test_pairs (190 pairs x 20 seeds) and test_triples (40 triples x 10 seeds), onset 160,
              windows after a shutdown in any twin removed; four-fault windows superposed from test_pairs singles
Windows of a faulty run that is frozen after a plant shutdown are removed everywhere.
Outputs (OUT/MODE/): metrics.json, records_<method>.pkl (per-window predicted sets for every evaluation group).
"""
import os, sys, time, json, pickle, hashlib, itertools
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

MODE = os.environ.get('V3_MODE', 'f0')
FUNC = os.environ.get('V3_FUNC', '0') == '1'
OUT = ('/home/user' if os.path.exists('/path/to/vioexplain') else '/home/user') + '/vioexplain-v3/results/' + \
    os.environ.get('V3_RES', 'final_v1') + '/' + MODE + '/'
os.makedirs(OUT, exist_ok=True)
ON_GPU_SERVER = os.path.exists('/path/to/vioexplain')
if ON_GPU_SERVER:
    TRAIN = '/path/to/vioexplain/data/rieth2017/training_arrays/'
    TEST = '/path/to/vioexplain/data/rieth2017/testing_arrays/'
    PART = '/path/to/vioexplain/data/rieth2017/run_partitions.csv'
    SIM = '/path/to/vioexplain/data/tepsim/'
else:
    TRAIN = '/path/to/vioexplain/data/rieth2017/training_arrays/'
    TEST = '/path/to/vioexplain/data/rieth2017/testing_arrays/'
    PART = '/path/to/vioexplain/results/data_split/run_partitions.csv'
    SIM = '/path/to/vioexplain/tepsim_data/'
GDIR = ('/home/user' if ON_GPU_SERVER else '/home/user') + '/vioexplain-v3/results/final_v1_gpu/'
W = 64; C = 21; M = 52; K = 20; NJ = int(os.environ.get('NJ', '16'))
EFF = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
t0 = time.time()
LOG = open(OUT + os.environ.get('V3_LOG', 'log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a) + '  [%.0fs]' % (time.time() - t0)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


tt_ = np.arange(W) - (W - 1) / 2
TT = (tt_ ** 2).sum()


def feats(X):
    return np.stack([X.mean(1), X.std(1), (X * tt_[None, :, None]).sum(1) / TT, X.min(1), X.max(1),
                     np.median(X, 1), X[:, -8:].mean(1) - X[:, :8].mean(1)], 2)


def frozen(Xw):
    """True for windows in which at least 30 sensors are constant (data frozen after a shutdown)."""
    return (Xw.std(1) < 1e-9).sum(1) >= 30


def runs_of(split):
    ids = []
    for line in open(PART).read().strip().split('\n')[1:]:
        f, r, s, _ = line.split(',')
        if f == '0' and s == split: ids.append(int(r) - 1)
    return sorted(ids)


def wins(a, runs, onset):
    out = []
    for r in runs:
        x = np.asarray(a[r])
        for s in range(onset, x.shape[0] - W + 1, W): out.append(x[s:s + W])
    return np.stack(out).astype(np.float32)


def hsplit(r):
    return int(hashlib.sha256(('vioexplain-v3-test-split:%d' % r).encode()).hexdigest(), 16) % 5 == 0


FIT = runs_of('fit'); CALR = runs_of('calibration') or runs_of('cal')
TESTRUNS = list(range(500)); TCAL = [r for r in TESTRUNS if hsplit(r)]; TEV = [r for r in TESTRUNS if not hsplit(r)]
log('runs fit', len(FIT), 'cal', len(CALR), 'tcal', len(TCAL), 'eval', len(TEV))
tr_arr = {c: np.load(TRAIN + 'class_%02d.npy' % c, mmap_mode='r') for c in range(C)}
te_arr = {c: np.load(TEST + 'class_%02d.npy' % c, mmap_mode='r') for c in range(C)}
RAW = {'tr': {c: wins(tr_arr[c], FIT, 20) for c in range(C)},
       'cal': {c: wins(tr_arr[c], CALR, 20) for c in range(C)},
       'tcal': {c: wins(te_arr[c], TCAL, 160) for c in range(C)},
       'ev': {c: wins(te_arr[c], TEV, 160) for c in range(C)}}
FR = {sp: {c: frozen(RAW[sp][c]) for c in range(C)} for sp in RAW}
F0 = feats(RAW['tr'][0]); MU = F0.mean(0); SD = F0.std(0) + 1e-8


# ---------------- functional constraints (V3_FUNC=1) ----------------
# Sparse linear relations x_j[t] ~ b0 + beta^T x_P[t] (at most 5 partners chosen by orthogonal matching pursuit) are
# mined on normal fit windows; relations with R^2 >= 0.9 on normal calibration windows are kept (one per variable set).
# The window statistics of their residual series, standardized by normal fit windows, are extra units of the description.
# Typed violations of the 52 sensors (tviol_raw, tviol_z) are unchanged; tviol_all also covers the relation units.
REL = []
if FUNC:
    from sklearn.linear_model import OrthogonalMatchingPursuit
    Xn_ = RAW['tr'][0].reshape(-1, M).astype(np.float64); Xk_ = RAW['cal'][0].reshape(-1, M).astype(np.float64)
    mu_ = Xn_.mean(0); sd_ = Xn_.std(0); live_ = np.where(sd_ > 1e-6 * (np.abs(mu_) + 1))[0]
    Xs_ = (Xn_ - mu_) / np.where(sd_ > 0, sd_, 1); cands_ = []
    for j in live_:
        oth = np.array([i for i in live_ if i != j])
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5, fit_intercept=False).fit(Xs_[:, oth], Xs_[:, j])
        P_ = [int(i) for i in oth[np.flatnonzero(omp.coef_)]]
        A_ = np.c_[np.ones(len(Xn_)), Xn_[:, P_]]; coef = np.linalg.lstsq(A_, Xn_[:, j], rcond=None)[0]
        pk_ = coef[0] + Xk_[:, P_] @ coef[1:]
        r2c = 1 - ((Xk_[:, j] - pk_) ** 2).sum() / ((Xk_[:, j] - Xk_[:, j].mean()) ** 2).sum()
        cands_.append(dict(target=int(j), partners=P_, b0=float(coef[0]), beta=[float(b) for b in coef[1:]], r2_cal=float(r2c)))
    seen_ = set()
    for c_ in sorted(cands_, key=lambda d: -d['r2_cal']):
        vs_ = frozenset([c_['target']] + c_['partners'])
        if c_['r2_cal'] >= 0.9 and vs_ not in seen_: REL.append(c_); seen_.add(vs_)
    REL.sort(key=lambda d: d['target'])
    json.dump(REL, open(OUT + 'relations.json', 'w'), indent=1)
NR = len(REL); MT = M + NR
BREL = np.zeros((M, NR)); B0REL = np.zeros(NR); TGT = np.array([r['target'] for r in REL], int)
for i_, r_ in enumerate(REL):
    BREL[r_['partners'], i_] = r_['beta']; B0REL[i_] = r_['b0']


def resid(X):
    return (X[:, :, TGT].astype(np.float64) - X.astype(np.float64) @ BREL - B0REL).astype(np.float32)


if NR:
    FE0 = feats(resid(RAW['tr'][0])); MUE = FE0.mean(0); SDE = FE0.std(0) + 1e-8


def Fn(X):
    z = ((feats(X) - MU) / SD).astype(np.float32)
    if not NR: return z
    return np.concatenate([z, ((feats(resid(X)) - MUE) / SDE).astype(np.float32)], 1)


Z = {sp: {c: Fn(RAW[sp][c]) for c in range(C)} for sp in RAW}
NOTWIN = os.environ.get('V3_NOTWIN') == '1'
if NOTWIN:
    # variant without paired runs: the reference normal window of every faulty window comes from another normal run
    for sp_, runs_ in (('tr', FIT), ('cal', CALR)):
        n_ = len(RAW[sp_][0]); per_ = n_ // len(runs_); perm_ = (np.arange(n_) + per_) % n_
        RAW[sp_][0] = RAW[sp_][0][perm_]; Z[sp_][0] = Z[sp_][0][perm_]; FR[sp_][0] = FR[sp_][0][perm_]
N3 = F0[:, :, :3]; C3 = N3.mean(0); S3 = N3.std(0) + 1e-12


def tviol_raw(X):
    a = (np.stack([X.mean(1), X.std(1), (X * tt_[None, :, None]).sum(1) / TT], 2) - C3) / S3
    return np.stack([a > 3, -a > 3], 3).reshape(len(X), -1)


def tviol_z(Zw):
    Zw = Zw[:, :M]
    return np.stack([Zw[:, :, :3] > 3, -Zw[:, :, :3] > 3], 3).reshape(len(Zw), -1)


def tviol_all(Zw):
    return np.stack([Zw[:, :, :3] > 3, -Zw[:, :, :3] > 3], 3).reshape(len(Zw), -1)


ntr = len(Z['tr'][0]); ncal = len(Z['cal'][0])
log('windows', {sp: len(Z[sp][0]) for sp in Z}, 'frozen tr', int(sum(FR['tr'][c].sum() for c in range(C))), 'units', Z['tr'][0].shape[1], 'relations', NR)
rng = np.random.default_rng(0)
ok_tr = {c: np.where(~FR['tr'][c])[0] for c in range(C)}
Supp = {h: np.where((np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]) > 3).any(2).mean(0) >= 0.5)[0] for h in range(1, C)}
SUPPSET = {h: set(Supp[h].tolist()) for h in range(1, C)}
CHS = {h: np.isin(np.arange(M), Supp[h]) for h in range(1, C)}
FOOT = {h: np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]).mean(axis=(0, 2)) for h in range(1, C)}

# ---------------- counterfactual composition ----------------
pairs_all = list(itertools.combinations(range(1, C), 2))
AUG = []; TRI_TRAIN = []
PCT = 0 if MODE == 'f0' else int(MODE[1:-1])
COVERED = set()
if PCT > 0:
    order_ = np.random.default_rng(7).permutation(len(pairs_all))
    COVERED = set(pairs_all[i] for i in order_[:int(round(len(pairs_all) * PCT / 100))])
    per_pair = 84
    for (A, B) in pairs_all:
        if (A, B) not in COVERED: continue
        good = np.intersect1d(ok_tr[A], ok_tr[B]); idx = rng.choice(good, min(per_pair, len(good)), replace=False)
        for first, second in ((A, B), (B, A)):
            Xs = RAW['tr'][first][idx] + RAW['tr'][second][idx] - RAW['tr'][0][idx]
            AUG.append(((first, second), Fn(Xs), Z['tr'][second][idx], idx))
    target_, got_, tries_ = int(round(300 * PCT / 100)), 0, 0
    while got_ < target_ and tries_ < 200000:
        tries_ += 1
        tri = tuple(sorted(rng.choice(range(1, C), 3, replace=False)))
        if not all(p_ in COVERED for p_ in itertools.combinations(tri, 2)): continue
        good = np.intersect1d(np.intersect1d(ok_tr[tri[0]], ok_tr[tri[1]]), ok_tr[tri[2]])
        idx = rng.choice(good, min(30, len(good)), replace=False)
        Xs = RAW['tr'][tri[0]][idx] + RAW['tr'][tri[1]][idx] + RAW['tr'][tri[2]][idx] - 2 * RAW['tr'][0][idx]
        Zs = Fn(Xs); AUG.append((tri, Zs, None, idx)); TRI_TRAIN.append((tri, idx, Zs)); got_ += 1
log('composition sets', len(AUG))

# ---------------- footprint deduction operators ----------------
OP = {}
for h in range(1, C):
    s = Supp[h]
    if len(s) == 0: OP[h] = None; continue
    g = ok_tr[h]
    Xin = [Z['tr'][h][g][:, s, :].reshape(len(g), -1)]; Y = [(Z['tr'][h][g] - Z['tr'][0][g]).reshape(len(g), -1)]
    for ev, Zs, Zrest, idx in AUG:
        if Zrest is None or ev[0] != h: continue
        Xin.append(Zs[:, s, :].reshape(len(Zs), -1)); Y.append((Zs - Zrest).reshape(len(Zs), -1))
    OP[h] = Ridge(alpha=10.0).fit(np.concatenate(Xin), np.concatenate(Y))


def deduct(Zw, h):
    if OP[h] is None or len(Zw) == 0: return Zw.copy()
    s = Supp[h]
    return (Zw - OP[h].predict(Zw[:, s, :].reshape(len(Zw), -1)).reshape(Zw.shape)).astype(np.float32)


def deduct_groups(R, events):
    """Deduct a (possibly different) event from every row of R."""
    out = R.copy(); events = np.asarray(events)
    for h in np.unique(events):
        m = events == h; out[m] = deduct(R[m], int(h))
    return out


def setf1(p, t):
    p = set(p); t = set(t)
    return 1.0 if not p and not t else 2 * len(p & t) / (len(p) + len(t))


# ---------------- evaluation data ----------------
def eff_single(sp):
    V0 = tviol_raw(RAW[sp][0]); out = [[] for _ in range(len(V0))]; keep = [np.ones(len(V0), bool)]
    for c in range(1, C):
        e = (tviol_raw(RAW[sp][c]) & ~V0).any(1); out += [[c] if x else [] for x in e]; keep.append(~FR[sp][c])
    return out, np.concatenate(keep)


EV_Z = np.concatenate([Z['ev'][c] for c in range(C)]); EV_Y = np.concatenate([[c] * len(Z['ev'][c]) for c in range(C)])
EV_T, EV_KEEP = eff_single('ev'); EV_Z = EV_Z[EV_KEEP]; EV_Y = EV_Y[EV_KEEP]; EV_T = [t for t, k in zip(EV_T, EV_KEEP) if k]
TC_Z = np.concatenate([Z['tcal'][c] for c in range(C)]); TC_Y = np.concatenate([[c] * len(Z['tcal'][c]) for c in range(C)])
TC_T, TC_KEEP = eff_single('tcal'); TC_Z = TC_Z[TC_KEEP]; TC_Y = TC_Y[TC_KEEP]; TC_T = [t for t, k in zip(TC_T, TC_KEEP) if k]
TC_RUN = np.concatenate([1000 * c + np.repeat(np.arange(len(TCAL)), len(Z['tcal'][c]) // len(TCAL)) for c in range(C)])[TC_KEEP]


def conformal_upper(scores, groups, alpha=0.05, seed=17):
    """Split-conformal threshold: one score per calibration run (chosen at random), the ceil((1-alpha)(n+1))-th
    smallest value; a test score strictly above it is flagged with probability at most alpha."""
    r_ = np.random.default_rng(seed); groups = np.asarray(groups); pick = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; pick.append(idx[r_.integers(len(idx))])
    v = np.sort(np.asarray(scores)[pick]); n = len(v)
    return float(v[min(n - 1, int(np.ceil((1 - alpha) * (n + 1))) - 1)])
NEV0 = int((EV_Y == 0).sum())
log('single eval windows', len(EV_Z), 'tcal', len(TC_Z))


def load_meta(d): return json.load(open(SIM + d + '/meta.json'))


MP = load_meta('test_pairs'); MT = load_meta('test_triples')


def sim_windows(d, meta, events):
    onset = meta['onset']; subs = [tuple(sorted(c)) for r in range(len(events) + 1) for c in itertools.combinations(events, r)]
    out = {}; valid = None
    for s_ in subs:
        key = 'normal' if not s_ else '+'.join(str(e) for e in s_)
        a = np.load(SIM + d + '/set_%s.npy' % key, mmap_mode='r'); sh = [r['shutdown_sample'] for r in meta['runs'][key]]
        ws = []; ok = []
        for i in range(a.shape[0]):
            for st in range(onset, a.shape[1] - W + 1, W):
                ws.append(np.asarray(a[i, st:st + W])); ok.append(sh[i] < 0 or sh[i] >= st + W)
        out[s_] = np.stack(ws).astype(np.float32); ok = np.array(ok); valid = ok if valid is None else valid & ok
    return out, valid


def eff_truth(Wd, events):
    full = tuple(sorted(events)); V = tviol_raw(Wd[full]); V0 = tviol_raw(Wd[()]); res = [[] for _ in range(len(V))]
    causes = {}
    for e in events:
        rest = tuple(sorted(set(events) - {e})); Vr = tviol_raw(Wd[rest]); Ve = tviol_raw(Wd[(e,)]); caused = V & ~V0
        ce = caused & (~Vr | Ve); causes[e] = ce
        for i in np.where(ce.any(1))[0]: res[i].append(e)
    return res, V, causes


GRAW = {}
COMP = []
for (A, B) in pairs_all:
    Wd, valid = sim_windows('test_pairs', MP, (A, B)); idx = np.where(valid)[0]
    if len(idx) == 0: continue
    tru, V, causes = eff_truth({k: v[idx] for k, v in Wd.items()}, (A, B))
    GRAW['pair_%d+%d' % (A, B)] = Wd[(A, B)][idx]; COMP.append(((A, B), Fn(GRAW['pair_%d+%d' % (A, B)]), tru, V, causes))
TRIP = []
for name in MT['sets']:
    if name == 'normal' or name.count('+') != 2: continue
    ev = tuple(int(x) for x in name.split('+'))
    Wd, valid = sim_windows('test_triples', MT, ev); idx = np.where(valid)[0]
    if len(idx) == 0: continue
    tru, V, causes = eff_truth({k: v[idx] for k, v in Wd.items()}, ev)
    GRAW['triple_' + name] = Wd[ev][idx]; TRIP.append((ev, Fn(Wd[ev][idx]), tru, V, causes))
QUAD = []
qrng = np.random.default_rng(11)
singles = {}


def sim_single(e):
    if e not in singles:
        Wd, valid = sim_windows('test_pairs', MP, (e,)); singles[e] = (Wd[(e,)], valid, Wd[()])
    return singles[e]


for _ in range(40):
    q = tuple(sorted(qrng.choice(EFF, 4, replace=False)))
    parts = [sim_single(e) for e in q]; valid = parts[0][1].copy()
    for p_ in parts: valid &= p_[1]
    idx = np.where(valid)[0]
    if len(idx) == 0: continue
    X0 = parts[0][2][idx]

    def comb(sub):
        x = X0.copy()
        for e in sub: x = x + sim_single(e)[0][idx] - X0
        return x
    Wd = {tuple(sorted(s)): comb(s) for r in range(5) for s in itertools.combinations(q, r)}
    tru, V, causes = eff_truth(Wd, q)
    GRAW['quad_' + '+'.join(map(str, q))] = Wd[q]; QUAD.append((q, Fn(Wd[q]), tru, V, causes))
log('concurrent groups: pairs', len(COMP), 'windows', sum(len(c[1]) for c in COMP), 'triples', len(TRIP),
    'windows', sum(len(c[1]) for c in TRIP), 'quads', len(QUAD))


def attribution(sets, V, causes):
    tp = npred = ntrue = 0
    evs = list(causes.keys())
    for i, S in enumerate(sets):
        keys = np.where(V[i])[0]
        for k in keys:
            j = k // 6; cand = [h for h in S if j in SUPPSET.get(h, set())]
            if cand:
                h = max(cand, key=lambda h_: FOOT[h_][j]); npred += 1; tp += int(h in causes and causes[h][i, k])
        anyc = np.zeros(V.shape[1], bool)
        for e in evs: anyc |= causes[e][i]
        ntrue += int(anyc.sum())
    return tp, npred, ntrue


RES = {}
KEYOF = {id(EV_Z): 'ev', id(TC_Z): 'tcal'}
for ev_, Zw_, *_ in COMP: KEYOF[id(Zw_)] = 'pair_%d+%d' % ev_
for ev_, Zw_, *_ in TRIP: KEYOF[id(Zw_)] = 'triple_' + '+'.join(map(str, ev_))
for ev_, Zw_, *_ in QUAD: KEYOF[id(Zw_)] = 'quad_' + '+'.join(map(str, ev_))
EV_RAW = np.concatenate([RAW['ev'][c] for c in range(C)])[EV_KEEP]
TC_RAW = np.concatenate([RAW['tcal'][c] for c in range(C)])[TC_KEEP]
GRAW['ev'] = EV_RAW; GRAW['tcal'] = TC_RAW


def evaluate(name, fn):
    rec = {}
    ps = fn(EV_Z); rec['single'] = ps
    lab = [s[0] if s else 0 for s in ps]
    ne = [i for i, t in enumerate(EV_T) if t]
    out = dict(single_macroF1_gated=float(f1_score(EV_Y, lab, average='macro')),
               single_eff=float(np.mean([setf1(a, b) for a, b in zip(ps, EV_T)])),
               single_eff_nonempty=float(np.mean([setf1(ps[i], EV_T[i]) for i in ne])),
               normal_named=float(np.mean([len(s) > 0 for s, y in zip(ps, EV_Y) if y == 0])))
    fs, fu = [], []
    for gname, groups in (('pair', COMP), ('triple', TRIP), ('quad', QUAD)):
        f = []; tp = npred = ntrue = 0; rec[gname] = []
        for ev, Zw, tru, V, causes in groups:
            S = fn(Zw); rec[gname].append((ev, S)); fg = [setf1(a, b) for a, b in zip(S, tru)]; f += fg
            if gname == 'pair': (fs if tuple(sorted(ev)) in COVERED else fu).extend(fg)
            a, b, c = attribution(S, V, causes); tp += a; npred += b; ntrue += c
        out[gname + '_eff'] = float(np.mean(f)); out[gname + '_attrF1'] = 2 * tp / max(npred + ntrue, 1)
    if fs: out['pair_eff_seen'] = float(np.mean(fs))
    if fu: out['pair_eff_unseen'] = float(np.mean(fu))
    RES[name] = out; log(MODE, name, json.dumps(out))
    pickle.dump(rec, open(OUT + 'records_%s.pkl' % name, 'wb'))
    json.dump(RES, open(OUT + os.environ.get('V3_METRICS', 'metrics.json'), 'w'), indent=1)


def conformal_empty(P):
    """tau0 at alpha 0.05 from normal test-calibration windows, one window per run."""
    zs = Z['tcal'][0]; per = len(zs) // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
    s = 1 - P(zs[pick])[:, 0]; n = len(s); q = np.sort(s)[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
    return 1 - q


def mk_lgb(n=400):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, n_jobs=NJ, verbose=-1)


ok_rows = [ok_tr[c] for c in range(C)]
XB = np.concatenate([Z['tr'][c][ok_rows[c]] for c in range(C)]); YB = np.concatenate([[c] * len(ok_rows[c]) for c in range(C)])
FLAT = lambda Zw: Zw.reshape(len(Zw), -1)


# ---------------- composed calibration windows (shared by every method in the settings with composition) ----------------
def cal_compositions(n=2700, seed=5, all_pairs=False):
    """Counterfactual compositions of the calibration runs of the training data: one, two or three effective faults
    superposed on the same process noise, every pair inside a composed set being a covered pair. Returns the raw
    windows, their descriptions and their effective event sets (paired-run truth). The addition decision of VioExplain
    is fitted on these windows, and every parameter that decides how many events a baseline names is chosen on them
    together with the single-fault test-calibration windows (all_pairs=True for methods trained on single faults only,
    so that their calibration windows equal those of VioExplain)."""
    crng_ = np.random.default_rng(seed); ok_cal_ = {c_: ~FR['cal'][c_] for c_ in range(C)}; batch = []; tries = 0

    def cw(events, i):
        x = RAW['cal'][0][i].copy()
        for e in events: x = x + RAW['cal'][e][i] - RAW['cal'][0][i]
        return x
    while len(batch) < n and tries < 200000:
        tries += 1
        k = int(crng_.choice([1, 2, 2, 3, 3])); ev = [int(e) for e in crng_.choice(EFF, k, replace=False)]; i = int(crng_.integers(ncal))
        if not all(ok_cal_[e][i] for e in ev): continue
        if not all_pairs and not all(tuple(sorted(p_)) in COVERED for p_ in itertools.combinations(ev, 2)): continue
        batch.append((ev, i))
    Xc = np.stack([cw(ev, i) for ev, i in batch]); truth = []
    for (ev, i), x in zip(batch, Xc):
        V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
        for e in ev:
            Vr = tviol_raw(cw([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cw([e], i)[None])[0]
            if (caused & (~Vr | Ve)).any(): t.append(e)
        truth.append(t)
    return Xc, Fn(Xc), truth


def ml_gate(p_tcal):
    """Empty-set rule shared with every other method: a multi-label method names nothing on a window whose largest
    event probability does not exceed the split-conformal threshold of the fault-free test-calibration windows
    (one window per run, alpha 0.05)."""
    nm = np.where(TC_Y == 0)[0]
    return conformal_upper(np.asarray(p_tcal)[nm].max(1), TC_RUN[nm])


def ml_sets(P, ta, q0):
    return [[] if p.max() <= q0 else [k + 1 for k in np.where(p > ta)[0]] for p in P]


def tune_threshold(p_single, p_comp, truth_comp, grid, tag='', q0=-1.0):
    """Probability threshold of a multi-label method: maximizes the mean of the event-set F1 on single-fault
    test-calibration windows and on composed calibration windows (the same calibration windows as VioExplain).
    The grid is extended outward (halving toward 0, or toward 1) while the optimum sits at an end."""
    def score(ta):
        f = float(np.mean([setf1(a, t) for a, t in zip(ml_sets(p_single, ta, q0), TC_T)]))
        if p_comp is not None:
            f = 0.5 * f + 0.5 * float(np.mean([setf1(a, t) for a, t in zip(ml_sets(p_comp, ta, q0), truth_comp)]))
        return f
    grid = sorted(grid); vals = {ta: score(ta) for ta in grid}
    for _ in range(12):
        best = max(grid, key=lambda t: (vals[t], -t))
        if best == grid[0] and grid[0] > 1e-6: nt = grid[0] / 2
        elif best == grid[-1] and grid[-1] < 1 - 1e-6: nt = 1 - (1 - grid[-1]) / 2
        else: break
        grid = sorted(grid + [nt]); vals[nt] = score(nt)
    best = max(grid, key=lambda t: (vals[t], -t))
    for ta in grid: log('  %s threshold grid' % tag, ta, round(vals[ta], 4))
    log('%s threshold' % tag, (best, vals[best]), 'grid ends', grid[0], grid[-1], 'interior', grid[0] < best < grid[-1],
        'composed calibration windows', 0 if p_comp is None else len(p_comp))
    return best
