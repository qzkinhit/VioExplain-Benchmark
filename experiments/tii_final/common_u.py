"""common.py with one fault held out (V3_HELD=h), used only by final_unknown_v3.py (TEP-U).
Differences from common.py: KN = faults 1..20 without h. Supp, SUPPSET, CHS, FOOT and OP exist for KN only; the
counterfactual composition (pairs and random triples) uses KN only; XB/YB (single-fault scorer data) exclude class h.
The single-fault evaluation and calibration arrays and the simulator pair groups (COMP) still contain h, the runner
splits them into known and unknown parts. Triples, quads, EV_RAW/TC_RAW and evaluate() are not built.
Original docstring of common.py follows.
Shared data, knowledge and evaluation for the VioExplain v3 final runs (imported by every runner).
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
HELD = int(os.environ['V3_HELD']); KN = [e for e in range(1, C) if e != HELD]; KEFF = [e for e in EFF if e != HELD]
LABS = [0] + KN
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
Supp = {h: np.where((np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]) > 3).any(2).mean(0) >= 0.5)[0] for h in KN}
SUPPSET = {h: set(Supp[h].tolist()) for h in KN}
CHS = {h: np.isin(np.arange(M), Supp[h]) for h in KN}
FOOT = {h: np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]).mean(axis=(0, 2)) for h in KN}

# ---------------- counterfactual composition (known faults only) ----------------
pairs_all = list(itertools.combinations(range(1, C), 2))
pairs_known = list(itertools.combinations(KN, 2))
AUG = []; TRI_TRAIN = []
PCT = 0 if MODE == 'f0' else int(MODE[1:-1])
COVERED = set()
if PCT > 0:
    order_ = np.random.default_rng(7).permutation(len(pairs_known))
    COVERED = set(pairs_known[i] for i in order_[:int(round(len(pairs_known) * PCT / 100))])
    per_pair = 84
    for (A, B) in pairs_known:
        if (A, B) not in COVERED: continue
        good = np.intersect1d(ok_tr[A], ok_tr[B]); idx = rng.choice(good, min(per_pair, len(good)), replace=False)
        for first, second in ((A, B), (B, A)):
            Xs = RAW['tr'][first][idx] + RAW['tr'][second][idx] - RAW['tr'][0][idx]
            AUG.append(((first, second), Fn(Xs), Z['tr'][second][idx], idx))
    target_, got_, tries_ = int(round(300 * PCT / 100)), 0, 0
    while got_ < target_ and tries_ < 200000:
        tries_ += 1
        tri = tuple(sorted(int(x_) for x_ in rng.choice(KN, 3, replace=False)))
        if not all(p_ in COVERED for p_ in itertools.combinations(tri, 2)): continue
        good = np.intersect1d(np.intersect1d(ok_tr[tri[0]], ok_tr[tri[1]]), ok_tr[tri[2]])
        idx = rng.choice(good, min(30, len(good)), replace=False)
        Xs = RAW['tr'][tri[0]][idx] + RAW['tr'][tri[1]][idx] + RAW['tr'][tri[2]][idx] - 2 * RAW['tr'][0][idx]
        Zs = Fn(Xs); AUG.append((tri, Zs, None, idx)); TRI_TRAIN.append((tri, idx, Zs)); got_ += 1
log('composition sets', len(AUG))

# ---------------- footprint deduction operators ----------------
OP = {}
for h in KN:
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
TRIP = []; QUAD = []
log('concurrent groups: pairs', len(COMP), 'windows', sum(len(c[1]) for c in COMP))


def conformal_empty(P):
    """tau0 at alpha 0.05 from normal test-calibration windows, one window per run."""
    zs = Z['tcal'][0]; per = len(zs) // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
    s = 1 - P(zs[pick])[:, 0]; n = len(s); q = np.sort(s)[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
    return 1 - q


def mk_lgb(n=400):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, n_jobs=NJ, verbose=-1)


ok_rows = {c: ok_tr[c] for c in LABS}
XB = np.concatenate([Z['tr'][c][ok_rows[c]] for c in LABS]); YB = np.concatenate([[c] * len(ok_rows[c]) for c in LABS])
FLAT = lambda Zw: Zw.reshape(len(Zw), -1)
