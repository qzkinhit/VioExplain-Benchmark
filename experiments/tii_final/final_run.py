"""VioExplain v3 final experiment runner (official protocol).

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

MODE = sys.argv[1]
OUT = '/path/to/vioexplain/results/final_v1/' + MODE + '/'
os.makedirs(OUT, exist_ok=True)
TRAIN = '/path/to/vioexplain/data/rieth2017/training_arrays/'
TEST = '/path/to/vioexplain/data/rieth2017/testing_arrays/'
PART = '/path/to/vioexplain/results/data_split/run_partitions.csv'
SIM = '/path/to/vioexplain/tepsim_data/'
W = 64; C = 21; M = 52; K = 20; NJ = int(os.environ.get('NJ', '16'))
EFF = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
t0 = time.time()
LOG = open(OUT + 'log.txt', 'a')


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


def Fn(X): return ((feats(X) - MU) / SD).astype(np.float32)


Z = {sp: {c: Fn(RAW[sp][c]) for c in range(C)} for sp in RAW}
N3 = F0[:, :, :3]; C3 = N3.mean(0); S3 = N3.std(0) + 1e-12


def tviol_raw(X):
    a = (np.stack([X.mean(1), X.std(1), (X * tt_[None, :, None]).sum(1) / TT], 2) - C3) / S3
    return np.stack([a > 3, -a > 3], 3).reshape(len(X), -1)


def tviol_z(Zw):
    return np.stack([Zw[:, :, :3] > 3, -Zw[:, :, :3] > 3], 3).reshape(len(Zw), -1)


ntr = len(Z['tr'][0]); ncal = len(Z['cal'][0])
log('windows', {sp: len(Z[sp][0]) for sp in Z}, 'frozen tr', int(sum(FR['tr'][c].sum() for c in range(C))))
rng = np.random.default_rng(0)
ok_tr = {c: np.where(~FR['tr'][c])[0] for c in range(C)}
Supp = {h: np.where((np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]) > 3).any(2).mean(0) >= 0.5)[0] for h in range(1, C)}
SUPPSET = {h: set(Supp[h].tolist()) for h in range(1, C)}
CHS = {h: np.isin(np.arange(M), Supp[h]) for h in range(1, C)}
FOOT = {h: np.abs(Z['tr'][h][ok_tr[h]] - Z['tr'][0][ok_tr[h]]).mean(axis=(0, 2)) for h in range(1, C)}

# ---------------- counterfactual composition ----------------
pairs_all = list(itertools.combinations(range(1, C), 2))
AUG = []; TRI_TRAIN = []
if MODE == 'f100t':
    per_pair = 84
    for (A, B) in pairs_all:
        good = np.intersect1d(ok_tr[A], ok_tr[B]); idx = rng.choice(good, min(per_pair, len(good)), replace=False)
        for first, second in ((A, B), (B, A)):
            Xs = RAW['tr'][first][idx] + RAW['tr'][second][idx] - RAW['tr'][0][idx]
            AUG.append(((first, second), Fn(Xs), Z['tr'][second][idx], idx))
    for _ in range(300):
        tri = tuple(sorted(rng.choice(range(1, C), 3, replace=False)))
        good = np.intersect1d(np.intersect1d(ok_tr[tri[0]], ok_tr[tri[1]]), ok_tr[tri[2]])
        idx = rng.choice(good, min(30, len(good)), replace=False)
        Xs = RAW['tr'][tri[0]][idx] + RAW['tr'][tri[1]][idx] + RAW['tr'][tri[2]][idx] - 2 * RAW['tr'][0][idx]
        Zs = Fn(Xs); AUG.append((tri, Zs, None, idx)); TRI_TRAIN.append((tri, idx, Zs))
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


COMP = []
for (A, B) in pairs_all:
    Wd, valid = sim_windows('test_pairs', MP, (A, B)); idx = np.where(valid)[0]
    if len(idx) == 0: continue
    tru, V, causes = eff_truth({k: v[idx] for k, v in Wd.items()}, (A, B))
    COMP.append(((A, B), Fn(Wd[(A, B)][idx]), tru, V, causes))
TRIP = []
for name in MT['sets']:
    if name == 'normal' or name.count('+') != 2: continue
    ev = tuple(int(x) for x in name.split('+'))
    Wd, valid = sim_windows('test_triples', MT, ev); idx = np.where(valid)[0]
    if len(idx) == 0: continue
    tru, V, causes = eff_truth({k: v[idx] for k, v in Wd.items()}, ev)
    TRIP.append((ev, Fn(Wd[ev][idx]), tru, V, causes))
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
    QUAD.append((q, Fn(Wd[q]), tru, V, causes))
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


def evaluate(name, fn):
    rec = {}
    ps = fn(EV_Z); rec['single'] = ps
    lab = [s[0] if s else 0 for s in ps]
    ne = [i for i, t in enumerate(EV_T) if t]
    out = dict(single_macroF1_gated=float(f1_score(EV_Y, lab, average='macro')),
               single_eff=float(np.mean([setf1(a, b) for a, b in zip(ps, EV_T)])),
               single_eff_nonempty=float(np.mean([setf1(ps[i], EV_T[i]) for i in ne])),
               normal_named=float(np.mean([len(s) > 0 for s, y in zip(ps, EV_Y) if y == 0])))
    for gname, groups in (('pair', COMP), ('triple', TRIP), ('quad', QUAD)):
        f = []; tp = npred = ntrue = 0; rec[gname] = []
        for ev, Zw, tru, V, causes in groups:
            S = fn(Zw); rec[gname].append((ev, S)); f += [setf1(a, b) for a, b in zip(S, tru)]
            a, b, c = attribution(S, V, causes); tp += a; npred += b; ntrue += c
        out[gname + '_eff'] = float(np.mean(f)); out[gname + '_attrF1'] = 2 * tp / max(npred + ntrue, 1)
    RES[name] = out; log(MODE, name, json.dumps(out))
    pickle.dump(rec, open(OUT + 'records_%s.pkl' % name, 'wb'))
    json.dump(RES, open(OUT + 'metrics.json', 'w'), indent=1)


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

# ---------------- single-label baselines ----------------
if MODE == 'f0':
    for bname, model in (('RF', RandomForestClassifier(500, n_jobs=NJ, random_state=0)), ('MC-LGBM', mk_lgb())):
        model.fit(FLAT(XB), YB); P = lambda Zw, m=model: m.predict_proba(FLAT(Zw))
        RES[bname + '_label_macroF1'] = float(f1_score(EV_Y, P(EV_Z).argmax(1), average='macro'))
        q0 = conformal_empty(P)
        evaluate(bname + '-top1', lambda Zw, P=P, q0=q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Zw)])
    try:
        import xgboost as xgb
        xm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, tree_method='hist')
        xm.fit(FLAT(XB), YB); P = lambda Zw: xm.predict_proba(FLAT(Zw))
        RES['XGB_label_macroF1'] = float(f1_score(EV_Y, P(EV_Z).argmax(1), average='macro'))
        q0 = conformal_empty(P)
        evaluate('XGB-top1', lambda Zw, q0=q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Zw)])
    except Exception as e:
        log('xgboost failed', e)

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
evaluate('BR-LGBM', lambda Zw: [[k + 1 for k in np.where(p > TH_BR)[0]] for p in br_probs(Zw)])

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
SC = mk_lgb(); SC.fit(FLAT(Xa), ya)
PS = lambda Zw: SC.predict_proba(FLAT(Zw))
RES['Ours_label_macroF1'] = float(f1_score(EV_Y, PS(EV_Z).argmax(1), average='macro'))
log('ours label macro F1', RES['Ours_label_macroF1'])


# ---------------- cooperative add decision ----------------
def cand_matrix(Zw, R, S_list, BRW):
    """Feature rows (n, 20, 8) for every candidate event at the current step, batched over windows."""
    n = len(Zw); pr = PS(R); brr = br_probs(R); rn = (np.clip(R, -50, 50) ** 2).sum(axis=(1, 2))
    vch = tviol_z(Zw).reshape(n, M, 6).any(2)
    expl = np.zeros((n, M), bool)
    for i, S in enumerate(S_list):
        for g in S: expl[i] |= CHS[g]
    un = vch & ~expl; nun = un.sum(1)
    rank = np.argsort(np.argsort(-pr[:, 1:], 1), 1)
    F = np.zeros((n, K, 8), np.float32)
    for e in range(1, C):
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
rows = []; lab = []
batch = []
for it in range(3000):
    k = int(crng.choice([1, 2, 2, 3, 3, 4])); ev = list(crng.choice(EFF, k, replace=False)); i = int(crng.integers(ncal))
    if not all(ok_cal[e][i] for e in ev): continue
    batch.append((ev, i))
Xc = np.stack([cal_window(ev, i) for ev, i in batch]); Zc = Fn(Xc); Vc = tviol_raw(Xc)
truth = []
for (ev, i), x in zip(batch, Xc):
    V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
    for e in ev:
        Vr = tviol_raw(cal_window([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cal_window([e], i)[None])[0]
        if (caused & (~Vr | Ve)).any(): t.append(e)
    truth.append(set(t))
p = PS(Zc); first = p[:, 1:].argmax(1) + 1
keep = np.array([f in t for f, t in zip(first, truth)])
Zc = Zc[keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]
S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = br_probs(Zc)
active = np.arange(len(Zc))
for step in range(4):
    if len(active) == 0: break
    F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active])
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
DEC = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
log('cooperative decision rows', len(lab), 'positives', int(np.sum(lab)))
# stop threshold: conformal on single-fault test-calibration windows whose first pick is correct
single_idx = np.where((TC_Y > 0) & np.array([len(t) > 0 for t in TC_T]))[0]
ps_ = PS(TC_Z[single_idx]); fp = ps_[:, 1:].argmax(1) + 1; good = single_idx[fp == TC_Y[single_idx]]
Rg = deduct_groups(TC_Z[good], TC_Y[good]); Fg = cand_matrix(TC_Z[good], Rg, [[int(y)] for y in TC_Y[good]], br_probs(TC_Z[good]))
pp = DEC.predict_proba(Fg.reshape(-1, 8))[:, 1].reshape(len(good), K)
pp[np.arange(len(good)), TC_Y[good] - 1] = -1
TAU_C = float(np.quantile(pp.max(1), 0.95))
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


evaluate('VioExplain', explain)
log('DONE')
