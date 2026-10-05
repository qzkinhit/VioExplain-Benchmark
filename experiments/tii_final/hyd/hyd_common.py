"""VioExplain on the UCI hydraulic condition monitoring data (Helwig et al. 2015, UCI id 447): shared data,
knowledge and evaluation, mirroring common.py of the TEP runs with the no-twin adaptations listed below.

Environment
  HYD_MODE    f0 | f100t | fall      (fall = baselines also trained on real multi-fault fit cycles, supervised reference)
  HYD_EVENTS  comp | sev             (comp: 4 component events; sev: 10 severity-level events mapped back to components)
  HYD_SD      pooled | normal        (feature scale: pooled within-condition std of normal and single-fault fit cycles,
                                      or std of the normal fit cycles only)
  HYD_OUTSUB  ''  | e.g. final/      (subdirectory of results/hyd_v1 for the outputs)
  HYD_FUNC    0 | 1                  (functional-constraint units as in common.py with V3_FUNC=1)
  HYD_CAL     one | single           (calibration rule of the parameters that decide how many events a baseline names;
                                      one: single-fault and normal test-calibration cycles plus composed calibration
                                      cycles, the calibration data of VioExplain; single: test-calibration cycles only)
  HYD_GRID    wide | old | ext       (grids under HYD_CAL=single; old is the short grid;
                                      ext is a diagnostic with values beyond the ends of the wide grids)
  HYD_CALSET  cal | tcal             (composed cycles of HYD_CAL=one; cal: the 2700 compositions of calibration cycles
                                      that fit the addition decision; tcal: diagnostic, the 1500 compositions of
                                      test-calibration cycles of the variant choice in hyd_final.py)
  HYD_MLGATE  1 | 0                  (under HYD_CAL=one: split-conformal empty-set gate of the multi-label methods)
Data
  2205 load cycles of 60 s, every sensor averaged to 1 Hz, so one cycle is one window of 60 x 17.
  Events: cooler < 100, valve < 100, pump leakage > 0, accumulator < 130. Truth of a cycle = its set of degraded components.
  Split: within every condition group (cooler, valve, pump, accumulator), cycles ordered by sha256('vioexplain-hyd:%d');
         rank r mod 5 gives the role. Normal and single-fault groups: test, tcal, cal, fit, fit.
         Multi-fault groups (used for testing, and for fitting only in fall): test, test, tcal, fit, fit.
No-twin adaptations
  footprint and support against the normal fit mean; deduction target Z_h - mean Z_0;
  composition x = sum_e x_e - (k - 1) x_n with randomly drawn single-fault cycles x_e and a random normal cycle x_n,
  the counterfactual rest is the composition without one event, and the composed truth is the composed set.
"""
import os, sys, time, json, pickle, hashlib, itertools, collections
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

MODE = os.environ.get('HYD_MODE', 'f0')
EVT = os.environ.get('HYD_EVENTS', 'comp')
SDMODE = os.environ.get('HYD_SD', 'pooled')
TAG = os.environ.get('HYD_TAG', '')
NPAIR = int(os.environ.get('HYD_NPAIR', '60')); NTRI = int(os.environ.get('HYD_NTRI', '60'))
MCS = int(os.environ.get('HYD_MCS', '20'))           # LightGBM min_child_samples (TEP default 20)
CW = os.environ.get('HYD_CW', 'none')                # class weights for every tree classifier: none | balanced
ROOT = '/path/to/vioexplain/'
OUTSUB = os.environ.get('HYD_OUTSUB', '')
FUNC = os.environ.get('HYD_FUNC', '0') == '1'
GRID = os.environ.get('HYD_GRID', 'wide')
OUT = ROOT + 'results/hyd_v1/%s%s%s%s/' % (OUTSUB, MODE, '' if EVT == 'comp' else '_' + EVT, TAG)
os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get('NJ', '8'))
t0 = time.time()
LOG = open(OUT + os.environ.get('HYD_LOG', 'log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a) + '  [%.0fs]' % (time.time() - t0)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


DAT = np.load(ROOT + 'data/hydraulic/hyd_1hz.npz')
XALL = DAT['X'].astype(np.float32); PROF = DAT['profile']; NCYC, W, M = XALL.shape
SENS = [str(s) for s in DAT['sensors']]
COMPS = ['cooler', 'valve', 'pump', 'accumulator']
CEV = np.stack([PROF[:, 0] < 100, PROF[:, 1] < 100, PROF[:, 2] > 0, PROF[:, 3] < 130], 1)
NEV = CEV.sum(1)
TRUTH = [[int(j) + 1 for j in np.where(CEV[i])[0]] for i in range(NCYC)]
LEVELS = [(0, 20), (0, 3), (1, 90), (1, 80), (1, 73), (2, 1), (2, 2), (3, 115), (3, 100), (3, 90)]
if EVT == 'comp':
    K = 4; COMPOF = np.arange(5)
    EVS = TRUTH
else:
    K = len(LEVELS); COMPOF = np.array([0] + [l[0] + 1 for l in LEVELS])
    EVS = [[e + 1 for e, (j, v) in enumerate(LEVELS) if int(PROF[i, j]) == v] for i in range(NCYC)]
C = K + 1
EVENTS_OF_COMP = {j: [e for e in range(1, C) if COMPOF[e] == j] for j in range(1, 5)}


def to_comp(S): return sorted(set(int(COMPOF[e]) for e in S))


SALT = os.environ.get('HYD_SALT', 'vioexplain-hyd'); SEED = int(os.environ.get('HYD_SEED', '0'))


def hval(i): return int(hashlib.sha256((SALT + ':%d' % i).encode()).hexdigest(), 16)


GRP = collections.defaultdict(list)
for i in range(NCYC): GRP[tuple(PROF[i, :4].astype(int))].append(i)
ROLE = np.empty(NCYC, dtype=object)
for g, idx in GRP.items():
    idx = sorted(idx, key=hval); k = int(NEV[idx[0]])
    pat = ['test', 'tcal', 'cal', 'fit', 'fit'] if k <= 1 else ['test', 'test', 'tcal', 'fit', 'fit']
    for r, i in enumerate(idx): ROLE[i] = pat[r % 5]
LAB = np.array([0 if NEV[i] == 0 else (EVS[i][0] if NEV[i] == 1 else -1) for i in range(NCYC)])
SPL = ('fit', 'cal', 'tcal', 'test')
IDX = {sp: {c: np.where((ROLE == sp) & (LAB == c))[0] for c in range(C)} for sp in SPL}
RAW = {sp: {c: XALL[IDX[sp][c]] for c in range(C)} for sp in SPL}
log('MODE', MODE, 'EVT', EVT, 'SD', SDMODE, 'K', K, 'SALT', SALT, 'SEED', SEED)
log('split', {sp: {k: int(((ROLE == sp) & (NEV == k)).sum()) for k in range(5)} for sp in SPL})
log('single-fault/normal cycles per class', {sp: [len(IDX[sp][c]) for c in range(C)] for sp in SPL})

tt_ = np.arange(W) - (W - 1) / 2
TT = (tt_ ** 2).sum()


def feats(X):
    return np.stack([X.mean(1), X.std(1), (X * tt_[None, :, None]).sum(1) / TT, X.min(1), X.max(1),
                     np.median(X, 1), X[:, -8:].mean(1) - X[:, :8].mean(1)], 2)


F0 = feats(RAW['fit'][0]); MU = F0.mean(0)
FIT_GROUPS = [[i for i in idx if ROLE[i] == 'fit'] for g, idx in GRP.items() if NEV[idx[0]] <= 1]
FIT_GROUPS = [ii for ii in FIT_GROUPS if len(ii) >= 2]
if SDMODE == 'normal':
    SD = F0.std(0) + 1e-8
else:
    ss = 0.0; dof = 0
    for ii in FIT_GROUPS:
        Fg = feats(XALL[ii]); ss = ss + ((Fg - Fg.mean(0)) ** 2).sum(0); dof += len(ii) - 1
    SD = np.sqrt(ss / dof) + 1e-8
log('feature scale', SDMODE, 'median SD ratio pooled/normal-only', float(np.median(SD / (F0.std(0) + 1e-8))))


# ---------------- functional constraints (HYD_FUNC=1), as in common.py ----------------
# Sparse linear relations x_j[t] ~ b0 + beta^T x_P[t] (at most 5 partners chosen by orthogonal matching pursuit) are
# mined on the samples of the normal fit cycles; relations with R^2 >= 0.9 on the samples of the normal calibration
# cycles are kept (one per variable set). The window statistics of their residual series are extra units of the
# description, centred by the normal fit cycles and scaled as the sensor units (HYD_SD). Typed violations (tviol_z)
# and the covered-sensor masks (CHS) stay on the M sensors.
REL = []; REL_CANDS = []
if FUNC:
    from sklearn.linear_model import OrthogonalMatchingPursuit
    Xn_ = RAW['fit'][0].reshape(-1, M).astype(np.float64); Xk_ = RAW['cal'][0].reshape(-1, M).astype(np.float64)
    mu_ = Xn_.mean(0); sd_ = Xn_.std(0); live_ = np.where(sd_ > 1e-6 * (np.abs(mu_) + 1))[0]
    Xs_ = (Xn_ - mu_) / np.where(sd_ > 0, sd_, 1)
    for j in live_:
        oth = np.array([i for i in live_ if i != j])
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5, fit_intercept=False).fit(Xs_[:, oth], Xs_[:, j])
        P_ = [int(i) for i in oth[np.flatnonzero(omp.coef_)]]
        A_ = np.c_[np.ones(len(Xn_)), Xn_[:, P_]]; coef = np.linalg.lstsq(A_, Xn_[:, j], rcond=None)[0]
        pk_ = coef[0] + Xk_[:, P_] @ coef[1:]
        r2c = 1 - ((Xk_[:, j] - pk_) ** 2).sum() / ((Xk_[:, j] - Xk_[:, j].mean()) ** 2).sum()
        REL_CANDS.append(dict(target=int(j), target_name=SENS[j], partners=P_, partner_names=[SENS[i] for i in P_],
                              b0=float(coef[0]), beta=[float(b) for b in coef[1:]], r2_cal=float(r2c)))
    seen_ = set()
    for c_ in sorted(REL_CANDS, key=lambda d: -d['r2_cal']):
        vs_ = frozenset([c_['target']] + c_['partners'])
        if c_['r2_cal'] >= 0.9 and vs_ not in seen_: REL.append(c_); seen_.add(vs_)
    REL.sort(key=lambda d: d['target'])
    json.dump({'n_fit_normal_cycles': int(len(RAW['fit'][0])), 'n_cal_normal_cycles': int(len(RAW['cal'][0])),
               'n_candidates': len(REL_CANDS), 'n_r2_ge_0.9': int(sum(c_['r2_cal'] >= 0.9 for c_ in REL_CANDS)),
               'n_relations': len(REL), 'relations': REL, 'candidates': REL_CANDS}, open(OUT + 'relations.json', 'w'), indent=1)
    log('functional relations: candidates', len(REL_CANDS), 'with cal R2 >= 0.9', int(sum(c_['r2_cal'] >= 0.9 for c_ in REL_CANDS)),
        'kept (one per variable set)', len(REL), 'cal R2', {c_['target_name']: round(c_['r2_cal'], 3) for c_ in REL_CANDS})
NR = len(REL); MT = M + NR
UNITS = SENS + ['rel:' + SENS[r_['target']] for r_ in REL]
BREL = np.zeros((M, NR)); B0REL = np.zeros(NR); TGT = np.array([r_['target'] for r_ in REL], int)
for i_, r_ in enumerate(REL):
    BREL[r_['partners'], i_] = r_['beta']; B0REL[i_] = r_['b0']


def resid(X):
    return (X[:, :, TGT].astype(np.float64) - X.astype(np.float64) @ BREL - B0REL).astype(np.float32)


if NR:
    MUE = feats(resid(RAW['fit'][0])).mean(0)
    if SDMODE == 'normal':
        SDE = feats(resid(RAW['fit'][0])).std(0) + 1e-8
    else:
        ss = 0.0; dof = 0
        for ii in FIT_GROUPS:
            Fg = feats(resid(XALL[ii])); ss = ss + ((Fg - Fg.mean(0)) ** 2).sum(0); dof += len(ii) - 1
        SDE = np.sqrt(ss / dof) + 1e-8


def Fn(X):
    z = ((feats(X) - MU) / SD).astype(np.float32)
    if not NR: return z
    return np.concatenate([z, ((feats(resid(X)) - MUE) / SDE).astype(np.float32)], 1)


Z = {sp: {c: Fn(RAW[sp][c]) if len(RAW[sp][c]) else np.zeros((0, MT, 7), np.float32) for c in range(C)} for sp in SPL}
Z0BAR = Z['fit'][0].mean(0)


def tviol_z(Zw):
    D = Zw[:, :M, :3] - Z0BAR[None, :M, :3]
    return np.stack([D > 3, -D > 3], 3).reshape(len(Zw), -1)


rng = np.random.default_rng(SEED)
Supp = {h: np.where((np.abs(Z['fit'][h] - Z0BAR) > 3).any(2).mean(0) >= 0.5)[0] for h in range(1, C)}
SUPPSET = {h: set(Supp[h].tolist()) for h in range(1, C)}
CHS = {h: np.isin(np.arange(M), Supp[h]) for h in range(1, C)}
FOOT = {h: np.abs(Z['fit'][h] - Z0BAR).mean(axis=(0, 2)) for h in range(1, C)}
log('support sensors', {h: [UNITS[j] for j in Supp[h]] for h in range(1, C)})

# ---------------- counterfactual composition (no-twin) ----------------
def comp_ok(S): return len(set(int(COMPOF[e]) for e in S)) == len(S)


pairs_all = [p for p in itertools.combinations(range(1, C), 2) if comp_ok(p)]
tris_all = [t for t in itertools.combinations(range(1, C), 3) if comp_ok(t)]
AUG = []; TRI_TRAIN = []; AUG_RAW = []
if MODE == 'f100t':
    npair = max(4, NPAIR * 6 // len(pairs_all)); ntri = max(4, NTRI * 4 // len(tris_all))
    for (A, B) in pairs_all:
        ia = rng.integers(len(RAW['fit'][A]), size=npair); ib = rng.integers(len(RAW['fit'][B]), size=npair)
        i0 = rng.integers(len(RAW['fit'][0]), size=npair)
        Xs = RAW['fit'][A][ia] + RAW['fit'][B][ib] - RAW['fit'][0][i0]; Zs = Fn(Xs); AUG_RAW.append(((A, B), Xs))
        AUG.append(((A, B), Zs, Z['fit'][B][ib], None)); AUG.append(((B, A), Zs, Z['fit'][A][ia], None))
    for tri in tris_all:
        ii = [rng.integers(len(RAW['fit'][e]), size=ntri) for e in tri]; i0 = rng.integers(len(RAW['fit'][0]), size=ntri)
        Xs = sum(RAW['fit'][e][j] for e, j in zip(tri, ii)) - 2 * RAW['fit'][0][i0]; Zs = Fn(Xs)
        AUG.append((tri, Zs, None, None)); TRI_TRAIN.append((tri, None, Zs)); AUG_RAW.append((tri, Xs))
log('composition sets', len(AUG), 'rows', sum(len(a[1]) for a in AUG))

# ---------------- footprint deduction operators ----------------
OP = {}
for h in range(1, C):
    s = Supp[h]
    if len(s) == 0: OP[h] = None; continue
    Zh = Z['fit'][h]
    Xin = [Zh[:, s, :].reshape(len(Zh), -1)]; Y = [(Zh - Z0BAR).reshape(len(Zh), -1)]
    for ev, Zs, Zrest, _ in AUG:
        if Zrest is None or ev[0] != h: continue
        Xin.append(Zs[:, s, :].reshape(len(Zs), -1)); Y.append((Zs - Zrest).reshape(len(Zs), -1))
    OP[h] = Ridge(alpha=10.0).fit(np.concatenate(Xin), np.concatenate(Y))


def deduct(Zw, h):
    if OP[h] is None or len(Zw) == 0: return Zw.copy()
    s = Supp[h]
    return (Zw - OP[h].predict(Zw[:, s, :].reshape(len(Zw), -1)).reshape(Zw.shape)).astype(np.float32)


def deduct_groups(R, events):
    out = R.copy(); events = np.asarray(events)
    for h in np.unique(events):
        m = events == h; out[m] = deduct(R[m], int(h))
    return out


def setf1(p, t):
    p = set(p); t = set(t)
    return 1.0 if not p and not t else 2 * len(p & t) / (len(p) + len(t))


# ---------------- evaluation data ----------------
TEST_IDX = np.where(ROLE == 'test')[0]
ZT = Fn(XALL[TEST_IDX]); KT = NEV[TEST_IDX]; TT_TRUTH = [TRUTH[i] for i in TEST_IDX]
TC1 = np.where((ROLE == 'tcal') & (NEV <= 1))[0]
TC_Z = Fn(XALL[TC1]); TC_Y = LAB[TC1]; TC_T = [TRUTH[i] for i in TC1]
TCA = np.where(ROLE == 'tcal')[0]
TCA_Z = Fn(XALL[TCA]); TCA_T = [TRUTH[i] for i in TCA]
GRAW = {'test': XALL[TEST_IDX], 'tcal1': XALL[TC1], 'tcalall': XALL[TCA]}
KEYOF = {id(ZT): 'test', id(TC_Z): 'tcal1', id(TCA_Z): 'tcalall'}
log('test cycles by number of events', {k: int((KT == k).sum()) for k in range(5)}, 'tcal(normal+single)', len(TC1))


def tune_sets(sets_ev, truth):
    return float(np.mean([setf1(to_comp(a), b) for a, b in zip(sets_ev, truth)]))


RES = {}


def evaluate(name, fn):
    Sev = fn(ZT); S = [to_comp(s) for s in Sev]; out = {}
    for k in (1, 2, 3, 4):
        m = np.where(KT == k)[0]
        out['k%d_setF1' % k] = float(np.mean([setf1(S[i], TT_TRUTH[i]) for i in m]))
        out['k%d_exact' % k] = float(np.mean([set(S[i]) == set(TT_TRUTH[i]) for i in m]))
        out['k%d_size' % k] = float(np.mean([len(S[i]) for i in m]))
    m01 = np.where(KT <= 1)[0]   # gated label macro F1 as in common.py: first element of the set, 0 if empty
    out['single_normal_macroF1_gated'] = float(f1_score([TT_TRUTH[i][0] if TT_TRUTH[i] else 0 for i in m01],
                                                        [int(COMPOF[Sev[i][0]]) if Sev[i] else 0 for i in m01], average='macro'))
    m0 = np.where(KT == 0)[0]
    out['normal_named'] = float(np.mean([len(S[i]) > 0 for i in m0])); out['n_normal'] = int(len(m0))
    mm = np.where(KT >= 2)[0]; out['multi_setF1'] = float(np.mean([setf1(S[i], TT_TRUTH[i]) for i in mm]))
    out['all_setF1'] = float(np.mean([setf1(S[i], TT_TRUTH[i]) for i in range(len(S))]))
    Pm = np.zeros((len(S), 4), bool); Tm = np.zeros((len(S), 4), bool)
    for i in range(len(S)):
        for j in S[i]: Pm[i, j - 1] = True
        for j in TT_TRUTH[i]: Tm[i, j - 1] = True
    out['comp_F1'] = {COMPS[j]: float(f1_score(Tm[:, j], Pm[:, j])) for j in range(4)}
    RES[name] = out; log(MODE, EVT, name, json.dumps(out))
    pickle.dump({'test_idx': TEST_IDX, 'pred_events': Sev, 'pred_comp': S}, open(OUT + 'records_%s.pkl' % name, 'wb'))
    json.dump(out, open(OUT + 'metrics_%s.json' % name, 'w'), indent=1)


def conformal_empty(P, alpha=0.05):
    """tau0 from all normal test-calibration cycles (one window per cycle)."""
    s = 1 - P(Z['tcal'][0])[:, 0]; n = len(s); q = np.sort(s)[min(n - 1, int(np.ceil((1 - alpha) * (n + 1))) - 1)]
    return 1 - q


def mk_lgb(n=400):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, min_child_samples=MCS, n_jobs=NJ, verbose=-1,
                              class_weight=None if CW == 'none' else 'balanced')


XB = np.concatenate([Z['fit'][c] for c in range(C)]); YB = np.concatenate([[c] * len(Z['fit'][c]) for c in range(C)])
FLAT = lambda Zw: Zw.reshape(len(Zw), -1)
XBR = np.concatenate([RAW['fit'][c] for c in range(C)])
# test cycles that are normal or single-fault, for the label macro F1 of single-label classifiers
T01 = np.where(KT <= 1)[0]; T01_Y = LAB[TEST_IDX[T01]]

# ---------------- multi-label training data (BR / CC and the scorer features of VioExplain) ----------------
Xml = [XB]; Yml = [np.zeros((len(YB), K), int)]
Yml[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = 1
for ev, Zs, Zrest, _ in AUG:                          # as in final_run.py, a pair enters once per order
    Y = np.zeros((len(Zs), K), int); Y[:, np.array(ev) - 1] = 1; Xml.append(Zs); Yml.append(Y)
if MODE == 'fall':
    FM = np.where((ROLE == 'fit') & (NEV >= 2))[0]
    Y = np.zeros((len(FM), K), int)
    for r, i in enumerate(FM): Y[r, np.array(EVS[i]) - 1] = 1
    Xml.append(Fn(XALL[FM])); Yml.append(Y)
    log('fall: real multi-fault fit cycles added', len(FM))
Xml = np.concatenate(Xml); Yml = np.concatenate(Yml)
log('multi-label rows', len(Xml))
_BR = []


def br_models():
    if not _BR:
        for k in range(K):
            m = mk_lgb(300); m.fit(FLAT(Xml), Yml[:, k]); _BR.append(m)
    return _BR


def br_probs(Zw):
    Xf = FLAT(Zw); return np.stack([m.predict_proba(Xf)[:, 1] for m in br_models()], 1)


THR = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6] + ([0.7, 0.8, 0.9, 0.95] if GRID != 'old' else [])
if GRID == 'ext': THR = [0.05, 0.1] + THR + [0.98, 0.99]   # diagnostic only: beyond both ends of the wide grid


def record_tuning(name, value, table, grid, extra=None):
    """Log the chosen value of a tuned parameter, the score of every grid point, and whether the choice is interior.
    The first grid point with the best calibration score is chosen, so on a plateau the smallest value wins."""
    top = max(f for _, f in table); arg = [g for g, f in table if f >= top - 1e-12]
    if isinstance(value, (tuple, list)):
        dims = [sorted(set(g[d] for g in grid)) for d in range(len(value))]
        interior = [bool(dims[d][0] < value[d] < dims[d][-1]) for d in range(len(value))]
    else:
        interior = bool(min(grid) < value < max(grid))
    rec = {'name': name, 'chosen': value, 'score': float(top), 'interior': interior, 'grid_ends': [grid[0], grid[-1]],
           'n_grid_points_with_best_score': len(arg), 'best_points': arg[:40], 'table': table}
    if extra: rec.update(extra)
    log('TUNING', name, 'chosen', value, 'calibration set F1', round(float(top), 4), 'interior', interior,
        'grid points with the best score', len(arg), 'of', len(table))
    json.dump(rec, open(OUT + 'tuning_%s.json' % name, 'w'), indent=1, default=float)
    return rec


def tune_thr(pml, Zt_truth, name=None):
    best = None; tab = []
    for ta in THR:
        s = [[k + 1 for k in np.where(p > ta)[0]] for p in pml]; f = tune_sets(s, Zt_truth); tab.append((ta, round(f, 6)))
        if best is None or f > best[1]: best = (ta, f)
    if name:   # scores below the grid are logged for information only and never chosen
        low = [(ta, round(tune_sets([[k + 1 for k in np.where(p > ta)[0]] for p in pml], Zt_truth), 6)) for ta in (0.05, 0.1)]
        record_tuning(name, best[0], tab, THR, {'below_grid_not_used': low})
    return best


# ---------------- one calibration rule for every method (HYD_CAL=one) ----------------
# Every parameter that decides how many events a method names (probability thresholds of the multi-label methods, the
# continuation multiplier of PCA-RBC, the costs of AEC and MinExplain) is chosen on the calibration data of VioExplain:
# the normal and single-fault test-calibration cycles and the composed calibration cycles on which its addition decision
# is fitted (one, two or three events). The objective is the mean of the event-set F1 on the two sets. This holds in f0
# as well. No test cycle and no real multi-fault cycle is used.
CAL = os.environ.get('HYD_CAL', 'one'); CALSET = os.environ.get('HYD_CALSET', 'cal')
NCOMP = int(os.environ.get('HYD_NCOMP', '2700'))
THR_ONE = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
# below 0.005 the extension continues until the choice is interior; threshold 0 would name every event (natural bound)
THR_EXT = ([0.002, 0.001, 0.0005, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 1e-8, 1e-9, 1e-10], [0.998, 0.999, 0.9999])


def compose_raw(role, k_choices, n_target, by_iter, seed):
    """No-twin compositions of cycles of one role: x = x_n + sum_e (x_e - x_n) with random single-fault cycles x_e of
    distinct components and a random normal cycle x_n. Returns the draws (events, cycle picks, normal pick) and the raw
    cycles. by_iter: n_target draws, otherwise n_target accepted compositions."""
    r_ = np.random.default_rng(seed); batch = []; it = 0
    while (it < n_target) if by_iter else (len(batch) < n_target and it < 200000):
        it += 1
        k = int(r_.choice(k_choices)); comps = r_.choice(4, k, replace=False) + 1
        ev = [int(r_.choice(EVENTS_OF_COMP[int(c)])) for c in comps]
        if any(len(RAW[role][e]) == 0 for e in ev): continue
        picks = [int(r_.integers(len(RAW[role][e]))) for e in ev]; i0 = int(r_.integers(len(RAW[role][0])))
        batch.append((ev, picks, i0))
    Xc = np.stack([RAW[role][0][i0] + sum(RAW[role][e][j] - RAW[role][0][i0] for e, j in zip(ev, picks)) for ev, picks, i0 in batch])
    return batch, Xc


_CALC = {}


def calc():
    """Composed calibration cycles shared by all methods: raw cycles X, descriptions Z, component truth T. With
    HYD_CALSET=cal these are the compositions on which hyd_final.py fits the addition decision (same draws)."""
    if not _CALC:
        if CALSET == 'cal': batch, Xc = compose_raw('cal', [1, 2, 2, 3, 3], NCOMP, False, 5 + SEED)
        else: batch, Xc = compose_raw('tcal', [2, 3], 1500, False, 23 + SEED)
        _CALC.update(X=Xc, Z=Fn(Xc), T=[to_comp(ev) for ev, _, _ in batch])
        log('composed calibration cycles', CALSET, len(Xc), 'by size', dict(collections.Counter(len(ev) for ev, _, _ in batch)))
    return _CALC


def tune_one(name, sets_fn, dims, ext=None, natural=None, extra=None):
    """Grid search under the one calibration rule. sets_fn(g) returns the event sets on the normal and single-fault
    test-calibration cycles (TC_Z) and on the composed calibration cycles (calc()['Z']); g is a scalar for one
    dimension and a tuple otherwise. dims holds the grid values per dimension; the grid is their product, visited in
    this order, and a later point must be strictly better. ext holds, per dimension, the values tried beyond the lower
    and the upper end: while a grid point with the best objective (the choice or a point tied with it) has the
    smallest or largest value of a dimension, the next value on that side is added and the new grid points are
    evaluated, so a plateau that reaches the end of a grid is followed. natural[d](value) returns a note when a value is a bound of the
    parameter itself. Returns (chosen, objective)."""
    CC = calc(); nd = len(dims); dims = [list(d) for d in dims]
    ext = [(list(lo), list(up)) for lo, up in (ext or [([], [])] * nd)]
    cache = {}; best = [None, -1.0]

    def sweep():
        for key in itertools.product(*dims):
            if key in cache: continue
            s1, s2 = sets_fn(key[0] if nd == 1 else key)
            f1 = tune_sets(s1, TC_T); f2 = float(np.mean([setf1(to_comp(a), t) for a, t in zip(s2, CC['T'])]))
            f = 0.5 * f1 + 0.5 * f2; cache[key] = (f, f1, f2)
            if best[0] is None or f > best[1]: best[0], best[1] = key, f
    sweep(); rounds = 0
    while rounds < 40:
        grew = False; tied = [k for k, v in cache.items() if v[0] >= best[1] - 1e-12]
        for d in range(nd):
            vals = set(k[d] for k in tied)
            if min(dims[d]) in vals and ext[d][0]: dims[d] = [ext[d][0].pop(0)] + dims[d]; grew = True
            if max(dims[d]) in vals and ext[d][1]: dims[d] = dims[d] + [ext[d][1].pop(0)]; grew = True
        if not grew: break
        rounds += 1; sweep()
    key, f = best; ties = [k for k, v in cache.items() if v[0] >= f - 1e-12]
    interior = [bool(min(dims[d]) < key[d] < max(dims[d])) for d in range(nd)]
    also = [bool(any(min(dims[d]) < k[d] < max(dims[d]) for k in ties)) for d in range(nd)]
    notes = [natural[d](key[d]) if natural and natural[d] else None for d in range(nd)]
    one = nd == 1; value = key[0] if one else list(key)
    rec = {'name': name, 'rule': 'one', 'composed_set': CALSET, 'n_composed': len(CC['T']), 'n_single_tcal': len(TC_T),
           'chosen': value, 'objective': float(f), 'setF1_single_tcal': cache[key][1], 'setF1_composed': cache[key][2],
           'interior': interior[0] if one else interior, 'best_objective_also_at_interior_value': also[0] if one else also,
           'natural_bound': notes[0] if one else notes, 'grid_final': dims[0] if one else dims, 'extension_rounds': rounds,
           'n_grid_points': len(cache), 'n_grid_points_with_best_objective': len(ties),
           'best_points': [k[0] if one else list(k) for k in ties[:40]],
           'table': [(k[0] if one else list(k), round(v[0], 6), round(v[1], 6), round(v[2], 6)) for k, v in sorted(cache.items())]}
    if extra: rec.update(extra)
    log('TUNING one rule', name, 'chosen', value, 'objective', round(float(f), 4), 'single', round(cache[key][1], 4), 'composed',
        round(cache[key][2], 4), 'interior', rec['interior'], 'natural bound', rec['natural_bound'], 'ties', len(ties), 'of',
        len(cache), 'extension rounds', rounds)
    json.dump(rec, open(OUT + 'tuning_%s.json' % name, 'w'), indent=1, default=float)
    return (key[0] if one else key), float(f)


MLGATE = os.environ.get('HYD_MLGATE', '1') == '1'


def ml_gate(p_tcal):
    """Empty-set rule shared with every other method (ml_gate of common.py): a multi-label method names nothing
    on a cycle whose largest event probability does not exceed the split-conformal threshold of the normal
    test-calibration cycles, the ceil((1 - 0.05)(n + 1))-th smallest value of their largest event probability. A cycle is
    one window and there is no run structure, so every normal test-calibration cycle enters once. With n = 4 the index
    exceeds n and the threshold is the maximum."""
    v = np.sort(np.asarray(p_tcal)[TC_Y == 0].max(1)); n = len(v)
    return float(v[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)])


def ml_sets(P, ta, q0):
    return [[] if p.max() <= q0 else [k + 1 for k in np.where(p > ta)[0]] for p in P]


def tune_ml_probs(name, p_single, p_comp):
    """Probability threshold of a multi-label method under the one calibration rule. With HYD_MLGATE=1 the empty-set
    gate is computed first from the test-calibration probabilities and the threshold search applies it.
    Returns ((threshold, gate), objective); gate = -1 means no gate."""
    q0 = ml_gate(p_single) if MLGATE else -1.0
    ta, f = tune_one(name, lambda ta: (ml_sets(p_single, ta, q0), ml_sets(p_comp, ta, q0)), [THR_ONE], [THR_EXT],
                     extra={'empty_set_gate': q0, 'gate_rule': 'split-conformal, alpha 0.05, normal test-calibration cycles' if MLGATE else 'none'})
    log(name, 'empty-set gate', q0)
    return (ta, q0), f
