"""[scripts/simproc2 copy: the exact-representation threshold grid of AEC and MinExplain is extended from (0.7, 0.9, 0.97) to
(0.7, 0.9, 0.97, 0.99, 0.997), and the AEC and MinExplain parameters are searched by tune_axes, which extends every
grid outward while the optimum sits at its end (the instruction: grids extended until the optimum is interior)]
VioExplain v3 final runs, additional CPU baselines on the shared protocol of common.py.

Usage:  V3_MODE=f0 V3_LOG=log_base.txt V3_METRICS=metrics_base.json python final_base.py NAME [NAME ...]
NAME in
  RF, MC-LGBM, XGB  single-label classifiers on the window description, top-1 with a conformal empty set
  FDA       Fisher discriminant analysis (shrinkage LDA) on the window statistics, top-1 with a conformal empty set
  PCA-RBC   PCA monitoring with the combined index and reconstruction-based identification along fault directions
            learned from paired runs, extended greedily to several directions until the reconstructed index is in control
  CC-LGBM   classifier chain of gradient boosting models (same training data as BR-LGBM, incl. composition in f100t)
  AEC       anomaly-explanation covering on typed violations: exact and possible
            representations from paired runs, greedy weighted covering, thresholds tuned on test calibration runs
  MinExplain  probabilistic minimum-cost explanation (uncapacitated facility location with a free background facility)
            on typed violations, density greedy, opening cost tuned on test calibration runs
"""
import sys
from common import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import ClassifierChain

NAMES = sys.argv[1:]
THR = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
# calibration windows shared with VioExplain: single-fault test-calibration windows and compositions of calibration runs
CALC = cal_compositions(all_pairs=(PCT == 0)) if any(n_ in NAMES for n_ in ('BR-LGBM', 'CC-LGBM', 'PCA-RBC', 'AEC', 'MinExplain')) else None
CSUB = np.arange(0, 2700, 2) if CALC is not None else None
TSUB = np.arange(0, len(TC_Z), 3)   # calibration windows used for parameter search of the slow covering baselines


def top1_gate(P):
    q0 = conformal_empty(P)
    return lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Zw)]


def tune_on_tcal(make_fn, grid, sub=None):
    """Parameters that decide how many events a method names: maximize the mean of the event-set F1 on single-fault
    test-calibration windows and on composed calibration windows."""
    best = None; Zs = TC_Z if sub is None else TC_Z[sub]; Ts = TC_T if sub is None else [TC_T[i] for i in sub]
    Zc = CALC[1][CSUB[CSUB < len(CALC[1])]]; Tc = [CALC[2][i] for i in CSUB[CSUB < len(CALC[1])]]
    for g in grid:
        fn = make_fn(g); f1 = np.mean([setf1(a, b_) for a, b_ in zip(fn(Zs), Ts)]); f2 = np.mean([setf1(a, b_) for a, b_ in zip(fn(Zc), Tc)])
        f = 0.5 * f1 + 0.5 * f2; log('  grid', g, round(float(f1), 4), round(float(f2), 4))
        if best is None or f > best[1]: best = (g, f)
    log('tuned', best, 'grid ends', grid[0], grid[-1])
    return make_fn(best[0])


def tune_axes(make_fn, axes, ext, sub=None, rounds=8):
    """tune_on_tcal over the product of per-parameter grids (same objective, same windows, first maximum in product
    order). While the optimum sits at an end of a grid that can be extended (ext[i](value, side) gives the next value
    outward, None at a natural bound), that grid is extended and the product is searched again (scores are cached)."""
    Zs = TC_Z if sub is None else TC_Z[sub]; Ts = TC_T if sub is None else [TC_T[i] for i in sub]
    Zc = CALC[1][CSUB[CSUB < len(CALC[1])]]; Tc = [CALC[2][i] for i in CSUB[CSUB < len(CALC[1])]]
    cache = {}; axes = [sorted(a) for a in axes]

    def score(g):
        if g not in cache:
            fn = make_fn(g); f1 = np.mean([setf1(a, b_) for a, b_ in zip(fn(Zs), Ts)]); f2 = np.mean([setf1(a, b_) for a, b_ in zip(fn(Zc), Tc)])
            cache[g] = 0.5 * f1 + 0.5 * f2; log('  grid', g, round(float(f1), 4), round(float(f2), 4))
        return cache[g]
    for r in range(rounds + 1):
        prod = list(itertools.product(*axes)); vals = [score(g) for g in prod]; best = prod[int(np.argmax(vals))]
        changed = False
        for i, a in enumerate(axes):
            for end, side in ((a[0], 'lo'), (a[-1], 'hi')):
                if best[i] == end and r < rounds:
                    nv = ext[i](end, side)
                    if nv is not None and nv not in a: a.append(nv); a.sort(); changed = True
        if not changed: break
    inter = [a[0] < b < a[-1] or ext[i](b, 'lo' if b == a[0] else 'hi') is None for i, (a, b) in enumerate(zip(axes, best))]
    log('tuned', (best, cache[best]), 'grids', axes, 'interior or natural bound per parameter', inter, 'evaluated', len(cache))
    return make_fn(best)


E_TH = lambda v, s: (v - 0.2 if v - 0.2 >= 0.05 else None) if s == 'lo' else (1 - (1 - v) / 3 if v < 0.9999 else None)
E_MUL = lambda f, lo0=False: (lambda v, s: ((v / f if v > 1e-3 else None) if not lo0 else (None if v == 0 else (v / f if v >= 0.03 else 0.0))) if s == 'lo' else v * f)
from sklearn.ensemble import RandomForestClassifier
for bname in ('RF', 'MC-LGBM', 'XGB'):
    if bname not in NAMES: continue
    if bname == 'RF': model = RandomForestClassifier(500, n_jobs=NJ, random_state=0)
    elif bname == 'MC-LGBM': model = mk_lgb()
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, tree_method='hist')
    model.fit(FLAT(XB), YB); P = lambda Zw, m=model: m.predict_proba(FLAT(Zw))
    RES[bname + '_label_macroF1'] = float(f1_score(EV_Y, P(EV_Z).argmax(1), average='macro'))
    evaluate(bname + '-top1', top1_gate(P))

if 'FDA' in NAMES:
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(FLAT(XB), YB)
    P = lambda Zw: lda.predict_proba(FLAT(Zw))
    RES['FDA_label_macroF1'] = float(f1_score(EV_Y, P(EV_Z).argmax(1), average='macro'))
    evaluate('FDA', top1_gate(P))

if 'PCA-RBC' in NAMES:
    XN = FLAT(Z['tr'][0][ok_tr[0]]).astype(np.float64); mu_n = XN.mean(0); Xc = XN - mu_n
    S = np.cov(Xc.T); ev_, V_ = np.linalg.eigh(S); o = np.argsort(ev_)[::-1]; ev_ = ev_[o]; V_ = V_[:, o]
    l = int(np.searchsorted(np.cumsum(ev_) / ev_.sum(), 0.90)) + 1; Pm = V_[:, :l]; lam = ev_[:l]
    # combined index phi = T2 / mean(T2) + SPE / mean(SPE) on normal fit windows (Yue and Qin)
    T2m = Pm @ np.diag(1 / lam) @ Pm.T; Q = np.eye(len(mu_n)) - Pm @ Pm.T
    t2 = np.einsum('ij,jk,ik->i', Xc, T2m, Xc); spe = np.einsum('ij,jk,ik->i', Xc, Q, Xc)
    PHI = T2m / t2.mean() + Q / spe.mean()
    XI = {}
    for h in range(1, C):
        g = ok_tr[h]; D = (FLAT(Z['tr'][h][g]) - FLAT(Z['tr'][0][g])).astype(np.float64)
        U, s, _ = np.linalg.svd(D.T, full_matrices=False); r = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.8)) + 1
        XI[h] = U[:, :min(r, 3)]
    log('PCA-RBC components', l, 'fault direction ranks', {h: XI[h].shape[1] for h in XI})

    def phi(X): return np.einsum('ij,jk,ik->i', X, PHI, X)

    def rec_index(X, H):
        Xi = np.concatenate([XI[h] for h in H], 1); A = Xi.T @ PHI @ Xi
        f = np.linalg.lstsq(A, Xi.T @ PHI @ X.T, rcond=None)[0]; return phi(X - (Xi @ f).T)

    zs = Z['tcal'][0]; per = len(zs) // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
    s0 = np.sort(phi(FLAT(zs[pick]) - mu_n)); LIM = s0[min(len(s0) - 1, int(np.ceil(0.95 * (len(s0) + 1))) - 1)]

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
                rows = np.array(rows); best = np.full(len(rows), np.inf); arg = np.zeros(len(rows), int)
                for h in range(1, C):
                    if h in Hs: continue
                    v = rec_index(X[rows], list(Hs) + [h]); b = v < best; best[b] = v[b]; arg[b] = h
                for n_, i in enumerate(rows):
                    S[i].append(int(arg[n_]))
                    if best[n_] > kappa * LIM: nxt.append(i)
            act = np.array(nxt, int)
        return S
      return rbc
    evaluate('PCA-RBC', tune_on_tcal(make_rbc, [0.25, 0.5, 1, 2, 4, 8, 16, 64, 256, 1024, 1e12], TSUB))

if 'CC-LGBM' in NAMES:
    Xml = [XB]; Yml = [np.zeros((len(YB), K), int)]
    Yml[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = 1
    for ev, Zs, Zrest, idx in AUG:
        Y = np.zeros((len(Zs), K), int); Y[:, np.array(ev) - 1] = 1; Xml.append(Zs); Yml.append(Y)
    Xml = np.concatenate(Xml); Yml = np.concatenate(Yml)
    cc = ClassifierChain(mk_lgb(300), order='random', random_state=0).fit(FLAT(Xml), Yml)
    ptc = cc.predict_proba(FLAT(TC_Z)); q_cc = ml_gate(ptc); log('CC empty-set gate', q_cc)
    ta_cc = tune_threshold(ptc, cc.predict_proba(FLAT(CALC[1])), CALC[2], THR, 'CC', q_cc)
    evaluate('CC-LGBM', lambda Zw, ta=ta_cc, q0=q_cc: ml_sets(cc.predict_proba(FLAT(Zw)), ta, q0))

if 'BR-LGBM' in NAMES:
    Xml = [XB]; Yml = [np.zeros((len(YB), K), bool)]
    Yml[0][np.arange(len(YB))[YB > 0], YB[YB > 0] - 1] = True
    for ev, Zs, Zrest, idx in AUG:
        Y = np.zeros((len(Zs), K), bool); Y[:, np.array(ev) - 1] = True; Xml.append(Zs); Yml.append(Y)
    Xml = np.concatenate(Xml); Yml = np.concatenate(Yml)
    brs = [mk_lgb(300).fit(FLAT(Xml), Yml[:, k].astype(int)) for k in range(K)]
    brp = lambda Zw: np.stack([m.predict_proba(FLAT(Zw))[:, 1] for m in brs], 1)
    ptb = brp(TC_Z); q_br = ml_gate(ptb); log('BR empty-set gate', q_br)
    ta_br = tune_threshold(ptb, brp(CALC[1]), CALC[2], THR, 'BR', q_br)
    evaluate('BR-LGBM', lambda Zw, ta=ta_br, q0=q_br: ml_sets(brp(Zw), ta, q0))

if 'AEC' in NAMES or 'MinExplain' in NAMES:
    # violation knowledge from paired runs: Pr_h(k) = share of fit windows in which h causes typed violation k
    V0 = tviol_all(Z['tr'][0]); NK = V0.shape[1]
    PR = np.zeros((C, NK))
    for h in range(1, C):
        g = ok_tr[h]; PR[h] = (tviol_all(Z['tr'][h][g]) & ~V0[g]).mean(0)
    P0 = np.clip(V0.mean(0), 1e-4, None)

if 'AEC' in NAMES:
    def make_aec(g):
        th_e, th_p, lam, tau = g
        EX = PR[1:] >= th_e; PO = PR[1:] >= th_p; Wt = np.where(PO, PR[1:], 0.0)

        def aec(Zw, maxk=5):
            V = tviol_all(Zw); out = []
            miss = (EX[None] & ~V[:, None, :]).sum(2)            # exact violations absent from the window
            for i in range(len(V)):
                unc = V[i].copy(); S = []
                for _ in range(maxk):
                    sc = Wt[:, unc].sum(1) - lam * miss[i]
                    for e in S: sc[e - 1] = -np.inf
                    j = int(np.argmax(sc))
                    if sc[j] < tau: break
                    S.append(j + 1); unc &= ~PO[j]
                    if not unc.any(): break
                out.append(S)
            return out
        return aec
    E_P = lambda v, s: (v / 2 if v > 1e-4 else None) if s == 'lo' else (min(v * 2.5, 0.9) if v < 0.9 else None)
    E_TAU = lambda v, s: (v - 5 if v > -50 else None) if s == 'lo' else (v * 2.5 if v > 0 else None)
    evaluate('AEC', tune_axes(make_aec, [(0.7, 0.9, 0.97, 0.99, 0.997), (0.005, 0.01, 0.02, 0.05, 0.2), (0.5, 2.0, 5.0, 10.0, 20.0),
                                         (-5.0, -2.0, -0.5, 0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.5)],
                              [E_TH, E_P, E_MUL(4, True), E_TAU], TSUB))

if 'MinExplain' in NAMES:
    def make_minexp(g):
        w, mu, th_e = g
        cost = -np.log(np.clip(PR[1:], 1e-4, 1.0)); c0 = -np.log(P0); EXm = PR[1:] >= th_e

        def minexp(Zw, maxk=5):
            V = tviol_all(Zw); out = []
            miss = (EXm[None] & ~V[:, None, :]).sum(2)            # exact representations absent from the window
            for i in range(len(V)):
                ks = np.where(V[i])[0]; cur = c0[ks].copy(); S = []
                for _ in range(maxk):
                    if len(ks) == 0: break
                    gain = np.maximum(cur[None] - cost[:, ks], 0).sum(1) - w - mu * miss[i]
                    for e in S: gain[e - 1] = -np.inf
                    j = int(np.argmax(gain))
                    if gain[j] <= 0: break
                    S.append(j + 1); cur = np.minimum(cur, cost[j, ks])
                out.append(S)
            return out
        return minexp
    evaluate('MinExplain', tune_axes(make_minexp, [(0.25, 1, 4, 16, 64, 256), (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0), (0.7, 0.9, 0.97, 0.99, 0.997)],
                                     [E_MUL(4), E_MUL(4, True), E_TH], TSUB))
log('DONE', NAMES)
