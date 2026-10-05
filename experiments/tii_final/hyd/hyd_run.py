"""VioExplain and baselines on the hydraulic benchmark (protocol of final_run.py, final_base.py, final_rocket.py).

Usage:  HYD_MODE=f0 python hyd_run.py NAME [NAME ...]
NAME in top1 (RF, MC-LGBM, XGB top-1, f0 only), BR, ours, FDA, PCA-RBC, CC, AEC, MinExplain, MiniRocket, CNN
"""
import sys
from hyd_common import *

NAMES = sys.argv[1:]


def top1_gate(P):
    q0 = conformal_empty(P)
    return lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Zw)]


def tune_on_tcal(make_fn, grid, name=None, extend=False):
    """Tuning on the normal and single-fault test-calibration cycles. The first grid point with the best set F1 wins.
    extend=True (scalar grids): while the choice is the first or last grid point, one more point is tried beyond that
    end (half the first, four times the last) and it is taken only if it is strictly better."""
    grid = list(grid); best = None; tab = []

    def score(g):
        f = tune_sets(make_fn(g)(TC_Z), TC_T); tab.append((g, round(f, 6))); return f
    for g in grid:
        f = score(g)
        if best is None or f > best[1]: best = (g, f)
    tried = []
    while extend and len(tried) < 8 and best[0] in (grid[0], grid[-1]):
        up = best[0] == grid[-1]; g = grid[-1] * 4 if up else grid[0] / 2; f = score(g); tried.append((g, round(f, 6)))
        grid = grid + [g] if up else [g] + grid
        if f > best[1]: best = (g, f)
        else: break
    log('tuned', best)
    if name: record_tuning(name, best[0], sorted(tab) if extend else tab, grid, {'points_tried_beyond_the_given_grid': tried})
    return make_fn(best[0])


def label_f1(name, P):
    RES[name + '_label_macroF1'] = float(f1_score(T01_Y, P(ZT[T01]).argmax(1), average='macro'))
    log(name, 'label macro F1 (normal+single test cycles)', RES[name + '_label_macroF1'])
    json.dump({'label_macroF1': RES[name + '_label_macroF1']}, open(OUT + 'label_%s.json' % name, 'w'))


# ---------------- single-label baselines ----------------
if 'top1' in NAMES and MODE == 'f0':
    from sklearn.ensemble import RandomForestClassifier
    cw = None if CW == 'none' else 'balanced'
    for bname, model in (('RF', RandomForestClassifier(500, n_jobs=NJ, random_state=0, class_weight=cw)), ('MC-LGBM', mk_lgb())):
        model.fit(FLAT(XB), YB); P = lambda Zw, m=model: m.predict_proba(FLAT(Zw))
        label_f1(bname, P); evaluate(bname + '-top1', top1_gate(P))
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight
    xm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.5, n_jobs=NJ, tree_method='hist')
    xm.fit(FLAT(XB), YB, sample_weight=None if CW == 'none' else compute_sample_weight('balanced', YB))
    P = lambda Zw: xm.predict_proba(FLAT(Zw)); label_f1('XGB', P); evaluate('XGB-top1', top1_gate(P))

# ---------------- multi-label baselines ----------------
TUNE_Z, TUNE_T = (TCA_Z, TCA_T) if MODE == 'fall' else (TC_Z, TC_T)
ONE = CAL == 'one' and MODE != 'fall'   # one calibration rule (hyd_common.py); fall keeps its supervised tuning


def tune_ml(name, P):
    """Probability threshold and empty-set gate of a multi-label method; P maps descriptions to event probabilities.
    Returns (threshold, gate, objective); gate = -1 means no gate (HYD_CAL=single)."""
    if not ONE:
        ta, f = tune_thr(P(TUNE_Z), TUNE_T, name); return ta, -1.0, f
    (ta, q0), f = tune_ml_probs(name, P(TC_Z), P(calc()['Z'])); return ta, q0, f


def tune_fn(name, make_fn, dims, ext, natural=None):
    g, _ = tune_one(name, lambda g_: (make_fn(g_)(TC_Z), make_fn(g_)(calc()['Z'])), dims, ext, natural)
    return make_fn(g)
if 'BR' in NAMES:
    best = tune_ml('BR-LGBM', br_probs); log('BR threshold', best)
    evaluate('BR-LGBM', lambda Zw: ml_sets(br_probs(Zw), best[0], best[1]))

if 'CC' in NAMES:
    from sklearn.multioutput import ClassifierChain
    cc = ClassifierChain(mk_lgb(300), order='random', random_state=0).fit(FLAT(Xml), Yml)
    bestc = tune_ml('CC-LGBM', lambda Zw: cc.predict_proba(FLAT(Zw))); log('CC threshold', bestc)
    evaluate('CC-LGBM', lambda Zw: ml_sets(cc.predict_proba(FLAT(Zw)), bestc[0], bestc[1]))

if 'RFML' in NAMES:
    from sklearn.ensemble import RandomForestClassifier
    rfm = RandomForestClassifier(500, n_jobs=NJ, random_state=0).fit(FLAT(Xml), Yml)
    PR_ = lambda Zw: np.stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(p)) for p in rfm.predict_proba(FLAT(Zw))], 1)
    bestr = tune_ml('RF-ML', PR_); log('RF-ML threshold', bestr)
    evaluate('RF-ML', lambda Zw: ml_sets(PR_(Zw), bestr[0], bestr[1]))

# ---------------- FDA ----------------
if 'FDA' in NAMES and MODE == 'f0':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(FLAT(XB), YB)
    P = lambda Zw: lda.predict_proba(FLAT(Zw)); label_f1('FDA', P); evaluate('FDA', top1_gate(P))

# ---------------- PCA with reconstruction-based identification ----------------
if 'PCA-RBC' in NAMES and MODE == 'f0':
    # normal operating model: centre = mean of normal fit cycles; covariance = pooled within-condition covariance of
    # the normal and single-fault fit cycles (only %d normal fit cycles exist, too few for a 119-dimensional covariance)
    mu_n = FLAT(Z['fit'][0]).astype(np.float64).mean(0)
    Rs = []
    for ii in FIT_GROUPS:
        Fg = FLAT(Fn(XALL[ii])).astype(np.float64); Rs.append(Fg - Fg.mean(0))
    Rs = np.concatenate(Rs); S = Rs.T @ Rs / (len(Rs) - len(FIT_GROUPS))
    ev_, V_ = np.linalg.eigh(S); o = np.argsort(ev_)[::-1]; ev_ = np.clip(ev_[o], 1e-8, None); V_ = V_[:, o]
    l = int(np.searchsorted(np.cumsum(ev_) / ev_.sum(), 0.90)) + 1; Pm = V_[:, :l]; lam = ev_[:l]
    T2m = Pm @ np.diag(1 / lam) @ Pm.T; Q = np.eye(len(mu_n)) - Pm @ Pm.T
    t2 = np.einsum('ij,jk,ik->i', Rs, T2m, Rs); spe = np.einsum('ij,jk,ik->i', Rs, Q, Rs)
    PHI = T2m / t2.mean() + Q / spe.mean()
    XI = {}
    for h in range(1, C):
        D = (FLAT(Z['fit'][h]) - mu_n).astype(np.float64)
        U, s, _ = np.linalg.svd(D.T, full_matrices=False); r = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.8)) + 1
        XI[h] = U[:, :min(r, 3)]
    log('PCA-RBC components', l, 'fault direction ranks', {h: XI[h].shape[1] for h in XI})

    def phi(X): return np.einsum('ij,jk,ik->i', X, PHI, X)

    def rec_index(X, H):
        Xi = np.concatenate([XI[h] for h in H], 1); A = Xi.T @ PHI @ Xi
        f = np.linalg.lstsq(A, Xi.T @ PHI @ X.T, rcond=None)[0]; return phi(X - (Xi @ f).T)

    s0 = np.sort(phi(FLAT(Z['tcal'][0]).astype(np.float64) - mu_n)); LIM = s0[min(len(s0) - 1, int(np.ceil(0.95 * (len(s0) + 1))) - 1)]

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
                        if h in Hs or not comp_ok(list(Hs) + [h]): continue
                        v = rec_index(X[rows], list(Hs) + [h]); b = v < best; best[b] = v[b]; arg[b] = h
                    for n_, i in enumerate(rows):
                        if arg[n_] == 0: continue
                        S[i].append(int(arg[n_]))
                        if best[n_] > kappa * LIM: nxt.append(i)
                act = np.array(nxt, int)
            return S
        return rbc
    if ONE:
        evaluate('PCA-RBC', tune_fn('PCA-RBC', make_rbc, [[0.25, 0.5, 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 1e12]], [([0.125, 0.0625, 0.03125], [])],
                                    [lambda v: 'at 1e12 the search stops after the first event' if v >= 1e12 else None]))
    else:
        evaluate('PCA-RBC', tune_on_tcal(make_rbc, [1, 2, 4, 8, 16, 64] + ([256, 1024, 4096, 1e12] if GRID != 'old' else []), 'PCA-RBC'))

# ---------------- AEC and MinExplain on typed violations ----------------
if ('AEC' in NAMES or 'MinExplain' in NAMES) and MODE == 'f0':
    V0 = tviol_z(Z['fit'][0]); NKV = V0.shape[1]
    PR = np.zeros((C, NKV))
    for h in range(1, C): PR[h] = np.clip(tviol_z(Z['fit'][h]).mean(0) - V0.mean(0), 0, 1)
    P0 = np.clip(V0.mean(0), 1e-4, None)

if 'AEC' in NAMES and MODE == 'f0':
    def make_aec(g):
        th_e, th_p, lam, tau = g
        EX = PR[1:] >= th_e; PO = PR[1:] >= th_p; Wt = np.where(PO, PR[1:], 0.0)

        def aec(Zw, maxk=5):
            V = tviol_z(Zw); out = []
            miss = (EX[None] & ~V[:, None, :]).sum(2)
            for i in range(len(V)):
                unc = V[i].copy(); S = []
                for _ in range(maxk):
                    sc = Wt[:, unc].sum(1) - lam * miss[i]
                    for e in S: sc[e - 1] = -np.inf
                    for e in range(1, C):
                        if not comp_ok(S + [e]): sc[e - 1] = -np.inf
                    j = int(np.argmax(sc))
                    if sc[j] < tau: break
                    S.append(j + 1); unc &= ~PO[j]
                    if not unc.any(): break
                out.append(S)
            return out
        return aec
    grid = [(e, p, l, t) for e in (0.7, 0.9) for p in (0.05, 0.2) for l in (0.5, 2.0) for t in (0.5, 1.5, 3.0)]
    if GRID != 'old':
        grid = [(e, p, l, t) for e in (0.7, 0.9, 0.97) for p in (0.02, 0.05, 0.2) for l in (0.5, 2.0, 5.0, 10.0) for t in (0.1, 0.25, 0.5, 1.5)]
    if GRID == 'ext':   # diagnostic only: values beyond every end of the wide grid
        grid = [(e, p, l, t) for e in (0.5, 0.7, 0.9, 0.97, 0.99) for p in (0.005, 0.01, 0.02, 0.05, 0.2, 0.5)
                for l in (0.1, 0.25, 0.5, 2.0, 5.0, 10.0, 20.0) for t in (0.02, 0.05, 0.1, 0.25, 0.5, 1.5, 3.0)]
    if ONE:
        minpos = float(PR[1:][PR[1:] > 0].min())
        TZ = os.environ.get('HYD_AEC_TAUZERO', '0') == '1'   # diagnostic: tau = 0 in the grid, extension into negative tau
        evaluate('AEC', tune_fn('AEC', make_aec, [(0.7, 0.9, 0.97), (0.005, 0.01, 0.02, 0.05, 0.2), (0.5, 2.0, 5.0, 10.0, 20.0),
                                                  ((0.0,) if TZ else ()) + (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.5)],
                                [([0.5, 0.3], [0.99, 1.0]), ([0.002, 0.001], [0.5]), ([0.25, 0.1, 0.0], [50.0, 100.0]),
                                 ([-0.1, -0.25, -0.5, -1.0, -2.0] if TZ else [0.001, 0.0, -0.1, -0.25, -0.5, -1.0, -2.0, -5.0, -10.0, -1e9], [3.0, 5.0])],
                                [lambda v: 'th_e = 1 keeps only violations present in every fit cycle of the event' if v >= 1.0 else None,
                                 lambda v: 'th_p is at or below the smallest positive violation rate %.4f; lower values give the same sets' % minpos if v <= minpos else None,
                                 lambda v: 'lam = 0 removes the penalty' if v == 0 else None,
                                 lambda v: ('tau = 0 also accepts events with net score 0 (no covered weight and no penalty) while violations remain '
                                            'uncovered' if v == 0 else 'tau = -1e9: no score threshold, events are added while violations remain uncovered' if v <= -1e9 else
                                            'a negative tau accepts events whose penalty exceeds their covered weight' if v < 0 else None)]))
    else:
        evaluate('AEC', tune_on_tcal(make_aec, grid, 'AEC'))

if 'MinExplain' in NAMES and MODE == 'f0':
    def make_minexp(g):
        # g = (w, mu, th_e) as in final_base.py: opening cost w plus mu per exact representation (rate >= th_e) that the
        # cycle does not show; a scalar g is the opening cost alone
        w, mu, th_e = g if isinstance(g, tuple) else (g, 0.0, 2.0)
        cost = -np.log(np.clip(PR[1:], 1e-4, 1.0)); c0 = -np.log(P0); EXm = PR[1:] >= th_e

        def minexp(Zw, maxk=5):
            V = tviol_z(Zw); out = []
            miss = (EXm[None] & ~V[:, None, :]).sum(2)
            for i in range(len(V)):
                ks = np.where(V[i])[0]; cur = c0[ks].copy(); S = []
                for _ in range(maxk):
                    if len(ks) == 0: break
                    gain = np.maximum(cur[None] - cost[:, ks], 0).sum(1) - w - (mu * miss[i] if mu else 0.0)
                    for e in range(1, C):
                        if e in S or not comp_ok(S + [e]): gain[e - 1] = -np.inf
                    j = int(np.argmax(gain))
                    if gain[j] <= 0: break
                    S.append(j + 1); cur = np.minimum(cur, cost[j, ks])
                out.append(S)
            return out
        return minexp
    if ONE:
        evaluate('MinExplain', tune_fn('MinExplain', make_minexp, [(0.25, 1, 4, 16, 64, 256), (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0), (0.7, 0.9, 0.97)],
                                       [([0.0625, 0.0], [1024, 4096]), ([], [1000.0]), ([0.5, 0.3], [0.99, 1.0])],
                                       [lambda v: 'w = 0 opens an event at no cost' if v == 0 else None,
                                        lambda v: 'mu = 0 charges nothing for absent exact representations; th_e then has no effect' if v == 0 else None,
                                        lambda v: 'th_e = 1 keeps only violations present in every fit cycle of the event' if v >= 1.0 else None]))
    else:
        evaluate('MinExplain', tune_on_tcal(make_minexp, [8, 16, 32, 64, 128, 256, 512, 1024, 4096] if GRID != 'old' else [1, 2, 4, 8, 16, 32],
                                            'MinExplain', extend=GRID != 'old'))

# ---------------- MiniRocket ----------------
nm = RAW['fit'][0].reshape(-1, M); PM_ = nm.mean(0)
_res = np.concatenate([(XALL[ii] - XALL[ii].mean(0)).reshape(-1, M) for ii in FIT_GROUPS]); PSD_ = _res.std(0) + 1e-8


def prep(X): return ((X - PM_) / PSD_).astype(np.float32).transpose(0, 2, 1).copy()


def raw_of(Zw):
    return GRAW[KEYOF[id(Zw)]] if id(Zw) in KEYOF else None


if 'MiniRocket' in NAMES and MODE == 'f0':
    from aeon.classification.convolution_based import MiniRocketClassifier
    clf = MiniRocketClassifier(n_jobs=NJ, random_state=0).fit(prep(XBR), YB)
    Pz = {}

    def Prk(Zw):
        k = KEYOF[id(Zw)] if id(Zw) in KEYOF else None
        if k is None: raise RuntimeError('raw windows unknown')
        if k not in Pz: Pz[k] = clf.predict_proba(prep(GRAW[k]))
        return Pz[k]
    s = np.sort(1 - clf.predict_proba(prep(RAW['tcal'][0]))[:, 0]); n = len(s)
    q0 = 1 - s[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
    pt = Prk(ZT); RES['MiniRocket_label_macroF1'] = float(f1_score(T01_Y, pt[T01].argmax(1), average='macro'))
    log('MiniRocket label macro F1', RES['MiniRocket_label_macroF1'])
    evaluate('MiniRocket', lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in Prk(Zw)])

# ---------------- small 1D-CNN (CPU torch) ----------------
if 'CNN' in NAMES:
    import torch, torch.nn as nn
    torch.set_num_threads(min(NJ, 8)); torch.manual_seed(SEED)
    multilabel = MODE != 'f0'

    def net():
        return nn.Sequential(nn.Conv1d(M, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                             nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                             nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                             nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, K if multilabel else C))
    if multilabel:
        Xr = [XBR]; Yr = [Yml[:len(XB)]]
        for ev, Xs in AUG_RAW:
            Y = np.zeros((len(Xs), K), int); Y[:, np.array(ev) - 1] = 1; Xr.append(Xs); Yr.append(Y)
        if MODE == 'fall':
            FM = np.where((ROLE == 'fit') & (NEV >= 2))[0]; Xr.append(XALL[FM]); Yr.append(Yml[-len(FM):])
        Xtr = torch.tensor(prep(np.concatenate(Xr))); Ytr = torch.tensor(np.concatenate(Yr), dtype=torch.float32)
    else:
        Xtr = torch.tensor(prep(XBR)); Ytr = torch.tensor(YB, dtype=torch.long)
    model = net(); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(300):
        model.train(); perm = torch.randperm(len(Xtr), generator=gen)
        for b in range(0, len(Xtr), 64):
            j = perm[b:b + 64]
            if len(j) < 2: continue
            opt.zero_grad(); l_ = lossf(model(Xtr[j]), Ytr[j]); l_.backward(); opt.step()
    model.eval()

    def Pcnn_raw(X):
        with torch.no_grad():
            o = model(torch.tensor(prep(X)))
            return (torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).numpy()
    if multilabel:
        tz = TCA if MODE == 'fall' else TC1
        bestn = tune_thr(Pcnn_raw(XALL[tz]), [TRUTH[i] for i in tz]); log('CNN threshold', bestn)
        evaluate('1D-CNN', lambda Zw: [[k + 1 for k in np.where(p > bestn[0])[0]] for p in Pcnn_raw(raw_of(Zw))])
    else:
        s = np.sort(1 - Pcnn_raw(RAW['tcal'][0])[:, 0]); n = len(s); q0c = 1 - s[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
        RES['CNN_label_macroF1'] = float(f1_score(T01_Y, Pcnn_raw(GRAW['test'])[T01].argmax(1), average='macro'))
        log('CNN label macro F1', RES['CNN_label_macroF1'])
        evaluate('1D-CNN', lambda Zw: [[] if p[0] >= q0c else [int(np.argmax(p[1:])) + 1] for p in Pcnn_raw(raw_of(Zw))])

# ---------------- VioExplain ----------------
if 'ours' in NAMES and MODE != 'fall':
    Xa = [XB]; ya = [YB]
    for ev, Zs, Zrest, _ in AUG:
        if Zrest is None: continue
        Xa.append(deduct(Zs, ev[1])); ya.append(np.full(len(Zs), ev[0]))
    for tri, _, Zs in TRI_TRAIN:
        for keep in tri:
            R = Zs.copy()
            for g in tri:
                if g != keep: R = deduct(R, g)
            Xa.append(R); ya.append(np.full(len(R), keep))
    for h in range(1, C):
        if OP[h] is not None:
            nh = len(Z['fit'][h]); i2 = rng.choice(nh, max(1, nh // 3), replace=False)
            Xa.append(deduct(Z['fit'][h][i2], h)); ya.append(np.zeros(len(i2), int))
    Xa = np.concatenate(Xa); ya = np.concatenate(ya)
    SC = mk_lgb(); SC.fit(FLAT(Xa), ya)
    PS = lambda Zw: SC.predict_proba(FLAT(Zw))
    label_f1('Ours', PS)

    def cand_matrix(Zw, R, S_list, BRW):
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

    def blocked(S):
        """events that cannot be added: already in S or of a component already in S (severity events only)."""
        cs = set(int(COMPOF[g]) for g in S); return [e for e in range(1, C) if e in S or int(COMPOF[e]) in cs]

    # cooperative add decision trained on compositions of calibration cycles (no-twin: random single-fault cycles of
    # each event and a random normal cycle; truth = composed set)
    crng = np.random.default_rng(5 + SEED)
    batch = []
    for it in range(3000):
        k = int(crng.choice([1, 2, 2, 3, 3, 4])); comps = crng.choice(4, k, replace=False) + 1
        ev = [int(crng.choice(EVENTS_OF_COMP[int(c)])) for c in comps]
        if any(len(RAW['cal'][e]) == 0 for e in ev): continue
        picks = [int(crng.integers(len(RAW['cal'][e]))) for e in ev]; i0 = int(crng.integers(len(RAW['cal'][0])))
        batch.append((ev, picks, i0))
    Xc = np.stack([RAW['cal'][0][i0] + sum(RAW['cal'][e][j] - RAW['cal'][0][i0] for e, j in zip(ev, picks)) for ev, picks, i0 in batch])
    Zc = Fn(Xc); truth = [set(ev) for ev, _, _ in batch]
    p = PS(Zc); first = p[:, 1:].argmax(1) + 1
    keep = np.array([f in t for f, t in zip(first, truth)])
    log('cal compositions', len(batch), 'first pick correct', float(keep.mean()))
    Zc = Zc[keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]
    S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = br_probs(Zc)
    active = np.arange(len(Zc)); rows = []; lab = []
    for step in range(4):
        if len(active) == 0: break
        F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active])
        nxt_active = []
        for n_, i in enumerate(active):
            bl = set(blocked(S_list[i]))
            for e in range(1, C):
                if e in bl: continue
                rows.append(F[n_, e - 1]); lab.append(int(e in truth[i]))
            rem = [e for e in truth[i] if e not in S_list[i]]
            if rem:
                nxt = max(rem, key=lambda e: F[n_, e - 1, 0]); S_list[i].append(nxt); nxt_active.append(i)
        if nxt_active:
            idx = np.array(nxt_active); R[idx] = deduct_groups(R[idx], [S_list[i][-1] for i in idx])
        active = np.array(nxt_active, int)
    DEC = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
    log('cooperative decision rows', len(lab), 'positives', int(np.sum(lab)))
    single_idx = np.where(TC_Y > 0)[0]
    ps_ = PS(TC_Z[single_idx]); fp = ps_[:, 1:].argmax(1) + 1; good = single_idx[fp == TC_Y[single_idx]]
    Rg = deduct_groups(TC_Z[good], TC_Y[good]); Fg = cand_matrix(TC_Z[good], Rg, [[int(y)] for y in TC_Y[good]], br_probs(TC_Z[good]))
    pp = DEC.predict_proba(Fg.reshape(-1, 8))[:, 1].reshape(len(good), K)
    for n_, y in enumerate(TC_Y[good]):
        for e in blocked([int(y)]): pp[n_, e - 1] = -1
    TAU_C = float(np.quantile(pp.max(1), 0.95))
    TAU0 = conformal_empty(PS)
    log('tau0', TAU0, 'tau_c', TAU_C, 'tcal singles with correct first pick', len(good), 'of', len(single_idx))
    json.dump({'tau0': TAU0, 'tau_c': TAU_C}, open(OUT + 'thresholds_ours.json', 'w'))

    def explain(Zw, maxk=5, P1=None, tau0=None):
        p = PS(Zw) if P1 is None else P1(Zw); n = len(Zw); tau0 = TAU0 if tau0 is None else tau0
        S_list = [[] if p[i, 0] >= tau0 else [int(np.argmax(p[i, 1:])) + 1] for i in range(n)]
        act = np.array([i for i in range(n) if S_list[i]], int)
        if len(act):
            R = Zw.copy(); R[act] = deduct_groups(Zw[act], [S_list[i][0] for i in act]); BRW = br_probs(Zw)
            for step in range(maxk - 1):
                if len(act) == 0: break
                F = cand_matrix(Zw[act], R[act], [S_list[i] for i in act], BRW[act])
                pp = DEC.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K)
                for n_, i in enumerate(act):
                    for g in blocked(S_list[i]): pp[n_, g - 1] = -1
                j = pp.argmax(1); add = pp[np.arange(len(act)), j] >= TAU_C
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        return S_list

    evaluate('VioExplain', explain)

    # ---------------- VioExplain-T: temporal evidence in the first selection (final_fuse.py) ----------------
    def mix(ps, pt, a):
        l = (1 - a) * np.log(np.clip(ps, 1e-9, 1)) + a * np.log(np.clip(pt, 1e-9, 1)); l -= l.max(1, keepdims=True)
        e = np.exp(l); return e / e.sum(1, keepdims=True)
    DEEP = ROOT + 'results/hyd_v1/f0%s%s/' % ('' if EVT == 'comp' else '_' + EVT, TAG)
    cands = {nm: dict(np.load(DEEP + 'probs_%s.npz' % nm)) for nm in ('1D-CNN', 'ResNet') if os.path.exists(DEEP + 'probs_%s.npz' % nm)}
    if cands:
        ps_tc = PS(TC_Z); best = None; tab = []
        for nm, PT in cands.items():
            for a in (0.0, 0.25, 0.5, 0.75, 0.9):
                pm = mix(ps_tc, PT['tcal1'], a); f = f1_score(TC_Y, pm.argmax(1), average='macro')
                nll = float(-np.mean(np.log(np.clip(pm[np.arange(len(TC_Y)), TC_Y], 1e-12, 1))))
                tab.append((nm, a, round(float(f), 4), round(nll, 4)))
                # selection on test-calibration cycles: label macro F1 as in final_fuse.py, ties broken by the
                # log loss of the mixture (a proper scoring rule on the same calibration cycles)
                if best is None or (f, -nll) > best[0]: best = ((f, -nll), nm, a)
        log('fusion candidates (net, a, tcal macroF1, tcal NLL)', tab)
        _, TNET, A = best; PT = cands[TNET]
        s_ = np.sort(1 - mix(PS(Z['tcal'][0]), PT['tcal1'][TC_Y == 0], A)[:, 0]); n_ = len(s_)
        TAU0F = 1 - s_[min(n_ - 1, int(np.ceil(0.95 * (n_ + 1))) - 1)]
        log('fusion choice', TNET, 'a', A, 'tau0 fused', TAU0F)
        json.dump({'tau0': TAU0, 'tau_c': TAU_C, 'fusion_net': TNET, 'fusion_a': A, 'tau0_fused': float(TAU0F),
                   'fusion_candidates': tab}, open(OUT + 'thresholds_ours.json', 'w'))
        RES['OursT_label_macroF1'] = float(f1_score(T01_Y, mix(PS(ZT[T01]), PT['test'][T01], A).argmax(1), average='macro'))
        evaluate('VioExplain-T', lambda Zw: explain(Zw, P1=lambda Zq: mix(PS(Zq), PT[KEYOF[id(Zq)]], A), tau0=TAU0F))
    else:
        log('no deep probabilities found in', DEEP, '; VioExplain-T skipped')
log('DONE', NAMES)
