"""Baselines for the knowledge-update (Update) panel, mode f0, reduced setting of final_update.py.

Usage:  NJ=2 python final_update_base.py H[,H...] METHOD[,METHOD...] [SMOKE]
METHOD in LGBM (top-1), BR-LGBM, CC-LGBM, AEC, MinExplain, PCA-RBC (CPU) and ResNet, LSTM, InceptionTime, ML-CNN (GPU,
V3_DEV, default cuda:0). V3_TAG adds a suffix to the output file (variants such as V3_EP=240).

Protocol (same held-out faults, windows, metric and calibration rules as final_update.py):
  knowledge  every class except H: normal plus the 19 known faults, every SUB-th (6) non-frozen fit window per class
             (350 windows per class), the reduced setting of final_update.py; LightGBM models 150 trees, learning rate 0.1.
  update n   the first n non-frozen fit-partition training windows of H in run order (the order final_update.py scans),
             WITH the true label H and without any flagging. The n windows are replicated to the per-class row count of
             the known classes (350; V3_BAL=0 switches this off) so that H has the weight of a known class. Methods that
             use paired runs for the known faults (AEC, PCA-RBC) build the knowledge of H from the n windows without
             paired runs, as VioExplain does (PCA-RBC: directions of the windows minus the normal mean; AEC: violation
             rates of the windows against randomly drawn normal fit windows). Models of the normal class alone (PCA
             monitoring model of PCA-RBC, normal violation rates of AEC) use all normal fit windows, as the unknown score
             of final_update.py does. Networks: training loop of gpu_save2.py, V3_EP epochs over the subsampled rows.
  full       H in the knowledge from the start like every other class (its 350 subsampled windows, paired runs where a
             method uses them), the counterpart of ref.json.
  calibration the rules of the main table. Empty set: split-conformal on fault-free test-calibration windows (one window
             per run, alpha 0.05; top-1 normal probability or largest event probability). Parameters that decide how many
             events a multi-label or covering method names: maximize the mean of event-set F1 on single-fault test-
             calibration windows and on composed calibration windows (cal_compositions of the f0 setting, seed 5, 2700
             windows). Under update the test-calibration windows of H and compositions containing H are not available
             (as for the tau_c and DEC of VioExplain in final_update.py), so the curve uses the known-class test-calibration
             windows and compositions of the known effective faults; full uses all of them.
  evaluation alone: eval test windows of H (official test set, onset 160, frozen removed, effective-event truth).
             with_known: simulator test_pairs windows of every pair that contains H. Event-set F1 per window, averaged.
Output: results/final_v1/update_base/h<H>_<METHOD>[_nobal].json, rewritten after every n.
"""
import os, sys, json, time, hashlib
HS = [int(x) for x in sys.argv[1].split(',')]; METHODS = sys.argv[2].split(','); SMOKE = len(sys.argv) > 3 and sys.argv[3] == 'SMOKE'
NJ_ = os.environ.get('NJ', '2'); os.environ['NJ'] = NJ_
for v_ in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v_] = NJ_
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['V3_MODE'] = 'f0'; os.environ['V3_FUNC'] = '0'
os.environ['V3_LOG'] = 'log_updbase_%s_%s%s.txt' % (sys.argv[1].replace(',', '-'), sys.argv[2].replace(',', '-'), '_smoke' if SMOKE else '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from sklearn.multioutput import ClassifierChain

ROOT = '/path/to/vioexplain/' if os.path.exists('/path/to/vioexplain') else '/path/to/vioexplain/'
UB = ROOT + 'results/final_v1/update_base/'; os.makedirs(UB, exist_ok=True)
SUB = 6; ITER = 150; LR = 0.1
BAL = os.environ.get('V3_BAL', '1') == '1'
CFGS = [4, 128, 'full', 16, 64, 8, 32]           # endpoints first so that partial runs cover the curve
if SMOKE: CFGS = [4, 'full']
if os.environ.get('V3_CFGS'): CFGS = [c_ if c_ == 'full' else int(c_) for c_ in os.environ['V3_CFGS'].split(',')]
TRI = {c: ok_tr[c][::SUB] for c in range(C)}; NPC = len(TRI[0])
THRG = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]

# ---------- helpers of the shared protocol (identical to common.py on hit; defined here when the local common.py lacks them)
if 'tviol_all' not in globals(): tviol_all = tviol_z
if 'TC_RUN' not in globals():
    TC_RUN = np.concatenate([1000 * c + np.repeat(np.arange(len(TCAL)), len(Z['tcal'][c]) // len(TCAL)) for c in range(C)])[TC_KEEP]


def conformal_upper_(scores, groups, alpha=0.05, seed=17):
    r_ = np.random.default_rng(seed); groups = np.asarray(groups); pick = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; pick.append(idx[r_.integers(len(idx))])
    v = np.sort(np.asarray(scores)[pick]); n = len(v)
    return float(v[min(n - 1, int(np.ceil((1 - alpha) * (n + 1))) - 1)])


NM_TC = np.where(TC_Y == 0)[0]


def ml_gate_(p_tc_normal):
    """largest event probability on the fault-free test-calibration windows (rows NM_TC), one window per run"""
    return conformal_upper_(np.asarray(p_tc_normal).max(1), TC_RUN[NM_TC])


def ml_sets_(P, ta, q0):
    return [[] if p.max() <= q0 else [k + 1 for k in np.where(p > ta)[0]] for p in P]


def tune_thr(p_single, t_single, p_comp, t_comp, grid, tag, q0):
    def score(ta):
        f1 = float(np.mean([setf1(a, t) for a, t in zip(ml_sets_(p_single, ta, q0), t_single)]))
        f2 = float(np.mean([setf1(a, t) for a, t in zip(ml_sets_(p_comp, ta, q0), t_comp)]))
        return 0.5 * f1 + 0.5 * f2
    grid = sorted(grid); vals = {ta: score(ta) for ta in grid}
    for _ in range(12):
        best = max(grid, key=lambda t: (vals[t], -t))
        if best == grid[0] and grid[0] > 1e-6: nt = grid[0] / 2
        elif best == grid[-1] and grid[-1] < 1 - 1e-6: nt = 1 - (1 - grid[-1]) / 2
        else: break
        grid = sorted(grid + [nt]); vals[nt] = score(nt)
    best = max(grid, key=lambda t: (vals[t], -t))
    log('%s threshold' % tag, best, round(vals[best], 4), 'interior', grid[0] < best < grid[-1])
    return best, vals[best]


def comps(effs, n=2700, seed=5):
    """cal_compositions of common.py (all_pairs=True, the f0 setting) restricted to the event list effs."""
    crng_ = np.random.default_rng(seed); ok_cal_ = {c_: ~FR['cal'][c_] for c_ in range(C)}; batch = []; tries = 0

    def cw(events, i):
        x = RAW['cal'][0][i].copy()
        for e in events: x = x + RAW['cal'][e][i] - RAW['cal'][0][i]
        return x
    while len(batch) < n and tries < 200000:
        tries += 1
        k = int(crng_.choice([1, 2, 2, 3, 3])); ev = [int(e) for e in crng_.choice(effs, k, replace=False)]; i = int(crng_.integers(ncal))
        if not all(ok_cal_[e][i] for e in ev): continue
        batch.append((ev, i))
    Xc = np.stack([cw(ev, i) for ev, i in batch]); truth = []
    for (ev, i), x in zip(batch, Xc):
        V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
        for e in ev:
            Vr = tviol_raw(cw([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cw([e], i)[None])[0]
            if (caused & (~Vr | Ve)).any(): t.append(e)
        truth.append(t)
    return Xc, Fn(Xc), truth


def mkl():
    return lgb.LGBMClassifier(n_estimators=8 if SMOKE else ITER, learning_rate=LR, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, n_jobs=NJ, verbose=-1)


def f1m(P, T): return float(np.mean([setf1(a, b) for a, b in zip(P, T)]))


def named(P, T, h):
    m = [h in a for a, b in zip(P, T) if h in b]; return float(np.mean(m)) if m else float('nan')


def hsh(T): return hashlib.sha1(json.dumps([sorted(int(e) for e in t) for t in T]).encode()).hexdigest()[:12]


CAL_ALL = comps(EFF)
log('calibration compositions (all effective faults)', len(CAL_ALL[2]))
SLNETS = ('ResNet', 'LSTM', 'InceptionTime')
GPU = any(m_ in SLNETS + ('ML-CNN',) for m_ in METHODS)
if GPU:
    import torch, torch.nn as nn
    DEV = os.environ.get('V3_DEV', 'cuda:0'); EP = int(os.environ.get('V3_EP', '40'))
    nm_ = RAW['tr'][0].reshape(-1, M); PM = nm_.mean(0); PSD = nm_.std(0) + 1e-8

    def to_t(X): return torch.as_tensor(((X - PM) / PSD).astype(np.float32).transpose(0, 2, 1))

    class CNN(nn.Module):
        def __init__(s, out=20):
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

    class LSTM(nn.Module):
        def __init__(s):
            super().__init__(); s.r = nn.LSTM(M, 128, num_layers=2, batch_first=True, dropout=0.1); s.h = nn.Linear(128, C)

        def forward(s, x): o, _ = s.r(x.transpose(1, 2)); return s.h(o[:, -1])

    class Inception(nn.Module):
        def __init__(s, i, nf=32):
            super().__init__(); s.b = nn.Conv1d(i, 32, 1, bias=False) if i > 1 else nn.Identity(); bi = 32 if i > 1 else i
            s.k = nn.ModuleList([nn.Conv1d(bi, nf, k, padding=k // 2, bias=False) for k in (9, 19, 39)])
            s.p = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(i, nf, 1, bias=False)); s.n = nn.BatchNorm1d(4 * nf)

        def forward(s, x):
            z = s.b(x); return torch.relu(s.n(torch.cat([c(z)[..., :x.shape[-1]] for c in s.k] + [s.p(x)], 1)))

    class InceptionTime(nn.Module):
        def __init__(s):
            super().__init__(); s.m = nn.ModuleList([Inception(M if d == 0 else 128) for d in range(6)])
            s.r = nn.ModuleList([nn.Sequential(nn.Conv1d(M, 128, 1), nn.BatchNorm1d(128)), nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128))])
            s.h = nn.Linear(128, C)

        def forward(s, x):
            res = x
            for d, m in enumerate(s.m):
                x = m(x)
                if d % 3 == 2: x = torch.relu(x + s.r[d // 3](res)); res = x
            return s.h(x.mean(-1))

    def train_net(net, X, Y, multilabel):
        """training loop of gpu_run2.py / gpu_save2.py (AdamW 1e-3, cosine schedule, batch 512, seed 0)"""
        torch.manual_seed(0); net = net.to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        ep_ = 1 if SMOKE else EP; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep_)
        Xt = to_t(X).to(DEV); Yt = torch.as_tensor(Y).to(DEV)
        for ep in range(ep_):
            net.train(); perm = torch.randperm(len(Xt), device=DEV)
            for s in range(0, len(Xt), 512):
                b = perm[s:s + 512]; o = net(Xt[b])
                loss = nn.functional.binary_cross_entropy_with_logits(o, Yt[b]) if multilabel else nn.functional.cross_entropy(o, Yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        net.eval()

        def P(Xr):
            out = []
            with torch.no_grad():
                for s in range(0, len(Xr), 4096):
                    o = net(to_t(Xr[s:s + 4096]).to(DEV)); out.append((torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).cpu().numpy())
            return np.concatenate(out)
        return P

for H in HS:
    KNOWN = [c for c in range(1, C) if c != H]; KEFF = [e for e in EFF if e != H]
    CAL_K = comps(KEFF); TCK = np.where(TC_Y != H)[0]; TCA = np.arange(len(TC_Y))
    iA = np.where(EV_Y == H)[0]; TA = [EV_T[i] for i in iA]
    jH = [j for j, g in enumerate(COMP) if H in g[0]]
    if SMOKE: iA = iA[::10]; TA = [EV_T[i] for i in iA]; jH = jH[:3]
    ZA = EV_Z[iA]; ZP = np.concatenate([COMP[j][1] for j in jH]); TP = sum([list(COMP[j][2]) for j in jH], [])
    RA = EV_RAW[iA]; RP = np.concatenate([GRAW['pair_%d+%d' % tuple(COMP[j][0])] for j in jH])
    assert len(RP) == len(ZP)
    NEVAL = dict(alone=int(len(iA)), pair=int(len(ZP)), pair_groups=len(jH), tcal_known=int(len(TCK)), cal_comp_known=len(CAL_K[2]),
                 hash_alone_truth=hsh(TA), hash_pair_truth=hsh(TP))
    log('H', H, 'eval', NEVAL)
    Xk = [Z['tr'][c][TRI[c]] for c in range(C) if c != H]; Rk = [RAW['tr'][c][TRI[c]] for c in range(C) if c != H]
    Yk = [np.full(len(TRI[c]), c) for c in range(C) if c != H]

    for meth in METHODS:
        OUTJ = UB + 'h%d_%s%s%s%s.json' % (H, meth, '' if BAL else '_nobal', ('_' + os.environ['V3_TAG']) if os.environ.get('V3_TAG') else '',
                                         '_smoke' if SMOKE else '')
        RJ = json.load(open(OUTJ)) if os.path.exists(OUTJ) and not SMOKE else {}
        RJ.update(H=H, method=meth, mode='f0', sub=SUB, rows_per_class=NPC, lgbm_trees=ITER, lgbm_lr=LR, balanced=BAL,
                  n_eval=NEVAL, host='hit' if 'user' in ROOT else 'cpu1')
        RJ.setdefault('curve', {})
        for cfg in CFGS:
            if str(cfg) in RJ['curve'] and not SMOKE: continue
            t1 = time.time()
            full = cfg == 'full'
            idxH = TRI[H] if full else ok_tr[H][:cfg]
            idxR = idxH if (full or not BAL) else np.resize(idxH, NPC)
            Ztr = np.concatenate(Xk + [Z['tr'][H][idxR]]); Ytr = np.concatenate(Yk + [np.full(len(idxR), H)])
            tci = TCA if full else TCK; CAL = CAL_ALL if full else CAL_K
            TCT = [TC_T[i] for i in tci]
            info = {}
            if meth in ('BR-LGBM', 'CC-LGBM', 'ML-CNN'):
                Yml = np.zeros((len(Ytr), K), int); Yml[np.arange(len(Ytr))[Ytr > 0], Ytr[Ytr > 0] - 1] = 1
            if meth == 'LGBM':
                m = mkl().fit(FLAT(Ztr), Ytr); cols = m.classes_.astype(int)

                def P(Zw, m=m, cols=cols):
                    p = m.predict_proba(FLAT(Zw)); out = np.zeros((len(Zw), C)); out[:, cols] = p; return out
                q0 = conformal_empty(P); info['q0'] = float(q0)
                fn = lambda Zw, P=P, q0=q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Zw)]
                PA, PP = fn(ZA), fn(ZP)
            elif meth in ('BR-LGBM', 'CC-LGBM'):
                if meth == 'BR-LGBM':
                    brs = [mkl().fit(FLAT(Ztr), Yml[:, k]) for k in range(K)]
                    P = lambda Zw, brs=brs: np.stack([b_.predict_proba(FLAT(Zw))[:, 1] for b_ in brs], 1)
                else:
                    cc = ClassifierChain(mkl(), order='random', random_state=0).fit(FLAT(Ztr), Yml)
                    P = lambda Zw, cc=cc: cc.predict_proba(FLAT(Zw))
                ptc = P(TC_Z); q0 = ml_gate_(ptc[NM_TC])
                ta, sc = tune_thr(ptc[tci], TCT, P(CAL[1]), CAL[2], THRG, '%s H%d n%s' % (meth, H, cfg), q0)
                info.update(q0=float(q0), threshold=float(ta), cal_score=float(sc))
                PA, PP = ml_sets_(P(ZA), ta, q0), ml_sets_(P(ZP), ta, q0)
            elif meth in SLNETS + ('ML-CNN',):
                Rtr = np.concatenate(Rk + [RAW['tr'][H][idxR]])
                if meth in SLNETS:
                    P = train_net({'ResNet': ResNet, 'LSTM': LSTM, 'InceptionTime': InceptionTime}[meth](), Rtr, Ytr.astype(np.int64), False)
                    n0 = len(Z['tcal'][0]); per = n0 // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
                    s = np.sort(1 - P(RAW['tcal'][0][pick])[:, 0]); q0 = 1 - s[min(len(s) - 1, int(np.ceil(0.95 * (len(s) + 1))) - 1)]
                    info['q0'] = float(q0)
                    fn = lambda Xr, P=P, q0=q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P(Xr)]
                    PA, PP = fn(RA), fn(RP)
                else:
                    P = train_net(CNN(20), Rtr, Yml.astype(np.float32), True)
                    ptc = P(TC_RAW); q0 = ml_gate_(ptc[NM_TC])
                    ta, sc = tune_thr(ptc[tci], TCT, P(CAL[0]), CAL[2], THRG, '%s H%d n%s' % (meth, H, cfg), q0)
                    info.update(q0=float(q0), threshold=float(ta), cal_score=float(sc))
                    PA, PP = ml_sets_(P(RA), ta, q0), ml_sets_(P(RP), ta, q0)
            elif meth in ('AEC', 'MinExplain'):
                V0 = tviol_all(Z['tr'][0]); PR = np.zeros((C, V0.shape[1])); P0 = np.clip(V0.mean(0), 1e-4, None)
                for h in KNOWN:
                    g = TRI[h]; PR[h] = (tviol_all(Z['tr'][h][g]) & ~V0[g]).mean(0)
                if full: PR[H] = (tviol_all(Z['tr'][H][idxH]) & ~V0[idxH]).mean(0)
                else:
                    jn = np.random.default_rng(1000 + H).choice(ok_tr[0], len(idxH), replace=False)
                    PR[H] = (tviol_all(Z['tr'][H][idxH]) & ~V0[jn]).mean(0)

                def make_aec(g):
                    th_e, th_p, lam, tau = g
                    EX = PR[1:] >= th_e; PO = PR[1:] >= th_p; Wt = np.where(PO, PR[1:], 0.0)

                    def aec(Zw, maxk=5):
                        V = tviol_all(Zw); out = []
                        miss = (EX[None] & ~V[:, None, :]).sum(2)
                        for i in range(len(V)):
                            unc = V[i].copy(); S = []
                            for _ in range(maxk):
                                sc_ = Wt[:, unc].sum(1) - lam * miss[i]
                                for e in S: sc_[e - 1] = -np.inf
                                j = int(np.argmax(sc_))
                                if sc_[j] < tau: break
                                S.append(j + 1); unc &= ~PO[j]
                                if not unc.any(): break
                            out.append(S)
                        return out
                    return aec
                def make_minexp(g):
                    w, mu, th_e = g
                    cost = -np.log(np.clip(PR[1:], 1e-4, 1.0)); c0 = -np.log(P0); EXm = PR[1:] >= th_e

                    def minexp(Zw, maxk=5):
                        V = tviol_all(Zw); out = []
                        miss = (EXm[None] & ~V[:, None, :]).sum(2)
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
                if meth == 'AEC':
                    grid = [(e, p, l, t) for e in (0.7, 0.9, 0.97) for p in (0.005, 0.01, 0.02, 0.05, 0.2) for l in (0.5, 2.0, 5.0, 10.0, 20.0)
                            for t in (-5.0, -2.0, -0.5, 0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.5)]; make_ = make_aec
                else:
                    grid = [(w, mu, e) for w in (0.25, 1, 4, 16, 64, 256) for mu in (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0)
                            for e in (0.7, 0.9, 0.97)]; make_ = make_minexp
                if SMOKE: grid = grid[::50]
                ts_ = tci[::3]; Zs = TC_Z[ts_]; Ts = [TC_T[i] for i in ts_]; Zc = CAL[1][::2]; Tc = CAL[2][::2]
                best = None
                for g in grid:
                    fa = make_(g); f = 0.5 * f1m(fa(Zs), Ts) + 0.5 * f1m(fa(Zc), Tc)
                    if best is None or f > best[1]: best = (g, f)
                log(meth, 'H', H, 'n', cfg, 'tuned', best)
                info.update(params=list(best[0]), cal_score=float(best[1]))
                fa = make_(best[0]); PA, PP = fa(ZA), fa(ZP)
            elif meth == 'PCA-RBC':
                XN = FLAT(Z['tr'][0][ok_tr[0]]).astype(np.float64); mu_n = XN.mean(0); Xc_ = XN - mu_n
                S_ = np.cov(Xc_.T); ev_, V_ = np.linalg.eigh(S_); o = np.argsort(ev_)[::-1]; ev_ = ev_[o]; V_ = V_[:, o]
                l = int(np.searchsorted(np.cumsum(ev_) / ev_.sum(), 0.90)) + 1; Pm = V_[:, :l]; lam_ = ev_[:l]
                T2m = Pm @ np.diag(1 / lam_) @ Pm.T; Q = np.eye(len(mu_n)) - Pm @ Pm.T
                t2 = ((Xc_ @ T2m) * Xc_).sum(1); spe = ((Xc_ @ Q) * Xc_).sum(1); PHI = T2m / t2.mean() + Q / spe.mean()
                XI = {}
                for h in range(1, C):
                    g = TRI[h]
                    if h != H or full: D = (FLAT(Z['tr'][h][g]) - FLAT(Z['tr'][0][g])).astype(np.float64)
                    else: D = (FLAT(Z['tr'][H][idxH]) - FLAT(Z['tr'][0][ok_tr[0]]).mean(0)).astype(np.float64)
                    U, s, _ = np.linalg.svd(D.T, full_matrices=False); r = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.8)) + 1
                    XI[h] = U[:, :min(r, 3)]
                phi = lambda X: ((X @ PHI) * X).sum(1)
                zs = Z['tcal'][0]; per = len(zs) // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
                s0 = np.sort(phi(FLAT(zs[pick]) - mu_n)); LIM = s0[min(len(s0) - 1, int(np.ceil(0.95 * (len(s0) + 1))) - 1)]

                def rbc_path(Zw, maxk=5):
                    """greedy reconstruction path of final_base.py PCA-RBC, run to maxk steps for every window above the
                    control limit; the set for a stop factor kappa is the path cut at the first step whose reconstructed
                    index is at most kappa * LIM (identical to running make_rbc(kappa))."""
                    X = FLAT(Zw).astype(np.float64) - mu_n; ph = phi(X); S = [[] for _ in range(len(X))]; B = [[] for _ in range(len(X))]
                    act = np.where(ph > LIM)[0]; XP = X @ PHI
                    for step in range(maxk):
                        keys = {}
                        for i in act: keys.setdefault(tuple(S[i]), []).append(i)
                        for Hs, rows in keys.items():
                            rows = np.array(rows); best = np.full(len(rows), np.inf); arg = np.zeros(len(rows), int)
                            for h in range(1, C):
                                if h in Hs: continue
                                Xi = np.concatenate([XI[g] for g in list(Hs) + [h]], 1); A = Xi.T @ PHI @ Xi
                                b = Xi.T @ XP[rows].T; f = np.linalg.lstsq(A, b, rcond=None)[0]
                                Y = X[rows] - (Xi @ f).T; v = ((Y @ PHI) * Y).sum(1)
                                bb = v < best; best[bb] = v[bb]; arg[bb] = h
                            for n_, i in enumerate(rows): S[i].append(int(arg[n_])); B[i].append(float(best[n_]))
                    return S, B

                def cut(path, kappa):
                    S, B = path; out = []
                    for s_, b_ in zip(S, B):
                        t = len(s_)
                        for k_, v in enumerate(b_):
                            if v <= kappa * LIM: t = k_ + 1; break
                        out.append(s_[:t])
                    return out
                ts_ = tci[::3]; Ts = [TC_T[i] for i in ts_]; Tc = CAL[2][::2]
                pth_s = rbc_path(TC_Z[ts_]); pth_c = rbc_path(CAL[1][::2]); best = None
                for kappa in [0.25, 0.5, 1, 2, 4, 8, 16, 64, 256, 1024, 1e12]:
                    f = 0.5 * f1m(cut(pth_s, kappa), Ts) + 0.5 * f1m(cut(pth_c, kappa), Tc)
                    if best is None or f > best[1]: best = (kappa, f)
                log('PCA-RBC H', H, 'n', cfg, 'components', l, 'rank H', XI[H].shape[1], 'tuned', best)
                info.update(kappa=float(best[0]), cal_score=float(best[1]), components=int(l), rank_H=int(XI[H].shape[1]))
                PA, PP = cut(rbc_path(ZA), best[0]), cut(rbc_path(ZP), best[0])
            else:
                raise ValueError(meth)
            a = f1m(PA, TA); w = f1m(PP, TP)
            e = dict(alone=a, with_known=w, mean=0.5 * (a + w), alone_named=named(PA, TA, H), pair_named=named(PP, TP, H),
                     alone_mean_set_size=float(np.mean([len(s_) for s_ in PA])), n_H_windows=int(len(idxH)), n_H_rows=int(len(idxR)),
                     seconds=round(time.time() - t1), **info)
            RJ['curve'][str(cfg)] = e; json.dump(RJ, open(OUTJ, 'w'), indent=1)
            log(meth, 'H', H, 'n', cfg, json.dumps(e))
log('DONE', HS, METHODS)
