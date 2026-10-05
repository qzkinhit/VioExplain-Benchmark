"""VioExplain v3 knowledge-update (Update) learning curve, mode f0.

Usage:  NJ=1 python final_update.py H [SMOKE]     (H=0: reference, all 20 faults in the knowledge with paired runs)
Compute-reduced setting: the scorer and BR training sets use every SUB-th fit window per class,
scorer = LightGBM ITER trees at learning rate LR, BR detectors BR_ITER trees. The reference (H=0) uses the same setting.
For a held-out fault H:
  1. knowledge of the other 19 faults (single-fault data only, paired runs for the known faults as in final_run.py f0):
     BR detectors, deduction-consistent scorer, cooperative decision DEC, conformal tau0 and stop threshold tau_c.
  2. explain() plus the unknown score (Mahalanobis distance from the composed template mu_normal + sum of mean
     footprints of the selected events, pooled within-class covariance; threshold = 95 percent quantile on the
     test-calibration windows of known classes and normal) over the fit-partition training windows of H in run order;
     flagged windows enter the Update buffer.
  3. after the first n flagged windows (n in NS) a new event is created from the buffer only (no paired runs):
     support, footprint and ridge deduction operator against the normal mean; the scorer is refit with the buffer
     windows (label H), their self-deducted versions (label normal) and compositions X_buf + (X_B - X_0) with paired
     known-event effects (deduct B -> label H, deduct H -> label B); one BR detector is fit for H; DEC is kept;
     tau0 and tau_c are recalibrated as in final_run.py (tau_c on known-class test-calibration windows).
  4. event-set F1 on eval test windows of H alone and on simulator test pairs that contain H, plus a retention check on
     a fixed subsample of known windows; reference = f0 VioExplain records (H in the knowledge with paired runs).
Output: results/final_v1/update/h<H>.json (rewritten after every n).
"""
import os, sys, json, time, pickle
H = int(sys.argv[1]); SMOKE = len(sys.argv) > 2 and sys.argv[2] == 'SMOKE'
NJ_ = os.environ.get('NJ', '2'); os.environ['NJ'] = NJ_
for v_ in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v_] = NJ_
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['V3_MODE'] = 'f0'; os.environ['V3_LOG'] = 'log_update_h%d%s.txt' % (H, '_smoke' if SMOKE else '')
sys.path.insert(0, '/path/to/vioexplain/scripts')
from common import *  # noqa

UPD = '/path/to/vioexplain/results/final_v1/update/'
os.makedirs(UPD, exist_ok=True)
NOFLAG = os.environ.get('V3_NOFLAG') == '1'   # diagnostic: buffer = first n windows of H in run order, flag ignored
OPMASK = os.environ.get('V3_OPMASK') == '1'   # variant: operator target zero outside the buffer support
BRREFIT = os.environ.get('V3_BRREFIT') == '1'  # variant: known BR detectors refit with buffer windows and compositions
OUTJ = UPD + ('h%d%s%s%s%s.json' % (H, '_noflag' if NOFLAG else '', '_opmask' if OPMASK else '', '_brrefit' if BRREFIT else '',
                                     '_smoke' if SMOKE else '') if H else 'ref%s.json' % ('_smoke' if SMOKE else ''))
if os.environ.get('V3_TAG'): OUTJ = OUTJ[:-5] + '_' + os.environ['V3_TAG'] + '.json'
NS = [4, 8] if SMOKE else [4, 16, 64, 128, 8, 32]      # endpoints first so that partial runs cover the curve
if os.environ.get('V3_NS'): NS = [int(x) for x in os.environ['V3_NS'].split(',')]
SUB = int(os.environ.get('V3_SUB', '6')); ITER = int(os.environ.get('V3_ITER', '150')); LR = float(os.environ.get('V3_LR', '0.1'))
BR_ITER = int(os.environ.get('V3_BRITER', '150'))
NCOMP = int(round(2100 / SUB / 19))      # compositions per known event: about one known class worth of rows
REF_HS = [1, 2, 6, 7, 13, 8, 12, 14, 17, 18]
KNOWN = [c for c in range(1, C) if c != H]; KEFF = [e for e in EFF if e != H]
if H: TWIN_SUPP = Supp[H].copy(); TWIN_FOOT = FOOT[H].copy()


def mkl(n=ITER):
    return lgb.LGBMClassifier(n_estimators=8 if SMOKE else n, learning_rate=LR, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, n_jobs=NJ, verbose=-1)


def set_event(h, supp, foot, op):
    Supp[h] = supp; SUPPSET[h] = set(supp.tolist()); CHS[h] = np.isin(np.arange(M), supp); FOOT[h] = foot; OP[h] = op


if H: set_event(H, np.array([], int), np.zeros(M), None)   # H is absent from the knowledge
kmask = YB != H; XBk = XB[kmask]; YBk = YB[kmask]; XBkf = FLAT(XBk)        # full rows: unknown-score means and covariance
TRI = {c: ok_tr[c][::SUB] for c in range(C)}
XBs = np.concatenate([Z['tr'][c][TRI[c]] for c in range(C) if c != H or not H])
YBs = np.concatenate([np.full(len(TRI[c]), c) for c in range(C) if c != H or not H]); XBsf = FLAT(XBs)
RESJ = dict(H=H, mode='f0', NS=NS, ncomp_per_known=NCOMP, smoke=SMOKE, sub=SUB, iters=ITER, lr=LR, br_iters=BR_ITER,
            scorer_rows_per_class=int(len(TRI[0])))
log('H', H, 'rows per class', len(TRI[0]), 'scorer base rows', len(XBs), 'NCOMP', NCOMP)


def dump():
    json.dump(RESJ, open(OUTJ, 'w'), indent=1)


# ---------------- known BR detectors ----------------
brs = {}
for k in KNOWN: brs[k] = mkl(BR_ITER).fit(XBsf, (YBs == k).astype(int))
log('H', H, 'known BR detectors', len(brs))


def make_brp(dets):
    def brp(Zw):
        Xf = FLAT(Zw); out = np.zeros((len(Zw), K), np.float32)
        for k, m in dets.items(): out[:, k - 1] = m.predict_proba(Xf)[:, 1]
        return out
    return brp


# ---------------- deduction-consistent scorer ----------------
srng = np.random.default_rng(1)
SELF0 = []
for h in KNOWN:
    if OP[h] is not None:
        i2 = srng.choice(TRI[h], len(TRI[h]) // 3, replace=False); SELF0.append(deduct(Z['tr'][h][i2], h))
SELF0 = np.concatenate(SELF0)


def fit_scorer(extraX=(), extray=()):
    Xa = np.concatenate([XBs, SELF0] + list(extraX))
    ya = np.concatenate([YBs, np.zeros(len(SELF0), int)] + list(extray)).astype(int)
    sc = mkl().fit(FLAT(Xa), ya); cols = sc.classes_.astype(int)

    def PS(Zw):
        p = sc.predict_proba(FLAT(Zw)); out = np.zeros((len(Zw), C)); out[:, cols] = p; return out
    return PS


def cand_matrix(Zw, R, S_list, BRW, PS, brp, cands):
    n = len(Zw); pr = PS(R); brr = brp(R); rn = (np.clip(R, -50, 50) ** 2).sum(axis=(1, 2))
    vch = tviol_z(Zw).reshape(n, M, 6).any(2)
    expl = np.zeros((n, M), bool)
    for i, S in enumerate(S_list):
        for g in S: expl[i] |= CHS[g]
    un = vch & ~expl; nun = un.sum(1)
    rank = np.argsort(np.argsort(-pr[:, 1:], 1), 1)
    F = np.zeros((n, K, 8), np.float32); ln = np.array([len(S) for S in S_list])
    for e in cands:
        red = (rn - (np.clip(deduct(R, e), -50, 50) ** 2).sum(axis=(1, 2))) / (rn + 1.0)
        cov = np.where(nun > 0, (un & CHS[e][None]).sum(1) / np.maximum(nun, 1), 0.0)
        F[:, e - 1] = np.stack([pr[:, e], rank[:, e - 1], pr[:, 0], BRW[:, e - 1], brr[:, e - 1], cov, red, ln], 1)
    return F


def cmask(cands):
    cm = np.zeros(K, bool); cm[np.array(cands) - 1] = True; return cm


PS0 = fit_scorer(); BRP0 = make_brp(brs)
log('H', H, 'base scorer fitted')

# ---------------- cooperative decision (known events only) ----------------
crng = np.random.default_rng(5)


def cal_window(events, i):
    x = RAW['cal'][0][i].copy()
    for e in events: x = x + RAW['cal'][e][i] - RAW['cal'][0][i]
    return x


ok_cal = {c: ~FR['cal'][c] for c in range(C)}
rows = []; lab = []; batch = []
for it in range(3000):
    k = int(crng.choice([1, 2, 2, 3, 3, 4])); ev = [int(e) for e in crng.choice(KEFF, k, replace=False)]; i = int(crng.integers(ncal))
    if not all(ok_cal[e][i] for e in ev): continue
    batch.append((ev, i))
Xc = np.stack([cal_window(ev, i) for ev, i in batch]); Zc = Fn(Xc)
truth = []
for (ev, i), x in zip(batch, Xc):
    V = tviol_raw(x[None])[0]; V0 = tviol_raw(RAW['cal'][0][i][None])[0]; caused = V & ~V0; t = []
    for e in ev:
        Vr = tviol_raw(cal_window([g for g in ev if g != e], i)[None])[0]; Ve = tviol_raw(cal_window([e], i)[None])[0]
        if (caused & (~Vr | Ve)).any(): t.append(e)
    truth.append(set(t))
p = PS0(Zc); first = p[:, 1:].argmax(1) + 1
keep = np.array([f in t for f, t in zip(first, truth)])
Zc = Zc[keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]
S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = BRP0(Zc)
active = np.arange(len(Zc))
for step in range(4):
    if len(active) == 0: break
    F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active], PS0, BRP0, KNOWN)
    nxt_active = []
    for n_, i in enumerate(active):
        for e in KNOWN:
            if e in S_list[i]: continue
            rows.append(F[n_, e - 1]); lab.append(int(e in truth[i]))
        rem = [e for e in truth[i] if e not in S_list[i]]
        if rem:
            nxt = max(rem, key=lambda e: F[n_, e - 1, 0]); S_list[i].append(nxt); nxt_active.append(i)
    if nxt_active:
        idx = np.array(nxt_active); R[idx] = deduct_groups(R[idx], [S_list[i][-1] for i in idx])
    active = np.array(nxt_active, int)
DEC = lgb.LGBMClassifier(n_estimators=8 if SMOKE else 300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
log('H', H, 'cooperative decision rows', len(lab), 'positives', int(np.sum(lab)))

TC_NE = np.array([len(t) > 0 for t in TC_T])


def calib_tauc(PS, brp, cands):
    """Stop threshold as in final_run.py, on single-fault test-calibration windows of the known classes."""
    cm = cmask(cands)
    single_idx = np.where(np.isin(TC_Y, KNOWN) & TC_NE)[0]
    if SMOKE: single_idx = single_idx[::20]
    ps_ = PS(TC_Z[single_idx]); fp = ps_[:, 1:].argmax(1) + 1; good = single_idx[fp == TC_Y[single_idx]]
    Rg = deduct_groups(TC_Z[good], TC_Y[good])
    Fg = cand_matrix(TC_Z[good], Rg, [[int(y)] for y in TC_Y[good]], brp(TC_Z[good]), PS, brp, cands)
    pp = DEC.predict_proba(Fg.reshape(-1, 8))[:, 1].reshape(len(good), K)
    pp[:, ~cm] = -1; pp[np.arange(len(good)), TC_Y[good] - 1] = -1
    return float(np.quantile(pp.max(1), 0.95))


def make_explain(PS, brp, tau0, tauc, cands):
    cm = cmask(cands)

    def explain(Zw, maxk=5, chunk=20000):
        out = []
        for s0 in range(0, len(Zw), chunk):
            Zb = Zw[s0:s0 + chunk]; p = PS(Zb); n = len(Zb); p1 = p[:, 1:].copy(); p1[:, ~cm] = -1
            S_list = [[] if p[i, 0] >= tau0 else [int(np.argmax(p1[i])) + 1] for i in range(n)]
            act = np.array([i for i in range(n) if S_list[i]], int)
            if len(act):
                R = Zb.copy(); R[act] = deduct_groups(Zb[act], [S_list[i][0] for i in act]); BRW = brp(Zb)
                for step in range(maxk - 1):
                    if len(act) == 0: break
                    F = cand_matrix(Zb[act], R[act], [S_list[i] for i in act], BRW[act], PS, brp, cands)
                    pp = DEC.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K); pp[:, ~cm] = -1
                    for n_, i in enumerate(act):
                        for g in S_list[i]: pp[n_, g - 1] = -1
                    j = pp.argmax(1); add = pp[np.arange(len(act)), j] >= tauc
                    nxt = act[add]
                    for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                    if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                    act = nxt
            out += S_list
        return out
    return explain


TAU0_0 = conformal_empty(PS0); TAUC_0 = calib_tauc(PS0, BRP0, KNOWN)
EX0 = make_explain(PS0, BRP0, TAU0_0, TAUC_0, KNOWN)
RESJ['base'] = dict(tau0=float(TAU0_0), tau_c=TAUC_0)
log('H', H, 'base tau0', TAU0_0, 'tau_c', TAUC_0)

IK_ALL = np.arange(len(EV_Z))[::40]                      # known-window retention subsample (fixed across H)
PKS = 20                                                   # stride inside each pair group for the retention subsample


def f1m(P, T): return float(np.mean([setf1(a, b) for a, b in zip(P, T)]))


def named(P, T, h):
    m = [h in a for a, b in zip(P, T) if h in b]; return float(np.mean(m)) if m else float('nan')


def metrics_h(h, PA, TA, PP, TP, PK, TK, PPK, TPK):
    ne = [i for i, t in enumerate(TA) if t]
    return dict(alone_f1=f1m(PA, TA), alone_f1_nonempty=float(np.mean([setf1(PA[i], TA[i]) for i in ne])) if ne else float('nan'),
                alone_named=named(PA, TA, h), pair_f1=f1m(PP, TP), pair_named=named(PP, TP, h),
                known_single_f1=f1m(PK, TK), known_pair_f1=f1m(PPK, TPK))


if H == 0:
    # reference: every fault in the knowledge from the start (paired runs), same compute setting
    HS = REF_HS[:2] if SMOKE else REF_HS
    iA_all = np.where(np.isin(EV_Y, HS))[0]
    jfull = [j for j, g in enumerate(COMP) if set(g[0]) & set(HS)]
    parts = [EV_Z[iA_all], EV_Z[IK_ALL]] + [COMP[j][1] for j in jfull] + [COMP[j][1][::PKS] for j in range(len(COMP))]
    sizes = [len(p_) for p_ in parts]; allP = EX0(np.concatenate(parts)); cut = np.cumsum([0] + sizes)
    seg = [allP[cut[i]:cut[i + 1]] for i in range(len(parts))]
    PA_all, PK_all = seg[0], seg[1]; Pfull = dict(zip(jfull, seg[2:2 + len(jfull)])); Psub = seg[2 + len(jfull):]
    out = {}
    for h in HS:
        a = [k for k, i in enumerate(iA_all) if EV_Y[i] == h]; jh = [j for j in jfull if h in COMP[j][0]]
        kk = [k for k, i in enumerate(IK_ALL) if EV_Y[i] != h]; jk = [j for j in range(len(COMP)) if h not in COMP[j][0]]
        out[str(h)] = metrics_h(h, [PA_all[k] for k in a], [EV_T[iA_all[k]] for k in a],
                                sum([list(Pfull[j]) for j in jh], []), sum([list(COMP[j][2]) for j in jh], []),
                                [PK_all[k] for k in kk], [EV_T[IK_ALL[k]] for k in kk],
                                sum([list(Psub[j]) for j in jk], []), sum([list(COMP[j][2][::PKS]) for j in jk], []))
        log('REF', h, json.dumps(out[str(h)]))
    RESJ['reference'] = out
    RESJ['all_pair_eff'] = f1m(sum([list(Psub[j]) for j in range(len(COMP))], []), sum([list(COMP[j][2][::PKS]) for j in range(len(COMP))], []))
    dump(); log('REF DONE'); sys.exit(0)

# ---------------- unknown score ----------------
labs = [0] + KNOWN
means = {c: XBkf[YBk == c].mean(0).astype(np.float64) for c in labs}
resid = XBkf - np.stack([means[c] for c in labs])[np.searchsorted(labs, YBk)]
cov = np.cov(resid.T); cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov)) + 1e-3 * np.eye(cov.shape[0]); icov = np.linalg.inv(cov)
del resid


def unk_score(Zw, S):
    X = FLAT(Zw).astype(np.float64)
    T = np.stack([means[0] + sum((means[k] - means[0] for k in s), np.zeros_like(means[0])) for s in S])
    D = X - T; return np.einsum('ij,jk,ik->i', D, icov, D)


calK = np.where(np.isin(TC_Y, labs))[0]
if SMOKE: calK = calK[::10]
THR = float(np.quantile(unk_score(TC_Z[calK], EX0(TC_Z[calK])), 0.95))
# flag pass over the fit-partition training windows of H in run order (FIT runs sorted, windows in time order)
order = ok_tr[H]; Ztr = Z['tr'][H][order]; S_tr = EX0(Ztr); s_tr = unk_score(Ztr, S_tr)
flag_pos = np.where(s_tr > THR)[0]; FLAG = order[flag_pos]
RESJ['base_named_on_train_H'] = {str(e): int(sum(e in s_ for s_ in S_tr)) for e in range(1, C) if sum(e in s_ for s_ in S_tr) > 0}
RESJ['base_empty_on_train_H'] = int(sum(len(s_) == 0 for s_ in S_tr))
if NOFLAG: flag_pos = np.arange(len(order)); FLAG = order
RESJ['unknown'] = dict(threshold=THR, n_train_windows=int(len(order)), n_flagged=int(len(FLAG)),
                       flag_rate=float(len(FLAG) / len(order)),
                       scanned_for_n={str(n): int(flag_pos[n - 1] + 1) for n in NS if n <= len(flag_pos)})
log('H', H, 'unknown threshold', THR, 'flagged', len(FLAG), 'of', len(order))
dump()

# ---------------- evaluation sets ----------------
iA = np.where(EV_Y == H)[0]; TA = [EV_T[i] for i in iA]
jH = [j for j, g in enumerate(COMP) if H in g[0]]
jK = [j for j, g in enumerate(COMP) if H not in g[0]]
iK = IK_ALL[EV_Y[IK_ALL] != H]
if SMOKE: iA = iA[::10]; TA = [EV_T[i] for i in iA]; jH = jH[:3]; jK = jK[::20]; iK = iK[::10]
PST = 1 if not SMOKE else 10
ZP = np.concatenate([COMP[j][1][::PST] for j in jH]); TP = sum([list(COMP[j][2][::PST]) for j in jH], [])
ZPK = np.concatenate([COMP[j][1][::PKS] for j in jK]); TPK = sum([list(COMP[j][2][::PKS]) for j in jK], [])
TK = [EV_T[i] for i in iK]


def metrics(PA, PP, PK, PPK):
    return metrics_h(H, PA, TA, PP, TP, PK, TK, PPK, TPK)


def run_eval(ex):
    parts = [EV_Z[iA], ZP, EV_Z[iK], ZPK]; cut = np.cumsum([0] + [len(p_) for p_ in parts]); P = ex(np.concatenate(parts))
    return metrics(*[P[cut[i]:cut[i + 1]] for i in range(4)])


# reference: f0 VioExplain records (H in the knowledge with paired runs)
ref = pickle.load(open(UPD + 'ref_records_f0_VioExplain.pkl', 'rb'))
assert len(ref['single']) == len(EV_Z) and len(ref['pair']) == len(COMP)
assert all(tuple(ref['pair'][j][0]) == tuple(COMP[j][0]) for j in range(len(COMP)))
ref_all_pair = float(np.mean([setf1(a, b) for j in range(len(COMP)) for a, b in zip(ref['pair'][j][1], COMP[j][2])]))
RESJ['reference_full_f0'] = metrics([ref['single'][i] for i in iA], sum([list(ref['pair'][j][1][::PST]) for j in jH], []),
                            [ref['single'][i] for i in iK], sum([list(ref['pair'][j][1][::PKS]) for j in jK], []))
RESJ['reference_full_f0']['all_pair_eff_check'] = ref_all_pair
RESJ['n_eval'] = dict(alone=int(len(iA)), pair=int(len(ZP)), pair_groups=len(jH), known_single=int(len(iK)), known_pair=int(len(ZPK)))
log('H', H, 'reference_full_f0', json.dumps(RESJ['reference_full_f0']))
RESJ['curve'] = {}
m0 = run_eval(EX0); m0.update(n=0, tau0=float(TAU0_0), tau_c=TAUC_0); RESJ['curve']['0'] = m0
log('H', H, 'n 0', json.dumps(m0)); dump()

# ---------------- Update from the buffer ----------------
Zbar0 = Z['tr'][0].mean(0)
for n in NS:
    if n > len(FLAG): log('H', H, 'only', len(FLAG), 'flagged windows, stop at n', n); break
    t1 = time.time()
    idx = FLAG[:n]; Zb = Z['tr'][H][idx]; Xb = RAW['tr'][H][idx]
    dev = np.abs(Zb - Zbar0)
    supp = np.where((dev > 3).any(2).mean(0) >= 0.5)[0]; foot = dev.mean(axis=(0, 2))
    Yop = (Zb - Zbar0).copy()
    if OPMASK: Yop[:, ~np.isin(np.arange(M), supp), :] = 0
    op = Ridge(alpha=10.0).fit(Zb[:, supp, :].reshape(n, -1), Yop.reshape(n, -1)) if len(supp) else None
    set_event(H, supp, foot, op)
    urng = np.random.default_rng(1000 + H)
    exX = [Zb, deduct(Zb, H)]; exy = [np.full(n, H), np.zeros(n, int)]; brpos = [Zb]
    for B in KNOWN:
        jj = urng.integers(n, size=NCOMP); ii = urng.choice(ok_tr[B], NCOMP, replace=False)
        Zs = Fn(Xb[jj] + RAW['tr'][B][ii] - RAW['tr'][0][ii])
        exX += [deduct(Zs, B), deduct(Zs, H)]; exy += [np.full(NCOMP, H), np.full(NCOMP, B)]; brpos.append(Zs)
    PS1 = fit_scorer(exX, exy)
    Zcomp = np.concatenate(brpos[1:]); brpos = FLAT(np.concatenate(brpos))
    bh = mkl(BR_ITER).fit(np.concatenate([XBsf, brpos]), np.r_[np.zeros(len(XBsf), int), np.ones(len(brpos), int)])
    dets = dict(brs); dets[H] = bh
    if BRREFIT:
        compB = np.repeat(KNOWN, NCOMP); Xbr = np.concatenate([XBsf, brpos])   # brpos = buffer windows then compositions
        for k in KNOWN:
            yk = np.r_[YBs == k, np.zeros(n, bool), compB == k].astype(int); dets[k] = mkl(BR_ITER).fit(Xbr, yk)
    BRP1 = make_brp(dets); cands1 = KNOWN + [H]
    tau0 = conformal_empty(PS1); tauc = calib_tauc(PS1, BRP1, cands1)
    m = run_eval(make_explain(PS1, BRP1, tau0, tauc, cands1))
    ts = set(TWIN_SUPP.tolist()); us = set(supp.tolist())
    m.update(n=n, tau0=float(tau0), tau_c=tauc, supp_size=int(len(supp)), twin_supp_size=int(len(TWIN_SUPP)),
             supp_jaccard_twin=float(len(ts & us) / max(len(ts | us), 1)),
             foot_cos_twin=float(foot @ TWIN_FOOT / (np.linalg.norm(foot) * np.linalg.norm(TWIN_FOOT) + 1e-12)),
             scanned_train_windows=int(flag_pos[n - 1] + 1), seconds=round(time.time() - t1))
    RESJ['curve'][str(n)] = m; dump()
    log('H', H, 'n', n, json.dumps(m))
log('H', H, 'DONE')
