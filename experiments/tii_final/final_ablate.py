"""VioExplain v3 ablations and coverage curve on the shared protocol of common.py.

Usage:  V3_MODE=f100t|f50t|f25t [V3_NOTWIN=1] V3_ABL=full|noDeductTrain|noCoop|noOp V3_LOG=... V3_METRICS=... python final_ablate.py
  full           the method of final_run.py (BR-LGBM is evaluated as well when V3_ABL=full)
  noDeductTrain  scorer trained on single faults only (no composed windows with deducted footprints, no self-deduction)
  noCoop         second and later events added by the residual posterior alone with a conformal threshold
  noOp           footprint deduction by resetting the support channels to the normal center (no learned operator)
  V3_NOTWIN=1    knowledge and composition without paired runs (reference normal window from another run)
Coverage modes fXt compose only X percent of the fault pairs (and triples whose pairs are all covered); the metrics
then include set F1 on test pairs whose combination was composed (pair_eff_seen) and was never seen (pair_eff_unseen).
"""
from common import *
ABL = os.environ.get('V3_ABL', 'full')
if ABL == 'noOp':
    def deduct(Zw, h):
        out = Zw.copy(); out[:, Supp[h], :] = 0; return out

    def deduct_groups(R, events):
        out = R.copy(); events = np.asarray(events)
        for h in np.unique(events):
            m = events == h; out[m] = deduct(R[m], int(h))
        return out
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
if ABL == 'full': evaluate('BR-LGBM', lambda Zw: [[k + 1 for k in np.where(p > TH_BR)[0]] for p in br_probs(Zw)])

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
if ABL == 'noDeductTrain': Xa, ya = XB, YB
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
if ABL == 'noCoop':
    pr_ = PS(Rg)[:, 1:].copy(); pr_[np.arange(len(good)), TC_Y[good] - 1] = -1; TAU_C = float(np.quantile(pr_.max(1), 0.95))
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
                pp = DEC.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K) if ABL != 'noCoop' else PS(R[act])[:, 1:].copy()
                for n_, i in enumerate(act):
                    for g in S_list[i]: pp[n_, g - 1] = -1
                j = pp.argmax(1); add = pp[np.arange(len(act)), j] >= TAU_C
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        out += S_list
    return out


evaluate('VioExplain' + ('' if ABL == 'full' else '-' + ABL) + ('-noTwin' if NOTWIN else ''), explain)
log('DONE')
