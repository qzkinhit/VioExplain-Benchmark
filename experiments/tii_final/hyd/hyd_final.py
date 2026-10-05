"""VioExplain on the hydraulic benchmark with the final method configuration (final_fuse5.py, run label B of the TEP
runs: learned addition decision without the number of selected events, split-conformal stop threshold, temporal
evidence with the weight chosen on calibration). Protocol f100t only: knowledge from normal and single-fault cycles
and compositions built from them; no real multi-fault cycle enters any fitted or calibrated quantity.

Usage:  HYD_MODE=f100t HYD_OUTSUB=final/ [HYD_FUNC=1] [HYD_LADDER=1] python hyd_final.py
Needs results/hyd_v1/final/f0<TAG>/tnet_<1D-CNN|ResNet>.pt from hyd_tnet.py.

Differences to the 'ours' block of hyd_run.py; the scorer, the detectors, the deduction
operators and their compositions (all pairs and all triples, hyd_common.py) are unchanged:
  1. the addition decision does not see the number of selected events (the step feature is zero);
  2. the compositions that fit the addition decision have one, two or three events (the 'ours' block also draws four);
     HYD_NCOMP of them are accepted (2700, as in final_fuse5.py);
  3. tau_1 is split-conformal: on the single-fault test-calibration cycles whose first selection is correct, the score
     is the largest addition score of a false event after deducting the true event; the threshold is the
     ceil((1 - 0.05)(n + 1))-th smallest score and an event is added only if its score is strictly above it. A cycle
     is one window and the data have no run structure, so every such cycle enters once (on TEP: one window per run);
  4. temporal evidence: the first selection uses p ~ PS^(1-a) PT^a. The pair (a, net), a in {0, .25, .5, .75, .9},
     net in {1D-CNN, ResNet}, maximizes the label macro F1 on the normal and single-fault test-calibration cycles plus
     the first-selection accuracy on the composed calibration cycles; candidates are visited by increasing a and a
     later one must be strictly better, so a tie keeps the smaller weight and a = 0 means no temporal evidence. The
     first selection of the addition-decision compositions and of the tau_1 cycles uses the same mixture;
  5. HYD_FUNC=1 appends the functional-constraint units of hyd_common.py.
With HYD_LADDER=1 the script also evaluates the intermediate configurations L0 (the 'ours' block of hyd_run.py), L1 (= L0 + 1),
L2 (= L1 + 2), L3 (= L2 + 3); the final method is L3 + 4. Every configuration is saved as metrics_<name>.json.
A calibration-only criterion for the choice between the variants with and without functional units is saved to
calib_<name>.json: the set F1 of the final method on the normal and single-fault test-calibration cycles and on
compositions (two or three events) of test-calibration cycles, which no fitted model has seen.
"""
import sys
from hyd_common import *
import torch
from hyd_tnet import build, tnet_probs
assert MODE == 'f100t', 'hyd_final.py implements the protocol with compositions (f100t)'
torch.set_num_threads(NJ)
FTAG = 'func' if FUNC else 'nofunc'
LADDER = os.environ.get('HYD_LADDER', '1') == '1'
NCOMP = int(os.environ.get('HYD_NCOMP', '2700')); ALPHA = 0.05
TDIR = ROOT + 'results/hyd_v1/final/f0%s%s/' % ('' if EVT == 'comp' else '_' + EVT, TAG)
TNETS = {}
for nm_ in ('1D-CNN', 'ResNet'):
    ck_ = torch.load(TDIR + 'tnet_%s.pt' % nm_, map_location='cpu', weights_only=False)
    TNETS[nm_] = build(nm_); TNETS[nm_].load_state_dict(ck_['state']); TNETS[nm_].eval()


def pt_raw(nm, X): return np.concatenate([tnet_probs(TNETS[nm], X[s:s + 2048]) for s in range(0, len(X), 2048)])


def mix(ps, pt, a):
    l = (1 - a) * np.log(np.clip(ps, 1e-9, 1)) + a * np.log(np.clip(pt, 1e-9, 1)); l -= l.max(1, keepdims=True)
    e = np.exp(l); return e / e.sum(1, keepdims=True)


# ---------------- scorer (deduction-consistent), identical to hyd_run.py ----------------
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
log('units', MT, 'relations', NR, 'scorer rows', len(ya), 'label macro F1 (normal+single test cycles)',
    float(f1_score(T01_Y, PS(ZT[T01]).argmax(1), average='macro')))


def cand_matrix(Zw, R, S_list, BRW, step):
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
                                np.array([len(S) for S in S_list]) if step else np.zeros(n)], 1)
    return F


def blocked(S):
    """events that cannot be added: already in S or of a component already in S (severity events only)."""
    cs = set(int(COMPOF[g]) for g in S); return [e for e in range(1, C) if e in S or int(COMPOF[e]) in cs]


def compose(role, k_choices, n_target, by_iter, seed):
    """No-twin compositions of cycles of one role (compose_raw of hyd_common.py); truth = composed set."""
    batch, Xc = compose_raw(role, k_choices, n_target, by_iter, seed)
    Zc = Fn(Xc)
    return dict(X=Xc, Z=Zc, truth=[set(ev) for ev, _, _ in batch], ps=PS(Zc), pt={nm: pt_raw(nm, Xc) for nm in TNETS},
                sizes=collections.Counter(len(ev) for ev, _, _ in batch))


CP = {True: compose('cal', [1, 2, 2, 3, 3, 4], 3000, True, 5 + SEED) if LADDER else None,     # 'ours' block of hyd_run.py
      False: compose('cal', [1, 2, 2, 3, 3], NCOMP, False, 5 + SEED)}                          # final
log('addition-decision compositions by size: final', dict(CP[False]['sizes']), 'L0', dict(CP[True]['sizes']) if LADDER else None)
PS_TC = PS(TC_Z); PT_TC = {nm: pt_raw(nm, GRAW['tcal1']) for nm in TNETS}; PT_TEST = {nm: pt_raw(nm, GRAW['test']) for nm in TNETS}
BR_TC = br_probs(TC_Z); N0 = TC_Y == 0


def fit_decision(cp, first, step):
    truth = cp['truth']; keep = np.array([f in t for f, t in zip(first, truth)])
    Zc = cp['Z'][keep]; truth = [t for t, k_ in zip(truth, keep) if k_]; first = first[keep]
    S_list = [[int(f)] for f in first]; R = deduct_groups(Zc, first); BRW = br_probs(Zc)
    active = np.arange(len(Zc)); rows = []; lab = []
    for stp in range(4):
        if len(active) == 0: break
        F = cand_matrix(Zc[active], R[active], [S_list[i] for i in active], BRW[active], step)
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
    dec = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=NJ, verbose=-1).fit(np.array(rows), np.array(lab))
    return dec, dict(decision_rows=len(lab), decision_positives=int(np.sum(lab)), composed_first_correct=float(keep.mean()))


def run_config(name, step, four, conformal, temporal, final=False):
    cp = CP[four]; info = dict(name=name, step_feature=bool(step), four_event_compositions=bool(four),
                               tau1_rule='split-conformal, strict' if conformal else 'quantile 0.95, >=', units=MT, relations=NR)
    # ---- temporal evidence: weight and encoder chosen on calibration, ties keep the smaller weight ----
    net, A = '1D-CNN', 0.0
    if temporal:
        best = None; tab = []
        for a in (0.0, 0.25, 0.5, 0.75, 0.9):
            for nm in TNETS:
                f_single = float(f1_score(TC_Y, mix(PS_TC, PT_TC[nm], a).argmax(1), average='macro'))
                pk = mix(cp['ps'], cp['pt'][nm], a)[:, 1:].argmax(1) + 1
                f_comp = float(np.mean([int(f_) in t for f_, t in zip(pk, cp['truth'])]))
                tab.append((nm, a, round(f_single, 6), round(f_comp, 6)))
                if best is None or f_single + f_comp > best[0]: best = (f_single + f_comp, nm, a)
        _, net, A = best
        info.update(fusion_candidates=tab, fusion_criterion=float(best[0]))
        log(name, 'fusion candidates (net, a, tcal macro F1, composed first-selection accuracy)', tab)
    info.update(fusion_net=net if A > 0 else None, fusion_a=float(A))
    P1 = (lambda ps, pt: ps) if A == 0 else (lambda ps, pt: mix(ps, pt, A))
    # ---- addition decision ----
    first = P1(cp['ps'], cp['pt'][net])[:, 1:].argmax(1) + 1
    dec, st = fit_decision(cp, first, step); info.update(st)
    # ---- stop threshold on single-fault test-calibration cycles whose first selection is correct ----
    single_idx = np.where(TC_Y > 0)[0]
    fp = P1(PS_TC[single_idx], PT_TC[net][single_idx])[:, 1:].argmax(1) + 1; good = single_idx[fp == TC_Y[single_idx]]
    Rg = deduct_groups(TC_Z[good], TC_Y[good]); Fg = cand_matrix(TC_Z[good], Rg, [[int(y)] for y in TC_Y[good]], BR_TC[good], step)
    pp = dec.predict_proba(Fg.reshape(-1, 8))[:, 1].reshape(len(good), K)
    for n_, y in enumerate(TC_Y[good]):
        for e in blocked([int(y)]): pp[n_, e - 1] = -1
    v = np.sort(pp.max(1)); n = len(v)
    tau1 = float(v[min(n - 1, int(np.ceil((1 - ALPHA) * (n + 1))) - 1)]) if conformal else float(np.quantile(v, 0.95))
    # ---- empty-set threshold on the normal test-calibration cycles ----
    s0 = np.sort(1 - P1(PS_TC[N0], PT_TC[net][N0])[:, 0]); n0 = len(s0)
    tau0 = float(1 - s0[min(n0 - 1, int(np.ceil((1 - ALPHA) * (n0 + 1))) - 1)])
    info.update(tau0=tau0, tau1=tau1, tau1_n=int(n), tcal_single=int(len(single_idx)), tau1_scores_top=[float(x) for x in v[-8:]])
    log(name, json.dumps({k: info[k] for k in ('fusion_net', 'fusion_a', 'decision_rows', 'decision_positives', 'composed_first_correct',
                                               'tau0', 'tau1', 'tau1_n', 'tcal_single')}))

    def explain(Zw, PTw, maxk=5):
        p = P1(PS(Zw), PTw); n = len(Zw)
        S_list = [[] if p[i, 0] >= tau0 else [int(np.argmax(p[i, 1:])) + 1] for i in range(n)]
        act = np.array([i for i in range(n) if S_list[i]], int)
        if len(act):
            R = Zw.copy(); R[act] = deduct_groups(Zw[act], [S_list[i][0] for i in act]); BRW = br_probs(Zw)
            for stp in range(maxk - 1):
                if len(act) == 0: break
                F = cand_matrix(Zw[act], R[act], [S_list[i] for i in act], BRW[act], step)
                pp = dec.predict_proba(F.reshape(-1, 8))[:, 1].reshape(len(act), K)
                for n_, i in enumerate(act):
                    for g in blocked(S_list[i]): pp[n_, g - 1] = -1
                j = pp.argmax(1); sc = pp[np.arange(len(act)), j]; add = (sc > tau1) if conformal else (sc >= tau1)
                nxt = act[add]
                for i, e in zip(nxt, j[add] + 1): S_list[i].append(int(e))
                if len(nxt): R[nxt] = deduct_groups(R[nxt], [S_list[i][-1] for i in nxt])
                act = nxt
        return S_list

    if final:
        # calibration-only criterion (no test cycle, no real multi-fault cycle)
        Stc = explain(TC_Z, PT_TC[net]); f_tc = tune_sets(Stc, TC_T)
        q = compose('tcal', [2, 3], 1500, False, 23 + SEED); Sq = explain(q['Z'], q['pt'][net])
        fq = np.array([setf1(to_comp(a), to_comp(t)) for a, t in zip(Sq, q['truth'])]); kq = np.array([len(t) for t in q['truth']])
        cal = dict(tcal_normal_single_setF1=float(f_tc), tcal_normal_named=float(np.mean([len(Stc[i]) > 0 for i in np.where(N0)[0]])),
                   tcal_composed_k2_setF1=float(fq[kq == 2].mean()), tcal_composed_k3_setF1=float(fq[kq == 3].mean()),
                   tcal_composed_setF1=float(fq.mean()), n_composed=int(len(fq)))
        cal['criterion'] = 0.5 * (cal['tcal_normal_single_setF1'] + cal['tcal_composed_setF1'])
        info['calibration'] = cal; log(name, 'CALIBRATION CRITERION', json.dumps(cal))
        json.dump(cal, open(OUT + 'calib_%s.json' % name, 'w'), indent=1)
    json.dump(info, open(OUT + 'thresholds_%s.json' % name, 'w'), indent=1)
    evaluate(name, lambda Zw: explain(Zw, PT_TEST[net]))


if LADDER:
    run_config('L0-base-' + FTAG, step=True, four=True, conformal=False, temporal=False)
    run_config('L1-nostep-' + FTAG, step=False, four=True, conformal=False, temporal=False)
    run_config('L2-comp3-' + FTAG, step=False, four=False, conformal=False, temporal=False)
    run_config('L3-conformal-' + FTAG, step=False, four=False, conformal=True, temporal=False)
run_config('VioExplain-' + FTAG, step=False, four=False, conformal=True, temporal=True, final=True)
log('DONE', FTAG)
