"""Recovery certificate of Theorem 3 measured with the FINAL pipeline (final_fuse5.py, tag B) on TEP-C pairs and triples.

Usage:
  V3_MODE=f100t V3_ABL=full V3_FUNC=1 V3_NOTWIN=0 V3_NOSTEP=1 V3_TFEAT=0 V3_RES=final_v4/cert_work/<tag> \
  V3_TMODEL=.../resnet_model.pt NJ=3 OMP_NUM_THREADS=3 CERT_CACHE=<dir> python final_cert5.py .../resnet_probs_v2.npz
The code of final_fuse5.py is executed verbatim up to its final evaluate() call, so knowledge, functional-constraint
units, deduction operators, scorer PS, BR detectors, the addition decision DEC (no step feature, compositions of at most
three events), the fusion weight A, TAU0F (tau_0 of the mixed posterior) and TAU_C (tau_1, split-conformal, one window per
calibration run) are built exactly as in the final run. Fitted LightGBM models are cached in CERT_CACHE (key = parameters
and training data), so a second execution skips training.

Algorithm traced (explain_t of final_fuse5.py):
  step 0: p = mix(PS(window), PT(window), A); output {} if p_0 >= TAU0F, else first event = argmax_{e>=1} p_e, deduct it
  later : z = DEC(cand_matrix(window, residual, S, BRW)); candidate = argmax over unselected events; add iff z > TAU_C
          (strict); deduct; at most 5 events
Certificate (Theorem 3 as stated; H = effective set from eff_truth, windows with empty H are skipped):
  p_0 < TAU0F;  eta_0 = max_{e in H} p_e - max_{e notin H} p_e > 0;
  every visited prefix S strictly inside H: psi_S = Fn(paired run in which only H minus S act),
      z_ref = DEC(cand_matrix(window, psi_S, S, BRW)), z_act the same on the actual residual,
      omega_S = max over unselected events of |z_act - z_ref|,
      eta_S = max_{E in H minus S} min(z_ref_E - max_{e notin H} z_ref_e, z_ref_E - TAU_C),  holds iff eta_S > 2 omega_S;
  S = H: psi_H = Fn(paired fault-free run), eta_H = TAU_C - max_{e notin H} z_ref_e, holds iff eta_H > omega_H.
Failure labels at a step: stop_empty (p_0 >= TAU0F), first_choice (eta_0 <= 0), selection (reference lead of the best
true event over false events <= 0 at a prefix), stop_early (best true reference score <= TAU_C), stop_late (eta_H <= 0),
perturbation (reference margin positive but not above the omega bound).
Interaction strength: ||phi(X_E) - sum_e phi(X_e) + (m-1) phi(X_0)|| / sqrt(sum_e ||phi(X_e) - phi(X_0)||^2), phi = Fn
clipped to [-50, 50] (primary) and unclipped (secondary), Frobenius norms.
Outputs in OUT: certificate_run.json (summary), certificate_windows.pkl (per-window records).
"""
import os, sys, json, pickle, hashlib
SCRIPTS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SCRIPTS)
import numpy as _np
import lightgbm as _lgb

CACHE = os.environ['CERT_CACHE']; os.makedirs(CACHE, exist_ok=True)
SMOKE = os.environ.get('CERT_SMOKE') == '1'
_Base = _lgb.LGBMClassifier
_NFIT = [0, 0, 0]
# Optional controls for reproducing the stored run (defaults leave final_fuse5.py unchanged):
#   CERT_DEC_FORCE=col|row   histogram mode of the addition decision (LightGBM otherwise picks it by a timing test)
#   CERT_ORDER_FALLBACK=1    models other than the addition decision missing in the cache are taken by fit order
DEC_FORCE = os.environ.get('CERT_DEC_FORCE', ''); ORDER_FALLBACK = os.environ.get('CERT_ORDER_FALLBACK') == '1'
import glob as _glob
_ORDER = sorted(_glob.glob(os.path.join(CACHE, '*.pkl')), key=os.path.getmtime)


class LGBMClassifier(_Base):
    """LightGBM classifier whose fitted state is cached on disk (key: parameters without n_jobs, X and y)."""
    def fit(self, X, y, **kw):
        if SMOKE: self.set_params(n_estimators=6)
        idx = _NFIT[2]; _NFIT[2] += 1; is_dec = self.get_params().get('num_leaves') == 15
        if is_dec and DEC_FORCE: self.set_params(**{'force_%s_wise' % DEC_FORCE: True})
        X = _np.ascontiguousarray(X); y = _np.ascontiguousarray(y)
        h = hashlib.sha1(repr(sorted((k, repr(v)) for k, v in self.get_params().items() if k != 'n_jobs')).encode())
        h.update(repr((X.shape, str(X.dtype), y.shape, str(y.dtype))).encode()); h.update(X.tobytes()); h.update(y.tobytes())
        if is_dec and DEC_FORCE: h.update(('n_jobs=%s' % self.n_jobs).encode())
        p = os.path.join(CACHE, h.hexdigest() + '.pkl')
        if not os.path.exists(p) and ORDER_FALLBACK and not is_dec and idx < len(_ORDER):
            print('cert5 cache: content key missing, model %d taken by fit order' % idx, flush=True); p = _ORDER[idx]
        if os.path.exists(p):
            nj = self.n_jobs; self.__dict__.update(pickle.load(open(p, 'rb'))); self.n_jobs = nj; _NFIT[1] += 1
            return self
        _Base.fit(self, X, y, **kw); _NFIT[0] += 1
        pickle.dump(self.__dict__, open(p + '.tmp', 'wb')); os.replace(p + '.tmp', p)
        return self


_lgb.LGBMClassifier = LGBMClassifier
_src = open(os.path.join(SCRIPTS, 'final_fuse5.py')).read()
_key = "evaluate('VioExplain' + ("
assert _src.count(_key) == 1
exec(compile(_src.split(_key)[0], os.path.join(SCRIPTS, 'final_fuse5.py'), 'exec'))

assert not NOCOMP and NOSTEP and not TFEAT and not DFEAT and FUNC and not NOTWIN and NF == 8
STORED = dict(label_macroF1=0.918962931718175, A=0.5, TAU0=0.09988482553445, TAU_C=0.001571928581570646,
              TAU0F=0.11486759672991098, dec_rows=105643, dec_pos=4139, pair_eff=0.9139356765243659,
              triple_eff=0.9220834765341581)
HERE = dict(label_macroF1=RES['Ours_label_macroF1'], A=float(A), TAU0=float(TAU0), TAU_C=float(TAU_C), TAU0F=float(TAU0F),
            dec_rows=len(lab), dec_pos=int(np.sum(lab)))
log('cert5: models trained', _NFIT[0], 'loaded from cache', _NFIT[1], 'NJ', NJ, 'OMP', os.environ.get('OMP_NUM_THREADS'), 'DEC_FORCE', DEC_FORCE)
log('cert5: here  ', json.dumps(HERE))
log('cert5: stored', json.dumps(STORED))
log('cert5: bit-identical thresholds', all(HERE[k] == STORED[k] for k in HERE))


def zscores(Zw, R, S_list, BRW):
    F = cand_matrix(Zw, R, S_list, BRW)
    return DEC.predict_proba(F.reshape(-1, NF))[:, 1].reshape(len(Zw), K)


def bitmask(ev, sub): return sum(1 << ev.index(e) for e in sub)


def group_data(kind):
    groups = COMP if kind == 'pair' else TRIP
    d, meta = ('test_pairs', MP) if kind == 'pair' else ('test_triples', MT)
    Zs, FS, EVS, H, PTr, I_c, I_u, GK = [], [], [], [], [], [], [], []
    for ev, Zw, tru, V, causes in groups:
        ev = tuple(ev); m = len(ev)
        Wd, valid = sim_windows(d, meta, ev); idx = np.where(valid)[0]
        F = np.zeros((len(idx), 2 ** m) + Zw.shape[1:], np.float32)
        for s_, X in Wd.items(): F[:, bitmask(ev, s_)] = Fn(X[idx])
        assert F.shape[0] == len(Zw) and np.array_equal(F[:, 2 ** m - 1], Zw)
        for clip, store in ((True, I_c), (False, I_u)):
            Fc = np.clip(F, -50, 50).astype(np.float64) if clip else F.astype(np.float64)
            ds = [Fc[:, 1 << j] - Fc[:, 0] for j in range(m)]
            na = Fc[:, 2 ** m - 1] - Fc[:, 0] - sum(ds)
            den = np.sqrt(sum((x ** 2).sum(axis=(1, 2)) for x in ds))
            store.append(np.sqrt((na ** 2).sum(axis=(1, 2))) / np.maximum(den, 1e-9))
        key = KEYOF[id(Zw)]; assert len(PT[key]) == len(Zw)
        Zs.append(Zw); FS += list(F); EVS += [ev] * len(Zw); H += [set(t) for t in tru]; PTr.append(PT[key])
        GK += [key] * len(Zw)
    return dict(Z=np.concatenate(Zs), FS=FS, EVS=EVS, H=H, PT=np.concatenate(PTr), I=np.concatenate(I_c),
                I_raw=np.concatenate(I_u), key=GK)


def step0(P, tau0, H):
    S_init, s0 = [], []
    for i in range(len(P)):
        S_init.append([] if P[i, 0] >= tau0 else [int(np.argmax(P[i, 1:])) + 1])
        if not H[i]: s0.append(None); continue
        tr = np.array(sorted(H[i])); fa = np.array([e for e in range(1, C) if e not in H[i]])
        es = float(P[i, tr].max() - P[i, fa].max()); ez = float(tau0 - P[i, 0])
        f = 'stop_empty' if ez <= 0 else ('first_choice' if es <= 0 else None)
        s0.append(dict(S=(), kind='empty', eta_s=es, eta_z=ez, omega=0.0, fail=f, fail_old=f, fail_all=f))
    return S_init, s0


def trace(Zw, BRW, FS, EVS, H, S_init, maxk=5):
    """explain_t() from the given first-step result, with the reference scores on the true path."""
    n = len(Zw); S = [list(s) for s in S_init]; rec = [[] for _ in range(n)]
    onpath = np.array([len(S[i]) > 0 and set(S[i]) <= H[i] for i in range(n)])
    act = np.array([i for i in range(n) if S[i]], int)
    R = Zw.copy()
    if len(act): R[act] = deduct_groups(Zw[act], [S[i][0] for i in act])
    for step in range(maxk - 1):
        if len(act) == 0: break
        za = zscores(Zw[act], R[act], [S[i] for i in act], BRW[act]); zam = za.copy()
        for n_, i in enumerate(act):
            for g in S[i]: zam[n_, g - 1] = -1
        j = zam.argmax(1); zbest = zam[np.arange(len(act)), j]; add = zbest > TAU_C
        rr = np.array([n_ for n_, i in enumerate(act) if onpath[i]], int)
        if len(rr):
            ii = act[rr]
            psi = np.stack([FS[i][bitmask(EVS[i], H[i] - set(S[i]))] for i in ii])
            zr = zscores(Zw[ii], psi, [S[i] for i in ii], BRW[ii])
            for r_, (n_, i) in enumerate(zip(rr, ii)):
                cand = np.array([e for e in range(1, C) if e not in S[i]])
                om = float(np.abs(za[n_, cand - 1] - zr[r_, cand - 1]).max())
                oma = float(np.abs(za[n_] - zr[r_]).max())
                rem = sorted(H[i] - set(S[i])); fa = np.array([e for e in range(1, C) if e not in H[i]])
                zf = float(zr[r_, fa - 1].max())
                base = dict(S=tuple(S[i]), omega=om, omega_all=oma, z_act_best=float(zbest[n_]), cand=int(j[n_]) + 1,
                            added=bool(add[n_]), z_ref_false=zf, z_act_false=float(za[n_, fa - 1].max()))
                if rem:
                    zt = float(zr[r_, np.array(rem) - 1].max()); es = zt - zf; ez = zt - TAU_C; eta = min(es, ez)
                    pre = 'selection' if es <= 0 else 'stop_early' if ez <= 0 else None
                    f = pre or ('perturbation' if eta <= 2 * om else None)
                    fo = pre or ('perturbation' if (es <= 2 * om or ez <= om) else None)
                    fa_ = pre or ('perturbation' if eta <= 2 * oma else None)
                    rec[i].append(dict(base, kind='prefix', eta_s=es, eta_z=ez, eta=eta, z_ref_true=zt,
                                       z_act_true=float(za[n_, np.array(rem) - 1].max()), fail=f, fail_old=fo, fail_all=fa_))
                else:
                    eh = TAU_C - zf
                    pre = 'stop_late' if eh <= 0 else None
                    f = pre or ('perturbation' if eh <= om else None)
                    fa_ = pre or ('perturbation' if eh <= oma else None)
                    rec[i].append(dict(base, kind='H', eta_H=eh, eta=eh, fail=f, fail_old=f, fail_all=fa_))
        nxt = act[add]
        for i, e in zip(nxt, j[add] + 1):
            S[i].append(int(e)); onpath[i] = onpath[i] and (int(e) in H[i])
        for i in act[~add]: onpath[i] = False
        if len(nxt): R[nxt] = deduct_groups(R[nxt], [S[i][-1] for i in nxt])
        act = nxt
    return S, rec


CAT = {'first_choice': 'first_choice', 'selection': 'first_choice', 'perturbation': 'perturbation', 'stop_empty': 'stop',
       'stop_early': 'stop', 'stop_late': 'stop'}


def per_window(H, S_fin, s0, rec):
    out = []
    for i in range(len(H)):
        if not H[i]: out.append(None); continue
        steps = [s0[i]] + rec[i]
        fails = [(k, st['fail']) for k, st in enumerate(steps) if st['fail'] is not None]
        reachedH = any(st['kind'] == 'H' for st in steps)
        cert = (not fails) and reachedH
        cert_old = reachedH and all(st['fail_old'] is None for st in steps)
        cert_all = reachedH and all(st['fail_all'] is None for st in steps)
        recov = set(S_fin[i]) == H[i]
        out.append(dict(cert=cert, cert_old=cert_old, cert_all=cert_all, recov=recov,
                        first_fail=fails[0][1] if fails else None, first_fail_step=fails[0][0] if fails else None,
                        dev_fail=None if recov else steps[-1]['fail'], nH=len(H[i]), reachedH=reachedH, steps=steps,
                        pred=list(S_fin[i]), H=sorted(H[i])))
    return out


def q(x, p): return float(np.quantile(x, p)) if len(x) else None


def where(w):
    st = w['steps'][w['first_fail_step']]; f = w['first_fail']
    lab_ = {'empty': 'step0', 'prefix': 'prefix', 'H': 'at_H'}[st['kind']] + ':' + f
    if f == 'perturbation' and st['kind'] == 'prefix':
        a = st['eta_s'] <= 2 * st['omega']; b = st['eta_z'] <= 2 * st['omega']
        lab_ += ':' + ('selection_and_threshold' if a and b else 'selection' if a else 'threshold')
    return lab_


def summarize(win, I, I_raw):
    sel = [i for i, w in enumerate(win) if w is not None]
    W_ = [win[i] for i in sel]; cert = np.array([w['cert'] for w in W_]); rec_ = np.array([w['recov'] for w in W_])
    nr = [w for w in W_ if not w['recov']]; unc = [w for w in W_ if not w['cert']]
    out = dict(n_windows=len(W_), n_windows_empty_H=len(win) - len(W_), certified=float(cert.mean()),
               recovered_exact=float(rec_.mean()), n_certified=int(cert.sum()), n_recovered=int(rec_.sum()),
               recovered_given_certified=float(rec_[cert].mean()) if cert.any() else None,
               certified_given_recovered=float(cert[rec_].mean()) if rec_.any() else None,
               n_certified_not_recovered=int((cert & ~rec_).sum()), n_not_recovered=len(nr), n_uncertified=len(unc),
               n_not_recovered_without_failed_condition=int(sum(w['first_fail'] is None for w in nr)),
               n_uncertified_without_failed_condition=int(sum(w['first_fail'] is None for w in unc)),
               certified_rule_of_earlier_analysis=float(np.mean([w['cert_old'] for w in W_])),
               certified_omega_over_all_events=float(np.mean([w['cert_all'] for w in W_])))
    for lab_, key in (('first_fail', 'first_fail'), ('deviation_step_fail', 'dev_fail')):
        sub = {}; cat = {'first_choice': 0, 'perturbation': 0, 'stop': 0}
        for w in nr:
            f = w[key]; sub[str(f)] = sub.get(str(f), 0) + 1
            if f in CAT: cat[CAT[f]] += 1
        out['not_recovered_' + lab_] = {k: v / max(len(nr), 1) for k, v in cat.items()}
        out['not_recovered_' + lab_ + '_detail'] = {k: v / max(len(nr), 1) for k, v in sub.items()}
    cat = {'first_choice': 0, 'perturbation': 0, 'stop': 0}
    for w in unc:
        if w['first_fail'] is not None: cat[CAT[w['first_fail']]] += 1
    out['uncertified_first_fail'] = {k: v / max(len(unc), 1) for k, v in cat.items()}
    for nm, ws in (('not_recovered_first_fail_by_step', nr), ('uncertified_first_fail_by_step', unc)):
        cnt = {}
        for w in ws:
            k = where(w) if w['first_fail'] is not None else 'none'; cnt[k] = cnt.get(k, 0) + 1
        out[nm] = {k: v / max(len(ws), 1) for k, v in sorted(cnt.items())}
    out['by_H_size'] = {}
    for k in sorted(set(w['nH'] for w in W_)):
        m = np.array([w['nH'] == k for w in W_])
        out['by_H_size'][str(k)] = dict(n=int(m.sum()), certified=float(cert[m].mean()), recovered_exact=float(rec_[m].mean()))
    pre = [st for w in W_ for st in w['steps'] if st['kind'] == 'prefix']
    hs = [st for w in W_ for st in w['steps'] if st['kind'] == 'H']
    e0 = [st for w in W_ for st in w['steps'] if st['kind'] == 'empty']
    out['margins'] = dict(
        empty_step=dict(n=len(e0), eta_0_median=q([s['eta_s'] for s in e0], .5), tau0_minus_p0_median=q([s['eta_z'] for s in e0], .5),
                        share_eta_0_pos=float(np.mean([s['eta_s'] > 0 for s in e0])), share_p0_below_tau0=float(np.mean([s['eta_z'] > 0 for s in e0]))),
        prefix_steps=dict(n=len(pre), omega_median=q([s['omega'] for s in pre], .5), omega_p90=q([s['omega'] for s in pre], .9),
                          eta_S_median=q([s['eta'] for s in pre], .5), eta_S_p10=q([s['eta'] for s in pre], .1),
                          share_eta_S_pos=float(np.mean([s['eta'] > 0 for s in pre])) if pre else None,
                          share_holds=float(np.mean([s['fail'] is None for s in pre])) if pre else None),
        H_steps=dict(n=len(hs), omega_median=q([s['omega'] for s in hs], .5), omega_p90=q([s['omega'] for s in hs], .9),
                     eta_H_median=q([s['eta_H'] for s in hs], .5), eta_H_p10=q([s['eta_H'] for s in hs], .1),
                     share_eta_H_pos=float(np.mean([s['eta_H'] > 0 for s in hs])) if hs else None,
                     share_holds=float(np.mean([s['fail'] is None for s in hs])) if hs else None))
    h2 = np.array([w['nH'] >= 2 for w in W_])
    for nm, II in (('interaction_terciles', I), ('interaction_terciles_unclipped', I_raw)):
        x = II[sel]; cuts = np.quantile(x, [1 / 3, 2 / 3]); t = np.digitize(x, cuts)
        out[nm] = dict(cutpoints=[float(c) for c in cuts], median_all=float(np.median(x)), terciles=[
            dict(tercile=k + 1, n=int((t == k).sum()), I_median=float(np.median(x[t == k])),
                 certified=float(cert[t == k].mean()), recovered_exact=float(rec_[t == k].mean()),
                 certified_given_recovered=float(cert[(t == k) & rec_].mean()) if ((t == k) & rec_).any() else None,
                 certified_H2=float(cert[(t == k) & h2].mean()) if ((t == k) & h2).any() else None)
            for k in range(3)])
    return out


SAVED = pickle.load(open('/path/to/vioexplain/results/final_v4/f100t/records_VioExplain-nostep-B.pkl', 'rb'))
RESULT = dict(pipeline='final_fuse5.py tag B (V3_MODE=f100t V3_ABL=full V3_FUNC=1 V3_NOTWIN=0 V3_NOSTEP=1 V3_TFEAT=0)',
              threads=dict(NJ=NJ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')), dec_histogram_mode=DEC_FORCE or 'auto',
              models_trained_here=_NFIT[0], models_loaded_from_cache=_NFIT[1], retrain=HERE, stored_run=STORED,
              thresholds_bit_identical_to_stored_run=all(HERE[k] == STORED[k] for k in HERE), groups={})
WIN = {}
for kind in ('pair', 'triple'):
    G = group_data(kind); n = len(G['Z']); H = G['H']
    log('cert5', kind, 'windows', n, 'nonempty H', sum(bool(h) for h in H))
    BRW = br_probs(G['Z']); P_fused = mix(PS(G['Z']), G['PT'], A)
    Si, s0 = step0(P_fused, TAU0F, H)
    S_f, rec = trace(G['Z'], BRW, G['FS'], G['EVS'], H, Si)
    log('cert5', kind, 'traced')
    groups = COMP if kind == 'pair' else TRIP
    off = np.cumsum([0] + [len(g[1]) for g in groups])
    # (a) same models: explain_t() of final_fuse5.py on whole groups
    gsel = range(len(groups)) if kind == 'triple' else np.random.default_rng(3).choice(len(groups), 25, replace=False)
    mism = 0; nchk = 0
    for g in gsel:
        Zw = groups[g][1]; a, b = off[g], off[g + 1]; nchk += len(Zw)
        mism += sum(x != y for x, y in zip(explain_t(Zw), S_f[a:b]))
    # (b) stored run: saved predictions of the final run (another training of the same code)
    sv = SAVED[kind]; assert len(sv) == len(groups); S_saved = []
    for (ev_s, S_s), g in zip(sv, groups):
        assert tuple(ev_s) == tuple(g[0]) and len(S_s) == len(g[1]); S_saved += [list(s) for s in S_s]
    tru_all = [sorted(h) for h in H]
    same_list = float(np.mean([a == b for a, b in zip(S_f, S_saved)]))
    same_set = float(np.mean([set(a) == set(b) for a, b in zip(S_f, S_saved)]))
    eff_here = float(np.mean([setf1(a, b) for a, b in zip(S_f, tru_all)]))
    eff_saved = float(np.mean([setf1(a, b) for a, b in zip(S_saved, tru_all)]))
    rec_here = np.array([set(a) == set(b) for a, b in zip(S_f, tru_all)]); rec_saved = np.array([set(a) == set(b) for a, b in zip(S_saved, tru_all)])
    ne = np.array([bool(h) for h in H])
    chk = dict(same_models_windows=nchk, same_models_mismatches=int(mism), stored_windows=n,
               stored_identical_ordered_share=same_list, stored_identical_set_share=same_set,
               stored_n_different_sets=int(sum(set(a) != set(b) for a, b in zip(S_f, S_saved))),
               eff_traced=eff_here, eff_stored_records=eff_saved, eff_stored_metrics=STORED[kind + '_eff'],
               exact_recovery_traced_nonemptyH=float(rec_here[ne].mean()), exact_recovery_stored_nonemptyH=float(rec_saved[ne].mean()))
    log('cert5', kind, 'implementation check', json.dumps(chk))
    w = per_window(H, S_f, s0, rec)
    # certificate against the stored predictions: certified windows whose stored prediction is not H
    cert_idx = [i for i, x in enumerate(w) if x is not None and x['cert']]
    chk['certified_windows_recovered_in_stored_records'] = float(np.mean([rec_saved[i] for i in cert_idx])) if cert_idx else None
    chk['n_certified_not_recovered_in_stored_records'] = int(sum(not rec_saved[i] for i in cert_idx))
    WIN[kind] = dict(win=w, I=G['I'], I_raw=G['I_raw'], key=G['key'], saved=S_saved)
    sm = summarize(w, G['I'], G['I_raw']); sm['implementation_check'] = chk
    RESULT['groups'][kind] = sm
    log('cert5', kind, json.dumps({k: sm[k] for k in ('n_windows', 'certified', 'recovered_exact', 'recovered_given_certified',
                                                       'certified_given_recovered', 'n_certified_not_recovered',
                                                       'n_not_recovered_without_failed_condition', 'not_recovered_first_fail',
                                                       'not_recovered_first_fail_by_step', 'certified_rule_of_earlier_analysis',
                                                       'certified_omega_over_all_events')}))
    log('cert5', kind, 'terciles', json.dumps(sm['interaction_terciles']), 'unclipped', json.dumps(sm['interaction_terciles_unclipped']))
    log('cert5', kind, 'margins', json.dumps(sm['margins']), 'by_H', json.dumps(sm['by_H_size']))
    json.dump(RESULT, open(OUT + 'certificate_run.json', 'w'), indent=1)
pickle.dump(WIN, open(OUT + 'certificate_windows.pkl', 'wb'))
log('cert5 DONE')
