"""MantisV2 baseline on the hydraulic benchmark, as in gpu_run.py of the TEP runs: frozen MantisV2 embedding of every
sensor of a cycle (60 samples resized to 512), PCA(16) per sensor fitted on the normal and single-fault fit cycles,
LightGBM head, top-1 with the conformal empty set of the other single-label methods. Protocol f0.

Usage:  HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_mantis.py prep     writes OUT/mantis_in.npz (standardized cycles, N x M x W)
        (GPU)                     python hyd_mantis_embed.py cuda:0  writes mantis_emb<TAG>.npz (N x M x D, float16)
        HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_mantis.py score    PCA + LightGBM head + evaluation, key MantisV2
The input standardization is that of hyd_deep.py (centre: normal fit cycles; scale: pooled within-condition std).
"""
import sys
from hyd_common import *
PM_ = RAW['fit'][0].reshape(-1, M).mean(0)
PSD_ = np.concatenate([(XALL[ii] - XALL[ii].mean(0)).reshape(-1, M) for ii in FIT_GROUPS]).std(0) + 1e-8
if sys.argv[1] == 'prep':
    S = ((XALL - PM_) / PSD_).astype(np.float32).transpose(0, 2, 1).copy()
    np.savez_compressed(OUT + 'mantis_in.npz', S=S); log('mantis input saved', S.shape)
else:
    from sklearn.decomposition import PCA
    E = np.load(OUT + 'mantis_emb.npz')['E']; assert E.shape[:2] == (NCYC, M), E.shape
    fit_idx = np.concatenate([IDX['fit'][c] for c in range(C)])
    pcas = [PCA(16, random_state=0).fit(E[fit_idx][:, j, :].astype(np.float32)) for j in range(M)]

    def red(idx): return np.concatenate([pcas[j].transform(E[idx][:, j, :].astype(np.float32)) for j in range(M)], 1)

    clf = mk_lgb().fit(red(fit_idx), YB)
    PR = {'test': clf.predict_proba(red(TEST_IDX)), 'tcal1': clf.predict_proba(red(TC1))}
    np.savez_compressed(OUT + 'probs_MantisV2.npz', **PR)
    s = np.sort(1 - PR['tcal1'][TC_Y == 0][:, 0]); n = len(s); q0 = 1 - s[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
    f = float(f1_score(T01_Y, PR['test'][T01].argmax(1), average='macro'))
    json.dump({'label_macroF1': f}, open(OUT + 'label_MantisV2.json', 'w'))
    log('MantisV2 embedding', E.shape, 'label macro F1 (normal+single test cycles)', f, 'q0', float(q0))
    evaluate('MantisV2', lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in PR[KEYOF[id(Zw)]]])
log('DONE', sys.argv[1:])
