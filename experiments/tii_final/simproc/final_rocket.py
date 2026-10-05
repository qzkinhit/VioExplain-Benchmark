"""VioExplain v3 final runs, time-series classification baselines (aeon) on the shared protocol of common.py.

Usage:  V3_MODE=f0 V3_LOG=log_NAME.txt V3_METRICS=metrics_NAME.json python final_rocket.py NAME
NAME in MiniRocket, MultiRocket, MultiRocketHydra, QUANT. Ridge-based classifiers give one-hot probabilities, so
their saved probabilities are re-scored with final_probs.py kind=argmax (empty set iff the predicted class is normal). Trained on raw single-fault fit windows standardized per sensor by the
normal fit windows; top-1 prediction with a conformal empty set calibrated on normal test-calibration windows.
"""
import os, sys
os.environ.setdefault('NUMBA_THREADING_LAYER', 'omp')   # the workqueue layer is not thread-safe and crashed with n_jobs > 1
from common import *
NAME = sys.argv[1]
from aeon.classification.convolution_based import MiniRocketClassifier, MultiRocketHydraClassifier, MultiRocketClassifier
from aeon.classification.interval_based import QUANTClassifier

nm = RAW['tr'][0][ok_tr[0]].reshape(-1, M); PM = nm.mean(0); PSD = nm.std(0) + 1e-8


def prep(X): return ((X - PM) / PSD).astype(np.float32).transpose(0, 2, 1).copy()


XTR = np.concatenate([RAW['tr'][c][ok_tr[c]] for c in range(C)])
mk = {'MiniRocket': lambda: MiniRocketClassifier(n_jobs=NJ, random_state=0),
      'MultiRocketHydra': lambda: MultiRocketHydraClassifier(n_jobs=NJ, random_state=0),
      'MultiRocket': lambda: MultiRocketClassifier(n_jobs=NJ, random_state=0),
      'QUANT': lambda: QUANTClassifier(random_state=0)}[NAME]
clf = mk().fit(prep(XTR), YB)
log(NAME, 'trained on', len(XTR), 'classes', list(clf.classes_)[:3], '...')


RIDGE = hasattr(clf, 'pipeline_') and hasattr(clf.pipeline_, 'decision_function')


def probs_raw(X, chunk=8000):
    """Class scores of a window batch. Ridge-based classifiers give hard predictions, so their decision values are
    turned into a softmax score; the conformal empty-set rule then applies to them as to every other classifier."""
    out = []
    for s in range(0, len(X), chunk):
        if RIDGE:
            d = clf.pipeline_.decision_function(prep(X[s:s + chunk])); d = 8.0 * (d - d.max(1, keepdims=True)); e = np.exp(d)
            out.append(e / e.sum(1, keepdims=True))
        else:
            out.append(clf.predict_proba(prep(X[s:s + chunk])))
    return np.concatenate(out)


CACHE = {}


def P_key(Zw):
    k = KEYOF[id(Zw)]
    if k not in CACHE: CACHE[k] = probs_raw(GRAW[k])
    return CACHE[k]


n0 = len(Z['tcal'][0]); per = n0 // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
s = np.sort(1 - probs_raw(TC_RAW[:n0][pick])[:, 0]); q0 = 1 - s[min(len(s) - 1, int(np.ceil(0.95 * (len(s) + 1))) - 1)]
RES[NAME + '_label_macroF1'] = float(f1_score(EV_Y, P_key(EV_Z).argmax(1), average='macro'))
evaluate(NAME, lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P_key(Zw)])
np.savez_compressed(OUT + 'probs_%s.npz' % NAME, **CACHE)
log('DONE', NAME)
LOG.flush(); sys.stdout.flush(); os._exit(0)   # skip interpreter teardown, where the numba thread pool aborted
