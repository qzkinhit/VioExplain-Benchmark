"""Deep and aeon baselines on the hydraulic benchmark (CPU), protocol of gpu_run.py, gpu_run2.py, final_rocket.py
and final_probs.py on the windows of hyd_common.py.

Usage:  HYD_MODE=f0 python hyd_deep.py 1D-CNN LSTM ResNet InceptionTime ML-CNN MiniRocket MultiRocket QUANT
        HYD_MODE=f100t python hyd_deep.py ML-CNN          (multi-label CNN with the composition of hyd_common.py)
Single-label nets (softmax over normal + K events) are trained on normal and single-fault fit cycles; top-1 with a
conformal empty set from the normal test-calibration cycles. Ridge-based aeon classifiers (MiniRocket, MultiRocket) give
one-hot probabilities; with HYD_ROCKET=soft (default) their decision values are turned into a softmax score
(temperature factor 8, as in final_rocket.py) and the conformal empty set of the other single-label methods applies; with
HYD_ROCKET=hard the set is empty iff the predicted class is normal. QUANT (extra trees) uses the conformal
empty set.
ML-CNN: sigmoid per event, threshold tuned as for BR-LGBM (HYD_CAL=one: normal and single-fault test-calibration cycles
plus composed calibration cycles, scored by the same network). Weights are saved to OUT/model_<name>.pt.
Architectures as in gpu_run.py / gpu_run2.py. Training: AdamW 1e-3, weight decay 1e-4, cosine schedule; the TEP runs use
40 epochs of batch 512 on about 38000 windows (about 3000 steps); here batch 32 and HYD_EP epochs (default 150), which
gives about 1100 steps on the 232 fit cycles. Probabilities are saved to OUT/probs_<name>.npz (keys test, tcal1).
"""
import sys
from hyd_common import *
import torch, torch.nn as nn
NAMES = sys.argv[1:]
torch.set_num_threads(NJ)
EP = int(os.environ.get('HYD_EP', '150')); BS = 32
PM_ = RAW['fit'][0].reshape(-1, M).mean(0)
PSD_ = np.concatenate([(XALL[ii] - XALL[ii].mean(0)).reshape(-1, M) for ii in FIT_GROUPS]).std(0) + 1e-8


def prep(X): return ((X - PM_) / PSD_).astype(np.float32).transpose(0, 2, 1).copy()


SETS = {'test': GRAW['test'], 'tcal1': GRAW['tcal1']}
N0 = TC_Y == 0


class CNN(nn.Module):
    def __init__(s, out=C):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(128, out)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))


class LSTM(nn.Module):
    def __init__(s):
        super().__init__(); s.r = nn.LSTM(M, 128, num_layers=2, batch_first=True, dropout=0.1); s.h = nn.Linear(128, C)
    def forward(s, x): o, _ = s.r(x.transpose(1, 2)); return s.h(o[:, -1])


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


def run_net(name, net, X, Y, multilabel=False, sets=None):
    sets = SETS if sets is None else sets
    torch.manual_seed(SEED); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    Xt = torch.as_tensor(prep(X)); Yt = torch.as_tensor(Y); gen = torch.Generator().manual_seed(SEED)
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(Xt), generator=gen)
        for s in range(0, len(Xt), BS):
            b = perm[s:s + BS]
            if len(b) < 2: continue
            o = net(Xt[b])
            loss = nn.functional.binary_cross_entropy_with_logits(o, Yt[b]) if multilabel else nn.functional.cross_entropy(o, Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for k, v in sets.items():
            o = net(torch.as_tensor(prep(v))); res[k] = (torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).numpy()
    np.savez_compressed(OUT + 'probs_%s.npz' % name, **res); log(name, 'trained', EP, 'epochs; probabilities saved')
    torch.save({'state': net.state_dict(), 'PM': PM_, 'PSD': PSD_, 'name': name}, OUT + 'model_%s.pt' % name)
    old = ROOT + 'results/hyd_v1/%s%s%s/probs_%s.npz' % (MODE, '' if EVT == 'comp' else '_' + EVT, TAG, name)
    if OUTSUB and os.path.exists(old):   # distance of this training from the probabilities stored in the default output directory
        o_ = dict(np.load(old)); chk = {'threads': NJ}
        for k in ('test', 'tcal1'):
            chk['max_abs_diff_' + k] = float(np.abs(res[k] - o_[k]).max())
            chk['share_same_side_of_0.5_' + k if multilabel else 'argmax_agree_' + k] = float(((res[k] > 0.5) == (o_[k] > 0.5)).mean()) if multilabel else float((res[k].argmax(1) == o_[k].argmax(1)).mean())
        log(name, 'check against the probabilities of the default output directory', json.dumps(chk)); json.dump(chk, open(OUT + 'retrain_check_%s.json' % name, 'w'), indent=1)
    return res


def score(name, PR, kind):
    P_key = lambda Zw: PR[KEYOF[id(Zw)]]
    if kind != 'multi':
        RES[name + '_label_macroF1'] = float(f1_score(T01_Y, PR['test'][T01].argmax(1), average='macro'))
        json.dump({'label_macroF1': RES[name + '_label_macroF1']}, open(OUT + 'label_%s.json' % name, 'w'))
    if kind == 'argmax':
        evaluate(name, lambda Zw: [[] if np.argmax(p) == 0 else [int(np.argmax(p))] for p in P_key(Zw)])
    elif kind == 'single':
        s = np.sort(1 - PR['tcal1'][N0][:, 0]); n = len(s); q0 = 1 - s[min(n - 1, int(np.ceil(0.95 * (n + 1))) - 1)]
        evaluate(name, lambda Zw: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P_key(Zw)])
    elif CAL == 'one' and 'calcomp' in PR:
        best, _ = tune_ml_probs(name, PR['tcal1'], PR['calcomp']); log(name, 'threshold and gate (one calibration rule)', best)
        evaluate(name, lambda Zw: ml_sets(P_key(Zw), best[0], best[1]))
        # same network with the single-fault rule (test-calibration cycles only), kept apart for attribution
        prev = tune_thr(PR['tcal1'], TC_T); log(name, 'threshold (single-fault rule, attribution only)', prev)
        evaluate(name + '-singlecal', lambda Zw: [[k + 1 for k in np.where(p > prev[0])[0]] for p in P_key(Zw)])
    else:
        best = tune_thr(PR['tcal1'], TC_T, name); log(name, 'threshold', best)
        evaluate(name, lambda Zw: [[k + 1 for k in np.where(p > best[0])[0]] for p in P_key(Zw)])


nets = {'1D-CNN': lambda: CNN(C), 'LSTM': LSTM, 'ResNet': ResNet, 'InceptionTime': InceptionTime}
if MODE == 'f0':
    for nm in nets:
        if nm in NAMES: score(nm, run_net(nm, nets[nm](), XBR, YB.astype(np.int64)), 'single')
if 'ML-CNN' in NAMES:
    XA = [XBR]; YA = [Yml[:len(XB)].astype(np.float32)]
    for ev, Xs in AUG_RAW:
        y = np.zeros((len(Xs), K), np.float32); y[:, np.array(ev) - 1] = 1; XA.append(Xs); YA.append(y)
    XA = np.concatenate(XA); YA = np.concatenate(YA); log('ML-CNN training cycles', len(XA))
    score('ML-CNN', run_net('ML-CNN', CNN(K), XA, YA, multilabel=True, sets=dict(SETS, calcomp=calc()['X']) if CAL == 'one' else None), 'multi')
if MODE == 'f0':
    from aeon.classification.convolution_based import MiniRocketClassifier, MultiRocketClassifier
    from aeon.classification.interval_based import QUANTClassifier
    mk = {'MiniRocket': (lambda: MiniRocketClassifier(n_jobs=NJ, random_state=0), 'argmax'),
          'MultiRocket': (lambda: MultiRocketClassifier(n_jobs=NJ, random_state=0), 'argmax'),
          'QUANT': (lambda: QUANTClassifier(random_state=0), 'single')}
    for nm, (f, kind) in mk.items():
        if nm not in NAMES: continue
        clf = f().fit(prep(XBR), YB)
        ridge = os.environ.get('HYD_ROCKET', 'soft') == 'soft' and hasattr(clf, 'pipeline_') and hasattr(clf.pipeline_, 'decision_function')

        def pr(v, clf=clf, ridge=ridge):
            if not ridge: return clf.predict_proba(prep(v))
            d = clf.pipeline_.decision_function(prep(v)); d = 8.0 * (d - d.max(1, keepdims=True)); e = np.exp(d)
            return e / e.sum(1, keepdims=True)
        PR = {k: pr(v) for k, v in SETS.items()}
        np.savez_compressed(OUT + 'probs_%s.npz' % nm, **PR); log(nm, 'trained', 'softmax of ridge decision values' if ridge else 'predict_proba')
        score(nm, PR, 'single' if ridge else kind)
log('DONE', NAMES)
