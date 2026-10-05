"""Retrain the multi-label CNNs (f0 and f100t) exactly as gpu_run2.py and save weights for CPU inference.
VioExplain v3 GPU baselines, second batch (same windows as final_run.py and gpu_run.py).

Single-label deep diagnosers trained on single-fault fit windows: LSTM, ResNet (Wang et al. 2017) and InceptionTime
(one network). Multi-label 1D-CNN with a sigmoid output per fault: ML-CNN-f0 on single faults only and ML-CNN-f100t
with the same counterfactual composition as final_run.py (all fault pairs superposed from paired runs plus random
triples). Saves per-window probabilities for every evaluation group to OUT/<name>_probs.npz.
Usage: python gpu_run2.py cuda:N
"""
import os, sys, time, json, hashlib, itertools
import numpy as np
import torch, torch.nn as nn
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
ROOT = '/path/to/vioexplain/'
TRAIN = ROOT + 'data/rieth2017/training_arrays/'; TEST = ROOT + 'data/rieth2017/testing_arrays/'
PART = ROOT + 'data/rieth2017/run_partitions.csv'; SIM = ROOT + 'data/tepsim/'
OUT = ROOT + 'results/final_v1_gpu/'; os.makedirs(OUT, exist_ok=True)
W = 64; C = 21; M = 52
EFF = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
t0 = time.time()


def log(*a):
    s = ' '.join(str(x) for x in a) + '  [%.0fs]' % (time.time() - t0); print(s, flush=True)
    open(OUT + 'log_save2.txt', 'a').write(s + '\n')


def frozen(Xw): return (Xw.std(1) < 1e-9).sum(1) >= 30


def runs_of(split):
    ids = []
    for line in open(PART).read().strip().split('\n')[1:]:
        f, r, s, _ = line.split(',')
        if f == '0' and s == split: ids.append(int(r) - 1)
    return sorted(ids)


def wins(a, runs, onset):
    out = []
    for r in runs:
        x = np.asarray(a[r])
        for s in range(onset, x.shape[0] - W + 1, W): out.append(x[s:s + W])
    return np.stack(out).astype(np.float32)


def hsplit(r): return int(hashlib.sha256(('vioexplain-v3-test-split:%d' % r).encode()).hexdigest(), 16) % 5 == 0


FIT = runs_of('fit'); TCAL = [r for r in range(500) if hsplit(r)]; TEV = [r for r in range(500) if not hsplit(r)]
tr = {c: np.load(TRAIN + 'class_%02d.npy' % c, mmap_mode='r') for c in range(C)}
te = {c: np.load(TEST + 'class_%02d.npy' % c, mmap_mode='r') for c in range(C)}
RAW = {'tr': {c: wins(tr[c], FIT, 20) for c in range(C)}, 'tcal': {c: wins(te[c], TCAL, 160) for c in range(C)},
       'ev': {c: wins(te[c], TEV, 160) for c in range(C)}}
FR = {sp: {c: frozen(RAW[sp][c]) for c in range(C)} for sp in RAW}
nm = RAW['tr'][0].reshape(-1, M); PM = nm.mean(0); PSD = nm.std(0) + 1e-8


def std(X): return ((X - PM) / PSD).astype(np.float32)


XTR = np.concatenate([RAW['tr'][c][~FR['tr'][c]] for c in range(C)]); YTR = np.concatenate([[c] * int((~FR['tr'][c]).sum()) for c in range(C)])
# evaluation sets identical to final_run.py (single sets keep normal windows and drop frozen faulty windows)
def single(sp):
    X = [RAW[sp][0]] + [RAW[sp][c][~FR[sp][c]] for c in range(1, C)]
    return np.concatenate(X)
SETS = {'ev': single('ev'), 'tcal': single('tcal')}
MP = json.load(open(SIM + 'test_pairs/meta.json')); MT = json.load(open(SIM + 'test_triples/meta.json'))


def sim_windows(d, meta, events):
    onset = meta['onset']; subs = [tuple(sorted(c)) for r in range(len(events) + 1) for c in itertools.combinations(events, r)]
    out = {}; valid = None
    for s_ in subs:
        key = 'normal' if not s_ else '+'.join(str(e) for e in s_)
        a = np.load(SIM + d + '/set_%s.npy' % key, mmap_mode='r'); sh = [r['shutdown_sample'] for r in meta['runs'][key]]
        ws = []; ok = []
        for i in range(a.shape[0]):
            for st in range(onset, a.shape[1] - W + 1, W):
                ws.append(np.asarray(a[i, st:st + W])); ok.append(sh[i] < 0 or sh[i] >= st + W)
        out[s_] = np.stack(ws).astype(np.float32); ok = np.array(ok); valid = ok if valid is None else valid & ok
    return out, valid


GROUPS = {}
for (A, B) in itertools.combinations(range(1, C), 2):
    Wd, valid = sim_windows('test_pairs', MP, (A, B)); idx = np.where(valid)[0]
    if len(idx): GROUPS['pair_%d+%d' % (A, B)] = Wd[(A, B)][idx]
for name in MT['sets']:
    if name == 'normal' or name.count('+') != 2: continue
    ev = tuple(int(x) for x in name.split('+')); Wd, valid = sim_windows('test_triples', MT, ev); idx = np.where(valid)[0]
    if len(idx): GROUPS['triple_' + name] = Wd[ev][idx]
qrng = np.random.default_rng(11); singles = {}
def sim_single(e):
    if e not in singles:
        Wd, valid = sim_windows('test_pairs', MP, (e,)); singles[e] = (Wd[(e,)], valid, Wd[()])
    return singles[e]
for _ in range(40):
    q = tuple(sorted(qrng.choice(EFF, 4, replace=False))); parts = [sim_single(e) for e in q]; valid = parts[0][1].copy()
    for p_ in parts: valid &= p_[1]
    idx = np.where(valid)[0]
    if len(idx) == 0: continue
    X0 = parts[0][2][idx]; x = X0.copy()
    for e in q: x = x + sim_single(e)[0][idx] - X0
    GROUPS['quad_' + '+'.join(map(str, q))] = x
log('data ready: train', len(XTR), 'ev', len(SETS['ev']), 'groups', len(GROUPS))

def run_net(name, net, X, Y, multilabel=False, EP=40):
    torch.manual_seed(0); net = net.to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    Xt = torch.as_tensor(std(X).transpose(0, 2, 1)); Yt = torch.as_tensor(Y)
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 512):
            b = perm[s:s + 512]; x = Xt[b].to(DEV); y = Yt[b].to(DEV); o = net(x)
            loss = nn.functional.binary_cross_entropy_with_logits(o, y) if multilabel else nn.functional.cross_entropy(o, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    log(name, 'trained')
    net.eval(); res = {}
    with torch.no_grad():
        for k, v in {**SETS, **GROUPS}.items():
            out = []
            for s in range(0, len(v), 4096):
                o = net(to_t(v[s:s + 4096])); out.append((torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).cpu().numpy())
            res[k] = np.concatenate(out)
    np.savez_compressed(OUT + name + '_probs_v2.npz', **res); log(name, 'probabilities saved')
    torch.save({'state': net.state_dict(), 'PM': PM, 'PSD': PSD}, OUT + name + '_model.pt'); log(name, 'model saved')


def to_t(X): return torch.as_tensor(std(X).transpose(0, 2, 1), device=DEV)


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


# multi-label CNN, single faults only
YML = np.zeros((len(YTR), 20), np.float32); YML[np.arange(len(YTR))[YTR > 0], YTR[YTR > 0] - 1] = 1
run_net('mlcnn_f0', CNN(20), XTR, YML, multilabel=True)
# multi-label CNN with the counterfactual composition of final_run.py (same sampling recipe)
rng = np.random.default_rng(0); ok = {c: np.where(~FR['tr'][c])[0] for c in range(C)}
XA = [XTR]; YA = [YML]
for (A, B) in itertools.combinations(range(1, C), 2):
    good = np.intersect1d(ok[A], ok[B]); idx = rng.choice(good, min(84, len(good)), replace=False)
    for first, second in ((A, B), (B, A)):
        XA.append(RAW['tr'][first][idx] + RAW['tr'][second][idx] - RAW['tr'][0][idx]); y = np.zeros((len(idx), 20), np.float32); y[:, [A - 1, B - 1]] = 1; YA.append(y)
for _ in range(300):
    tri = tuple(sorted(rng.choice(range(1, C), 3, replace=False)))
    good = np.intersect1d(np.intersect1d(ok[tri[0]], ok[tri[1]]), ok[tri[2]]); idx = rng.choice(good, min(30, len(good)), replace=False)
    XA.append(RAW['tr'][tri[0]][idx] + RAW['tr'][tri[1]][idx] + RAW['tr'][tri[2]][idx] - 2 * RAW['tr'][0][idx])
    y = np.zeros((len(idx), 20), np.float32); y[:, np.array(tri) - 1] = 1; YA.append(y)
XA = np.concatenate(XA).astype(np.float32); YA = np.concatenate(YA)
log('composition training windows', len(XA))
run_net('mlcnn_f100t', CNN(20), XA, YA, multilabel=True)
log('DONE')
