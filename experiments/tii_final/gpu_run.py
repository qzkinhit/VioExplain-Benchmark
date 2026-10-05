"""VioExplain v3 GPU baselines and temporal evidence on hit (same windows as final_run.py).

Trains (1) a 1D-CNN fault diagnoser on raw windows and (2) a frozen MantisV2 per-sensor embedding classifier, and saves
per-window class probabilities for every evaluation group so that cpu1 can score them with the shared metric code
and fuse the temporal evidence into the first selection of VioExplain.
Usage: python gpu_run.py cuda:N
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
    open(OUT + 'log.txt', 'a').write(s + '\n')


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


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(128, C)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))


def to_t(X): return torch.as_tensor(std(X).transpose(0, 2, 1), device=DEV)


torch.manual_seed(0); net = CNN().to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
Xt = torch.as_tensor(std(XTR).transpose(0, 2, 1)); Yt = torch.as_tensor(YTR)
EP = 40; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
for ep in range(EP):
    net.train(); perm = torch.randperm(len(Xt))
    for s in range(0, len(Xt), 512):
        b = perm[s:s + 512]; x = Xt[b].to(DEV); y = Yt[b].to(DEV)
        loss = nn.functional.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
log('cnn trained')


def cnn_probs(X):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), 4096): out.append(torch.softmax(net(to_t(X[s:s + 4096])), 1).cpu().numpy())
    return np.concatenate(out)


np.savez_compressed(OUT + 'cnn_probs.npz', **{k: cnn_probs(v) for k, v in {**SETS, **GROUPS}.items()})
log('cnn probabilities saved')
# frozen MantisV2 per-sensor embeddings -> per-sensor PCA(16) -> LightGBM
sys.path.insert(0, ROOT + 'envs')
from smoke_v3 import load_mantis
model = load_mantis(os.environ['V3_MANTIS'], DEV)


def embed(X):
    S = std(X).transpose(0, 2, 1).reshape(-1, W); out = []
    with torch.inference_mode():
        for s in range(0, len(S), 4096):
            t = torch.as_tensor(S[s:s + 4096, None, :], device=DEV)
            t = torch.nn.functional.interpolate(t, size=512, mode='linear', align_corners=False)
            out.append(model(t).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out).reshape(len(X), M, -1)


Etr = embed(XTR); log('mantis train embedded', Etr.shape)
from sklearn.decomposition import PCA
pcas = [PCA(16, random_state=0).fit(Etr[:, j, :].astype(np.float32)) for j in range(M)]


def red(E): return np.concatenate([pcas[j].transform(E[:, j, :].astype(np.float32)) for j in range(M)], 1)


import lightgbm as lgb
clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.5, n_jobs=16, verbose=-1).fit(red(Etr), YTR)
np.savez_compressed(OUT + 'mantis_probs.npz', **{k: clf.predict_proba(red(embed(v))) for k, v in {**SETS, **GROUPS}.items()})
log('mantis probabilities saved; DONE')
