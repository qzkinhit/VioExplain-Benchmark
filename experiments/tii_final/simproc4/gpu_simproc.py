"""Deep diagnosers, multi-label CNNs and MantisV2 for a simulated process (hit, GPU), on the windows of common.py.

Usage:  V3_DATA=cstr V3_MODE=f100t V3_RES=simproc_v1/cstr python gpu_simproc.py cuda:N NAME [NAME ...]
NAME in cnn resnet lstm inceptiontime mlcnn_f0 mlcnn_f100t mantis
Architectures and training as in gpu_run.py, gpu_run2.py and gpu_save2.py of the TEP runs (40 epochs, batch 512, AdamW
1e-3 with weight decay 1e-4, cosine schedule, seed 0, per-sensor standardization by the normal fit windows). MantisV2
as in gpu_run.py (frozen per-sensor embedding, per-sensor PCA(16), LightGBM head). The single-label networks and
ML-CNN-f0 learn from normal and single-fault fit windows; ML-CNN-f100t also learns from the compositions that BR-LGBM
and CC-LGBM receive in f100t (common.AUG, same draws).
Writes results/<V3_RES>/gpu/<NAME>_probs_v2.npz (keys ev, tcal and every concurrent group of common.GRAW) and, for cnn,
resnet and the multi-label CNNs, <NAME>_model.pt (weights and normalization) for CPU inference.
"""
import sys
DEV = sys.argv[1]; NAMES = sys.argv[2:]
from common import *
import torch, torch.nn as nn
assert PCT == 100, 'run with V3_MODE=f100t so that the compositions of BR-LGBM are available'
GOUT = os.path.dirname(OUT.rstrip('/')) + '/gpu/'; os.makedirs(GOUT, exist_ok=True)
XTR = np.concatenate([RAW['tr'][c][ok_tr[c]] for c in range(C)]); YTR = YB.astype(np.int64)
nm = RAW['tr'][0].reshape(-1, M); PM = nm.mean(0); PSD = nm.std(0) + 1e-8


def std(X): return ((X - PM) / PSD).astype(np.float32)


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


def run_net(name, net, X, Y, multilabel=False, EP=40, save=False):
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
    log(name, 'trained on', len(X), 'windows')
    net.eval(); res = {}
    with torch.no_grad():
        for k, v in GRAW.items():
            out = []
            for s in range(0, len(v), 4096):
                o = net(to_t(v[s:s + 4096])); out.append((torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).cpu().numpy())
            res[k] = np.concatenate(out)
    if not multilabel:
        log(name, 'label macro F1 on evaluation windows', float(f1_score(EV_Y, res['ev'].argmax(1), average='macro')))
    np.savez_compressed(GOUT + name + '_probs_v2.npz', **res); log(name, 'probabilities saved')
    if save:
        torch.save({'state': net.cpu().state_dict(), 'PM': PM, 'PSD': PSD}, GOUT + name + '_model.pt'); log(name, 'model saved')


YML = np.zeros((len(YTR), K), np.float32); YML[np.arange(len(YTR))[YTR > 0], YTR[YTR > 0] - 1] = 1
for name in NAMES:
    if name == 'cnn': run_net('cnn', CNN(), XTR, YTR, save=True)
    elif name == 'resnet': run_net('resnet', ResNet(), XTR, YTR, save=True)
    elif name == 'lstm': run_net('lstm', LSTM(), XTR, YTR)
    elif name == 'inceptiontime': run_net('inceptiontime', InceptionTime(), XTR, YTR)
    elif name == 'mlcnn_f0': run_net('mlcnn_f0', CNN(K), XTR, YML, multilabel=True, save=True)
    elif name == 'mlcnn_f100t':
        XA = [XTR]; YA = [YML]
        for ev, Zs, Zrest, idx in AUG:
            if Zrest is not None: x = RAW['tr'][ev[0]][idx] + RAW['tr'][ev[1]][idx] - RAW['tr'][0][idx]
            else: x = sum(RAW['tr'][e][idx] for e in ev) - (len(ev) - 1) * RAW['tr'][0][idx]
            assert np.allclose(Fn(x), Zs, atol=1e-3)
            y = np.zeros((len(idx), K), np.float32); y[:, np.array(ev) - 1] = 1; XA.append(x); YA.append(y)
        XA = np.concatenate(XA).astype(np.float32); YA = np.concatenate(YA); log('composition training windows', len(XA))
        run_net('mlcnn_f100t', CNN(K), XA, YA, multilabel=True, save=True)
    elif name == 'mantis':
        sys.path.insert(0, '/path/to/vioexplain/envs')
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
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, n_jobs=NJ, verbose=-1).fit(red(Etr), YTR)
        res = {k: clf.predict_proba(red(embed(v))) for k, v in GRAW.items()}
        log('mantis label macro F1 on evaluation windows', float(f1_score(EV_Y, res['ev'].argmax(1), average='macro')))
        np.savez_compressed(GOUT + 'mantis_probs_v2.npz', **res); log('mantis probabilities saved')
log('DONE', NAMES)
