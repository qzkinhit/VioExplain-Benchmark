"""Temporal diagnosers of hyd_deep.py (1D-CNN and ResNet, protocol f0) with saved weights: the final VioExplain needs
their posterior on composed calibration cycles (fusion-weight choice and first selection of the addition-decision
compositions, as in final_fuse5.py).

Usage:  HYD_MODE=f0 HYD_OUTSUB=final/ NJ=<threads> python hyd_tnet.py 1D-CNN|ResNet
The architecture, optimizer, schedule, batch order and initialization follow hyd_deep.py line by line (the network is
built before torch.manual_seed, as there). The script saves OUT/tnet_<name>.pt (weights, input centre and scale) and
OUT/tnet_<name>_check.json with the largest absolute difference to the probabilities that hyd_deep.py stored in the
default output directory (results/hyd_v1/f0<TAG>/probs_<name>.npz). Those files are not modified.
"""
import sys
from hyd_common import *
import torch, torch.nn as nn
torch.set_num_threads(NJ)
EP = int(os.environ.get('HYD_EP', '150')); BS = 32
PM_ = RAW['fit'][0].reshape(-1, M).mean(0)
PSD_ = np.concatenate([(XALL[ii] - XALL[ii].mean(0)).reshape(-1, M) for ii in FIT_GROUPS]).std(0) + 1e-8


def prep(X): return ((X - PM_) / PSD_).astype(np.float32).transpose(0, 2, 1).copy()


class CNN(nn.Module):
    def __init__(s, out=C):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(128, out)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))


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


def build(name): return CNN(C) if name == '1D-CNN' else ResNet()


def tnet_probs(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.as_tensor(prep(X))), 1).numpy()


if __name__ == '__main__':
    NAME = sys.argv[1]; net = build(NAME)
    torch.manual_seed(SEED); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    Xt = torch.as_tensor(prep(XBR)); Yt = torch.as_tensor(YB.astype(np.int64)); gen = torch.Generator().manual_seed(SEED)
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(Xt), generator=gen)
        for s in range(0, len(Xt), BS):
            b = perm[s:s + BS]
            if len(b) < 2: continue
            o = net(Xt[b]); loss = nn.functional.cross_entropy(o, Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    res = {'test': tnet_probs(net, GRAW['test']), 'tcal1': tnet_probs(net, GRAW['tcal1'])}
    torch.save({'state': net.state_dict(), 'PM': PM_, 'PSD': PSD_, 'name': NAME}, OUT + 'tnet_%s.pt' % NAME)
    np.savez_compressed(OUT + 'tnet_probs_%s.npz' % NAME, **res)
    old = ROOT + 'results/hyd_v1/f0%s%s/probs_%s.npz' % ('' if EVT == 'comp' else '_' + EVT, TAG, NAME)
    chk = {'threads': NJ}
    if os.path.exists(old):
        o = dict(np.load(old))
        for k in res:
            chk['max_abs_diff_' + k] = float(np.abs(res[k] - o[k]).max())
            chk['argmax_agree_' + k] = float((res[k].argmax(1) == o[k].argmax(1)).mean())
    log(NAME, 'trained', EP, 'epochs; weights saved; check against the default output directory', json.dumps(chk))
    json.dump(chk, open(OUT + 'tnet_%s_check.json' % NAME, 'w'), indent=1)
    log('DONE', NAME)
