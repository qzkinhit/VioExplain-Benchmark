"""Score probability files produced on the GPU server with the shared protocol of common.py.

Usage:  V3_MODE=f0 V3_LOG=log_probs.txt V3_METRICS=metrics_probs.json python final_probs.py NAME:path.npz:kind ...
kind = single  softmax over 21 classes (normal + 20 faults): top-1 with a conformal empty set (alpha 0.05, one normal
               test-calibration window per run)
kind = argmax  hard predictions (one-hot probabilities of ridge-based classifiers): empty set iff the predicted class
               is normal
kind = multi   sigmoid per fault (20 columns): threshold tuned on test-calibration single-fault set F1, as for BR-LGBM
"""
import sys
from common import *

for arg in sys.argv[1:]:
    name, path, kind = arg.split(':')
    D = np.load(path); PR = {k: D[k] for k in D.files}
    assert len(PR['ev']) == len(EV_Z), (len(PR['ev']), len(EV_Z))
    P_key = lambda Zw, PR=PR: PR[KEYOF[id(Zw)]]
    if kind == 'argmax':
        RES[name + '_label_macroF1'] = float(f1_score(EV_Y, PR['ev'].argmax(1), average='macro'))
        evaluate(name, lambda Zw, P_key=P_key: [[] if np.argmax(p) == 0 else [int(np.argmax(p))] for p in P_key(Zw)])
    elif kind == 'single':
        n0 = len(Z['tcal'][0]); per = n0 // len(TCAL); pick = np.arange(len(TCAL)) * per + rng.integers(per, size=len(TCAL))
        s = np.sort(1 - PR['tcal'][:n0][pick][:, 0]); q0 = 1 - s[min(len(s) - 1, int(np.ceil(0.95 * (len(s) + 1))) - 1)]
        RES[name + '_label_macroF1'] = float(f1_score(EV_Y, PR['ev'].argmax(1), average='macro'))
        evaluate(name, lambda Zw, P_key=P_key, q0=q0: [[] if p[0] >= q0 else [int(np.argmax(p[1:])) + 1] for p in P_key(Zw)])
    else:
        # multi-label CNN: the threshold is chosen on single-fault test-calibration windows and on composed calibration
        # windows (scored by the saved network given as V3_MLMODEL), the calibration windows shared by all methods
        pc = tc = None
        if os.environ.get('V3_MLMODEL'):
            import torch, torch.nn as nn
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
            ck = torch.load(os.environ['V3_MLMODEL'], map_location='cpu', weights_only=False)
            net = nn.Module(); net.f = nn.Sequential(nn.Conv1d(M, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=2),
                                                     nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                                                     nn.AdaptiveAvgPool1d(1)); net.h = nn.Linear(128, K)
            net.load_state_dict(ck['state']); net.eval(); Xc, _, tc = cal_compositions(all_pairs=(PCT == 0))
            with torch.no_grad():
                x = torch.as_tensor(((Xc - ck['PM']) / ck['PSD']).astype(np.float32).transpose(0, 2, 1)); pc = torch.sigmoid(net.h(net.f(x).squeeze(-1))).numpy()
        q0 = ml_gate(PR['tcal']); log(name, 'empty-set gate', q0)
        ta = tune_threshold(PR['tcal'], pc, tc, [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995], name, q0)
        evaluate(name, lambda Zw, P_key=P_key, ta=ta, q0=q0: ml_sets(P_key(Zw), ta, q0))
log('DONE')
