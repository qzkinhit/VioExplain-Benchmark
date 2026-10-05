"""Re-score the multi-label CNN without training it again.

Usage:  HYD_MODE=f0|f100t HYD_OUTSUB=<out>/ python hyd_rescore.py ML-CNN [WEIGHTS_SUBDIR]
With WEIGHTS_SUBDIR (e.g. final_onerule/) the network saved by hyd_deep.py in
results/hyd_v1/<WEIGHTS_SUBDIR><MODE><TAG>/model_<name>.pt scores the test-calibration, composed calibration and test
cycles, and the threshold follows the one calibration rule with the composed set given by HYD_CALSET.
Without it, the probabilities stored in the default output directory (results/hyd_v1/<MODE><TAG>/probs_<name>.npz, keys
test and tcal1) are re-scored with the single-fault rule (test-calibration cycles only, grid HYD_GRID).
"""
import sys
from hyd_common import *
name = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else None
tail = '%s%s%s/' % (MODE, '' if EVT == 'comp' else '_' + EVT, TAG)
if sub is None:
    src = ROOT + 'results/hyd_v1/' + tail + 'probs_%s.npz' % name
    PR = dict(np.load(src)); ta, _ = tune_thr(PR['tcal1'], TC_T, name); best = (ta, -1.0); log(name, 'threshold', best, 'probabilities from', src)
else:
    import torch
    from hyd_tnet import CNN
    src = ROOT + 'results/hyd_v1/' + sub + tail + 'model_%s.pt' % name
    ck = torch.load(src, map_location='cpu', weights_only=False); net = CNN(K); net.load_state_dict(ck['state']); net.eval()

    def pr(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.as_tensor(((X - ck['PM']) / ck['PSD']).astype(np.float32).transpose(0, 2, 1).copy()))).numpy()
    PR = {'test': pr(GRAW['test']), 'tcal1': pr(GRAW['tcal1']), 'calcomp': pr(calc()['X'])}
    best, _ = tune_ml_probs(name, PR['tcal1'], PR['calcomp']); log(name, 'threshold and gate (one calibration rule)', best, 'weights from', src)
evaluate(name, lambda Zw: ml_sets(PR[KEYOF[id(Zw)]], best[0], best[1]))
log('DONE', sys.argv[1:])
