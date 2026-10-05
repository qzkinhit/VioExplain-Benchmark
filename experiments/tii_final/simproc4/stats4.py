"""Effective-event statistics of one simproc4 data set on the windows of common.py (no method is run).
Usage: V3_DATA=dist V3_RES=simproc_v1/dist V3_MODE=f0 V3_FUNC=1 V3_LOG=log_stats.txt python stats4.py
Writes results/<V3_RES>/dataset_truth_stats.json: per group (single, pair, triple, quad) the number of windows and
the distribution of the size of the effective-event truth, and per fault the share of single-fault evaluation windows
in which the fault is effective (it causes a typed violation against the paired normal window)."""
import collections
from common import *
out = {'relations': NR, 'units': int(Z['tr'][0].shape[1])}
eff = collections.defaultdict(list)
for t, y in zip(EV_T, EV_Y):
    if y > 0: eff[int(y)].append(len(t) > 0)
out['single_eval_windows'] = len(EV_T); out['single_normal_eval_windows'] = int((EV_Y == 0).sum())
out['single_effective_share_per_fault'] = {str(k): float(np.mean(v)) for k, v in sorted(eff.items())}
for g, groups in (('pair', COMP), ('triple', TRIP), ('quad', QUAD)):
    sizes = collections.Counter(); full = 0; n = 0
    for ev, Zw, tru, V, causes in groups:
        for t in tru: sizes[len(t)] += 1; full += int(len(t) == len(ev)); n += 1
    out[g] = dict(sets=len(groups), windows=n, truth_size_counts={str(k): v for k, v in sorted(sizes.items())},
                  share_all_events_effective=full / max(n, 1), mean_truth_size=sum(k * v for k, v in sizes.items()) / max(n, 1))
json.dump(out, open(os.path.dirname(OUT.rstrip('/')) + '/dataset_truth_stats.json', 'w'), indent=1)
log(json.dumps(out)); log('DONE')
