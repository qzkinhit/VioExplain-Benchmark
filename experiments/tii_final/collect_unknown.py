"""Collect TEP-U per-fault results (results/final_v3/unknown/h<h>.json) into results/final_v3/unknown_final.json
(per held-out fault and the mean over the faults present) and print a compact table."""
import json, os, glob
import numpy as np
U = '/path/to/vioexplain/results/final_v3/unknown/'
HELD = [1, 2, 6, 7, 8, 12, 13, 14, 17, 18]
per = {}
for h in HELD:
    p = U + 'h%d.json' % h
    if os.path.exists(p): per[h] = json.load(open(p))
METH = ['Ours', 'MDS', 'MSP', 'Energy']
KEYS = ['auroc_all', 'auroc_single', 'recall_U2', 'recall_U1', 'falseflag_K2', 'falseflag_K1',
        'misassign', 'misassign_only', 'misassign_gated', 'misassign_only_gated']
mean = {m: {k: float(np.nanmean([r['methods'][m][k] for r in per.values()])) for k in KEYS} for m in METH} if per else {}
out = dict(protocol='TEP-U, final pipeline (final_fuse5 full, V3_FUNC=1, f100t) with h held out of every component',
           held_done=sorted(per), held_missing=[h for h in HELD if h not in per],
           mean=mean, per_held={str(h): r for h, r in per.items()})
json.dump(out, open('/path/to/vioexplain/results/final_v3/unknown_final.json', 'w'), indent=1)
print('done', sorted(per), 'missing', out['held_missing'])
print('%-8s' % 'method' + ''.join('%14s' % k[:14] for k in KEYS))
for m in METH:
    if mean: print('%-8s' % m + ''.join('%14.3f' % mean[m][k] for k in KEYS))
for h, r in per.items():
    print('h=%d n=%s A=%.2f sanity=%s' % (h, r['n'], r['fusion_weight'], {k: round(v, 3) for k, v in r['sanity'].items()}))
    for m in METH: print('   %-7s' % m + ''.join('%8.3f' % r['methods'][m][k] for k in KEYS))
