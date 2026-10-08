"""Aggregate results/final_v1/update_base/h<H>_<METHOD>.json (final_update_base.py) and the VioExplain Update curve
(update_curve.json, variant refit_br, reference ref.json) into results/final_v1/update_baselines.json.

mean -> method -> {n, alone, with_known, mean, full}: averages over the held-out faults; a value at n
is reported only when every held-out fault has it (else null, and per_fault shows what exists).
"""
import json, glob, re, os
import numpy as np
D = '/path/to/vioexplain/results/final_v1/'
U = D + 'update_base/'
HS = [1, 2, 6, 8, 12, 13]
NS = [4, 8, 16, 32, 64, 128]
ORDER = ['BR-LGBM', 'CC-LGBM', 'LGBM', 'ResNet', 'ML-CNN', 'PCA-RBC', 'AEC', 'MinExplain', 'LSTM', 'InceptionTime']

per = {}
for f in sorted(glob.glob(U + 'h*.json')):
    m = re.match(r'.*/h(\d+)_(.+)\.json$', f)
    if not m or m.group(2).endswith('_smoke'): continue
    h, meth = int(m.group(1)), m.group(2)
    per.setdefault(meth, {})[h] = json.load(open(f))
# helper runs (V3_TAG=part2) computed other n values of the same method: merge, the main file wins on overlap
for meth in [m_ for m_ in per if m_.endswith('_part2')]:
    base = meth[:-len('_part2')]
    for h, r in per.pop(meth).items():
        if h not in per.setdefault(base, {}): per[base][h] = r
        else:
            for k, v in r['curve'].items(): per[base][h]['curve'].setdefault(k, v)


def avg(vals):
    return float(np.mean(vals)) if len(vals) == len(HS) else None


out = dict(description=(
    'Knowledge-update panel, mode f0, reduced setting of final_update.py (every 6th non-frozen fit window per class, 350 '
    'per class; LightGBM 150 trees, learning rate 0.1). VioExplain: Update curve of '
    'update_curve.json (variant refit_br), new event built from the first n flagged training windows of H without labels '
    'or paired runs; full = ref.json (H in the knowledge from the start, same reduced setting). Baselines: retrained on '
    'the known classes plus the first n non-frozen fit windows of H in run order WITH the true label (no flagging), '
    'replicated to 350 rows (the per-class count) unless the method name ends in _nobal; full = H in the knowledge like '
    'every other class (its 350 subsampled windows; paired runs for AEC and PCA-RBC). Calibration: conformal empty set '
    'on fault-free test-calibration windows (alpha 0.05, one window per run); thresholds and covering parameters by the '
    'uniform rule (mean of event-set F1 on single-fault test-calibration windows and composed calibration windows), '
    'restricted to known classes and compositions of known effective faults on the curve (as tau_c and DEC of '
    'VioExplain), all classes for full. alone = eval test windows of H; with_known = simulator test_pairs windows of '
    'pairs containing H; mean = (alone + with_known) / 2; values are event-set F1 averaged over windows, then over the '
    'held-out faults. Networks (ResNet, ML-CNN, LSTM, InceptionTime) use the training loop of gpu_save2.py on the subsampled '
    'raw windows, 40 epochs; suffix _ep240 = 240 epochs (about the optimizer steps of the full-data training), n 4, 16, '
    '128 and full only. Suffix _nobal = the n windows of H are not replicated (sensitivity).'),
    faults=HS, mean={}, per_fault={})

# VioExplain from update_curve.json (refit_br) and ref.json
uc = json.load(open(D + 'update_curve.json'))['refit_br']
pf = uc['per_fault']
vx = dict(n=NS, alone=[], with_known=[], mean=[])
for n in NS:
    a = [pf[str(h)]['curve'][str(n)]['alone_f1'] for h in HS if str(n) in pf[str(h)]['curve']]
    w = [pf[str(h)]['curve'][str(n)]['pair_f1'] for h in HS if str(n) in pf[str(h)]['curve']]
    vx['alone'].append(avg(a)); vx['with_known'].append(avg(w))
    vx['mean'].append(None if avg(a) is None else 0.5 * (avg(a) + avg(w)))
ra = avg([pf[str(h)]['reference']['alone_f1'] for h in HS]); rw = avg([pf[str(h)]['reference']['pair_f1'] for h in HS])
vx['full'] = dict(alone=ra, with_known=rw, mean=0.5 * (ra + rw), source='ref.json (same reduced setting)')
vx['n0'] = dict(alone=avg([pf[str(h)]['curve']['0']['alone_f1'] for h in HS]), with_known=avg([pf[str(h)]['curve']['0']['pair_f1'] for h in HS]))
out['mean']['VioExplain'] = vx
out['per_fault']['VioExplain'] = {str(h): {str(n): dict(alone=pf[str(h)]['curve'][str(n)]['alone_f1'], with_known=pf[str(h)]['curve'][str(n)]['pair_f1'])
                                           for n in NS if str(n) in pf[str(h)]['curve']} for h in HS}
vx_neval = {str(h): pf[str(h)]['n_eval'] for h in HS}

checks = {}
for meth in ORDER + sorted(m_ for m_ in per if m_ not in ORDER):
    if meth not in per: continue
    R = per[meth]; e = dict(n=NS, alone=[], with_known=[], mean=[])
    for n in NS:
        rows = [R[h]['curve'][str(n)] for h in HS if h in R and str(n) in R[h]['curve']]
        a = avg([r['alone'] for r in rows]); w = avg([r['with_known'] for r in rows])
        e['alone'].append(a); e['with_known'].append(w); e['mean'].append(None if a is None else 0.5 * (a + w))
    rows = [R[h]['curve']['full'] for h in HS if h in R and 'full' in R[h]['curve']]
    a = avg([r['alone'] for r in rows]); w = avg([r['with_known'] for r in rows])
    e['full'] = dict(alone=a, with_known=w, mean=None if a is None else 0.5 * (a + w))
    e['faults_done'] = {str(h): sorted(R[h]['curve'].keys()) for h in R}
    out['mean'][meth] = e
    out['per_fault'][meth] = {str(h): R[h]['curve'] for h in sorted(R)}
    # same evaluation windows as the VioExplain Update runs
    for h in R:
        ne = R[h]['n_eval']; ve = vx_neval[str(h)]
        checks.setdefault(meth, {})[str(h)] = dict(alone=ne['alone'] == ve['alone'], pair=ne['pair'] == ve['pair'],
                                                   pair_groups=ne['pair_groups'] == ve['pair_groups'],
                                                   truth_hash=ne['hash_alone_truth'] + '/' + ne['hash_pair_truth'], host=R[h].get('host'))
for meth, e in out['mean'].items():
    # full uses the 350 subsampled windows of H with no replication, so it is the same run with or without balancing
    if meth.endswith('_nobal') and e['full']['alone'] is None and meth[:-6] in out['mean']:
        e['full'] = dict(out['mean'][meth[:-6]]['full']); e['full']['source'] = 'identical to ' + meth[:-6] + ' (no replication at full)'
out['eval_window_check'] = checks
json.dump(out, open(D + 'update_baselines.json', 'w'), indent=1)

fmt = lambda v: '  -  ' if v is None else '%.3f' % v
print('%-12s %-10s' % ('method', 'metric') + ''.join('%7s' % n for n in NS) + '   full')
for meth, e in out['mean'].items():
    for k in ('alone', 'with_known', 'mean'):
        print('%-12s %-10s' % (meth, k) + ''.join('%7s' % fmt(v) for v in e[k]) + '  ' + fmt(e['full'][k]))
bad = {m_: {h: c for h, c in v.items() if not (c['alone'] and c['pair'] and c['pair_groups'])} for m_, v in checks.items()}
print('eval window mismatches:', {m_: v for m_, v in bad.items() if v})
hashes = {}
for m_, v in checks.items():
    for h, c in v.items(): hashes.setdefault(h, set()).add(c['truth_hash'])
print('truth hashes per fault (one value = identical windows on both hosts):', {h: len(s) for h, s in hashes.items()})
