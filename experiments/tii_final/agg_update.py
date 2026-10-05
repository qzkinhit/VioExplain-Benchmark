"""Aggregate results/final_v1/update/*.json into results/final_v1/update_curve.json and print compact tables.
Variants
  fixed_br  : protocol as specified, known BR detectors kept from the 19-fault knowledge (files h<H>.json)
  refit_br  : known BR detectors refit at every n with the buffer windows (negatives) and compositions (files h<H>_brrefit*.json)
Diagnostics: fault 7 with the flag ignored (h7_noflag*.json), fault 6 with the support-masked operator (h6_opmask.json)."""
import json, glob, os, re
import numpy as np
D = '/path/to/vioexplain/results/final_v1/'
U = D + 'update/'
NS = [0, 4, 8, 16, 32, 64, 128]
KEYS = ('alone_f1', 'pair_f1', 'alone_named', 'pair_named', 'known_single_f1', 'known_pair_f1')
EXTRA = ('supp_size', 'twin_supp_size', 'supp_jaccard_twin', 'foot_cos_twin', 'scanned_train_windows', 'tau0', 'tau_c')
REFJ = json.load(open(U + 'ref.json'))


def load_merged(paths):
    r = None
    for p in paths:
        x = json.load(open(p))
        if r is None: r = x
        else: r['curve'].update(x.get('curve', {}))
    return r


def fault_entry(h, r):
    ref = REFJ['reference'][str(h)]; rff = r['reference_full_f0']
    pf = dict(reference={k: ref[k] for k in KEYS}, reference_full_f0={k: rff[k] for k in KEYS}, unknown=r['unknown'],
              n_eval=r['n_eval'], curve={})
    for n in NS:
        c = r['curve'].get(str(n))
        if c is None: continue
        e = {k: c[k] for k in KEYS}
        e['ratio_alone'] = c['alone_f1'] / ref['alone_f1']; e['ratio_pair'] = c['pair_f1'] / ref['pair_f1']
        e['ratio_mean'] = (e['ratio_alone'] + e['ratio_pair']) / 2
        e['ratio_alone_fullref'] = c['alone_f1'] / rff['alone_f1']; e['ratio_pair_fullref'] = c['pair_f1'] / rff['pair_f1']
        for k in EXTRA:
            if k in c: e[k] = c[k]
        pf['curve'][str(n)] = e
    for th in (0.8, 0.9):
        for rk in ('ratio_alone', 'ratio_pair', 'ratio_mean'):
            hit = [n for n in NS if str(n) in pf['curve'] and pf['curve'][str(n)][rk] >= th]
            pf['n_to_%d_%s' % (int(th * 100), rk)] = hit[0] if hit else None
    return pf


def build(files):
    out = dict(per_fault={}, curve={})
    for h, paths in sorted(files.items()):
        r = load_merged(paths)
        if r is None or str(h) not in REFJ['reference']: continue
        out['per_fault'][str(h)] = fault_entry(h, r)
    pfs = out['per_fault']
    for n in NS:
        hs = [h for h in pfs if str(n) in pfs[h]['curve']]
        if not hs: continue
        rows = [pfs[h]['curve'][str(n)] for h in hs]
        e = {k: float(np.mean([x[k] for x in rows])) for k in KEYS + ('ratio_alone', 'ratio_pair', 'ratio_mean', 'ratio_alone_fullref', 'ratio_pair_fullref')}
        e['faults'] = sorted(int(h) for h in hs); out['curve'][str(n)] = e
    upd = [h for h in pfs if any(str(n) in pfs[h]['curve'] for n in NS if n > 0)]
    out['faults_with_update'] = sorted(int(h) for h in upd)
    out['reference_mean'] = {k: float(np.mean([pfs[h]['reference'][k] for h in upd])) for k in KEYS} if upd else {}
    out['reference_full_f0_mean'] = {k: float(np.mean([pfs[h]['reference_full_f0'][k] for h in upd])) for k in KEYS} if upd else {}
    for th in (0.8, 0.9):
        for rk in ('ratio_alone', 'ratio_pair', 'ratio_mean'):
            hit = [n for n in NS if n > 0 and str(n) in out['curve'] and set(out['curve'][str(n)]['faults']) >= set(int(h) for h in upd)
                   and out['curve'][str(n)][rk] >= th]
            out['n_to_%d_%s' % (int(th * 100), rk)] = hit[0] if hit else None
    return out


def show(name, out):
    print('==', name, 'faults with update', out['faults_with_update'])
    print('%5s %7s %7s %7s %7s %7s %7s %7s  %s' % ('n', 'alone', 'pair', 'r_alone', 'r_pair', 'r_mean', 'kn_sgl', 'kn_pair', 'faults'))
    for n in NS:
        if str(n) not in out['curve']: continue
        e = out['curve'][str(n)]
        print('%5d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f  %s' % (n, e['alone_f1'], e['pair_f1'], e['ratio_alone'], e['ratio_pair'],
                                                                e['ratio_mean'], e['known_single_f1'], e['known_pair_f1'], e['faults']))
    for lab, rm in (('ref', out['reference_mean']), ('full', out['reference_full_f0_mean'])):
        if rm: print('%5s %7.3f %7.3f %23s %7.3f %7.3f' % (lab, rm['alone_f1'], rm['pair_f1'], '', rm['known_single_f1'], rm['known_pair_f1']))
    print({k: v for k, v in out.items() if k.startswith('n_to')})
    for h, pf in out['per_fault'].items():
        print('H', h, 'ref alone %.3f pair %.3f | full-ref alone %.3f pair %.3f' % (
            pf['reference']['alone_f1'], pf['reference']['pair_f1'], pf['reference_full_f0']['alone_f1'], pf['reference_full_f0']['pair_f1']),
            '| flagged', pf['unknown']['n_flagged'], 'of', pf['unknown']['n_train_windows'])
        for n, e in pf['curve'].items():
            print('   n %4s alone %.3f pair %.3f r_alone %.3f r_pair %.3f named_a %.3f named_p %.3f kn %.3f/%.3f supp %s/%s jac %s scanned %s' % (
                n, e['alone_f1'], e['pair_f1'], e['ratio_alone'], e['ratio_pair'], e['alone_named'], e['pair_named'],
                e['known_single_f1'], e['known_pair_f1'], e.get('supp_size', '-'), e.get('twin_supp_size', '-'),
                ('%.2f' % e['supp_jaccard_twin']) if 'supp_jaccard_twin' in e else '-', e.get('scanned_train_windows', '-')))


fixed = {int(m.group(1)): [f] for f in glob.glob(U + 'h*.json') for m in [re.match(r'.*/h(\d+)\.json$', f)] if m}
refit = {}
for f in sorted(glob.glob(U + 'h*_brrefit*.json')):
    m = re.match(r'.*/h(\d+)_brrefit(_b)?\.json$', f)
    if m: refit.setdefault(int(m.group(1)), []).append(f)
diag = {}
for name, pat in (('fault7_noflag_fixed_br', 'h7_noflag.json'), ('fault7_noflag_refit_br', 'h7_noflag_brrefit.json'),
                  ('fault6_opmask_fixed_br', 'h6_opmask.json')):
    if os.path.exists(U + pat): diag[name] = build({int(pat[1]): [U + pat]})
res = dict(description='Update learning curve, mode f0. Held-out fault H is removed from the knowledge, the new event is built from '
                       'the first n flagged fit-partition training windows of H without paired runs. reference = same pipeline and '
                       'compute setting with H in the knowledge from the start (paired runs, ref.json). reference_full_f0 = full-'
                       'setting f0 VioExplain records of final_run.py on the same windows. alone = eval test windows of H (onset 160, '
                       'frozen removed, effective-event truth). pair = simulator test pairs with H and one known fault. known_* = '
                       'fixed subsample of known windows (every 40th single eval window, every 20th window of pairs without H).',
           setting={k: REFJ[k] for k in ('sub', 'iters', 'lr', 'br_iters', 'ncomp_per_known', 'scorer_rows_per_class')},
           fixed_br=build(fixed), refit_br=build(refit), diagnostics=diag)
json.dump(res, open(D + 'update_curve.json', 'w'), indent=1)
show('fixed_br', res['fixed_br']); show('refit_br', res['refit_br'])
for k, v in diag.items(): show(k, v)
