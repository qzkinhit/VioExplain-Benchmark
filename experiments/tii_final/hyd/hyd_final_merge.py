"""Final HYD tables: mean and sd over the three hash splits -> results/hyd_v1/metrics_<protocol>_final_3splits.json
(same structure as metrics_<protocol>_3splits.json) and metrics_final_3splits_meta.json.

Per-split metrics come from
  results/hyd_v1/final_gate/<protocol><split>/     multi-label methods (BR-LGBM, CC-LGBM, RF-ML, ML-CNN) under the
                                                   uniform calibration rule with the split-conformal empty-set gate,
  results/hyd_v1/final_uniform2/<protocol><split>/ PCA-RBC, AEC, MinExplain under the uniform calibration rule
                                                   (HYD_CAL=one, composed calibration cycles of VioExplain),
  results/hyd_v1/final/<protocol><split>/          final VioExplain, softmax Rocket scores, MantisV2,
  results/hyd_v1/<protocol><split>/                every other row (default output directory of hyd_run.py and
                                                   hyd_deep.py). Its VioExplain and VioExplain-T entries are not copied.
VioExplain = the variant (with or without functional units) with the higher mean calibration criterion of
hyd_final.py over the three splits; an exact tie keeps the variant without functional units. No test cycle enters
this choice. The other variant is stored as VioExplain-func or VioExplain-nofunc.
"""
import json, glob, os, numpy as np
R = '/path/to/vioexplain/results/hyd_v1/'
KEYS = ['k1_setF1', 'k2_setF1', 'k3_setF1', 'k4_setF1', 'multi_setF1', 'single_normal_macroF1_gated', 'normal_named']
ORDER = ['VioExplain', 'VioExplain-func', 'VioExplain-nofunc', 'PCA-RBC', 'FDA', 'RF-top1', 'XGB-top1', 'MC-LGBM-top1', 'MiniRocket',
         'MultiRocket', 'QUANT', 'MantisV2', '1D-CNN', 'LSTM', 'ResNet', 'InceptionTime', 'BR-LGBM', 'CC-LGBM', 'ML-CNN', 'RF-ML', 'AEC', 'MinExplain']
SUF = ('', '_splitB', '_splitC')
TUNED = ['BR-LGBM', 'CC-LGBM', 'RF-ML', 'ML-CNN', 'PCA-RBC', 'AEC', 'MinExplain']
SUBDIR = {'f0': TUNED + ['MiniRocket', 'MultiRocket', 'MantisV2'], 'f100t': ['BR-LGBM', 'CC-LGBM', 'RF-ML', 'ML-CNN']}
DIR_OF = {m: 'final_gate/' if m in ('BR-LGBM', 'CC-LGBM', 'RF-ML', 'ML-CNN') else 'final_uniform2/' for m in TUNED}; DIR_OF.update({m: 'final/' for m in ('MiniRocket', 'MultiRocket', 'MantisV2')})


def agg(v):
    out = {'splits': sorted(v)}
    for k in KEYS:
        a = np.array([v[s][k] for s in v if k in v[s]], float)
        out[k] = [float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, int(len(a))]
    return out


def fmt(o): return ' '.join('%6.3f+-%.3f' % (o[k][0], o[k][1]) for k in KEYS)


agg_now = {}
HEAD = '%-22s ' % 'method' + ' '.join('%12s' % k.replace('_setF1', '').replace('single_normal_macroF1_gated', 'mF1').replace('normal_named', 'normNamed') for k in KEYS)
meta = {'keys': KEYS, 'source': {}, 'tuning': {}, 'differs_from_default_directory': {}, 'calibration_rule': (
    'uniform rule: every parameter that decides how many events a baseline names is chosen by maximizing 0.5 * set F1 on the '
    '116 normal and single-fault test-calibration cycles + 0.5 * set F1 on the 2700 composed calibration cycles (one, two or '
    'three events) on which the addition decision of VioExplain is fitted; in f0 and f100t; the first grid point with the best '
    'objective wins; a grid is extended while a grid point with the best objective lies at one of its ends'), 'single_fault_rule': {}, 'diag_tcalcomp': {},
    'mlcnn_singlecal': {}, 'uniform_rule_extension_stops_on_ties': {}, 'uniform_rule_without_gate': {},
    'multilabel_gate': 'a multi-label method names nothing on a cycle whose largest event probability does not exceed the '
    'ceil(0.95 (n + 1))-th smallest largest event probability of the n = 4 normal test-calibration cycles (their maximum); '
    'the threshold search applies the same gate'}
# ---- choice between the variants with and without functional units, calibration data only ----
cal = {v: [json.load(open(R + 'final/f100t%s/calib_VioExplain-%s.json' % (s, v))) for s in SUF] for v in ('nofunc', 'func')}
crit = {v: float(np.mean([c['criterion'] for c in cal[v]])) for v in cal}
REC = 'func' if crit['func'] > crit['nofunc'] else 'nofunc'; OTHER = 'nofunc' if REC == 'func' else 'func'
thr = {v: [json.load(open(R + 'final/f100t%s/thresholds_VioExplain-%s.json' % (s, v))) for s in SUF] for v in cal}
meta['variant_choice'] = {
    'rule': 'higher mean over the three splits of 0.5 * (set F1 on normal and single-fault test-calibration cycles + set F1 on 1500 '
            'two- and three-event compositions of test-calibration cycles); exact tie keeps nofunc; no test cycle is used',
    'mean_criterion': crit, 'per_split_criterion': {v: [c['criterion'] for c in cal[v]] for v in cal}, 'recommended': REC,
    'per_split_calibration': cal, 'relations_kept_per_split': [t['relations'] for t in thr['func']],
    'fusion_per_split': {v: [(t['fusion_net'], t['fusion_a']) for t in thr[v]] for v in thr},
    'tau0_per_split': {v: [t['tau0'] for t in thr[v]] for v in thr}, 'tau1_per_split': {v: [t['tau1'] for t in thr[v]] for v in thr},
    'tau1_n_per_split': {v: [t['tau1_n'] for t in thr[v]] for v in thr}}
print('variant choice by calibration criterion:', crit, '-> VioExplain =', REC, '| per split', meta['variant_choice']['per_split_criterion'])
for mode in ('f0', 'f100t'):
    per = {}
    for suf in SUF:
        for f in glob.glob(R + mode + suf + '/metrics_*.json'):
            nm = os.path.basename(f)[8:-5]
            if nm.startswith('VioExplain'): continue
            per.setdefault(nm, {})[suf or 'main'] = json.load(open(f))
    first = {m: agg(v) for m, v in per.items()}
    src = {m: 'default directory (results/hyd_v1/%s*/)' % mode for m in per}
    for m in SUBDIR[mode]:
        new = {}
        for suf in SUF:
            f = R + DIR_OF[m] + '%s%s/metrics_%s.json' % (mode, suf, m)
            if os.path.exists(f): new[suf or 'main'] = json.load(open(f))
        if len(new) == 3:
            per[m] = new; src[m] = 'method directory (results/hyd_v1/%s%s*/)' % (DIR_OF[m], mode)
        else:
            print('WARNING', mode, m, 'method directory incomplete', sorted(new), '- default directory kept' if m in per else '- absent')
        if m in TUNED:
            meta['tuning'][mode + '/' + m] = []
            for suf in SUF:
                f = R + DIR_OF[m] + '%s%s/tuning_%s.json' % (mode, suf, m)
                if os.path.exists(f):
                    t = json.load(open(f)); meta['tuning'][mode + '/' + m].append({k: t[k] for k in t if k not in ('table', 'best_points')} | {'split': suf or 'main'})
            for key, sub in (('single_fault_rule', 'final/'), ('diag_tcalcomp', 'final_uniform2/diag_tcalcomp_'),
                             ('uniform_rule_extension_stops_on_ties', 'final_uniform/'), ('uniform_rule_without_gate', 'final_uniform2/')):
                fs = [R + sub + '%s%s/metrics_%s.json' % (mode, suf, m) for suf in SUF]
                if all(os.path.exists(f) for f in fs):
                    meta[key][mode + '/' + m] = agg({suf or 'main': json.load(open(f)) for suf, f in zip(SUF, fs)})
                    if key != 'single_fault_rule':
                        meta[key][mode + '/' + m]['chosen'] = [json.load(open(R + sub + '%s%s/tuning_%s.json' % (mode, suf, m)))['chosen'] for suf in SUF]
            if m == 'ML-CNN':
                fs = [R + 'final_onerule/%s%s/metrics_ML-CNN-singlecal.json' % (mode, suf) for suf in SUF]
                if all(os.path.exists(f) for f in fs):
                    meta['mlcnn_singlecal'][mode] = agg({suf or 'main': json.load(open(f)) for suf, f in zip(SUF, fs)})
    if mode == 'f100t':
        for key, v in (('VioExplain', REC), ('VioExplain-' + OTHER, OTHER)):
            per[key] = {suf or 'main': json.load(open(R + 'final/f100t%s/metrics_VioExplain-%s.json' % (suf, v))) for suf in SUF}
            src[key] = 'final method, variant %s (results/hyd_v1/final/f100t*/metrics_VioExplain-%s.json)' % (v, v)
    out = {}
    print('\n== %s final (mean +- sd over splits)' % mode); print(HEAD)
    for m in ORDER + sorted(set(per) - set(ORDER)):
        if m not in per: continue
        out[m] = agg(per[m]); print('%-22s ' % m + fmt(out[m]), len(per[m]), '' if src[m].startswith('default') else '*')
        if m in first and not src[m].startswith('default'):
            d = {k: out[m][k][0] - first[m][k][0] for k in KEYS}
            if any(abs(x) > 1e-12 for x in d.values()): meta['differs_from_default_directory'][mode + '/' + m] = {'default_directory': {k: first[m][k][:2] for k in KEYS}, 'final': {k: out[m][k][:2] for k in KEYS}}
            print('%-22s ' % ('   default directory') + fmt(first[m]), '  changed' if mode + '/' + m in meta['differs_from_default_directory'] else '  identical')
    meta['source'][mode] = src
    json.dump(out, open(R + 'metrics_%s_final_3splits.json' % mode, 'w'), indent=1)
    for m in TUNED:
        if m in out: agg_now[mode + '/' + m] = out[m]['multi_setF1'][0]
# ---- intermediate configurations of hyd_final.py (our method, f100t) ----
lad = {}
old = {suf or 'main': json.load(open(R + 'f100t%s/metrics_VioExplain.json' % suf)) for suf in SUF}
print('\n== VioExplain f100t: ours block of hyd_run.py, intermediate configurations, final'); print(HEAD)
print('%-22s ' % 'hyd_run.py ours' + fmt(agg(old))); lad['hyd_run.py ours'] = agg(old)
for ft in ('nofunc', 'func'):
    for n in ('L0-base-', 'L1-nostep-', 'L2-comp3-', 'L3-conformal-', 'VioExplain-'):
        fs = [R + 'final/f100t%s/metrics_%s%s.json' % (suf, n, ft) for suf in SUF]
        if all(os.path.exists(f) for f in fs):
            lad[n + ft] = agg({suf or 'main': json.load(open(f)) for suf, f in zip(SUF, fs)}); print('%-22s ' % (n + ft) + fmt(lad[n + ft]))
meta['ladder_f100t'] = lad
print('\ntuning records (chosen value per split, interior, ties, natural bound):')
for k, v in meta['tuning'].items():
    print(' ', k, [(t['split'], t['chosen'], t['interior'], t.get('n_grid_points_with_best_objective', t.get('n_grid_points_with_best_score')),
                    t.get('natural_bound')) for t in v])
print('\nmulti-label methods with the gate (k2, k3, k4, k>=2, normal named) against the uniform rule without the gate:')
for k in meta['uniform_rule_without_gate']:
    if k.split('/')[1] in ('BR-LGBM', 'CC-LGBM', 'RF-ML', 'ML-CNN'):
        g = meta['uniform_rule_without_gate'][k]; print('  %-16s without gate %s' % (k, ' '.join('%.3f' % g[q][0] for q in ('k2_setF1', 'k3_setF1', 'k4_setF1', 'multi_setF1', 'normal_named'))))
print('\nsingle-fault rule (wide grids) and diagnostic with composed test-calibration cycles, k>=2 set F1:')
for k in meta['single_fault_rule']:
    d = meta['diag_tcalcomp'].get(k)
    print('  %-16s single-fault rule %.3f  uniform rule %s  diag tcal compositions %s' % (k, meta['single_fault_rule'][k]['multi_setF1'][0],
          '%.3f' % agg_now[k] if k in agg_now else '-', '%.3f %s' % (d['multi_setF1'][0], d['chosen']) if d else '-'))
for mode, a in meta['mlcnn_singlecal'].items(): print('  ML-CNN, single-fault rule,', mode, 'k>=2 %.3f' % a['multi_setF1'][0])
json.dump(meta, open(R + 'metrics_final_3splits_meta.json', 'w'), indent=1, default=float)
