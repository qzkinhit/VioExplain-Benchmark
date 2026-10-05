"""Collect the simulated-process results (cpu-server, results/simproc_v1/<process>/) into summary files. Rerun at any time;
rows whose run has not finished are null.

Comparison (A), as the main table of the paper: every baseline learns from normal and single-fault fit runs only (f0),
VioExplain composes counterfactual windows from the same runs (f100t, run label B: learned addition decision without the
step feature, ResNet temporal encoder, functional-constraint units). Comparison (B), as the composition table: BR, CC
and ML-CNN receive the same compositions as VioExplain (f100t). Extra row: VioExplain without any composition (f0).
Columns: macro F1 (gated, first named event or normal), share of normal windows with a named event, event-set F1 on
pairs, triples and four faults, attribution F1 (mean over pairs, triples and four faults).
Writes results/simproc_v1/<process>/summary.json and summary.md, and results/simproc_v1/summary_all.json.
"""
import json, os
R = '/path/to/vioexplain/results/simproc_v1/'
A_ROWS = [('Process monitoring', 'PCA-RBC', 'f0/metrics_PCA-RBC.json', 'PCA-RBC'),
          ('Process monitoring', 'FDA', 'f0/metrics_FDA.json', 'FDA'),
          ('Fault classifiers', 'RF', 'f0/metrics_RF.json', 'RF-top1'),
          ('Fault classifiers', 'XGB', 'f0/metrics_XGB.json', 'XGB-top1'),
          ('Fault classifiers', 'LGBM', 'f0/metrics_MC-LGBM.json', 'MC-LGBM-top1'),
          ('Time-series classifiers', 'MiniRocket', 'f0/metrics_MiniRocket.json', 'MiniRocket'),
          ('Time-series classifiers', 'MultiRocket', 'f0/metrics_MultiRocket.json', 'MultiRocket'),
          ('Time-series classifiers', 'QUANT', 'f0/metrics_QUANT.json', 'QUANT'),
          ('Deep diagnosers', '1D-CNN', 'f0/metrics_1D-CNN.json', '1D-CNN'),
          ('Deep diagnosers', 'LSTM', 'f0/metrics_LSTM.json', 'LSTM'),
          ('Deep diagnosers', 'ResNet', 'f0/metrics_ResNet.json', 'ResNet'),
          ('Deep diagnosers', 'InceptionTime', 'f0/metrics_InceptionTime.json', 'InceptionTime'),
          ('Time-series foundation model', 'MantisV2', 'f0/metrics_MantisV2.json', 'MantisV2'),
          ('Multi-label diagnosis', 'BR', 'f0/metrics_BR-LGBM.json', 'BR-LGBM'),
          ('Multi-label diagnosis', 'CC', 'f0/metrics_CC-LGBM.json', 'CC-LGBM'),
          ('Multi-label diagnosis', 'ML-CNN', 'f0/metrics_mlcnn.json', 'ML-CNN'),
          ('Knowledge-based explanation', 'AEC', 'f0/metrics_AEC.json', 'AEC'),
          ('Knowledge-based explanation', 'MinExplain', 'f0/metrics_MinExplain.json', 'MinExplain')]
OURS = ('VioExplain', 'f100t/metrics_B.json', 'VioExplain-nostep-B')
B_ROWS = [('BR', 'f100t/metrics_BR-LGBM.json', 'BR-LGBM'), ('CC', 'f100t/metrics_CC-LGBM.json', 'CC-LGBM'),
          ('ML-CNN', 'f100t/metrics_mlcnn.json', 'ML-CNN')]
EXTRA = [('VioExplain without composition (f0)', 'f0/metrics_B.json', 'VioExplain-nostep-B')]
COLS = ['macroF1', 'normal_pct', 'pairs', 'triples', 'four', 'attrF1']


def load(proc, path, key):
    try: m = json.load(open(R + proc + '/' + path))[key]
    except Exception: return None
    return dict(macroF1=m['single_macroF1_gated'], normal_pct=100 * m['normal_named'], pairs=m['pair_eff'], triples=m['triple_eff'],
                four=m['quad_eff'], attrF1=(m['pair_attrF1'] + m['triple_attrF1'] + m['quad_attrF1']) / 3,
                single_eff=m['single_eff'], source=proc + '/' + path + ':' + key)


def pct(a, b): return None if a is None or b is None else round(100 * (a / b - 1), 1)


ALL = {}
for proc in ('cstr', 'qtank'):
    if not os.path.isdir(R + proc): continue
    A = [dict(family=f, method=n, **(load(proc, p, k) or {})) for f, n, p, k in A_ROWS]
    o = load(proc, OURS[1], OURS[2]); A.append(dict(family='Ours', method='VioExplain', **(o or {})))
    B = [dict(method=n, **(load(proc, p, k) or {})) for n, p, k in B_ROWS] + [dict(method='VioExplain', **(o or {}))]
    X = [dict(method=n, **(load(proc, p, k) or {})) for n, p, k in EXTRA]
    ml0 = [r for r in A if r['method'] in ('BR', 'CC', 'ML-CNN') and 'pairs' in r]
    ml1 = [r for r in B[:3] if 'pairs' in r]
    base = [r for r in A[:-1] if 'pairs' in r]
    gains = {}
    if o:
        for c in ('pairs', 'triples', 'four'):
            if ml0: gains['A_gain_pct_over_strongest_multilabel_' + c] = pct(o[c], max(r[c] for r in ml0))
            if base: gains['A_gain_pct_over_strongest_baseline_' + c] = pct(o[c], max(r[c] for r in base))
            if ml1: gains['B_gain_pct_over_strongest_multilabel_' + c] = pct(o[c], max(r[c] for r in ml1))
        if base: gains['A_attrF1_points_over_best_baseline'] = round(100 * (o['attrF1'] - max(r['attrF1'] for r in base)), 1)
        ke = [r for r in A if r['method'] in ('AEC', 'MinExplain') and 'four' in r]
        if len(ke) == 2: gains['A_four_gain_pct_over_stronger_of_AEC_MinExplain'] = pct(o['four'], max(r['four'] for r in ke))
        if ml0 and o['normal_pct'] > 0: gains['A_normal_named_ratio_multilabel_over_ours'] = round(max(r['normal_pct'] for r in ml0) / o['normal_pct'], 1)
    best = {c: max([r[c] for r in A if c in r] or [None]) if c != 'normal_pct' else None for c in COLS}
    done = sum('pairs' in r for r in A) + sum('pairs' in r for r in B[:3])
    S = dict(process=proc, comparison_A=A, comparison_B=B, extra=X, gains=gains, column_best_A=best,
             finished_rows=done, expected_rows=len(A) + 3)
    json.dump(S, open(R + proc + '/summary.json', 'w'), indent=1); ALL[proc] = S
    fmt = lambda r, c: ('%.1f' % r[c] if c == 'normal_pct' else '%.3f' % r[c]) if c in r else '--'
    L = ['# %s: comparison (A), baselines fitted on single-fault runs' % proc, '',
         '| No. | Family | Method | Macro F1 | Normal (%) | Pairs | Triples | Four | Attr. F1 |', '|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(A): L.append('| %d | %s | %s | %s |' % (i + 1, r['family'], r['method'], ' | '.join(fmt(r, c) for c in COLS)))
    L += ['', '# %s: comparison (B), multi-label methods with the compositions of VioExplain' % proc, '',
          '| Method | Pairs | Triples | Four | Macro F1 | Normal (%) |', '|---|---|---|---|---|---|']
    for r in B: L.append('| %s | %s |' % (r['method'], ' | '.join(fmt(r, c) for c in ('pairs', 'triples', 'four', 'macroF1', 'normal_pct'))))
    L += ['', '| Extra | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    for r in X: L.append('| %s | %s |' % (r['method'], ' | '.join(fmt(r, c) for c in COLS)))
    L += ['', 'Gains: ' + json.dumps(gains), '', 'Finished rows: %d of %d' % (done, len(A) + 3)]
    open(R + proc + '/summary.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
json.dump(ALL, open(R + 'summary_all.json', 'w'), indent=1)
