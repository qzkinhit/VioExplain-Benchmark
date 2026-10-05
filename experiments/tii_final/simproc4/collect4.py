"""Collect the simproc4 results (results/simproc_v1/{ferm,hex}/) into the delivery files. Rerun at any time;
methods whose run has not finished are missing from the files (and listed under 'missing' in simproc4_logs/summary_simproc4.json).

metrics_main.json      comparison (A): every baseline learns from normal and single-fault fit runs only (f0), VioExplain
                       composes counterfactual windows from the same runs (f100t, run label B: learned addition decision
                       without the step feature, ResNet temporal encoder, functional-constraint units)
metrics_samecomp.json  comparison (B): BR-LGBM, CC-LGBM and ML-CNN receive the compositions of VioExplain (f100t), plus
                       VioExplain itself
Fields per method: macro_f1 (gated macro F1 on single-fault evaluation windows, first named event or normal),
normal_named (share of normal evaluation windows with a named event), k1_f1 (event-set F1 against the effective-event
truth on single-fault evaluation windows), pair_f1, triple_f1, quad_f1 (event-set F1 on two, three and four
simultaneous faults), attr_f1 (attribution F1, mean over pairs, triples and four faults), n_pair, n_triple, n_quad
(evaluated windows), source (file and key).
metrics_extra.json     VioExplain without any composition (f0), for reference
Also writes dataset_stats.json per data set and simproc4_logs/summary_simproc4.md (all three tables) under results/simproc_v1/.
"""
import json, os, re
R = '/path/to/vioexplain/results/simproc_v1/'
DIRS = [(p, p) for p in ("ferm", "hex")]
A = [('PCA-RBC', 'f0/metrics_base.json', 'PCA-RBC'), ('FDA', 'f0/metrics_base.json', 'FDA'),
     ('RF', 'f0/metrics_base.json', 'RF-top1'), ('XGB', 'f0/metrics_base.json', 'XGB-top1'), ('LGBM', 'f0/metrics_base.json', 'MC-LGBM-top1'),
     ('MiniRocket', 'f0/metrics_MiniRocket.json', 'MiniRocket'), ('MultiRocket', 'f0/metrics_MultiRocket.json', 'MultiRocket'),
     ('QUANT', 'f0/metrics_QUANT.json', 'QUANT'), ('1D-CNN', 'f0/metrics_probs.json', '1D-CNN'), ('LSTM', 'f0/metrics_probs.json', 'LSTM'),
     ('ResNet', 'f0/metrics_probs.json', 'ResNet'), ('InceptionTime', 'f0/metrics_probs_late.json', 'InceptionTime'),
     ('MantisV2', 'f0/metrics_probs_late.json', 'MantisV2'), ('BR-LGBM', 'f0/metrics_base.json', 'BR-LGBM'),
     ('CC-LGBM', 'f0/metrics_base.json', 'CC-LGBM'), ('ML-CNN', 'f0/metrics_mlcnn.json', 'ML-CNN'),
     ('AEC', 'f0/metrics_base.json', 'AEC'), ('MinExplain', 'f0/metrics_base.json', 'MinExplain'),
     ('VioExplain', 'f100t/metrics_B.json', 'VioExplain-nostep-B')]
B = [('BR-LGBM', 'f100t/metrics_base.json', 'BR-LGBM'), ('CC-LGBM', 'f100t/metrics_base.json', 'CC-LGBM'),
     ('ML-CNN', 'f100t/metrics_mlcnn.json', 'ML-CNN'), ('VioExplain', 'f100t/metrics_B.json', 'VioExplain-nostep-B')]


def counts(d):
    for f in sorted(os.listdir(R + d + '/f0')) if os.path.isdir(R + d + '/f0') else []:
        if not f.startswith('log'): continue
        for line in open(R + d + '/f0/' + f):
            m = re.search(r'concurrent groups: pairs (\d+) windows (\d+) triples (\d+) windows (\d+) quads (\d+)', line)
            if m: return dict(n_pair_sets=int(m[1]), n_pair=int(m[2]), n_triple_sets=int(m[3]), n_triple=int(m[4]), n_quad_sets=int(m[5]), n_quad=30 * 12 * int(m[5]))
    return {}


def load(d, path, key, cnt):
    try: m = json.load(open(R + d + '/' + path))[key]
    except Exception: return None
    return dict(macro_f1=m['single_macroF1_gated'], normal_named=m['normal_named'], pair_f1=m['pair_eff'], triple_f1=m['triple_eff'],
                quad_f1=m['quad_eff'], attr_f1=(m['pair_attrF1'] + m['triple_attrF1'] + m['quad_attrF1']) / 3, k1_f1=m['single_eff'],
                n_pair=cnt.get('n_pair'), n_triple=cnt.get('n_triple'), n_quad=cnt.get('n_quad'), source=d + '/' + path + ':' + key)


COLS = ['macro_f1', 'normal_named', 'k1_f1', 'pair_f1', 'triple_f1', 'quad_f1', 'attr_f1']
L = []; SUM = {}
for d, proc in DIRS:
    if not os.path.isdir(R + d): continue
    cnt = counts(d); main = {}; same = {}; miss = []
    for n, p, k in A:
        v = load(d, 'f0/metrics_kb.json', k, cnt) if n in ('AEC', 'MinExplain') else None   # extended grids (dist, csth)
        v = v or load(d, p, k, cnt)
        if v: main[n] = v
        else: miss.append('A:' + n)
    for n, p, k in B:
        v = load(d, p, k, cnt)
        if v: same[n] = v
        else: miss.append('B:' + n)
    json.dump(main, open(R + d + '/metrics_main.json', 'w'), indent=1); json.dump(same, open(R + d + '/metrics_samecomp.json', 'w'), indent=1)
    extra = {}
    v = load(d, 'f0/metrics_B.json', 'VioExplain-nostep-B', cnt)
    if v: extra['VioExplain without composition (f0)'] = v
    json.dump(extra, open(R + d + '/metrics_extra.json', 'w'), indent=1)
    try:
        DS = json.load(open('/path/to/vioexplain/data/simproc4/%s/dataset.json' % proc))
        st = dict(process=proc, results_dir=d, sample_period=DS['sample_period'], window=DS['W'], variables=DS['vars'], n_variables=DS['M'],
                  faults=DS['faults'], n_fault_types=DS['C'] - 1, fit_runs_per_class=DS['n_fit'], cal_runs_per_class=DS['n_cal'],
                  test_runs_per_class=DS['n_test'], train_samples=DS['T_train'], train_onset=DS['onset_train'], test_samples=DS['T_test'],
                  test_onset=DS['onset_test'], pair_sets=DS['n_pairs'], pair_seeds=DS['pair_seeds'], triple_sets=DS['n_triples'],
                  triple_seeds=DS['triple_seeds'], composed_triple_sets=DS['tri_sets'], trips=DS['stats'], **cnt)
        for f in os.listdir(R + d + '/f100t') if os.path.isdir(R + d + '/f100t') else []:
            if f == 'relations.json': st['functional_relations'] = len(json.load(open(R + d + '/f100t/' + f)))
        for line in open(R + d + '/f0/log_base.txt') if os.path.exists(R + d + '/f0/log_base.txt') else []:
            m = re.search(r'single eval windows (\d+) tcal (\d+)', line)
            if m: st['single_eval_windows'] = int(m[1]); st['single_tcal_windows'] = int(m[2])
        json.dump(st, open(R + d + '/dataset_stats.json', 'w'), indent=1)
    except Exception as e:
        st = {'error': str(e)}
    SUM[d] = dict(missing=miss, counts=cnt)
    f = lambda r, c: ('%.1f' % (100 * r[c]) if c == 'normal_named' else '%.3f' % r[c])
    L += ['## %s (%s): comparison (A), baselines fitted on single-fault runs' % (d, proc), '', '| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |', '|---|---|---|---|---|---|---|---|']
    for n, _, _ in A:
        if n in main: L.append('| %s | %s |' % (n, ' | '.join(f(main[n], c) for c in COLS)))
        else: L.append('| %s | %s |' % (n, ' | '.join(['--'] * len(COLS))))
    L += ['', '## %s: comparison (B), set-valued methods with the compositions of VioExplain' % d, '', '| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |', '|---|---|---|---|---|---|---|---|']
    for n, _, _ in B:
        if n in same: L.append('| %s | %s |' % (n, ' | '.join(f(same[n], c) for c in COLS)))
        else: L.append('| %s | %s |' % (n, ' | '.join(['--'] * len(COLS))))
    for n, v in extra.items(): L += ['', '| Extra: %s | %s |' % (n, ' | '.join(f(v, c) for c in COLS))]
    L += ['', 'windows: %s, missing: %s' % (json.dumps(cnt), ', '.join(miss) or 'none'), '']
open(R + 'simproc4_logs/summary_simproc4.md', 'w').write('\n'.join(L) + '\n'); json.dump(SUM, open(R + 'simproc4_logs/summary_simproc4.json', 'w'), indent=1)
print('\n'.join(L))
