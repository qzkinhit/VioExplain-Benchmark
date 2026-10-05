"""Print the main table of the paper from the result files of this directory.

Usage:  python results/print_main_table.py

Columns: TEP-R, FERM, HEX (macro F1, MF1), TEP-C, HYD, CSTR, QTank, DIST, CSTH, DTS200 (event-set F1 on windows with
two or more concurrent events, SF1), TEP-C, EVAP, PH (attribution F1, AF1). On TEP-C and the simulated processes
SF1 and AF1 are the equal-weight mean over two, three and four faults. On HYD the value is the mean over three splits.
The rank orders the methods by their mean rank over the columns.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ['PCA-RBC', 'FDA', 'RF', 'XGB', 'LGBM', 'MiniRocket', 'MultiRocket', 'QUANT', '1D-CNN', 'LSTM', 'ResNet',
           'InceptionTime', 'MantisV2', 'BR', 'CC', 'ML-CNN', 'AEC', 'MinExplain', 'VioExplain']
ALIAS = {'RF': ['RF-top1', 'RF'], 'XGB': ['XGB-top1', 'XGB'], 'LGBM': ['MC-LGBM-top1', 'LGBM'], 'BR': ['BR-LGBM', 'BR'],
         'CC': ['CC-LGBM', 'CC']}


def load(rel):
    with open(os.path.join(HERE, rel)) as f:
        return json.load(f)


def row_of(d, method):
    """Metrics of one method in a result file: the entry under one of its names, or the only entry of VioExplain."""
    for k in ALIAS.get(method, [method]):
        if isinstance(d.get(k), dict):
            return d[k]
    ks = [k for k, v in d.items() if isinstance(v, dict) and k.startswith(method)]
    return d[ks[0]]


def mean3(*v):
    return sum(v) / 3


cols = {}
tep = {m: row_of(load('tep/main/%s.json' % m), m) for m in METHODS}
cols['TEP-R MF1'] = {m: r['single_macroF1_gated'] for m, r in tep.items()}
cols['TEP-C SF1'] = {m: mean3(r['pair_eff'], r['triple_eff'], r['quad_eff']) for m, r in tep.items()}
cols['TEP-C AF1'] = {m: mean3(r['pair_attrF1'], r['triple_attrF1'], r['quad_attrF1']) for m, r in tep.items()}
h0, h1 = load('hyd/single_fault_knowledge.json'), load('hyd/with_compositions.json')
cols['HYD SF1'] = {m: (h1['VioExplain'] if m == 'VioExplain' else row_of(h0, m))['multi_setF1'][0] for m in METHODS}
for p in ('cstr', 'qtank'):
    rows = {r['method']: r for r in load('simulated/%s/summary.json' % p)['comparison_A']}
    cols[p.upper() + ' SF1'] = {m: mean3(rows[m]['pairs'], rows[m]['triples'], rows[m]['four']) for m in METHODS}
for p in ('dist', 'csth', 'dts200'):
    d = load('simulated/%s/metrics_main.json' % p)
    cols[p.upper() + ' SF1'] = {m: mean3(*(row_of(d, m)[k] for k in ('pair_f1', 'triple_f1', 'quad_f1'))) for m in METHODS}
for p, key, tag in (('ferm', 'macro_f1', 'MF1'), ('hex', 'macro_f1', 'MF1'), ('evap', 'attr_f1', 'AF1'), ('ph', 'attr_f1', 'AF1')):
    d = load('simulated/%s/metrics_main.json' % p)
    cols['%s %s' % (p.upper(), tag)] = {m: row_of(d, m)[key] for m in METHODS}
ORDER = ['TEP-R MF1', 'FERM MF1', 'HEX MF1', 'TEP-C SF1', 'HYD SF1', 'CSTR SF1', 'QTANK SF1', 'DIST SF1', 'CSTH SF1',
         'DTS200 SF1', 'TEP-C AF1', 'EVAP AF1', 'PH AF1']

ranks = {m: [] for m in METHODS}
for c in ORDER:
    vals = sorted(((round(cols[c][m], 3), m) for m in METHODS), key=lambda t: -t[0])   # ranks follow the printed values
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and abs(vals[j + 1][0] - vals[i][0]) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[vals[k][1]].append((i + j) / 2 + 1)
        i = j + 1
mean_rank = {m: sum(r) / len(r) for m, r in ranks.items()}
mean_val = {m: sum(round(cols[c][m], 3) for c in ORDER) / len(ORDER) for m in METHODS}
final = {m: i + 1 for i, m in enumerate(sorted(METHODS, key=lambda m: (mean_rank[m], -mean_val[m])))}

print('%4s  %-14s' % ('Rank', 'Method') + ''.join('%11s' % c for c in ORDER))
for m in METHODS:
    print('%4d  %-14s' % (final[m], m) + ''.join('%11.3f' % cols[c][m] for c in ORDER))
