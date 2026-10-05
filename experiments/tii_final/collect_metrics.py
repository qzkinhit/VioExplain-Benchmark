"""Collect every metrics*.json of the final runs on cpu1 into one local table (all_metrics.json and a printed view).
Run locally: python collect_metrics.py  (pulls with ssh/scp)."""
import json, os, subprocess, glob
HERE = os.path.dirname(os.path.abspath(__file__)); DST = os.path.join(HERE, 'final_metrics')
os.makedirs(DST, exist_ok=True)
subprocess.run(['rsync', '-a', '--include=*/', '--include=metrics*.json', '--include=*.json', '--exclude=*',
                'cpu1:/path/to/vioexplain/results/final_v1/', DST + '/'], check=True)
ALL = {}
for f in sorted(glob.glob(DST + '/**/*.json', recursive=True)):
    rel = os.path.relpath(f, DST); mode = rel.split('/')[0] if '/' in rel else 'root'
    try: d = json.load(open(f))
    except Exception: continue
    for k, v in d.items():
        if isinstance(v, dict) and 'pair_eff' in v: ALL.setdefault(mode, {})[k] = v
        elif isinstance(v, float): ALL.setdefault(mode, {}).setdefault('_scalars', {})[k] = v
json.dump(ALL, open(os.path.join(HERE, 'all_metrics.json'), 'w'), indent=1)
cols = ['single_macroF1_gated', 'normal_named', 'pair_eff', 'triple_eff', 'quad_eff', 'pair_attrF1', 'triple_attrF1', 'quad_attrF1', 'pair_eff_unseen']
for mode in sorted(ALL):
    print('==', mode); print('%-26s' % 'method' + ''.join('%9s' % c.replace('single_macroF1_gated', 'macroF1').replace('_eff', '').replace('_attrF1', 'Att')[:9] for c in cols))
    for m, v in sorted(ALL[mode].items(), key=lambda kv: -kv[1].get('pair_eff', 0) if isinstance(kv[1], dict) and 'pair_eff' in kv[1] else 0):
        if m == '_scalars': continue
        print('%-26s' % m + ''.join('%9.3f' % v[c] if c in v else '%9s' % '-' for c in cols))
