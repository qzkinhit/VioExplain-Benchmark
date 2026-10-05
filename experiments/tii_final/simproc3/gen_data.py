"""Generate one simulated benchmark in the directory layout of the TEP runs (read by common.py with V3_DATA=<process>).

Usage:  SIM_NPROC=5 python gen_data.py evap|ph   (copy of scripts/simproc2/gen_data.py; output under data/simproc3/)
Layout under data/simproc3/<process>/ :
  training_arrays/class_XX.npy   (400 runs, 512 samples, M)  XX = 00 normal, 01.. single fault; run r of every class uses
                                 seed r (paired runs); runs 0-299 fit, 300-399 calibration; fault onset at sample 64
  testing_arrays/class_XX.npy    (500 runs, 960 samples, M)  seeds 100000+r; onset 160; split into test calibration and
                                 evaluation by the hash of the TEP protocol
  */shutdown_XX.npy              first frozen sample of every run after an interlock trip (-1: none)
  test_pairs/set_<key>.npy       (30 seeds, 960, M) for normal, every single fault and every pair (key '3+7'); onset 160
  test_triples/set_<key>.npy     (15 seeds, 960, M) for a random sample of triples and every subset of them; onset 160
  test_*/meta.json               onset, set names, seed and shutdown sample of every run
  dataset.json                   shapes, names, protocol constants, trip statistics
Every run with the same seed shares its noise and disturbance streams and the magnitude of each fault (common random
numbers), so the runs of all subsets of a fault set are paired runs.
"""
import sys, os, json, itertools, importlib, time
import numpy as np
proc = sys.argv[1]; sim = importlib.import_module('sim_' + proc); t0 = time.time()
ROOT = ('/home/user' if os.path.exists('/path/to/vioexplain') else '/home/user') + '/vioexplain-v3/'
OUT = ROOT + 'data/simproc3/%s/' % proc
W = 64; NFIT, NCAL, NTEST = 300, 100, 500; ON_TR = 64; T_TR = ON_TR + 7 * W; ON_TE = 160; T_TE = 960
NPS, NTS = 30, 15
F = sorted(sim.FAULTS); C = len(F) + 1; M = len(sim.VARS)
for d in ('training_arrays', 'testing_arrays', 'test_pairs', 'test_triples'): os.makedirs(OUT + d, exist_ok=True)


def key(fs): return 'normal' if not fs else '+'.join(str(f) for f in sorted(fs))


stats = {}
# single-fault parts
for part, n, s0, T, on in (('training_arrays', NFIT + NCAL, 0, T_TR, ON_TR), ('testing_arrays', NTEST, 100000, T_TE, ON_TE)):
    runs = [(s0 + r, () if c == 0 else (c,)) for c in range(C) for r in range(n)]
    X, sh = sim.simulate(runs, T, on); X = X.reshape(C, n, T, M); sh = sh.reshape(C, n)
    for c in range(C):
        np.save(OUT + part + '/class_%02d.npy' % c, X[c]); np.save(OUT + part + '/shutdown_%02d.npy' % c, sh[c])
    stats[part] = {'runs_per_class': n, 'samples': T, 'onset': on, 'tripped_runs_per_class': [int((sh[c] >= 0).sum()) for c in range(C)]}
    print(part, X.shape, 'tripped', stats[part]['tripped_runs_per_class'], '[%.0fs]' % (time.time() - t0), flush=True)
# concurrent parts
pairs = list(itertools.combinations(F, 2)); alltri = list(itertools.combinations(F, 3))
NTRI = min(5 * len(F), len(alltri))   # triple samples scaled to the number of faults (5 per fault, as in the CSTR and tank runs)
tri = [alltri[i] for i in sorted(np.random.default_rng(sim.BASE + 3).choice(len(alltri), NTRI, replace=False))]
for part, sets, ns, s0 in (('test_pairs', [()] + [(f,) for f in F] + pairs, NPS, 200000),
                           ('test_triples', sorted(set(tuple(c) for t in tri for r in range(4) for c in itertools.combinations(t, r)), key=lambda x: (len(x), x)), NTS, 300000)):
    runs = [(s0 + r, fs) for fs in sets for r in range(ns)]
    X, sh = sim.simulate(runs, T_TE, ON_TE); X = X.reshape(len(sets), ns, T_TE, M); sh = sh.reshape(len(sets), ns)
    meta = {'onset': ON_TE, 'W': W, 'sets': [key(fs) for fs in sets], 'runs': {}}
    for i, fs in enumerate(sets):
        np.save(OUT + part + '/set_%s.npy' % key(fs), X[i])
        meta['runs'][key(fs)] = [{'seed': s0 + r, 'shutdown_sample': int(sh[i, r])} for r in range(ns)]
    json.dump(meta, open(OUT + part + '/meta.json', 'w'))
    bysize = {}
    for i, fs in enumerate(sets): bysize.setdefault(len(fs), []).append(float((sh[i] >= 0).mean()))
    stats[part] = {'sets': len(sets), 'seeds': ns, 'samples': T_TE, 'onset': ON_TE,
                   'share_of_tripped_runs_by_number_of_faults': {k: float(np.mean(v)) for k, v in bysize.items()}}
    print(part, X.shape, stats[part]['share_of_tripped_runs_by_number_of_faults'], '[%.0fs]' % (time.time() - t0), flush=True)
npairs = len(pairs)
DS = dict(name=proc, vars=sim.VARS, faults={str(k): v for k, v in sim.FAULTS.items()}, sample_period=sim.SAMPLE_PERIOD, W=W, C=C, M=M,
          EFF=F, onset_train=ON_TR, onset_test=ON_TE, T_train=T_TR, T_test=T_TE, n_fit=NFIT, n_cal=NCAL, n_test=NTEST,
          pair_seeds=NPS, triple_seeds=NTS, n_pairs=npairs, n_triples=NTRI, triples=[key(t) for t in tri],
          tri_sets=int(round(300 * npairs / 190)),   # composed triple sets of the training side, same ratio to the pairs as on TEP
          params={k: (list(v) if isinstance(v, tuple) else v) for k, v in sim.P.items()}, stats=stats, doc=sim.__doc__)
json.dump(DS, open(OUT + 'dataset.json', 'w'), indent=1)
print('DONE', proc, '[%.0fs]' % (time.time() - t0))
