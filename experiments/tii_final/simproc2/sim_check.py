"""Quick behaviour check of a simulator before data generation (no method is run here).
Usage: python sim_check.py cstr|qtank
Prints, for normal runs and every single fault, the per-variable shift of the window mean, standard deviation and slope
in units of the normal between-window standard deviation (windows of 64 samples after the onset), the share of windows
with at least one typed violation (|z| > 3 against the paired normal window being in control), and the trip counts.
Also checks determinism (two calls give identical arrays) and the pairing (identical records before the onset)."""
import sys, importlib, time
import numpy as np
sim = importlib.import_module('sim_' + sys.argv[1]); W = 64; T = 160 + 12 * W; ON = 160; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
t0 = time.time(); F = sorted(sim.FAULTS)
runs = [(s, ()) for s in range(NS)] + [(s, (f,)) for f in F for s in range(NS)]
X, sh = sim.simulate(runs, T, ON); print('simulated', X.shape, '%.0fs' % (time.time() - t0))
X2, sh2 = sim.simulate(runs[:NS] + runs[NS:2 * NS], T, ON); print('deterministic', bool((X2 == X[:2 * NS]).all()), 'paired before onset', bool((X[NS:2 * NS, :ON] == X[:NS, :ON]).all()))
tt = np.arange(W) - (W - 1) / 2; TT = (tt ** 2).sum()


def st(x):   # (runs, T, M) -> (runs, windows, M, 3)
    w = np.stack([x[:, s:s + W] for s in range(ON, T - W + 1, W)], 1)
    return np.stack([w.mean(2), w.std(2), (w * tt[None, None, :, None]).sum(2) / TT], 3)


S0 = st(X[:NS].astype(np.float64)); mu = S0.reshape(-1, S0.shape[2], 3).mean(0); sd = S0.reshape(-1, S0.shape[2], 3).std(0) + 1e-12
np.set_printoptions(precision=1, suppress=True, linewidth=220)
print('vars', sim.VARS); print('normal mean', mu[:, 0]); print('normal window std of mean', sd[:, 0]); print('normal within-window std', mu[:, 1])
z0 = (S0 - mu) / sd; v0 = np.abs(z0) > 3; print('normal windows with a violation', round(float(v0.any((2, 3)).mean()), 3), 'trips', int((sh[:NS] >= 0).sum()))
for i, f in enumerate(F):
    Xf = X[(i + 1) * NS:(i + 2) * NS].astype(np.float64); s = sh[(i + 1) * NS:(i + 2) * NS]; z = (st(Xf) - mu) / sd
    caused = (np.abs(z) > 3) & ~v0
    print('\nfault %d %s: trips %d (first at %s); windows with a caused violation: first window %.2f, all %.2f' % (
        f, sim.FAULTS[f], int((s >= 0).sum()), s[s >= 0].min() if (s >= 0).any() else '-', caused[:, 0].any((1, 2)).mean(), caused.any((2, 3)).mean()))
    dz = (st(Xf) - S0) / sd
    for nm, j in (('mean', 0), ('std', 1), ('slope', 2)):
        print('   d%-5s first win' % nm, dz[:, 0, :, j].mean(0), '\n          last win ', dz[:, -1, :, j].mean(0))
