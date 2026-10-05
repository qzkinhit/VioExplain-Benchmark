"""Closed-loop continuous fermenter (Henson and Seborg 1992) with concurrent faults and paired runs.

Model. The continuous fermenter of Henson and Seborg (Chem. Eng. Sci. 47 (1992) 821-835, "Nonlinear control
strategies for continuous fermenters"), biomass X, substrate S and product P (g/l):
    dX/dt = -D X + mu X
    dS/dt = D (Sf - S) - mu X / Yxs
    dP/dt = -D P + (alpha mu + beta) X
    mu = mu_m (1 - P / Pm) S / (Km + S + S^2 / Ki)
Parameters taken from the source: Yxs = 0.4 g/g, alpha = 2.2 g/g, beta = 0.2 1/h, mu_m = 0.48 1/h, Pm = 50 g/l,
Km = 1.2 g/l, Ki = 22 g/l, Sf = 20 g/l, D = 0.202 1/h, which give the published steady state X = 6.0, S = 5.0,
P = 19.14 g/l.
Deviations, all introduced here: a liquid volume balance dV/dt = Fin - Fout - Fx (nominal V 1 l, so D = Fin / V)
with a level loop on the outlet pump; a feed flow loop (PI on the feed pump, flow meter); an outer biomass loop that
sets the feed flow set point from the optical-density biomass measurement (the biomass is controlled by the dilution
rate as in the source's control study); substrate and product analyzers with 0.25 h sampling and 0.25 h dead time;
an off-gas CO2 signal 0.956 (alpha mu + beta) X V (g/h, ethanol stoichiometry); first-order autoregressive
disturbances of Sf and of mu_m (temperature effect) and of the feed supply; Gaussian noise on the control
measurements and on every recorded variable; ten fault types (list FAULTS), each with a magnitude drawn per run index
from U(0.6, 1.4) times a nominal size in P; interlocks on V and X (wash-out).
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: hour. Sampling period 0.05 h (3 min), control
interval 0.025 h, explicit Euler step 0.0125 h.
"""
import os
import numpy as np

NAME = 'ferm'
VARS = ['Fin_cmd', 'Fin', 'Fout', 'V', 'X_od', 'S', 'P', 'CO2']
FAULTS = {1: 'feed substrate concentration drop (disturbance)', 2: 'inhibitor in the feed, growth rate drop (disturbance)',
          3: 'culture degeneration, biomass yield ramp (process)', 4: 'contaminant consuming substrate, ramp (process)',
          5: 'feed pump gain loss (actuator)', 6: 'outlet valve stiction (actuator)',
          7: 'biomass optical-density sensor drift (sensor, in loop)', 8: 'level sensor bias (sensor, in loop)',
          9: 'substrate analyzer bias (sensor, out of loop)', 10: 'feed flow meter bias (sensor, inner loop)'}
SAMPLE_PERIOD = '0.05 h'
P = dict(Yxs=0.4, alpha=2.2, beta=0.2, mum=0.48, Pm=50.0, Km=1.2, Ki=22.0, Sf=20.0, D0=0.202, V0=1.0,
         Finmax=0.6, Foutmax=0.6, an_T=0.25,
         KF=50.0, TiF=0.1, KX=0.02, TiX=3.0, KV=1.0, TiV=0.5,
         d_sig=(0.3, 0.015, 0.03), d_tau=(4.0, 2.0, 0.3),          # Sf (g/l), mu_m factor, feed supply factor
         c_sig=(0.002, 0.03, 0.003),                                 # control measurement noise Fin (l/h), X (g/l), V (l)
         m_sig=(0.0, 0.002, 0.002, 0.003, 0.03, 0.05, 0.15, 0.05),
         f1_step=-2.5, f2_drop=0.08, f3_rate=0.02, f4_rate=0.04, f5_loss=0.12, f6_band=3.0, f7_rate=0.08,
         f8_bias=0.04, f9_bias=0.8, f10_bias=0.012, V_hi=1.3, V_lo=0.7, X_lo=1.0)
DT = 0.0125; NSUB = 2; NCTL = 2; BURN = 60.0
NF_ = len(FAULTS); BASE = 20261024


def mu(S, Pp, mum):
    return mum * np.maximum(1 - Pp / P['Pm'], 0.0) * S / (P['Km'] + S + S ** 2 / P['Ki'])


def steady():
    from scipy.optimize import fsolve
    D = P['D0']
    def f(x):
        X, S, Pp = x; m = mu(S, Pp, P['mum'])
        return [-D * X + m * X, D * (P['Sf'] - S) - m * X / P['Yxs'], -D * Pp + (P['alpha'] * m + P['beta']) * X]
    return fsolve(f, [6.0, 5.0, 19.14], xtol=1e-13)


SS = steady()


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = int(round((BURN + T * 0.05) / (DT * NSUB))) + 2
    return g.standard_normal((nc, 6)).astype(np.float32), g.standard_normal((T, len(VARS))).astype(np.float32), \
        g.standard_normal(3).astype(np.float32), g.standard_normal((nc // 10 + 2, 2)).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids); T and onset in samples. Returns X (n, T, 8) float32 and shutdown_sample."""
    npr = int(os.environ.get('SIM_NPROC', '1'))
    if npr > 1 and len(runs) > 64:
        from multiprocessing import Pool
        ch = [c for c in np.array_split(np.arange(len(runs)), npr * 3) if len(c)]
        with Pool(npr) as pool:
            res = pool.starmap(_simulate, [([runs[i] for i in c], T, onset) for c in ch])
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
    return _simulate(runs, T, onset)


def _simulate(runs, T, onset):
    n = len(runs); seeds = sorted(set(int(s) for s, _ in runs)); pos = {s: i for i, s in enumerate(seeds)}
    sidx = np.array([pos[int(s)] for s, _ in runs]); NS = [noise_streams(s, T) for s in seeds]
    dist = np.stack([a[0] for a in NS]); meas = np.stack([a[1] for a in NS]); d0 = np.stack([a[2] for a in NS]); ann = np.stack([a[3] for a in NS])
    act = np.zeros((n, NF_ + 1), bool); mag = np.zeros((n, NF_ + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    X0, S0, P0 = SS; Fin0 = P['D0'] * P['V0']; uin0 = 100 * Fin0 / P['Finmax']; uout0 = 100 * Fin0 / P['Foutmax']
    Xb = np.full(n, X0); Sb = np.full(n, S0); Pb = np.full(n, P0); V = np.full(n, P['V0'])
    uin = np.full(n, uin0); uout = np.full(n, uout0); pout = np.full(n, uout0); Fsp = np.full(n, Fin0)
    IF = np.zeros(n); IX = np.zeros(n); IV = np.zeros(n)
    held = np.tile(np.array([S0, P0]), (n, 1)); pend = held.copy(); an_k = 0
    dtc = DT * NSUB; dsig = np.array(P['d_sig']); phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig']); csig = np.array(P['c_sig'])
    Xr = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    spp = 0.05; nc = int(round((BURN + T * spp) / dtc)); per = int(round(spp / dtc)); anper = int(round(P['an_T'] / dtc))
    Fin = np.full(n, Fin0); Fout = np.full(n, Fin0); co2 = np.zeros(n)
    for k in range(nc):
        t = ((k + 1) * dtc - BURN) / spp; tau = max(t - onset, 0.0) * spp; on = t > onset
        dk = dist[sidx, k].astype(np.float64); d = phi * d + dinn * dk[:, :3]; cs = csig * dk[:, 3:]
        Sf = P['Sf'] + d[:, 0]; mum = P['mum'] * (1 + d[:, 1]); sup = 1 + d[:, 2]
        Y = np.full(n, P['Yxs']); cons = np.zeros(n); gin = np.ones(n); bX = np.zeros(n); bV = np.zeros(n); bS = np.zeros(n); bF = np.zeros(n)
        if on:
            Sf = Sf + am[:, 1] * P['f1_step']; mum = mum * (1 - am[:, 2] * P['f2_drop'])
            Y = Y * np.maximum(1 - am[:, 3] * P['f3_rate'] * tau, 0.6); cons = am[:, 4] * np.minimum(P['f4_rate'] * tau, 0.4)
            gin = 1 - am[:, 5] * P['f5_loss']; bX = am[:, 7] * P['f7_rate'] * tau; bV = am[:, 8] * P['f8_bias']
            bS = am[:, 9] * P['f9_bias']; bF = am[:, 10] * P['f10_bias']
        # control: biomass -> feed flow set point -> feed pump; level -> outlet valve
        Fm = Fin + bF + cs[:, 0]; Xm = Xb + bX + cs[:, 1]; Vm = V + bV + cs[:, 2]
        ex = Xm - X0; Fsp = Fin0 + P['KX'] * (ex + IX); IX = np.where((Fsp > 0.9 * P['Finmax']) & (ex > 0), IX, IX + ex * dtc / P['TiX'])
        Fsp = np.clip(Fin0 + P['KX'] * (ex + IX), 0.02, 0.9 * P['Finmax'])
        ef = Fsp - Fm; uin = uin0 + P['KF'] * (ef + IF); sf = ((uin >= 100) & (ef > 0)) | ((uin <= 0) & (ef < 0))
        IF = np.where(sf, IF, IF + ef * dtc / P['TiF']); uin = np.clip(uin0 + P['KF'] * (ef + IF), 0, 100)
        ev = Vm - P['V0']; uo = uout0 + 100 / P['Foutmax'] * P['KV'] * (ev + IV); so = ((uo >= 100) & (ev > 0)) | ((uo <= 0) & (ev < 0))
        IV = np.where(so, IV, IV + ev * dtc / P['TiV']); uout = np.clip(uout0 + 100 / P['Foutmax'] * P['KV'] * (ev + IV), 0, 100)
        if on:
            s6 = act[:, 6] & (np.abs(uout - pout) <= mag[:, 6] * P['f6_band']); pout = np.where(s6, pout, uout)
        else:
            pout = uout
        for _ in range(NSUB):
            Fin = P['Finmax'] * uin / 100 * gin * np.maximum(sup, 0.0); Fout = P['Foutmax'] * pout / 100 * np.sqrt(np.maximum(V, 0.0) / P['V0'])
            D = Fin / np.maximum(V, 1e-3); m = mu(Sb, Pb, mum); qp = (P['alpha'] * m + P['beta']) * Xb
            dX = -D * Xb + m * Xb; dS = D * (Sf - Sb) - m * Xb / Y - cons * Sb / (P['Km'] + Sb); dP = -D * Pb + qp
            Xb = Xb + DT * dX; Sb = np.maximum(Sb + DT * dS, 0.0); Pb = Pb + DT * dP; V = V + DT * (Fin - Fout)
            co2 = 0.956 * qp * V
        if (k + 1) % anper == 0:   # substrate and product analyzers: 0.25 h sampling, 0.25 h dead time
            a_ = ann[sidx, an_k % ann.shape[1]].astype(np.float64); an_k += 1
            held = pend; pend = np.stack([Sb + bS + msig[5] * a_[:, 0], Pb + msig[6] * a_[:, 1]], 1)
        tripped = tripped | (V > P['V_hi']) | (V < P['V_lo']) | (Xb < P['X_lo']) | ~np.isfinite(Xb)
        if tripped.any():
            Xb = np.where(tripped, X0, Xb); Sb = np.where(tripped, S0, Sb); Pb = np.where(tripped, P0, Pb); V = np.where(tripped, P['V0'], V)
        if (k + 1) * dtc > BURN + 1e-9 and (k + 1) % per == 0:
            j = int(round(((k + 1) * dtc - BURN) / spp)) - 1
            if j < 0 or j >= T: continue
            mn = meas[sidx, j].astype(np.float64) * msig; mn[:, 5:7] = 0
            row = np.stack([uin, Fin + bF, Fout, V + bV, Xb + bX, held[:, 0], held[:, 1], co2], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; Xr[:, j] = row
    return Xr, shut
