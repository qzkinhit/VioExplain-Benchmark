"""Closed-loop binary distillation column (column A of Skogestad and Morari) with concurrent faults and paired runs.

Model. Column A as in Skogestad's colamod.m (Skogestad and Morari 1988; Skogestad 1997): 41 stages (stage 1 the
reboiler, stage 41 the total condenser), feed on stage 21, constant relative volatility alpha = 1.5, constant molar
flows, linearized liquid flow dynamics L_i = L0 + (M_i - M0) / tau_L with tau_L = 0.063 min and lambda = 0, holdup
M0 = 0.5 kmol on every stage, nominal F = 1 kmol/min, zF = 0.5, qF = 1, L = 2.70629, V = 3.20629, D = B = 0.5,
xD = 0.99, xB = 0.01. Material balances of total and light-component holdup on every stage as in colamod.m.
LV configuration as in cola_lv.m: condenser holdup by the distillate D and reboiler holdup by the bottoms B, both
proportional controllers with gain 10. Composition loops as in Skogestad's colas_PI.m (LV
configuration): PI control of the top composition by the reflux L (gain 26.1, integral time 3.76 min) and of the
bottom composition by the boilup V (gain -37.5 on the error set point minus measurement, integral time 3.31 min), on
mole fractions, with a measurement delay of 1 min on each composition.
Deviations from the published model, all introduced here:
  * the composition controllers run every 0.25 min (discrete PI) with conditional integration as anti-windup;
  * reflux and boilup act through first-order valves (time constant 0.1 min, range twice the nominal flow), which
    allows valve stiction (stick-slip) and gain loss;
  * Murphree vapor efficiency E on the rectifying trays (E = 1 nominally, as in colamod.m) for the efficiency fault;
  * tray temperatures from the linear boiling-point rule T = 100 - 20 x (deg C), an assumption of this file;
  * three stochastic disturbances (feed flow, feed composition, feed liquid fraction) as first-order autoregressive
    processes updated every 0.25 min, and Gaussian measurement noise on every recorded variable;
  * ten fault types (list FAULTS), each with a magnitude drawn per run index from U(0.6, 1.4) times a nominal size.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: minute. Sampling period 1 min, control interval
0.25 min, explicit Euler step 0.025 min.
"""
import os
import numpy as np

NAME = 'dist'
TSTAGES = [4, 8, 12, 16, 26, 30, 34, 38]                     # 1-based stages with a temperature sensor
VARS = ['F', 'zF', 'L', 'V', 'D', 'B', 'MD', 'MB', 'xD', 'xB'] + ['T%d' % s for s in TSTAGES] + ['uL', 'uV']
FAULTS = {1: 'feed composition step (disturbance)', 2: 'feed flow variability (disturbance)',
          3: 'feed flow step (disturbance)', 4: 'feed liquid fraction drop, partly vaporized feed (disturbance)',
          5: 'reflux valve stiction (actuator)', 6: 'boilup valve gain loss (actuator)',
          7: 'top composition analyzer bias (sensor, in loop)', 8: 'bottom composition analyzer drift (sensor, in loop)',
          9: 'stage-8 temperature sensor drift (sensor, out of loop)', 10: 'tray efficiency loss in the rectifying section (process)'}
SAMPLE_PERIOD = '1 min'
P = dict(NT=41, NF=21, alpha=1.5, F0=1.0, zF0=0.5, qF0=1.0, L0=2.70629, V0=3.20629, D0=0.5, B0=0.5, M0=0.5, taul=0.063,
         KcD=10.0, KcB=10.0, xDsp=0.99, xBsp=0.01, KL=26.1, TiL=3.76, KV=-37.5, TiV=3.31, tauv=0.1, ana_delay=1.0,
         d_sig=(0.010, 0.008, 0.010), d_tau=(10.0, 30.0, 20.0),           # F, zF, qF: std and correlation time (min)
         m_sig=(0.005, 0.002, 0.01, 0.01, 0.005, 0.005, 0.003, 0.003, 0.0, 0.0) + (0.05,) * 8 + (0.0, 0.0),
         f1_step=0.03, f2_mult=4.0, f3_step=0.06, f4_step=-0.15, f5_band=0.8, f6_loss=0.10, f7_bias=0.002,
         f8_rate=8.0e-6, f9_rate=0.01, f10_loss=0.2)
DT = 0.025; NSUB = 10; NCTL = 4; BURN = 100
NF_ = len(FAULTS); BASE = 20261006
NT = P['NT']; FS = P['NF'] - 1                                           # 0-based feed stage
TIDX = np.array(TSTAGES) - 1


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = (BURN + T) * NCTL
    return g.standard_normal((nc, 3)).astype(np.float32), g.standard_normal((nc, 2)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(3).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def logit(x): return np.log(np.clip(x, 1e-9, 1 - 1e-9) / (1 - np.clip(x, 1e-9, 1 - 1e-9)))


def derivs(Mh, x, LT, VB, F, zF, qF, E):
    a = P['alpha']; ys = a * x[:, :NT - 1] / (1 + (a - 1) * x[:, :NT - 1])
    if E is not None:                                                    # Murphree vapor efficiency on the rectifying trays
        y = ys.copy()
        for i in range(FS + 1, NT - 1): y[:, i] = y[:, i - 1] + E * (ys[:, i] - y[:, i - 1])
    else:
        y = ys
    n = len(x); V = np.repeat(VB[:, None], NT - 1, 1); V[:, FS:] += ((1 - qF) * F)[:, None]
    L = np.empty((n, NT)); L0b = P['L0'] + P['qF0'] * P['F0']
    L[:, 1:FS + 1] = L0b + (Mh[:, 1:FS + 1] - P['M0']) / P['taul']; L[:, FS + 1:NT - 1] = P['L0'] + (Mh[:, FS + 1:NT - 1] - P['M0']) / P['taul']
    L[:, NT - 1] = LT
    D = np.maximum(P['D0'] + P['KcD'] * (Mh[:, NT - 1] - P['M0']), 0.0); B = np.maximum(P['B0'] + P['KcB'] * (Mh[:, 0] - P['M0']), 0.0)
    dM = np.empty_like(Mh); dMx = np.empty_like(Mh)
    dM[:, 1:NT - 1] = L[:, 2:] - L[:, 1:NT - 1] + V[:, :NT - 2] - V[:, 1:]
    dMx[:, 1:NT - 1] = L[:, 2:] * x[:, 2:] - L[:, 1:NT - 1] * x[:, 1:NT - 1] + V[:, :NT - 2] * y[:, :NT - 2] - V[:, 1:] * y[:, 1:]
    dM[:, FS] += F; dMx[:, FS] += F * zF
    dM[:, 0] = L[:, 1] - V[:, 0] - B; dMx[:, 0] = L[:, 1] * x[:, 1] - V[:, 0] * y[:, 0] - B * x[:, 0]
    dM[:, NT - 1] = V[:, NT - 2] - LT - D; dMx[:, NT - 1] = V[:, NT - 2] * y[:, NT - 2] - (LT + D) * x[:, NT - 1]
    return dM, dMx, D, B


_SS = None


def steady():
    """Composition profile at the nominal inputs (levels at their set points), solved once."""
    global _SS
    if _SS is None:
        from scipy.optimize import fsolve
        one = np.ones(1); Mh = np.full((1, NT), P['M0'])

        def res(z):
            x = 1 / (1 + np.exp(-z[None])); _, dMx, _, _ = derivs(Mh, x, P['L0'] * one, P['V0'] * one, P['F0'] * one, P['zF0'] * one, P['qF0'] * one, None)
            return dMx[0] * 100
        z = fsolve(res, np.linspace(-4.6, 4.6, NT), xtol=1e-13); _SS = 1 / (1 + np.exp(-z))
    return _SS.copy()


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids). Returns X (n, T, len(VARS)) float32 and shutdown_sample (n,)."""
    npr = int(os.environ.get('SIM_NPROC', '1')); steady()
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
    dist = np.stack([a[0] for a in NS]); ctl = np.stack([a[1] for a in NS]); meas = np.stack([a[2] for a in NS]); d0 = np.stack([a[3] for a in NS])
    act = np.zeros((n, NF_ + 1), bool); mag = np.zeros((n, NF_ + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    xs = steady(); x = np.repeat(xs[None], n, 0); Mh = np.full((n, NT), P['M0']); Mx = Mh * x
    Lmax, Vmax = 2 * P['L0'], 2 * P['V0']
    xvL = np.full(n, 50.0); xvV = np.full(n, 50.0); pL = np.full(n, 50.0); ID = np.zeros(n); IB = np.zeros(n)
    dsig = np.array(P['d_sig']); dtc = DT * NSUB; phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    Eeff = 1 - am[:, 10] * P['f10_loss']; gV = 1 - am[:, 6] * P['f6_loss']; anyE = bool(act[:, 10].any())
    nc = (BURN + T) * NCTL; ND = int(round(P['ana_delay'] / dtc)); bufD = np.full((ND, n), P['xDsp']); bufB = np.full((ND, n), P['xBsp']); bp = 0
    for k in range(nc):
        t = (k + 1) * dtc - BURN; tau = max(t - onset, 0.0); on = t > onset
        e = dist[sidx, k].astype(np.float64); mult = np.ones((n, 3))
        if on: mult[:, 0] = np.where(act[:, 2], P['f2_mult'], 1.0)
        d = phi * d + dinn * mult * e
        F = P['F0'] + d[:, 0]; zF = P['zF0'] + d[:, 1]; qF = P['qF0'] + d[:, 2]
        bD = np.zeros(n); bB = np.zeros(n); bT8 = np.zeros(n); g = np.ones(n); E = None
        if on:
            zF = zF + am[:, 1] * P['f1_step']; F = F + am[:, 3] * P['f3_step']; qF = qF + am[:, 4] * P['f4_step']
            bD = am[:, 7] * P['f7_bias']; bB = am[:, 8] * P['f8_rate'] * tau; bT8 = am[:, 9] * P['f9_rate'] * tau; g = gV
            if anyE: E = Eeff
        zF = np.clip(zF, 0.05, 0.95)
        cn = ctl[sidx, k].astype(np.float64)
        xDm = bufD[bp].copy(); xBm = bufB[bp].copy()                   # analyzer readings taken 1 min earlier
        bufD[bp] = x[:, NT - 1] + bD + msig[8] * cn[:, 0]; bufB[bp] = x[:, 0] + bB + msig[9] * cn[:, 1]; bp = (bp + 1) % ND
        eD = P['xDsp'] - xDm; eB = P['xBsp'] - xBm
        uL = 100 * (P['L0'] + P['KL'] * (eD + ID)) / Lmax; uV = 100 * (P['V0'] + P['KV'] * (eB + IB)) / Vmax
        satL = ((uL >= 100) & (eD > 0)) | ((uL <= 0) & (eD < 0)); satV = ((uV >= 100) & (eB < 0)) | ((uV <= 0) & (eB > 0))
        ID = np.where(satL, ID, ID + eD * dtc / P['TiL']); IB = np.where(satV, IB, IB + eB * dtc / P['TiV'])
        uL = np.clip(100 * (P['L0'] + P['KL'] * (eD + ID)) / Lmax, 0, 100); uV = np.clip(100 * (P['V0'] + P['KV'] * (eB + IB)) / Vmax, 0, 100)
        if on:
            stick = act[:, 5] & (np.abs(uL - pL) <= mag[:, 5] * P['f5_band']); pL = np.where(stick, pL, uL)
        else:
            pL = uL
        for _ in range(NSUB):
            xvL = xvL + DT / P['tauv'] * (pL - xvL); xvV = xvV + DT / P['tauv'] * (uV - xvV)
            LT = Lmax * xvL / 100; VB = Vmax * g * xvV / 100
            dM, dMx, D, B = derivs(Mh, x, LT, VB, F, zF, qF, E)
            Mh = Mh + DT * dM; Mx = Mx + DT * dMx; x = np.clip(Mx / np.maximum(Mh, 1e-6), 0.0, 1.0)
        bad = ~np.isfinite(Mh).all(1) | (Mh.min(1) < 0.05) | (Mh.max(1) > 2.0)
        tripped = tripped | bad
        if tripped.any():   # the record is frozen after a trip; the states are reset only to keep them finite
            Mh = np.where(tripped[:, None], P['M0'], Mh); x = np.where(tripped[:, None], xs[None], x); Mx = Mh * x
        if k >= BURN * NCTL and (k + 1) % NCTL == 0:
            j = (k + 1) // NCTL - BURN - 1; mn = meas[sidx, j].astype(np.float64) * msig
            Tst = 100 - 20 * x[:, TIDX]; Tst[:, 1] += bT8
            row = np.concatenate([np.stack([F, zF, LT, VB, D, B, Mh[:, NT - 1], Mh[:, 0], bufD[(bp - 1) % ND], bufB[(bp - 1) % ND]], 1), Tst,
                                  np.stack([uL, uV], 1)], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
