"""Closed-loop forced-circulation evaporator (Newell and Lee 1989) with concurrent faults and paired runs.

Model. The forced-circulation evaporator of Newell and Lee (Applied Process Control: A Case Study, Prentice Hall,
1989), in the form used in process-control benchmarks (for example Kariwala and Cao, Ind. Eng. Chem. Res. 2009;
Govatsmark and Skogestad 2001). Three states, separator level L2 (m), product composition X2 (%) and operating
pressure P2 (kPa):
    20 dL2/dt = F1 - F4 - F2
    20 dX2/dt = F1 X1 - F2 X2
     4 dP2/dt = F4 - F5
with the algebraic relations
    T2 = 0.5616 P2 + 0.3126 X2 + 48.43,   T3 = 0.507 P2 + 55.0,   T100 = 0.1538 P100 + 90.0
    Q100 = UA1 (T100 - T2),  UA1 = 0.16 (F1 + F3),   F100 = Q100 / 36.6
    F4 = (Q100 - 0.07 F1 (T2 - T1)) / 38.5
    Q200 = UA2 (T3 - T200) / (1 + UA2 / (0.14 F200)),  UA2 = 6.84,  T201 = T200 + Q200 / (0.07 F200),  F5 = Q200 / 38.5.
Nominal values taken from the source: F1 = 10 kg/min, X1 = 5 %, T1 = 40 C, F2 = 2 kg/min, X2 = 25 %, L2 = 1 m,
P2 = 50.5 kPa, F3 = 50 kg/min, P100 = 194.7 kPa, F200 = 208 kg/min, T200 = 25 C (these close all balances: F4 = F5
= 8 kg/min, T2 = 84.6 C, T3 = 80.6 C, Q100 = 339 kW, Q200 = 308 kW, T201 = 46.1 C). Control structure as in the
benchmark: level L2 by the product flow F2, product composition X2 by the steam pressure P100, operating pressure P2
by the cooling-water flow F200. The equations and values were written from the benchmark description in the
literature and checked by closing the steady-state balances above; the book itself was not at hand.
Deviations from the published model, all introduced here:
  * PI settings chosen on this model (level K 5 kg/min/m, Ti 20 min; composition K 1.5 kPa/%, Ti 10 min;
    pressure K 20 kg/min/kPa, Ti 10 min), discrete PI every 0.2 min with conditional integration;
  * first-order actuators for P100 and F200 (time constant 0.5 min) and for F2 (0.2 min), which allow valve
    stiction and gain loss; a 1 min analyzer delay and 1 min first-order lag on the composition measurement;
  * four stochastic disturbances (feed flow F1, feed composition X1, feed temperature T1, cooling-water inlet
    temperature T200) as first-order autoregressive processes, and Gaussian measurement noise on every recorded
    variable;
  * ten fault types (list FAULTS), each with a magnitude drawn per run index from U(0.6, 1.4) times a nominal size
    in P; the fault blocks are written here, the source defines no faults;
  * interlocks on the level, the pressure and the composition; no run of the generated data trips them.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: minute. Sampling period 1 min, control interval
0.2 min, explicit Euler step 0.1 min.
"""
import os
import numpy as np

NAME = 'evap'
VARS = ['L2', 'X2', 'P2', 'F1', 'X1', 'T1', 'T200', 'F2', 'F3', 'F100', 'P100', 'F200', 'T2', 'T3', 'T201',
        'uP100', 'uF200']
FAULTS = {1: 'feed flow step (disturbance)', 2: 'feed composition step (disturbance)',
          3: 'cooling-water inlet temperature step (disturbance)', 4: 'evaporator heat-transfer fouling (process)',
          5: 'condenser heat-transfer fouling (process)', 6: 'circulation pump degradation (process)',
          7: 'steam valve stiction (actuator)', 8: 'cooling-water valve gain loss (actuator)',
          9: 'product composition analyzer bias (sensor, in loop)', 10: 'operating pressure sensor drift (sensor, in loop)'}
SAMPLE_PERIOD = '1 min'
P = dict(F1_0=10.0, X1_0=5.0, T1_0=40.0, F2_0=2.0, X2sp=25.0, L2sp=1.0, P2sp=50.5, F3_0=50.0, P100_0=194.7,
         F200_0=208.0, T200_0=25.0, UA2=6.84, C=0.07, lam=38.5, lams=36.6,
         KL=5.0, TiL=20.0, KX=1.5, TiX=10.0, KP=20.0, TiP=10.0, tau_s=0.5, tau_c=0.5, tau_p=0.2, ana_delay=1.0, ana_lag=1.0,
         P100max=400.0, F200max=400.0, F2max=4.0,
         d_sig=(0.15, 0.08, 1.0, 0.5), d_tau=(10.0, 20.0, 20.0, 30.0),                     # F1, X1, T1, T200
         m_sig=(0.01, 0.05, 0.1, 0.05, 0.02, 0.2, 0.1, 0.02, 0.2, 0.05, 0.3, 1.0, 0.1, 0.1, 0.1, 0.0, 0.0),
         f1_step=0.4, f2_step=0.25, f3_step=1.5, f4_loss=0.04, f5_loss=0.05, f6_loss=0.06, f7_band=8.0,
         f8_loss=0.3, f9_bias=1.0, f10_rate=0.008,
         L_hi=2.5, L_lo=0.1, P_hi=80.0, P_lo=20.0, X_hi=60.0, X_lo=5.0)
DT = 0.1; NSUB = 2; NCTL = 5; BURN = 200
NF_ = len(FAULTS); BASE = 20261012
NDX = int(round(P['ana_delay'] / (DT * NSUB)))              # analyzer delay in control steps


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = (BURN + T) * NCTL
    return g.standard_normal((nc, 4)).astype(np.float32), g.standard_normal((nc, 3)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(4).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def algebra(L2, X2, P2, F1, X1, T1, F3, P100, F200, T200, ua1f, ua2f):
    T2 = 0.5616 * P2 + 0.3126 * X2 + 48.43; T3 = 0.507 * P2 + 55.0; T100 = 0.1538 * P100 + 90.0
    UA1 = 0.16 * (F1 + F3) * ua1f; Q100 = UA1 * (T100 - T2); F100 = Q100 / P['lams']
    F4 = (Q100 - P['C'] * F1 * (T2 - T1)) / P['lam']
    UA2 = P['UA2'] * ua2f; F200 = np.maximum(F200, 1.0)
    Q200 = UA2 * (T3 - T200) / (1 + UA2 / (2 * P['C'] * F200)); T201 = T200 + Q200 / (P['C'] * F200); F5 = Q200 / P['lam']
    return T2, T3, F100, F4, F5, T201


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids); T and onset in samples. Returns X (n, T, 17) float32 and shutdown_sample."""
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
    dist = np.stack([a[0] for a in NS]); ctl = np.stack([a[1] for a in NS]); meas = np.stack([a[2] for a in NS]); d0 = np.stack([a[3] for a in NS])
    act = np.zeros((n, NF_ + 1), bool); mag = np.zeros((n, NF_ + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    L2 = np.full(n, P['L2sp']); X2 = np.full(n, P['X2sp']); P2 = np.full(n, P['P2sp'])
    aP = np.full(n, P['P100_0']); aF = np.full(n, P['F200_0']); aF2 = np.full(n, P['F2_0'])     # actuator states
    uP = aP.copy(); uF = aF.copy(); uF2 = aF2.copy(); pP = uP.copy()                             # demands, stuck position
    IL = np.zeros(n); IX = np.zeros(n); IP = np.zeros(n)
    xlag = np.full(n, P['X2sp']); bufX = np.full((NDX, n), P['X2sp']); bp = 0
    dsig = np.array(P['d_sig']); dtc = DT * NSUB; phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    nc = int(round((BURN + T) / dtc)); per = int(round(1.0 / dtc))
    for k in range(nc):
        t = (k + 1) * dtc - BURN; tau = max(t - onset, 0.0); on = t > onset
        d = phi * d + dinn * dist[sidx, k].astype(np.float64)
        F1 = P['F1_0'] + d[:, 0]; X1 = P['X1_0'] + d[:, 1]; T1 = P['T1_0'] + d[:, 2]; T200 = P['T200_0'] + d[:, 3]
        F3 = np.full(n, P['F3_0']); ua1f = np.ones(n); ua2f = np.ones(n); gF = np.ones(n); bX = np.zeros(n); bP = np.zeros(n)
        if on:
            F1 = F1 + am[:, 1] * P['f1_step']; X1 = X1 + am[:, 2] * P['f2_step']; T200 = T200 + am[:, 3] * P['f3_step']
            ua1f = 1 - am[:, 4] * P['f4_loss']; ua2f = 1 - am[:, 5] * P['f5_loss']; F3 = F3 * (1 - am[:, 6] * P['f6_loss'])
            gF = 1 - am[:, 8] * P['f8_loss']; bX = am[:, 9] * P['f9_bias']; bP = am[:, 10] * P['f10_rate'] * tau
        # composition analyzer: first-order lag, then transport delay
        xlag = xlag + dtc / P['ana_lag'] * (X2 + bX - xlag)
        xd = bufX[bp].copy(); bufX[bp] = xlag; bp = (bp + 1) % NDX
        cn = ctl[sidx, k].astype(np.float64)
        Lm = L2 + msig[0] * cn[:, 0]; Xm = xd + msig[1] * cn[:, 1]; Pm = P2 + bP + msig[2] * cn[:, 2]
        eL = Lm - P['L2sp']; eX = P['X2sp'] - Xm; eP = Pm - P['P2sp']
        uF2 = P['F2_0'] + P['KL'] * (eL + IL); uP = P['P100_0'] + P['KX'] * (eX + IX); uF = P['F200_0'] + P['KP'] * (eP + IP)
        sL = ((uF2 >= P['F2max']) & (eL > 0)) | ((uF2 <= 0) & (eL < 0))
        sX = ((uP >= P['P100max']) & (eX > 0)) | ((uP <= 0) & (eX < 0))
        sP = ((uF >= P['F200max']) & (eP > 0)) | ((uF <= 0) & (eP < 0))
        IL = np.where(sL, IL, IL + eL * dtc / P['TiL']); IX = np.where(sX, IX, IX + eX * dtc / P['TiX']); IP = np.where(sP, IP, IP + eP * dtc / P['TiP'])
        uF2 = np.clip(P['F2_0'] + P['KL'] * (eL + IL), 0, P['F2max']); uP = np.clip(P['P100_0'] + P['KX'] * (eX + IX), 0, P['P100max'])
        uF = np.clip(P['F200_0'] + P['KP'] * (eP + IP), 0, P['F200max'])
        if on:
            stk = act[:, 7] & (np.abs(uP - pP) <= mag[:, 7] * P['f7_band']); pP = np.where(stk, pP, uP)
        else:
            pP = uP
        for _ in range(NSUB):
            aP = aP + DT / P['tau_s'] * (pP - aP); aF = aF + DT / P['tau_c'] * (gF * uF - aF); aF2 = aF2 + DT / P['tau_p'] * (uF2 - aF2)
            T2, T3, F100, F4, F5, T201 = algebra(L2, X2, P2, F1, X1, T1, F3, aP, aF, T200, ua1f, ua2f)
            L2 = L2 + DT * (F1 - F4 - aF2) / 20.0; X2 = X2 + DT * (F1 * X1 - aF2 * X2) / 20.0; P2 = P2 + DT * (F4 - F5) / 4.0
        tripped = tripped | (L2 > P['L_hi']) | (L2 < P['L_lo']) | (P2 > P['P_hi']) | (P2 < P['P_lo']) | (X2 > P['X_hi']) | (X2 < P['X_lo'])
        if tripped.any():
            L2 = np.where(tripped, P['L2sp'], L2); X2 = np.where(tripped, P['X2sp'], X2); P2 = np.where(tripped, P['P2sp'], P2)
        if (k + 1) * dtc > BURN and (k + 1) % per == 0:
            j = int(round((k + 1) * dtc - BURN)) - 1
            if j < 0 or j >= T: continue
            T2, T3, F100, F4, F5, T201 = algebra(L2, X2, P2, F1, X1, T1, F3, aP, aF, T200, ua1f, ua2f)
            mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([L2, xd, P2 + bP, F1, X1, T1, T200, aF2, F3, F100, aP, aF, T2, T3, T201, uP, uF], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
