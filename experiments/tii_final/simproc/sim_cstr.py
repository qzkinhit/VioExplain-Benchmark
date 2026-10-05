"""Closed-loop continuous stirred tank reactor (CSTR) with concurrent faults and paired runs.

Model. Exothermic first-order reaction A -> B in a jacketed reactor, three balances of the classical non-isothermal
CSTR (reactant concentration C, reactor temperature T, jacket temperature Tc) in the form used by Pilario and Cao
(IEEE TII 2018) for incipient faults, with the multiplicative catalyst activity a and heat-transfer efficiency b:
    dV/dt  = Qf - Qo
    dC/dt  = Qf / V (Ci - C) - a k0 exp(-E/(R T)) C
    dT/dt  = Qf / V (Ti - T) + a (-dH) k0 exp(-E/(R T)) C / (rho Cp) - b UA / (rho Cp V) (T - Tc)
    dTc/dt = Qc / Vc (Tci - Tc) + b UA / (rhoc Cpc Vc) (T - Tc)
Parameter values (Q 100 L/min, V 150 L, Vc 10 L, dH -2e5 cal/mol, UA 7e5 cal/(min K), k0 7.2e10 1/min, E/R 1e4 K,
rho Cp 1000 cal/(L K), Ci 1 mol/L, Ti 350 K, Tci 350 K, T set point 430 K) are the ones commonly quoted for this
benchmark. They were written from memory and not checked against the paper, so this file does not reproduce a
published data set. Deviations from the published model, all introduced here:
  * a liquid-level balance with a PI level loop acting on the outlet valve (the level is a state, UA does not depend
    on it), and a cascade for the reactor temperature (outer PI gives the jacket temperature set point, inner PI acts
    on the coolant valve), with controller settings chosen here from a linear stability scan of this model;
  * first-order valve actuators, a stick-slip valve model for the stiction fault, flow sensors on both valves;
  * four stochastic disturbances (inlet concentration, inlet temperature, coolant inlet temperature, feed flow) as
    first-order autoregressive processes updated every control interval, Gaussian measurement noise;
  * twelve fault types of this benchmark design (list FAULTS), each with a magnitude drawn per run index;
  * interlocks (reactor temperature or level outside limits) that trip the plant, after which the record is frozen.
Common random numbers. Every random number of a run is a function of the run index (seed) only: disturbance
innovations, measurement noise and the magnitude of each fault type. Runs with the same seed and different fault
sets therefore share the same noise and disturbance streams, and a fault has the same magnitude in every fault set
that contains it. Time unit: minute. Sampling period 1 min, control interval 0.05 min, integration step 0.01 min.
"""
import numpy as np

NAME = 'cstr'
VARS = ['Ci', 'Ti', 'Tci', 'Qf', 'C', 'T', 'Tc', 'h', 'Qc', 'Qo', 'Tcsp', 'uT', 'uh']
FAULTS = {1: 'catalyst decay (process, exponential)', 2: 'heat-transfer fouling (process, exponential)',
          3: 'inlet concentration step (disturbance)', 4: 'inlet temperature step (disturbance)',
          5: 'coolant inlet temperature step (disturbance)', 6: 'coolant inlet temperature variability (disturbance)',
          7: 'reactor temperature sensor bias (sensor, in loop)', 8: 'concentration sensor drift (sensor)',
          9: 'level sensor drift (sensor, in loop)', 10: 'coolant valve stiction (actuator)',
          11: 'outlet valve gain loss (actuator)', 12: 'feed flow step (disturbance)'}
SAMPLE_PERIOD = '1 min'
P = dict(Qf0=100.0, V0=150.0, Vc=10.0, dH=2.0e5, UA=7.0e5, k0=7.2e10, EoR=1.0e4, rcp=1000.0, Ci0=1.0, Ti0=350.0, Tci0=350.0,
         Tsp=430.0, hsp=100.0, QCMAX=300.0, QOMAX=200.0, tauv=0.05,
         KT=3.0, TiT=4.0, KI=4.0, TiI=0.3, KL=2.0, TiL=8.0,      # outer PI (K per K, min), inner PI (valve % per K, min), level PI
         d_sig=(0.010, 0.6, 0.8, 1.0), d_tau=(20.0, 30.0, 30.0, 10.0),   # disturbances Ci, Ti, Tci, Qf: std and correlation time
         m_sig=(0.004, 0.15, 0.15, 0.4, 0.0015, 0.05, 0.10, 0.15, 0.6, 0.4, 0.0, 0.0, 0.0),   # measurement noise per variable
         f1_rate=0.0010, f2_rate=0.0008, f3_step=0.06, f4_step=-3.0, f5_step=5.0, f6_mult=4.0, f7_bias=2.0, f8_rate=0.0005,
         f9_rate=0.02, f10_band=4.0, f11_loss=0.25, f12_step=6.0,
         T_hi=445.0, T_lo=405.0, h_hi=130.0, h_lo=70.0)
DT = 0.01; NSUB = 5; NCTL = 20; BURN = 120
NF = len(FAULTS); BASE = 20261005


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = (BURN + T) * NCTL
    return g.standard_normal((nc, 4)).astype(np.float32), g.standard_normal((nc, 3)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(4).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def steady():
    """Nominal steady state at the temperature set point: C, Tc, Qc and the valve positions."""
    k = P['k0'] * np.exp(-P['EoR'] / P['Tsp']); C = P['Ci0'] / (1 + k * P['V0'] / P['Qf0'])
    q = P['dH'] * k * C * P['V0'] - P['rcp'] * P['Qf0'] * (P['Tsp'] - P['Ti0']); Tc = P['Tsp'] - q / P['UA']
    Qc = q / (P['rcp'] * (Tc - P['Tci0']))
    return C, Tc, Qc, 100 * Qc / P['QCMAX'], 100 * P['Qf0'] / P['QOMAX']


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids). Returns X (n, T, 13) float32 and shutdown_sample (n,) (-1: no trip)."""
    n = len(runs); seeds = sorted(set(int(s) for s, _ in runs)); pos = {s: i for i, s in enumerate(seeds)}
    sidx = np.array([pos[int(s)] for s, _ in runs]); NS = [noise_streams(s, T) for s in seeds]
    dist = np.stack([a[0] for a in NS]); ctl = np.stack([a[1] for a in NS]); meas = np.stack([a[2] for a in NS]); d0 = np.stack([a[3] for a in NS])
    act = np.zeros((n, NF + 1), bool); mag = np.zeros((n, NF + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    C0, Tc0, Qc0, uT0, uh0 = steady()
    V = np.full(n, P['V0']); C = np.full(n, C0); Tr = np.full(n, P['Tsp']); Tc = np.full(n, Tc0)
    xc = np.full(n, uT0); xo = np.full(n, uh0); pc = np.full(n, uT0); IT = np.zeros(n); II = np.zeros(n); IL = np.zeros(n)
    dsig = np.array(P['d_sig']); phi = np.exp(-(DT * NSUB) / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx] * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    dtc = DT * NSUB; kap = P['UA'] / (P['rcp'] * P['Vc']); nc = (BURN + T) * NCTL
    for k in range(nc):
        t = (k + 1) * dtc - BURN; tau = max(t - onset, 0.0); on = t > onset
        e = dist[sidx, k].astype(np.float64); mult = np.ones((n, 4))
        if on: mult[:, 2] = np.where(act[:, 6], P['f6_mult'], 1.0)
        d = phi * d + dinn * mult * e
        Ci = P['Ci0'] + d[:, 0]; Ti = P['Ti0'] + d[:, 1]; Tci = P['Tci0'] + d[:, 2]; Qf = P['Qf0'] + d[:, 3]
        a = np.ones(n); b = np.ones(n); go = np.ones(n); bT = np.zeros(n); bC = np.zeros(n); bh = np.zeros(n)
        if on:
            Ci = Ci + am[:, 3] * P['f3_step']; Ti = Ti + am[:, 4] * P['f4_step']
            Tci = Tci + am[:, 5] * P['f5_step']; Qf = Qf + am[:, 12] * P['f12_step']
            a = np.exp(-am[:, 1] * P['f1_rate'] * tau); b = np.exp(-am[:, 2] * P['f2_rate'] * tau)
            bT = am[:, 7] * P['f7_bias']; bC = am[:, 8] * P['f8_rate'] * tau
            bh = am[:, 9] * P['f9_rate'] * tau; go = 1 - am[:, 11] * P['f11_loss']
        cn = ctl[sidx, k].astype(np.float64)
        Tm = Tr + bT + msig[5] * cn[:, 0]; hm = 100 * V / P['V0'] + bh + msig[7] * cn[:, 1]
        Tcm = Tc + msig[6] * cn[:, 2]
        eT = Tm - P['Tsp']; Tcsp = Tc0 - P['KT'] * (eT + IT); eI = Tcm - Tcsp; uT = uT0 + P['KI'] * (eI + II)
        sat = ((uT >= 100) & (eI > 0)) | ((uT <= 0) & (eI < 0))          # conditional integration in both loops (anti-windup)
        IT = np.where(sat, IT, IT + eT * dtc / P['TiT']); II = np.where(sat, II, II + eI * dtc / P['TiI'])
        Tcsp = Tc0 - P['KT'] * (eT + IT); uT = np.clip(uT0 + P['KI'] * (Tcm - Tcsp + II), 0, 100)
        eL = hm - P['hsp']; uh = uh0 + P['KL'] * (eL + IL); satL = ((uh >= 100) & (eL > 0)) | ((uh <= 0) & (eL < 0))
        IL = np.where(satL, IL, IL + eL * dtc / P['TiL']); uh = np.clip(uh0 + P['KL'] * (eL + IL), 0, 100)
        if on:
            stick = act[:, 10] & (np.abs(uT - pc) <= mag[:, 10] * P['f10_band']); pc = np.where(stick, pc, uT)
        else:
            pc = uT
        for _ in range(NSUB):
            xc = xc + DT / P['tauv'] * (pc - xc); xo = xo + DT / P['tauv'] * (uh - xo)
            Qc = P['QCMAX'] * xc / 100; Qo = P['QOMAX'] * go * xo / 100
            kr = a * P['k0'] * np.exp(-P['EoR'] / Tr)
            dV = Qf - Qo; dC = Qf / V * (Ci - C) - kr * C
            dT = Qf / V * (Ti - Tr) + P['dH'] / P['rcp'] * kr * C - b * P['UA'] / (P['rcp'] * V) * (Tr - Tc)
            Tc = (Tc + DT * (Qc / P['Vc'] * Tci + b * kap * Tr)) / (1 + DT * (Qc / P['Vc'] + b * kap))
            V = V + DT * dV; C = np.maximum(C + DT * dC, 0.0); Tr = Tr + DT * dT
        hl = 100 * V / P['V0']
        tripped = tripped | (Tr > P['T_hi']) | (Tr < P['T_lo']) | (hl > P['h_hi']) | (hl < P['h_lo'])
        if tripped.any():   # the record is frozen after a trip; the states are reset only to keep them finite
            Tr = np.where(tripped, P['Tsp'], Tr); V = np.where(tripped, P['V0'], V); C = np.where(tripped, C0, C); Tc = np.where(tripped, Tc0, Tc)
        if k >= BURN * NCTL and (k + 1) % NCTL == 0:
            j = (k + 1) // NCTL - BURN - 1; mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([Ci, Ti, Tci, Qf, C + bC, Tr + bT, Tc, hl + bh, Qc, Qo, Tcsp, uT, uh], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)      # first frozen sample
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
