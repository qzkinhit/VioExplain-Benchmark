"""Closed-loop counter-current shell-and-tube heat exchanger under outlet-temperature control, with concurrent faults
and paired runs.

Model. The lumped (cell) model of a counter-current shell-and-tube heat exchanger with wall storage, as derived in
process-dynamics textbooks (for example Luyben, Process Modeling, Simulation and Control for Chemical Engineers,
2nd ed. 1990, distributed heat-exchanger examples; Seborg, Edgar, Mellichamp and Doyle, Process Dynamics and
Control, heat-exchanger models). N = 8 cells; tube side (process water, flowing 1 -> N), wall, shell side (hot oil,
flowing N -> 1):
    m_t c_t dTt_i/dt = Fc c_t (Tt_{i-1} - Tt_i) + hA_t (Tw_i - Tt_i)
    m_w c_w dTw_i/dt = hA_s (Ts_i - Tw_i) - hA_t (Tw_i - Tt_i)
    m_s c_s dTs_i/dt = Fh c_s (Ts_{i+1} - Ts_i) + hA_s (Tw_i - Ts_i)
with film coefficients scaling with flow^0.8 (Dittus-Boelter). The model structure is the textbook one; all numerical
values are chosen here (no single published parameter set is followed): c_t 4.18, c_s 2.3 kJ/(kg K), m_t 2 kg,
m_s 6 kg, m_w c_w 5 kJ/K per cell, hA_t = hA_s = 1.125 kW/K per cell at nominal flow, Fc 2 kg/s, inlet 20 C, outlet
set point 60 C, hot oil 150 C, about 2.1 kg/s; tube-side pressure drop 40 kPa (Fc / Fc0)^2 (1 + 0.3 (1 - f_t)), f_t the tube-side fouling factor on hA_t.
Control: process flow loop (PI on the process valve), outlet temperature PI on the hot-oil valve (valve lags 2 s and
3 s, outlet thermometer with a 2 s first-order lag).
Disturbances and noise: first-order autoregressive process inlet temperature, hot-oil supply temperature and the two
supply pressures; Gaussian noise on the control measurements and on every recorded variable. Ten fault types (list
FAULTS), each with a magnitude drawn per run index from U(0.6, 1.4) times a nominal size in P; interlocks on the outlet
temperature.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: second. Sampling period 2 s, control interval 0.5 s,
explicit Euler step 0.1 s.
"""
import os
import numpy as np

NAME = 'hex'
VARS = ['Fc', 'Fh', 'Tc_in', 'Tc_out', 'Tc_mid', 'Th_in', 'Th_out', 'dP_tube', 'hot_valve', 'proc_valve']
FAULTS = {1: 'tube-side fouling, ramp (process)', 2: 'shell-side fouling, ramp (process)',
          3: 'hot-oil supply temperature drop (disturbance)', 4: 'process inlet temperature drop (disturbance)',
          5: 'hot-oil valve stiction (actuator)', 6: 'hot-oil valve gain loss (actuator)',
          7: 'outlet temperature sensor bias (sensor, in loop)', 8: 'process flow meter drift (sensor, in loop)',
          9: 'internal bypass leakage of process fluid (process)', 10: 'hot-oil outlet temperature sensor drift (sensor, out of loop)'}
SAMPLE_PERIOD = '2 s'
NC = 8
P = dict(ct=4.18, cs=2.3, mt=2.0, ms=6.0, mwcw=5.0, hAt=1.125, hAs=1.125, Fc0=2.0, Tcin=20.0, Tsp=60.0, Thin=150.0,
         dP0=40.0, Fcmax=4.0, Fhmax=5.0, vc_tau=2.0, vh_tau=3.0, th_tau=2.0,
         KC=8.0, TiC=4.0, KT=1.5, TiT=40.0,
         d_sig=(0.4, 1.0, 0.02, 0.02), d_tau=(120.0, 200.0, 30.0, 30.0),   # Tc_in, Th_in (C), hot-oil and process supply pressure factors
         c_sig=(0.01, 0.05),                                               # control measurement noise Fc (kg/s), Tc_out (C)
         m_sig=(0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.0, 0.0),
         f1_rate=2.5e-4, f2_rate=4.0e-4, f3_step=-3.0, f4_step=-1.5, f5_band=3.5, f6_loss=0.08, f7_bias=1.0,
         f8_rate=1.5e-4, f9_frac=0.025, f10_rate=4e-3, T_hi=80.0, T_lo=40.0)
DT = 0.1; NSUB = 5; NCTL = 4; BURN = 900
NF_ = len(FAULTS); BASE = 20261025


def steady_solve(Fh):
    """Steady temperatures for a given hot-oil flow (no fouling); returns Tt (NC), Tw, Ts."""
    Tt = np.linspace(25, 60, NC); Ts = np.linspace(80, 150, NC); Tw = 0.5 * (Tt + Ts)
    for _ in range(6000):
        Tt_up = np.concatenate([[P['Tcin']], Tt[:-1]]); Ts_up = np.concatenate([Ts[1:], [P['Thin']]])
        Tt = Tt + 0.2 * (P['Fc0'] * P['ct'] * (Tt_up - Tt) + P['hAt'] * (Tw - Tt)) / (P['mt'] * P['ct'])
        Tw = Tw + 0.2 * (P['hAs'] * (Ts - Tw) - P['hAt'] * (Tw - Tt)) / P['mwcw']
        Ts = Ts + 0.2 * (Fh * P['cs'] * (Ts_up - Ts) + P['hAs'] * (Tw - Ts)) / (P['ms'] * P['cs'])
    return Tt, Tw, Ts


def steady():
    lo, hi = 0.5, 4.9
    for _ in range(40):
        mid = 0.5 * (lo + hi); Tt, Tw, Ts = steady_solve(mid)
        if Tt[-1] > P['Tsp']: hi = mid
        else: lo = mid
    Tt, Tw, Ts = steady_solve(0.5 * (lo + hi)); return 0.5 * (lo + hi), Tt, Tw, Ts


FH0, TT0, TW0, TS0 = steady()


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = int((BURN + T * 2) / (DT * NSUB)) + 2
    return g.standard_normal((nc, 6)).astype(np.float32), g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(4).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids); T and onset in samples. Returns X (n, T, 10) float32 and shutdown_sample."""
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
    dist = np.stack([a[0] for a in NS]); meas = np.stack([a[1] for a in NS]); d0 = np.stack([a[2] for a in NS])
    act = np.zeros((n, NF_ + 1), bool); mag = np.zeros((n, NF_ + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    Tt = np.tile(TT0, (n, 1)); Tw = np.tile(TW0, (n, 1)); Ts = np.tile(TS0, (n, 1))
    vc0 = 100 * P['Fc0'] / P['Fcmax']; vh0 = 100 * FH0 / P['Fhmax']
    xc = np.full(n, vc0); xh = np.full(n, vh0); ph = np.full(n, vh0); vc = np.full(n, vc0); vh = np.full(n, vh0)
    Tth = np.full(n, P['Tsp']); IC = np.zeros(n); IT = np.zeros(n)
    dtc = DT * NSUB; dsig = np.array(P['d_sig']); phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig']); csig = np.array(P['c_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    spp = 2.0; nc = int((BURN + T * spp) / dtc); per = int(round(spp / dtc))
    Fc = np.full(n, P['Fc0']); Fh = np.full(n, FH0); ct, cs = P['ct'], P['cs']
    for k in range(nc):
        t = ((k + 1) * dtc - BURN) / spp; tau = max(t - onset, 0.0) * spp; on = t > onset
        dk = dist[sidx, k].astype(np.float64); d = phi * d + dinn * dk[:, :4]; cn = csig * dk[:, 4:]
        Tcin = P['Tcin'] + d[:, 0]; Thin = P['Thin'] + d[:, 1]; psh = 1 + d[:, 2]; psc = 1 + d[:, 3]
        ft = np.ones(n); fs_ = np.ones(n); gh = np.ones(n); bT = np.zeros(n); bF = np.zeros(n); byp = np.zeros(n); bTh = np.zeros(n)
        if on:
            ft = np.maximum(1 - am[:, 1] * P['f1_rate'] * tau, 0.4); fs_ = np.maximum(1 - am[:, 2] * P['f2_rate'] * tau, 0.4)
            Thin = Thin + am[:, 3] * P['f3_step']; Tcin = Tcin + am[:, 4] * P['f4_step']; gh = 1 - am[:, 6] * P['f6_loss']
            bT = am[:, 7] * P['f7_bias']; bF = am[:, 8] * P['f8_rate'] * tau; byp = am[:, 9] * P['f9_frac']; bTh = am[:, 10] * P['f10_rate'] * tau
        Tout = (1 - byp) * Tt[:, -1] + byp * Tcin
        Fcm = Fc + bF + cn[:, 0]; Tm = Tth + bT + cn[:, 1]
        ec = P['Fc0'] - Fcm; vc = vc0 + P['KC'] * (ec + IC); sc = ((vc >= 100) & (ec > 0)) | ((vc <= 0) & (ec < 0))
        IC = np.where(sc, IC, IC + ec * dtc / P['TiC']); vc = np.clip(vc0 + P['KC'] * (ec + IC), 0, 100)
        et = P['Tsp'] - Tm; vh = vh0 + P['KT'] * (et + IT); sh = ((vh >= 100) & (et > 0)) | ((vh <= 0) & (et < 0))
        IT = np.where(sh, IT, IT + et * dtc / P['TiT']); vh = np.clip(vh0 + P['KT'] * (et + IT), 0, 100)
        if on:
            s5 = act[:, 5] & (np.abs(vh - ph) <= mag[:, 5] * P['f5_band']); ph = np.where(s5, ph, vh)
        else:
            ph = vh
        for _ in range(NSUB):
            xc = xc + DT / P['vc_tau'] * (vc - xc); xh = xh + DT / P['vh_tau'] * (ph * gh - xh)
            Fc = P['Fcmax'] * xc / 100 * np.sqrt(np.maximum(psc, 0.0)); Fh = P['Fhmax'] * xh / 100 * np.sqrt(np.maximum(psh, 0.0))
            Fb = Fc * (1 - byp)
            hAt = P['hAt'] * ft[:, None] * (np.maximum(Fb, 1e-3) / P['Fc0'])[:, None] ** 0.8
            hAs = P['hAs'] * fs_[:, None] * (np.maximum(Fh, 1e-3) / FH0)[:, None] ** 0.8
            Tt_up = np.concatenate([Tcin[:, None], Tt[:, :-1]], 1); Ts_up = np.concatenate([Ts[:, 1:], Thin[:, None]], 1)
            qt = hAt * (Tw - Tt); qs = hAs * (Ts - Tw)
            Tt = Tt + DT * (Fb[:, None] * ct * (Tt_up - Tt) + qt) / (P['mt'] * ct)
            Tw = Tw + DT * (qs - qt) / P['mwcw']
            Ts = Ts + DT * (Fh[:, None] * cs * (Ts_up - Ts) - qs) / (P['ms'] * cs)
            Tout = (1 - byp) * Tt[:, -1] + byp * Tcin
            Tth = Tth + DT / P['th_tau'] * (Tout - Tth)
        tripped = tripped | (Tout > P['T_hi']) | (Tout < P['T_lo']) | ~np.isfinite(Tout)
        if tripped.any():
            Tt = np.where(tripped[:, None], TT0, Tt); Tw = np.where(tripped[:, None], TW0, Tw); Ts = np.where(tripped[:, None], TS0, Ts)
        if (k + 1) * dtc > BURN and (k + 1) % per == 0:
            j = int(round(((k + 1) * dtc - BURN) / spp)) - 1
            if j < 0 or j >= T: continue
            mn = meas[sidx, j].astype(np.float64) * msig
            dP = P['dP0'] * (Fc / P['Fc0']) ** 2 * (1 + 0.3 * (1 - ft))
            row = np.stack([Fc + bF, Fh, Tcin, Tth + bT, Tt[:, NC // 2 - 1], Thin, Ts[:, 0] + bTh, dP, vh, vc], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
