"""Closed-loop continuous stirred tank heater (CSTH) with concurrent faults and paired runs.

Model. The stirred tank heater of Thornhill, Patwardhan and Shah (J. Process Control 18 (2008) 347-360), following
the Simulink model distributed with the paper (CSTHSimulation.zip, Zenodo record 10093059: CSTHMaster.mdl,
CSTHDisturbedStdOp*.mdl, index.htm), standard operating point 2 (hot-water valve 5.5 mA, level 12 mA = 20.475 cm,
temperature 10.5 mA = 42.517 C, cold-water flow 3.8226e-5 m3/s, cold water 24 C, hot water 50 C). Taken from the
distributed model: the volume-level table of the tank, the outflow law f_out = 1e-4 (0.1013 sqrt(55 + h) + 0.0237)
m3/s with h in cm, the cold-water valve (1 s dead time, first-order lag 3.8 s, valve-signal-to-flow and
valve-signal-to-flow-measurement tables), the hot-water valve table, the steam valve to heat input table (kW), the
level and temperature mA tables, the 8 s delay of the temperature measurement (0.5 s plus 7.5 s), and the three PI
loops in mA (level P 2, I 0.1 per s, output = cold-water flow set point; cold-water flow P 0.5, I 0.2 per s;
temperature P 3, I 0.1 per s on the steam valve), hot-water valve in manual. Balances:
    dV/dt = f_cw + f_hw - f_out(h) - f_leak(h),   V dT/dt = f_cw (T_cw - T) + f_hw (T_hw - T) + Q_st / (rho c_p)
Deviations from the distributed model, all introduced here:
  * constant rho c_p = 4.18e6 J/(m3 K) instead of the enthalpy and density tables (the steady temperature at operating
    point 2 is 42.48 C instead of 42.517 C; the burn-in absorbs the difference);
  * the recorded noise sequences of the model (level, cold-water flow, temperature, in mA) are replaced by first-order
    autoregressive sequences with the same standard deviations (0.0288, 0.0851, 0.0334 mA) and correlation times of
    3, 2 and 1 s chosen here;
  * disturbances absent from the distributed model and added here: hot-water supply temperature, cold-water supply
    temperature and steam supply pressure as first-order autoregressive processes; a hot-water flow meter and a
    hot-water temperature sensor with Gaussian noise (both not in the distributed model);
  * ten fault types (list FAULTS), each with a magnitude drawn per run index from U(0.6, 1.4) times a nominal size;
    the fault blocks (stiction, sensor bias, leak, supply faults) are written here, since the distributed files contain
    none.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: second. Sampling period 1 s, control interval 0.25 s,
explicit Euler step 0.125 s. Recorded signals are in mA as in the distributed model, except the two added sensors.
"""
import os
import numpy as np

NAME = 'csth'
VARS = ['level_mA', 'T_mA', 'Fcw_mA', 'Fhw_Lmin', 'Thw_C', 'LC_out_mA', 'CWvalve_mA', 'STvalve_mA']
FAULTS = {1: 'steam valve stiction (actuator)', 2: 'cold-water valve stiction (actuator)',
          3: 'level sensor bias (sensor, in loop)', 4: 'temperature sensor bias (sensor, in loop)',
          5: 'cold-water flow sensor bias (sensor, inner loop)', 6: 'steam supply pressure drop (disturbance)',
          7: 'steam supply pressure oscillation (disturbance)', 8: 'tank leak (process)',
          9: 'hot-water supply temperature drop (disturbance)', 10: 'hot-water valve signal offset, more hot water (actuator)'}
SAMPLE_PERIOD = '1 s'
VOL = 1e-3 * np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 2.75, 3, 4, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5])
LEV = np.array([1.0, 1.6, 3.2, 4.8, 6.5, 8.3, 10.0, 11.8, 13.5, 15.3, 16.9, 18.5, 24.9, 31.2, 32.8, 34.3, 35.9, 37.5, 39.1, 40.7, 42.4, 44.0, 45.6, 47.3])
LEV_MA = np.array([3.559, 3.883, 4.551, 5.293, 6.016, 6.77, 7.539, 8.285, 9.027, 9.797, 10.49, 11.17, 13.86, 16.54, 17.21, 17.88, 18.54, 19.24, 19.96, 20.63, 21.36, 22.05, 22.75, 23.47])
CW_MA = np.array([0, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24.]); CW_F = np.array([0, 0, 1.9e-5, 2.8e-5, 4.0e-5, 5.6e-5, 7.7e-5, 1.05e-4, 1.38e-4, 1.75e-4, 2.08e-4, 2.08e-4])
CWM_MA = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 21, 24.]); CWM_OUT = np.array([4.34, 4.335, 4.335, 5.71, 6.32, 6.925, 7.5, 8.075, 8.87, 9.78, 11.94, 14.68, 19.53, 24.2, 24.2])
HW_MA = np.array([0, 4, 5, 6, 7, 8, 9, 10, 12.]); HW_F = 1e-5 * np.array([0, 0, 3.07, 7.36, 11.29, 14.88, 17.87, 20.56, 23.91])
ST_MA = np.array([0, 4, 7.5, 9, 11, 14, 17, 20, 25.]); ST_KW = np.array([0, 0, 2.24, 2.61, 4.65, 8.89, 13.60, 15.04, 15.04])
T_C = np.array([24, 30, 31, 36.5, 44.5, 48, 61, 65.]); T_MA = np.array([7.785, 8.61, 8.76, 9.59, 10.80, 11.38, 13.52, 14.20])
P = dict(rcp=4.18e6, Tcw0=24.0, Thw0=50.0, hw_mA=5.5, Lsp=12.0, Tsp=10.5, ucw0=7.7043, ust0=6.0532, uL0=7.330, T0=42.517, h0=20.475,
         KL=2.0, IL=0.1, KF=0.5, IF=0.2, KT=3.0, IT=0.1, cw_delay=1.0, cw_tau=3.8, T_delay=8.0,
         n_sig=(0.0288, 0.0851, 0.0334), n_tau=(3.0, 2.0, 1.0),                  # level, cold-water flow, temperature noise (mA)
         d_sig=(0.3, 0.2, 0.02), d_tau=(120.0, 300.0, 30.0),                     # added: T_hw, T_cw, steam pressure factor
         m_sig=(0.0, 0.0, 0.0, 0.05, 0.1, 0.0, 0.0, 0.0),
         f1_band=0.4, f2_band=0.5, f3_bias=0.8, f4_bias=0.25, f5_bias=0.3, f6_drop=0.3, f7_amp=0.4, f7_period=60.0,
         f8_leak=0.05, f9_step=-4.0, f10_off=0.25, V_hi=7.3e-3, V_lo=0.5e-3)
DT = 0.125; NSUB = 2; NCTL = 4; BURN = 600
NF_ = len(FAULTS); BASE = 20261007
NDCW = int(round(P['cw_delay'] / (DT * NSUB))); NDT = int(round(P['T_delay'] / (DT * NSUB)))


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = (BURN + T) * NCTL
    return g.standard_normal((nc, 6)).astype(np.float32), g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(6).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def fout(hcm): return 1e-4 * (0.1013 * np.sqrt(55 + hcm) + 0.0237)


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids). Returns X (n, T, len(VARS)) float32 and shutdown_sample (n,)."""
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
    V = np.full(n, float(np.interp(P['h0'], LEV, VOL))); Tt = np.full(n, P['T0'])
    xcw = np.full(n, P['ucw0']); pcw = np.full(n, P['ucw0']); pst = np.full(n, P['ust0'])
    bcw = np.full((NDCW, n), P['ucw0']); bT = np.full((NDT, n), P['T0']); bpc = 0; bpt = 0
    Il = np.zeros(n); If = np.zeros(n); It = np.zeros(n)
    dtc = DT * NSUB; sig = np.array(P['n_sig'] + P['d_sig']); tau_ = np.array(P['n_tau'] + P['d_tau'])
    phi = np.exp(-dtc / tau_); dinn = sig * np.sqrt(1 - phi ** 2); d = d0[sidx].astype(np.float64) * sig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    nc = (BURN + T) * NCTL; V0 = float(np.interp(P['h0'], LEV, VOL))
    for k in range(nc):
        t = (k + 1) * dtc - BURN; tau = max(t - onset, 0.0); on = t > onset
        d = phi * d + dinn * dist[sidx, k].astype(np.float64)
        nL, nF, nT = d[:, 0], d[:, 1], d[:, 2]; Thw = P['Thw0'] + d[:, 3]; Tcw = P['Tcw0'] + d[:, 4]; ps = 1 + d[:, 5]
        hw = np.full(n, P['hw_mA']); bl = np.zeros(n); bt = np.zeros(n); bf = np.zeros(n); lk = np.zeros(n)
        if on:
            ps = ps * (1 - am[:, 6] * P['f6_drop']) * (1 + am[:, 7] * P['f7_amp'] * np.sin(2 * np.pi * tau / P['f7_period']))
            Thw = Thw + am[:, 9] * P['f9_step']; hw = hw + am[:, 10] * P['f10_off']
            bl = am[:, 3] * P['f3_bias']; bt = am[:, 4] * P['f4_bias']; bf = am[:, 5] * P['f5_bias']; lk = am[:, 8] * P['f8_leak']
        # measurements (mA) at the control instant; the temperature is the tank temperature 8 s earlier
        hcm = np.interp(V, VOL, LEV); Lm = np.interp(hcm, LEV, LEV_MA) + nL + bl
        Tdel = bT[bpt]; Tm = np.interp(Tdel, T_C, T_MA) + nT + bt
        Fm = np.interp(xcw + nF, CWM_MA, CWM_OUT) + bf
        # level PI -> cold-water flow set point (mA), flow PI -> cold-water valve, temperature PI -> steam valve
        el = P['Lsp'] - Lm; Il = Il + el * dtc; uL = np.clip(P['uL0'] + P['KL'] * el + P['IL'] * Il, 0, 25)
        ef = uL - Fm; If = If + ef * dtc; ucw = np.clip(P['ucw0'] + P['KF'] * ef + P['IF'] * If, 0, 25)
        et = P['Tsp'] - Tm; It = It + et * dtc; ust = np.clip(P['ust0'] + P['KT'] * et + P['IT'] * It, 0, 25)
        if on:
            s1 = act[:, 1] & (np.abs(ust - pst) <= mag[:, 1] * P['f1_band']); pst = np.where(s1, pst, ust)
            s2 = act[:, 2] & (np.abs(ucw - pcw) <= mag[:, 2] * P['f2_band']); pcw = np.where(s2, pcw, ucw)
        else:
            pst = ust; pcw = ucw
        ucw_del = bcw[bpc].copy(); bcw[bpc] = pcw; bpc = (bpc + 1) % NDCW     # 1 s dead time of the cold-water valve
        Q = 1000 * np.interp(pst, ST_MA, ST_KW) * np.maximum(ps, 0.0); fhw = np.interp(hw, HW_MA, HW_F)
        for _ in range(NSUB):
            xcw = xcw + DT / P['cw_tau'] * (ucw_del - xcw)
            fcw = np.interp(xcw + nF, CW_MA, CW_F); hcm = np.interp(V, VOL, LEV); fo = fout(hcm); fl = lk * fo
            dT = (fcw * (Tcw - Tt) + fhw * (Thw - Tt) + Q / P['rcp']) / np.maximum(V, 1e-4)
            V = V + DT * (fcw + fhw - fo - fl); Tt = Tt + DT * dT
        bT[bpt] = Tt; bpt = (bpt + 1) % NDT
        tripped = tripped | (V > P['V_hi']) | (V < P['V_lo']) | ~np.isfinite(Tt)
        if tripped.any():
            V = np.where(tripped, V0, V); Tt = np.where(tripped, P['T0'], Tt)
        if k >= BURN * NCTL and (k + 1) % NCTL == 0:
            j = (k + 1) // NCTL - BURN - 1; mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([Lm, Tm, Fm, 60000 * fhw, Thw, uL, pcw, pst], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
