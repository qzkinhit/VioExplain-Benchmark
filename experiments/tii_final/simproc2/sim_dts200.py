"""Closed-loop three-tank system (Amira DTS200 benchmark) with concurrent faults and paired runs.

Model. Three cylindrical tanks T1, T3, T2 in series (cross-section A = 0.0154 m2, connecting pipes of cross-section
Sn = 5e-5 m2, outflow coefficients mu13 = mu32 = 0.5 and mu20 = 0.6), pump 1 feeding T1 and pump 2 feeding T2
(at most 1e-4 m3/s each), nominal outflow from T2, Torricelli flows q = mu Sn sgn(dh) sqrt(2 g |dh|):
    A dh1/dt = Q1 - q13 - q1L
    A dh3/dt = q13 - q32 - q3L
    A dh2/dt = Q2 + q32 - q20 - q2L
These are the DTS200 equations and parameter values as quoted in the fault-diagnosis literature (for example the
COSY three-tank benchmark); the values were written from memory and not checked against the Amira manual. Levels h1
and h2 are controlled by two decentralized PI loops on the pumps (set points 0.45 m and 0.25 m, so h3 = 0.35 m).
Leaks q_iL are flows through an extra opening at the tank bottom; clogging lowers an outflow coefficient.
Deviations, all introduced here: the PI settings (chosen on this model); first-order autoregressive disturbances on
both pump deliveries and on the outflow coefficient of T2; Gaussian measurement noise; a linear regularization of the
square root for level differences below 1 mm; ten fault types (list FAULTS), each with a magnitude drawn per run
index from U(0.6, 1.4) times a nominal size; overflow and dry-tank interlocks.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: second. Sampling period 2 s, control interval 0.5 s,
explicit Euler step 0.25 s. Flows are recorded in mL/s.
"""
import os
import numpy as np

NAME = 'dts200'
VARS = ['h1', 'h2', 'h3', 'Q1', 'Q2', 'u1', 'u2']
FAULTS = {1: 'leak in tank 1 (process)', 2: 'leak in tank 2 (process)', 3: 'leak in tank 3 (process)',
          4: 'clogging of the pipe between tanks 1 and 3 (process)', 5: 'clogging of the pipe between tanks 3 and 2 (process)',
          6: 'clogging of the outflow pipe of tank 2 (process)', 7: 'pump 1 gain loss (actuator)',
          8: 'pump 2 delivery offset (actuator)', 9: 'level sensor 1 bias (sensor, in loop)',
          10: 'level sensor 3 drift (sensor, out of loop)'}
SAMPLE_PERIOD = '2 s'
P = dict(A=0.0154, Sn=5e-5, mu13=0.5, mu32=0.5, mu20=0.6, g=9.81, Qmax=1e-4, h1sp=0.45, h2sp=0.25,
         K1=250.0, Ti1=150.0, K2=250.0, Ti2=150.0,
         d_sig=(0.01, 0.01, 0.01), d_tau=(20.0, 20.0, 60.0),               # pump 1, pump 2, mu20 factor
         m_sig=(0.001, 0.001, 0.001, 0.5, 0.5, 0.0, 0.0),
         f_leak=0.03, f4_clog=0.1, f5_clog=0.1, f6_clog=0.08, f7_loss=0.08, f8_off=-1.5e-6, f9_bias=0.01, f10_rate=3.0e-5,
         h_hi=0.62, h_lo=0.01)
DT = 0.25; NSUB = 2; NCTL = 4; BURN = 600
NF_ = len(FAULTS); BASE = 20261008


def tor(mu, dh):
    """Torricelli flow mu Sn sgn(dh) sqrt(2 g |dh|), linear below 1 mm."""
    a = np.abs(dh); r = np.where(a > 1e-3, np.sqrt(2 * P['g'] * np.maximum(a, 1e-3)), a * np.sqrt(2 * P['g'] * 1e-3) / 1e-3)
    return mu * P['Sn'] * np.sign(dh) * r


def steady():
    h1, h2 = P['h1sp'], P['h2sp']; h3 = 0.5 * (h1 + h2)   # mu13 = mu32
    Q1 = float(tor(P['mu13'], h1 - h3)); Q2 = float(tor(P['mu20'], h2)) - Q1
    return h1, h2, h3, Q1, Q2


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = (BURN // 2 + T) * NCTL
    return g.standard_normal((nc, 3)).astype(np.float32), g.standard_normal((nc, 2)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(3).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids); T and onset in samples. Returns X (n, T, 7) float32 and shutdown_sample."""
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
    h10, h20, h30, Q10, Q20 = steady(); u10 = 100 * Q10 / P['Qmax']; u20 = 100 * Q20 / P['Qmax']
    h1 = np.full(n, h10); h2 = np.full(n, h20); h3 = np.full(n, h30); I1 = np.zeros(n); I2 = np.zeros(n)
    dsig = np.array(P['d_sig']); dtc = DT * NSUB; phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    burn_s = BURN; spp = 2.0; nc = int((burn_s + T * spp) / dtc); per = int(spp / dtc)
    for k in range(nc):
        t = ((k + 1) * dtc - burn_s) / spp; tau = max(t - onset, 0.0) * spp; on = t > onset
        d = phi * d + dinn * dist[sidx, k].astype(np.float64)
        g1 = 1 + d[:, 0]; g2 = 1 + d[:, 1]; m20 = P['mu20'] * (1 + d[:, 2]); m13 = np.full(n, P['mu13']); m32 = np.full(n, P['mu32'])
        lk = np.zeros((n, 3)); off2 = np.zeros(n); b1 = np.zeros(n); b3 = np.zeros(n)
        if on:
            lk = am[:, 1:4] * P['f_leak']; m13 = m13 * (1 - am[:, 4] * P['f4_clog']); m32 = m32 * (1 - am[:, 5] * P['f5_clog'])
            m20 = m20 * (1 - am[:, 6] * P['f6_clog']); g1 = g1 * (1 - am[:, 7] * P['f7_loss']); off2 = am[:, 8] * P['f8_off']
            b1 = am[:, 9] * P['f9_bias']; b3 = am[:, 10] * P['f10_rate'] * tau
        cn = ctl[sidx, k].astype(np.float64)
        h1m = h1 + b1 + msig[0] * cn[:, 0]; h2m = h2 + msig[1] * cn[:, 1]
        e1 = P['h1sp'] - h1m; e2 = P['h2sp'] - h2m; u1 = u10 + P['K1'] * (e1 + I1); u2 = u20 + P['K2'] * (e2 + I2)
        s1 = ((u1 >= 100) & (e1 > 0)) | ((u1 <= 0) & (e1 < 0)); s2 = ((u2 >= 100) & (e2 > 0)) | ((u2 <= 0) & (e2 < 0))
        I1 = np.where(s1, I1, I1 + e1 * dtc / P['Ti1']); I2 = np.where(s2, I2, I2 + e2 * dtc / P['Ti2'])
        u1 = np.clip(u10 + P['K1'] * (e1 + I1), 0, 100); u2 = np.clip(u20 + P['K2'] * (e2 + I2), 0, 100)
        Q1 = np.maximum(P['Qmax'] * u1 / 100 * g1, 0.0); Q2 = np.maximum(P['Qmax'] * u2 / 100 * g2 + off2, 0.0)
        for _ in range(NSUB):
            q13 = tor(m13, h1 - h3); q32 = tor(m32, h3 - h2); q20 = tor(m20, np.maximum(h2, 0.0))
            ql = [tor(0.5 * lk[:, i], np.maximum(hh, 0.0)) for i, hh in enumerate((h1, h2, h3))]
            h1 = h1 + DT * (Q1 - q13 - ql[0]) / P['A']; h3 = h3 + DT * (q13 - q32 - ql[2]) / P['A']; h2 = h2 + DT * (Q2 + q32 - q20 - ql[1]) / P['A']
        hm = np.stack([h1, h2, h3], 1)
        tripped = tripped | (hm.max(1) > P['h_hi']) | (hm.min(1) < P['h_lo'])
        if tripped.any():
            h1 = np.where(tripped, h10, h1); h2 = np.where(tripped, h20, h2); h3 = np.where(tripped, h30, h3)
        if (k + 1) * dtc > burn_s and (k + 1) % per == 0:
            j = int(round(((k + 1) * dtc - burn_s) / spp)) - 1
            if j < 0 or j >= T: continue
            mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([h1 + b1, h2, h3 + b3, 1e6 * Q1, 1e6 * Q2, u1, u2], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
