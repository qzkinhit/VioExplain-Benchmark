"""Quadruple-tank process under decentralized level control with concurrent faults and paired runs.

Model. The four-tank laboratory process of Johansson (IEEE TCST 2000), nonlinear balances
    A1 dh1/dt = -a1 sqrt(2 g h1) + a3 sqrt(2 g h3) + gamma1 k1 v1
    A2 dh2/dt = -a2 sqrt(2 g h2) + a4 sqrt(2 g h4) + gamma2 k2 v2
    A3 dh3/dt = -a3 sqrt(2 g h3) + (1 - gamma2) k2 v2
    A4 dh4/dt = -a4 sqrt(2 g h4) + (1 - gamma1) k1 v1
with the published parameters of the minimum-phase setting (A1 = A3 = 28 cm2, A2 = A4 = 32 cm2, a1 = a3 = 0.071 cm2,
a2 = a4 = 0.057 cm2, k1 = 3.33, k2 = 3.35 cm3/(V s), gamma1 = 0.70, gamma2 = 0.60, v1 = v2 = 3 V, g = 981 cm/s2) and the
decentralized PI controllers of that paper (K1 = 3.0, Ti1 = 30 s, K2 = 2.7, Ti2 = 40 s on the level-sensor voltage,
sensor gain 0.5 V/cm). The values were written from memory and not checked against the paper, so this file does not
reproduce a published data set. Additions introduced here:
  * four stochastic disturbances as first-order autoregressive processes (multiplicative gain fluctuation of each pump,
    unmeasured inflow to each upper tank), Gaussian measurement noise, flow sensors on both pumps;
  * ten fault types of this benchmark design (list FAULTS), each with a magnitude drawn per run index;
  * interlocks (overflow of any tank at 20 cm, lower tank below 3 cm) that trip the plant and freeze the record.
Common random numbers as in sim_cstr.py: every random number is a function of the run index only.
Time unit: second. Sampling period 5 s, control interval 1 s, integration step 0.5 s.
"""
import numpy as np

NAME = 'qtank'
VARS = ['h1', 'h2', 'h3', 'h4', 'v1', 'v2', 'q1', 'q2']
FAULTS = {1: 'pump 1 gain loss (actuator)', 2: 'pump 2 gain variability (actuator)', 3: 'valve 1 split shift (valve)',
          4: 'valve 2 split shift (valve)', 5: 'leak in tank 1 (leak)', 6: 'leak in tank 4 (leak)',
          7: 'clogged outlet of tank 3 (process)', 8: 'level sensor 1 bias (sensor, in loop)',
          9: 'level sensor 2 drift (sensor, in loop)', 10: 'level sensor 4 bias (sensor)'}
SAMPLE_PERIOD = '5 s'
P = dict(A=(28.0, 32.0, 28.0, 32.0), a=(0.071, 0.057, 0.071, 0.057), g=981.0, k=(3.33, 3.35), gam=(0.70, 0.60), v0=(3.0, 3.0),
         Kc=(1.5, 1.35), Ti=(30.0, 40.0),
         d_sig=(0.015, 0.015, 0.25, 0.25), d_tau=(100.0, 100.0, 60.0, 60.0),      # pump gain 1, 2 (relative), inflow tank 3, 4
         m_sig=(0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05),
         f1_loss=0.15, f2_mult=8.0, f3_shift=-0.08, f4_shift=0.08, f5_leak=0.25, f6_leak=0.30, f7_clog=0.25, f8_bias=1.5,
         f9_rate=0.0012, f10_bias=0.8, h_hi=20.0, h_lo=3.0)
DT = 0.5; NSUB = 2; NCTL = 5; BURN = 120
NF = len(FAULTS); BASE = 20261005


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 21, int(seed)]); nc = (BURN + T) * NCTL
    return g.standard_normal((nc, 4)).astype(np.float32), g.standard_normal((nc, 2)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(4).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 22, int(seed), int(f)]).uniform(0.6, 1.4))


def steady():
    a = np.array(P['a']); s = np.sqrt(2 * P['g']); k1, k2 = P['k']; g1, g2 = P['gam']; v1, v2 = P['v0']
    h3 = ((1 - g2) * k2 * v2 / (a[2] * s)) ** 2; h4 = ((1 - g1) * k1 * v1 / (a[3] * s)) ** 2
    h1 = ((a[2] * s * np.sqrt(h3) + g1 * k1 * v1) / (a[0] * s)) ** 2; h2 = ((a[3] * s * np.sqrt(h4) + g2 * k2 * v2) / (a[1] * s)) ** 2
    return np.array([h1, h2, h3, h4])


def simulate(runs, T, onset):
    """runs: list of (seed, tuple of fault ids). Returns X (n, T, 8) float32 and shutdown_sample (n,) (-1: no trip)."""
    n = len(runs); seeds = sorted(set(int(s) for s, _ in runs)); pos = {s: i for i, s in enumerate(seeds)}
    sidx = np.array([pos[int(s)] for s, _ in runs]); NS = [noise_streams(s, T) for s in seeds]
    dist = np.stack([x[0] for x in NS]); ctl = np.stack([x[1] for x in NS]); meas = np.stack([x[2] for x in NS]); d0 = np.stack([x[3] for x in NS])
    act = np.zeros((n, NF + 1), bool); mag = np.zeros((n, NF + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    hs = steady(); h = np.tile(hs, (n, 1)); I1 = np.zeros(n); I2 = np.zeros(n)
    A = np.array(P['A']); a0 = np.array(P['a']); s2g = np.sqrt(2 * P['g']); k1, k2 = P['k']; v10, v20 = P['v0']
    dsig = np.array(P['d_sig']); dtc = DT * NSUB; phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx] * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    nc = (BURN + T) * NCTL; onset_t = onset * NCTL * dtc
    for k in range(nc):
        t = (k + 1) * dtc - BURN * NCTL * dtc; tau = max(t - onset_t, 0.0); on = t > onset_t
        e = dist[sidx, k].astype(np.float64); mult = np.ones((n, 4))
        if on: mult[:, 1] = np.where(act[:, 2], P['f2_mult'], 1.0)
        d = phi * d + dinn * mult * e
        g1 = np.full(n, P['gam'][0]); g2 = np.full(n, P['gam'][1]); loss1 = np.zeros(n); aL1 = np.zeros(n); aL4 = np.zeros(n); a3 = np.full(n, a0[2])
        b1 = np.zeros(n); b2 = np.zeros(n); b4 = np.zeros(n)
        if on:
            loss1 = am[:, 1] * P['f1_loss']; g1 = g1 + am[:, 3] * P['f3_shift']; g2 = g2 + am[:, 4] * P['f4_shift']
            aL1 = am[:, 5] * P['f5_leak'] * a0[0]; aL4 = am[:, 6] * P['f6_leak'] * a0[3]
            a3 = a0[2] * (1 - am[:, 7] * P['f7_clog']); b1 = am[:, 8] * P['f8_bias']
            b2 = am[:, 9] * P['f9_rate'] * tau; b4 = am[:, 10] * P['f10_bias']
        cn = ctl[sidx, k].astype(np.float64)
        e1 = hs[0] - (h[:, 0] + b1 + msig[0] * cn[:, 0]); e2 = hs[1] - (h[:, 1] + b2 + msig[1] * cn[:, 1])
        u1 = v10 + P['Kc'][0] * (e1 + I1); sat1 = ((u1 >= 10) & (e1 > 0)) | ((u1 <= 0) & (e1 < 0))
        I1 = np.where(sat1, I1, I1 + e1 * dtc / P['Ti'][0]); v1 = np.clip(v10 + P['Kc'][0] * (e1 + I1), 0, 10)
        u2 = v20 + P['Kc'][1] * (e2 + I2); sat2 = ((u2 >= 10) & (e2 > 0)) | ((u2 <= 0) & (e2 < 0))
        I2 = np.where(sat2, I2, I2 + e2 * dtc / P['Ti'][1]); v2 = np.clip(v20 + P['Kc'][1] * (e2 + I2), 0, 10)
        q1 = k1 * (1 + d[:, 0]) * (1 - loss1) * v1; q2 = k2 * (1 + d[:, 1]) * v2
        for _ in range(NSUB):
            r = s2g * np.sqrt(np.maximum(h, 0.0))
            dh1 = (-(a0[0] + aL1) * r[:, 0] + a3 * r[:, 2] + g1 * q1) / A[0]
            dh2 = (-a0[1] * r[:, 1] + a0[3] * r[:, 3] + g2 * q2) / A[1]
            dh3 = (-a3 * r[:, 2] + (1 - g2) * q2 + d[:, 2]) / A[2]
            dh4 = (-(a0[3] + aL4) * r[:, 3] + (1 - g1) * q1 + d[:, 3]) / A[3]
            h = np.maximum(h + DT * np.stack([dh1, dh2, dh3, dh4], 1), 0.0)
        tripped = tripped | (h > P['h_hi']).any(1) | (h[:, :2] < P['h_lo']).any(1)
        if tripped.any(): h = np.where(tripped[:, None], hs[None], h)   # record frozen after a trip; states kept finite
        if k >= BURN * NCTL and (k + 1) % NCTL == 0:
            j = (k + 1) // NCTL - BURN - 1; mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([h[:, 0] + b1, h[:, 1] + b2, h[:, 2], h[:, 3] + b4, v1, v2, q1, q2], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
