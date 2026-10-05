"""Closed-loop pH neutralization process (Henson and Seborg benchmark) with concurrent faults and paired runs.

Model. The pH neutralization process of Henson and Seborg (IEEE Trans. Control Systems Technology 2(3):169-182,
1994), in the single-tank reduced form used by most simulation studies of this benchmark. A stirred tank
(cross-section A = 207 cm2) receives an acid stream (HNO3, flow q1), a buffer stream (NaHCO3, flow q2) and a base
stream (NaOH, flow q3); the effluent q4 leaves through a valve. Balances (the paper's equations (13)-(16) with the
two-tank inlet replaced by q1 directly):
    A dh/dt = q1 + q2 + q3 - q4 - q_leak
    A h dWa4/dt = q1 (Wa1 - Wa4) + q2 (Wa2 - Wa4) + q3 (Wa3 - Wa4)
    A h dWb4/dt = q1 (Wb1 - Wb4) + q2 (Wb2 - Wb4) + q3 (Wb3 - Wb4)
where Wa4 and Wb4 are the reaction invariants of the acid-base system (Wa = [H+] - [OH-] - [HCO3-] - 2 [CO3(2-)],
Wb = [H2CO3] + [HCO3-] + [CO3(2-)]). The pH follows from the electroneutrality equation (9) of the paper,
    Wb4 (Ka1/[H+] + 2 Ka1 Ka2/[H+]^2) / (1 + Ka1/[H+] + Ka1 Ka2/[H+]^2) + Wa4 + Kw/[H+] - [H+] = 0,
solved here by a vectorized Newton iteration in log10([H+]); pH = -log10([H+]).
Parameters taken from the paper Table I (checked against the original paper and its reproductions):
A = 207 cm2, q1 = 16.6, q2 = 0.55, q3 = 15.55, q4 = 32.7 ml/s (the value that closes the balances; the table prints
15.6 and 32.8), h = 14.0 cm, pH = 7.0, Wa1 = 3.0e-3, Wb1 = 0, Wa2 = -3.0e-2, Wb2 = 3.0e-2, Wa3 = -3.05e-3,
Wb3 = 5.0e-5 mol/L, Ka1 = 4.47e-7, Ka2 = 5.62e-11, Kw = 1.0e-14, transmitter lags tau_pH = tau_h = 15 s, pH
measurement delay 10 s, valve lag 6 s; Cv = 8.75 (ml/s)/cm^0.5 is the single-tank valve coefficient of the
reduced model (8.75 sqrt(14) = 32.7).
Deviations from the paper, all introduced here:
  * the acid buffer tank of the original (A2 = 42 cm2, q1e = Cv1 sqrt(h2)) is dropped; the acid stream enters the
    neutralization tank directly as q1;
  * the effluent valve is linear, q4 = q4max u4 / 100, instead of the power-law valve equation (14), and the level
    is held by a PI loop on the effluent flow. The paper regulates only the pH (by q3) and leaves the effluent to
    the valve equation; Boehling, Seborg and Hespanha (IFAC 2005) regulate the level by the acid flow instead;
  * the two loops run every 1 s instead of the 15 s pH control period of the paper;
  * first-order autoregressive disturbances on the acid flow, the buffer flow, the acid concentration and the
    buffer concentration (the paper treats q1 and q2 as unmeasured disturbances and estimates q2 adaptively);
  * Gaussian measurement noise on every recorded variable;
  * ten fault types (list FAULTS), each with a magnitude drawn per run index from U(0.6, 1.4) times a nominal
    size in P; the fault blocks (stiction, transmitter bias and drift, meter bias, leak) are written here;
  * interlocks on the level and on the pH; no run of the generated data trips them.
Common random numbers. Every random number of a run is a function of the run index (seed) only, so runs with the
same seed and different fault sets are paired runs. Time unit: second. Sampling period 2 s, control interval 1 s,
explicit Euler step 0.25 s. The recorded signals are the transmitter readings (pH and level, lagged; the pH with
an additional transport delay) and flow-meter readings in ml/s, plus the two valve commands in per cent.
"""
import os
import numpy as np

NAME = 'ph'
VARS = ['pH', 'h', 'q1', 'q2', 'q3', 'q4', 'u3', 'u4']
FAULTS = {1: 'acid flow step (disturbance)', 2: 'buffer flow step (disturbance)',
          3: 'acid concentration drop (disturbance)', 4: 'buffer concentration drop (disturbance)',
          5: 'base valve stiction (actuator)', 6: 'effluent valve stiction (actuator)',
          7: 'pH transmitter bias (sensor, in loop)', 8: 'level transmitter drift (sensor, in loop)',
          9: 'acid flow meter bias (sensor, out of loop)', 10: 'tank leak (process)'}
SAMPLE_PERIOD = '2 s'
P = dict(A=207.0, q1_0=16.6, q2_0=0.55, q3_0=15.55, q4_0=32.7, h_sp=14.0, pH_sp=7.0,
         Wa1=3.0e-3, Wb1=0.0, Wa2=-3.0e-2, Wb2=3.0e-2, Wa3=-3.05e-3, Wb3=5.0e-5,
         Ka1=4.47e-7, Ka2=5.62e-11, Kw=1.0e-14, Wa4_0=-4.32e-4, Wb4_0=5.28e-4,
         q3max=31.0, q4max=65.0, Kp3=5.0, Ti3=150.0, Kp4=5.0, Ti4=60.0,
         tau_v=6.0, tau_ph=15.0, tau_h=15.0, theta_ph=10.0,
         d_sig=(0.33, 0.028, 3.0e-5, 3.0e-4), d_tau=(30.0, 60.0, 120.0, 120.0),        # q1, q2, Wa1, Wb2
         m_sig=(0.02, 0.02, 0.08, 0.005, 0.08, 0.16, 0.0, 0.0),
         f1_step=1.0, f2_step=0.15, f3_step=-2.0e-4, f4_step=-1.5e-2,
         f5_band=5.0, f6_band=2.5, f7_bias=0.3, f8_rate=0.004, f9_bias=1.0, f10_leak=1.0,
         h_hi=26.0, h_lo=2.0, pH_hi=12.5, pH_lo=2.0)
DT = 0.25; NSUB = 4; NCTL = 2; BURN = 600
NF_ = len(FAULTS); BASE = 20261010
NDPH = int(round(P['theta_ph'] / (DT * NSUB)))               # pH measurement delay in control steps


def noise_streams(seed, T):
    g = np.random.default_rng([BASE, 11, int(seed)]); nc = BURN + T * 2
    return g.standard_normal((nc, 4)).astype(np.float32), g.standard_normal((nc, 2)).astype(np.float32), \
        g.standard_normal((T, len(VARS))).astype(np.float32), g.standard_normal(4).astype(np.float32)


def fault_mag(seed, f):
    return float(np.random.default_rng([BASE, 12, int(seed), int(f)]).uniform(0.6, 1.4))


def ph_of(Wa, Wb):
    """pH from the electroneutrality equation (9) of the paper, vectorized bisection on pH in [-2, 16].
    g(pH) = Wb (Ka1/[H+] + 2 Ka1 Ka2/[H+]^2) / (1 + Ka1/[H+] + Ka1 Ka2/[H+]^2) + Wa + Kw/[H+] - [H+] is strictly
    increasing in pH (the buffer term rises from 0 to 2 Wb, Kw/[H+] grows, [H+] shrinks), so 30 bisection steps
    return the unique root to about 1e-8 pH units."""
    Ka1 = P['Ka1']; Ka2 = P['Ka2']; Kw = P['Kw']
    lo = np.full(Wa.shape, -2.0); hi = np.full(Wa.shape, 16.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi); x = 10.0 ** (-mid)
        g = Wb * (Ka1 * x + 2.0 * Ka1 * Ka2) / (x * x + Ka1 * x + Ka1 * Ka2) + Wa + Kw / x - x
        pos = g > 0.0
        hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


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
    dist = np.stack([a[0] for a in NS]); ctl = np.stack([a[1] for a in NS]); meas = np.stack([a[2] for a in NS]); d0 = np.stack([a[3] for a in NS])
    act = np.zeros((n, NF_ + 1), bool); mag = np.zeros((n, NF_ + 1))
    for i, (s, fs) in enumerate(runs):
        for f in fs: act[i, f] = True; mag[i, f] = fault_mag(s, f)
    am = act * mag
    h = np.full(n, P['h_sp']); Wa = np.full(n, P['Wa4_0']); Wb = np.full(n, P['Wb4_0'])
    u30 = 100.0 * P['q3_0'] / P['q3max']; u40 = 100.0 * P['q4_0'] / P['q4max']
    u3 = np.full(n, u30); u4 = np.full(n, u40); p3 = u3.copy(); p4 = u4.copy()
    xv3 = u3.copy(); xv4 = u4.copy(); I3 = np.zeros(n); I4 = np.zeros(n)
    txp = np.full(n, P['pH_sp']); txh = np.full(n, P['h_sp'])
    bufP = np.full((NDPH, n), P['pH_sp']); bp = 0
    dsig = np.array(P['d_sig']); dtc = DT * NSUB; phi = np.exp(-dtc / np.array(P['d_tau'])); dinn = dsig * np.sqrt(1 - phi ** 2)
    d = d0[sidx].astype(np.float64) * dsig; msig = np.array(P['m_sig'])
    X = np.zeros((n, T, len(VARS)), np.float32); shut = np.full(n, -1); tripped = np.zeros(n, bool); last = np.zeros((n, len(VARS)))
    burn_s = BURN; spp = 2.0; nc = int((burn_s + T * spp) / dtc); per = int(spp / dtc)
    for k in range(nc):
        t = ((k + 1) * dtc - burn_s) / spp; tau = max(t - onset, 0.0) * spp; on = t > onset
        d = phi * d + dinn * dist[sidx, k].astype(np.float64)
        q1 = P['q1_0'] + d[:, 0]; q2 = P['q2_0'] + d[:, 1]; Wa1 = P['Wa1'] + d[:, 2]; Wb2 = P['Wb2'] + d[:, 3]
        bph = np.zeros(n); bh = np.zeros(n); bq1 = np.zeros(n); lk = np.zeros(n)
        if on:
            q1 = q1 + am[:, 1] * P['f1_step']; q2 = q2 + am[:, 2] * P['f2_step']
            Wa1 = Wa1 + am[:, 3] * P['f3_step']; Wb2 = Wb2 + am[:, 4] * P['f4_step']
            bph = am[:, 7] * P['f7_bias']; bh = am[:, 8] * P['f8_rate'] * tau; bq1 = am[:, 9] * P['f9_bias']; lk = am[:, 10] * P['f10_leak']
        q1 = np.maximum(q1, 0.1); q2 = np.maximum(q2, 0.01); Wa1 = np.maximum(Wa1, 0.0); Wb2 = np.maximum(Wb2, 0.0)
        Wa2 = -Wb2                                   # NaHCO3 stream: Wa2 = -Wb2 (paper Table I: -3.0e-2 and 3.0e-2)
        ph = ph_of(Wa, Wb)
        txp = txp + dtc / P['tau_ph'] * (ph + bph - txp)
        txh = txh + dtc / P['tau_h'] * (h + bh - txh)
        phd = bufP[bp].copy(); bufP[bp] = txp; bp = (bp + 1) % NDPH
        cn = ctl[sidx, k].astype(np.float64)
        phm = phd + msig[0] * cn[:, 0]; hm = txh + msig[1] * cn[:, 1]
        e4 = hm - P['h_sp']; e3 = P['pH_sp'] - phm
        u4 = u40 + P['Kp4'] * e4 + I4; u3 = u30 + P['Kp3'] * e3 + I3
        s4 = ((u4 >= 100) & (e4 > 0)) | ((u4 <= 0) & (e4 < 0)); s3 = ((u3 >= 100) & (e3 > 0)) | ((u3 <= 0) & (e3 < 0))
        I4 = np.where(s4, I4, I4 + e4 * dtc / P['Ti4']); I3 = np.where(s3, I3, I3 + e3 * dtc / P['Ti3'])
        u4 = np.clip(u40 + P['Kp4'] * e4 + I4, 0, 100); u3 = np.clip(u30 + P['Kp3'] * e3 + I3, 0, 100)
        if on:
            st3 = act[:, 5] & (np.abs(u3 - p3) <= mag[:, 5] * P['f5_band']); p3 = np.where(st3, p3, u3)
            st4 = act[:, 6] & (np.abs(u4 - p4) <= mag[:, 6] * P['f6_band']); p4 = np.where(st4, p4, u4)
        else:
            p3 = u3; p4 = u4
        for _ in range(NSUB):
            xv3 = xv3 + DT / P['tau_v'] * (p3 - xv3); xv4 = xv4 + DT / P['tau_v'] * (p4 - xv4)
            q3 = P['q3max'] * xv3 / 100.0; q4 = P['q4max'] * xv4 / 100.0
            h = h + DT * (q1 + q2 + q3 - q4 - lk) / P['A']
            den = P['A'] * np.maximum(h, 1e-3)
            Wa = Wa + DT * (q1 * (Wa1 - Wa) + q2 * (Wa2 - Wa) + q3 * (P['Wa3'] - Wa)) / den
            Wb = Wb + DT * (q1 * (P['Wb1'] - Wb) + q2 * (Wb2 - Wb) + q3 * (P['Wb3'] - Wb)) / den
        tripped = tripped | (h > P['h_hi']) | (h < P['h_lo']) | (ph < P['pH_lo']) | (ph > P['pH_hi']) | ~np.isfinite(Wa) | ~np.isfinite(Wb)
        if tripped.any():
            h = np.where(tripped, P['h_sp'], h); Wa = np.where(tripped, P['Wa4_0'], Wa); Wb = np.where(tripped, P['Wb4_0'], Wb)
        if (k + 1) * dtc > burn_s and (k + 1) % per == 0:
            j = int(round(((k + 1) * dtc - burn_s) / spp)) - 1
            if j < 0 or j >= T: continue
            mn = meas[sidx, j].astype(np.float64) * msig
            row = np.stack([phd, txh, q1 + bq1, q2, q3, q4, u3, u4], 1) + mn
            shut = np.where(tripped & (shut < 0), j, shut)
            if j > 0: row = np.where(tripped[:, None], last, row)
            last = row; X[:, j] = row
    return X, shut
