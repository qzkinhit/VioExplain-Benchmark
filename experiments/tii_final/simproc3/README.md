# Two more simulated benchmarks with concurrent events: forced-circulation evaporator (EVAP) and pH neutralization (PH)

This folder adds two closed-loop process simulators with paired concurrent-fault runs and runs VioExplain and the 18
baselines on them under the protocol of `scripts/simproc2/` (see its README). `common.py`, `final_base.py`,
`final_fuse5.py`, `final_probs.py`, `final_rocket.py`, `gpu_simproc.py`, `gen_data.py` and `sim_check.py` are copies
of the `scripts/simproc2/` files. The copies differ from them in the following places.

1. `common.py` reads its data from `data/simproc3/<process>/`.
2. `gen_data.py` writes to `data/simproc3/<process>/` and samples min(5 x number of fault types, all triples) random
   triples, which is 50 for both processes (as in `scripts/simproc2/`).
3. `launch_gpu3.sh` takes the GPU from `V3_DEV` (on gpu-server, cuda:2 for EVAP and cuda:5 for PH, because the other four
   cards had about 4 GB free memory at launch time), and `launch_after_gpu3.sh` also runs VioExplain without
   composition (f0) for `metrics_extra.json`.

The method (run label B: learned addition decision without the step feature, ResNet temporal encoder,
functional-constraint units), the 18 baselines, the calibration rule, the conformal empty set, the composition recipe,
the effective-event truth, the metrics and the MultiRocket kernel count (1000) are those of `scripts/simproc2/`.
No parameter of any method was chosen on evaluation windows. The fault sizes of the simulators were set before any
method was run, with `sim_check.py` only (share of windows in which a fault causes a typed violation against the
paired normal window), with the aim that every fault type is effective in a part of its windows.

## Simulators and provenance

Each simulator is a vectorized explicit-Euler integration of the published balances with PI control, first-order
autoregressive disturbances and Gaussian measurement noise. Each docstring lists the model equations, the parameters
taken from the source and every deviation. Neither data set reproduces a published data set; the faults, the
disturbance and noise models and the controller settings are written here.

**`sim_evap.py`, forced-circulation evaporator** (Newell and Lee 1989, *Applied Process Control: A Case Study*, in the
form used in process-control benchmarks, for example Kariwala and Cao 2009). Taken from the source: the three state
equations for separator level L2, product composition X2 and operating pressure P2 (holdup 20 kg, pressure constant
4 kg/kPa), the algebraic relations for T2, T3, T100, Q100, UA1 = 0.16 (F1 + F3), F4, F100, Q200 with UA2 = 6.84,
T201 and F5, the constants (C = 0.07, latent heats 38.5 and 36.6), the nominal operating point (F1 = 10, X1 = 5 %,
T1 = 40 C, F2 = 2, X2 = 25 %, L2 = 1 m, P2 = 50.5 kPa, F3 = 50, P100 = 194.7 kPa, F200 = 208, T200 = 25 C) and the
control structure (L2 by F2, X2 by P100, P2 by F200). The equations and values were written from the benchmark
description in the literature, not from the book, and were checked by closing the steady-state balances (F4 = F5 = 8,
T2 = 84.6 C, Q100 = 339, Q200 = 308, T201 = 46.1 C). Deviations: PI settings chosen on this model; first-order
actuators for P100, F200 and F2; a 1 min delay and 1 min lag on the composition analyzer; autoregressive disturbances
of F1, X1, T1 and T200. 17 recorded variables (L2, X2 analyzer, P2, F1, X1, T1, T200, F2, F3, F100, P100, F200, T2, T3,
T201 and the two controller outputs for P100 and F200), sampling 1 min, 10 fault types.

**`sim_ph.py`, pH neutralization** (Henson and Seborg 1994, IEEE Trans. Control Systems Technology 2(3):169-182).
Taken from the source: the acid (HNO3), buffer (NaHCO3) and base (NaOH) streams, the reaction invariants Wa and Wb
with their balances, the electroneutrality equation for pH (solved by bisection on pH), the parameter values of the
paper's Table I (A = 207 cm2, q1 = 16.6, q2 = 0.55, q3 = 15.55 ml/s, h = 14 cm, pH 7, stream invariants, Ka1, Ka2,
Kw), the 15 s transmitter lags, the 10 s pH measurement delay and the 6 s valve lag, and pH control by the base flow.
Deviations: the acid buffer tank of the original is dropped and the acid stream enters the tank directly; the
effluent valve is linear and the level is held by a PI loop on the effluent flow (the paper leaves the effluent to
the valve equation); the loops run every 1 s instead of 15 s; autoregressive disturbances of the acid flow, the buffer
flow, the acid concentration and the buffer concentration. 8 recorded variables (pH, level, the four flows, the two
valve commands), sampling 2 s, 10 fault types.

Fault types (magnitude of each fault type drawn per run index from U(0.6, 1.4) times the nominal size in `P`):

| No. | EVAP | PH |
|---|---|---|
| 1 | feed flow step (disturbance) | acid flow step (disturbance) |
| 2 | feed composition step (disturbance) | buffer flow step (disturbance) |
| 3 | cooling-water inlet temperature step (disturbance) | acid concentration drop (disturbance) |
| 4 | evaporator heat-transfer fouling (process) | buffer concentration drop (disturbance) |
| 5 | condenser heat-transfer fouling (process) | base valve stiction (actuator) |
| 6 | circulation pump degradation (process) | effluent valve stiction (actuator) |
| 7 | steam valve stiction (actuator) | pH transmitter bias (sensor, in loop) |
| 8 | cooling-water valve gain loss (actuator) | level transmitter drift (sensor, in loop) |
| 9 | product composition analyzer bias (sensor, in loop) | acid flow meter bias (sensor, out of loop) |
| 10 | operating pressure sensor drift (sensor, in loop) | tank leak (process) |

Common random numbers: every random number of a run (disturbance and noise streams, fault magnitudes) is a function
of its seed only, so the runs of all subsets of a fault set are paired runs. `sim_check.py` confirmed determinism and
identical records before the onset for both simulators. No run of the generated data trips an interlock.

## Data (`gen_data.py`, `data/simproc3/<process>/`, identical on cpu-server and gpu-server)

Same layout and protocol as `scripts/simproc2/`: 400 training runs per class (runs 0 to 299 fit, 300 to 399
calibration, 512 samples, onset 64, 7 windows each), 500 test runs per class (960 samples, onset 160, hash split into
test calibration and evaluation, 12 windows each), test_pairs with 30 seeds for normal, every single fault and all 45
pairs, test_triples with 15 seeds for 50 random triples and all their subsets, four-fault windows superposed from the
paired single-fault runs of test_pairs. Window length 64 samples. `gen_all3.sh` generates both data sets.

## Run order

`gen_all3.sh` (cpu-server and gpu-server), `launch_cpu3.sh` (cpu-server; after the first minutes its lane loops were replaced by `launch_cpu3_split.sh`, which runs the same commands with the same threads for one data set per call, so that EVAP and PH ran in parallel), `launch_gpu3.sh` (gpu-server, `V3_DEV`), `relay3.sh early|late`
(workstation), `collect3.py` (cpu-server). Results: `results/simproc_v1/{evap,ph}/metrics_main.json`,
`metrics_samecomp.json`, `metrics_extra.json`, `dataset_stats.json`, `dataset_truth_stats.json` (written by `stats3.py`). All runs finished on 2026-10-05 between 21:53 and 22:22; no method is missing.
