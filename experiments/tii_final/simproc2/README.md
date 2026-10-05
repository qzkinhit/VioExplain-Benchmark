# Three more benchmarks with concurrent events: distillation column A, stirred tank heater, three-tank system

This folder adds three closed-loop process simulators with paired concurrent-fault runs and runs VioExplain and the 18
baselines on them under the TEP protocol. It reuses the runner and loader of `../simproc/` (CSTR, quadruple tank)
without modifying them: `common.py`, `final_base.py`, `final_fuse5.py`, `final_probs.py`, `final_rocket.py`,
`gpu_simproc.py`, `gen_data.py` and `sim_check.py` are copies of the `../simproc/` files.
The copies differ from the originals in the following places.

1. `common.py` reads its data from `data/simproc2/<process>/` instead of `data/simproc/<process>/`.
2. `gen_data.py` writes to `data/simproc2/<process>/` and samples 50 random triples for each process.
3. `final_rocket.py` reads the number of MultiRocket kernels from `V3_MR_KERNELS`. The aeon default is 10000. The runs
   here use 1000. On the tank data of `../simproc/` the default MultiRocket needed 45 minutes to train and had not
   finished scoring 40 minutes later, and a later attempt on the CSTR and tank data ended with a segmentation fault.
   With 1000 kernels it takes 5 to 8 minutes per data set here.

Two later changes concern only the copies of this folder. `final_base.py` searches the AEC and MinExplain parameters
with `tune_axes`, which uses the objective, windows and tie rule of `tune_on_tcal` and extends every grid outward while
the optimum sits at its end (the first pass with the fixed grids of `../simproc/` left the exact-representation
threshold at its end on DIST and CSTH). Those reruns are `f0/metrics_kb.json`, and the earlier fixed-grid
results remain in `f0/metrics_base.json` and `f0/metrics_kb_grid5.json` for reference. The scoring of the GPU outputs is
split into an early part (VioExplain, 1D-CNN, ResNet, LSTM, ML-CNN) and a late part (InceptionTime, MantisV2,
`f0/metrics_probs_late.json`), because InceptionTime takes longer to train.

The method, the 18 baselines, the calibration rule, the conformal empty set, the composition recipe, the effective-event
truth and the metrics are therefore those of the CSTR and tank runs, which are those of the TEP runs.

## Simulators and provenance

Each simulator is a vectorized explicit-Euler integration of the published balances with PI control, first-order
autoregressive disturbances and Gaussian measurement noise. Each docstring lists the model equations, the parameters
taken from the source and every deviation. Summary:

**`sim_dist.py`, distillation column A** (Skogestad and Morari 1988, Skogestad's `colamod.m`, `cola_lv.m`,
`colas_PI.m`, checked against https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html and the two m-files). Taken from
the source: 41 stages, feed stage 21, alpha 1.5, F 1, zF 0.5, qF 1, L 2.70629, V 3.20629, D = B = 0.5, xD 0.99, xB 0.01,
holdup 0.5 kmol on every stage, tau_L 0.063 min, lambda 0, the material balances of `colamod.m`, proportional level
control of condenser and reboiler (gain 10, LV configuration), and the composition PI settings of `colas_PI.m` (top
composition by L, gain 26.1, integral time 3.76 min, bottom composition by V, gain -37.5, integral time 3.31 min, on mole
fractions, 1 min measurement delay on each composition). Added here: first-order reflux and boilup valves (0.1 min,
range twice the nominal flow), Murphree vapor efficiency on the rectifying trays (1 nominally), tray temperatures from
the linear rule T = 100 - 20 x (deg C), discrete PI every 0.25 min, autoregressive disturbances of feed flow, feed
composition and feed liquid fraction. 20 recorded variables (F, zF, L, V, D, B, condenser and reboiler holdup, xD and xB
analyzer readings, eight tray temperatures, two valve demands), sampling 1 min, 10 fault types.

**`sim_csth.py`, continuous stirred tank heater** (Thornhill, Patwardhan and Shah 2008), following the Simulink model
distributed with the paper (Zenodo record 10093059, `CSTHMaster.mdl`, `CSTHDisturbedStdOp*.mdl`, `index.htm`), operating
point 2. Taken from the source: the volume-level table of the tank, the outflow law f_out = 1e-4 (0.1013 sqrt(55 + h) +
0.0237) m3/s, the cold-water valve (1 s dead time, 3.8 s lag, signal-to-flow and signal-to-flow-measurement tables), the
hot-water valve table, the steam valve to heat input table, the level and temperature mA tables, the 8 s delay of the
temperature measurement, the three PI loops in mA (level P 2, I 0.1, cascaded onto the cold-water flow loop P 0.5, I 0.2,
temperature P 3, I 0.1 on the steam valve), hot water in manual, cold water 24 C and hot water 50 C. Deviations: constant
rho c_p instead of the enthalpy and density tables, autoregressive noise with the standard deviations of the recorded
noise sequences instead of the recordings, and added disturbances (hot-water and cold-water supply temperature, steam
supply pressure) and two added sensors (hot-water flow and temperature), because the distributed model has no supply
disturbances. The distributed files contain no fault
blocks, so all fault blocks are written here. 8 recorded variables, sampling 1 s, 10 fault types.

**`sim_dts200.py`, three-tank system DTS200** (Amira, used as the COSY benchmark). Equations and parameter values as
commonly quoted in the fault-diagnosis literature (A 0.0154 m2, Sn 5e-5 m2, outflow coefficients 0.5, 0.5, 0.6, pumps up
to 1e-4 m3/s), not checked against the Amira manual. Two PI level loops (h1 at 0.45 m, h2 at
0.25 m, chosen here), autoregressive pump and outflow-coefficient disturbances. 7 recorded variables (three levels, two
pump flows, two pump demands), sampling 2 s, 10 fault types.

Fault types (magnitude of each fault type drawn per run index from U(0.6, 1.4) times the nominal size in `P`):

| No. | DIST | CSTH | DTS200 |
|---|---|---|---|
| 1 | feed composition step | steam valve stiction | leak in tank 1 |
| 2 | feed flow variability | cold-water valve stiction | leak in tank 2 |
| 3 | feed flow step | level sensor bias (in loop) | leak in tank 3 |
| 4 | feed liquid fraction drop | temperature sensor bias (in loop) | clogging, pipe 1 to 3 |
| 5 | reflux valve stiction | cold-water flow sensor bias (inner loop) | clogging, pipe 3 to 2 |
| 6 | boilup valve gain loss | steam supply pressure drop | clogging, outflow of tank 2 |
| 7 | top analyzer bias (in loop) | steam supply pressure oscillation | pump 1 gain loss |
| 8 | bottom analyzer drift (in loop) | tank leak | pump 2 delivery offset |
| 9 | stage-8 temperature sensor drift | hot-water supply temperature drop | level sensor 1 bias (in loop) |
| 10 | rectifying tray efficiency loss | hot-water valve signal offset | level sensor 3 drift |

Common random numbers: every random number of a run (disturbance and noise streams, fault magnitudes) is a function
of its seed only, so the runs of all subsets of a fault set are paired runs. `sim_check.py` confirmed determinism and
identical records before the onset. No run of the generated data trips an interlock.

## Data (`gen_data.py`, `data/simproc2/<process>/`)

Same layout and protocol as `../simproc/`: 400 training runs per class (seeds 0 to 399, 512 samples, onset 64,
runs 0 to 299 fit, 300 to 399 calibration, 7 windows each), 500 test runs per class (960 samples, onset 160, hash split
into test calibration and evaluation, 12 windows each), test_pairs with 30 seeds for normal, every single fault and all
45 pairs, test_triples with 15 seeds for 50 random triples and all their subsets, four-fault windows superposed from
the paired single-fault runs of test_pairs (40 random four-fault sets). Window length 64 samples. All 10 faults are
effective. 71 composed triple sets on the training side (300 times 45 over 190, as on TEP). `gen_data.py` generates the
three data sets.

## Running

- `gen_data.py PROCESS`: data of one process (`dist`, `csth` or `dts200`).
- `final_base.py`: all CPU baselines under f0 and BR-LGBM, CC-LGBM under f100t. `final_rocket.py`: MiniRocket,
  MultiRocket (1000 kernels) and QUANT.
- `gpu_simproc.py`: 1D-CNN, ResNet, LSTM, InceptionTime, ML-CNN under f0 and f100t, MantisV2.
- `final_fuse5.py`: VioExplain (run label B, f100t, V3_FUNC=1 V3_NOSTEP=1 V3_ABL=full, ResNet temporal encoder).
  `final_probs.py` scores the network outputs.
- AEC and MinExplain are rerun with the auto-extended grids (`f0/metrics_kb.json`).
- Extra run: VioExplain without composition (f0), `f0/metrics_B.json`, collected in `metrics_extra.json`.
- `stats2.py`: effective-event statistics. `collect2.py`: delivery files (rerun at any time).

## Results (`results/simproc_v1/{dist,csth,third}/`)

- `metrics_main.json`: comparison (A), method name to {macro_f1, normal_named, pair_f1, triple_f1, quad_f1, attr_f1,
  k1_f1, n_pair, n_triple, n_quad, source}. Every baseline learns from normal and single-fault fit runs (f0), VioExplain
  composes counterfactual windows from the same runs (f100t).
- `metrics_samecomp.json`: comparison (B), BR-LGBM, CC-LGBM and ML-CNN with the compositions of VioExplain (f100t), and
  VioExplain.
- `dataset_stats.json`, `dataset_truth_stats.json`: sizes, variables, faults, relation units, effective-event statistics.
- `f0/`, `f100t/`: logs, per-method metrics, per-window records. `gpu/`: GPU outputs.

## Dataset statistics

| | DIST | CSTH | DTS200 (`third`) |
|---|---|---|---|
| recorded variables | 20 | 8 | 7 |
| fault types (all effective) | 10 | 10 | 10 |
| sampling period, window | 1 min, 64 min | 1 s, 64 s | 2 s, 128 s |
| functional-constraint units kept (R^2 >= 0.9 on calibration windows) | 1 | 1 | 0 |
| description units (sensors plus relation units) | 21 | 9 | 7 |
| fit and calibration windows per class | 2100, 700 | 2100, 700 | 2100, 700 |
| single-fault evaluation windows (of which normal) | 54384 (4944) | 54384 (4944) | 54384 (4944) |
| single-fault test-calibration windows | 11616 | 11616 | 11616 |
| pair sets, windows | 45, 16200 | 45, 16200 | 45, 16200 |
| triple sets, windows | 50, 9000 | 50, 9000 | 50, 9000 |
| four-fault sets, windows (superposed) | 40, 14400 | 40, 14400 | 40, 14400 |
| lowest share of single-fault evaluation windows in which the fault is effective | 0.857 (fault 5) | 0.757 (fault 1) | 0.735 (fault 8) |
| share of windows in which every event of the set is effective: pairs, triples, four | 0.940, 0.885, 0.870 | 0.745, 0.555, 0.481 | 0.876, 0.734, 0.514 |
| mean size of the effective-event truth: pairs, triples, four | 1.94, 2.88, 3.87 | 1.74, 2.45, 3.36 | 1.88, 2.70, 3.39 |
| runs with an interlock trip | 0 | 0 | 0 |

Windows removed as invalid: none (no trip and no frozen record in any run).

## Results

Columns: gated macro F1 on single-fault evaluation windows, share of normal evaluation windows with a named event,
event-set F1 on single-fault windows (k=1), on two, three and four simultaneous faults, attribution F1 (mean over pairs,
triples and four faults). AEC and MinExplain rows are the reruns with the auto-extended grids (`f0/metrics_kb.json`).
All tuned parameters that decide how many events a method names are interior or at a natural bound (zero) of their
grids: BR, CC and ML-CNN thresholds by `tune_threshold`, PCA-RBC continuation 8, 256 and 4 inside 0.25 to 1e12, AEC and
MinExplain by `tune_axes` (on CSTH the exact-representation grid was extended down to 0.3 and lambda down to
0.125, on DTS200 lambda down to 0.125, before the optimum was interior).

### dist (dist): comparison (A), baselines fitted on single-fault runs

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| PCA-RBC | 0.925 | 2.6 | 0.889 | 0.724 | 0.702 | 0.739 | 0.602 |
| FDA | 0.984 | 4.2 | 0.974 | 0.682 | 0.517 | 0.413 | 0.509 |
| RF | 0.987 | 2.6 | 0.982 | 0.682 | 0.515 | 0.412 | 0.509 |
| XGB | 0.990 | 7.1 | 0.973 | 0.685 | 0.519 | 0.413 | 0.627 |
| LGBM | 0.991 | 2.5 | 0.979 | 0.684 | 0.518 | 0.413 | 0.608 |
| MiniRocket | 0.989 | 2.1 | 0.980 | 0.685 | 0.520 | 0.414 | 0.652 |
| MultiRocket | 0.980 | 1.2 | 0.980 | 0.683 | 0.519 | 0.413 | 0.646 |
| QUANT | 0.991 | 3.3 | 0.979 | 0.685 | 0.518 | 0.412 | 0.629 |
| 1D-CNN | 0.995 | 4.6 | 0.974 | 0.674 | 0.494 | 0.360 | 0.503 |
| LSTM | 0.992 | 7.0 | 0.971 | 0.662 | 0.510 | 0.410 | 0.492 |
| ResNet | 0.995 | 4.2 | 0.974 | 0.679 | 0.518 | 0.411 | 0.495 |
| InceptionTime | 0.996 | 1.1 | 0.978 | 0.681 | 0.520 | 0.414 | 0.508 |
| MantisV2 | 0.984 | 1.2 | 0.981 | 0.681 | 0.517 | 0.410 | 0.676 |
| BR-LGBM | 0.980 | 2.2 | 0.976 | 0.940 | 0.883 | 0.896 | 0.788 |
| CC-LGBM | 0.977 | 2.4 | 0.973 | 0.941 | 0.889 | 0.899 | 0.787 |
| ML-CNN | 0.986 | 2.3 | 0.973 | 0.816 | 0.592 | 0.426 | 0.580 |
| AEC | 0.974 | 4.6 | 0.960 | 0.858 | 0.748 | 0.715 | 0.699 |
| MinExplain | 0.967 | 7.6 | 0.934 | 0.860 | 0.777 | 0.745 | 0.692 |
| VioExplain | 0.993 | 7.0 | 0.956 | 0.965 | 0.964 | 0.964 | 0.789 |

### dist: comparison (B), set-valued methods with the compositions of VioExplain

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| BR-LGBM | 0.984 | 2.4 | 0.976 | 0.974 | 0.955 | 0.950 | 0.801 |
| CC-LGBM | 0.984 | 2.5 | 0.976 | 0.974 | 0.955 | 0.951 | 0.801 |
| ML-CNN | 0.991 | 1.9 | 0.975 | 0.974 | 0.957 | 0.926 | 0.791 |
| VioExplain | 0.993 | 7.0 | 0.956 | 0.965 | 0.964 | 0.964 | 0.789 |

| Extra: VioExplain without composition (f0) | 0.996 | 3.0 | 0.956 | 0.923 | 0.885 | 0.874 | 0.770 |

windows: {"n_pair_sets": 45, "n_pair": 16200, "n_triple_sets": 50, "n_triple": 9000, "n_quad_sets": 40, "n_quad": 14400}, missing: none

### csth (csth): comparison (A), baselines fitted on single-fault runs

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| PCA-RBC | 0.716 | 2.8 | 0.769 | 0.614 | 0.533 | 0.445 | 0.617 |
| FDA | 0.934 | 2.0 | 0.927 | 0.716 | 0.580 | 0.464 | 0.641 |
| RF | 0.965 | 2.8 | 0.955 | 0.736 | 0.600 | 0.470 | 0.670 |
| XGB | 0.983 | 4.0 | 0.947 | 0.727 | 0.588 | 0.465 | 0.669 |
| LGBM | 0.983 | 3.9 | 0.947 | 0.721 | 0.581 | 0.463 | 0.659 |
| MiniRocket | 0.964 | 3.5 | 0.925 | 0.708 | 0.547 | 0.455 | 0.639 |
| MultiRocket | 0.959 | 4.3 | 0.922 | 0.706 | 0.542 | 0.434 | 0.623 |
| QUANT | 0.988 | 2.0 | 0.947 | 0.713 | 0.584 | 0.462 | 0.656 |
| 1D-CNN | 0.990 | 2.7 | 0.945 | 0.728 | 0.585 | 0.451 | 0.625 |
| LSTM | 0.982 | 3.1 | 0.936 | 0.721 | 0.586 | 0.456 | 0.624 |
| ResNet | 0.991 | 1.7 | 0.945 | 0.729 | 0.591 | 0.466 | 0.612 |
| InceptionTime | 0.990 | 3.1 | 0.941 | 0.737 | 0.600 | 0.470 | 0.640 |
| MantisV2 | 0.985 | 2.0 | 0.944 | 0.689 | 0.554 | 0.448 | 0.665 |
| BR-LGBM | 0.971 | 3.0 | 0.939 | 0.846 | 0.745 | 0.681 | 0.754 |
| CC-LGBM | 0.965 | 3.8 | 0.937 | 0.854 | 0.768 | 0.701 | 0.762 |
| ML-CNN | 0.979 | 4.3 | 0.934 | 0.789 | 0.654 | 0.522 | 0.654 |
| AEC | 0.907 | 3.0 | 0.960 | 0.807 | 0.698 | 0.695 | 0.700 |
| MinExplain | 0.918 | 2.9 | 0.941 | 0.840 | 0.795 | 0.785 | 0.736 |
| VioExplain | 0.991 | 3.8 | 0.926 | 0.913 | 0.876 | 0.880 | 0.785 |

### csth: comparison (B), set-valued methods with the compositions of VioExplain

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| BR-LGBM | 0.970 | 1.3 | 0.945 | 0.920 | 0.870 | 0.857 | 0.787 |
| CC-LGBM | 0.968 | 1.5 | 0.944 | 0.920 | 0.868 | 0.856 | 0.786 |
| ML-CNN | 0.980 | 2.8 | 0.941 | 0.918 | 0.861 | 0.846 | 0.781 |
| VioExplain | 0.991 | 3.8 | 0.926 | 0.913 | 0.876 | 0.880 | 0.785 |

| Extra: VioExplain without composition (f0) | 0.992 | 3.6 | 0.928 | 0.852 | 0.766 | 0.725 | 0.768 |

windows: {"n_pair_sets": 45, "n_pair": 16200, "n_triple_sets": 50, "n_triple": 9000, "n_quad_sets": 40, "n_quad": 14400}, missing: none

### third (dts200): comparison (A), baselines fitted on single-fault runs

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| PCA-RBC | 0.770 | 7.3 | 0.758 | 0.523 | 0.500 | 0.446 | 0.471 |
| FDA | 0.958 | 4.4 | 0.930 | 0.607 | 0.474 | 0.377 | 0.531 |
| RF | 0.983 | 5.7 | 0.948 | 0.597 | 0.465 | 0.392 | 0.543 |
| XGB | 0.984 | 2.9 | 0.952 | 0.598 | 0.461 | 0.386 | 0.546 |
| LGBM | 0.984 | 4.5 | 0.950 | 0.598 | 0.470 | 0.394 | 0.552 |
| MiniRocket | 0.976 | 4.0 | 0.949 | 0.612 | 0.460 | 0.385 | 0.539 |
| MultiRocket | 0.941 | 4.4 | 0.923 | 0.596 | 0.435 | 0.368 | 0.513 |
| QUANT | 0.984 | 4.5 | 0.950 | 0.579 | 0.431 | 0.354 | 0.512 |
| 1D-CNN | 0.985 | 3.2 | 0.952 | 0.582 | 0.427 | 0.365 | 0.531 |
| LSTM | 0.985 | 2.2 | 0.954 | 0.569 | 0.439 | 0.356 | 0.531 |
| ResNet | 0.985 | 4.9 | 0.950 | 0.588 | 0.441 | 0.359 | 0.541 |
| InceptionTime | 0.985 | 5.3 | 0.949 | 0.602 | 0.467 | 0.385 | 0.566 |
| MantisV2 | 0.984 | 4.9 | 0.949 | 0.597 | 0.455 | 0.354 | 0.532 |
| BR-LGBM | 0.963 | 2.9 | 0.943 | 0.680 | 0.566 | 0.479 | 0.658 |
| CC-LGBM | 0.969 | 2.9 | 0.945 | 0.685 | 0.575 | 0.504 | 0.669 |
| ML-CNN | 0.948 | 4.0 | 0.935 | 0.616 | 0.443 | 0.396 | 0.552 |
| AEC | 0.705 | 1.8 | 0.812 | 0.648 | 0.583 | 0.532 | 0.649 |
| MinExplain | 0.727 | 1.2 | 0.819 | 0.635 | 0.556 | 0.492 | 0.633 |
| VioExplain | 0.984 | 4.8 | 0.934 | 0.870 | 0.797 | 0.774 | 0.791 |

### third: comparison (B), set-valued methods with the compositions of VioExplain

| Method | Macro F1 | Normal (%) | k=1 | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|
| BR-LGBM | 0.973 | 3.0 | 0.947 | 0.890 | 0.799 | 0.760 | 0.810 |
| CC-LGBM | 0.975 | 3.3 | 0.948 | 0.887 | 0.791 | 0.746 | 0.800 |
| ML-CNN | 0.968 | 3.1 | 0.941 | 0.868 | 0.771 | 0.685 | 0.751 |
| VioExplain | 0.984 | 4.8 | 0.934 | 0.870 | 0.797 | 0.774 | 0.791 |

| Extra: VioExplain without composition (f0) | 0.986 | 2.0 | 0.939 | 0.778 | 0.683 | 0.655 | 0.696 |

windows: {"n_pair_sets": 45, "n_pair": 16200, "n_triple_sets": 50, "n_triple": 9000, "n_quad_sets": 40, "n_quad": 14400}, missing: none
