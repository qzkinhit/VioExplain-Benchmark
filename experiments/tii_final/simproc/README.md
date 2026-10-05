# Simulated processes with paired concurrent-fault runs (CSTR, quadruple tank)

Two closed-loop process simulators generate data in the layout of the TEP runs, and the TEP protocol of `scripts/common.py` runs on them with only the data paths and sizes switched (`V3_DATA=cstr` or `V3_DATA=qtank`). Code is in `scripts/simproc/` on cpu-server and gpu-server (identical copies) and archived in `VioExplain/paper/research/v3/final_scripts/simproc/`. Results are in `results/simproc_v1/<process>/` on cpu-server.

## Simulators

- `sim_cstr.py` is a jacketed exothermic CSTR (reactant concentration, reactor and jacket temperature, liquid level) with a level loop on the outlet valve and a temperature cascade (outer loop sets the jacket temperature, inner loop moves the coolant valve). Parameter values are the commonly quoted ones for this benchmark, written from memory and not checked against a publication. 13 recorded variables, 12 fault types (process, disturbance, sensor in and out of loop, actuator). Time unit minute, one sample per minute.
- `sim_qtank.py` is the four-tank process of Johansson (2000) in the minimum-phase setting with two decentralized PI level loops, parameter values written from memory. 8 recorded variables, 10 fault types (pump, valve split, leak, clogging, sensor in and out of loop). One sample per 5 s.
- Both add first-order autoregressive disturbances and Gaussian measurement noise. Every random number of a run is a function of its seed only (disturbance and noise streams and the magnitude of each fault type), so runs with the same seed and different fault sets are paired runs and the runs of all subsets of a fault set share their noise.
- Interlocks exist (temperature and level limits for the CSTR, overflow and low level for the tanks), and no run of the generated data trips them.
- `sim_check.py cstr|qtank` prints, for each single fault, the shift of window statistics against the paired normal run and checks determinism and pairing.
- The controller settings of the CSTR cascade were chosen from a linear stability scan of the model (damping of the closed-loop poles), after a single temperature loop was found to oscillate and trip the plant under several faults.

## Data (`gen_data.py PROCESS`, `data/simproc/<process>/`)

| Part | Runs per class or set | Samples | Onset | Role |
|---|---|---|---|---|
| training_arrays | 400 (seeds 0 to 399) | 512 | 64 | runs 0 to 299 fit, 300 to 399 calibration, 7 windows each |
| testing_arrays | 500 (seeds 100000+) | 960 | 160 | hash split into test calibration and evaluation, 12 windows each |
| test_pairs | 30 seeds per set | 960 | 160 | normal, every single fault, every pair |
| test_triples | 15 seeds per set | 960 | 160 | 60 (CSTR) or 50 (tank) random triples and all their subsets |

Four-fault windows are superposed from the paired single-fault runs of test_pairs, as on TEP. The window length is 64 samples. Generation is deterministic, and the arrays generated on cpu-server and on gpu-server are bitwise identical.

## Loader contract (for other datasets)

`common.py` reads, for `V3_DATA=<name>`, the directory `data/simproc/<name>/` (cpu-server `/path/to/vioexplain/`, gpu-server `/path/to/vioexplain/`).

- `dataset.json` with the fields `W`, `C` (classes including normal), `M` (variables), `EFF` (effective faults, used for four-fault superpositions and for the composed calibration windows), `onset_train`, `onset_test`, `tri_sets` (composed triple sets under f100t), `n_fit`, `n_cal`, `n_test`.
- `training_arrays/class_XX.npy` of shape (n_fit + n_cal, T, M) for XX = 00 (normal) to C-1. Run r of every class must be paired with run r of class 00 (same noise before and after the onset). Runs 0 to n_fit-1 are fit runs, the next n_cal are calibration runs.
- `testing_arrays/class_XX.npy` of shape (n_test, T, M), also paired by run index. The test calibration runs are those with `sha256('vioexplain-v3-test-split:%d' % r) % 5 == 0`.
- `test_pairs/set_<key>.npy` of shape (seeds, T, M) for key `normal`, every single fault `e` and every pair `a+b` (a < b), all with the same seeds, and `test_pairs/meta.json` with `onset`, `sets` and `runs[key]` (a list with `shutdown_sample` per seed, -1 if none). Windows after a shutdown are removed.
- `test_triples/` in the same format with every triple `a+b+c` and all its subsets.
- A window counts as frozen when at least ceil(30 M / 52) variables are constant (30 of 52 on TEP).

## Protocol (TEP reference scripts of 2026-10-05, 13:55)

- `common.py` is built from the TEP reference `common.py` by `patch_common.py` (data switch only). Fit and calibration runs, the test hash split, the description, the functional-constraint units, the effective-event truth from paired runs, the attribution metric and the composition recipe are unchanged. All faults are effective. The number of composed triple sets is scaled to the number of fault pairs (104 for the CSTR, 71 for the tank, against 300 for 190 TEP pairs).
- `final_base.py`, `final_rocket.py` and `final_fuse5.py` are the TEP reference scripts. `final_probs.py` is the reference with the number of output columns of the multi-label CNN taken from the data.
- Every parameter that decides how many events a method names is chosen on single-fault test-calibration windows and on composed calibration windows (one rule for every method). The multi-label threshold grid runs from 0.005 to 0.995 and is extended outward while the optimum sits at an end, and the AEC tau grid runs from -5 to 1.5.
- Every method shares the split-conformal empty-set rule on the fault-free test-calibration windows (one window per run, alpha 0.05). Single-label methods name their top event only above it. BR, CC and ML-CNN name nothing when their largest event probability does not exceed the conformal threshold of that probability (`ml_gate`, `ml_sets`), and their tuned threshold applies after this gate, also inside the threshold search.
- `gpu_simproc.py` (gpu-server, GPUs 3 to 5) trains the 1D-CNN, ResNet, LSTM, InceptionTime and the multi-label CNNs with the TEP architectures and training settings, and runs MantisV2 (frozen embedding, PCA, LightGBM head). ML-CNN under f100t is trained on exactly the compositions that BR and CC receive (`common.AUG`).
- VioExplain is run label B of the TEP paper (learned addition decision without the step feature, ResNet temporal encoder, functional-constraint units).

Comparison (A) follows the main table: every baseline learns from normal and single-fault fit runs (f0), and VioExplain composes counterfactual windows from the same runs (f100t). Comparison (B) follows the composition table: BR, CC and ML-CNN receive the compositions of VioExplain (f100t).

## Launchers and results

1. `gen_data.py PROCESS` on cpu-server and on gpu-server.
2. `launch_cpu.sh` (cpu-server) runs the CPU baselines of `final_base.py` and `final_rocket.py`.
3. `launch_gpu.sh` (gpu-server) runs `gpu_simproc.py`. Its outputs go to `results/simproc_v1/<process>/gpu/` and are copied to the same path on cpu-server (`relay.sh`, run on a machine that reaches both servers).
4. `launch_after_gpu.sh PROCESS JOB` (cpu-server) scores the GPU outputs (`final_probs.py`) and runs VioExplain (`final_fuse5.py`) once the files are on cpu-server.
5. `collect_simproc.py` (cpu-server) writes `results/simproc_v1/<process>/summary.json` and `summary.md` (both comparisons, gains over the strongest multi-label method and over the strongest baseline) and `results/simproc_v1/summary_all.json`. It can be rerun at any time, and rows whose run has not finished are empty.

`results/simproc_v1/<process>/{f0,f100t}/metrics_<method>.json` hold the metrics of every run, written when the run ends, with logs and per-window records next to them. Runs superseded by protocol updates are kept: `superseded_grid1303/` (BR, CC, ML-CNN and AEC with the fixed grids of 13:03) and `superseded_nogate1350/` (BR, CC and ML-CNN without the empty-set gate).

MultiRocket crashes with a segmentation fault on cpu-server when numba runs with two or four threads (logs `MultiRocket_f0_try1.out`, `_try2.out`). The rows in the tables come from gpu-server (eight threads, `final_rocket.py` as above, ridge fit 52 min for the CSTR and 38 min for the tank), whose metrics, records, probabilities and log were copied to cpu-server (`logs/MultiRocket_f0_hit.out`). `launch_multirocket.sh` runs it on cpu-server with one numba thread and 24 BLAS threads, and those runs were stopped once the gpu-server runs had finished.
