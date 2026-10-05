# Hydraulic test rig (HYD)

These scripts run VioExplain and the baselines on the hydraulic condition monitoring data of Helwig et al. (2015, UCI repository, id 447). The data are downloaded from the repository into `data/hydraulic/` under the root set at the top of `hyd_common.py` (`/path/to/vioexplain` is a placeholder).

## Data and events

`hyd_prepare.py` averages the 17 sensors to 1 Hz, so one load cycle is one window of 60 samples and 17 variables. The data hold 2205 cycles. The four events are a degraded cooler (cooler condition below 100), a lagging valve (valve condition below 100), internal pump leakage (leakage above 0) and accumulator pressure loss (pressure below 130). The truth of a cycle is its set of degraded components.

The split is stratified by condition group. Within a group the cycles are ordered by a hash of the salt and the cycle index, and the rank modulo 5 gives the role. Fault-free and single-fault groups follow the pattern test, test calibration, calibration, fit, fit. Groups with several faults are used for testing. The three splits use `HYD_SALT=vioexplain-hyd` with `HYD_TAG` empty, `HYD_SALT=vioexplain-hyd-b` with `HYD_TAG=_splitB` and `HYD_SALT=vioexplain-hyd-c` with `HYD_TAG=_splitC`.

The data have no paired runs. A composition adds the deviations of randomly drawn single-fault cycles from a fault-free cycle, and the knowledge of an event is built against the fault-free fit mean.

## Scripts

| Script | Role |
|---|---|
| `hyd_prepare.py` | Resampling of the raw sensor files to `hyd_1hz.npz` |
| `hyd_common.py` | Data, split, window descriptions, event knowledge, compositions, calibration rule and evaluation |
| `hyd_run.py` | RF, XGB and LGBM (`top1`), FDA, PCA-RBC, BR, CC, AEC, MinExplain |
| `hyd_deep.py` | 1D-CNN, LSTM, ResNet, InceptionTime, ML-CNN, MiniRocket, MultiRocket, QUANT |
| `hyd_mantis.py`, `hyd_mantis_embed.py` | MantisV2 embedding classifier |
| `hyd_tnet.py` | 1D-CNN and ResNet diagnosers with saved weights, the temporal evidence of VioExplain |
| `hyd_final.py` | VioExplain with compositions |
| `hyd_rescore.py` | Scoring of a saved ML-CNN under the shared calibration rule |
| `hyd_final_merge.py` | Mean and standard deviation over the three splits |

`HYD_MODE=f0` fits every method on fault-free and single-fault cycles. `HYD_MODE=f100t` adds the compositions. `HYD_OUTSUB` names the subdirectory of `results/hyd_v1/` that receives the outputs.

## Commands

The commands below are for one split and are repeated with the salt and tag of the other two.

```bash
python hyd_prepare.py
# single-label and monitoring baselines
HYD_MODE=f0 python hyd_run.py top1 FDA
HYD_MODE=f0 python hyd_deep.py 1D-CNN LSTM ResNet InceptionTime QUANT
HYD_MODE=f100t python hyd_run.py ours
# Rocket classifiers and MantisV2
HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_deep.py MiniRocket MultiRocket
HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_mantis.py prep
PYTHONPATH=.. V3_MANTIS=/path/to/mantis_weights python hyd_mantis_embed.py cuda:0
HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_mantis.py score
# VioExplain
HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_tnet.py 1D-CNN
HYD_MODE=f0 HYD_OUTSUB=final/ python hyd_tnet.py ResNet
HYD_MODE=f100t HYD_OUTSUB=final/ HYD_FUNC=0 python hyd_final.py
HYD_MODE=f100t HYD_OUTSUB=final/ HYD_FUNC=1 python hyd_final.py
# knowledge-based and monitoring methods under the shared calibration rule
HYD_MODE=f0 HYD_CAL=one HYD_CALSET=cal HYD_OUTSUB=final_uniform2/ python hyd_run.py PCA-RBC AEC MinExplain
# multi-label methods under the shared calibration rule with the empty-set rule
HYD_MODE=f0 HYD_CAL=one HYD_CALSET=cal HYD_OUTSUB=final_onerule/ python hyd_deep.py ML-CNN
HYD_MODE=f100t HYD_CAL=one HYD_CALSET=cal HYD_OUTSUB=final_onerule/ python hyd_deep.py ML-CNN
HYD_MODE=f0 HYD_CAL=one HYD_CALSET=cal HYD_MLGATE=1 HYD_OUTSUB=final_gate/ python hyd_run.py BR CC
HYD_MODE=f100t HYD_CAL=one HYD_CALSET=cal HYD_MLGATE=1 HYD_OUTSUB=final_gate/ python hyd_run.py BR CC
HYD_MODE=f0 HYD_CAL=one HYD_CALSET=cal HYD_MLGATE=1 HYD_OUTSUB=final_gate/ python hyd_rescore.py ML-CNN final_onerule/
HYD_MODE=f100t HYD_CAL=one HYD_CALSET=cal HYD_MLGATE=1 HYD_OUTSUB=final_gate/ python hyd_rescore.py ML-CNN final_onerule/
# tables
python hyd_final_merge.py
```

`hyd_mantis.py prep` writes `mantis_in.npz` to its output directory. `hyd_mantis_embed.py` reads `data/hydraulic/mantis_in<TAG>.npz` and writes `results/hyd_v1/mantis_emb<TAG>.npz`, and `hyd_mantis.py score` reads `mantis_emb.npz` from its output directory, so the two files are copied between these places. `hyd_mantis_embed.py` imports the loader `smoke_v3.py` of the parent folder, which `PYTHONPATH=..` makes visible.

## Calibration and outputs

Every parameter that decides how many events a baseline names is chosen on the fault-free and single-fault test-calibration cycles together with the composed calibration cycles on which the addition decision of VioExplain is fitted (`HYD_CAL=one`). A multi-label method names nothing on a cycle whose largest event probability does not exceed the split-conformal threshold of the fault-free test-calibration cycles (`HYD_MLGATE=1`). No test cycle and no measured cycle with several faults enters a fitted or calibrated quantity.

`hyd_final.py` runs VioExplain with and without functional constraint units. `hyd_final_merge.py` keeps the variant with the higher mean calibration criterion over the three splits, where the criterion uses test-calibration cycles and their compositions only.

`hyd_final_merge.py` writes `metrics_f0_final_3splits.json` and `metrics_f100t_final_3splits.json`. They are the files `results/hyd/single_fault_knowledge.json` and `results/hyd/with_compositions.json` of this repository, restricted to the methods of the paper. Each entry holds the mean, the standard deviation and the number of splits. `k2_setF1`, `k3_setF1` and `k4_setF1` are SF1 on cycles with two, three and four faults, and `multi_setF1` is SF1 on all cycles with two or more faults. The script also collects tuning records and diagnostic configurations from further output directories when they exist.
