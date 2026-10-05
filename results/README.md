# Result files of the paper

This directory holds the result files behind the tables, figures and quoted numbers of the paper. Every file is the JSON output of a run, restricted to the methods of the paper. `python results/print_main_table.py` prints the main table from these files.

## Layout

| Path | Content |
|---|---|
| `tep/main/<Method>.json` | TEP-R and TEP-C, every method fitted on fault-free and single-fault runs, VioExplain with its counterfactual compositions |
| `tep/same_compositions/{BR,CC,ML-CNN}.json` | TEP-C, multi-label methods fitted with the compositions of VioExplain |
| `tep/coverage/{010,025,050,075}/` | TEP-C, 10, 25, 50 and 75 percent of the fault pairs composed, for VioExplain, BR, CC and ML-CNN |
| `tep/ablation/` | VioExplain without the temporal posterior, without the learned operator, without paired runs, without functional constraint units |
| `tep/unknown_events.json` | Unknown score of VioExplain, MDS, MSP and Energy, the held-out TEP error types |
| `tep/update_curve.json`, `tep/update_baselines.json` | Knowledge update on held-out faults, VioExplain and the baselines |
| `tep/certificate.json` | Recovery certificate on two-fault and three-fault windows |
| `tep/timing.json` | CPU time per window |
| `tep/dataset_statistics.json`, `tep/addition_statistics.json` | Window counts and violation statistics quoted in the text |
| `hyd/single_fault_knowledge.json` | HYD, every baseline, mean and standard deviation over three splits |
| `hyd/with_compositions.json` | HYD, VioExplain and the multi-label methods with the same compositions |
| `simulated/<process>/` | `metrics_main.json` (or `summary.json` for CSTR and QTank) and `dataset_stats.json` of the nine simulated processes |

In a file of `tep/`, the entry whose value is a dictionary holds the metrics of the method. `single_macroF1_gated` is MF1 on TEP-R. `pair_eff`, `triple_eff` and `quad_eff` are SF1 on two, three and four faults. `pair_attrF1`, `triple_attrF1` and `quad_attrF1` are AF1. `normal_named` is the naming rate on fault-free windows. `pair_eff_unseen` is SF1 on fault pairs that were never composed.

## Tables

| Item | Files and fields |
|---|---|
| Dataset table, sizes of TEP-R and TEP-C | `tep/dataset_statistics.json`: `n_single_eval`, and `n_pair` + `n_triple` + `n_quad` |
| Dataset table, simulated processes | `simulated/<process>/dataset_stats.json`: `n_fault_types`, `n_variables`, and `single_eval_windows` + `n_pair` + `n_triple` + `n_quad`. The sizes of CSTR and QTank come from the run logs and have no file here. |
| Main table, TEP-R | `tep/main/<Method>.json`: `single_macroF1_gated` |
| Main table, TEP-C (SF1) | `tep/main/<Method>.json`: mean of `pair_eff`, `triple_eff`, `quad_eff` |
| Main table, TEP-C (AF1) | `tep/main/<Method>.json`: mean of `pair_attrF1`, `triple_attrF1`, `quad_attrF1` |
| Main table, HYD | `hyd/single_fault_knowledge.json` for the baselines and `hyd/with_compositions.json` for VioExplain: first element of `multi_setF1` |
| Main table, CSTR and QTank | `simulated/{cstr,qtank}/summary.json`, `comparison_A`: mean of `pairs`, `triples`, `four` |
| Main table, DIST, CSTH, DTS200 | `simulated/<process>/metrics_main.json`: mean of `pair_f1`, `triple_f1`, `quad_f1` |
| Main table, FERM and HEX | `simulated/<process>/metrics_main.json`: `macro_f1` |
| Main table, EVAP and PH | `simulated/<process>/metrics_main.json`: `attr_f1` |
| Main table, rank | Mean rank over the 13 columns, computed by `print_main_table.py` |

## Figures

| Item | Files and fields |
|---|---|
| Results figure, panel (a), SF1 against the number of faults on TEP-C | `tep/main/<Method>.json`: `single_eff_nonempty`, `pair_eff`, `triple_eff`, `quad_eff` |
| Results figure, panel (b), coverage | `tep/coverage/<percent>/<Method>.json`: `pair_eff` and `pair_eff_unseen`. The 100 percent point is `tep/main/VioExplain.json` and `tep/same_compositions/`. |
| Results figure, panel (c), HYD | `hyd/single_fault_knowledge.json` and `hyd/with_compositions.json` (VioExplain): `k2_setF1`, `k3_setF1`, `multi_setF1` |
| Results figure, panel (d), knowledge update | `tep/update_baselines.json`: `mean` → method → `with_known` against `n` |
| Composition and ablation figure, panel (a) | `tep/same_compositions/{BR,CC,ML-CNN}.json` and `tep/main/VioExplain.json`: `pair_eff`, `triple_eff`, `quad_eff` |
| Composition and ablation figure, panel (b) | `tep/main/VioExplain.json` and `tep/ablation/*.json`: `single_macroF1_gated`, `pair_eff`, `triple_eff`, `quad_eff` |

The motivation figure shows one paired run of TEP-R and one two-fault window of TEP-C and has no result file. The framework figure is a diagram.

## Numbers quoted in the text

| Statement | Files and fields |
|---|---|
| Violations per window of IDV(1), fault-free windows with a violation, share of windows of IDV(3), IDV(9), IDV(15) with a caused violation | `tep/dataset_statistics.json`: `f1_caused_mean`, `f1_all_mean`, `f1_sensors_mean`, `normal_mean`, `normal_share`, `sub_share` |
| First selection is a true fault, second fault added correctly or falsely without compositions | `tep/addition_statistics.json` |
| Gain over the strongest multi-label method on two, three and four faults | `tep/main/VioExplain.json` against `tep/main/{BR,CC,ML-CNN}.json` |
| Gain over AEC and MinExplain on four faults | `quad_eff` of `tep/main/VioExplain.json`, `AEC.json`, `MinExplain.json` |
| Contribution of the temporal posterior to MF1 | `tep/main/VioExplain.json` and `tep/ablation/no_temporal_posterior.json`: `single_macroF1_gated` |
| Naming rate of the multi-label methods relative to VioExplain | `normal_named` of `tep/main/` |
| Gain on never-composed pairs at 50 percent coverage | `tep/coverage/050/`: `pair_eff_unseen` |
| False flags on windows of two known faults against MDS | `tep/unknown_events.json`: `mean` → `falseflag_K2` |
| Windows needed to reach 80 percent of the SF1 under full knowledge, value after 128 windows | `tep/update_curve.json`: `refit_br` → `n_to_80_ratio_mean`, `curve` → `128` → `ratio_mean` |
| Gain over the best baseline after 16 and 128 windows | `tep/update_baselines.json`: `mean` → method → `with_known` |
| Gains on HYD | `multi_setF1` of `hyd/with_compositions.json` (VioExplain) against `hyd/single_fault_knowledge.json`, and against BR, CC, ML-CNN inside `hyd/with_compositions.json` |
| Gains on the simulated processes | Columns of the main table |
| Ablation losses on four faults | `quad_eff` of `tep/main/VioExplain.json` minus `tep/ablation/*.json` |
| Share of windows that satisfy the certificate, share of recovered windows that it covers, shares by interaction strength | `tep/certificate.json`: `pair_cert_share`, `pair_recovered_cert_share`, `pair_cert_share_low_interaction`, `pair_cert_share_high_interaction` |
| Time per window | `tep/timing.json`: `methods` |

## Method names in the files

RF, XGB and LGBM appear as `RF-top1`, `XGB-top1` and `MC-LGBM-top1` in the files of `tep/` and `hyd/`. BR and CC appear as `BR-LGBM` and `CC-LGBM`. The entry of VioExplain in `tep/` carries the tag of its run (`VioExplain-nostep-B`).
