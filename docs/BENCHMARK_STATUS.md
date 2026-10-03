# Benchmark status

All nine predefined baseline run units are complete. The additional locked event-matching panel is also complete. These statements describe the fixed experiments, not a state-of-the-art performance claim.

| Released batch | Scope | State |
|---|---|---|
| [SMD statistical detection and BARO component](../results_and_logs/summary/formal_v1/cpu_smd/COMPLETE.json) | 28 machines | Completed |
| [SKAB statistical detection](../results_and_logs/summary/formal_v1/cpu_skab/COMPLETE.json) | 34 files | Completed |
| [TEP statistical detection](../results_and_logs/summary/formal_v1/cpu_tep/COMPLETE.json) | 22 trajectories | Completed |
| [TreeSHAP-IsolationForest](../results_and_logs/summary/formal_v1/shap_smd/COMPLETE.json) | 28 machines, three model seeds | Completed |
| [TEP supervised classifiers](../results_and_logs/summary/formal_v1/tep_classifiers/COMPLETE.json) | 22 trajectories, 267 windows | Completed |
| [TranAD on SMD](../results_and_logs/summary/formal_v1/tranad_smd/COMPLETE.json) | 28 machines, three seeds | Completed |
| [TranAD on SKAB](../results_and_logs/summary/formal_v1/tranad_skab/COMPLETE.json) | 34 files, three seeds | Completed |
| [TranAD on TEP](../results_and_logs/summary/formal_v1/tranad_tep/COMPLETE.json) | 22 trajectories, three seeds | Completed |
| [SARAD on SMD](../results_and_logs/summary/formal_v1/sarad_smd/COMPLETE.json) | 28 machines, three seeds | Completed |
| [Event matching with AEC-Prototype, AEC-CostAdapted and MinExplain](../results_and_logs/summary/event_matching_v1/COMPLETE.json) | Same 44 TEP files; interval, raw temporal and frozen Chronos-2 comparisons | Completed |

The [formal report](FORMAL_BENCHMARK_REPORT_zh.md) and [event report](EVENT_MATCHING_REPORT_zh.md) keep detection, conditional attribution and event-set explanation distinct. SMD confidence intervals resample complete machines; SKAB and TEP intervals describe their recorded files or faults and do not establish cross-factory generalization.

The frozen-model MinExplain variant improved its matched raw-temporal control but remained below stronger statistical classifiers. Normal false explanations and missing-knowledge rejection remain unresolved. GDN, MOMENT, SWaT, WADI, HAI and full Exathlon evaluation were not run in this fixed batch.
