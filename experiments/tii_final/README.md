# Experiment scripts of the paper

These scripts ran the experiments on the Tennessee Eastman process, and the `simproc*/` folders hold the simulators and runners of the simulated processes. Set the data and result directories at the top of `common.py` (`/path/to/vioexplain` and `/home/user` are placeholders).

| Script | Role |
|---|---|
| `common.py` | Data access, window descriptions with functional-constraint units, event knowledge from paired runs, counterfactual compositions, evaluation and the split-conformal thresholds shared by all methods |
| `final_fuse5.py` | VioExplain with the deduction-consistent scorer, the learned addition decision, the temporal posterior in the first selection and the calibrated thresholds, and its ablations through `V3_ABL`, `V3_NOTWIN` and `V3_FUNC` |
| `final_base.py` | RF, XGB, LGBM, FDA, PCA-RBC, BR, CC, AEC, MinExplain |
| `final_rocket.py` | MiniRocket, MultiRocket, QUANT |
| `gpu_run.py`, `gpu_run2.py`, `gpu_save.py`, `gpu_save2.py`, `gpu_save3.py` | Training of 1D-CNN, LSTM, ResNet, InceptionTime, ML-CNN and the MantisV2 embedding classifier |
| `final_probs.py` | Scoring of the saved network probabilities with the shared protocol |
| `final_run.py`, `final_ablate.py` | Reference runner of the shared protocol and its coverage modes |
| `final_update_base.py`, `fold_part2.py`, `agg_update_base.py` | Baselines of the knowledge-update experiment and their aggregation |
| `stats_extra.py`, `finding4.py` | Statistics quoted in the text |
| `collect_metrics.py` | Collection of the metric files into one table |
| `simproc/` | CSTR and QTank |
| `simproc2/` | DIST, CSTH and DTS200 |
| `simproc3/` | EVAP and PH |
| `simproc4/` | FERM and HEX |

The usage line of every script is in its docstring. The result files of the paper are in `results/` at the top of the repository, and the proofs of the theoretical results are in `docs/proofs/VioExplain_proofs.pdf`.
