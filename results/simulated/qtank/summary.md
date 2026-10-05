# qtank: comparison (A), baselines fitted on single-fault runs

| No. | Family | Method | Macro F1 | Normal (%) | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Process monitoring | PCA-RBC | 0.937 | 4.8 | 0.568 | 0.533 | 0.595 | 0.644 |
| 2 | Process monitoring | FDA | 0.946 | 4.4 | 0.626 | 0.535 | 0.473 | 0.652 |
| 3 | Fault classifiers | RF | 0.982 | 2.7 | 0.625 | 0.534 | 0.438 | 0.620 |
| 4 | Fault classifiers | XGB | 0.993 | 3.6 | 0.627 | 0.538 | 0.435 | 0.616 |
| 5 | Fault classifiers | LGBM | 0.994 | 2.9 | 0.622 | 0.534 | 0.439 | 0.616 |
| 6 | Time-series classifiers | MiniRocket | 0.986 | 2.6 | 0.677 | 0.550 | 0.480 | 0.718 |
| 7 | Time-series classifiers | MultiRocket | 0.965 | 8.3 | 0.666 | 0.530 | 0.453 | 0.680 |
| 8 | Time-series classifiers | QUANT | 0.979 | 5.3 | 0.675 | 0.547 | 0.416 | 0.719 |
| 9 | Deep diagnosers | 1D-CNN | 0.995 | 3.5 | 0.647 | 0.516 | 0.445 | 0.718 |
| 10 | Deep diagnosers | LSTM | 0.995 | 3.1 | 0.633 | 0.514 | 0.413 | 0.672 |
| 11 | Deep diagnosers | ResNet | 0.993 | 5.7 | 0.644 | 0.506 | 0.423 | 0.714 |
| 12 | Deep diagnosers | InceptionTime | 0.995 | 4.0 | 0.652 | 0.520 | 0.446 | 0.722 |
| 13 | Time-series foundation model | MantisV2 | 0.979 | 5.1 | 0.674 | 0.548 | 0.412 | 0.716 |
| 14 | Multi-label diagnosis | BR | 0.970 | 8.7 | 0.755 | 0.735 | 0.726 | 0.743 |
| 15 | Multi-label diagnosis | CC | 0.987 | 7.5 | 0.746 | 0.720 | 0.708 | 0.732 |
| 16 | Multi-label diagnosis | ML-CNN | 0.983 | 5.8 | 0.752 | 0.655 | 0.596 | 0.731 |
| 17 | Knowledge-based explanation | AEC | 0.800 | 2.9 | 0.690 | 0.647 | 0.637 | 0.653 |
| 18 | Knowledge-based explanation | MinExplain | 0.801 | 3.4 | 0.658 | 0.599 | 0.617 | 0.624 |
| 19 | Ours | VioExplain | 0.994 | 4.5 | 0.798 | 0.756 | 0.805 | 0.830 |

# qtank: comparison (B), multi-label methods with the compositions of VioExplain

| Method | Pairs | Triples | Four | Macro F1 | Normal (%) |
|---|---|---|---|---|---|
| BR | 0.824 | 0.763 | 0.779 | 0.981 | 1.5 |
| CC | 0.827 | 0.764 | 0.774 | 0.980 | 1.1 |
| ML-CNN | 0.796 | 0.729 | 0.771 | 0.994 | 1.3 |
| VioExplain | 0.798 | 0.756 | 0.805 | 0.994 | 4.5 |

| Extra | macroF1 | normal_pct | pairs | triples | four | attrF1 |
|---|---|---|---|---|---|---|
| VioExplain without composition (f0) | 0.990 | 8.9 | 0.742 | 0.671 | 0.717 | 0.817 |

Gains: {"A_gain_pct_over_strongest_multilabel_pairs": 5.7, "A_gain_pct_over_strongest_baseline_pairs": 5.7, "B_gain_pct_over_strongest_multilabel_pairs": -3.5, "A_gain_pct_over_strongest_multilabel_triples": 2.9, "A_gain_pct_over_strongest_baseline_triples": 2.9, "B_gain_pct_over_strongest_multilabel_triples": -1.1, "A_gain_pct_over_strongest_multilabel_four": 10.9, "A_gain_pct_over_strongest_baseline_four": 10.9, "B_gain_pct_over_strongest_multilabel_four": 3.2, "A_attrF1_points_over_best_baseline": 8.7, "A_four_gain_pct_over_stronger_of_AEC_MinExplain": 26.2, "A_normal_named_ratio_multilabel_over_ours": 1.9}

Finished rows: 22 of 22
