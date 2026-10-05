# cstr: comparison (A), baselines fitted on single-fault runs

| No. | Family | Method | Macro F1 | Normal (%) | Pairs | Triples | Four | Attr. F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Process monitoring | PCA-RBC | 0.889 | 3.4 | 0.650 | 0.599 | 0.608 | 0.602 |
| 2 | Process monitoring | FDA | 0.949 | 2.9 | 0.669 | 0.507 | 0.406 | 0.567 |
| 3 | Fault classifiers | RF | 0.972 | 2.6 | 0.657 | 0.485 | 0.406 | 0.577 |
| 4 | Fault classifiers | XGB | 0.983 | 3.0 | 0.653 | 0.485 | 0.400 | 0.552 |
| 5 | Fault classifiers | LGBM | 0.983 | 4.1 | 0.660 | 0.490 | 0.404 | 0.572 |
| 6 | Time-series classifiers | MiniRocket | 0.976 | 3.9 | 0.671 | 0.508 | 0.404 | 0.562 |
| 7 | Time-series classifiers | MultiRocket | 0.971 | 6.1 | 0.670 | 0.505 | 0.385 | 0.533 |
| 8 | Time-series classifiers | QUANT | 0.970 | 3.1 | 0.639 | 0.453 | 0.374 | 0.555 |
| 9 | Deep diagnosers | 1D-CNN | 0.974 | 4.2 | 0.641 | 0.445 | 0.306 | 0.464 |
| 10 | Deep diagnosers | LSTM | 0.979 | 4.4 | 0.620 | 0.433 | 0.338 | 0.469 |
| 11 | Deep diagnosers | ResNet | 0.983 | 1.3 | 0.650 | 0.471 | 0.348 | 0.497 |
| 12 | Deep diagnosers | InceptionTime | 0.984 | 6.6 | 0.650 | 0.473 | 0.359 | 0.489 |
| 13 | Time-series foundation model | MantisV2 | 0.976 | 5.1 | 0.666 | 0.501 | 0.407 | 0.538 |
| 14 | Multi-label diagnosis | BR | 0.973 | 3.0 | 0.933 | 0.893 | 0.889 | 0.825 |
| 15 | Multi-label diagnosis | CC | 0.975 | 4.6 | 0.928 | 0.883 | 0.881 | 0.817 |
| 16 | Multi-label diagnosis | ML-CNN | 0.887 | 4.2 | 0.666 | 0.402 | 0.210 | 0.448 |
| 17 | Knowledge-based explanation | AEC | 0.875 | 1.3 | 0.882 | 0.840 | 0.818 | 0.787 |
| 18 | Knowledge-based explanation | MinExplain | 0.930 | 3.5 | 0.872 | 0.806 | 0.794 | 0.787 |
| 19 | Ours | VioExplain | 0.983 | 1.9 | 0.958 | 0.948 | 0.936 | 0.844 |

# cstr: comparison (B), multi-label methods with the compositions of VioExplain

| Method | Pairs | Triples | Four | Macro F1 | Normal (%) |
|---|---|---|---|---|---|
| BR | 0.959 | 0.931 | 0.921 | 0.978 | 1.4 |
| CC | 0.958 | 0.927 | 0.916 | 0.975 | 0.6 |
| ML-CNN | 0.946 | 0.886 | 0.864 | 0.965 | 1.5 |
| VioExplain | 0.958 | 0.948 | 0.936 | 0.983 | 1.9 |

| Extra | macroF1 | normal_pct | pairs | triples | four | attrF1 |
|---|---|---|---|---|---|---|
| VioExplain without composition (f0) | 0.984 | 3.0 | 0.928 | 0.889 | 0.912 | 0.827 |

Gains: {"A_gain_pct_over_strongest_multilabel_pairs": 2.6, "A_gain_pct_over_strongest_baseline_pairs": 2.6, "B_gain_pct_over_strongest_multilabel_pairs": -0.2, "A_gain_pct_over_strongest_multilabel_triples": 6.2, "A_gain_pct_over_strongest_baseline_triples": 6.2, "B_gain_pct_over_strongest_multilabel_triples": 1.8, "A_gain_pct_over_strongest_multilabel_four": 5.3, "A_gain_pct_over_strongest_baseline_four": 5.3, "B_gain_pct_over_strongest_multilabel_four": 1.6, "A_attrF1_points_over_best_baseline": 1.9, "A_four_gain_pct_over_stronger_of_AEC_MinExplain": 14.3, "A_normal_named_ratio_multilabel_over_ours": 2.4}

Finished rows: 22 of 22
