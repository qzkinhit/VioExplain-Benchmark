# Third-party notices

The root [MIT license](LICENSE) applies to the code of this repository. Third-party datasets, libraries and model weights keep their own terms. No external dataset, model weight or third-party source code is bundled.

| Material | Use in this repository | Handling |
|---|---|---|
| Extended Tennessee Eastman data of Rieth et al. (2017) | TEP-R, single-fault runs and official test set | Downloaded separately from its owners |
| Tennessee Eastman simulator (Downs and Vogel 1993) with the plant-wide control scheme of Lyman and Georgakis (1995) | Generation of the TEP-C runs | Not bundled |
| Hydraulic test rig data of Helwig et al. (2015), UCI repository | HYD | Downloaded separately from its owners |
| Published process models (stirred-tank reactor, quadruple tank, distillation column A, stirred tank heater, three-tank system, forced-circulation evaporator, pH neutralization, continuous fermenter, heat exchanger) | Simulators in `experiments/tii_final/simproc*/`, written for this repository from the published equations | Sources and deviations are listed in the docstring of each simulator |
| aeon | MiniRocket, MultiRocket, QUANT | Installed separately |
| LightGBM, XGBoost, scikit-learn, PyTorch, NumPy, SciPy | Baselines and method components | Installed separately |
| MantisV2 | Frozen time-series foundation model | Weights and loader obtained separately from its authors |

This notice does not assign a new license to any third-party material.
