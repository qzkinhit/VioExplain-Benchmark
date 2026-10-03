# Baseline identity and fidelity

A named baseline is considered an official reproduction only when its upstream implementation, revision, dependencies, preprocessing, inputs, parameters, and evaluation mapping are recorded. Similar scoring rules must be named as adaptations.

| Family | Released component or status | Permitted interpretation |
|---|---|---|
| AEC-Prototype and later MinExplain | Implemented in `publicexp/`, `mincost/`, and retained prototype modules | Core explanation comparison |
| Violation-only dimensions | Direct violation evidence | Mandatory control for the dimension-signature adapter |
| Robust median/IQR or robust-z scores | Statistical control in prior pilots; formal runner records exact definition | Do not label a short score implementation as complete official BARO |
| Persistence, context median, seasonal naive | Implemented in `tii/residuals.py` | Forecast/residual controls |
| Isolation Forest | Library estimator; runner must freeze parameters and scale | Detection control, not semantic diagnosis |
| Ridge linear/VAR references | Prior exploratory control; formal configuration required | Relation or forecast control |
| Chronos-2 frozen forecast/representation | Locked TEP event panel completed; weights not bundled | Matched-structure ranking gain, still below strong statistical classifiers |
| Official BARO RobustScorer | Complete SMD28 component evaluation, pinned upstream source | Not the full BOCPD plus attribution chain |
| Official TranAD network | Formal runner and pinned source; completion recorded in run artifacts | Five-epoch adapted protocol, not the published table |
| Official TreeSHAP + Isolation Forest | Formal runner with SHAP 0.46.0 | Explains path length, not physical causality |
| Official SARAD network | SMD28 with three seeds completed; exact network and loss checked | Shared model with documented chronological and machine-boundary adaptations |
| EXstream, GDN | Official reproduction not completed in this release | Historical simplified EXstream is not an official baseline |
| Other recent time-series foundation models | Planned unless accompanied by a completed run manifest | No borrowed scores from incompatible published protocols |

All attribution methods must receive the same permitted context and output budget. Fixed Top-k methods cannot conceal forced explanations on normal examples. Detector AUPRC and explanation support F1 answer different questions. Numerical benchmark results require per-entity output and an explicit label-access protocol.

## Original method versus later formalization

`AEC-Prototype` denotes only the supplied legacy `setcover/` implementation and its preserved `MyCover`/`Select` behavior. `MinExplain` denotes the later opening-plus-assignment problem in `mincost/`. They must occupy distinct method rows. The default `api.explain` and the historical `publicexp/aec.py` adapter use MinExplain, even though the latter retains `aec` in its compatibility filename. Matching filenames do not prove algorithmic equivalence.


The [formal benchmark guide](FORMAL_BENCHMARK.md) records exact official revisions and command lines.
