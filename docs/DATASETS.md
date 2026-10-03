# Dataset scope and evidence

| Dataset | Authority and relevance | What can be evaluated | What must not be inferred |
|---|---|---|---|
| [SMD](../data/cards/SMD.md) | Official OmniAnomaly data accompanying KDD 2019 | Native point detection and anomaly-contributing dimensions | Contribution annotations do not establish physical causality |
| [SKAB](../data/cards/SKAB.md) | Official controlled industrial testbed benchmark | Native anomaly detection and change points | Binary anomaly labels are not root-dimension labels |
| [Exathlon](../data/cards/EXATHLON.md) | Official explainable anomaly benchmark accompanying VLDB 2021 | Event types and distinct root-cause/effect intervals | Event type is not a complete feature-level causal ground truth |
| [Tennessee Eastman](../data/cards/TEP.md) | Established simulated chemical process | Fault identity and detection under a specified distribution and explicit onset convention | Simulated fault identity does not identify every affected variable as a root cause |
| Controlled injections | Explicitly generated supports on separate backgrounds | Mechanism, identifiability boundaries, support recovery | Injected variants are not independent real industrial events |

Availability of a dataset does not mean the benchmark has been run. Read the formal protocol and run manifest for the evaluated entities and completed methods. Public accessibility also does not replace license verification.
