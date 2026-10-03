# VioExplain Benchmark

[简体中文](README_zh.md)

VioExplain explains constraint violations through anomaly representations and minimum-cost covering. It preserves the original sequence of **violation extraction, representation matching, anomaly explanation, and knowledge update**. This repository provides the method implementation, explicit handling of unmatched evidence, public-data protocols, baseline adapters, and recorded exploratory results.

The foundation-model components are research extensions. Existing pilot and formal event results do not establish a consistent advantage over statistical controls or a state-of-the-art result. Synthetic checks, native anomaly detection, dimension attribution, and event diagnosis are reported separately.

## Method identities

| Method | Objective and solver | Entry point |
|---|---|---|
| `AEC-Prototype` | Supplied interval-based event-covering prototype with its original `Select` procedure | `vioexplain.setcover.set_covering.MyCover` |
| `MinExplain` | Later opening-cost plus violation-assignment objective; density greedy or primal-dual | `vioexplain.api.explain`, `vioexplain.mincost` |

These are different objectives and algorithms. The default public API is **MinExplain**. `AEC-Prototype` remains separately callable and is verified against the supplied legacy demonstrations. The reference lineage is the supplied original English manuscript and code. Repairs to adapters and interfaces are not research contributions.

## Quick start

Use Python 3.10 or later in a fresh environment. No GPU, model weights, or dataset is required for the smoke check.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash run.sh smoke
bash run.sh prototype-smoke
python -m pytest -q
```

`smoke` runs a tiny synthetic explanation with two matchable violations and one unmatched violation. It checks the API contract and does not produce a paper result.

## Repository map

| Path | Purpose |
|---|---|
| [`src/vioexplain/`](src/vioexplain/README.md) | Public API, violation extraction, representations, covering, knowledge update |
| `benchmark/` | Official-data adapters, evaluation, and frozen protocols |
| [`data/`](data/README.md) | Data cards and downloads pinned to upstream commits; raw data remain local |
| `configs/` | Named smoke and experiment configurations |
| `run_vioexplain/` | Experiment entry points |
| [`result/`](result/README.md) | Tracked aggregate evidence and local run outputs |
| [`docs/`](docs/REPRODUCING.md) | Reproduction, baseline fidelity, dataset scope, provenance, and limitations |
| `tests/` | API, partial coverage, optimization, and experimental-component checks |

Start with [`vioexplain.api.explain`](src/vioexplain/api.py). The public result separates evidence supported by selected representations from unmatched evidence; an unknown violation is never silently counted as explained.

```python
from vioexplain import Violation, Representation, explain

violations = [Violation("temperature", "domain", 1, 1.0, (1.0, 1.0), 3)]
knowledge = [Representation({"temperature"}, w=1.0, support=2)]
result = explain(violations, knowledge, theta=0.6)
print(result.to_dict())
```

An optional `distances={(violation_index, representation_index): nonnegative_cost}` supplies a matching model without replacing the covering problem. Missing edges are incompatible. [API semantics](docs/API.md) define identities, cost, assignments, and unknown evidence.

## Data and evaluation

```bash
python data/download.py smd --show-source
python data/download.py smd
python data/download.py skab
bash run.sh formal --help
```

SMD supplies native anomaly-contributing dimensions. SKAB supplies anomaly and change-point labels, so it cannot by itself validate root-dimension explanation. The formal statistical runners also support the classic Tennessee Eastman simulation. Exathlon remains a separately documented candidate extension. [Dataset cards](docs/DATASETS.md) identify what each dataset can support; [baseline notes](docs/BASELINES.md) distinguish official implementations from adaptations.

## Formal benchmark commands

```bash
# CPU statistical detection. Runs all official entities in the supplied data root.
bash run.sh formal statistical --dataset smd --data data/raw/smd --output result/runs/smd-statistical
bash run.sh formal statistical --dataset skab --data data/raw/skab --output result/runs/skab-statistical
```

The SMD conditional attribution track can additionally invoke the official BARO RobustScorer component. TranAD and TreeSHAP have their own entry points and dependency requirements. See [formal reproduction](docs/FORMAL_BENCHMARK.md) for pinned source acquisition, commands, adaptation boundaries, and status. Statistical scores, conditional attribution, and event-class diagnosis remain separate tasks.

## Evidence status

The software release and research validation have different scopes. The inherited AEC core and exploratory solvers are implemented. [Historical aggregates](result/summary/historical/manifest.json) preserve both gains and negative results, including the absence of cross-domain gains for a frozen foundation-model matching head. They are not a new blind benchmark. [Completed formal batches](docs/BENCHMARK_STATUS.md) have their own protocol, per-entity records and manifest. These baseline runs do not establish an advantage for a new VioExplain method.

[Reproduction](docs/REPRODUCING.md) describes the commands and result contracts. [Limitations](docs/LIMITATIONS.md) records known boundaries. The release contains no private industrial measurements, dissertation text, model weights, or copied third-party baseline source.

## License and citation

Project code is released under [MIT](LICENSE). Datasets, external implementations, and model weights retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). [CITATION.cff](CITATION.cff) identifies the software. This release does not claim a published journal article or an assigned paper DOI.

Historical method names are resolved in the [method registry](docs/METHOD_IDENTITIES.md).

The [locked TEP event panel](docs/EVENT_MATCHING.md) adds the original interval prototype, an explicit cost adaptation, and MinExplain with interval, raw temporal or frozen Chronos-2 matching. All nine formal baseline batches and this event panel are complete; gains and negative results are reported together.

Commands retain console output and execution evidence in `log/runs/`. Published snapshots are in `result/summary/`; use `result/runs/` for new runs. See [the recording contract](docs/EXPERIMENT_RECORDS.md).
