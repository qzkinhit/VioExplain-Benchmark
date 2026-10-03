# Method names and historical labels

The historical label `AEC` has been used for three different implementations. Preserve original CSV labels for traceability and resolve identity using the source path. Do not merge rows solely by the displayed method name.

| Canonical identity | Source lineage | Meaning |
|---|---|---|
| `AEC-Prototype` | `setcover/set_covering.py:MyCover` and `Select` | Supplied original interval/event-covering prototype |
| `MinExplain-DimensionAdapter` | `publicexp/aec.py:explain_event` | Later opening-plus-assignment objective with dimension-signature matching; some historical tables display `AEC` |
| `AEC-SimulationHeuristic` | `compare/methods.py:aec` in the original workspace | Controlled-simulation heuristic; not exported in this release |

The stable `vioexplain.api.explain` interface calls the MinExplain formulation. It does not call `MyCover`. Cost values under different formulations are not directly comparable. Golden demonstration tests check the supplied prototype implementation; they do not establish equivalence of every historical table or of the final dissertation version.

The current research comparison is grounded in the supplied original English manuscript and code. The supplied prototype is checked against its own demonstration cases and remains separate from later formulations.

[`METHOD_REGISTRY.json`](METHOD_REGISTRY.json) records canonical identities and exact official component scopes. Use canonical identities in new experiments, and retain an explicit original-label field when importing historical data. Adapter bug fixes, input validation, and repository cleanup are reproducibility work and are not method innovations.

## Locked TEP event panel

AEC-Prototype consumes real magnitude intervals using the supplied MyCover/Select procedure. AEC-CostAdapted is an explicitly named edge-cost interface under the single-series/no-association-rule setting. It does not fabricate intervals to reproduce learned costs. MinExplain optimizes its separate opening-plus-assignment objective. Interval-only, raw-temporal and frozen-Chronos-2 versions of each adapted solver share a candidate graph; none is merged into the original AEC label. Their exact result labels are retained in `event_matching_v1/`.
