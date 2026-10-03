# Changelog

## 0.1.0, 2026-10-03

- Exported the original violation, representation, covering, and knowledge-update components through an explicit allowlist.
- Included the corrected partial AEC behavior that preserves known explanations when unmatched violations occur.
- Added a public explanation result with separate unknown-evidence indices and assignments, plus nonnegative-cost input validation.
- Added packaging, a synthetic smoke command, baseline/data scope documentation, and CPU tests.
- Added upstream-commit-pinned SMD/SKAB acquisition with retained notices and per-file SHA256 manifests.
- Preserved historical exploratory aggregates, including negative results, without presenting them as a new formal benchmark.
- Kept foundation-model solvers explicitly exploratory and documented historical correlation-extractor limitations.

- Explicitly separated the supplied AEC-Prototype from the later MinExplain formulation, and provided a runnable prototype smoke with inherited golden checks.

- Completed and released all nine fixed formal baseline batches, with per-entity data, clustered intervals, adapter audits and exact executed source snapshots.
- Added the locked TEP event panel, preserving true prototype intervals and separately identifying AEC-CostAdapted and MinExplain with interval, raw temporal and frozen Chronos-2 costs.
- Published event predictions, negative outcomes, immutable LOCK and explicit metadata-redaction provenance; fitted objects and model/embedding files remain unbundled.

## 2026-10-04

Separate `result/` and `log/` directories. Command wrappers now preserve failed exits, console output and source hashes in unique run directories. Released numerical artifacts are unchanged; 43 remote CPU tests and artifact-integrity checks passed.

The streaming wrapper flushes each available child output chunk before process exit. MinExplain documentation now states the support-rescan runtime bound and exposes implementation assumptions without references to private paper files. Numerical solvers are unchanged.
