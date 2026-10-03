# Results and logs

`summary/` contains tracked, deliberately small evidence files. `runs/` is ignored and is the default place for new experiment outputs. Do not commit raw datasets, weights, copied third-party code, host-specific paths, or unredacted environment logs.

The [`summary/historical/`](summary/historical/manifest.json) records are exact aggregate exports from prior exploratory runs. Its manifest identifies the data regime and the source CSV hash. Native detection and injected-support experiments are separate. Large predictions and original execution environments remain outside this release, so these aggregates alone are not a complete reproduction package.

A completed new benchmark must provide its own frozen protocol, per-entity rows, failed/skipped cells, calibration rules, source/input/model hashes, and environment. Missing or failed results must not be replaced with zero scores. Summary performance does not authorize a state-of-the-art claim on a different target or split.
