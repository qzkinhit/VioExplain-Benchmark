# Reproduction

Run from the repository root. Python 3.10 or later is required. Install the declared CPU dependencies in a fresh virtual environment.

```bash
python -m pip install -e '.[test]'
bash run.sh smoke
python -m pytest -q
python data/download.py smd
python data/download.py skab
bash run.sh formal --help
```

The first two checks need no downloaded data or models. The minimum-cost tests compare approximation solvers and safe pruning against exact small-instance enumeration. These are implementation checks and are not paper-performance measurements. An optional exact-solver installation is `python -m pip install -e '.[exact]'`.

## Data acquisition

`data/sources.lock.json` fixes each supported upstream repository and commit, or the official TEP archive SHA256. The downloader verifies the source, retains its notices, and records SHA256 hashes of all downloaded files. It never installs or executes upstream code. It refuses to overwrite an existing dataset. Data are placed under `data/raw/<dataset>/`, excluded from Git.

Downloading the same commit permits input verification; it does not imply that a historical private working copy was byte-identical. Each formal experiment must record the actual input hashes. Source commit identity and source-file hashes should both be recorded when there are uncommitted changes.

## Evaluation contract

Separate fitting, threshold calibration, model selection, and test evaluation. Thresholds and hyperparameters cannot be selected on final labels. Record anomaly-window access explicitly when evaluating explanations on true event intervals. Such a protocol measures conditional explanation, not end-to-end detection and explanation.

SMD native contribution labels, SKAB anomaly labels, simulated Tennessee Eastman fault identities, and injected channel supports are different targets. Report them in separate panels. Do not manufacture root-dimension labels for datasets that provide only anomaly labels. Fixed Top-k attribution and variable-size support estimation must also remain separate; report errors on normal intervals.

An experiment output should contain the locked protocol, source and input hashes, model revision when used, method identity, seed, per-entity measurements, normal false-explanation or false-alarm rates, aggregate definitions, failures, and timings. The independent statistical unit is a machine, testbed file, or nonoverlapping background, not every injected variant on the same background.

## Historical results

`result/summary/historical/` contains exact aggregate CSV exports from prior experiments. Its manifest identifies source CSV hashes and the data regime. Large predictions, fitted artifacts, and original private execution manifests are not redistributed. These aggregates are auditable records of the listed observations; this package alone does not reconstruct every historical run. New formal results belong to a separate protocol and output directory.

## Optional foundation models

The `tii/` numerical components can be imported without Torch or model weights. GPU runners obtain model weights separately and must record the official model identifier, revision, file hashes, environment, and forecast timing. No universal GPU requirements are installed by the CPU package. A frozen model, a fitted matching head, and a fine-tuned model are distinct experimental conditions.

## Supplied original prototype

```bash
bash run.sh prototype-smoke
python -m pytest -q tests/test_setcover.py
```

The command runs the six tiny demonstrations supplied with AEC-Prototype. The test compares their costs against the inherited golden values. It validates this supplied prototype only, not equivalence with an unavailable authoritative dissertation version. Demonstrations are synthetic checks and not benchmark performance.

