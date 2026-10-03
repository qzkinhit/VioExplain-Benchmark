# Locked event-matching results

This is the completed TEP event panel, separate from the nine formal baseline runs and earlier synthetic pilots. It contains the fixed protocol, LOCK, test-start marker, fitted-knowledge metadata, complete predictions, metrics, candidate diagnostics, normal errors and missing-knowledge outcomes.

`export_manifest.json` is an explicit allowlist with original and exported hashes. `environment.json` removes only host, working directory and site-specific GPU-index fields. All numerical outputs, LOCK and executed source snapshots retain their original bytes. The metadata correction changes only the descriptive count of normal calibration windows. No fitted pickle, model weights, raw dataset or embedding array is bundled.

```bash
python -m run_vioexplain.verify_results
```

The verification command checks exported hashes and matches every frozen source to the LOCK. `postprocess_predictions.py` can reconstruct descriptive class and candidate diagnostics from the saved predictions without training or model inference. The original full audit script requires unbundled fitted objects and embedding arrays, so it is not presented as a runnable public check.

The current rerun entry is `bash run.sh formal event-matching --help`. Use a new output directory for development and testing. Exact executed scripts under `source_snapshot/` are provenance records, not the preferred entry point. See the [method guide](../../../docs/EVENT_MATCHING.md) and [full report](../../../docs/EVENT_MATCHING_REPORT_zh.md).
