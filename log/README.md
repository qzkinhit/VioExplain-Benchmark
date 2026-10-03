# Execution logs

`bash run.sh smoke`, `prototype-smoke`, and `formal` retain combined stdout/stderr, actual exit status, timing, arguments and source hashes under `log/runs/<task>/<run-id>/`. Each run gets a new directory. Failed executions remain failures; no scientific completion marker is fabricated.

The outer wrapper records process evidence. Dataset-specific runners additionally save the protocol, fitted artifacts, intermediate provenance, predictions and summaries in the chosen `result/runs/` directory. See [the recording contract](../docs/EXPERIMENT_RECORDS.md). Do not publish logs containing credentials or private machine paths without review. Historical outputs expose only artifacts that actually exist.
