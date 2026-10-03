# Formal v1 released measurements

`INDEX.json` lists completed runs and SHA256 hashes of the released files. Each run retains its completion marker, sanitized manifest, per-entity measurements and aggregates. No raw data, model weights or host-specific paths are included.

Detection is separate from conditional attribution and supervised fault classification. The TEP event-class panel uses official fault IDs under a stated window/onset protocol. Unknown/unrun methods receive no fabricated score. Historical pilots are stored separately in `../historical/`.

Scores and checkpoint arrays are retained in the research execution archive and are not bundled in this lightweight release. The current repository contains the rerun code and pinned data/source acquisition. The manifest's source paths and hashes identify actual executed scripts, whose original directory layout differs from the public layout. These exported records are not regenerated to make their source hashes match later path or startup-check changes. Exact project-owned executed scripts are preserved under `executed_sources/` with their own hash manifest; they are provenance snapshots, not the preferred rerun entry points.

```bash
python -m run_vioexplain.verify_results
```

This command checks the tracked file hashes. It does not rerun the model or independently reconstruct metrics from unbundled raw predictions.
