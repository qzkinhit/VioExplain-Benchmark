# Data

Raw data are not included in this software release. Obtain data from their upstream owners and retain their notices.

```bash
python data/download.py smd --show-source
python data/download.py smd
python data/download.py skab
python data/download.py tep
```

The default layout is `data/raw/smd/{train,test,test_label,interpretation_label}` and `data/raw/skab/` with the upstream subdirectories. Git downloads contain `UPSTREAM_NOTICES/`; the TEP archive retains its own readme. Every download contains `download_manifest.json`. The latter records the fixed Git revision and SHA256 of the actual files.

| Card | Native evaluation target | Download status |
|---|---|---|
| [SMD](cards/SMD.md) | Point anomalies and event-contributing dimensions | Pinned official download implemented |
| [SKAB](cards/SKAB.md) | Point anomalies and change points | Pinned official download implemented |
| [Exathlon](cards/EXATHLON.md) | Anomaly type and root-cause interval | Manual upstream acquisition; no data bundled |
| [Tennessee Eastman](cards/TEP.md) | Simulated process fault identity | Official archive SHA256 pinned; protocol-derived onset labels |

Synthetic fixtures in the smoke command are hand-constructed method checks. They do not reproduce physical industrial faults and do not appear in benchmark leaderboards.
