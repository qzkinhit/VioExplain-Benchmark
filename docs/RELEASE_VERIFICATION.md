# Software release verification

Final validation ran on a remote Linux CPU server on 2026-10-03. No research experiment or pytest suite was run on the local workstation.

| Check | Observed result |
|---|---|
| Complete public test suite | 41 passed in 1.68 seconds, including five event-matching invariants |
| MinExplain synthetic smoke | Partial explanation preserves the unmatched violation independently |
| AEC-Prototype synthetic smoke | Six supplied demonstrations execute; golden-cost regression passes |
| Source distribution and wheel | Both built with the correct package identity |
| Separately installed wheel | All 33 package modules import outside the source checkout; both smoke commands pass |
| Formal CLI | Wrapper, statistical, TEP classifier and event-matching help succeed |
| SMD source identity | The formal experiment's 112 raw files match the pinned official Git blobs |
| SKAB download and loader | Pinned commit verified; all 34 anomalous files loaded |
| TEP download and loader | Official archive SHA256 verified; all 22 test trajectories loaded |
| Baseline result hashes | 61 indexed files from nine completed batches and seven executed scripts match |
| Event result hashes | All 45 allowlisted artifacts and all 15 source hashes in LOCK match; locked protocol and test-start marker also match |
| Release review | Public text contains no personal filesystem paths, private server identities or credentials; local Markdown links resolve |

The tested core environment used Python 3.10.12, NumPy 1.24.3, pandas 1.5.3, SciPy 1.15.3, scikit-learn 1.7.2 and pytest 8.4.2. `requirements-cpu-tested.txt` records it. The recorded GPU event experiment has a separate [environment manifest](../result/summary/event_matching_v1/environment.json).

These checks establish software and artifact consistency. Method performance, failure modes and dataset scope are reported separately in the [benchmark status](BENCHMARK_STATUS.md) and [event-matching guide](EVENT_MATCHING.md). Reproducible software does not establish overall method superiority.
