# Server Machine Dataset

The [official OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly) accompanies KDD 2019. It contains 28 machines with 38 metrics each. The authors provide separate train/test series, point anomaly labels, and `interpretation_label` files listing anomaly-contributing dimensions. The official description recommends treating machine subsets separately.

This benchmark uses the pinned commit in `../sources.lock.json`. The repository-level MIT notice is retained in each download; this software repository redistributes no raw data.

An interpretation row is parsed as an interval and a list of one-based dimension indices. The historical adapter treats interval ends as exclusive. Formal evaluation must record this convention and audit alignment against point labels because individual boundaries can disagree. All ground-truth intervals must remain outside fitting and calibration. Using their boundaries to locate explanation windows is conditional event explanation and must be identified as such.

The labels identify contributing dimensions, not adjudicated physical causes. Cloud/server telemetry is relevant to operation monitoring but does not alone substantiate a physical process diagnosis claim.
