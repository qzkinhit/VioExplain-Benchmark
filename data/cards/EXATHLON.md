# Exathlon

[Exathlon](https://github.com/exathlonbenchmark/exathlon) accompanies a VLDB 2021 explainable anomaly-detection benchmark for Spark applications. The official data distinguish anomaly types, root-cause intervals, and extended effects. Obtain the desired version directly from the authors; this release does not bundle traces or automate bulk acquisition.

Native event type and interval annotations are suitable for event-conditioned explanation and type discrimination. They are not a general root-dimension truth table. Root-cause intervals and extended-effect intervals must not be merged without identifying the evaluation target. Missing metrics, dropped columns, and imputation require a frozen convention; backward filling uses future observations and cannot be described as a strictly online transformation.

The code and data may have different licenses. Retain and verify both upstream notices for the chosen version. A historical six-trace subset is not equivalent to the full official benchmark.
