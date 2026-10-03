# Skoltech Anomaly Benchmark

The [official SKAB repository](https://github.com/waico/SKAB) documents measurements from a water-circulation testbed. Version 0.9 describes 34 anomaly-containing series and one anomaly-free reference file, with eight measured channels. The native targets are binary `anomaly` and `changepoint` labels.

Use `python data/download.py skab`. The exact upstream revision is fixed in `../sources.lock.json`. The upstream repository contains a GPL-3.0 notice, copied into the ignored download directory. Dataset-specific use and redistribution terms should be checked with the owner; no raw SKAB files or upstream Python source are redistributed here.

Binary anomaly labels do not identify root dimensions. Sensor support labels created by injection are synthetic labels and belong to a separate experiment. A candidate model that uses a trusted normal prefix must state its availability assumption and report prefix contamination after predictions are frozen. Repeated tuning on already scored files invalidates a fresh holdout claim.
