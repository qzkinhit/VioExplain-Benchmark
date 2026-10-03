# Formal benchmark v1

The [locked configuration](../configs/formal/locked_v1.json) and [Chinese protocol](../benchmark/protocol/formal_v1_zh.md) define the 2026-10-03 formal batch. This batch is separate from the historical pilot results.

## Tasks and scope

| Track | Data and labels | Evaluation |
|---|---|---|
| Native detection | All 28 SMD machines; official train/test and point labels | Point F1, AUPRC/AUROC, normal false alarms, event hits and delay |
| Conditional dimension attribution | SMD native contributing-dimension annotations; event intervals are supplied | Fixed K=1,3,5 precision/recall/F1, NDCG and hit; group 1 development, groups 2/3 evaluation |
| Native industrial detection | All 34 anomaly-containing SKAB files; separate anomaly-free reference | File-level macro detection metrics; no invented dimension labels |
| Simulated-fault detection | Classic TEP d00 through d21 test trajectories | All 21 fault cases plus normal control; protocol-derived fault onset at zero-based index 160; not a separate official label file |
| Event-class diagnosis | Separate TEP fault training and testing trajectories | Separate protocol and results required; not implied by the detection runner |

Each normal reference is divided chronologically into 60% fit, 20% scale/development, and 20% calibration. Statistical controls do not add the unused middle segment to fitting. The threshold is the finite-sample upper 0.99 order statistic of normal calibration scores. Temporal dependence prevents an unconditional 1% false-alarm guarantee.

No point adjustment or test-label-selected threshold is used. Single-class evaluation metrics are treated according to their defined domain. Missed events do not receive zero delay. SMD attribution receives all 38 dimensions and fixed K, not the true support cardinality. Event boundaries make this conditional attribution rather than end-to-end explanation.

## Commands

Install the project and download the data first. Supply fresh output directories.

```bash
python -m pip install -e '.[test]'
python data/download.py smd
python data/download.py skab
python data/download.py tep
bash run.sh formal statistical --dataset smd --data data/raw/smd --output result/runs/smd-statistical
bash run.sh formal statistical --dataset skab --data data/raw/skab --output result/runs/skab-statistical
bash run.sh formal statistical --dataset tep --data data/raw/tep --output result/runs/tep-statistical
```

The TEP data root must contain `d00.dat` and `d00_te.dat` through `d21_te.dat`, directly or in `TE_process/`. Obtain the classic distribution using the data-card instructions. The loader transposes the 52-by-500 normal training file and checks for 52 columns. They comprise XMEAS(1–41) and XMV(1–11), not an invented 22/30 split. This is a simulated chemical process, not measurements from an operating factory.

## Official BARO component

Obtain the official repository separately. It is ignored and not redistributed.

```bash
python -m pip install -e '.[baro]'
git clone https://github.com/phamquiluan/baro.git vendor/baro
git -C vendor/baro checkout --detach e35f4ec1095e5cac891d52de9ad18a5b32a37ec8
bash run.sh formal statistical --dataset smd --data data/raw/smd --output result/runs/smd-with-baro --baro-source vendor/baro
```

The runner calls the unchanged DataFrame `RobustScorer` path. It does not run BARO's full change-point detection chain. Reference observations are the observed equally sized segment before the event, or the tail of the normal fit segment when the event starts at zero. Ground truth is not used to filter that reference. The output row must be named `baro_robust_scorer`, not a full BARO reproduction.

## Official TranAD network

Use an appropriate separate PyTorch 2 CUDA environment and obtain the official source.

```bash
git clone https://github.com/imperial-qore/TranAD.git vendor/TranAD
git -C vendor/TranAD checkout --detach 7ffb98d0c18189cc3d9ab732b4cb0278200a0af0
bash run.sh formal tranad --dataset smd --data data/raw/smd --vendor vendor/TranAD --output result/runs/smd-tranad --seeds 0 1 2 --epochs 5
```

The adapter extracts the official network classes with AST to avoid unrelated dependencies and global argument parsing. A compatible one-layer container omits new PyTorch causal-mask keywords, preserving the old layer computation. The protocol fixes five epochs, float64, batch size 128, AdamW learning rate 0.0001, weight decay 0.00001, and StepLR(5,0.9).

Fit-segment min/max normalization is applied to every partition without test fitting or clipping. Scores align with the observation reconstructed. Bounded inference batches control memory. Normal calibration replaces the upstream test-aware threshold and point-adjusted evaluation. Consequently these scores are not numerically interchangeable with the original paper table. Changing epochs or seeds creates a different experimental condition; the runner's arguments must match its recorded protocol.

## Official TreeSHAP attribution

```bash
python -m pip install -e '.[shap]'
bash run.sh formal treeshap --data data/raw/smd --output result/runs/smd-treeshap
```

This explains a fitted 200-tree Isolation Forest using the official `TreeExplainer` with path-dependent contributions. It averages absolute contributions on at most 30 evenly spaced points per supplied event. The explained target is expected path length, not physical causality, and this is not a reproduction of a different classifier's SHAP system.

## Result and implementation status

The completed CPU batch covers SMD28, SKAB34 and TEP22, with RobustZ/PCA and three Isolation Forest seeds. SMD also includes the official BARO component with no failed event calls in the corrected batch. TranAD has completed all three datasets, TreeSHAP has completed SMD28, and SARAD has completed SMD28. Their completion markers, per-entity results and hash checks are released. GDN and Exathlon expansion were not run.

Every run retains per-file detection, per-event attribution where applicable, thresholds, failures, source hashes and data hashes. GPU runs additionally retain model/checkpoint information and epoch losses. Use independent entities as resampling units; repeated injected variants and repeated seeds are not new industrial installations. Formal result summaries are added only after their run manifests are checked.

## TEP conditional fault classification

```bash
bash run.sh formal tep-classifiers --data data/raw/tep --output result/runs/tep-classifiers
```

The released classifier panel uses 64-point nonoverlapping windows, all 21 fault IDs and the normal class, with separate official training and testing trajectories. Features are per-channel mean, standard deviation and least-squares slope. RBF-SVM chooses C and gamma on development windows only; shrinkage LDA and 500-tree Random Forest provide additional supervised controls. RF uses seeds 0,1,2. The exact candidate grid and split boundaries are saved in each manifest.

The protocol excludes the first 20 fault-training samples as a preset warm-up choice. This is not a claim that fault begins at training sample 20. The official 480-point fault-training files should not be confused with another distribution's 500-point files. Test fault windows begin at the prescribed sample 160. There are 267 complete test windows from 22 trajectories; windows from one trajectory are not independent trials. Trajectory voting uses the smallest class ID to break ties.

Completed, sanitized run artifacts are listed in [`formal_v1/INDEX.json`](../result/summary/formal_v1/INDEX.json). They include per-entity or per-window measurements and source/input hashes. Actual executed source paths refer to the original research layout (`experiments/formal/`); public entry points reside in `benchmark/evaluation/`. Current wrappers additionally verify pinned official vendor-file hashes before loading them. Those acquisition and path changes are not counted as model improvements.

## Official SARAD candidate

```bash
python -m pip install -e '.[gpu]'
git clone https://github.com/daidahao/SARAD.git vendor/SARAD
git -C vendor/SARAD checkout --detach 24854d9723b4eed31b547344061671c08fbfb3e2
bash run.sh formal sarad --data data/raw/smd --vendor vendor/SARAD --output result/runs/smd-sarad --seeds 0 1 2 --epochs 3
```

This separate protocol uses the official 512-dimensional, three-layer, eight-head network with three epochs and a 10% fit-window sample. The model is shared across machines; windows never join the end of one machine to the start of another. Fitting, score-scale estimation and threshold calibration use separate chronological segments. This run is complete on all 28 SMD machines and three seeds; final artifacts are linked in the [status table](BENCHMARK_STATUS.md). Do not label this constrained training budget as the paper's original reported setting.

Exact project-owned executed source snapshots are included in the formal result directory. Original vendor code remains separately acquired under its own license.
