# VioExplain

[English](README.md)

本仓库包含 VioExplain 一文的代码、结果文件与证明。VioExplain 对工业过程的一个窗口输出三项内容，即一组异常事件、约束违反到事件的归属、未被解释的剩余违反。方法只用无故障运行与单故障运行学习，用配对运行合成并发故障的反事实窗口，并把含未知事件的窗口标记出来用于知识更新。

## 目录

| 路径 | 内容 |
|---|---|
| [`experiments/tii_final/`](experiments/tii_final/README.md) | Tennessee Eastman 过程（TEP）实验的脚本 |
| [`experiments/tii_final/tep_generation/`](experiments/tii_final/tep_generation/README.md) | TEP 仿真器的构建脚本与 TEP-C 配对并发故障运行的生成脚本 |
| [`experiments/tii_final/hyd/`](experiments/tii_final/hyd/README.md) | 液压试验台（HYD）实验的脚本 |
| `experiments/tii_final/simproc/`、`simproc2/`、`simproc3/`、`simproc4/` | 九个仿真过程的仿真器、数据生成脚本与运行脚本 |
| [`results/`](results/README.md) | 文中每张表、每幅图对应的结果文件及索引 |
| [`docs/proofs/VioExplain_proofs.pdf`](docs/proofs/VioExplain_proofs.pdf) | 理论结果的证明 |

## 数据集

| 数据集 | 来源 | 文中任务 |
|---|---|---|
| TEP-R | Rieth 等（2017）的扩展 Tennessee Eastman 数据，20 种扰动 IDV(1) 至 IDV(20)，使用官方测试集 | 单故障诊断 |
| TEP-C | 用 TEP 仿真器在全厂控制方案下生成，扰动单独注入以及两两、三三组合注入，四故障窗口由配对的单故障运行叠加得到 | 并发故障解释、违反归因 |
| TEP 留出错误类型 | TEP 错误类型轮流留出，窗口取自 TEP-R 与 TEP-C | 未知事件与知识更新 |
| HYD | Helwig 等（2015）的液压试验台数据，取自 UCI 数据库，含四个退化部件 | 并发故障解释 |
| CSTR、QTank | 搅拌釜反应器与四容水箱，见 `simproc/` | 并发故障解释 |
| DIST、CSTH、DTS200 | 精馏塔、搅拌釜加热器与三容水箱，见 `simproc2/` | 并发故障解释 |
| EVAP、PH | 强制循环蒸发器与 pH 中和过程，见 `simproc3/` | 违反归因 |
| FERM、HEX | 连续发酵罐与管壳式换热器，见 `simproc4/` | 单故障诊断 |

仓库不附带数据。TEP-R 与 HYD 需从数据所有者处下载。TEP-C 的运行数据由 `experiments/tii_final/tep_generation/` 调用仿真器的 Fortran 代码生成，该代码需另行获取。仿真过程由各目录下的 `gen_data.py` 生成，生成过程是确定性的。

## 方法

VioExplain 与 18 种基线比较。

| 类别 | 方法 |
|---|---|
| 过程监控 | PCA-RBC、FDA |
| 单标签诊断 | RF、XGB、LGBM、MiniRocket、MultiRocket、QUANT、1D-CNN、LSTM、ResNet、InceptionTime、MantisV2 |
| 多标签诊断 | BR、CC、ML-CNN |
| 知识型解释 | AEC、MinExplain |

未知事件实验把 VioExplain 的未知得分与 MDS、MSP、Energy 比较。

## 运行实验

脚本需要 Python 3.10 及以上版本和 `requirements.txt` 中的包。MantisV2 还需要其公开权重与加载代码。数据目录与结果目录在各目录的 `common.py` 顶部设置，其中 `/path/to/vioexplain` 与 `/home/user` 是占位路径。`final_run.py`、`gpu_*.py`、`agg_update_base.py`、`final_update_base.py`、`final_update.py`、`agg_update.py`、`final_unknown_v3.py`、`collect_unknown.py`、`final_cert5.py`、`cert5_finalize.py`、`final_timing.py`、`gen_data.py` 与 `hyd/hyd_common.py` 的顶部写有同样的目录。

每个脚本从环境变量读取设置。`V3_MODE` 取 `f0` 时只用单故障运行的知识，取 `f100t` 时加入反事实合成窗口，取 `f10t` 至 `f75t` 时只合成相应比例的故障对。`V3_RES` 指定结果目录，`V3_LOG` 与 `V3_METRICS` 指定输出文件名。各脚本的文档字符串给出用法。

TEP 实验在 `experiments/tii_final/` 下运行。

```bash
# 基于窗口描述与类型化违反的 CPU 基线
V3_MODE=f0 V3_METRICS=metrics_base.json python final_base.py RF MC-LGBM XGB FDA PCA-RBC BR-LGBM CC-LGBM AEC MinExplain
# 时间序列分类器
V3_MODE=f0 V3_METRICS=metrics_QUANT.json python final_rocket.py QUANT
# 深度诊断器与多标签网络，随后对其输出概率评分
python gpu_run.py cuda:0 && python gpu_run2.py cuda:0 && python gpu_save.py cuda:0 && python gpu_save2.py cuda:0
V3_MODE=f0 V3_METRICS=metrics_probs.json python final_probs.py ResNet:/path/to/resnet_probs.npz:single
# VioExplain
V3_MODE=f100t V3_FUNC=1 V3_NOSTEP=1 V3_ABL=full V3_TMODEL=/path/to/resnet_model.pt V3_TNAME=B \
  V3_METRICS=metrics_B.json python final_fuse5.py /path/to/resnet_probs_v2.npz
```

消融实验在最后一条命令中设置 `V3_ABL`（`noTemp`、`noOp`）或 `V3_NOTWIN=1`、`V3_FUNC=0`。覆盖率实验把 `V3_MODE` 设为 `f10t`、`f25t`、`f50t` 或 `f75t`，ML-CNN 用 `gpu_save3.py` 训练。`final_update_base.py` 与 `agg_update_base.py` 运行知识更新实验中的基线。`stats_extra.py` 与 `finding4.py` 计算正文引用的统计量。

未知事件得分、VioExplain 的知识更新曲线、恢复证书与计时同样在 `experiments/tii_final/` 下运行。

```bash
# 未知事件，先训练不含留出错误类型的时序诊断器，再对每个留出错误类型运行一次
python gpu_unknown.py cuda:0 <held-out fault ids>
V3_HELD=1 V3_MODE=f100t V3_FUNC=1 V3_TFEAT=0 V3_RES=final_v3/unknown/h1 python final_unknown_v3.py
python collect_unknown.py
# VioExplain 的知识更新，参考运行（0）与每个留出错误类型各一次
NJ=1 python final_update.py 0
V3_BRREFIT=1 NJ=1 python final_update.py 1
python agg_update.py
# 恢复证书
V3_MODE=f100t V3_ABL=full V3_FUNC=1 V3_NOTWIN=0 V3_NOSTEP=1 V3_TFEAT=0 V3_RES=final_v4/cert_work/nj8 \
  V3_TMODEL=/path/to/resnet_model.pt V3_TNAME=B NJ=8 OMP_NUM_THREADS=3 CERT_CACHE=/path/to/cert_cache \
  python final_cert5.py /path/to/resnet_probs_v2.npz
python cert5_finalize.py nj8
# 每个窗口的 CPU 时间
V3_MODE=f0 V3_LOG=log_timing.txt NJ=2 OMP_NUM_THREADS=2 python final_timing.py /path/to/cnn_probs.npz
```

`final_unknown_v3.py` 导入 `common_u.py`，后者是从每个组件中移除一个错误类型后的共享协议。`final_timing.py` 执行 `final_fuse.py` 中构建模型的部分，`final_cert5.py` 执行 `final_fuse5.py` 中构建模型的部分。`smoke_v3.py` 提供 `gpu_run.py` 导入的 MantisV2 加载函数。

HYD 实验在 `experiments/tii_final/hyd/` 下运行，TEP-C 的运行数据在 `experiments/tii_final/tep_generation/` 下生成。两个目录的 README 列出各自的命令。

仿真过程在 `experiments/tii_final/simproc*/` 下运行。

```bash
python gen_data.py cstr
V3_DATA=cstr V3_RES=simproc_v1/cstr V3_MODE=f0 V3_METRICS=metrics_base.json python final_base.py RF MC-LGBM XGB FDA PCA-RBC BR-LGBM CC-LGBM AEC MinExplain
V3_DATA=cstr V3_RES=simproc_v1/cstr V3_MODE=f100t python gpu_simproc.py cuda:0 cnn resnet lstm inceptiontime mlcnn_f0 mlcnn_f100t mantis
```

之后依次运行 `final_rocket.py`、`final_probs.py`、`final_fuse5.py` 与该目录的汇总脚本。各目录的 README 说明仿真器、注入的故障与数据布局。

## 核对表格

```bash
python results/print_main_table.py
```

该脚本从 `results/` 下的文件打印文中的主表。[`results/README.md`](results/README.md) 把每张表、每幅图和正文引用的每个数字对应到文件与字段。

## 许可

代码以 [MIT 许可](LICENSE) 发布。数据集、外部库与模型权重遵守各自的条款，见[第三方声明](THIRD_PARTY_NOTICES.md)。
