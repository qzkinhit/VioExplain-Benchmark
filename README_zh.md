# VioExplain Benchmark

[English](README.md)

VioExplain 以异常表征解释约束违反，通过最小代价覆盖组织解释。仓库保留**违反提取、表征匹配、异常解释、知识更新**这一原有方法主线，提供方法实现、未知违反的独立记录、公开数据协议、基线适配器及已有探索结果。

时序基模组件属于研究扩展。目前的小规模试验与正式事件实验尚未证明相对于统计对照的稳定优势，也未证明达到最优性能。合成检查、原生异常检测、异常贡献维归因和事件诊断分别报告。

## 方法身份

| 方法 | 目标与求解方式 | 入口 |
|---|---|---|
| `AEC-Prototype` | 所提供的区间表征事件覆盖原型及原 `Select` 过程 | `vioexplain.setcover.set_covering.MyCover` |
| `MinExplain` | 后加的表征开设代价与违反指派代价目标，采用密度贪心或原始对偶求解 | `vioexplain.api.explain`、`vioexplain.mincost` |

二者的目标函数与算法不同。默认公开接口运行 **MinExplain**，`AEC-Prototype` 以独立入口保留，并与所提供旧原型的演示结果核对。当前对照以所提供的原英文稿和代码为依据。适配器与接口修复不属于研究创新。

## 快速开始

使用 Python 3.10 或更高版本，并建立独立环境。最小运行检查不需要 GPU、模型权重或数据集。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash run.sh smoke
bash run.sh prototype-smoke
python -m pytest -q
```

`smoke` 使用两条可匹配违反和一条未知违反组成的微型合成实例，仅检查接口约定，不产生论文实验指标。

## 目录说明

| 路径 | 内容 |
|---|---|
| [`src/vioexplain/`](src/vioexplain/README.md) | 公开接口、违反提取、异常表征、集合覆盖与知识更新 |
| `benchmark/` | 官方数据适配、指标计算与冻结协议 |
| [`data/`](data/README.md) | 数据卡与固定上游提交的下载脚本，原始数据仅保存在使用者本地 |
| `configs/` | 最小检查与正式实验配置 |
| `run_vioexplain/` | 实验入口 |
| [`result/`](result/README.md) | 纳入版本管理的汇总证据与本地运行输出 |
| [`docs/`](docs/REPRODUCING.md) | 复现、基线实现差异、数据适用范围、来源与局限 |
| `tests/` | 接口、部分覆盖、优化求解与探索组件的检查 |

方法入口是 [`vioexplain.api.explain`](src/vioexplain/api.py)。输出分别记录被所选表征支持的证据和未知证据，未知违反不会被计作已解释。

```python
from vioexplain import Violation, Representation, explain

violations = [Violation("temperature", "domain", 1, 1.0, (1.0, 1.0), 3)]
knowledge = [Representation({"temperature"}, w=1.0, support=2)]
result = explain(violations, knowledge, theta=0.6)
print(result.to_dict())
```

可选参数 `distances={(违反索引, 表征索引): 非负代价}` 用于接入匹配模型，覆盖问题保持一致。缺失的边表示不兼容。[接口说明](docs/API.md) 定义索引、代价、指派与未知违反的含义。

## 数据与评测

```bash
python data/download.py smd --show-source
python data/download.py smd
python data/download.py skab
bash run.sh formal --help
```

SMD 提供原生异常贡献维标注。SKAB 提供异常与变化点标注，不能单独验证根因维解释。正式统计基线入口还支持经典 Tennessee Eastman 模拟数据。Exathlon 仍作为单独记录的候选扩展。[数据卡](docs/DATASETS.md) 说明数据支持的结论，[基线说明](docs/BASELINES.md) 区分官方实现与方法适配。

## 正式基准命令

```bash
# CPU 统计检测，遍历给定目录中的官方完整对象。
bash run.sh formal statistical --dataset smd --data data/raw/smd --output result/runs/smd-statistical
bash run.sh formal statistical --dataset skab --data data/raw/skab --output result/runs/skab-statistical
```

SMD 条件归因轨道还可调用官方 BARO 的 RobustScorer 组件。TranAD 与 TreeSHAP 使用独立入口及依赖。[正式复现说明](docs/FORMAL_BENCHMARK.md) 给出固定来源的获取方式、运行命令、适配边界和状态。统计检测、条件维归因与事件类诊断分别评价。

## 证据状态

软件完成状态与研究验证状态分别记录。原有 AEC-Prototype 与后加 MinExplain 核心和探索求解器已经实现。[历史汇总](result/summary/historical/manifest.json) 同时保留改善与负结果，包括冻结基模匹配头未获得跨域增益的结果。这些历史试验不属于新的盲测基准。[已完成正式批次](docs/BENCHMARK_STATUS.md) 提供独立协议、逐实体记录与运行清单。这些基线结果本身不能证明新 VioExplain 方法具有优势。

[复现说明](docs/REPRODUCING.md) 给出命令与输出约定，[局限说明](docs/LIMITATIONS.md) 记录已知边界。仓库不包含私有工业测量、博士论文全文、模型权重或复制的第三方基线源码。

## 许可与引用

本项目代码采用 [MIT 许可](LICENSE)。数据、外部实现与模型权重遵守各自条款，详见[第三方声明](THIRD_PARTY_NOTICES.md)。[CITATION.cff](CITATION.cff) 描述本软件版本，不宣称已发表期刊论文或已获得论文 DOI。

历史同名方法的具体实现见[方法身份表](docs/METHOD_IDENTITIES.md)。

[锁定的 TEP 事件实验](docs/EVENT_MATCHING.md) 包含原始区间原型、显式代价适配及采用区间、原始时序或冻结 Chronos-2 匹配的 MinExplain。九个正式基线批次与该事件实验均已完成，改善与负结果同时保留。

运行入口自动将控制台输出及来源保存到`log/runs/`。已发布结果在`result/summary/`，新实验建议输出到`result/runs/`。详见[实验工件约定](docs/EXPERIMENT_RECORDS.md)。
