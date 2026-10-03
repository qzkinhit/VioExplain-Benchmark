# coding=utf-8
"""违反特征：事件窗口 -> 违反集 V（协议 §1.2 / §0 记号对齐 IDEA_THEORY_BLUEPRINT §3）。

一个事件窗口聚合出的违反维集合 V_dims 即该事件的「违反特征」，所有方法（AEC / BARO /
EXstream / SHAP / LIME）都在**同一个** V_dims 上运行以保公平。

每条违反 v 绑定：
  dim   : 维度（列名）
  ctype : 约束类型 'domain' / 'speed' / 'corr'
  k     : 涉及序列数（domain/speed=1，corr=2 多序列约束）
  mag   : 越界程度（量化区间的中心量，供 MinExplain 距离与 AEC 排序）
  interval : (lo, hi) 越界程度区间（协议 §0：每条违反绑定一个越界程度区间）

事件的违反特征 = 该窗口内所有触发的 (dim, ctype) 去重后聚合（每类每维一条违反）。
V_dims = 违反涉及的维度集合（用于事件级评测的 P_e 映射与真值 G_e 比对）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .constraints import ConstraintSet, detect_pointwise


@dataclass
class Violation:
    """一条违反特征 v。"""
    dim: str
    ctype: str          # 'domain' | 'speed' | 'corr'
    k: int              # 涉及序列数（多序列约束 k>1）
    mag: float          # 越界程度（越界点的最大程度）
    interval: Tuple[float, float]   # 越界程度区间 [min_mag, max_mag]
    count: int          # 窗口内该 (dim,ctype) 越界的点数


def extract_violations(window: pd.DataFrame, cons: ConstraintSet,
                       min_points: int = 1) -> List[Violation]:
    """从一个事件窗口抽违反特征列表（协议 §1.2 逐点检测后按 (dim,ctype) 聚合）。

    min_points: 某 (dim,ctype) 至少 min_points 个越界点才成一条违反（抗单点噪声）。
    corr 违反：残差越界的对，两维各记一条 ctype='corr' k=2 违反。
    """
    det = detect_pointwise(window, cons)
    cols = det["columns"]
    viols: List[Violation] = []

    for j, c in enumerate(cols):
        # domain
        mask = det["domain"][:, j]
        cnt = int(mask.sum())
        if cnt >= min_points:
            mags = det["dom_mag"][mask, j]
            viols.append(Violation(dim=c, ctype="domain", k=1,
                                   mag=float(mags.max()),
                                   interval=(float(mags.min()), float(mags.max())),
                                   count=cnt))
        # speed
        mask = det["speed"][:, j]
        cnt = int(mask.sum())
        if cnt >= min_points:
            mags = det["spd_mag"][mask, j]
            viols.append(Violation(dim=c, ctype="speed", k=1,
                                   mag=float(mags.max()),
                                   interval=(float(mags.min()), float(mags.max())),
                                   count=cnt))

    # corr：残差越界的对
    for (d1, d2), mask in det["corr"].items():
        cnt = int(mask.sum())
        if cnt >= min_points:
            for c in (d1, d2):
                viols.append(Violation(dim=c, ctype="corr", k=2,
                                       mag=1.0, interval=(0.0, 1.0), count=cnt))
    return viols


def violation_dims(viols: List[Violation]) -> Set[str]:
    """V_dims：违反涉及的维度集合。"""
    return {v.dim for v in viols}


def dims_to_1indexed(dims: Set[str]) -> Set[int]:
    """列名 'd7' -> 7（SMD 逐维根因真值为 1-indexed 列）。非 dN 格式忽略。"""
    out = set()
    for c in dims:
        if isinstance(c, str) and c.startswith("d") and c[1:].isdigit():
            out.add(int(c[1:]))
    return out
