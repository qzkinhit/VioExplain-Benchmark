# coding=utf-8
"""E-G 违反特征编码（协议 §3）：把 §1.2 约束在每点产生的逐维越界指示拼成额外特征列。

VioDetect 重定位为「违反特征编码层」：对任意检测器（LOF/IForest/AE），输入
  A（裸）      : 原始 m 维
  B（+违反编码）: 原始 m 维 ‖ 违反编码维（domain/speed 逐维 0/1，corr 对逐点 0/1）
用于测量违反编码对不同检测器的影响（ΔF1 = F1(B) − F1(A)），不预设增益。

编码来自 §1.2 约束（train 段挖），逐点检测，纯数值，零方法层依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import ConstraintSet, detect_pointwise


def violation_encoding(df: pd.DataFrame, cons: ConstraintSet) -> np.ndarray:
    """对整段 df 逐点产生违反编码矩阵 (T, k)。

    k = m（domain 逐维 0/1） + m（speed 逐维 0/1） + n_corr_pairs（corr 逐点 0/1）。
    """
    det = detect_pointwise(df, cons)
    dom = det["domain"].astype(np.float32)      # (T, m)
    spd = det["speed"].astype(np.float32)       # (T, m)
    parts = [dom, spd]
    if det["corr"]:
        T = dom.shape[0]
        corr = np.zeros((T, len(det["corr"])), dtype=np.float32)
        for ci, (_, mask) in enumerate(det["corr"].items()):
            corr[:, ci] = mask.astype(np.float32)
        parts.append(corr)
    return np.concatenate(parts, axis=1)


def concat_encoding(X: np.ndarray, enc: np.ndarray) -> np.ndarray:
    """原始特征 ‖ 违反编码。"""
    return np.concatenate([X.astype(np.float32), enc.astype(np.float32)], axis=1)
