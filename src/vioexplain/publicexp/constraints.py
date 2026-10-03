# coding=utf-8
"""约束挖掘与违反检测（协议 docs/protocol_public.md §1.2 / §2）。

三类约束，全部从**无标注的 train / 参考段**挖出（无标注泄漏防线）：
  domain[d] = (quantile(train[d], q_lo), quantile(train[d], q_hi))     值域界
  speed[d]  = (quantile(diff(train[d]), q_lo), quantile(diff(train[d]), q_hi))  一阶差分界
  corr 对   = {(d1,d2): |pearson(train[d1],train[d2])| >= rho_thr}    维间相关（多序列约束 k=2）

评测/知识构建时对任意窗口逐点检查：
  值越 domain 界            -> 该维 domain 违反（k=1）
  一阶差分越 speed 界        -> 该维 speed 违反（k=1）
  相关维对的线性残差越界      -> 两维各记一条 corr 违反（k=2，多序列约束）

约束挖掘方法无关：挖出的约束喂给所有方法（AEC/BARO/EXstream/SHAP/LIME）产生**同一**违反空间，
保证公平（协议 §1.2 末句）。纯 numpy / pandas，零方法层依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConstraintSet:
    """从 train 段挖出的约束（一台机器 / 一条 trace 一套）。方法无关。"""
    columns: List[str]
    domain: Dict[str, Tuple[float, float]]                 # 列 -> (lo, hi)
    speed: Dict[str, Tuple[float, float]]                  # 列 -> (diff_lo, diff_hi)
    corr_pairs: List[Tuple[str, str, float, float]]        # (d1, d2, slope, resid_std)
    q_lo: float = 0.005
    q_hi: float = 0.995
    rho_thr: float = 0.9
    resid_z: float = 4.0                                   # corr 残差越界的 z 倍数

    def summary(self) -> dict:
        return {"n_cols": len(self.columns),
                "n_domain": len(self.domain),
                "n_speed": len(self.speed),
                "n_corr_pairs": len(self.corr_pairs)}


def mine_constraints(train_df: pd.DataFrame,
                     q_lo: float = 0.005, q_hi: float = 0.995,
                     rho_thr: float = 0.9, resid_z: float = 4.0,
                     max_corr_pairs: int = 200) -> ConstraintSet:
    """从 train 段挖 domain / speed / corr 约束（协议 §1.2）。train 只含正常值，不碰标注。

    corr 对：|pearson| >= rho_thr 的维对，拟合线性 d2 ~ slope*d1，记残差标准差供越界判定。
    近似常数列（std 极小）跳过 corr（避免 nan 与病态拟合）。max_corr_pairs 限制对数（高维 trace）。
    """
    cols = list(train_df.columns)
    X = train_df.values.astype(np.float64)
    domain: Dict[str, Tuple[float, float]] = {}
    speed: Dict[str, Tuple[float, float]] = {}

    for j, c in enumerate(cols):
        x = X[:, j]
        lo, hi = np.quantile(x, q_lo), np.quantile(x, q_hi)
        domain[c] = (float(lo), float(hi))
        dx = np.diff(x)
        if dx.size == 0:
            speed[c] = (0.0, 0.0)
        else:
            slo, shi = np.quantile(dx, q_lo), np.quantile(dx, q_hi)
            speed[c] = (float(slo), float(shi))

    # corr 对：只在非常数列间找强相关（皮尔逊）
    stds = X.std(axis=0)
    active = [j for j in range(len(cols)) if stds[j] > 1e-9]
    corr_pairs: List[Tuple[str, str, float, float]] = []
    if len(active) >= 2:
        Xa = X[:, active]
        # 相关矩阵（列标准化后）
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(Xa, rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        na = len(active)
        for ia in range(na):
            for ib in range(ia + 1, na):
                if abs(C[ia, ib]) >= rho_thr:
                    j1, j2 = active[ia], active[ib]
                    x1, x2 = X[:, j1], X[:, j2]
                    v1 = x1.var()
                    slope = float(np.cov(x1, x2)[0, 1] / v1) if v1 > 1e-12 else 0.0
                    resid = x2 - slope * x1
                    rstd = float(resid.std()) + 1e-9
                    corr_pairs.append((cols[j1], cols[j2], slope, rstd))
                    if len(corr_pairs) >= max_corr_pairs:
                        break
            if len(corr_pairs) >= max_corr_pairs:
                break

    return ConstraintSet(columns=cols, domain=domain, speed=speed,
                         corr_pairs=corr_pairs, q_lo=q_lo, q_hi=q_hi,
                         rho_thr=rho_thr, resid_z=resid_z)


def detect_pointwise(window: pd.DataFrame, cons: ConstraintSet):
    """对一个窗口逐点检测三类违反。返回逐类越界的布尔矩阵与量化程度。

    返回 dict：
      'domain' : (T, m) 布尔，值越 domain 界
      'speed'  : (T, m) 布尔，一阶差分越 speed 界
      'corr'   : dict[(d1,d2)] -> (T,) 布尔，corr 残差越界（两维共享该布尔）
      'dom_mag': (T, m) float，domain 越界程度（越界量 / 界宽，供违反量化区间）
      'spd_mag': (T, m) float，speed 越界程度
    列顺序对齐 cons.columns（缺列按 0 处理）。
    """
    cols = cons.columns
    # 对齐窗口到约束列（窗口可能是 df 子集）
    W = np.zeros((len(window), len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        if c in window.columns:
            W[:, j] = window[c].values.astype(np.float64)
    T = W.shape[0]

    dom = np.zeros((T, len(cols)), dtype=bool)
    dom_mag = np.zeros((T, len(cols)), dtype=np.float64)
    spd = np.zeros((T, len(cols)), dtype=bool)
    spd_mag = np.zeros((T, len(cols)), dtype=np.float64)

    for j, c in enumerate(cols):
        lo, hi = cons.domain[c]
        x = W[:, j]
        below = x < lo
        above = x > hi
        dom[:, j] = below | above
        width = (hi - lo) if (hi - lo) > 1e-9 else 1.0
        dom_mag[below, j] = (lo - x[below]) / width
        dom_mag[above, j] = (x[above] - hi) / width

        slo, shi = cons.speed[c]
        dx = np.empty(T, dtype=np.float64)
        dx[0] = 0.0
        if T > 1:
            dx[1:] = np.diff(x)
        sbelow = dx < slo
        sabove = dx > shi
        spd[:, j] = sbelow | sabove
        swidth = (shi - slo) if (shi - slo) > 1e-9 else 1.0
        spd_mag[sbelow, j] = (slo - dx[sbelow]) / swidth
        spd_mag[sabove, j] = (dx[sabove] - shi) / swidth

    col_idx = {c: j for j, c in enumerate(cols)}
    corr: Dict[Tuple[str, str], np.ndarray] = {}
    for (d1, d2, slope, rstd) in cons.corr_pairs:
        j1, j2 = col_idx[d1], col_idx[d2]
        resid = W[:, j2] - slope * W[:, j1]
        corr[(d1, d2)] = np.abs(resid) > cons.resid_z * rstd

    return {"domain": dom, "dom_mag": dom_mag, "speed": spd, "spd_mag": spd_mag,
            "corr": corr, "columns": cols}
