# coding=utf-8
"""评测指标（协议 docs/protocol_public.md §1.4 / §3）。

事件级根因维 P/R/F1：
  P = |P_e ∩ G_e| / |P_e|
  R = |P_e ∩ G_e| / |G_e|
  F1 = 2PR/(P+R)
  micro：把所有事件的交/预测/真值分别累加后算一次（大事件主导）。
  macro：每事件先算 F1 再求均（每事件等权，协议主表口径）。

点级检测（E-G）：F1（非 PA）与 F1（PA=point-adjust：一段内命中任一点则整段视为全命中），
事件级召回（事件内有任一点被标为异常即算命中）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def event_prf(pred_dims: Set[int], true_dims: Set[int]) -> Tuple[float, float, float, int, int, int]:
    """单事件维集合 P/R/F1。返回 (P, R, F1, inter, |pred|, |true|)。"""
    inter = len(pred_dims & true_dims)
    p = inter / len(pred_dims) if pred_dims else 0.0
    r = inter / len(true_dims) if true_dims else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, inter, len(pred_dims), len(true_dims)


def aggregate_events(rows: List[dict]) -> dict:
    """把每事件一行（含 inter/|pred|/|true|/f1）聚合成 micro + macro。

    rows 每行需含键：'inter','n_pred','n_true','f1'（precision/recall 可选）。
    """
    if not rows:
        return {"micro_p": 0.0, "micro_r": 0.0, "micro_f1": 0.0,
                "macro_f1": 0.0, "macro_p": 0.0, "macro_r": 0.0, "n_events": 0}
    tot_i = sum(r["inter"] for r in rows)
    tot_p = sum(r["n_pred"] for r in rows)
    tot_t = sum(r["n_true"] for r in rows)
    micro_p = tot_i / tot_p if tot_p else 0.0
    micro_r = tot_i / tot_t if tot_t else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    macro_p = float(np.mean([r.get("precision", 0.0) for r in rows]))
    macro_r = float(np.mean([r.get("recall", 0.0) for r in rows]))
    return {"micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
            "macro_f1": macro_f1, "macro_p": macro_p, "macro_r": macro_r,
            "n_events": len(rows)}


# ------------------------------------------------------------- 点级检测（E-G）
def point_f1(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    """点级 P/R/F1（非 PA）。pred/label 为 (T,) 0/1。"""
    pred = pred.astype(bool)
    label = label.astype(bool)
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """point-adjust：若某真值异常段内有任一点被预测为异常，则整段预测置 1（惯例）。"""
    pred = pred.astype(bool).copy()
    label = label.astype(bool)
    T = len(label)
    i = 0
    while i < T:
        if label[i]:
            j = i
            while j < T and label[j]:
                j += 1
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1
    return pred.astype(np.int8)


def point_f1_pa(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    """PA 版点级 F1（point-adjust 后再算 P/R/F1）。"""
    return point_f1(point_adjust(pred, label), label)


def event_recall(pred: np.ndarray, intervals: Sequence[dict]) -> float:
    """事件级召回：事件区间 [start,end) 内有任一点被预测为异常即命中。"""
    pred = pred.astype(bool)
    if not intervals:
        return 0.0
    hit = 0
    for iv in intervals:
        s, e = iv["start"], iv["end"]
        if pred[s:e].any():
            hit += 1
    return hit / len(intervals)
