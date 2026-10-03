# coding=utf-8
"""异常表征知识集 R 的构建（协议 docs/protocol_public.md §1.3 / §2）。

无标签泄漏防线：R 的签名来自**训练组事件检测出的违反维模式**（用 §1.2 同一套约束抽 V_dims，
**不看该事件真值**），不是训练组的真值维集合本身，杜绝把答案抄进知识库。评测组事件标注不进 R。

流程（跨组，默认 SMD train_groups=[1] / Exathlon 学 trace）：
  1. 对训练组每个事件窗口，用约束抽违反维集合 V_dims（不看真值）。
  2. 把这些 V_dims 按 Jaccard 相似度做贪心聚类（阈值 jaccard_thr 默认 0.5）。
  3. 每个簇取维度出现频率 >= freq_thr 的维作为签名 sig(r)，
     先验代价 w_r = 1 / 簇内事件数（常见模式更便宜）。

得到 R = [Representation(sig, w)]，评测时用它覆盖新事件的违反（见 aec.py）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np


@dataclass
class Representation:
    """异常表征 r：签名维集合 sig 与先验代价 w。"""
    sig: Set[str]        # 签名维（列名集合）
    w: float             # 先验代价 w_r >= 0
    support: int         # 簇内事件数（支持度）


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build_knowledge(train_event_dims: List[Set[str]],
                    jaccard_thr: float = 0.5,
                    freq_thr: float = 0.8,
                    seed: int = 0) -> List[Representation]:
    """从训练组事件的违反维集合列表构建知识集 R（协议 §1.3）。

    train_event_dims: 每个训练组事件检测出的 V_dims（不含真值）。
    贪心聚类：seed 控制事件遍历顺序（协议 §1.4：聚类贪心顺序受 seed 控制）。
    """
    events = [set(d) for d in train_event_dims if d]
    if not events:
        return []
    rng = np.random.default_rng(seed)
    order = list(range(len(events)))
    rng.shuffle(order)

    clusters: List[List[int]] = []      # 每簇的事件索引
    centroids: List[Set[str]] = []      # 每簇的当前维并集（用于 Jaccard 判定）
    for idx in order:
        dims = events[idx]
        best_c, best_j = -1, jaccard_thr
        for ci, cen in enumerate(centroids):
            jj = _jaccard(dims, cen)
            if jj >= best_j:
                best_j, best_c = jj, ci
        if best_c >= 0:
            clusters[best_c].append(idx)
            centroids[best_c] = centroids[best_c] | dims
        else:
            clusters.append([idx])
            centroids.append(set(dims))

    R: List[Representation] = []
    for members in clusters:
        n = len(members)
        # 维度出现频率
        freq: Dict[str, int] = {}
        for idx in members:
            for d in events[idx]:
                freq[d] = freq.get(d, 0) + 1
        sig = {d for d, cnt in freq.items() if cnt / n >= freq_thr}
        if not sig:
            # 频率阈值下签名为空时退回取最高频维（保证表征非空）
            mx = max(freq.values())
            sig = {d for d, cnt in freq.items() if cnt == mx}
        R.append(Representation(sig=sig, w=1.0 / n, support=n))
    return R


def knowledge_summary(R: List[Representation]) -> dict:
    return {"n_representations": len(R),
            "avg_sig_size": float(np.mean([len(r.sig) for r in R])) if R else 0.0,
            "total_support": int(sum(r.support for r in R))}
