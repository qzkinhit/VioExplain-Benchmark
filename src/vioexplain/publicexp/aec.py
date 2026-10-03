# coding=utf-8
"""历史兼容适配器，实际调用后加 MinExplain，不能作为原 AEC-Prototype 的等价复现。

AEC 解释：违反 V + 知识 R -> MinExplain 实例 -> 求解 -> 映射回维集合（协议 §1.4）。

AEC = Anomaly Explanation by set Covering（集合覆盖式异常解释）。本模块把公开数据的违反特征
与知识集接到仓库既有 MinExplain 求解器上（复用 src/vioexplain/mincost 的纯函数，不修改它）：

  违反 v_i（一条 (dim,ctype) 违反）      -> MinExplain 的客户 v
  表征 r_j（签名维集合 sig, 先验代价 w） -> MinExplain 的设施 r
  匹配距离 d(v, r)：v.dim ∈ sig(r) 时 d = w_r 的距离项（越界程度调制），否则 +inf（类型不匹配）
  cover_theta(r) = {v : d(v,r) <= theta}

求解用密度贪心 greedy_density（T2 界，懒惰堆），先做安全剪枝 prune（T7）。
解释 (E,a) 映射回维集合（协议 §1.4 AEC 口径）：
  P_e = 选中表征集 E 的签名维并集  ∩  该事件实际违反的维（只报被证据支持的维）。

无候选的违反不进入覆盖求解器。其维度记录于 unmatched_dims，并保留在历史兼容的
pred_dims 中作为待复核证据；它们不计入 covered，不能解释为知识支持的事件诊断。
cost 仅包含可匹配子实例的开设与指派代价。知识为空时该子实例为空，代价与覆盖数均为零。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..mincost.core import build_instance, cheapest_assignment
from ..mincost.greedy import greedy_density
from ..mincost.prune import prune
from .features import Violation
from .knowledge import Representation


@dataclass
class Explanation:
    """AEC 解释输出。"""
    pred_dims: Set[str]          # P_e：预测的根因维集合（映射回维集合）
    selected: List[int]          # 选中表征在 R 中的索引
    cost: float
    covered: int                 # 被覆盖的违反数
    n_viol: int                  # 总违反数
    unmatched_dims: Set[str] = field(default_factory=set)  # 无候选支持的待复核维


def _match_distance(v: Violation, r: Representation, theta: float) -> float:
    """匹配距离 d(v,r)：v.dim 在 sig(r) 中则给有限距离（越界程度越大越贴合 -> 距离越小），
    否则 +inf（类型不匹配 / 维不匹配）。距离范围落在 [0, theta] 内才成边。

    设计（对齐 MinExplain：距离小表示 v 被 r 很好地说明）：
      base = 越界越强越应被解释 -> d 随 mag 下降；signature 越大（越泛）略加惩罚。
    """
    if v.dim not in r.sig:
        return float("inf")
    # 归一化越界程度到 (0,1]；mag 越大越像真异常，越贴合表征
    m = max(0.0, min(1.0, v.mag))
    base = 1.0 - m                      # mag=1 -> 0, mag=0 -> 1
    penalty = 0.02 * (len(r.sig))       # 泛表征轻惩罚（偏好精确签名）
    return base * theta + penalty * theta


def explain_event(viols: List[Violation], R: List[Representation],
                  theta: float = 0.6) -> Explanation:
    """对一个事件的违反特征跑 AEC 解释（协议 §1.4）。

    theta: 匹配阈值（协议默认 0.6；越界程度决定的距离 <= theta 才成边）。
    """
    n = len(viols)
    vio_dims = {v.dim for v in viols}
    if n == 0:
        return Explanation(pred_dims=set(), selected=[], cost=0.0, covered=0, n_viol=0)

    if not R:
        # 保留原始违反证据，但不能将未知事件算作已由知识解释。
        return Explanation(pred_dims=set(vio_dims), selected=[], cost=0.0,
                           covered=0, n_viol=n, unmatched_dims=set(vio_dims))

    m = len(R)
    # 稀疏距离 dict
    dist: Dict[Tuple[int, int], float] = {}
    for i, v in enumerate(viols):
        for j, r in enumerate(R):
            dvr = _match_distance(v, r, theta)
            if dvr <= theta:
                dist[(i, j)] = dvr

    w = [r.w for r in R]
    inst = build_instance(violations=list(range(n)), candidates=list(range(m)),
                          w=w, d=dist, theta=theta)

    unmatched_dims = {viols[i].dim for i in range(n) if not inst.L[i]}
    if not inst.feasible():
        # 必须先抽出可行子实例。否则任何未知违反都会使求解 cost=inf，
        # 旧实现据此丢弃有效的 sol.E，连已匹配事件也会丢失。
        matched = [i for i in range(n) if inst.L[i]]
        if not matched:
            return Explanation(pred_dims=set(vio_dims), selected=[], cost=0.0,
                               covered=0, n_viol=n, unmatched_dims=unmatched_dims)
        remap = {old: new for new, old in enumerate(matched)}
        matched_dist = {(remap[i], j): value for (i, j), value in dist.items()}
        inst = build_instance(violations=matched, candidates=list(range(m)),
                              w=w, d=matched_dist, theta=theta)

    pr = prune(inst)
    sol = greedy_density(pr.inst)
    selected_orig = sorted(set(list(pr.forced) + [pr.kept[j] for j in sol.E]))

    sig_dims: Set[str] = set()
    for j in selected_orig:
        sig_dims |= R[j].sig
    # 协议 §1.4：P_e = 选中表征签名维并集 ∩ 该事件实际违反的维（只报被证据支持的维）
    pred = (sig_dims & vio_dims) | unmatched_dims
    if not pred:
        # 覆盖到的表征签名与违反维无交（罕见）：退回违反维
        pred = set(vio_dims)
    covered = sum(1 for a in sol.assignment if a is not None)
    return Explanation(pred_dims=pred, selected=selected_orig,
                       cost=float(sol.cost), covered=covered, n_viol=n,
                       unmatched_dims=unmatched_dims)
