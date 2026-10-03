# coding=utf-8
"""T7(a) 聚合支配剪枝 + T7(b) 安全迫选（唯一覆盖必选）。纯函数，零 IO。

theorems_draft.tex：
  T7(a) Theorem 7.1（dominance）：若 cover(r1) ⊆ cover(r2) 且
      w_{r1} >= w_{r2} + Σ_{v∈cover(r1)} [d(v,r2) - d(v,r1)]_+     ... (eq:dom)
    则删 r1 不改变最优值（交换论证）。逐点形式（w_{r1}>=w_{r2} 且逐 v d(v,r1)>=d(v,r2)）为特例。
  T7(b) Theorem 7.2（unique cover forced）：若 L(v)={r}，则每个可行解含 r；
    固定 r∈E（w_r 只记一次）保最优。预指派 u∈cover(r) 到 r 仅当
    d(u,r)=min_{r'∈L(u)} d(u,r') 时 optimality-safe；一般情形指派留给后续优化。

审计裁决（docs/code_audit_pruning.md）：不整删 cover(r)。本模块的 prune() 只：
  - 迫选：把唯一覆盖的 r 计入 forced（记账 w_r 一次），仅在最小距离条件下预指派；
  - 支配：删去被支配的 r1（不动违反、不动指派）。
产出可交给 greedy/pd/exact 继续在缩减实例上求解。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .core import MinExplainInstance, INF, build_instance


@dataclass
class PruneResult:
    """剪枝产物。

    inst          : 缩减后的实例（表征集可能变少；违反集不变）。
    kept          : 保留的表征在原实例中的索引列表（inst 中表征 j 对应 kept[j]）。
    forced        : 被迫选的原表征索引集合（L(v)={r} 触发），其 w_r 已在 forced_open_cost 记账。
    pre_assign    : 安全预指派 {原违反索引 v: 原表征索引 r}（仅 d(v,r) 最小时预指派）。
    forced_open_cost : Σ_{r∈forced} w_r（开设代价，已固定，不应在后续重复计）。
    removed_dominated: 被支配删除的原表征索引集合。
    """
    inst: MinExplainInstance
    kept: List[int]
    forced: Set[int] = field(default_factory=set)
    pre_assign: Dict[int, int] = field(default_factory=dict)
    forced_open_cost: float = 0.0
    removed_dominated: Set[int] = field(default_factory=set)


def _dominates(inst: MinExplainInstance, r1: int, r2: int) -> bool:
    """T7(a) eq:dom：r2 支配 r1（cover(r1)⊆cover(r2) 且聚合代价条件成立）=> 可删 r1。"""
    if r1 == r2:
        return False
    c1 = set(inst.cover[r1])
    if not c1:
        return True  # 空覆盖表征永远可删（不覆盖任何违反）
    c2 = set(inst.cover[r2])
    if not c1.issubset(c2):
        return False
    # w_{r1} >= w_{r2} + Σ_{v∈cover(r1)} [d(v,r2) - d(v,r1)]_+
    extra = 0.0
    for v in c1:
        gap = inst.dist[(v, r2)] - inst.dist[(v, r1)]
        if gap > 0:
            extra += gap
    return inst.w[r1] >= inst.w[r2] + extra - 1e-12


def _dominates_pointwise(inst: MinExplainInstance, r1: int, r2: int) -> bool:
    """逐点特例：cover(r1)⊆cover(r2)，w_{r1}>=w_{r2}，逐 v d(v,r1)>=d(v,r2)。"""
    if r1 == r2:
        return False
    c1 = set(inst.cover[r1])
    if not c1:
        return True
    c2 = set(inst.cover[r2])
    if not c1.issubset(c2):
        return False
    if inst.w[r1] < inst.w[r2] - 1e-12:
        return False
    for v in c1:
        if inst.dist[(v, r1)] < inst.dist[(v, r2)] - 1e-12:
            return False
    return True


def prune(
    inst: MinExplainInstance,
    use_dominance: bool = True,
    pointwise: bool = False,
    use_forcing: bool = True,
) -> PruneResult:
    """对实例做安全剪枝。默认聚合支配 + 安全迫选；pointwise=True 用逐点支配特例。

    返回 PruneResult；缩减实例中每个 v 仍保留（可行性不破坏）。
    迫选只固定 r∈forced 与其 w_r 记账；预指派仅在最小距离条件下发生（optimality-safe）。
    支配删除是保最优的（T7(a)）。两步都不整删 cover(r)。
    """
    n, m = inst.n, inst.m
    removed: Set[int] = set()
    forced: Set[int] = set()
    pre_assign: Dict[int, int] = {}
    forced_open_cost = 0.0

    # ---- T7(b) 安全迫选：L(v) = {r} 的唯一覆盖 ----
    if use_forcing:
        for v in range(n):
            if len(inst.L[v]) == 1:
                r = inst.L[v][0]
                if r not in forced:
                    forced.add(r)
                    forced_open_cost += inst.w[r]
        # 安全预指派：对每个被迫选的 r，其 cover(r) 中若 u 的最小距离恰为 d(u,r)，可预指派
        for r in forced:
            for u in inst.cover[r]:
                # d(u,r) 是否为 L(u) 内最小
                min_d = min(inst.dist[(u, rp)] for rp in inst.L[u])
                if abs(inst.dist[(u, r)] - min_d) <= 1e-12:
                    pre_assign[u] = r

    # ---- T7(a) 支配剪枝 ----
    if use_dominance:
        dom_fn = _dominates_pointwise if pointwise else _dominates
        # 逐对判定；被迫选的 r 不删（它们必在解里，删掉会破坏 T7(b) 的锁定）
        for r1 in range(m):
            if r1 in removed or r1 in forced:
                continue
            for r2 in range(m):
                if r2 == r1 or r2 in removed:
                    continue
                if dom_fn(inst, r1, r2):
                    # 若 r1 支配 r2 且 r2 也支配 r1（等价），只删索引较大者以避免互删
                    if dom_fn(inst, r2, r1) and r2 < r1:
                        continue
                    removed.add(r1)
                    break

    kept = [r for r in range(m) if r not in removed]
    kept_pos = {r: j for j, r in enumerate(kept)}

    # 重建缩减实例（违反集不变，表征集为 kept；距离用 dict 形态直接复制存在的边）
    new_w = [inst.w[r] for r in kept]
    new_dist: Dict[Tuple[int, int], float] = {}
    for r in kept:
        j = kept_pos[r]
        for v in inst.cover[r]:
            new_dist[(v, j)] = inst.dist[(v, r)]
    reduced = build_instance(
        violations=list(range(n)),
        candidates=list(range(len(kept))),
        w=new_w,
        d=new_dist,
        theta=inst.theta,
    )

    return PruneResult(
        inst=reduced,
        kept=kept,
        forced=forced,
        pre_assign=pre_assign,
        forced_open_cost=forced_open_cost,
        removed_dominated=removed,
    )
