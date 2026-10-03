# coding=utf-8
"""T3 一遍 primal-dual（local-ratio）算法，O(N)，代价 <= f·OPT。

theorems_draft.tex §T3（Theorem T3 + Lemma pd-invariants）。
Bar-Yehuda--Even 局部比率法向带指派代价的推广。维护残余权 w~_r（初始化 w_r）。
按任意固定顺序遍历每个 v：
  1. t_v = min_{r∈L(v)} (w~_r + d(v,r))，取到最小的 r*_v；
  2. 对每个 r∈L(v)：z_{vr} = [t_v - d(v,r)]_+，w~_r -= z_{vr}；
  3. u_v = t_v，a(v) = r*_v（更新后 w~_{r*_v} = 0）。
输出 E = {a(v)}、指派 a。

保证：feasible，Cost(E,a) <= f·OPT。与 T2 合成实例自适应界 min(1+lnΔ, f)。
纯函数，零 IO。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .core import MinExplainInstance, Solution, INF


def pd_onepass(
    inst: MinExplainInstance,
    order: Optional[Sequence[int]] = None,
) -> Solution:
    """一遍 primal-dual。order 指定处理违反的顺序（默认 0..n-1）。

    返回 Solution(E, assignment, cost)。不可行（某 v 无候选）时该 v 指派 None、cost=+inf。
    """
    n, m = inst.n, inst.m
    if order is None:
        order = range(n)

    w_res = list(inst.w)                 # 残余权 w~_r
    assignment: List[Optional[int]] = [None] * n
    E_set = set()

    for v in order:
        Lv = inst.L[v]
        if not Lv:
            assignment[v] = None
            continue
        # step 1: t_v 与 r*_v
        t_v = INF
        r_star = None
        for r in Lv:
            val = w_res[r] + inst.dist[(v, r)]
            if val < t_v:
                t_v = val
                r_star = r
        # step 2: z 更新残余权（z_{vr}=[t_v-d(v,r)]_+，扣减 w~_r）
        for r in Lv:
            z = t_v - inst.dist[(v, r)]
            if z > 0:
                w_res[r] -= z
                if w_res[r] < 0:  # 数值护栏（理论上 >=0，Lemma pd-invariants(i)）
                    w_res[r] = 0.0
        # step 3: 指派
        assignment[v] = r_star
        E_set.add(r_star)

    E = sorted(E_set)
    # 代价：Σ_{r∈E} w_r + Σ_v d(v,a(v))
    feasible = all(assignment[v] is not None for v in range(n))
    if not feasible:
        cost = INF
    else:
        cost = sum(inst.w[r] for r in E_set)
        for v in range(n):
            cost += inst.dist[(v, assignment[v])]
    return Solution(E=E, assignment=assignment, cost=cost)
