# coding=utf-8
"""小实例精确求解：ILP（pulp/CBC 后端，OR-Tools 可选）+ 暴力枚举（对拍）。纯函数，零 IO。

theorems_draft.tex §T3 的 (P)：
  min Σ_r w_r x_r + Σ_v Σ_{r∈L(v)} d(v,r) y_{vr}
  s.t. Σ_{r∈L(v)} y_{vr} >= 1        ∀v
       y_{vr} <= x_r                 ∀v, r∈L(v)
       x,y ∈ {0,1}
整数解即精确 OPT（真最优）。给 E-I 精确对照与对拍的 ground truth。
关键（审计裁决）：y_{vr} 对 r∈L(v) 全建，cover(r) 内违反可自由改投任意已开设表征，绝不整删。

暴力枚举 brute_force：枚举 R 的所有子集 E，对每个可行 E 用 min_a Cost(E,a)（cheapest_assignment），
取最小。n,m 小（<=~12/子集数可控）时用于严格对拍最优性。
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

from .core import MinExplainInstance, Solution, INF, cheapest_assignment


def brute_force(inst: MinExplainInstance) -> Solution:
    """暴力枚举所有表征子集，返回精确最优 Solution。仅用于小实例对拍。

    覆盖每个 v 至少需一个 r∈L(v)。枚举 R 的子集（m 小），对可行子集用最便宜指派求代价。
    优化：只需枚举"极小可行子集"过多，故直接遍历 2^m（m<=~18 可接受）。
    """
    m, n = inst.m, inst.n
    if not inst.feasible():
        return Solution(E=[], assignment=[None] * n, cost=INF)

    best_cost = INF
    best_E: List[int] = []
    best_assign: List[Optional[int]] = [None] * n

    reps = list(range(m))
    # 按子集大小递增枚举，便于早停剪枝（一旦当前 w 下界已超 best 可跳过大子集）
    for size in range(1, m + 1):
        for combo in itertools.combinations(reps, size):
            E_set = set(combo)
            open_cost = sum(inst.w[r] for r in E_set)
            if open_cost >= best_cost:
                continue  # 仅开设代价已不优
            assignment, cost = cheapest_assignment(inst, combo)
            if cost < best_cost:
                best_cost = cost
                best_E = list(combo)
                best_assign = assignment
    return Solution(E=best_E, assignment=best_assign, cost=best_cost)


def _solve_pulp(inst: MinExplainInstance, time_limit: Optional[float]) -> Optional[Solution]:
    try:
        import pulp
    except Exception:
        return None
    n, m = inst.n, inst.m
    prob = pulp.LpProblem("MinExplain", pulp.LpMinimize)
    x = {r: pulp.LpVariable(f"x_{r}", cat="Binary") for r in range(m)}
    y = {}
    for v in range(n):
        for r in inst.L[v]:
            y[(v, r)] = pulp.LpVariable(f"y_{v}_{r}", cat="Binary")
    # 目标
    prob += (
        pulp.lpSum(inst.w[r] * x[r] for r in range(m))
        + pulp.lpSum(inst.dist[(v, r)] * y[(v, r)] for v in range(n) for r in inst.L[v])
    )
    # 覆盖约束
    for v in range(n):
        if inst.L[v]:
            prob += pulp.lpSum(y[(v, r)] for r in inst.L[v]) >= 1
        else:
            # 不可行
            return Solution(E=[], assignment=[None] * n, cost=INF)
    # 耦合约束 y<=x
    for (v, r) in y:
        prob += y[(v, r)] <= x[r]
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        return None
    E = [r for r in range(m) if x[r].value() is not None and x[r].value() > 0.5]
    assignment: List[Optional[int]] = [None] * n
    for v in range(n):
        for r in inst.L[v]:
            if y[(v, r)].value() is not None and y[(v, r)].value() > 0.5:
                assignment[v] = r
                break
    # 用最便宜指派重算代价（ILP 的 y 可能有并列，统一口径）
    assignment2, cost = cheapest_assignment(inst, E)
    return Solution(E=E, assignment=assignment2, cost=cost)


def exact_ilp(
    inst: MinExplainInstance,
    time_limit: Optional[float] = None,
    fallback_bruteforce: bool = True,
) -> Solution:
    """ILP 精确求解 MinExplain。优先 pulp/CBC；不可用或求解失败时回退暴力枚举。

    返回 Solution(E, assignment, cost)，cost 为精确 OPT（在给定 time_limit 下达最优时）。
    """
    if not inst.feasible():
        return Solution(E=[], assignment=[None] * inst.n, cost=INF)
    sol = _solve_pulp(inst, time_limit)
    if sol is not None:
        return sol
    if fallback_bruteforce:
        return brute_force(inst)
    raise RuntimeError("No ILP backend available (pulp not installed) and fallback disabled.")
