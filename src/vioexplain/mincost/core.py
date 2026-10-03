# coding=utf-8
"""MinExplain 实例表示与统一接口（纯函数，零 IO）。

记号严格照 开发稿（公开说明见 docs/MINEXPLAIN.md） Definition 1.1--1.3：
  V = {0,...,n-1}    违反集（用整数索引）
  R = {0,...,m-1}    知识/异常表征集（候选），先验代价 w_r >= 0
  d(v,r) in [0, +inf]  匹配距离，类型不匹配记 +inf
  theta >= 0           匹配阈值
  cover_theta(r) = { v : d(v,r) <= theta }
  L(v) = { r : v in cover_theta(r) }
  解释 (E, a)：E ⊆ R，a: V -> E 且 v in cover(a(v))
  Cost(E,a) = Σ_{r∈E} w_r + Σ_{v∈V} d(v, a(v))

参数：Δ = max_r |cover(r)|，f = max_v |L(v)|，N = Σ_v |L(v)| = 边数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

INF = math.inf

# d 的输入形态：稀疏 dict {(v,r): dist}，或稠密 [n][m] 矩阵，或 callable(v,r)->dist。
DistInput = Union[Dict[Tuple[int, int], float], Sequence[Sequence[float]], Callable[[int, int], float]]


@dataclass
class MinExplainInstance:
    """预处理后的 MinExplain 实例。边 (v,r) 存在当且仅当 d(v,r) <= theta。"""
    n: int                                   # |V|
    m: int                                   # |R|
    w: List[float]                           # w[r] >= 0
    theta: float
    # 稀疏邻接（只含 d<=theta 的边）
    cover: List[List[int]] = field(default_factory=list)      # cover[r] = 排序前的候选违反索引
    L: List[List[int]] = field(default_factory=list)          # L[v] = 候选表征索引
    dist: Dict[Tuple[int, int], float] = field(default_factory=dict)  # 边距离 d(v,r)（仅存在的边）

    @property
    def Delta(self) -> int:
        return max((len(c) for c in self.cover), default=0)

    @property
    def f(self) -> int:
        return max((len(l) for l in self.L), default=0)

    @property
    def N(self) -> int:
        return sum(len(l) for l in self.L)

    def d(self, v: int, r: int) -> float:
        """边距离；不存在的边返回 +inf。"""
        return self.dist.get((v, r), INF)

    def feasible(self) -> bool:
        """可行性：每个 v 至少有一个候选表征。"""
        return all(len(self.L[v]) > 0 for v in range(self.n))


@dataclass
class Solution:
    """统一输出：选中的表征集 E、指派 assignment[v]=r（未指派为 None）、总代价 cost。"""
    E: List[int]
    assignment: List[Optional[int]]
    cost: float

    def as_set(self) -> frozenset:
        return frozenset(self.E)


def _dist_value(d: DistInput, v: int, r: int) -> float:
    if isinstance(d, dict):
        return d.get((v, r), INF)
    if callable(d):
        return d(v, r)
    # 稠密矩阵
    val = d[v][r]
    return INF if val is None else float(val)


def build_instance(
    violations: Sequence,
    candidates: Sequence,
    w: Sequence[float],
    d: DistInput,
    theta: float,
) -> MinExplainInstance:
    """构造 MinExplain 实例。

    violations: 长度 n 的可迭代（内容任意，只用其长度定 n；索引即身份）。
    candidates: 长度 m 的可迭代（同上定 m）。
    w:          长度 m 的先验代价，w[r] >= 0。
    d:          距离，dict{(v,r):dist} / 矩阵[n][m] / callable(v,r)->dist；> theta 或 +inf 视为无边。
    theta:      匹配阈值 >= 0。
    仅保留 d(v,r) <= theta 的边（有限距离）。
    """
    n = len(violations)
    m = len(candidates)
    w = [float(x) for x in w]
    if len(w) != m:
        raise ValueError(f"len(w)={len(w)} != m={m}")
    if theta < 0:
        raise ValueError("theta must be >= 0")

    cover: List[List[int]] = [[] for _ in range(m)]
    L: List[List[int]] = [[] for _ in range(n)]
    dist: Dict[Tuple[int, int], float] = {}

    for v in range(n):
        for r in range(m):
            dvr = _dist_value(d, v, r)
            if dvr <= theta and dvr < INF:
                cover[r].append(v)
                L[v].append(r)
                dist[(v, r)] = float(dvr)

    inst = MinExplainInstance(n=n, m=m, w=w, theta=float(theta),
                              cover=cover, L=L, dist=dist)
    return inst


def cost_of(inst: MinExplainInstance, E: Sequence[int], assignment: Sequence[Optional[int]]) -> float:
    """按 Definition 1.2 计算给定解释 (E, a) 的代价；不校验可行性（校验用 is_feasible）。"""
    E_set = set(E)
    total = sum(inst.w[r] for r in E_set)
    for v in range(inst.n):
        r = assignment[v]
        if r is None:
            return INF
        total += inst.d(v, r)
    return total


def is_feasible(inst: MinExplainInstance, E: Sequence[int], assignment: Sequence[Optional[int]]) -> bool:
    """校验 (E,a) 可行：每个 v 被指派到 E 中且 v in cover(a(v))（即 d(v,a(v))<=theta）。"""
    E_set = set(E)
    for v in range(inst.n):
        r = assignment[v]
        if r is None or r not in E_set:
            return False
        if inst.dist.get((v, r), INF) > inst.theta:
            return False
    return True


def cheapest_assignment(inst: MinExplainInstance, E: Sequence[int]) -> Tuple[List[Optional[int]], float]:
    """给定已开设表征集 E，为每个 v 选 E∩L(v) 中距离最小的表征（min_a Cost(E,a) 口径）。

    返回 (assignment, cost)。若某 v 无法被 E 覆盖，assignment[v]=None 且 cost=+inf。
    这是 开发稿（公开说明见 docs/MINEXPLAIN.md） T4(ii) 中 Cost(F)=min_a Cost(F,a) 的实现。
    """
    E_set = set(E)
    assignment: List[Optional[int]] = [None] * inst.n
    total = sum(inst.w[r] for r in E_set)
    feasible = True
    for v in range(inst.n):
        best_r, best_d = None, INF
        for r in inst.L[v]:
            if r in E_set:
                dvr = inst.dist[(v, r)]
                if dvr < best_d:
                    best_d, best_r = dvr, r
        if best_r is None:
            feasible = False
            assignment[v] = None
        else:
            assignment[v] = best_r
            total += best_d
    if not feasible:
        return assignment, INF
    return assignment, total
