# coding=utf-8
"""T2 密度贪心（懒惰堆实现），T7(c) 复杂度。

theorems_draft.tex Definition 2.2（density greedy）+ Lemma 2.1（prefix optimality）
+ Lemma unimodal（prefix density 单峰，二分找最小）+ Theorem T7c（懒惰堆 O(N(log m+logΔ)+n log n)）。

每轮在所有 (r, S)（S ⊆ cover(r)∩alive）中选密度
    rho(r,S) = (w_r + Σ_{v∈S} d(v,r)) / |S|
最小者；由 Lemma 2.1 最优 S 必为按 d 升序的前缀，由 Lemma unimodal 前缀密度
phi(s)=(w_r+Σ_{i<=s} d_i)/s 单峰，最小点用二分（predicate d_{s+1} >= phi(s)）定位。
懒惰堆：堆键为各 r 当前最优前缀密度；某 r 覆盖的违反被删则置脏，弹出时重算。

保证（Theorem T2）：Cost(E,a) <= H(Δ)·OPT <= (1+ln Δ)·OPT。
纯函数，零 IO。
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

from .core import MinExplainInstance, Solution, INF


def _best_prefix(sorted_d: List[float], w_r: float) -> Tuple[float, int]:
    """在按升序排好的存活距离 sorted_d 上，用单峰二分找最小前缀密度。

    phi(s) = (w_r + prefix_sum(s)) / s，s in {1,...,c}。
    Lemma unimodal: phi(s+1) <= phi(s)  <=>  d_{s+1} <= phi(s)，且一旦
    d_{s+1} >= phi(s) 成立则对更大 s 恒成立。故 phi 单峰（先降后升），
    最小点是首个满足 d_{s+1} >= phi(s) 的 s（1-based），或 s=c。
    返回 (best_density, best_size)。sorted_d 为空返回 (+inf, 0)。
    """
    c = len(sorted_d)
    if c == 0:
        return INF, 0
    # 前缀和（用于任意 s 的 phi 求值）；c 可能很大，但单轮只对被弹出的 r 计算。
    # 二分谓词：P(s) := (s==c) or (d[s] >= phi(s))，其中 d 为 0-based，d[s] 即 d_{s+1}。
    # phi(s) 随 s 先降后升，最小点为首个 P(s) 为真的 s。P 关于 s 单调（False...False,True...True）。
    prefix = [0.0] * (c + 1)
    for i in range(c):
        prefix[i + 1] = prefix[i] + sorted_d[i]

    def phi(s: int) -> float:
        return (w_r + prefix[s]) / s

    def pred(s: int) -> bool:
        # s in [1, c]; s==c 时最小点即末端
        if s == c:
            return True
        return sorted_d[s] >= phi(s)  # d_{s+1} >= phi(s)

    lo, hi = 1, c
    # 找首个 pred(s)=True。pred 单调不减（False 段后接 True 段）。
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    best_s = lo
    return phi(best_s), best_s


def greedy_density(
    inst: MinExplainInstance,
    return_rounds: bool = False,
) -> Solution:
    """密度贪心，懒惰堆实现。返回 Solution(E, assignment, cost)。

    若实例不可行（某 v 无候选），返回 cost=+inf 的部分解。
    return_rounds=True 时在 Solution 上附加 .rounds（每轮选中的 (r, size)）供诊断（不改签名语义）。
    """
    n, m = inst.n, inst.m
    # 每个 r 的 cover 按距离升序（预处理 Σ c_r log c_r = O(N logΔ)）
    sorted_cover: List[List[int]] = []
    for r in range(m):
        cr = sorted(inst.cover[r], key=lambda v: inst.dist[(v, r)])
        sorted_cover.append(cr)

    alive = [True] * n            # 违反是否仍待覆盖
    n_alive = n
    assignment: List[Optional[int]] = [None] * n
    E: List[int] = []
    E_set = set()
    rounds: List[Tuple[int, int]] = []

    def alive_sorted_dists(r: int) -> Tuple[List[int], List[float]]:
        """r 的存活违反（按距离升序）及其距离列表。"""
        vs = [v for v in sorted_cover[r] if alive[v]]
        ds = [inst.dist[(v, r)] for v in vs]
        return vs, ds

    # 初始化堆：每个有存活覆盖的 r 计算最优前缀密度
    heap: List[Tuple[float, int, int]] = []  # (density, version_seen_nalive_marker, r)
    # 用 dirty 标记 + 惰性重算；堆项带一个"计算时的 n_alive 快照"不足以判脏，改用显式 dirty 位。
    dirty = [False] * m
    best_density = [INF] * m
    best_size = [0] * m

    for r in range(m):
        if sorted_cover[r]:
            _, ds = alive_sorted_dists(r)
            dens, sz = _best_prefix(ds, inst.w[r])
            best_density[r] = dens
            best_size[r] = sz
            if sz > 0:
                heapq.heappush(heap, (dens, r))  # type: ignore[arg-type]

    while n_alive > 0 and heap:
        dens, r = heapq.heappop(heap)  # type: ignore[misc]
        if dirty[r]:
            # 惰性重算：存活集收缩后密度只增（Theorem T7c laziness 正确性），重算后重入堆
            dirty[r] = False
            _, ds = alive_sorted_dists(r)
            new_dens, new_sz = _best_prefix(ds, inst.w[r])
            best_density[r] = new_dens
            best_size[r] = new_sz
            if new_sz > 0:
                heapq.heappush(heap, (new_dens, r))
            continue
        if abs(dens - best_density[r]) > 1e-12:
            # 陈旧堆项（此前重算后旧项残留），丢弃
            continue
        # clean 最小项即全局最小密度对（Lemma prefix + laziness）
        vs, ds = alive_sorted_dists(r)
        s = best_size[r]
        if s == 0:
            continue
        chosen = vs[:s]  # 选中的前缀违反
        if r not in E_set:
            E.append(r)
            E_set.add(r)
        for v in chosen:
            if alive[v]:
                assignment[v] = r
                alive[v] = False
                n_alive -= 1
                # 该违反从所有含它的其他 r 的存活集移除 -> 置脏
                for r2 in inst.L[v]:
                    if r2 != r and not dirty[r2]:
                        dirty[r2] = True
        rounds.append((r, len(chosen)))
        # r 自身也可能还有存活覆盖（其余违反），置脏后可再次被选
        dirty[r] = True
        _, ds2 = alive_sorted_dists(r)
        nd, nsz = _best_prefix(ds2, inst.w[r])
        best_density[r] = nd
        best_size[r] = nsz
        dirty[r] = False
        if nsz > 0:
            heapq.heappush(heap, (nd, r))

    # 代价
    if n_alive > 0:
        cost = INF
    else:
        cost = sum(inst.w[r] for r in E_set)
        for v in range(n):
            r = assignment[v]
            cost += inst.dist[(v, r)] if r is not None else INF

    sol = Solution(E=E, assignment=assignment, cost=cost)
    if return_rounds:
        sol.rounds = rounds  # type: ignore[attr-defined]
    return sol
