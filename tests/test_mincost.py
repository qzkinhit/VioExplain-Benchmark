# coding=utf-8
"""对拍验收 mincost/ 六个算法文件（core/greedy/primal_dual/prune/exact）。

事实源：docs/theory_drafts/theorems_draft.tex（T2/T3/T7）。断言口径：
  - ILP（pulp/CBC，回退 brute_force）== 暴力枚举最优（同一 ground truth）；
  - 密度贪心 Cost <= H(Δ)·OPT（Theorem T2）；
  - primal-dual Cost <= f·OPT（Theorem T3）；
  - 支配剪枝 + 安全迫选前后最优值不变、可行性保持（Theorem T7(a)/T7(b)）。
随机小实例 n<=12，>=200 个可行实例，多参数网格 + 定向构造。纯 CPU，秒级。
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vioexplain.mincost import (
    build_instance,
    greedy_density,
    pd_onepass,
    prune,
    exact_ilp,
    brute_force,
)
from vioexplain.mincost.core import is_feasible, cheapest_assignment, cost_of

INF = math.inf


def _harmonic(k: int) -> float:
    return sum(1.0 / i for i in range(1, k + 1)) if k > 0 else 1.0


def _random_instance(rng, n_max=12, m_max=9, edge_p=0.45, force_feasible=True):
    n = int(rng.integers(1, n_max + 1))
    m = int(rng.integers(1, m_max + 1))
    theta = 1.0
    d = {}
    for v in range(n):
        if force_feasible:
            fr = int(rng.integers(0, m))
            d[(v, fr)] = float(rng.random())  # in [0,1) <= theta，保证每个 v 至少一条边
        for r in range(m):
            if rng.random() < edge_p:
                d[(v, r)] = float(rng.random())
    w = [float(rng.random() * 3.0) for _ in range(m)]
    return build_instance(range(n), range(m), w, d, theta)


def _feasible_instances(seed, count, **kw):
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < count and tries < count * 50:
        tries += 1
        inst = _random_instance(rng, **kw)
        if inst.feasible():
            out.append(inst)
    assert len(out) == count, f"only generated {len(out)}/{count} feasible instances"
    return out


# 生成一次共享实例池（>=200），供多个断言复用。
INSTANCES = _feasible_instances(seed=20240703, count=240)


def test_pool_size():
    assert len(INSTANCES) >= 200


def test_ilp_equals_bruteforce():
    """ILP（pulp/CBC 或回退暴力）与独立暴力枚举给出相同 OPT。"""
    n_bad = 0
    for inst in INSTANCES:
        bf = brute_force(inst)
        ilp = exact_ilp(inst)
        assert math.isfinite(bf.cost)
        if abs(ilp.cost - bf.cost) > 1e-5:
            n_bad += 1
    assert n_bad == 0, f"{n_bad} instances where ILP != brute-force OPT"


def test_greedy_harmonic_bound_and_feasibility():
    """密度贪心：可行，且 Cost <= H(Δ)·OPT（Theorem T2）。"""
    n_bad = 0
    n_infeas = 0
    max_ratio = 0.0
    for inst in INSTANCES:
        opt = brute_force(inst)
        g = greedy_density(inst)
        if not is_feasible(inst, g.E, g.assignment):
            n_infeas += 1
            continue
        H = _harmonic(inst.Delta)
        if g.cost > H * opt.cost + 1e-6:
            n_bad += 1
        if opt.cost > 1e-9:
            max_ratio = max(max_ratio, g.cost / opt.cost)
    assert n_infeas == 0, f"{n_infeas} greedy outputs infeasible"
    assert n_bad == 0, f"{n_bad} instances violate H(Delta)*OPT bound"
    # 经验近似比应远优于最坏界（诊断，非硬断言下限）
    assert max_ratio < 10.0


def test_pd_frequency_bound_and_feasibility():
    """primal-dual：可行，且 Cost <= f·OPT（Theorem T3）。"""
    n_bad = 0
    n_infeas = 0
    max_ratio = 0.0
    for inst in INSTANCES:
        opt = brute_force(inst)
        pd = pd_onepass(inst)
        if not is_feasible(inst, pd.E, pd.assignment):
            n_infeas += 1
            continue
        if pd.cost > inst.f * opt.cost + 1e-6:
            n_bad += 1
        if opt.cost > 1e-9:
            max_ratio = max(max_ratio, pd.cost / opt.cost)
    assert n_infeas == 0, f"{n_infeas} pd outputs infeasible"
    assert n_bad == 0, f"{n_bad} instances violate f*OPT bound"
    assert max_ratio < 20.0


def test_pd_order_invariant_bound():
    """primal-dual 在不同处理顺序下仍满足 f·OPT 界且可行。"""
    rng = np.random.default_rng(99)
    for inst in INSTANCES[:80]:
        opt = brute_force(inst)
        order = list(range(inst.n))
        rng.shuffle(order)
        pd = pd_onepass(inst, order=order)
        assert is_feasible(inst, pd.E, pd.assignment)
        assert pd.cost <= inst.f * opt.cost + 1e-6


@pytest.mark.parametrize("pointwise", [False, True])
def test_prune_preserves_optimum_and_feasibility(pointwise):
    """剪枝（聚合支配 或 逐点特例）+ 安全迫选前后最优值不变、可行性保持（T7）。"""
    n_bad = 0
    n_infeas = 0
    for inst in INSTANCES:
        opt = brute_force(inst)
        pr = prune(inst, pointwise=pointwise)
        # 缩减实例仍可行（每个违反仍有候选，迫选/支配都不整删 cover）
        if not pr.inst.feasible():
            n_infeas += 1
            continue
        red = brute_force(pr.inst)
        if abs(red.cost - opt.cost) > 1e-6:
            n_bad += 1
    assert n_infeas == 0, f"{n_infeas} pruned instances became infeasible"
    assert n_bad == 0, f"{n_bad} instances where pruning changed the optimum"


def test_prune_then_solve_matches_opt():
    """在剪枝后的实例上跑 exact/greedy/pd，最优/近似性质仍成立。"""
    for inst in INSTANCES[:120]:
        opt = brute_force(inst)
        pr = prune(inst)
        assert pr.inst.feasible()
        e = exact_ilp(pr.inst)
        assert abs(e.cost - opt.cost) < 1e-5
        g = greedy_density(pr.inst)
        H = _harmonic(pr.inst.Delta)
        assert g.cost <= H * opt.cost + 1e-6
        pd = pd_onepass(pr.inst)
        assert pd.cost <= pr.inst.f * opt.cost + 1e-6


def test_dominance_actually_triggers():
    """定向构造：r0 被 r1 聚合支配（eq:dom），必须被删且最优值不变。"""
    n, m, theta = 3, 2, 1.0
    d = {(0, 0): 0.5, (1, 0): 0.5, (0, 1): 0.2, (1, 1): 0.2, (2, 1): 0.3}
    w = [2.0, 0.1]
    inst = build_instance(range(n), range(m), w, d, theta)
    pr = prune(inst)
    assert 0 in pr.removed_dominated
    assert pr.inst.feasible()
    assert abs(brute_force(pr.inst).cost - brute_force(inst).cost) < 1e-9


def test_unique_cover_forced():
    """定向构造：v 只被单一 r 覆盖 => r 被迫选并计入 forced_open_cost 一次。"""
    n, m, theta = 3, 2, 1.0
    d = {(0, 0): 0.3, (1, 0): 0.3, (0, 1): 0.4, (2, 1): 0.2}  # v1 只在 r0，v2 只在 r1
    w = [1.0, 1.0]
    inst = build_instance(range(n), range(m), w, d, theta)
    pr = prune(inst)
    assert 0 in pr.forced and 1 in pr.forced
    assert abs(pr.forced_open_cost - 2.0) < 1e-9
    assert pr.inst.feasible()
    assert abs(brute_force(pr.inst).cost - brute_force(inst).cost) < 1e-9


def test_mutual_domination_keeps_one():
    """互相支配（等价 rep）时只删索引大者，实例不塌成不可行。"""
    n, m, theta = 2, 2, 1.0
    d = {(0, 0): 0.5, (1, 0): 0.5, (0, 1): 0.5, (1, 1): 0.5}
    w = [1.0, 1.0]
    inst = build_instance(range(n), range(m), w, d, theta)
    pr = prune(inst)
    assert len(pr.kept) >= 1
    assert pr.inst.feasible()
    assert abs(brute_force(pr.inst).cost - brute_force(inst).cost) < 1e-9


def test_cheapest_assignment_consistency():
    """cheapest_assignment 给出的 assignment 与 cost 自洽（cost_of 复算一致）。"""
    for inst in INSTANCES[:100]:
        opt = brute_force(inst)
        assignment, cost = cheapest_assignment(inst, opt.E)
        if math.isfinite(cost):
            assert abs(cost_of(inst, opt.E, assignment) - cost) < 1e-9


def test_infeasible_instance_reports_inf():
    """某违反无任何候选 => 各算法返回 +inf / 不可行。"""
    n, m, theta = 2, 1, 1.0
    d = {(0, 0): 0.5}  # v1 无边
    w = [1.0]
    inst = build_instance(range(n), range(m), w, d, theta)
    assert not inst.feasible()
    assert brute_force(inst).cost == INF
    assert exact_ilp(inst).cost == INF
    assert greedy_density(inst).cost == INF
    assert pd_onepass(inst).cost == INF
