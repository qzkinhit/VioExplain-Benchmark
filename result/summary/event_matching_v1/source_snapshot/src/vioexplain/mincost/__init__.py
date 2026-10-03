# coding=utf-8
"""MinExplain 最小代价覆盖（带指派代价）的升级算法与精确求解。

事实源：docs/theory_drafts/theorems_draft.tex（T1--T7）+ docs/IDEA_THEORY_BLUEPRINT.md §3--§5。
本包为纯函数方法层，零 IO（不读文件、不 print、不落盘），供 experiments/ 薄壳脚本编排。

统一问题实例与接口（见 core.py）：
  build_instance(violations, candidates, w, d, theta) -> MinExplainInstance
  每个算法 solve 接口输入 (violations, candidates, w, d, theta)，输出 Solution(E, assignment, cost)。

算法：
  greedy_density   : T2 密度贪心，懒惰堆 + 按距离排序前缀 + 单峰二分（Lemma unimodal），T7(c) 复杂度。
  pd_onepass       : T3 一遍 primal-dual（local-ratio），O(N)，代价 <= f·OPT。
  prune            : T7(a) 聚合支配剪枝 + T7(b) 安全迫选（不整删 cover(r)）。
  exact_ilp        : 小实例 ILP 精确求解（pulp/CBC 后端，OR-Tools 可选），返回真 OPT。
  brute_force      : n<=~12 的暴力枚举精确解，供对拍。
"""
from .core import (
    MinExplainInstance,
    Solution,
    build_instance,
    cost_of,
)
from .greedy import greedy_density
from .primal_dual import pd_onepass
from .prune import prune, PruneResult
from .exact import exact_ilp, brute_force

__all__ = [
    "MinExplainInstance",
    "Solution",
    "build_instance",
    "cost_of",
    "greedy_density",
    "pd_onepass",
    "prune",
    "PruneResult",
    "exact_ilp",
    "brute_force",
]
