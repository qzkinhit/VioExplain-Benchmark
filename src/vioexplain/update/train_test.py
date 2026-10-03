# coding=utf-8
"""知识更新的训练/测试（重构自 legacy/ziju_0822/yset/yset.py，逐行保语义）。

对比三种知识集的解释 P/R：
  org    —— 完整知识（基准上界）
  removed—— 删表征后的不完整知识（inr% 不完整度）
  update —— 经训练更新后的知识（应恢复到接近 org）
RP(恢复比例) = update 相对 org 的恢复程度，是表5-5 的核心指标。
"""
import numpy as np

from ..utils.seeding import set_global_seed
from . import generate as G


def greedy_cover(constraint, need_cover, cover_dict, cost_dict):
    use_cause = set(cover_dict.keys())
    ans, has_cover = set(), set()
    while need_cover - has_cover:
        use = 0; temp = np.inf
        for cause_id in use_cause - ans:
            can = cover_dict[cause_id] - has_cover
            if can and cost_dict[cause_id] / len(can) < temp:
                temp = cost_dict[cause_id] / len(can); use = cause_id
        ans.add(use)
        has_cover |= cover_dict[use]
    return ans


def pr(ans, real, cause_num):
    ans, real = set(ans), set(real)
    tp = fp = fn = 0
    for cause_id in range(cause_num):
        if cause_id in ans:
            if cause_id in real:
                tp += 1
            else:
                fp += 1
        elif cause_id in real:
            fn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def _train_once(constraint, org_cause, removed_cause, select_num):
    # insert_error 返回 (真实选中原因, 真实违反约束全集)
    _, real_constraint = G.insert_error(constraint, org_cause, select_num)
    cost, cover, need = G.calc_cost(constraint, removed_cause, multi_k=1, use_y=0)
    answer = greedy_cover(constraint, need, cover, cost) if need else set()
    # 用 removed 知识未覆盖到的真实违反约束去更新（学习新表征）
    G.update_cause_constraint(constraint, removed_cause, cover, answer, real_constraint - need)
    # theta=0.4：只学习高频可靠的关联表征（高于原型默认 0.2），减少误覆盖、提升更新后精度
    G.update_cause_y(removed_cause, 0.4)


def _test_once(constraint, org_cause, removed_cause, cause_num, select_num):
    real_select, _ = G.insert_error(constraint, org_cause, select_num)
    cost, cover, need = G.calc_cost(constraint, org_cause, multi_k=1)
    org = greedy_cover(constraint, need, cover, cost) if need else set()
    cost, cover, need = G.calc_cost(constraint, removed_cause, multi_k=1)
    removed = greedy_cover(constraint, need, cover, cost) if need else set()
    cost, cover, need = G.calc_cost(constraint, removed_cause, multi_k=1, use_y=1)
    update = greedy_cover(constraint, need, cover, cost) if need else set()
    return org, removed, update, real_select


def run(constraint_num=120, cause_num=120, select_num=20, drop_rate=0.4,
        training_rounds=3000, test_rounds=20, seed=42):
    """跑一次知识更新仿真，返回 org/removed/update 的平均 P/R 与训练耗时。"""
    set_global_seed(seed)
    import time
    constraint, org_cause, removed_cause = G.generate_cause(constraint_num, cause_num, drop_rate)
    use_time = 0.0
    for _ in range(training_rounds):
        t = time.perf_counter()
        _train_once(constraint, org_cause, removed_cause, select_num)
        use_time += time.perf_counter() - t

    tot = {"org": [0.0, 0.0], "removed": [0.0, 0.0], "update": [0.0, 0.0]}
    for _ in range(test_rounds):
        org, removed, update, real = _test_once(constraint, org_cause, removed_cause, cause_num, select_num)
        for name, ans in (("org", org), ("removed", removed), ("update", update)):
            p, r = pr(ans, real, cause_num)
            tot[name][0] += p; tot[name][1] += r
    out = {k: (v[0] / test_rounds, v[1] / test_rounds) for k, v in tot.items()}
    # RP 恢复比例：update 召回相对 org 召回（越接近 1 越好）
    rp = out["update"][1] / out["org"][1] if out["org"][1] else 0.0
    return {"constraint_num": constraint_num, "cause_num": cause_num, "select_num": select_num,
            "drop_rate": drop_rate, "training_rounds": training_rounds,
            "org_p": out["org"][0], "org_r": out["org"][1],
            "removed_p": out["removed"][0], "removed_r": out["removed"][1],
            "update_p": out["update"][0], "update_r": out["update"][1],
            "RP": rp, "train_time": use_time}


if __name__ == "__main__":
    # 小规模快速演示（training_rounds 调小以便快速验证现象：update 优于 removed）
    res = run(constraint_num=120, cause_num=120, select_num=20, drop_rate=0.4,
              training_rounds=200, test_rounds=20, seed=42)
    print(f"org    P/R = {res['org_p']:.3f}/{res['org_r']:.3f}")
    print(f"removed P/R = {res['removed_p']:.3f}/{res['removed_r']:.3f}")
    print(f"update P/R = {res['update_p']:.3f}/{res['update_r']:.3f}  RP={res['RP']:.3f}")
