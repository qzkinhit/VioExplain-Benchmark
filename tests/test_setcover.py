# coding=utf-8
"""集合覆盖解集代价回归测试：重构版与原型逐位一致。"""
from vioexplain.setcover import demo_data
from vioexplain.setcover.set_covering import MyCover

# 原型 legacy/setcovering/SetCovering.py 6 组演示数据的解集代价（golden）
GOLDEN_COSTS = [
    1.182960911637382,
    1.3787955443600604,
    1.5358904433582299,
    0.9722610086282256,
    0.5390358660050614,
    0.5970033670033674,
]


def test_demo_costs_match_legacy():
    rules = demo_data.demo_rules()
    for i, F_f in enumerate(demo_data.demo_explanation_sets()):
        cost, H = MyCover(F_f, demo_data.demo_constraints(), demo_data.demo_constraints(), rules)
        assert abs(cost - GOLDEN_COSTS[i]) < 1e-9, f"F_f{i+1}: {cost} != {GOLDEN_COSTS[i]}"
        assert len(H) > 0
