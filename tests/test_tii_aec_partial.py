"""对未知违反的回归验证：未知证据不能抹去已匹配的事件。"""
import math

import pytest

from vioexplain.publicexp.aec import explain_event
from vioexplain.publicexp.features import Violation
from vioexplain.publicexp.knowledge import Representation


def violation(dim):
    return Violation(dim, "domain", 1, 1.0, (1.0, 1.0), 3)


def test_unmatched_violation_preserves_existing_explanation():
    # 两个候选均覆盖已知证据，无唯一候选可触发 forced。
    # 旧实现加入未知 d3 后会丢掉这两个已知维，并返回无穷代价。
    knowledge = [Representation({"d1", "d2"}, 1.0, 2),
                 Representation({"d1", "d2", "d4"}, 2.0, 1)]
    known = [violation("d1"), violation("d2")]
    base = explain_event(known, knowledge)
    partial = explain_event(known + [violation("d3")], knowledge)
    assert base.selected == partial.selected == [0]
    assert math.isfinite(partial.cost)
    assert partial.cost == pytest.approx(base.cost)
    assert partial.covered == 2 and partial.n_viol == 3
    assert partial.pred_dims == {"d1", "d2", "d3"}
    assert partial.unmatched_dims == {"d3"}


@pytest.mark.parametrize("knowledge", [[], [Representation({"d9"}, 1.0, 1)]])
def test_unknown_evidence_is_not_counted_as_explained(knowledge):
    result = explain_event([violation("d1"), violation("d2")], knowledge)
    assert result.covered == 0
    assert result.selected == []
    assert result.cost == 0.0
    assert result.unmatched_dims == result.pred_dims == {"d1", "d2"}


def test_empty_evidence_has_no_unknown_dimensions():
    result = explain_event([], [Representation({"d1"}, 1.0, 1)])
    assert result.cost == 0.0 and result.covered == 0
    assert result.unmatched_dims == set()
