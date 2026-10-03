import math
import pytest
from vioexplain import Representation, Violation, explain
from vioexplain.cli import smoke


def v(dim):
    return Violation(dim, "domain", 1, 1.0, (1.0, 1.0), 3)


def test_unknown_evidence_retained_without_diagnosis():
    out = explain([v("d1"), v("d2")], [Representation({"d1"}, 1, 1)])
    assert out.explained_dims == ("d1",)
    assert out.unmatched_dims == ("d2",)
    assert out.assignment == (0, None)
    assert out.status == "partial"
    assert out.cost == pytest.approx(1.012)


def test_same_dimension_can_contain_known_and_unknown_evidence():
    out = explain([v("d1"), v("d1")], [Representation({"d1"}, 1, 1)], distances={(0, 0): 0.2})
    assert out.explained_dims == out.unmatched_dims == ("d1",)
    assert out.unmatched_violation_indices == (1,)
    assert out.covered == 1


def test_empty_and_unknown_are_distinct():
    assert explain([], []).status == "empty"
    assert explain([v("d1")], []).status == "unknown"
    assert explain([v("d1")], []).cost == 0


@pytest.mark.parametrize("distance", [-1, math.nan, -math.inf])
def test_invalid_distances_are_rejected(distance):
    with pytest.raises(ValueError):
        explain([v("d1")], [Representation({"d1"}, 1, 1)], distances={(0, 0): distance})


def test_invalid_opening_cost_is_rejected():
    with pytest.raises(ValueError):
        explain([v("d1")], [Representation({"d1"}, -1, 1)])


def test_smoke_is_explicitly_synthetic():
    assert smoke()["purpose"] == "synthetic_smoke_only"
