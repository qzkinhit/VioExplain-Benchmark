"""MinExplain contract with unmatched evidence recorded separately.

This is the later opening-plus-assignment model, not AEC-Prototype.
The signature adapter is inherited from the public-data AEC reproduction.
A caller can supply nonnegative distances for a different matching model;
the explanation objective and the candidate identities remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping, Sequence

from .mincost import build_instance, greedy_density, prune
from .publicexp.aec import _match_distance
from .publicexp.features import Violation
from .publicexp.knowledge import Representation


@dataclass(frozen=True)
class ExplainResult:
    selected: tuple[int, ...]
    assignment: tuple[int | None, ...]
    explained_dims: tuple[str, ...]
    unmatched_dims: tuple[str, ...]
    unmatched_violation_indices: tuple[int, ...]
    cost: float
    covered: int
    n_violations: int
    status: str

    def to_dict(self) -> dict:
        """Return JSON-serializable fields, without inf or NaN."""
        return asdict(self)


def explain(
    violations: Sequence[Violation],
    knowledge: Sequence[Representation],
    *,
    theta: float = 0.6,
    distances: Mapping[tuple[int, int], float] | Callable[[int, int], float] | None = None,
) -> ExplainResult:
    """Explain the matchable subproblem, retaining all unmatched evidence.

    Candidate and violation identities are their zero-based input positions.
    Missing mapping entries and +inf mean incompatible. Distances and opening
    costs must be nonnegative; finite theta is fixed before evaluation.
    An empty knowledge base returns status='unknown', not a successful diagnosis.
    """
    if not math.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")
    for rep in knowledge:
        if not math.isfinite(rep.w) or rep.w < 0:
            raise ValueError("representation costs must be finite and nonnegative")
    for v in violations:
        if not isinstance(v.dim, str) or not v.dim or not math.isfinite(v.mag) or v.mag < 0:
            raise ValueError("each violation needs a named dimension and finite nonnegative magnitude")
    if isinstance(distances, Mapping):
        for key in distances:
            if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(i, int) for i in key):
                raise ValueError("distance keys must be (violation_index, representation_index)")
            if not (0 <= key[0] < len(violations) and 0 <= key[1] < len(knowledge)):
                raise ValueError("distance key is outside the input range")
    edges = {}
    for i, violation in enumerate(violations):
        for j, rep in enumerate(knowledge):
            if distances is None:
                value = _match_distance(violation, rep, theta)
            elif callable(distances):
                value = float(distances(i, j))
            else:
                value = float(distances.get((i, j), math.inf))
            if math.isnan(value) or value < 0:
                raise ValueError("distances must be nonnegative numbers or +inf")
            if value <= theta:
                edges[i, j] = value
    matched = sorted({i for i, _ in edges})
    matched_set = set(matched)
    unknown = tuple(i for i in range(len(violations)) if i not in matched_set)
    assignments: list[int | None] = [None] * len(violations)
    selected: tuple[int, ...] = ()
    cost = 0.0
    if matched:
        remap = {old: new for new, old in enumerate(matched)}
        inst = build_instance(matched, knowledge, [r.w for r in knowledge],
                              {(remap[i], j): d for (i, j), d in edges.items()}, theta)
        reduced = prune(inst)
        solution = greedy_density(reduced.inst)
        selected = tuple(sorted(set(reduced.kept[j] for j in solution.E) | reduced.forced))
        for local_i, local_j in enumerate(solution.assignment):
            assignments[matched[local_i]] = reduced.kept[local_j] if local_j is not None else None
        cost = float(solution.cost)
        if not math.isfinite(cost):
            raise RuntimeError("the matchable explanation subproblem returned an infeasible solution")
    status = "empty" if not violations else ("unknown" if not matched else ("partial" if unknown else "explained"))
    return ExplainResult(
        selected=selected, assignment=tuple(assignments),
        explained_dims=tuple(sorted({violations[i].dim for i in matched})),
        unmatched_dims=tuple(sorted({violations[i].dim for i in unknown})),
        unmatched_violation_indices=unknown, cost=cost, covered=len(matched),
        n_violations=len(violations), status=status,
    )
