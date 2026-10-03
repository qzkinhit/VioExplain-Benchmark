# Explanation contract

`explain(violations, knowledge, theta=0.6, distances=None)` accepts a sequence of `Violation` objects and a sequence of `Representation` objects. Input positions are stable zero-based identities. The default matcher is the inherited dimension-signature adapter. A distance mapping or callable can supply alternative evidence, provided costs are nonnegative and the matching threshold is fixed before evaluation.

| Field | Meaning |
|---|---|
| `selected` | Selected representation indices in the original knowledge sequence |
| `assignment` | One representation index per original violation; `None` means unmatched |
| `explained_dims` | Dimensions with at least one assigned violation |
| `unmatched_dims` | Dimensions with at least one violation with no feasible candidate |
| `unmatched_violation_indices` | Exact original indices of unmatched evidence |
| `cost` | Opening plus assignment cost of the matchable subproblem |
| `covered`, `n_violations` | Assigned evidence count and total evidence count |
| `status` | `empty`, `unknown`, `partial`, or `explained` |

A dimension may appear in both `explained_dims` and `unmatched_dims` when different violations on that dimension have different matching status. The explicit indices preserve this information. A cost of zero with `status="unknown"` describes an empty matchable subproblem, not a correct zero-cost diagnosis. The API does not infer physical causality or semantic event labels from channel names.

When no distance map is supplied, a representation can match only dimensions in its signature. Covering every matchable violation consequently reproduces the matchable violation dimensions. Event-set and cost improvements must be evaluated on selected representations; a dimension-level score alone cannot establish an improvement from covering.

## Relationship to AEC-Prototype

This public API solves the later MinExplain opening-plus-assignment formulation. The supplied original AEC prototype is separately exposed as `vioexplain.setcover.set_covering.MyCover`; it uses event/interval costs and its own `Select` procedure. Replacing the original objective or solver with MinExplain is a modeling change, not an equivalent refactoring. Its mathematical bounds do not automatically apply to AEC-Prototype.

For a comparison, supply each method its declared representation format, preserve the same observable event evidence and knowledge split, compare selected event identities under an explicit identity map, and keep each method's internal objective value separately named. A lower value under one objective cannot be claimed as a lower value under another objective.

