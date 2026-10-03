# MinExplain objective and implementation assumptions

This module is a later formalization of violation explanation. It is distinct from the supplied AEC-Prototype. It selects event representations and assigns individual violations to selected representations.

For a fixed feasible graph, let `w[h]` be a nonnegative opening cost and `d[v,h]` a nonnegative assignment cost. The objective is

```text
Cost(H, assignment) = sum(w[h] for h in H)
                   + sum(d[v, assignment[v]] for v in V)
```

Each violation must be assigned along a feasible edge. An observation without a compatible candidate is kept outside this feasible subproblem as unexplained evidence. It cannot be silently discarded when reporting the explanation task.

Let `n` be the number of violations, `m` the number of candidates, `N` the number of edges, `Delta` the largest candidate support, and `f` the largest number of candidates incident to a violation.

## Density greedy

`greedy_density` considers the cost of opening a candidate and assigning a nonempty subset of its remaining violations. For any subset size, the cheapest subset is the prefix of violations sorted by assignment cost. The prefix density has a unimodal minimum, which `_best_prefix` locates by binary search after constructing prefix sums.

The harmonic guarantee is `Cost <= H(Delta) * OPT` under the fixed graph and nonnegative-cost assumptions. To see the charging argument, consider the violations assigned to one optimum candidate. When `r` of them remain, that candidate supplies an available subset with density at most its full optimum contribution divided by `r`. Charging the greedy selections in removal order gives a harmonic sum. If the implementation selects the same event more than once, the final objective charges its opening only once, so its final cost is no larger than the accumulated selection charges.

The current implementation rescans a candidate support when recomputing a dirty key. Its conservative bound, after building the graph, is

```text
O(m + n + N log(Delta + 1)
  + (N + n) * (Delta + log(m + 1)))
```

Memory is `O(N + n + m)`. A claim of near-linear heap runtime would omit these support rescans. The generic `build_instance` additionally visits all `n*m` potential pairs, even when its input is a sparse distance dictionary.

## One-pass primal-dual solver

`pd_onepass` decreases the residual opening budgets through feasible dual charges. Selected openings are fully paid by those charges. A violation contributes to at most `f` incident openings, giving `Cost <= f * OPT` under the same assumptions. Running both solvers and keeping the lower-cost feasible result therefore gives the smaller of the two stated bounds. Empirical truth recovery is a separate question from either objective guarantee.

## Safe simplifications

A candidate can be removed by support inclusion only when the increased assignment costs are also covered by the opening-cost saving. Containing another candidate's support alone is insufficient.

If a violation has a unique candidate, that candidate is required in every feasible solution. This fact does not justify removing every violation that it could cover. Other violations must still choose their cheapest valid assignments. `prune.py` keeps this distinction.

The exact small-instance reference and solver tests verify numerical behavior on their stated cases; they do not establish semantic correctness of the event dictionary, industrial fault identities, learned costs, or the proofs themselves. Executed source snapshots remain authoritative for historical reported results.
