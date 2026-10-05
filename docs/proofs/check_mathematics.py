"""Finite checks for the proof document; these do not replace its proofs."""
from fractions import Fraction as Q
from itertools import combinations, product, permutations
import math
import random


def subsets(n):
    return [frozenset(c) for k in range(n + 1) for c in combinations(range(n), k)]


def check_profile():
    # Each coordinate describes one event incidence at a single unit.
    choices = [(None, False)] + [(g, exact) for g in (-2, -1, 0, 1, 2)
                                for exact in (False, True)]
    checked = 0
    for row in product(choices, repeat=3):
        for h in subsets(3):
            gains = [row[e][0] for e in h if row[e][0] is not None]
            owners = [row[e][0] for e in h if row[e][1]]
            m = max(gains, default=-math.inf)
            profile = m if owners else max(0, m)
            surrogate = max(0, m) - sum(max(0, -g) for g in owners)
            equality = (not owners or
                        (m >= 0 and all(g >= 0 for g in owners)) or
                        (m < 0 and len(owners) == 1 and owners[0] == m))
            assert surrogate <= profile
            assert (surrogate == profile) == equality
            checked += 1
    return checked


def check_binary():
    all_sets = subsets(3)
    checked = 0
    for supports in product(all_sets, repeat=3):
        for manifestations in all_sets:
            weights = (1, 2, 3)
            level = sum(weights) + 1
            rows = []
            for h in all_sets:
                union = frozenset().union(*(supports[e] for e in h))
                cost = sum(weights[e] for e in h)
                score = (level * len(manifestations & union)
                         - level * sum(len(supports[e] - manifestations) for e in h)
                         - cost)
                rows.append((h, union == manifestations, cost, score))
            covers = [r for r in rows if r[1]]
            if not covers:
                continue
            best_cost = min(r[2] for r in covers)
            best_score = max(r[3] for r in rows)
            expected = {r[0] for r in covers if r[2] == best_cost}
            actual = {r[0] for r in rows if r[3] == best_score}
            assert actual == expected
            checked += 1
    return checked


def check_facility_and_greedy():
    rng = random.Random(20261005)
    checked = 0
    for _ in range(200):
        n, m = 4, 3
        gains = [[rng.choice((None, 0, 1, 2, 3)) for _ in range(m)] for _ in range(n)]
        opening = [Q(rng.randrange(4)) for _ in range(m)]
        G = [Q(max((g for g in row if g is not None), default=0)) for row in gains]
        c = {(u, e): G[u] - gains[u][e] for u in range(n) for e in range(m)
             if gains[u][e] is not None}
        costs = []
        for h in subsets(m):
            cov = sum(max([0] + [gains[u][e] for e in h if (u, e) in c]) for u in range(n))
            F = cov - sum(opening[e] for e in h)
            C = sum(min([G[u]] + [c[u, e] for e in h if (u, e) in c]) for u in range(n))
            C += sum(opening[e] for e in h)
            assert sum(G) - F == C
            costs.append(C)
        uncovered = set(range(n))
        greedy_cost = Q(0)
        while uncovered:
            options = [(G[u], Q(1), {u}) for u in sorted(uncovered)]
            for e in range(m):
                ordered = sorted((u for u in uncovered if (u, e) in c), key=lambda u: c[u, e])
                for k in range(1, len(ordered) + 1):
                    star = set(ordered[:k])
                    options.append((opening[e] + sum(c[u, e] for u in star), Q(k), star))
            cost, size, star = min(options, key=lambda r: r[0] / r[1])
            greedy_cost += cost
            uncovered -= star
        delta = max([1] + [sum((u, e) in c for u in range(n)) for e in range(m)])
        harmonic = sum(Q(1, k) for k in range(1, delta + 1))
        assert greedy_cost <= harmonic * min(costs)
        checked += 1
    return checked


def quantile(values, alpha):
    n = len(values)
    q = (1 - alpha) * (n + 1)
    index = -(-q.numerator // q.denominator)
    return math.inf if index == n + 1 else sorted(values)[index - 1]


def check_ranks():
    checked = 0
    # Uniform random permutations give exchangeable joint score laws, including ties.
    for population in ((0, 1, 2, 3), (0, 0, 1, 2), (0, 0, 0, 0), (0,), (0, 1)):
        orders = list(permutations(population))
        for alpha in (Q(1, 20), Q(1, 4), Q(1, 2), Q(3, 4)):
            exceed = sum(p[-1] > quantile(p[:-1], alpha) for p in orders)
            probability = Q(exceed, len(orders))
            assert probability <= alpha
            if len(set(population)) == len(population):
                q = (1 - alpha) * len(population)
                index = -(-q.numerator // q.denominator)
                assert probability == 1 - Q(index, len(population))
            checked += 1
    assert quantile([], Q(1, 20)) == math.inf
    assert quantile(list(range(9)), Q(1, 20)) == math.inf
    return checked


def check_sharper_and_functional():
    omega, threshold, true_reference, false_reference = Q(1), Q(0), Q(3, 2), Q(-3, 2)
    d, a = true_reference - false_reference, true_reference - threshold
    assert d > 2 * omega and a > omega and not min(d, a) > 2 * omega
    for dt, df in product((Q(-1), Q(0), Q(1)), repeat=2):
        assert true_reference + dt > false_reference + df
        assert true_reference + dt > threshold
    # Independent unit-variance x1,x2,e and target x1+x2+e.
    r2 = Q(2, 3)
    general_squared = Q(1) / (3 * (1 - r2))
    old_squared = r2 / (1 - r2)
    assert general_squared == 1 and old_squared == 2
    return 10


if __name__ == '__main__':
    for name, check in (
        ('profile and equality', check_profile),
        ('binary consistent covering', check_binary),
        ('facility identity and greedy bound', check_facility_and_greedy),
        ('exchangeable ranks', check_ranks),
        ('sharper margins and functional changes', check_sharper_and_functional),
    ):
        print(f'PASS {name}: {check()} finite cases')
