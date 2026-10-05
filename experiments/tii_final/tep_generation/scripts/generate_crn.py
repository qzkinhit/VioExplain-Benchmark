#!/usr/bin/env python3
"""Generate common-random-numbers fault data: for every seed, the fault-free run and every subset of every requested
disturbance set, all with the same seed.  Subsets shared by several requests (the fault-free run, single faults) are
simulated once.

Output directory:
  set_<name>.npy     float64 array (n_seeds, n_samples, 52); <name> is "normal" or the IDVs joined by "+"
  meta.json          protocol, variant, seeds, generator states, shutdown step of every run

Examples
  # Rieth training protocol, all 190 pairs, 100 seeds, split build (exact CRN for all IDVs)
  python3 scripts/generate_crn.py --sets all-pairs --seeds 0-99 --protocol train --out /path/out
  # testing protocol, three chosen compound faults
  python3 scripts/generate_crn.py --sets 1+4,2+6,1+4+11 --seeds 0-49 --protocol test --out /path/out
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tepsim import simulate_many  # noqa: E402

PROTOCOLS = {"train": (500, 20), "test": (960, 160)}


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_sets(text):
    if text == "all-pairs":
        return [frozenset(p) for p in combinations(range(1, 21), 2)]
    if text == "all-singles":
        return [frozenset([k]) for k in range(1, 21)]
    return [frozenset(int(i) for i in s.split("+")) for s in text.split(",")]


def name(s):
    return "normal" if not s else "+".join(str(i) for i in sorted(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help='"all-pairs", "all-singles" or e.g. "1+4,2+6"')
    ap.add_argument("--seeds", required=True, help='e.g. "0-99" or "0,5,7"')
    ap.add_argument("--protocol", choices=PROTOCOLS, default="train")
    ap.add_argument("--n-samples", type=int, help="override the protocol length")
    ap.add_argument("--onset", type=int, help="override the protocol onset (samples)")
    ap.add_argument("--variant", choices=("original", "split"), default="split")
    ap.add_argument("--processes", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n, onset = PROTOCOLS[args.protocol]
    n = args.n_samples or n
    onset = args.onset if args.onset is not None else onset
    seeds = parse_seeds(args.seeds)
    needed = set()
    for s in parse_sets(args.sets):
        for r in range(len(s) + 1):
            needed.update(frozenset(c) for c in combinations(sorted(s), r))
    needed = sorted(needed, key=lambda s: (len(s), sorted(s)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(protocol=args.protocol, n_samples=n, onset=onset, variant=args.variant, seeds=seeds,
                sets=[name(s) for s in needed], runs={})
    t0 = time.time()
    # several sets per pool call so that every call has at least 4 jobs per worker process
    per_batch = max(1, -(-4 * args.processes // len(seeds)))
    for b in range(0, len(needed), per_batch):
        batch = needed[b:b + per_batch]
        jobs = [dict(n_samples=n, seed=sd, variant=args.variant, faults={k: onset for k in s})
                for s in batch for sd in seeds]
        res = simulate_many(jobs, processes=args.processes)
        for i, s in enumerate(batch):
            part = res[i * len(seeds):(i + 1) * len(seeds)]
            np.save(out / f"set_{name(s)}.npy", np.stack([r.X for r in part]))
            meta["runs"][name(s)] = [dict(seed=r.seed, g=r.g, g_final=r.g_final, shutdown_step=r.shutdown_step,
                                          shutdown_sample=r.shutdown_sample) for r in part]
            print(f"{name(s):>8s}: {len(part)} runs, shutdowns {sum(r.shutdown for r in part)}, "
                  f"elapsed {time.time() - t0:.0f} s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
