"""ctypes interface to the original closed-loop Tennessee Eastman simulator.

The dynamics are the Fortran code of Downs and Vogel (teprob.f) with the decentralised control of Russell, Chiang
and Braatz (temain_mod.f), the code behind the classic d00..d21 files and, judging by its output format and its
pre-onset twins, behind the Rieth et al. (2017) data.  build.py compiles that code without edits together with
fortran/tepsim_driver.f, a callable copy of the main program of temain_mod.f.

One call of TEPSimulator.run integrates 180*n_samples one-second Euler steps and returns the 52 variables
x = [XMEAS(1..41), XMV(1..11)] every 180 s, the same variables, order and sampling as the classic and Rieth data.

Time convention.  Sample k (k = 0, 1, ...) is written at integration step 180*(k+1), before that step is
integrated, as SUBROUTINE OUTPUT does in temain_mod.f.  A disturbance with onset s (in samples) is on for the
steps i >= 180*s.  Its first possible effect is therefore on sample s, and samples 0..s-1 are identical to the
fault-free run with the same seed.  Rieth training data use s = 20 (1 h) with 500 samples, Rieth testing data and
the classic d??_te.dat files use s = 160 (8 h) with 960 samples.

Seeds.  The only source of randomness is TESUB7, G <- mod(G*9228907, 2^32) in double precision, with G in
COMMON/RANDSD/.  TEINIT sets G = 4651207995 and the temain_mod.f instructions ask the user to edit that value before
each run; Braatz' list of the values used for d00..d21 is kept in the comments of teprob.f.  run(g=...) sets G
directly; run(seed=n) uses G = g_from_seed(n).

The product G*9228907 needs up to 56 bits, so for G > 2^53/9228907 it is rounded to a multiple of 2, 4 or 8.  Once
G is a multiple of 8 every later product is exact, and G = 2^v * odd stays on a cycle of 2^(30-v) states.  A 48 h
run draws about 4.6e7 numbers (264 per 1 s step), a 25 h run about 2.4e7, so states that settle at v >= 5
(cycle <= 3.4e7) repeat their noise inside a 48 h run.  g_from_seed only returns seeds that settle at v = 3, the
longest cycle (2^27 = 1.34e8 draws, about 141 simulated hours).

Variants.  "original" is the code as published.  "split" gives each of the 12 random walks of TEFUNC its own
generator state, so the measurement-noise stream is consumed identically by every run that does not shut down
(in "original", IDV 17, 18 and 20 skip one shared draw per pulse and shift all later noise).

Every run starts from zeroed COMMON blocks, the state of a freshly loaded program, so a run in a long-lived process or
a forked worker is bit-identical to the same run in a fresh process.  The Fortran state is global: use one TEPSimulator per
process; simulate_many runs jobs in worker processes.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIBDIR = ROOT / "lib"
STEPS_PER_SAMPLE = 180            # 1 s Euler steps per 3 min sample
N_VARS = 52
N_IDV = 20
DEFAULT_G = 4651207995.0          # value assigned to G in TEINIT of teprob.f
NEVER = 2 ** 31 - 1
VARIANTS = ("original", "split")
_MASK64 = (1 << 64) - 1

FaultSpec = Union[int, Tuple[int, Optional[int]]]


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


TWO32 = 4294967296.0
MULT = 9228907.0


def tesub7_next(g: float) -> float:
    """One step of TESUB7 (bit-identical to the compiled Fortran: IEEE double product, exact fmod)."""
    return math.fmod(g * MULT, TWO32)


def settle(g: float, max_steps: int = 1_000_000) -> Tuple[int, int, float]:
    """Iterate the state until it is a multiple of 8.  Returns (steps, v, state) with v the 2-adic valuation of the
    settled state (v = 99 for the absorbing state 0); the state then lies on a cycle of 2**(30 - v) states."""
    for k in range(max_steps + 1):
        if g == 0.0:
            return k, 99, g
        if math.fmod(g, 8.0) == 0.0:
            v, m = 0, int(g)
            while m % 2 == 0:
                m //= 2
                v += 1
            return k, v, g
        g = tesub7_next(g)
    raise RuntimeError("generator state did not settle")


def cycle_length(g: float) -> int:
    k, v, _ = settle(g)
    return 1 if v >= 32 else 2 ** (30 - v)


def g_from_seed(seed: int, stream: int = 0) -> float:
    """Initial generator state for integer seed `seed`; stream 0 is G, streams 1..12 are the walk states GW(1..12)
    of the split variant.  Candidates are odd 32-bit integers from splitmix64(seed, stream, attempt); the first one
    whose state settles at valuation 3 (cycle of 2^27 draws, see the module docstring) is returned."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    for attempt in range(64):
        v = _splitmix64((((int(seed) & ((1 << 48) - 1)) << 16) | (int(stream) << 8) | attempt))
        g = float((v >> 32) | 1)
        if settle(g)[1] == 3:
            return g
    raise RuntimeError("no full-cycle seed found")


@dataclass
class SimResult:
    X: np.ndarray                  # (n_samples, 52): XMEAS(1..41), XMV(1..11)
    g: float                       # initial G
    g_final: float                 # G after the run; equal finals <=> same number of draws (for the same g)
    faults: Dict[int, Tuple[int, Optional[int]]]
    variant: str
    shutdown_step: int             # first 1 s step whose TEFUNC call raised the shutdown flag, 0 if none
    n_written: int                 # samples written (< n_samples only with stop_on_shutdown)
    seed: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def shutdown(self) -> bool:
        return self.shutdown_step > 0

    @property
    def shutdown_sample(self) -> int:
        """First sample showing the shutdown state, or -1.  Sample k shows XMEAS computed in step 180*(k+1)-1, so a
        flag raised in step i first appears in sample i // 180."""
        if not self.shutdown:
            return -1
        return int(self.shutdown_step // STEPS_PER_SAMPLE)

    @property
    def shutdown_hours(self) -> float:
        return self.shutdown_step / 3600.0 if self.shutdown else float("nan")


def normalize_faults(faults) -> Dict[int, Tuple[int, Optional[int]]]:
    """Accepts {idv: onset}, {idv: (onset, duration)}, or an iterable of (idv, onset[, duration]).
    onset and duration are in samples (3 min); duration None means until the end of the run."""
    out: Dict[int, Tuple[int, Optional[int]]] = {}
    if faults is None:
        return out
    items = faults.items() if isinstance(faults, Mapping) else [(f[0], f[1:]) for f in faults]
    for idv, spec in items:
        if isinstance(spec, (tuple, list)):
            onset = int(spec[0]) if len(spec) > 0 else 0
            duration = None if len(spec) < 2 or spec[1] is None else int(spec[1])
        else:
            onset, duration = int(spec), None
        idv = int(idv)
        if not 1 <= idv <= N_IDV:
            raise ValueError(f"IDV index {idv} outside 1..20")
        if onset < 0 or (duration is not None and duration <= 0):
            raise ValueError(f"bad schedule for IDV({idv}): onset={onset}, duration={duration}")
        if idv in out:
            raise ValueError(f"IDV({idv}) given twice; one interval per disturbance")
        out[idv] = (onset, duration)
    return out


def round_e13_5(X: np.ndarray) -> np.ndarray:
    """Round to the 5 significant digits of the FORMAT(E13.5) used by SUBROUTINE OUTPUT of temain_mod.f, giving
    the float64 a reader of those files obtains (the precision of the classic and Rieth data)."""
    flat = np.asarray(X, dtype=np.float64).ravel()
    out = np.array([float("%.4e" % v) if np.isfinite(v) else v for v in flat])
    return out.reshape(np.shape(X))


class TEPSimulator:
    """One loaded copy of the Fortran simulator.  Not thread-safe (COMMON blocks are global)."""

    def __init__(self, variant: str = "original", libdir: Union[str, Path] = LIBDIR):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        libdir = Path(libdir)
        info = json.loads((libdir / "build_info.json").read_text())["variants"][variant]
        self.variant = variant
        self.lib = ctypes.CDLL(str(libdir / info["library"]))
        fn = self.lib.tepsim_
        P = ctypes.POINTER
        fn.argtypes = [P(ctypes.c_int), P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_int),
                       P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double), P(ctypes.c_int), P(ctypes.c_int)]
        fn.restype = None
        self._fn = fn
        # COMMON blocks.  None has a DATA/BLOCK DATA initialisation, so all of them live in .bss and a freshly
        # loaded program sees zeros.  _reset() writes those zeros before every run.  This matters: TESUB2 solves
        # for the temperatures by Newton iterations that start from the values left in COMMON/TEPROC/, so a run
        # started from the state left by an earlier run can differ in the last bits (about 1e-11 relative).
        # Zero-filling (rather than restoring a snapshot) also covers a simulator created in a forked worker,
        # whose inherited COMMON blocks are not those of a fresh program.
        self._commons = []
        for sym, size in info["commons"].items():
            buf = (ctypes.c_char * size).in_dll(self.lib, sym)
            self._commons.append((sym, buf, size))
        self._g = ctypes.c_double.in_dll(self.lib, "randsd_")

    def _reset(self):
        for _, buf, size in self._commons:
            ctypes.memset(buf, 0, size)

    def run(self, n_samples: int = 500, *, seed: Optional[int] = None, g: Optional[float] = None,
            faults=None, stop_on_shutdown: bool = False, round5: bool = False,
            _skip_reset: bool = False) -> SimResult:
        """Simulate n_samples 3-minute samples.

        seed / g : integer seed mapped by g_from_seed, or the raw initial G (overrides seed).  Neither: G=4651207995.
        faults   : {idv: onset} or {idv: (onset, duration)} in samples, any subset of IDV(1..20).
        stop_on_shutdown : stop integrating at the first shutdown; unwritten samples are NaN.  Default False keeps
                   the original behaviour (TEFUNC freezes the states and keeps going).
        round5   : return values rounded like FORMAT(E13.5), as stored in the classic and Rieth files.
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        sched = normalize_faults(faults)
        if g is None:
            g = DEFAULT_G if seed is None else g_from_seed(seed, 0)
        g = float(g)
        gw_seed = seed if seed is not None else int(g)
        gw = (ctypes.c_double * 12)(*[g_from_seed(gw_seed, j) for j in range(1, 13)])
        on = (ctypes.c_int * N_IDV)(*([NEVER] * N_IDV))
        off = (ctypes.c_int * N_IDV)(*([NEVER] * N_IDV))
        for idv, (onset, duration) in sched.items():
            on[idv - 1] = STEPS_PER_SAMPLE * onset
            off[idv - 1] = NEVER if duration is None else STEPS_PER_SAMPLE * (onset + duration)
        X = np.full((n_samples, N_VARS), np.nan, dtype=np.float64)   # C order == Fortran XOUT(52, NSAMP)
        nwrit = ctypes.c_int(0)
        isd = ctypes.c_int(0)
        if not _skip_reset:          # _skip_reset only exists to show that the reset is needed
            self._reset()
        self._fn(ctypes.byref(ctypes.c_int(n_samples)), ctypes.byref(ctypes.c_double(g)), gw, on, off,
                 ctypes.byref(ctypes.c_int(1 if stop_on_shutdown else 0)),
                 X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(nwrit), ctypes.byref(isd))
        g_final = float(self._g.value)
        if nwrit.value < n_samples:
            X[nwrit.value:] = np.nan
        if round5:
            X = round_e13_5(X)
        return SimResult(X=X, g=g, g_final=g_final, faults=sched, variant=self.variant,
                         shutdown_step=int(isd.value), n_written=int(nwrit.value), seed=seed)


# ----------------------------------------------------------------------------------------------------------------
# process-parallel batch runs
_WORKER: Dict[str, TEPSimulator] = {}


def _worker_run(job: dict) -> SimResult:
    variant = job.get("variant", "original")
    sim = _WORKER.get(variant)
    if sim is None:
        sim = _WORKER[variant] = TEPSimulator(variant)
    kw = {k: v for k, v in job.items() if k not in ("variant", "tag")}
    res = sim.run(**kw)
    res.meta["pid"] = os.getpid()
    if "tag" in job:
        res.meta["tag"] = job["tag"]
    return res


def simulate_many(jobs: Sequence[dict], processes: Optional[int] = None, chunksize: int = 1) -> List[SimResult]:
    """Run jobs (dicts of TEPSimulator.run keyword arguments, plus optional 'variant' and 'tag') in worker
    processes.  Results come back in job order and do not depend on which worker ran them."""
    import multiprocessing as mp
    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 2)
    if processes <= 1:
        return [_worker_run(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes) as pool:
        return pool.map(_worker_run, list(jobs), chunksize=chunksize)


def simulate(n_samples: int = 500, *, seed: Optional[int] = None, g: Optional[float] = None, faults=None,
             variant: str = "original", stop_on_shutdown: bool = False, round5: bool = False) -> SimResult:
    """One-off convenience wrapper (reuses a per-process simulator)."""
    return _worker_run(dict(n_samples=n_samples, seed=seed, g=g, faults=faults, variant=variant,
                            stop_on_shutdown=stop_on_shutdown, round5=round5))


def simulate_subsets(n_samples: int, faults, *, seed: Optional[int] = None, g: Optional[float] = None,
                     variant: str = "split", processes: Optional[int] = None, **kw) -> Dict[frozenset, SimResult]:
    """Common-random-numbers group: run every subset of `faults` (including the empty set) with the same seed.
    Returns {frozenset(idvs): SimResult}.  With k disturbances this is 2**k runs; for two disturbances A, B the
    keys are {}, {A}, {B}, {A, B}, and X[{A,B}] - X[{A}] - X[{B}] + X[{}] is their interaction.  The default
    variant is "split", whose noise stream is shared by all subsets for every IDV (see module docstring)."""
    from itertools import combinations
    sched = normalize_faults(faults)
    keys = [frozenset(c) for r in range(len(sched) + 1) for c in combinations(sorted(sched), r)]
    jobs = [dict(n_samples=n_samples, seed=seed, g=g, variant=variant,
                 faults={i: sched[i] for i in k}, **kw) for k in keys]
    res = simulate_many(jobs, processes=processes if processes is not None else min(len(jobs), os.cpu_count() or 1))
    return dict(zip(keys, res))
