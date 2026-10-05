"""tepsim: the original closed-loop Tennessee Eastman Fortran simulator with seeds and disturbance schedules.

    from tepsim import TEPSimulator
    sim = TEPSimulator()                                   # "original" build, one per process
    x0 = sim.run(500, seed=7).X                            # fault-free, (500, 52)
    x1 = sim.run(500, seed=7, faults={1: 20}).X            # IDV(1) from sample 20 (1 h), same random numbers
    x14 = sim.run(500, seed=7, faults={1: 20, 4: 20}).X    # compound IDV(1)+IDV(4)

    from tepsim import simulate_subsets                    # {}, {1}, {4}, {1,4} with one seed, in parallel
    group = simulate_subsets(500, {1: 20, 4: 20}, seed=7)

See core.py for the time and seed conventions.
"""
from .core import (DEFAULT_G, N_VARS, STEPS_PER_SAMPLE, VARIANTS, SimResult, TEPSimulator, cycle_length,
                   g_from_seed, normalize_faults, round_e13_5, settle, simulate, simulate_many, simulate_subsets,
                   tesub7_next)
from .names import DESCRIPTIONS, FEATURES, IDV_DESCRIPTIONS, RNG_SHIFTING_IDV

__all__ = ["TEPSimulator", "SimResult", "simulate", "simulate_many", "simulate_subsets", "g_from_seed",
           "settle", "cycle_length", "tesub7_next", "normalize_faults",
           "round_e13_5", "FEATURES", "DESCRIPTIONS", "IDV_DESCRIPTIONS", "RNG_SHIFTING_IDV", "DEFAULT_G",
           "N_VARS", "STEPS_PER_SAMPLE", "VARIANTS"]
