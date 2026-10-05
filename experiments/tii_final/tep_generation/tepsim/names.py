"""Variable and disturbance names, in the column order of the classic (Braatz) and Rieth et al. (2017) data."""

FEATURES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

DESCRIPTIONS = [
    "A feed (stream 1) kscmh", "D feed (stream 2) kg/h", "E feed (stream 3) kg/h", "A and C feed (stream 4) kscmh",
    "Recycle flow (stream 8) kscmh", "Reactor feed rate (stream 6) kscmh", "Reactor pressure kPa gauge",
    "Reactor level %", "Reactor temperature C", "Purge rate (stream 9) kscmh", "Product separator temperature C",
    "Product separator level %", "Product separator pressure kPa gauge", "Separator underflow (stream 10) m3/h",
    "Stripper level %", "Stripper pressure kPa gauge", "Stripper underflow (stream 11) m3/h",
    "Stripper temperature C", "Stripper steam flow kg/h", "Compressor work kW",
    "Reactor cooling water outlet temperature C", "Separator cooling water outlet temperature C",
] + [f"Reactor feed analysis {c} mol%" for c in "ABCDEF"] \
  + [f"Purge gas analysis {c} mol%" for c in "ABCDEFGH"] \
  + [f"Product analysis {c} mol%" for c in "DEFGH"] + [
    "D feed flow valve %", "E feed flow valve %", "A feed flow valve %", "A and C feed flow valve %",
    "Compressor recycle valve %", "Purge valve %", "Separator pot liquid flow valve %",
    "Stripper liquid product flow valve %", "Stripper steam valve %", "Reactor cooling water flow valve %",
    "Condenser cooling water flow valve %",
]

IDV_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (stream 4), step",
    2: "B composition, A/C ratio constant (stream 4), step",
    3: "D feed temperature (stream 2), step",
    4: "Reactor cooling water inlet temperature, step",
    5: "Condenser cooling water inlet temperature, step",
    6: "A feed loss (stream 1), step",
    7: "C header pressure loss, reduced availability (stream 4), step",
    8: "A, B, C feed composition (stream 4), random variation",
    9: "D feed temperature (stream 2), random variation",
    10: "C feed temperature (stream 4), random variation",
    11: "Reactor cooling water inlet temperature, random variation",
    12: "Condenser cooling water inlet temperature, random variation",
    13: "Reaction kinetics, slow drift",
    14: "Reactor cooling water valve, sticking",
    15: "Condenser cooling water valve, sticking",
    16: "Unknown (code: random walk 9, stripper steam heat transfer factor UAC)",
    17: "Unknown (code: pulse process 10, reactor heat transfer factor QUR)",
    18: "Unknown (code: pulse process 11, condenser heat transfer factor QUS)",
    19: "Unknown (code: sticking of valves 5, 7, 8, 9)",
    20: "Unknown (code: pulse process 12, reactor-to-separator flow factor FTM(8))",
}

# IDV(j) -> mechanism in teprob.f.  Pulse processes 10-12 consume a different number of generator draws when the
# disturbance is on, which breaks common random numbers in the "original" build (see core.py).
RNG_SHIFTING_IDV = (17, 18, 20)
