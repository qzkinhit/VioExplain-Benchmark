# Two simulated benchmarks for single-fault diagnosis: continuous fermenter (FERM) and heat exchanger (HEX)

This folder adds two closed-loop process simulators with paired runs and runs VioExplain and the 18 baselines on them
under the protocol of `../simproc2/` (see its README). `common.py`, `final_base.py`, `final_fuse5.py`,
`final_probs.py`, `final_rocket.py`, `gpu_simproc.py`, `gen_data.py` and `sim_check.py` are copies of the
`../simproc2/` files. `common.py` reads its data from `data/simproc4/<process>/` and `gen_data.py` writes there.

- `sim_ferm.py`: continuous fermenter of Henson and Seborg (1992) with a level loop, a feed flow loop and a biomass
  loop. 8 recorded variables, 10 fault types.
- `sim_hex.py`: counter-current shell-and-tube heat exchanger under outlet-temperature control, lumped cell model with
  wall storage. 10 recorded variables, 10 fault types.

Each docstring lists the model equations, the parameters taken from the source and every deviation. The fault blocks,
the disturbance and noise models and the controller settings are written here.

Run order: `gen_data.py ferm|hex`, `final_base.py` and `final_rocket.py` (CPU baselines), `gpu_simproc.py` (networks
and MantisV2), `final_probs.py`, `final_fuse5.py` (VioExplain), `stats4.py`, `collect4.py`. The paper reports the
macro F1 of both processes, and the result files are in `results/simulated/{ferm,hex}/` at the top of this repository.
