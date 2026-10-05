# Experiment scripts of the paper

These scripts produce every number of the paper. Set the root directory in `common.py` (`/path/to/vioexplain`) to the location of the data.

- `common.py`: data access, window descriptions with functional-constraint units, event knowledge from paired runs, counterfactual compositions, evaluation and the split-conformal thresholds shared by all methods.
- `final_fuse5.py`: VioExplain (deduction-consistent scorer, learned addition decision, temporal posterior in the first selection, calibrated thresholds).
- `final_base.py`, `final_rocket.py`, `final_probs.py`: the compared methods; `gpu_*.py`: training of the deep and multi-label networks.
- `simproc/`: simulators and runners of the further process benchmarks.

The proofs of the theoretical results are in `docs/proofs/VioExplain_proofs.pdf`.
