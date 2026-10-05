# Theoretical statements and proofs

[Read the proof document](VioExplain_proofs.pdf). Its [standalone LaTeX source](VioExplain_proofs.tex) contains the bibliography and requires no manuscript files. Result numbers retain their correspondence to the manuscript. Proposition S2 gives a stronger sufficient recovery certificate.

## Mathematical scope

- Theorem 1 bounds an event-prior-penalized **profile likelihood**, with the assignment treated as a nuisance parameter. It is not a joint or marginal posterior for an unspecified random assignment. The submodular coverage term, facility-cost equivalence, computational hardness, and stated greedy and primal-dual approximation guarantees are proved separately. The binary covering specialization has explicit consistency and cost assumptions.
- Theorem 2 applies to the particular interface supplied to it. Corollary 3 retains the complete violation-feature interface. A mismatch probability for keys alone cannot be substituted for the mismatch probability of features that also contain numerical intervals.
- Theorem 3 is a sufficient certificate along the actual algorithm path. Proposition S2 relaxes the addition-threshold margin from twice the score discrepancy to once the discrepancy, while retaining twice the discrepancy for ordering against false candidates.
- Theorem 4 assumes frozen training and tuning, separate normal, single-event, and unknown-score calibration pools, and the stated conditional exchangeability at each stage. The upper bounds allow ties and handle empty or insufficient calibration pools with an infinite upper threshold. The Beta fluctuation statement additionally requires conditional independent and identically distributed continuous scores.
- Proposition S1 concerns **changes** in degrees under paired perturbations. Its general multiple-predictor coefficient is distinguished from the simple-regression special case.

## Implementation scope

This document revision does not change any experiment code or recorded result. In `experiments/tii_final/final_fuse5.py`, fusion-weight selection uses test-calibration labels before the same calibration pool supplies naming thresholds. That workflow has not been converted here to the independent tuning and calibration protocol of Theorem 4. Selecting a weight before computing a threshold is insufficient if the selection used that threshold's calibration data.

Theorem 4(c) also requires that the complete event-selection rule be frozen before the unknown-score calibration pool is used. Reusing the data that determine that rule is not covered by the proof. The implementation must meet these assumptions before its empirical results can be described as enjoying this finite-sample guarantee. No experiment performance improvement is claimed from the sharper certificate alone.

## Build

With an existing LaTeX installation, run from this directory:

```bash
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build VioExplain_proofs.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build VioExplain_proofs.tex
cp build/VioExplain_proofs.pdf VioExplain_proofs.pdf
```

The source also compiles as a standalone document in an editor. It uses the standard article class and common mathematics packages. No BibTeX run is needed.

## Finite verification examples

```bash
python3 check_mathematics.py
```

The checks exercise finite instances of the algebraic claims, exact conformal rank counting, a sharper-margin example, and the general functional coefficient. They supplement the written proofs and do not establish the theorems by testing. They require only the Python standard library and do not load datasets or model weights.
