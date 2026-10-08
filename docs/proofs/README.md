# Theoretical statements and proofs

[VioExplain_proofs.pdf](VioExplain_proofs.pdf) proves the four theorems of the paper, with the same numbering. [VioExplain_proofs.tex](VioExplain_proofs.tex) is its standalone LaTeX source and contains its own bibliography.

- **Theorem 1** shows that the explanation objective is a lower bound of an event-prior-penalized profile likelihood and the largest such bound with one penalty per event, that it is a facility-location problem with a background facility of zero cost, and that it is NP-hard.
- **Theorem 2** gives a lower bound on the error of every rule that reads a window only through an interface, in terms of a probability measured on paired runs.
- **Theorem 3** gives a recovery condition. Under margins measured on paired runs, the algorithm returns exactly the true event set.
- **Theorem 4** bounds the rates of naming an event on a fault-free window, of adding a false event and of flagging a known window as unknown, for thresholds calibrated by split conformal prediction.

## Build

```bash
pdflatex VioExplain_proofs.tex
pdflatex VioExplain_proofs.tex
```
