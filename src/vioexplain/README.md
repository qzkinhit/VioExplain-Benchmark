# Method implementation

| Module | Role | Status |
|---|---|---|
| `api.py` | Stable explanation interface with explicit unmatched evidence | Public interface |
| `publicexp/constraints.py`, `features.py` | Historical public-data violation extractor | Compatibility adapter; see limitations |
| `publicexp/knowledge.py` | Representation signatures from training violation patterns | Compatibility adapter |
| `publicexp/aec.py` | Historical dimension-signature AEC adapter | Compatibility output includes unmatched dimensions in `pred_dims` |
| `mincost/` | Later MinExplain objective, density greedy, primal-dual, safe pruning, exact small-instance reference | Later formalization |
| `setcover/` | AEC-Prototype interval representations, original MyCover/Select, and graph-based update | Retained original prototype API |
| `update/` | Controlled synthetic incomplete-knowledge update study | Synthetic research component |
| `tii/` | Causal residual adapters and joint latent correction | Exploratory; no general superiority claim |

`api.py` is the recommended interface for new work. The historical adapter is retained so old reproduction code can still import it. Its `pred_dims` field merges unmatched evidence with predicted dimensions; this field must not be interpreted as a fully knowledge-supported diagnosis. The new API names and indices make this distinction explicit.

The minimum-cost objective sums the opening cost of each selected representation and the distance of each assigned violation. Approximation statements require nonnegative costs and the stated fixed feasible graph. Numerical checks on small instances do not replace a proof. The exact solver uses optional PuLP/CBC when installed and otherwise enumerates small candidate subsets.
