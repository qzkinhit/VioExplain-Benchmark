# Official data adapters

The executable adapters are `vioexplain.formal.data` and do not rename or invent labels.

| Track | Required directory | Label meaning |
|---|---|---|
| SMD | `train/`, `test/`, `test_label/`, `interpretation_label/` | Binary anomalies; event-conditional annotated dimensions |
| SKAB | `anomaly-free/`, `other/`, `valve1/`, `valve2/` | Binary anomalies and change points, no event identities |
| TEP | `d00.dat` to `d21.dat`, plus corresponding `_te.dat` | Normal operation and 21 simulated fault identities |

TEP columns are the official 41 measured variables followed by 11 manipulated variables. The transposed normal training file is handled explicitly. Fault-description uncertainty is preserved. No official labels are republished in this repository.
