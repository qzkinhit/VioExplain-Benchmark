# Tennessee Eastman process

Tennessee Eastman is an established simulated chemical-process benchmark. This formal track uses the [classic MIT/Braatz distribution](https://web.mit.edu/braatzgroup/TE_process.zip), identified by SHA256 `3ae9d9578cd0883ed81ee3c2b911f85e57e367016e08bd5f1bde963694e4c111`. The archive contains normal and fault training/testing files, its readme, and simulation source. The downloader verifies the archive before extraction and does not execute its source.

```bash
python data/download.py tep
```

The resulting `data/raw/tep/` contains `d00.dat` and the remaining training and test files. Retain the exact UIUC source/binary notice in the supplied readme. No TEP data or simulation source is bundled in this software release.

The 52 columns comprise 41 XMEAS measurements and 11 XMV manipulated variables. The normal training file is stored transposed, which the adapter corrects explicitly. Fault files are simulated experiments, not independent operating plants. The formal detection labels use a protocol-defined fault onset at zero-based sample 160 in each faulty test file; there is no separate official binary-label file. The archive driver `temain_mod.f.txt` provides the eight-hour fault switch and three-minute sampling that support this convention. Its retained example is fault 12, so it is not a complete generation record for every distributed file. Do not confuse it with learned change points or native human annotations.

A fault identifier supports event-class diagnosis under a documented training/test split. It is not an exhaustive root-dimension annotation. The supplied simulator comments describe fault identities 1–15, label 16–20 as unknown, and do not establish a reliable physical description for fault 21. Use identifiers when the physical semantics are unverified.
