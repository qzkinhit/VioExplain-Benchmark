# Generation of the concurrent Tennessee Eastman runs (TEP-C)

This folder builds the closed-loop Tennessee Eastman simulator as a shared library and generates the paired runs of TEP-C. Every fault set is simulated together with all of its subsets under one seed, so the fault-free run, the single-fault runs and the concurrent run of a seed share their random numbers.

## Files

| File | Role |
|---|---|
| `build.py` | Checks the upstream Fortran files, writes the driver and the two source variants, compiles `lib/libtep_original.so` and `lib/libtep_split.so` |
| `fortran/tepsim_driver.tmpl.f` | Callable form of the main program, with four markers that `build.py` fills from the upstream files |
| `tepsim/core.py` | `ctypes` interface, seeds, disturbance schedules, parallel runs |
| `tepsim/names.py` | Names of the 52 variables and of the 20 disturbances |
| `scripts/generate_crn.py` | Generation of the paired runs of a list of fault sets |

## Upstream code

The Fortran code of the process (`teprob.f`, Downs and Vogel) and of its closed-loop control (`temain_mod.f`, Russell, Chiang and Braatz) is not bundled. Place both files in `vendor/tennessee-eastman-profbraatz/`. `build.py` compares their SHA-256 with the values at its top and stops on a mismatch. It needs `gfortran`.

```bash
python3 build.py
```

## Variants

The `original` variant is the published code. Its disturbances IDV(17), IDV(18) and IDV(20) skip one draw of the shared random generator at the end of every pulse, so a faulty run stops sharing measurement noise with the fault-free run of the same seed. The `split` variant gives each of the 12 random walks its own generator state through the four patches listed in `build.py`. The measurement noise is then consumed identically by every run that does not shut down. TEP-C uses the `split` variant.

## Conventions

One integration step is 1 s and one sample is 180 steps. A disturbance with onset `s` is on from step `180 s`, and the samples before the onset equal the fault-free run of the same seed. The testing protocol has 960 samples and onset 160, and the training protocol has 500 samples and onset 20. After a plant shutdown the upstream code freezes the states, and the shutdown sample of every run is stored in `meta.json`.

## Commands

```bash
OUT=/path/to/vioexplain/data/tepsim
TRIPLES="1+2+19,1+4+6,1+5+9,1+6+14,1+7+11,1+7+12,1+9+18,1+14+16,1+16+20,2+3+10,2+3+11,2+3+13,2+4+13,2+6+9,2+7+11,2+9+15,2+11+12,2+13+16,3+11+14,4+7+20,4+10+17,4+10+20,5+6+17,5+7+13,5+7+20,5+13+18,5+19+20,6+14+15,7+13+16,7+13+19,8+14+16,9+12+13,9+13+15,9+14+15,10+13+20,12+13+15,12+16+20,13+18+20,14+15+17,14+17+20"
python3 scripts/generate_crn.py --sets all-pairs --seeds 2001-2020 --protocol test --variant split --out $OUT/test_pairs
python3 scripts/generate_crn.py --sets "$TRIPLES" --seeds 2101-2110 --protocol test --variant split --out $OUT/test_triples
python3 scripts/generate_crn.py --sets all-pairs --seeds 1001-1020 --protocol train --variant split --out $OUT/dev_pairs
python3 scripts/generate_crn.py --sets "$TRIPLES" --seeds 1101-1105 --protocol train --variant split --out $OUT/dev_triples
```

Each output directory holds one array `set_<name>.npy` of shape (seeds, samples, 52) per fault set, where `<name>` is `normal` or the disturbance numbers joined by `+`, and `meta.json` with the seeds, the generator states and the shutdown step of every run. The runners of `experiments/tii_final/` read `test_pairs` and `test_triples` from the directory `SIM` of `common.py`. Windows after a shutdown in any run of a paired group are removed there, and the four-fault windows are superposed from the paired single-fault runs of `test_pairs`.
