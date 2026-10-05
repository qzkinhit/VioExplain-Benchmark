#!/usr/bin/env python3
"""Build the Tennessee Eastman shared libraries used by tepsim.

Inputs (never modified): vendor/tennessee-eastman-profbraatz/{teprob.f,temain_mod.f},
the Russell-Chiang-Braatz distribution of the Downs-Vogel code (closed loop).

Outputs:
  fortran/teprob_original.f   verbatim copy of teprob.f
  fortran/teprob_split.f      teprob.f with one generator state per random walk (see SPLIT_PATCHES)
  fortran/temain_mod_subs.f   verbatim subroutines of temain_mod.f (everything after the main program)
  fortran/tepsim_driver.f     fortran/tepsim_driver.tmpl.f with four verbatim blocks of temain_mod.f/teprob.f
  lib/libtep_original.so, lib/libtep_split.so
  lib/build_info.json         checksums, compiler, flags, COMMON block symbols and sizes

Usage: python3 build.py [--fflags "-O2"]
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "tennessee-eastman-profbraatz"
FORT = ROOT / "fortran"
LIB = ROOT / "lib"

# sha256 of the upstream files (github.com/jkitchin/tennessee-eastman-profbraatz, commit 9a6c8e5, top level)
EXPECTED_SHA256 = {
    "teprob.f": "409975e070780a3b197d37125dddeca64e46c11d8c99028528a56a1f238d45e3",
    "temain_mod.f": "31b83c17f8d52332d6f6fe46d9da3ee7131de2d018bb83945453350bf635c8b6",
}

# The "split" variant gives each of the 12 random walks of TEFUNC its own generator state GW(I) and leaves G to
# the measurement noise (TESUB6).  In the original code the walks 10-12 (IDV 17, 18, 20) skip one draw of the shared
# generator at the end of every pulse, which shifts all later measurement noise of a faulty run against the
# fault-free run with the same seed.  With split streams the noise stream is consumed identically in every run that
# does not shut down, so x_fault - x_normal is a common-random-numbers difference for all 20 disturbances.
SPLIT_PATCHES = [
    (  # declare the walk generator states in TEFUNC
        "      INTEGER NN,I,ISD\n",
        "      INTEGER NN,I,ISD\n"
        "      DOUBLE PRECISION GW\n"
        "      INTEGER ISTRM\n"
        "      COMMON/RANDSW/GW(12),ISTRM\n",
    ),
    (  # walks 1-9: the three draws of TESUB5 come from GW(I)
        "      CALL TESUB5(SWLK,SPWLK,ADIST(I),BDIST(I),CDIST(I),\n"
        "     .DDIST(I),TLAST(I),TNEXT(I),HSPAN(I),HZERO(I),\n"
        "     .SSPAN(I),SZERO(I),SPSPAN(I),IDVWLK(I))\n",
        "      ISTRM=I\n"
        "      CALL TESUB5(SWLK,SPWLK,ADIST(I),BDIST(I),CDIST(I),\n"
        "     .DDIST(I),TLAST(I),TNEXT(I),HSPAN(I),HZERO(I),\n"
        "     .SSPAN(I),SZERO(I),SPSPAN(I),IDVWLK(I))\n"
        "      ISTRM=0\n",
    ),
    (  # walks 10-12: the interval draw comes from GW(I)
        "      HWLK=HSPAN(I)*TESUB7(ISD)+HZERO(I)\n",
        "      ISTRM=I\n"
        "      HWLK=HSPAN(I)*TESUB7(ISD)+HZERO(I)\n"
        "      ISTRM=0\n",
    ),
    (  # the generator itself: same recursion, state selected by ISTRM
        "      DOUBLE PRECISION FUNCTION TESUB7(I)\n"
        "      INTEGER I\n"
        "      DOUBLE PRECISION G,DMOD\n"
        "      COMMON/RANDSD/G\n"
        "      G=DMOD(G*9228907.D0,4294967296.D0)\n"
        "      IF(I.GE.0)TESUB7=G/4294967296.D0\n"
        "      IF(I.LT.0)TESUB7=2.D0*G/4294967296.D0-1.D0\n"
        "      RETURN\n"
        "      END\n",
        "      DOUBLE PRECISION FUNCTION TESUB7(I)\n"
        "      INTEGER I\n"
        "      DOUBLE PRECISION G,DMOD\n"
        "      COMMON/RANDSD/G\n"
        "      DOUBLE PRECISION GW\n"
        "      INTEGER ISTRM\n"
        "      COMMON/RANDSW/GW(12),ISTRM\n"
        "      IF(ISTRM.EQ.0)THEN\n"
        "      G=DMOD(G*9228907.D0,4294967296.D0)\n"
        "      IF(I.GE.0)TESUB7=G/4294967296.D0\n"
        "      IF(I.LT.0)TESUB7=2.D0*G/4294967296.D0-1.D0\n"
        "      ELSE\n"
        "      GW(ISTRM)=DMOD(GW(ISTRM)*9228907.D0,4294967296.D0)\n"
        "      IF(I.GE.0)TESUB7=GW(ISTRM)/4294967296.D0\n"
        "      IF(I.LT.0)TESUB7=2.D0*GW(ISTRM)/4294967296.D0-1.D0\n"
        "      ENDIF\n"
        "      RETURN\n"
        "      END\n",
    ),
]

DEFAULT_FFLAGS = "-O2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(lines, start_pred, end_pred, include_start=True, include_end=True, start_at=0):
    """Return lines[s:e] for the first start line matching start_pred at or after start_at and the first end line
    after it matching end_pred."""
    s = next(i for i in range(start_at, len(lines)) if start_pred(lines[i]))
    e = next(i for i in range(s + 1, len(lines)) if end_pred(lines[i]))
    return lines[s if include_start else s + 1: e + 1 if include_end else e], s, e


def extract(temain_text, teprob_text):
    tm = temain_text.splitlines(keepends=True)
    tp = teprob_text.splitlines(keepends=True)
    # main program of temain_mod.f ends at its first END statement
    end_main = next(i for i, l in enumerate(tm) if re.fullmatch(r"\s+END\s*", l.rstrip("\n")))
    # the leading comment block (authors, copyright and licence) is kept on top of the extracted subroutines,
    # as the licence asks for redistributed source
    first_code = next(i for i, l in enumerate(tm) if l[:1] not in ("C", "c", "*", "!", "\n", ""))
    header = ["C" + "=" * 70 + "\n",
              "C  temain_mod_subs.f: the subroutines of temain_mod.f (everything after\n",
              "C  the main program), copied verbatim by tepsim/build.py.  The header\n",
              "C  below is the original header of temain_mod.f.\n",
              "C" + "=" * 70 + "\n"] + tm[:first_code]
    subs = header + tm[end_main + 1:]
    decl, _, _ = block(tm, lambda l: "MEASUREMENT AND VALVE COMMON BLOCK" in l,
                       lambda l: "COMMON/CTRL22/" in l)
    setup, _, _ = block(tm, lambda l: "CALL TEINIT(NN,TIME,YY,YP)" in l,
                        lambda l: "Set all Disturbance Flags to OFF" in l, include_start=False, include_end=False)
    calls, _, _ = block(tm, lambda l: "TEST=MOD(I,3)" in l, lambda l: "CALL CONTRL20" in l)
    # COMMON/TEPROC/ declarations of TEFUNC: from the line after COMMON/DVEC/ to the line ending with IVST(12)
    tefunc = next(i for i, l in enumerate(tp) if "SUBROUTINE TEFUNC" in l)
    teproc, _, _ = block(tp, lambda l: "COMMON/DVEC/IDV(20)" in l, lambda l: "IVST(12)" in l,
                         include_start=False, start_at=tefunc)
    # sanity checks on the extracted blocks
    assert any("SUBROUTINE CONTRL1" in l for l in subs) and any("SUBROUTINE CONSHAND" in l for l in subs)
    n_common = sum("COMMON/" in l for l in decl if l[:1] not in "Cc*")
    assert n_common == 24, f"expected 24 COMMON statements in the main program, found {n_common}"
    assert sum(l.strip().startswith("SETPT(") for l in setup) == 20
    assert sum(l.strip().startswith("XMV(") for l in setup) == 11
    assert sum("CALL CONTRL" in l for l in calls) == 19
    assert any("COMMON/TEPROC/" in l for l in teproc)
    return "".join(subs), "".join(decl), "".join(setup), "".join(calls), "".join(teproc)


def make_driver(template, decl, setup, calls, teproc):
    out = []
    for line in template.splitlines(keepends=True):
        tag = line.strip()
        if tag == "C@@MAIN_DECLARATIONS":
            out.append("C     >>> verbatim from temain_mod.f (main program declarations)\n" + decl +
                       "C     <<< end of verbatim block\n")
        elif tag == "C@@TEPROC_DECLARATIONS":
            out.append("C     >>> verbatim from teprob.f (TEFUNC, COMMON/TEPROC/)\n" + teproc +
                       "C     <<< end of verbatim block\n")
        elif tag == "C@@CONTROLLER_SETUP":
            out.append("C     >>> verbatim from temain_mod.f (controller setup)\n" + setup +
                       "C     <<< end of verbatim block\n")
        elif tag == "C@@CONTROL_CALLS":
            out.append("C     >>> verbatim from temain_mod.f (simulation loop)\n" + calls +
                       "C     <<< end of verbatim block\n")
        else:
            out.append(line)
    text = "".join(out)
    assert "C@@" not in text, "unreplaced marker"
    return text


def make_split(teprob_text):
    text = teprob_text
    for old, new in SPLIT_PATCHES:
        n = text.count(old)
        assert n == 1, f"patch anchor found {n} times:\n{old}"
        text = text.replace(old, new)
    return text


def common_names(paths):
    names = set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line[:1] in ("C", "c", "*", "!"):       # fixed-form comment line
                continue
            for m in re.finditer(r"COMMON\s*/\s*(\w+)\s*/", line, flags=re.IGNORECASE):
                names.add(m.group(1).lower())
    return sorted(names)


def nm_symbols(so):
    out = subprocess.run(["nm", "-S", "--defined-only", str(so)], check=True, capture_output=True, text=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            addr, size, typ, name = parts
            syms[name] = {"type": typ, "size": int(size, 16)}
    return syms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fflags", default=DEFAULT_FFLAGS, help="optimisation flags passed to gfortran")
    ap.add_argument("--allow-mismatch", action="store_true", help="accept vendor files with other checksums")
    args = ap.parse_args()

    src = {name: VENDOR / name for name in EXPECTED_SHA256}
    for name, path in src.items():
        h = sha256(path)
        if h != EXPECTED_SHA256[name] and not args.allow_mismatch:
            sys.exit(f"{path}: sha256 {h} differs from the audited upstream file")
    temain_text = src["temain_mod.f"].read_text()
    teprob_text = src["teprob.f"].read_text()

    FORT.mkdir(exist_ok=True)
    LIB.mkdir(exist_ok=True)
    subs, decl, setup, calls, teproc = extract(temain_text, teprob_text)
    (FORT / "teprob_original.f").write_text(teprob_text)
    (FORT / "teprob_split.f").write_text(make_split(teprob_text))
    (FORT / "temain_mod_subs.f").write_text(subs)
    template = (FORT / "tepsim_driver.tmpl.f").read_text()
    (FORT / "tepsim_driver.f").write_text(make_driver(template, decl, setup, calls, teproc))

    gfortran = "gfortran"
    version = subprocess.run([gfortran, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    base = ["-fPIC", "-shared", "-std=legacy", "-w"]
    info = {"gfortran": version, "fflags": args.fflags.split() + base, "vendor_sha256": EXPECTED_SHA256,
            "generated_sha256": {}, "variants": {}}
    for variant in ("original", "split"):
        sources = [FORT / f"teprob_{variant}.f", FORT / "temain_mod_subs.f", FORT / "tepsim_driver.f"]
        so = LIB / f"libtep_{variant}.so"
        cmd = [gfortran] + args.fflags.split() + base + ["-o", str(so)] + [str(s) for s in sources]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        syms = nm_symbols(so)
        commons = {}
        for name in common_names(sources):
            sym = name + "_"
            if sym not in syms:
                sys.exit(f"COMMON block /{name}/ ({sym}) not found in {so}")
            commons[sym] = syms[sym]["size"]
        # local static data (would survive between runs and is not reset by tepsim); expected to be empty
        local_static = {k: v for k, v in syms.items() if v["type"] in "bd" and not k.startswith("_")
                        and not k.startswith("completed.")}
        info["variants"][variant] = {"library": so.name, "sha256": sha256(so), "commons": commons,
                                     "local_static_symbols": local_static,
                                     "sources": [s.name for s in sources]}
    for f in ("teprob_original.f", "teprob_split.f", "temain_mod_subs.f", "tepsim_driver.f"):
        info["generated_sha256"][f] = sha256(FORT / f)
    (LIB / "build_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
