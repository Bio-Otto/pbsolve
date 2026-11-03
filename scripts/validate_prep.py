#!/usr/bin/env python3
"""Compare `pbsolve prep` with pdb2pqr byte for byte.

    python scripts/validate_prep.py 1ycr.pdb 1tii.pdb ... [--ff amber parse charmm] [--keep]

Runs `pdb2pqr --ff=FF [--noopt] --drop-water --keep-chain --whitespace` and
`pbsolve.prep.prepare(..., forcefield=FF, optimize=...)` on every PDB file and reports
identical / differing lines. Needs pdb2pqr (BSD) on PATH. Exit 1 on any difference.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

from pbsolve.prep import prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="+")
    ap.add_argument("--ff", nargs="+", default=["amber", "parse", "charmm"])
    ap.add_argument("--keep", help="directory to keep the PQR files in")
    ap.add_argument("--mode", nargs="+", default=["opt", "noopt"], choices=["opt", "noopt"])
    a = ap.parse_args()
    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    if exe is None:
        print("pdb2pqr not found on PATH", file=sys.stderr)
        return 2
    out = pathlib.Path(a.keep or tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for pdb in a.pdb:
        stem = pathlib.Path(pdb).stem
        for ff, mode in [(f, m) for f in a.ff for m in a.mode]:
            ref = out / f"{stem}_{ff}_{mode}_pdb2pqr.pqr"
            flags = ["--noopt"] if mode == "noopt" else []
            r = subprocess.run([exe, f"--ff={ff.upper()}", *flags, "--drop-water", "--keep-chain", "--whitespace",
                                pdb, str(ref)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{stem:>12} {ff:>6} {mode:>5}: pdb2pqr failed: {r.stderr.strip().splitlines()[-1:]}")
                bad += 1
                continue
            try:
                _, rep = prepare(pdb, ff, optimize=mode == "opt")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{stem:>12} {ff:>6} {mode:>5}: pbsolve failed: {exc!r}")
                bad += 1
                continue
            (out / f"{stem}_{ff}_{mode}_pbsolve.pqr").write_text(rep.pqr_text)
            theirs = ref.read_text().splitlines()
            ours = rep.pqr_text.splitlines()
            diff = sum(1 for x, y in zip(ours, theirs) if x != y) + abs(len(ours) - len(theirs))
            status = "IDENTICAL" if diff == 0 else f"{diff} lines differ"
            print(f"{stem:>12} {ff:>6} {mode:>5}: {len(ours)} atoms, {status}")
            if diff:
                bad += 1
                for x, y in zip(ours, theirs):
                    if x != y:
                        print(f"    pbsolve: {x}\n    pdb2pqr: {y}")
                        break
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
