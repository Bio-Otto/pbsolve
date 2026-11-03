#!/usr/bin/env python3
"""Compare pbsolve with APBS on one PQR file (thin wrapper around ``pbsolve.apbs``).

    python scripts/compare_apbs.py protein.pqr --spacing 0.5 --ionic-strength 0.15
    python scripts/compare_apbs.py complex.pqr --ligand-chain B          # + receptor, ligand, ΔΔG

Equivalent to ``pbsolve apbs compare ...``; run ``pbsolve apbs run apbs.in``
to evaluate an existing APBS input file with pbsolve. Exit code 0 = PASS,
1 = FAIL, 2 = error (e.g. APBS not found, unsupported input).
"""

import sys

from pbsolve.apbs import main

if __name__ == "__main__":
    sys.exit(main(["compare", *sys.argv[1:]]))
