"""Installation check, in the spirit of ``python -m openmm.testInstallation``.

Solves a Born ion on every available platform, compares against the
analytic result and checks that the platforms agree with each other.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Sequence

import numpy as np

from . import Solver, __version__, available_platforms, cuda_info


def born_energy(radius: float = 2.0, pdie: float = 1.0, sdie: float = 78.54, temperature: float = 298.15) -> float:
    """Analytic Born solvation energy of a unit charge, kcal/mol."""
    e, eps0, kb, na = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 6.02214076e23
    lb0 = e * e / (4.0 * math.pi * eps0 * kb * temperature) * 1e10
    kt = kb * na * temperature / 4184.0
    return -(lb0 / (2.0 * radius)) * (1.0 / pdie - 1.0 / sdie) * kt


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ._console import utf8_stdio

    utf8_stdio()
    ap = argparse.ArgumentParser(prog="python -m pbsolve.test_installation")
    ap.add_argument("--platform", action="append", help="platform(s) to test (default: all available)")
    ap.add_argument("--spacing", type=float, default=0.25)
    args = ap.parse_args(argv)

    info = cuda_info()
    print(f"pbsolve version: {__version__}")
    print(f"CUDA support compiled: {info['compiled']} (CUDA {info['cuda_major'] or '-'})")
    print(f"CUDA available: {info['available']}" + ("" if info["available"] else f"  [{info['reason']}]"))
    platforms = args.platform or available_platforms()
    print(f"Testing platforms: {', '.join(platforms)}\n")

    coords = np.array([[0.1, -0.05, 0.02]])
    charges, radii = np.array([1.0]), np.array([2.0])
    exact = born_energy()
    tolerance = 0.03
    ok = True
    energies: dict[str, float] = {}
    for p in platforms:
        try:
            solver = Solver(platform=p, grid_spacing=args.spacing, fadd=12.0)
            t0 = time.perf_counter()
            res = solver.polar_solvation(coords, charges, radii)
            dt = time.perf_counter() - t0
        except Exception as exc:  # report every failure, keep going
            print(f"{p:>5}: FAILED - {exc}")
            ok = False
            continue
        err = (res.energy - exact) / abs(exact)
        good = abs(err) < tolerance
        ok &= good
        energies[p] = res.energy
        print(
            f"{p:>5}: {'ok' if good else 'INACCURATE':<10} ΔG = {res.energy:9.4f} kcal/mol "
            f"(analytic {exact:.4f}, error {100 * err:+.2f}%), "
            f"{sum(res.iterations)} CG iterations, {dt:.2f} s on {solver.device_name}"
        )

    names = list(energies)
    if len(names) > 1:
        print("\nPlatform agreement:")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rel = abs(energies[a] - energies[b]) / abs(energies[a])
                agree = rel < 1e-8
                ok &= agree
                print(f"  {a} vs {b}: relative difference {rel:.2e} {'ok' if agree else 'MISMATCH'}")

    print("\nAll tests passed." if ok else "\nSome tests FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
