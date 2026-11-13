#!/usr/bin/env python3
"""Observed convergence order of ΔG_polar against analytic references.

    python scripts/validate_analytic.py [--platform CPU] [--placements 6] [--surface mol]

Born ion and Kirkwood spheres (`pbsolve.analytic.kirkwood_energy`). With a
sharp dielectric boundary the error depends on where the surface cuts the
grid, so every spacing is averaged over random placements of the molecule on
a fixed grid; the order is the slope of log|mean error| against log h.
Errors are (ΔG − ΔG_analytic) / |ΔG_analytic|, as in docs/VALIDATION.md.
"""
import argparse

import numpy as np

import pbsolve
from pbsolve.analytic import kirkwood_energy

THREE = (np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]]), np.array([1.0, -1.0, 0.5]))
NEAR = (np.array([[4.0, 0.0, 0.0], [0.0, -1.5, 3.5]]), np.array([1.0, 1.0]))
CASES = [
    # name, sphere radius, (positions, charges), solver/reference settings
    ("Born ion, a = 2 Å", 2.0, (np.zeros((1, 3)), np.array([1.0])), dict(fadd=12.0)),
    ("Kirkwood, 3 charges, a = 5 Å", 5.0, THREE, dict()),
    ("  same, 0.15 M", 5.0, THREE, dict(ionic_strength=0.15)),
    ("  same, 0.15 M, pdie 4", 5.0, THREE, dict(ionic_strength=0.15, pdie=4.0)),
    ("Kirkwood, 2 charges 1–1.2 Å inside, 0.15 M", 5.0, NEAR, dict(ionic_strength=0.15)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--placements", type=int, default=6)
    ap.add_argument("--surface", default="mol")
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.5, 0.35, 0.25, 0.2])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    hs = a.spacings
    print("| Case | ΔG analytic (kcal/mol) | " + " | ".join(f"h = {h}" for h in hs) + " | order |")
    print("|---|---|" + "---|" * len(hs) + "---|")
    for name, radius, (pos, q), kw in CASES:
        ref_kw = {k: v for k, v in kw.items() if k in ("pdie", "ionic_strength")}
        exact = kirkwood_energy(pos, q, radius, **ref_kw)
        coords, charges = np.vstack([[0.0, 0.0, 0.0], pos]), np.r_[0.0, q]
        radii = np.r_[radius, np.zeros(len(q))]
        cells, means = [], []
        for h in hs:
            s = pbsolve.Solver(platform=a.platform, grid_spacing=h, surface=a.surface, **kw)
            grid = s.make_grid(coords, radii)
            errs = []
            for k in range(a.placements):
                shift = rng.uniform(0.0, h, 3) if k else np.zeros(3)
                e = s.polar_solvation(coords + shift, charges, radii, grid=grid).energy
                errs.append((e - exact) / abs(exact))
            errs = 100.0 * np.array(errs)
            means.append(errs.mean())
            cells.append(f"{errs.mean():+.2f} % ({errs.min():+.2f} … {errs.max():+.2f})")
        m = np.array(means)
        order = f"{np.polyfit(np.log(hs), np.log(np.abs(m)), 1)[0]:.2f}" if np.all(m * m[0] > 0) else "n/a"
        print(f"| {name} | {exact:.4f} | " + " | ".join(cells) + f" | {order} |", flush=True)


if __name__ == "__main__":
    main()
