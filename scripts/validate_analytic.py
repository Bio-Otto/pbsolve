#!/usr/bin/env python3
"""Observed convergence order of ΔG_polar against analytic references.

    python scripts/validate_analytic.py [--platform CPU] [--placements 6] [--surface mol] [--chgm spl0]

Born ion, Kirkwood spheres (`pbsolve.analytic.kirkwood_energy`) and two
spheres (`two_sphere_energy`). With a sharp dielectric boundary the error
depends on where the surface cuts the grid, so every spacing is averaged over
random placements of the molecule on a fixed grid; the order is the slope of
log|mean error| against log h. Errors are (ΔG − ΔG_exact) / |ΔG_exact|, as
in docs/VALIDATION.md; for the association of two spheres (ΔΔG = complex −
sphere 1 − sphere 2, all on the complex's grid) they are in kcal/mol.
"""
import argparse

import numpy as np

import pbsolve
from pbsolve.analytic import kirkwood_energy, two_sphere_energy

ORIGIN = np.zeros(3)
BORN = (ORIGIN, 2.0, np.zeros((1, 3)), np.array([1.0]))
THREE = (ORIGIN, 5.0, np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]]), np.array([1.0, -1.0, 0.5]))
NEAR = (ORIGIN, 5.0, np.array([[4.0, 0.0, 0.0], [0.0, -1.5, 3.5]]), np.array([1.0, 1.0]))
C2 = np.array([3.0, -4.0, 15.0])
SECOND = (C2, 4.0, C2 + np.array([[0.0, -1.0, 1.5], [1.2, 0.3, -0.4]]), np.array([-1.0, 0.7]))
SALT, PDIE4 = dict(ionic_strength=0.15), dict(ionic_strength=0.15, pdie=4.0)
CASES = [
    # name, spheres (centre, radius, charge positions, charges), settings, association?
    ("Born ion, a = 2 Å", [BORN], dict(fadd=12.0), False),
    ("Kirkwood, 3 charges, a = 5 Å", [THREE], {}, False),
    ("  same, 0.15 M", [THREE], SALT, False),
    ("  same, 0.15 M, pdie 4", [THREE], PDIE4, False),
    ("Kirkwood, 2 charges 1–1.2 Å inside, 0.15 M", [NEAR], SALT, False),
    ("Two spheres (5 + 4 Å, 15.8 Å apart)", [THREE, SECOND], {}, False),
    ("  same, 0.15 M", [THREE, SECOND], SALT, False),
    ("  association ΔΔG, 0 M (kcal/mol)", [THREE, SECOND], {}, True),
    ("  association ΔΔG, 0.15 M (kcal/mol)", [THREE, SECOND], SALT, True),
]


def exact(spheres, kw):
    kw = {k: v for k, v in kw.items() if k in ("pdie", "ionic_strength")}
    if len(spheres) == 1:
        c, a, pos, q = spheres[0]
        return kirkwood_energy(pos, q, a, center=c, **kw)
    pos = np.vstack([s[2] for s in spheres])
    q = np.concatenate([s[3] for s in spheres])
    return two_sphere_energy(pos, q, [s[0] for s in spheres], [s[1] for s in spheres], **kw)


def atoms(spheres):
    """One uncharged atom per sphere, then the charges as atoms of radius 0."""
    coords = np.vstack([[s[0] for s in spheres], *[s[2] for s in spheres]])
    charges = np.concatenate([np.zeros(len(spheres)), *[s[3] for s in spheres]])
    radii = np.concatenate([[s[1] for s in spheres], *[np.zeros(len(s[3])) for s in spheres]])
    owner = np.concatenate([np.arange(len(spheres)), *[np.full(len(s[3]), i) for i, s in enumerate(spheres)]])
    return coords, charges, radii, owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--placements", type=int, default=6)
    ap.add_argument("--surface", default="mol")
    ap.add_argument("--chgm", default="spl0", choices=["spl0", "spl2", "spl4"])
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.5, 0.35, 0.25, 0.2])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    hs = a.spacings
    print("| Case | exact (kcal/mol) | " + " | ".join(f"h = {h}" for h in hs) + " | order |")
    print("|---|---|" + "---|" * len(hs) + "---|")
    for name, spheres, kw, association in CASES:
        ref = exact(spheres, kw)
        if association:
            ref -= sum(exact([s], kw) for s in spheres)
        coords, charges, radii, owner = atoms(spheres)
        parts = [np.ones(len(owner), bool)] + ([owner == i for i in range(len(spheres))] if association else [])
        cells, means = [], []
        for h in hs:
            s = pbsolve.Solver(platform=a.platform, grid_spacing=h, surface=a.surface, chgm=a.chgm, **kw)
            grid = s.make_grid(coords, radii)
            errs = []
            for k in range(a.placements):
                x = coords + (rng.uniform(0.0, h, 3) if k else 0.0)
                e = [s.polar_solvation(x[m], charges[m], radii[m], grid=grid).energy for m in parts]
                value = e[0] - sum(e[1:])
                errs.append(value - ref if association else 100.0 * (value - ref) / abs(ref))
            errs = np.array(errs)
            means.append(errs.mean())
            unit, d = ("", 3) if association else (" %", 2)
            cells.append(f"{errs.mean():+.{d}f}{unit} ({errs.min():+.{d}f} … {errs.max():+.{d}f})")
        m = np.array(means)
        order = f"{np.polyfit(np.log(hs), np.log(np.abs(m)), 1)[0]:.2f}" if np.all(m * m[0] > 0) else "n/a"
        print(f"| {name} | {ref:.4f} | " + " | ".join(cells) + f" | {order} |", flush=True)


if __name__ == "__main__":
    main()
