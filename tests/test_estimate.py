"""Numerical error estimates (`pbsolve.estimate_error`)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main
from pbsolve.analytic import kirkwood_energy, two_sphere_energy

# A 5 Å sphere with three charges and a 4 Å sphere with two (as in test_analytic).
A, A2 = 5.0, 4.0
QPOS = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]])
QVAL = np.array([1.0, -1.0, 0.5])
C2 = np.array([3.0, -4.0, 15.0])
QPOS2 = C2 + np.array([[0.0, -1.0, 1.5], [1.2, 0.3, -0.4]])
QVAL2 = np.array([-1.0, 0.7])


def sphere():
    return np.vstack([[0.0, 0.0, 0.0], QPOS]), np.r_[0.0, QVAL], np.r_[A, 0.0, 0.0, 0.0]


def pair():
    coords = np.vstack([[0.0, 0.0, 0.0], C2, QPOS, QPOS2])
    charges = np.r_[0.0, 0.0, QVAL, QVAL2]
    radii = np.r_[A, A2, np.zeros(5)]
    first = np.r_[True, False, [True] * 3, [False] * 2]
    return coords, charges, radii, first


def test_placements_are_averaged():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, ionic_strength=0.15)
    coords, charges, radii = sphere()
    est = pbsolve.estimate_error(s, coords, charges, radii, placements=4)
    # The first placement is the automatic grid itself.
    assert est.energies[0] == s.polar_solvation(coords, charges, radii).energy
    assert est.placements == 4 and len(set(est.energies)) == 4
    # Every placement is the automatic grid moved by its offset, at most half a spacing.
    base = s.make_grid(coords, radii)
    assert est.offsets[0] == [0.0, 0.0, 0.0]
    for offset, energy in zip(est.offsets, est.energies):
        assert all(abs(d) <= h / 2 for d, h in zip(offset, base.fine_spacing))
        grid = pbsolve.GridSpec(list(np.add(base.center, offset)), base.dims, base.coarse_length, base.fine_length)
        assert s.polar_solvation(coords, charges, radii, grid=grid).energy == energy
    assert est.energy == pytest.approx(np.mean(est.energies), rel=1e-15)
    assert est.sd == pytest.approx(np.std(est.energies, ddof=1), rel=1e-12)
    assert est.sem == pytest.approx(est.sd / 2.0, rel=1e-15)
    assert est.spacings == [pytest.approx(np.mean(s.make_grid(coords, radii).fine_spacing))]
    assert est.extrapolated is None and est.discretisation_error is None
    assert "4 placements" in est.format()
    again = pbsolve.estimate_error(s, coords, charges, radii, placements=4)
    assert again.energies == est.energies
    other = pbsolve.estimate_error(s, coords, charges, radii, placements=4, seed=1)
    assert other.energies[0] == est.energies[0] and other.energies[1:] != est.energies[1:]


def test_two_spacings_are_extrapolated_at_first_order():
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15)
    coords, charges, radii = sphere()
    est = pbsolve.estimate_error(s, coords, charges, radii, placements=6, spacings=[0.35, 0.5])
    (h1, h2), (m1, m2) = est.spacings, est.means
    assert h1 > h2 and est.energy == m2
    assert est.extrapolated == pytest.approx(m2 + (m2 - m1) * h2 / (h1 - h2), rel=1e-14)
    assert est.discretisation_error == pytest.approx(est.energy - est.extrapolated, rel=1e-14)
    # Against Kirkwood's solution: the extrapolation removes most of the error
    # (measured: mean −1.8 %, extrapolated +0.7 %), and the estimated error of
    # the mean has the right sign and size.
    exact = kirkwood_energy(QPOS, QVAL, A, ionic_strength=0.15)
    assert abs(est.extrapolated - exact) < 0.5 * abs(est.energy - exact)
    assert 0.5 < (est.energy - exact) / est.discretisation_error < 2.0
    assert est.sem < est.sd < 0.01 * abs(exact)


def test_parts_give_the_association_energy():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6)
    coords, charges, radii, first = pair()
    est = pbsolve.estimate_error(s, coords, charges, radii, placements=3, parts=[first, ~first])
    grid = s.make_grid(coords, radii)
    e = [s.polar_solvation(coords[m], charges[m], radii[m], grid=grid).energy for m in (slice(None), first, ~first)]
    assert est.energies[0] == e[0] - e[1] - e[2]
    by_index = pbsolve.estimate_error(
        s, coords, charges, radii, placements=3, parts=[np.flatnonzero(first), np.flatnonzero(~first)]
    )
    assert by_index.energies == est.energies
    exact = two_sphere_energy(np.vstack([QPOS, QPOS2]), np.r_[QVAL, QVAL2], [[0.0, 0.0, 0.0], C2], [A, A2])
    exact -= kirkwood_energy(QPOS, QVAL, A) + kirkwood_energy(QPOS2 - C2, QVAL2, A2)
    # The mean is consistent with the exact value within its standard error
    # (plus ~0.015 kcal/mol from APBS' sampled surface, see docs/VALIDATION.md).
    assert abs(est.energy - exact) < 3 * est.sem + 0.05


def test_bad_arguments():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6)
    coords, charges, radii = sphere()
    with pytest.raises(ValueError, match="placements"):
        pbsolve.estimate_error(s, coords, charges, radii, placements=1)
    with pytest.raises(ValueError, match="mask"):
        pbsolve.estimate_error(s, coords, charges, radii, parts=[np.array([True, False])])
    with pytest.raises(ValueError, match="indices"):
        pbsolve.estimate_error(s, coords, charges, radii, parts=[np.array([0, 4])])


def test_command_line(capsys):
    pqr = str(Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")
    args = ["solvation", pqr, "--platform", "CPU", "--spacing", "0.9", "--placements", "3", "--extrapolate", "0.7"]
    assert main(args + ["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["placements"] == 3 and out["polar_sd_kcal_mol"] > 0
    assert out["polar_sem_kcal_mol"] == pytest.approx(out["polar_sd_kcal_mol"] / math.sqrt(3))
    h1, h2 = out["extrapolation_spacings"]
    m2 = out["polar_mean_kcal_mol"]
    assert h1 > h2 and math.isfinite(out["polar_extrapolated_kcal_mol"]) and m2 < 0
    assert main(["solvation", pqr, "--platform", "CPU", "--extrapolate", "0.7"]) == 2
    assert "--placements" in capsys.readouterr().err
