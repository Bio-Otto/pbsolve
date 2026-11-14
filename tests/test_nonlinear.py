"""The nonlinear Poisson–Boltzmann equation (`equation="npbe"`)."""

import json
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs
from pbsolve.__main__ import main

PQR = str(Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")
# Three overlapping atoms with mixed charges.
COORDS = np.array([[0.13, -0.21, 0.07], [2.31, 0.52, -0.33], [-1.12, 1.41, 0.88]])
CHARGES = np.array([-3.0, 1.5, -2.0])
RADII = np.array([2.2, 1.8, 1.6])


def solver(**kw):
    return pbsolve.Solver(platform="CPU", grid_spacing=0.4, pdie=2.0, ionic_strength=0.15, tol=1e-10, **kw)


def test_weak_potentials_follow_the_linearized_equation():
    q = 1e-3 * CHARGES
    lin, non = solver(), solver(equation="npbe")
    grid = lin.make_grid(COORDS, RADII)
    u_lin = lin.potential(COORDS, q, RADII, grid=grid)[0]
    u_non = non.potential(COORDS, q, RADII, grid=grid)[0]
    assert np.abs(u_non - u_lin).max() < 1e-8 * np.abs(u_lin).max()


def test_strong_potentials():
    # A -30 e sphere at 0.1 M: the ions screen far more strongly than the linearized equation assumes.
    x, q, r = np.array([[0.11, -0.07, 0.05]]), np.array([-30.0]), np.array([4.0])
    kw = dict(platform="CPU", grid_spacing=0.5, ionic_strength=0.1)
    lin, non = pbsolve.Solver(**kw), pbsolve.Solver(equation="npbe", **kw)
    grid = lin.make_grid(x, r)
    a, b = lin.polar_solvation(x, q, r, grid=grid), non.polar_solvation(x, q, r, grid=grid)
    assert b.energy < a.energy - 90.0  # measured 97.5 kcal/mol more favourable
    assert all(res < 1e-8 for res in b.residuals)
    maps = lin.maps(x, q, r, grid=grid)
    accessible = maps["kappa_sq"] > 0
    u_lin = lin.potential(x, q, r, grid=grid)[0][accessible].min()
    u_non = non.potential(x, q, r, grid=grid)[0][accessible].min()
    assert u_lin < -20.0 and -9.0 < u_non < -6.0  # kT/e at the ions


def test_energies():
    lin, non = solver(), solver(equation="npbe")
    grid = pbsolve.GridSpec(list(COORDS.mean(axis=0)), [41, 41, 41], [40.0] * 3, [16.0] * 3)
    assert len(grid.levels) == 2  # the coarse level adds outer terms to the functional
    a = lin.polar_solvation(COORDS, CHARGES, RADII, grid=grid)
    b = non.polar_solvation(COORDS, CHARGES, RADII, grid=grid)
    # The reference state (pdie, no ions) is linear either way.
    assert b.reference_kt == a.reference_kt
    # The solvated energy is the free-energy functional, which potential() reports too.
    assert non.potential(COORDS, CHARGES, RADII, grid=grid)[3] == pytest.approx(b.solvated_kt, rel=1e-12)
    assert b.energy == pytest.approx(a.energy, rel=0.02)
    # Without ions npbe is lpbe, bit for bit.
    c = pbsolve.Solver(platform="CPU", equation="npbe").polar_solvation(COORDS, CHARGES, RADII)
    d = pbsolve.Solver(platform="CPU").polar_solvation(COORDS, CHARGES, RADII)
    assert c.energy == d.energy


def test_solve_maps_is_nonlinear():
    # One level: solve_maps with the level's maps and boundary reproduces potential().
    non = pbsolve.Solver(platform="CPU", grid_spacing=0.5, ionic_strength=0.15, equation="npbe", tol=1e-11)
    c = list(COORDS.mean(axis=0))
    grid = pbsolve.GridSpec(c, [41, 41, 41], [20.0, 20.0, 20.0])
    u, *_ = non.potential(COORDS, 4 * CHARGES, RADII, grid=grid)
    m = non.maps(COORDS, 4 * CHARGES, RADII, grid=grid)
    v, iters, res = non.solve_maps(m["eps_x"], m["eps_y"], m["eps_z"], m["kappa_sq"], m["charge"], u, m["origin"], m["spacing"])
    assert np.abs(v - u).max() < 1e-8 * np.abs(u).max()
    lin = pbsolve.Solver(platform="CPU", ionic_strength=0.15)
    w, *_ = lin.solve_maps(m["eps_x"], m["eps_y"], m["eps_z"], m["kappa_sq"], m["charge"], u, m["origin"], m["spacing"])
    assert np.abs(w - u).max() > 5.0  # kT/e: the linearized potential near the surface (14 measured)


def test_equation_parameter():
    assert pbsolve.Solver(platform="CPU").params["equation"] == "lpbe"
    s = pbsolve.Solver(platform="CPU", equation="npbe", ionic_strength=0.15)
    assert s.params["equation"] == "npbe"
    assert pbsolve.estimate._with_spacing(s, 0.7).params["equation"] == "npbe"
    with pytest.raises(ValueError, match="equation"):
        pbsolve.Solver(platform="CPU", equation="smpbe")
    with pytest.raises(ValueError, match="coulomb"):
        pbsolve.Solver(platform="CPU", equation="npbe", coulomb="analytic")
    # The APBS input carries the keyword.
    grid = s.make_grid(COORDS, RADII)
    elec = apbs.elec_from_solver(s, grid, "solvated")
    assert elec.pbe == "npbe" and "    npbe\n" in apbs.write_input(["x.pqr"], [elec])
    assert elec.solver(platform="CPU").params["equation"] == "npbe"


def test_command_line(capsys):
    args = ["solvation", PQR, "--platform", "CPU", "--spacing", "0.9", "--ionic-strength", "0.15", "--json"]
    energies = []
    for equation in ("lpbe", "npbe"):
        assert main(args + ["--equation", equation]) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    # 1YCR is weakly charged: the two agree to a few 0.01 kcal/mol.
    assert energies[1] == pytest.approx(energies[0], abs=0.2) and energies[1] != energies[0]
