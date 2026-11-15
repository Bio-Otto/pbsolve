"""Finite-size ions (`equation="smpbe"`), and the planar double layer for `npbe`."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import pbsolve
from pbsolve import apbs
from pbsolve.__main__ import main

PER_MOLAR = 6.02214076e23 * 1e-27  # ions per Å³ at 1 mol/L
PQR = str(Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")


def planar_profile(xs, u0, kappa2, phi0):
    """Exact potential (kT/e) of a planar double layer in a 1:1 salt, u(0) = u0.

    From the first integral u'² = F(u): Gouy–Chapman for point ions
    (F = 4κ² sinh²(u/2)), and (2κ²/φ0) ln(1 − φ0 + φ0 cosh u) for lattice-gas
    ions with bulk volume fraction φ0 [Borukhov et al. 1997; Kilic et al. 2007].
    """
    if phi0 > 0:
        F = lambda v: 2 * kappa2 / phi0 * np.log1p(2 * phi0 * np.sinh(v / 2) ** 2)  # cosh v − 1 without cancellation
    else:
        F = lambda v: 4 * kappa2 * np.sinh(v / 2) ** 2
    x_of = lambda u: quad(lambda v: F(v) ** -0.5, u, u0, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    return np.array([u0 if x == 0 else brentq(lambda u: x_of(u) - x, 1e-12, u0, xtol=1e-15) for x in xs])


def planar_error(solver, h, phi0, u0=6.0, length=24.0):
    """Max error of the solver's interior potential on a slab whose boundary carries the exact profile."""
    nx = int(round(length / h)) + 1
    dims = (nx, 5, 5)
    kbar2 = solver.maps(np.zeros((1, 3)), np.zeros(1), np.zeros(1), grid=pbsolve.GridSpec([0, 0, 0], [9] * 3, [8.0] * 3))["kappa_bar_sq"]
    sdie = solver.params["sdie"]
    u = planar_profile(np.arange(nx) * h, u0, kbar2 / sdie, phi0)
    eps = np.full(dims, sdie)
    g = np.broadcast_to(u[:, None, None], dims).copy()
    v, *_ = solver.solve_maps(eps, eps, eps, np.full(dims, kbar2), np.zeros(dims), g, [0.0] * 3, [h] * 3)
    return np.abs(v[1:-1, 2, 2] - u[1:-1]).max()


@pytest.mark.parametrize("equation, size", [("npbe", 0.0), ("smpbe", 3.0), ("smpbe", 7.0)])
def test_planar_double_layer(equation, size):
    # u(0) = 6 kT/e at 0.15 M: strongly nonlinear, and crowded for 7 Å ions (D up to 13).
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15, equation=equation, ion_size=size, tol=1e-12)
    phi0 = 2 * 0.15 * PER_MOLAR * size**3
    coarse, fine = planar_error(s, 0.5, phi0), planar_error(s, 0.25, phi0)
    # measured: 1.1e-2 / 1.8e-3 (npbe), 3.4e-3 / 3.7e-4 (3 Å), 5.5e-5 / 5.0e-6 (7 Å); order 2.6–3.4
    assert fine < 2e-3 and coarse / fine > 4.0


def sphere(charge=-20.0, radius=10.0):
    return np.array([[0.11, -0.07, 0.05]]), np.array([charge]), np.array([radius])


def test_finite_size_penalty_grows_as_a_cubed():
    # A low-dielectric sphere with a central charge in 0.1 M salt, without a
    # Stern layer [Silalahi et al. 2010]: the entropic cost of finite ion
    # size is positive and grows as a³ while the ions are dilute (their eq. 11).
    x, q, r = sphere()
    kw = dict(platform="CPU", grid_spacing=0.6, pdie=4.0, ionic_strength=0.1, ion_radius=0.0, surface="vdw")
    grid = pbsolve.Solver(**kw).make_grid(x, r)
    point = pbsolve.Solver(equation="npbe", **kw).polar_solvation(x, q, r, grid=grid).energy
    cost = [
        pbsolve.Solver(equation="smpbe", ion_size=a, **kw).polar_solvation(x, q, r, grid=grid).energy - point
        for a in (1.0, 2.0)
    ]
    assert cost[0] > 0 and cost[1] / cost[0] == pytest.approx(8.0, rel=0.02)


def test_counterions_saturate():
    # At a −50 e sphere the 8 Å ions fill their cells (3.243 of 3.243 M); point ions would reach 84 M.
    x, q, r = sphere(-50.0)
    kw = dict(platform="CPU", grid_spacing=0.6, pdie=4.0, ionic_strength=0.1, ion_radius=0.0, surface="vdw")
    s = pbsolve.Solver(equation="smpbe", ion_size=8.0, **kw)
    grid = s.make_grid(x, r)
    u, *_ = s.potential(x, q, r, grid=grid)
    u = u[s.maps(x, q, r, grid=grid)["kappa_sq"] > 0]  # where the ions are
    phi = 0.1 * PER_MOLAR * 8.0**3
    counterions = 0.1 * np.exp(-u) / (1 - 2 * phi + 2 * phi * np.cosh(u))
    cell = 1 / (PER_MOLAR * 8.0**3)
    assert 0.5 * cell < counterions.max() <= cell


def test_parameters():
    assert pbsolve.Solver(platform="CPU").params["ion_size"] == 0.0
    s = pbsolve.Solver(platform="CPU", equation="smpbe", ion_size=6.0, ionic_strength=0.15)
    assert (s.params["equation"], s.params["ion_size"]) == ("smpbe", 6.0)
    assert pbsolve.estimate._with_spacing(s, 0.7).params["ion_size"] == 6.0
    with pytest.raises(ValueError, match="ion_size > 0"):
        pbsolve.Solver(platform="CPU", equation="smpbe")
    with pytest.raises(ValueError, match="applies to equation 'smpbe'"):
        pbsolve.Solver(platform="CPU", equation="npbe", ion_size=6.0)
    with pytest.raises(ValueError, match="fill the solvent"):
        pbsolve.Solver(platform="CPU", equation="smpbe", ion_size=12.0, ionic_strength=1.0)
    with pytest.raises(ValueError, match="coulomb"):
        pbsolve.Solver(platform="CPU", equation="smpbe", ion_size=6.0, coulomb="analytic")
    x, q, r = sphere()
    with pytest.raises(ValueError, match="disables smpbe"):
        apbs.elec_from_solver(s, s.make_grid(x, r), "solvated")


def test_command_line(capsys):
    args = ["solvation", PQR, "--platform", "CPU", "--spacing", "0.9", "--ionic-strength", "0.15", "--json"]
    energies = []
    for extra in (["--equation", "npbe"], ["--equation", "smpbe", "--ion-size", "6"]):
        assert main(args + extra) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    # 1YCR is weakly charged: finite size costs little, but it costs.
    assert energies[0] < energies[1] < energies[0] + 1.0
