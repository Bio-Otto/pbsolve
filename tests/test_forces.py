"""Forces on the atoms (`Solver.polar_forces`) and the spline surfaces they need."""

import json

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main
from pbsolve.analytic import kirkwood_energy

from conftest import synthetic_molecule

SHIFTS = np.array([[0.1, 0.4, 0.7], [0.6, 0.2, 0.9], [0.3, 0.8, 0.5], [0.9, 0.6, 0.1]])


def placements(solver, coords, radii, h):
    """The automatic grid moved by fractions of a cell (four placements)."""
    g0 = solver.make_grid(coords, radii)
    return [pbsolve.GridSpec(list(np.asarray(g0.center) + d * h), g0.dims, g0.coarse_length, g0.fine_length) for d in SHIFTS]


def test_polar_forces_api():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, surface="spl2")
    f = s.polar_forces(x, q, r)
    assert isinstance(f, pbsolve.PolarForces) and isinstance(f.result, pbsolve.PolarResult)
    for a in (f.forces, f.qf, f.db, f.ib):
        assert a.shape == (len(q), 3) and np.isfinite(a).all()
    assert np.array_equal(f.forces, f.qf + f.db + f.ib)
    # The same calculation as polar_solvation, bit for bit.
    assert f.energy == s.polar_solvation(x, q, r).energy == f.result.energy
    assert np.abs(f.db).max() > 0 and np.abs(f.ib).max() > 0
    assert "PolarForces(energy=" in repr(f)


def test_forces_need_a_smooth_surface_and_a_reference_state():
    x, q, r = synthetic_molecule()
    kw = dict(platform="CPU", grid_spacing=0.6, fadd=10.0)
    for bad in (dict(surface="mol"), dict(surface="vdw"), dict(surface="harmonic")):
        with pytest.raises(ValueError, match="spl2"):
            pbsolve.Solver(**kw, **bad).polar_forces(x, q, r)
    with pytest.raises(ValueError, match="coulomb"):
        pbsolve.Solver(**kw, surface="spl2", coulomb="analytic").polar_forces(x, q, r)
    # An atom whose boundary force reaches the edge of the grid: an error, not a silent zero (APBS).
    tight = pbsolve.GridSpec(list(x.mean(0)), [33, 33, 33], [14.0] * 3)
    with pytest.raises(ValueError, match="enlarge the fine grid"):
        pbsolve.Solver(**kw, surface="spl2").polar_forces(x, q, r, grid=tight)


def test_forces_are_minus_the_gradient_of_the_energy():
    # One level and zero boundary values: the exact gradient of the discrete energy, in kcal/mol/Å.
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15, surface="spl2", chgm="spl2", bcfl="zero", tol=1e-12)
    grid = pbsolve.GridSpec(list(x.mean(0) + 0.07), [49, 49, 49], [22.0] * 3)
    f = s.polar_forces(x, q, r, grid=grid)
    for n, axis in [(0, 0), (11, 1), (20, 2)]:  # atoms with radii

        def energy(d):
            y = x.copy()
            y[n, axis] += d
            return s.polar_solvation(y, q, r, grid=grid).energy

        central = lambda d: -(energy(d) - energy(-d)) / (2 * d)
        fd = (4 * central(5e-5) - central(1e-4)) / 3
        assert abs(f.forces[n, axis] - fd) < 1e-7 * np.abs(f.forces).max(), (n, axis, f.forces[n, axis], fd)


@pytest.mark.parametrize("surface", ["spl2", "spl4"])
def test_charge_in_a_dielectric_sphere(surface):
    # A unit charge 2.5 Å off the centre of an uncharged sphere (radius 6 Å, pdie 2):
    # * the reaction field pushes the charge outwards with Kirkwood's force for the
    #   sphere's effective radius, which the Born energy of the same surface gives
    #   (the switch moves the boundary inwards; measured 0.07-0.08 %);
    # * the dielectric-boundary force pushes the sphere the other way (momentum is
    #   conserved; the grid leaves ~1 %).
    a, s, pdie, sdie, h = 6.0, 2.5, 2.0, 78.54, 0.35
    # The default box (fadd 20 Å): with fadd 10 the boundary shifts the force by 0.6 %.
    sv = pbsolve.Solver(platform="CPU", grid_spacing=h, pdie=pdie, sdie=sdie, surface=surface, tol=1e-10)
    c = np.array([0.13, -0.08, 0.05])
    xb, qb, rb = np.array([c]), np.array([1.0]), np.array([a])
    born = np.mean([sv.polar_solvation(xb, qb, rb, grid=g).energy for g in placements(sv, xb, rb, h)])
    lb = pbsolve.bjerrum_length(298.15) * pbsolve.kt_kcal_per_mol(298.15)
    a_eff = -lb * (1 / pdie - 1 / sdie) / (2 * born)
    x, q, r = np.array([c, c + [s, 0, 0]]), np.array([0.0, 1.0]), np.array([a, 0.0])
    f = np.array([sv.polar_forces(x, q, r, grid=g).forces for g in placements(sv, x, r, h)])

    def kirkwood(radius, d=1e-4):
        e = lambda t: kirkwood_energy([[t, 0, 0]], [1.0], radius, pdie=pdie, sdie=sdie)
        return -(e(s + d) - e(s - d)) / (2 * d)

    charge, sphere = f[:, 1, 0].mean(), f[:, 0, 0].mean()
    assert 5.6 < a_eff < 5.9
    assert charge == pytest.approx(kirkwood(a_eff), rel=5e-3)
    assert charge > 1.1 * kirkwood(a)  # the sharp sphere would be 14 % off
    assert abs(charge + sphere) < 0.03 * charge
    assert np.abs(f[:, :, 1:]).max() < 0.03 * charge  # no force across the axis


def test_spline_surfaces():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.1, surface="spl4", swin=0.4)
    assert (s.params["surface"], s.params["swin"]) == ("spl4", 0.4)
    m = s.maps(x, q, r)
    eps = m["eps_x"]
    between = (eps > 1.0 + 1e-9) & (eps < 78.54 - 1e-9)
    assert between.sum() > 100  # a smooth boundary, not a jump
    k = m["kappa_sq"] / m["kappa_bar_sq"]
    assert k.min() == 0.0 and k.max() == 1.0 and ((k > 0) & (k < 1)).sum() > 100
    for bad in (dict(swin=0.0), dict(swin=-1.0), dict(interface="fractional")):
        with pytest.raises(ValueError):
            pbsolve.Solver(platform="CPU", surface="spl2", **bad)


def test_command_line_surface(capsys):
    pqr = str(__import__("pathlib").Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")
    args = ["solvation", pqr, "--platform", "CPU", "--spacing", "0.9", "--json"]
    energies = []
    for extra in (["--surface", "spl2"], ["--surface", "spl2", "--swin", "0.5"]):
        assert main(args + extra) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    assert energies[0] != energies[1]


@pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")
def test_forces_on_cuda_equal_the_cpu():
    x, q, r = synthetic_molecule()
    kw = dict(grid_spacing=0.5, fadd=10.0, ionic_strength=0.15, surface="spl2", chgm="spl2", equation="npbe")
    cpu = pbsolve.Solver(platform="CPU", **kw)
    grid = cpu.make_grid(x, r)
    a = cpu.polar_forces(x, q, r, grid=grid)
    b = pbsolve.Solver(platform="CUDA", **kw).polar_forces(x, q, r, grid=grid)
    assert np.abs(a.forces - b.forces).max() < 1e-9 * np.abs(a.forces).max()
    assert b.result.platform == "CUDA"
