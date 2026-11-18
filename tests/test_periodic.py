"""Periodic lateral boundaries (`period`): the solvated state is a periodic array of the solute in x and y."""

import json
import math

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli

from conftest import synthetic_molecule, write_pqr
from test_membrane import KT, LB, layered_reaction

SLAB = [(-15.0, 15.0, 2.0)]
MEMBRANE = dict(zmem=-15.0, lmem=30.0, mdie=2.0, pdie=2.0)
ION = (np.array([[0.0, 0.0, 0.0]]), np.array([1.0]), np.array([0.0]))


def _ghat(k, z0, layers, sdie, kbar2):
    """Hankel kernel ĝ(k) of the potential of a unit charge at z0 in the stratified medium of
    `layered_reaction` (u(ρ) = ∫ ĝ J0(kρ) k dk), for an array of k > 0."""
    lw = np.sqrt(k * k + kbar2 / sdie)
    y = sdie * lw
    for a, b, e in layers:
        hi = min(b, z0)
        if hi > a:
            r, t = y / (e * k), np.tanh(k * (hi - a))
            y = e * k * (r + t) / (1.0 + r * t)
        if hi == z0:
            break
    below, y = y, -sdie * lw
    for a, b, e in reversed(layers):
        lo = max(a, z0)
        if lo < b:
            r, t = y / (e * k), np.tanh(k * (b - lo))
            y = e * k * (r - t) / (1.0 - r * t)
        if lo == z0:
            break
    return 2.0 * LB / (below - y)


def periodic_reaction(z0, layers, sdie, kbar2, period, mu=0.05, gmax=40.0):
    """Reaction potential (kT/e) at a unit charge at z0 that repeats with `period` in x and y.

    Poisson summation of the images, Σ_R u(R) = (2π/A) Σ_G ĝ(|G|), with the free kernel of the
    charge's own layer split off as a screened one, ĝ_μ = l_B/(ε √(k² + μ²)) ↔ l_B e^{−μr}/(ε r):
    u_rf = (2π/A) Σ_G [ĝ − ĝ_μ] + Σ_{R≠0} l_B e^{−μR}/(ε R) − l_B μ/ε. The reciprocal sum is cut
    at gmax plus its tail l_B μ²/(2ε gmax) (ĝ_free − ĝ_μ ≈ l_B μ²/(2ε k³))."""
    eps0 = next(e for a, b, e in layers if a <= z0 <= b)
    px, py = period
    mx, my = int(gmax * px / (2 * math.pi)), int(gmax * py / (2 * math.pi))
    a, b = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
    g = np.hypot(2 * math.pi * a / px, 2 * math.pi * b / py).ravel()
    g = g[g <= gmax]
    recip = (2 * math.pi / (px * py)) * np.sum(
        _ghat(np.where(g == 0.0, 1e-9, g), z0, layers, sdie, kbar2) - LB / (eps0 * np.sqrt(g * g + mu * mu))
    )
    reach = 40.0 / mu
    a, b = np.meshgrid(np.arange(-int(reach / px) - 1, int(reach / px) + 2), np.arange(-int(reach / py) - 1, int(reach / py) + 2), indexing="ij")
    rr = np.hypot(a * px, b * py).ravel()
    rr = rr[rr > 0]
    real = np.sum(LB * np.exp(-mu * rr) / (eps0 * rr))
    return recip + LB * mu * mu / (2.0 * eps0 * gmax) + real - LB * mu / eps0


def _kbar2():
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15, **MEMBRANE)
    return s.maps(*ION, grid=pbsolve.GridSpec([0.1, -0.05, 0.03], [49] * 3, [150.0] * 3, [24.0] * 3))["kappa_bar_sq"]


def test_the_lattice_reference_is_consistent():
    kbar2 = _kbar2()
    for period in ((30.0, 30.0), (40.0, 52.0)):
        a = periodic_reaction(0.0, SLAB, 78.54, kbar2, period)
        b = periodic_reaction(0.0, SLAB, 78.54, kbar2, period, mu=0.1, gmax=60.0)
        assert a == pytest.approx(b, abs=1e-5)
    # Far-apart images: the isolated charge.
    iso = layered_reaction(0.0, SLAB, 78.54, kbar2)
    assert periodic_reaction(0.0, SLAB, 78.54, kbar2, (200.0, 200.0)) == pytest.approx(iso, abs=1e-6)


@pytest.mark.parametrize("period", [30.0, 60.0])
def test_a_charge_in_a_periodic_membrane_matches_the_lattice_solution(period):
    """A point charge in the middle of a uniform membrane slab and its lateral images (pdie = ε
    there, so the grid self-energy cancels). The images add +1.77 / +0.05 kcal/mol to ΔG at
    P = 30 / 60 Å; measured +3.4e-4 / −8.3e-5 (docs/VALIDATION.md). The coarse grid must resolve
    the slab in z: the images' charge sheet makes the energy track its discrete thickness."""
    kbar2 = _kbar2()
    exact = 0.5 * periodic_reaction(0.0, SLAB, 78.54, kbar2, (period, period)) * KT
    isolated = 0.5 * layered_reaction(0.0, SLAB, 78.54, kbar2) * KT
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15, period=(period, period), **MEMBRANE)
    grid = pbsolve.GridSpec([0.1, -0.05, 0.03], [97] * 3, [period, period, 80.0], [24.0] * 3)
    energy = s.polar_solvation(*ION, grid=grid).energy
    assert energy == pytest.approx(exact, rel=2e-3)
    assert abs(energy - isolated) > 20 * abs(energy - exact)


def _supercell(period, copies, **kw):
    """Potentials at the central copy's atoms: one periodic cell, or a plain box of copies."""
    x, q, r = synthetic_molecule()
    x = x - x.mean(axis=0)
    n, nz = int(round(period / 0.5)), 97
    if copies == 0:
        s = pbsolve.Solver(platform="CPU", period=(period, period), **kw)
        grid = pbsolve.GridSpec([0.0] * 3, [n + 1, n + 1, nz], [period, period, 48.0])
        xs, qs, rs = x, q, r
    else:
        m = 2 * copies + 1
        s = pbsolve.Solver(platform="CPU", **kw)
        grid = pbsolve.GridSpec([0.0] * 3, [m * n + 1, m * n + 1, nz], [m * period, m * period, 48.0])
        shifts = [(a * period, b * period, 0.0) for a in range(-copies, copies + 1) for b in range(-copies, copies + 1)]
        xs = np.concatenate([x + np.array(t) for t in shifts])
        qs, rs = np.tile(q, len(shifts)), np.tile(r, len(shifts))
    u, origin, spacing, _ = s.potential(xs, qs, rs, grid=grid)
    f = (x - np.array(origin)) / np.array(spacing)
    i0 = np.floor(f).astype(int)
    w = f - i0
    out = np.zeros(len(x))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ww = np.prod([w[:, a] if d else 1.0 - w[:, a] for a, d in enumerate((dx, dy, dz))], axis=0)
                out += ww * u[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def test_a_periodic_membrane_cell_is_the_limit_of_supercells():
    """Same nodes in the cell and the supercells: the potential at the central copy approaches the
    periodic one as more copies are added (in the membrane the images decay over ~L/π ≈ 10 Å).
    Measured 2.58 → 0.24 kT/e for 3×3 → 5×5 copies with P = 24 Å."""
    kw = dict(ionic_strength=0.15, tol=1e-12, **MEMBRANE)
    periodic = _supercell(24.0, 0, **kw)
    d3 = np.abs(_supercell(24.0, 1, **kw) - periodic).max()
    d5 = np.abs(_supercell(24.0, 2, **kw) - periodic).max()
    assert d5 < d3 / 5.0 and d5 < 0.5, (d3, d5)


def test_the_periodic_potential_and_grid():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, period=(30.0, 34.0))
    g = s.make_grid(x, r)
    assert list(g.coarse_length[:2]) == [30.0, 34.0]
    res = s.polar_solvation(x, q, r)
    assert list(res.grid.coarse_length[:2]) == [30.0, 34.0]
    plain = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    assert res.energy != plain.polar_solvation(x, q, r, grid=g).energy
    # In water, images 80 Å away are screened away: the plain box on the same grid (9e-9).
    wide = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, period=(80.0, 80.0))
    far = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    gw = wide.make_grid(x, r)
    assert wide.polar_solvation(x, q, r).energy == pytest.approx(far.polar_solvation(x, q, r, grid=gw).energy, rel=1e-7)
    # The finest grid spans the period: its last x and y planes repeat the first.
    one = pbsolve.GridSpec(g.center, [41, 45, 41], [30.0, 34.0, 30.0])
    u, *_ = s.potential(x, q, r, grid=one)
    assert np.array_equal(u[-1], u[0]) and np.array_equal(u[:, -1], u[:, 0])
    u_ref, *_ = s.potential(x, q, r, grid=one, reference=True)
    assert not np.array_equal(u_ref[-1], u_ref[0])


def test_invalid_periods():
    x, q, r = synthetic_molecule()
    for kw, match in (
        (dict(period=(30.0, 30.0)), "ions"),
        (dict(period=(30.0, -1.0), ionic_strength=0.1), "positive"),
        (dict(period=(30.0, 30.0), ionic_strength=0.1, bcfl="sdh"), "mdh"),
        (dict(period=(30.0, 30.0), ionic_strength=0.1, bcfl="zero"), "mdh"),
        (dict(period=(30.0, 30.0), ionic_strength=0.1, surface="gauss"), "gauss"),
        (dict(period=(30.0, 30.0), ionic_strength=0.1, equation="npbe"), "lpbe"),
        (dict(period=(30.0, 30.0), ionic_strength=0.1, coulomb="analytic"), "lpbe"),
    ):
        with pytest.raises(ValueError, match=match):
            pbsolve.Solver(**kw)
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.1, period=(9.0, 40.0))
    with pytest.raises(ValueError, match="periodic boundary along x"):
        s.polar_solvation(x, q, r)
    with pytest.raises(ValueError, match="exclusion"):
        pbsolve.Solver(platform="CPU", ionic_strength=0.1, period=(30.0, 30.0), zmem=-5.0, membrane_exclusion=(14.9, 1.0)).polar_solvation(x, q, r)
    ok = pbsolve.Solver(platform="CPU", ionic_strength=0.1, period=(40.0, 40.0), surface="spl2")
    with pytest.raises(ValueError, match="periodic"):
        ok.polar_forces(x, q, r)
    with pytest.raises(ValueError, match="periodic"):
        pbsolve.apbs.elec_from_solver(ok, ok.make_grid(x, r), "x")


def test_parameters_round_trip_and_the_command_line(tmp_path, capsys):
    s = pbsolve.Solver(ionic_strength=0.15, period=(40.0, 44.0), bcfl="mdh-fast")
    p = s.params
    assert p["period"] == (40.0, 44.0)
    params = dict(p)
    params.pop("ionic_strength")
    assert pbsolve.Solver(**params).params == p
    assert pbsolve.Solver().params["period"] is None
    x, q, r = synthetic_molecule()
    pqr = tmp_path / "m.pqr"
    write_pqr(pqr, x, q, r)
    args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--json", "--ionic-strength", "0.15"]
    assert cli(args + ["--period", "30", "34"]) == 0
    out = json.loads(capsys.readouterr().out)
    mol = pbsolve.io.load(str(pqr))
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.6, ionic_strength=0.15, period=(30.0, 34.0))
    assert out["polar_kcal_mol"] == py.polar_solvation(mol.coords, mol.charges, mol.radii).energy
    assert cli(args) == 0
    assert json.loads(capsys.readouterr().out)["polar_kcal_mol"] != out["polar_kcal_mol"]


@pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")
@pytest.mark.parametrize("extra", [dict(), dict(zmem=-6.0, lmem=12.0), dict(zmem=-8.0, lmem=16.0, membrane_profile="greene2019", membrane_fill=True)])
def test_cuda_matches_cpu(extra):
    """Measured ≤ 5e-14 relative, same iterations (2026-11-18)."""
    x, q, r = synthetic_molecule()
    kw = dict(grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, period=(30.0, 32.0), **extra)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(x, q, r)
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(x, q, r)
    assert gpu.platform == "CUDA"
    assert gpu.iterations == cpu.iterations
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-12)
