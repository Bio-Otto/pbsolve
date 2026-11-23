"""Exact gradients of ΔG_polar (`Solver.polar_gradient`, ROADMAP 4.20)."""

import numpy as np
import pytest

import pbsolve

from conftest import synthetic_molecule

KW = dict(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, pdie=2.0, tol=1e-12)


def test_polar_gradient_api():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(**KW)
    g = s.polar_gradient(x, q, r)
    assert isinstance(g, pbsolve.PolarGradient) and isinstance(g.result, pbsolve.PolarResult)
    assert g.charges.shape == q.shape and np.isfinite(g.charges).all()
    assert g.pdie is None and g.sdie is None
    # The same energy as polar_solvation, bit for bit.
    e = s.polar_solvation(x, q, r)
    assert g.energy == e.energy == g.result.energy
    assert np.array_equal(g.result.atom_energies, e.atom_energies)
    assert "PolarGradient(energy=" in repr(g)
    d = s.polar_gradient(x, q, r, dielectric_constants=True)
    assert np.array_equal(d.charges, g.charges)
    assert d.pdie > 0 and d.sdie < 0  # a weaker contrast costs solvation energy


def test_charge_derivatives_equal_central_differences():
    x, q, r = synthetic_molecule()
    for extra in (dict(), dict(bcfl="sdh", chgm="spl4", surface="spl2")):
        s = pbsolve.Solver(**KW, **extra)
        grid = s.make_grid(x, r)
        g = s.polar_gradient(x, q, r, grid=grid)
        for i in (0, 5, 11):  # 5 has radius 0
            h = 1e-3
            qp, qm = q.copy(), q.copy()
            qp[i] += h
            qm[i] -= h
            fd = (s.polar_solvation(x, qp, r, grid=grid).energy - s.polar_solvation(x, qm, r, grid=grid).energy) / (2 * h)
            assert g.charges[i] == pytest.approx(fd, abs=1e-7 * np.abs(g.charges).max()), (extra, i)
        # ΔG is quadratic in the charges.
        assert np.dot(q, g.charges) == pytest.approx(2 * g.energy, rel=1e-9)


def test_dielectric_derivatives_equal_central_differences():
    x, q, r = synthetic_molecule()
    for extra in (dict(), dict(interface="fractional", bcfl="mdh-fast"), dict(surface="spl4")):
        kw = {**KW, **extra}
        s = pbsolve.Solver(**kw)
        grid = s.make_grid(x, r)
        g = s.polar_gradient(x, q, r, grid=grid, dielectric_constants=True)
        for name in ("pdie", "sdie"):
            v = kw.get(name, 78.54)
            h = 1e-4 * v
            up = pbsolve.Solver(**{**kw, name: v + h}).polar_solvation(x, q, r, grid=grid).energy
            down = pbsolve.Solver(**{**kw, name: v - h}).polar_solvation(x, q, r, grid=grid).energy
            assert getattr(g, name) == pytest.approx((up - down) / (2 * h), rel=1e-6), (extra, name)


def test_per_atom_dielectrics_give_charge_derivatives():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(**{**KW, "ionic_strength": 0.0})
    eps = np.where(np.arange(len(q)) < 15, 2.0, 4.0)
    g = s.polar_gradient(x, q, r, dielectrics=eps)
    assert g.energy == s.polar_solvation(x, q, r, dielectrics=eps).energy
    assert np.dot(q, g.charges) == pytest.approx(2 * g.energy, rel=1e-9)
    with pytest.raises(ValueError, match="per-atom dielectrics"):
        s.polar_gradient(x, q, r, dielectrics=eps, dielectric_constants=True)


def test_unsupported_options_are_refused():
    x, q, r = synthetic_molecule()
    for bad, match in (
        (dict(equation="npbe"), "linearized"),
        (dict(coulomb="analytic"), "coulomb"),
        (dict(zmem=0.0), "membrane"),
    ):
        with pytest.raises(ValueError, match=match):
            pbsolve.Solver(**{**KW, **bad}).polar_gradient(x, q, r)
    r = np.where(r > 0, r, 1.1)  # gauss refuses charges of radius 0 outside its contour
    for bad, match in ((dict(surface="gauss"), "surface"), (dict(surface="harmonic"), "surface"), (dict(bcfl="sdh"), "sdh")):
        s = pbsolve.Solver(**{**KW, **bad})
        assert np.isfinite(s.polar_gradient(x, q, r).charges).all()
        with pytest.raises(ValueError, match=match):
            s.polar_gradient(x, q, r, dielectric_constants=True)
