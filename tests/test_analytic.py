"""Analytic references (`pbsolve.analytic`) and pbsolve against them."""

import math

import numpy as np
import pytest

import pbsolve
from pbsolve.analytic import _reaction_coefficients, kirkwood_energy
from pbsolve.test_installation import born_energy

LB, KT = pbsolve.bjerrum_length(), pbsolve.kt_kcal_per_mol()

# Three charges inside a 5 Å sphere (|r| = 2.0, 2.3, 2.7 Å).
A = 5.0
QPOS = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]])
QVAL = np.array([1.0, -1.0, 0.5])


def _kappa(ionic_strength, sdie=78.54):
    return math.sqrt(8.0 * math.pi * LB * 6.02214076e23 * 1e-27 * ionic_strength / sdie)


def _grounded(pos, q, a, pdie):
    """Image charges of a grounded sphere [Jackson, ch. 2], kcal/mol."""
    rho = np.linalg.norm(pos, axis=1)
    images = pos * (a * a / rho**2)[:, None]
    dist = np.linalg.norm(pos[:, None, :] - images[None, :, :], axis=2)  # field point i, image of j
    return -0.5 * LB / pdie * np.sum(np.outer(q, q) * a / (rho[None, :] * dist)) * KT


def test_kirkwood_reduces_to_born():
    assert kirkwood_energy([[0, 0, 0]], [1.0], 2.0) == pytest.approx(born_energy(), rel=1e-13)
    e = kirkwood_energy([[1.0, 2.0, 3.0]], [1.0], 2.0, center=[1.0, 2.0, 3.0], pdie=4.0, sdie=20.0)
    assert e == pytest.approx(born_energy(pdie=4.0, sdie=20.0), rel=1e-13)


@pytest.mark.parametrize("ion_radius", [0.0, 2.0])
def test_kirkwood_central_charge_in_salt(ion_radius):
    """Born with Debye–Hückel screening beyond the Stern radius b = a + ion_radius."""
    a, pdie, sdie, ionic = 5.0, 2.0, 78.54, 0.15
    k, b = _kappa(ionic), a + ion_radius
    exact = 0.5 * LB * ((1 / sdie - 1 / pdie) / a - k / (sdie * (1 + k * b))) * KT
    e = kirkwood_energy([[0, 0, 0]], [-2.0], a, pdie=pdie, ionic_strength=ionic, ion_radius=ion_radius)
    assert e == pytest.approx(4 * exact, rel=1e-13)


def test_kirkwood_grounded_sphere_limit():
    """sdie → ∞ or κ → ∞ at the sphere make it a grounded conductor: image charges,
    which also check the cross terms of several off-centre charges."""
    exact = _grounded(QPOS, QVAL, A, 2.0)
    assert kirkwood_energy(QPOS, QVAL, A, pdie=2.0, sdie=1e12) == pytest.approx(exact, rel=1e-10)
    e = kirkwood_energy(QPOS, QVAL, A, pdie=2.0, ionic_strength=1e8, ion_radius=0.0)
    assert e == pytest.approx(exact, rel=1e-5)  # finite κa ≈ 1.6e4
    near = np.array([[4.9, 0.0, 0.0]])  # ρ/a = 0.98: ~800 multipoles
    assert kirkwood_energy(near, [1.0], A, pdie=2.0, sdie=1e12) == pytest.approx(_grounded(near, [1.0], A, 2.0), rel=1e-10)


def _kn(n, x):
    """Modified spherical Bessel kₙ(x) (up to π/2) and its derivative, from the explicit sum."""
    c = [math.factorial(n + s) / (math.factorial(s) * math.factorial(n - s)) / 2**s for s in range(n + 1)]
    k = sum(cs * x ** -(s + 1) for s, cs in enumerate(c)) * math.exp(-x)
    dk = sum(cs * (-(x ** -(s + 1)) - (s + 1) * x ** -(s + 2)) for s, cs in enumerate(c)) * math.exp(-x)
    return k, dk


def test_kirkwood_coefficients_solve_the_interface_conditions():
    """β_n against the 4×4 matching conditions at the sphere and the Stern radius."""
    a, b, pdie, sdie = 5.0, 7.0, 2.0, 78.54
    ka = _kappa(0.15) * a
    beta = _reaction_coefficients(12, pdie, sdie, ka, b / a)
    for n in range(13):
        u = b / a
        k, dk = _kn(n, ka * u)
        dk *= ka
        # unknowns B, C, D, E; source term u^-(n+1)/pdie inside, u = r/a
        m = np.array(
            [
                [1.0, -1.0, -1.0, 0.0],
                [n * pdie, -n * sdie, (n + 1) * sdie, 0.0],
                [0.0, u**n, u ** -(n + 1), -k],
                [0.0, n * u ** (n - 1), -(n + 1) * u ** -(n + 2), -dk],
            ]
        )
        rhs = np.array([-1.0 / pdie, n + 1.0, 0.0, 0.0])
        assert beta[n] == pytest.approx(np.linalg.solve(m, rhs)[0], rel=1e-12), n


def test_kirkwood_is_rotation_invariant_and_checks_its_input():
    rot, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    kw = dict(pdie=4.0, ionic_strength=0.15)
    assert kirkwood_energy(QPOS @ rot.T, QVAL, A, **kw) == pytest.approx(kirkwood_energy(QPOS, QVAL, A, **kw), rel=1e-13)
    with pytest.raises(ValueError):
        kirkwood_energy([[5.0, 0.0, 0.0]], [1.0], A)
    with pytest.raises(ValueError):
        kirkwood_energy(QPOS, QVAL[:2], A)


@pytest.mark.parametrize("kw", [dict(), dict(ionic_strength=0.15, pdie=4.0)])
def test_solver_converges_to_kirkwood(kw):
    """The sphere as one atom of radius A with zero-radius charges inside: the
    7-point scheme with a sharp boundary is first order (docs/VALIDATION.md)."""
    coords = np.vstack([[0.0, 0.0, 0.0], QPOS])
    charges, radii = np.r_[0.0, QVAL], np.r_[A, 0.0, 0.0, 0.0]
    exact = kirkwood_energy(QPOS, QVAL, A, **kw)
    err = {}
    for h in (0.5, 0.25):
        e = pbsolve.Solver(platform="CPU", grid_spacing=h, **kw).polar_solvation(coords, charges, radii).energy
        err[h] = (e - exact) / abs(exact)
    # measured: −3.1 / −1.2 % (I = 0) and −2.7 / −1.0 % (0.15 M, pdie 4)
    assert abs(err[0.25]) < 0.015
    assert abs(err[0.25]) < 0.6 * abs(err[0.5])
