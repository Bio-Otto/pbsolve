"""Analytic references (`pbsolve.analytic`) and pbsolve against them."""

import math

import numpy as np
import pytest

import pbsolve
from pbsolve.analytic import _legendre, _log_derivatives, _reaction_coefficients, kirkwood_energy, two_sphere_energy
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


# A second sphere (4 Å) with two charges; centres 15.8 Å apart, so the ion-free
# shells (b = a + 2 Å) stay apart and the `mol` surface is the union of the spheres.
A2 = 4.0
QPOS2 = np.array([[0.0, -1.0, 1.5], [1.2, 0.3, -0.4]])
QVAL2 = np.array([-1.0, 0.7])


def _pair(offset):
    centers = np.array([[0.0, 0.0, 0.0], offset])
    return np.vstack([QPOS, QPOS2 + centers[1]]), np.r_[QVAL, QVAL2], centers, np.array([A, A2])


def test_legendre_functions_are_orthonormal():
    x, w = np.polynomial.legendre.leggauss(60)
    for m in (0, 1, 7, 20):
        p = _legendre(40, m, x)
        np.testing.assert_allclose((p * w) @ p.T, np.eye(41 - m), atol=1e-12)


def test_bessel_log_derivatives():
    """x·iₙ'/iₙ from the power series and x·kₙ'/kₙ from the explicit polynomial."""
    for x in (0.3, 1.0, 3.0):
        li, lk = _log_derivatives(30, x)
        for n in range(31):
            c = [1.0 / (2**k * math.factorial(k) * math.prod(range(1, 2 * n + 2 * k + 2, 2))) for k in range(40)]
            series = sum(ck * x ** (2 * k) for k, ck in enumerate(c))
            deriv = sum(ck * (n + 2 * k) * x ** (2 * k) for k, ck in enumerate(c))
            assert li[n] == pytest.approx(deriv / series, rel=1e-13), (x, n)
            if n <= 12:
                k, dk = _kn(n, x)
                assert lk[n] == pytest.approx(x * dk / k, rel=1e-13), (x, n)


def test_two_spheres_without_a_dielectric_jump():
    """pdie = sdie without salt: the re-expanded fields must rebuild the Coulomb
    potential between the spheres exactly, so nothing is left."""
    pos, q, c, a = _pair([3.0, -4.0, 15.0])
    assert abs(two_sphere_energy(pos, q, c, a, pdie=4.0, sdie=4.0)) < 1e-12


def test_two_spheres_conductor_limit():
    """sdie → ∞: each sphere is a grounded conductor on its own (image charges); the
    reference state still has the Coulomb interaction of the two spheres in pdie."""
    pos, q, c, a = _pair([3.0, -4.0, 15.0])
    n1 = len(QVAL)
    images = _grounded(QPOS, QVAL, A, 2.0) + _grounded(QPOS2, QVAL2, A2, 2.0)
    rij = np.linalg.norm(pos[:n1, None, :] - pos[None, n1:, :], axis=2)
    exact = images - LB / 2.0 * np.sum(np.outer(QVAL, QVAL2) / rij) * KT
    assert two_sphere_energy(pos, q, c, a, pdie=2.0, sdie=1e12) == pytest.approx(exact, rel=1e-10)


def test_two_spheres_far_apart():
    k1, k2 = (kirkwood_energy(QPOS, QVAL, A, pdie=2.0), kirkwood_energy(QPOS2, QVAL2, A2, pdie=2.0))
    # Without salt: the charges and dipoles of the spheres interact through the solvent;
    # a dielectric sphere passes on 3/(pdie + 2 sdie) of a dipole's field.
    d = 1000.0
    pos, q, c, a = _pair([0.0, 0.0, d])
    q1, q2, p1, p2 = QVAL.sum(), QVAL2.sum(), QVAL @ QPOS, QVAL2 @ QPOS2
    f = 3.0 / (2.0 + 2 * 78.54) - 1 / 2.0
    est = (q1 * q2 * (1 / 78.54 - 1 / 2.0) / d + f * (q2 * p1[2] - q1 * p2[2]) / d**2) * LB * KT
    assert two_sphere_energy(pos, q, c, a, pdie=2.0) - k1 - k2 == pytest.approx(est, rel=1e-3)  # O(1/d) more
    # With salt the solvent screens the spheres from each other: only the reference
    # state's Coulomb interaction is left, exactly.
    kw = dict(pdie=2.0, ionic_strength=0.15)
    pos, q, c, a = _pair([0.0, 0.0, 300.0])
    rij = np.linalg.norm(pos[: len(QVAL), None, :] - pos[None, len(QVAL) :, :], axis=2)
    cross = -LB / 2.0 * np.sum(np.outer(QVAL, QVAL2) / rij) * KT
    k = kirkwood_energy(QPOS, QVAL, A, **kw) + kirkwood_energy(QPOS2, QVAL2, A2, **kw)
    assert two_sphere_energy(pos, q, c, a, **kw) - k == pytest.approx(cross, rel=1e-10)


@pytest.mark.parametrize("kw", [dict(pdie=4.0), dict(pdie=4.0, ionic_strength=0.15)])
def test_two_spheres_are_symmetric_and_converged(kw):
    pos, q, c, a = _pair([3.0, -4.0, 15.0])
    e = two_sphere_energy(pos, q, c, a, **kw)
    assert two_sphere_energy(pos, q, c, a, order=80, **kw) == pytest.approx(e, rel=1e-11)
    assert two_sphere_energy(pos[::-1], q[::-1], c[::-1], a[::-1], **kw) == pytest.approx(e, rel=1e-11)
    rot, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    assert two_sphere_energy(pos @ rot.T + 1.0, q, c @ rot.T + 1.0, a, **kw) == pytest.approx(e, rel=1e-11)


def test_two_spheres_check_their_input():
    pos, q, c, a = _pair([0.0, 0.0, 15.0])
    with pytest.raises(ValueError, match="inside one"):
        two_sphere_energy(np.vstack([pos, [0.0, 0.0, 7.0]]), np.r_[q, 1.0], c, a)
    with pytest.raises(ValueError, match="overlap"):
        two_sphere_energy(pos, q, c, a, ionic_strength=0.15, ion_radius=3.5)  # shells 8.5 + 7.5 > 15
    with pytest.raises(ValueError, match="overlap"):
        two_sphere_energy(pos, q, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]]), a)


def test_solver_two_spheres():
    """Complex and association energy (MM-PBSA style, all three on the complex's grid).
    `vdw` is exactly the two spheres; `mol` adds APBS' sampled probe positions, whose
    cusps up to ~0.004 Å outside the spheres shift ΔΔG by ~0.015 kcal/mol at every spacing."""
    pos, q, c, a = _pair([3.0, -4.0, 15.0])
    exact = two_sphere_energy(pos, q, c, a)
    ddg = exact - kirkwood_energy(QPOS, QVAL, A) - kirkwood_energy(QPOS2, QVAL2, A2)
    coords, charges, radii = np.vstack([c, pos]), np.r_[0.0, 0.0, q], np.r_[a, np.zeros(len(q))]
    first = np.r_[True, False, [True] * len(QVAL), [False] * len(QVAL2)]
    err = {}
    for h in (0.5, 0.35):
        s = pbsolve.Solver(platform="CPU", grid_spacing=h, surface="vdw")
        g = s.make_grid(coords, radii)
        e = [s.polar_solvation(coords[m], charges[m], radii[m], grid=g).energy for m in (slice(None), first, ~first)]
        err[h] = (e[0] - exact) / abs(exact)
        assert abs(e[0] - e[1] - e[2] - ddg) < 0.006  # kcal/mol of 1.79; measured −0.0031 / −0.0007
    # measured −3.0 / −2.1 %
    assert abs(err[0.35]) < 0.025
    assert abs(err[0.35]) < 0.8 * abs(err[0.5])
