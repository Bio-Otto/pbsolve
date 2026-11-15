"""The Gaussian dielectric (`surface="gauss"`) against a semi-analytic reference."""

import json

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main

from conftest import synthetic_molecule

T = 298.15
SHIFTS = np.array([[0.1, 0.4, 0.7], [0.6, 0.2, 0.9], [0.3, 0.8, 0.5], [0.9, 0.6, 0.1]])


def radial_reference(R, sigma, pdie, sdie, contour, kbar2=0.0, born=0.0):
    """ΔG_polar (kcal/mol) of a unit charge at the centre of one Gaussian atom.

    Solvated: (ε r² u')' = κ̄² K r² u with ε(r) = pdie + (sdie − pdie)(1 − e^{−r²/σ²R²}) and
    K = exp(−born (1/ε − 1/sdie)); reference: ε up to the contour, pdie beyond, no ions.
    ΔG = ½ q [w_solvated(0) − w_reference(0)] with w = u − l_B q/(pdie r).
    """
    from scipy.integrate import quad, solve_ivp

    lb, kt, s = pbsolve.bjerrum_length(T), pbsolve.kt_kcal_per_mol(T), sigma * R
    eps = lambda r: pdie + (sdie - pdie) * (-np.expm1(-((r / s) ** 2)))
    c = lb / pdie

    def rhs(r, y, inhom):  # y = (w, ε r² u')
        k2 = kbar2 * np.exp(-born * (1 / eps(r) - 1 / sdie))
        return [(y[1] / eps(r) + inhom * c) / r**2, k2 * (r**2 * y[0] + inhom * c * r)]

    r0, rf = 1e-4 * s, max(12 * s, 30.0)
    opts = dict(method="DOP853", rtol=1e-12, atol=1e-14)
    # w'(0) = l_B (sdie − pdie)/(s² pdie²): start the particular solution there.
    p = solve_ivp(rhs, (r0, rf), [r0 * lb * (sdie - pdie) / (s * s * pdie**2), -lb], args=(1.0,), **opts).y[:, -1]
    h = solve_ivp(rhs, (r0, rf), [1.0, 0.0], args=(0.0,), **opts).y[:, -1]
    kap = np.sqrt(kbar2 / sdie)
    decay = lambda y, inhom: y[1] + sdie * rf * (kap * rf + 1) * (y[0] + inhom * c / rf)
    w_solvated = -decay(p, 1.0) / decay(h, 0.0)
    rc = s * np.sqrt(-np.log(1 - (contour - pdie) / (sdie - pdie)))
    w_reference = lb * quad(lambda r: (1 / eps(r) - 1 / pdie) / r**2, 0, rc, epsabs=0, epsrel=1e-13)[0]
    return 0.5 * (w_solvated - w_reference) * kt


def test_one_atom_against_the_radial_reference():
    # pdie 4, σR = 3.7 Å: the contour lies 1.8 Å from the charge. Measured: −0.94 %
    # at 0.35 Å (−1.3 % at 0.5, −0.45 % at 0.18; first order).
    x, q, r = np.array([[0.113, -0.071, 0.052]]), np.array([1.0]), np.array([4.0])
    energies = {}
    for salt in (0.0, 0.15):
        s = pbsolve.Solver(platform="CPU", grid_spacing=0.35, pdie=4.0, surface="gauss", ionic_strength=salt, tol=1e-10)
        g0, h = s.make_grid(x, r), 0.35
        es = [s.polar_solvation(x, q, r, grid=pbsolve.GridSpec(list(np.asarray(g0.center) + d * h), g0.dims, g0.coarse_length, g0.fine_length)).energy for d in SHIFTS]
        energies[salt] = np.mean(es)
        assert np.std(es) < 0.002 * abs(energies[salt])
    kbar2 = pbsolve.Solver(ionic_strength=0.15).maps(x, q, r, grid=pbsolve.GridSpec([0.0] * 3, [9] * 3, [4.0] * 3))["kappa_bar_sq"]
    born = pbsolve.bjerrum_length(T) / (2 * 2.0)  # z = 1, ion radius 2 Å
    exact = {salt: radial_reference(4.0, 0.93, 4.0, 78.54, 20.0, kb, born) for salt, kb in [(0.0, 0.0), (0.15, kbar2)]}
    for salt in exact:
        assert energies[salt] == pytest.approx(exact[salt], rel=0.015)  # −20.70 kcal/mol with salt
    # The ions' Born penalty: the salt effect to 0.2 % (−0.183 kcal/mol; −3.9 without the penalty).
    assert energies[0.15] - energies[0.0] == pytest.approx(exact[0.15] - exact[0.0], rel=2e-3)


def test_maps_and_settings():
    x, q, r = synthetic_molecule()
    r = np.where(r > 0, r, 1.0)
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.1, surface="gauss", gauss_sigma=0.8, gauss_contour=30.0)
    assert (s.params["surface"], s.params["gauss_sigma"], s.params["gauss_contour"]) == ("gauss", 0.8, 30.0)
    m, ref = s.maps(x, q, r), s.maps(x, q, r, reference=True)
    eps = m["eps_x"]
    assert eps.min() > 1.0 and ((eps > 2.0) & (eps < 77.0)).sum() > 1000  # graded, never exactly pdie
    # The reference keeps ε up to the contour, pdie beyond, and has no ions.
    assert np.array_equal(ref["eps_x"], np.where(eps > 30.0, 1.0, eps))
    assert not ref["kappa_sq"].any()
    k = m["kappa_sq"] / m["kappa_bar_sq"]
    assert k.max() == pytest.approx(1.0) and k.min() < 1e-10 and ((k > 0.01) & (k < 0.99)).sum() > 100


def test_what_gauss_refuses():
    x, q, r = synthetic_molecule()
    kw = dict(platform="CPU", grid_spacing=0.6, fadd=10.0, surface="gauss")
    # Atom 5 (radius 0, charged) lies in the solvent: outside the contour.
    with pytest.raises(ValueError, match="atom 5 .*gauss_contour"):
        pbsolve.Solver(**kw).polar_solvation(x, q, r)
    r = np.where(r > 0, r, 1.0)
    with pytest.raises(ValueError, match="coulomb 'grid'"):
        pbsolve.Solver(**kw, coulomb="analytic")
    with pytest.raises(ValueError, match="sharp contour"):
        pbsolve.Solver(**kw).polar_forces(x, q, r)
    with pytest.raises(ValueError, match="diverges"):
        pbsolve.Solver(**kw, equation="npbe")
    with pytest.raises(ValueError, match="one valence"):
        pbsolve.Solver(**kw, ions=[(2.0, 0.1, 2.0), (-1.0, 0.2, 2.0)])
    for bad in (dict(gauss_sigma=0.0), dict(gauss_contour=1.0)):
        with pytest.raises(ValueError):
            pbsolve.Solver(**kw, **bad)
    from pbsolve.apbs import elec_from_solver

    s = pbsolve.Solver(**kw)
    with pytest.raises(ValueError, match="no APBS equivalent"):
        elec_from_solver(s, s.make_grid(x, r), "x")


def test_command_line(capsys):
    pqr = str(__import__("pathlib").Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")
    args = ["solvation", pqr, "--platform", "CPU", "--spacing", "0.9", "--json", "--surface", "gauss"]
    energies = []
    # pdb2pqr's AMBER radii: the 18 charged hydrogens of radius 0 lie inside the contour.
    for extra in ([], ["--gauss-sigma", "0.97"], ["--gauss-contour", "25"]):
        assert main(args + extra) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    assert len(set(energies)) == 3


@pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")
def test_cuda_equals_the_cpu():
    x, q, r = synthetic_molecule()
    r = np.where(r > 0, r, 1.0)
    kw = dict(grid_spacing=0.5, fadd=10.0, ionic_strength=0.15, surface="gauss", equation="smpbe", ion_size=6.0)
    cpu = pbsolve.Solver(platform="CPU", **kw)
    grid = cpu.make_grid(x, r)
    a = cpu.polar_solvation(x, q, r, grid=grid)
    b = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(x, q, r, grid=grid)
    assert b.platform == "CUDA" and abs(a.energy - b.energy) < 1e-9 * abs(a.energy)
