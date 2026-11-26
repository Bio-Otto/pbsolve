"""Poisson–Boltzmann pKa values and titration (`pbsolve.pka`, ROADMAP 4.21)."""

import numpy as np
import pytest

import pbsolve
from pbsolve.pka import LN10, Site, SiteEnergies, site_energies, titrate

from conftest import synthetic_molecule

KT = pbsolve.kt_kcal_per_mol(298.15)


def energies(pka, acid, w):
    sites = [Site(f"s{i}", [i], [0.0], [-1.0 if a else 1.0], p, a) for i, (p, a) in enumerate(zip(pka, acid))]
    n = len(sites)
    return SiteEnergies(sites, np.zeros(n), np.zeros(n), np.asarray(w, float), np.asarray(pka, float), 298.15)


def test_independent_sites_follow_henderson_hasselbalch():
    e = energies([4.0, 10.5, 6.3], [True, False, False], np.zeros((3, 3)))
    t = titrate(e)
    want = 1.0 / (1.0 + 10.0 ** (t.ph[:, None] - e.intrinsic_pka[None, :]))
    assert np.abs(t.protonated - want).max() < 1e-12
    assert t.pka == pytest.approx(e.intrinsic_pka, abs=2e-3)  # linear interpolation on a 0.1 grid
    # The net charge: −1 per deprotonated acid, +1 per protonated base.
    assert t.charge[0] == pytest.approx(want[0, 1] + want[0, 2] - (1 - want[0, 0]), abs=1e-12)


def test_two_coupled_sites_match_their_partition_function():
    w = 1.7  # kcal/mol between the two charged states
    e = energies([4.0, 6.0], [True, True], [[0.0, w], [w, 0.0]])
    t = titrate(e, ph=[3.0, 5.0, 7.0])
    for k, ph in enumerate(t.ph):
        g = lambda x0, x1: KT * LN10 * ((4.0 - ph) * x0 + (6.0 - ph) * x1) + w * x0 * x1  # noqa: E731
        z = {s: np.exp(-g(*s) / KT) for s in [(0, 0), (1, 0), (0, 1), (1, 1)]}
        total = sum(z.values())
        assert t.protonated[k, 0] == pytest.approx((z[(0, 0)] + z[(0, 1)]) / total, rel=1e-12)
        assert t.protonated[k, 1] == pytest.approx((z[(0, 0)] + z[(1, 0)]) / total, rel=1e-12)


def test_monte_carlo_agrees_with_the_exact_sum():
    rng = np.random.default_rng(4)
    n = 9
    w = rng.normal(0.0, 1.2, (n, n))
    w = np.triu(w, 1) + np.triu(w, 1).T
    w[0, 1] = w[1, 0] = 4.5  # strongly coupled: pair moves
    e = energies(rng.uniform(3.0, 11.0, n), rng.random(n) < 0.5, w)
    ph = np.arange(0.0, 14.01, 1.0)
    exact = titrate(e, ph=ph, method="exact")
    mc = titrate(e, ph=ph, method="mc", sweeps=20000, seed=1)
    assert mc.method == "mc" and mc.error is not None
    assert np.abs(mc.protonated - exact.protonated).max() < max(0.015, 5 * mc.error.max())
    assert np.array_equal(titrate(e, ph=ph, method="mc", sweeps=2000, seed=1).protonated,
                          titrate(e, ph=ph, method="mc", sweeps=2000, seed=1).protonated)


def molecule_with_sites():
    x, q, r = synthetic_molecule()
    r = np.where(r > 0, r, 1.0)
    groups = [(0, 1, 2), (10, 11), (20, 21, 22)]
    sites = []
    for k, (atoms, acid) in enumerate(zip(groups, [True, False, True])):
        neutral = q[list(atoms)]
        change = np.full(len(atoms), (-1.0 if acid else 1.0) / len(atoms))
        model = sorted(set(atoms) | {atoms[0] + 3, atoms[0] + 4})
        sites.append(Site(f"site {k}", atoms, neutral, neutral + change, 4.0 + 3 * k, acid, model))
    return x, q, r, sites


def full_energy(solver, x, q, r, grid):
    """E(q): the reaction-field energy plus Coulomb's law in pdie between all pairs (kcal/mol)."""
    rf = solver.polar_solvation(x, q, r, grid=grid).energy if np.any(q) else 0.0
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    iu = np.triu_indices(len(q), 1)
    lb = pbsolve.bjerrum_length(298.15) * KT / solver.params["pdie"]
    return rf + lb * float(np.sum(q[iu[0]] * q[iu[1]] / d[iu]))


def test_site_terms_are_energy_differences():
    x, q, r, sites = molecule_with_sites()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, pdie=4.0, tol=1e-12)
    grid = s.make_grid(x, r)
    e = site_energies(s, x, q, r, sites, grid=grid)
    delta = []
    for site in sites:
        d = np.zeros(len(q))
        d[list(site.atoms)] = np.asarray(site.charged) - np.asarray(site.neutral)
        delta.append(d)
    scale = np.abs(e.background).max()
    for i, site in enumerate(sites):
        m = np.asarray(site.model_atoms)
        want = (full_energy(s, x, q + delta[i], r, grid) - full_energy(s, x, q, r, grid)) - (
            full_energy(s, x[m], (q + delta[i])[m], r[m], grid) - full_energy(s, x[m], q[m], r[m], grid)
        )
        assert e.born[i] + e.background[i] == pytest.approx(want, abs=1e-8 * scale), site.name
        sign = 1.0 if site.acid else -1.0
        assert e.intrinsic_pka[i] == pytest.approx(site.model_pka + sign * want / (KT * LN10), abs=1e-8)
        for j in range(i + 1, len(sites)):
            pair = (
                full_energy(s, x, q + delta[i] + delta[j], r, grid)
                - full_energy(s, x, q + delta[i], r, grid)
                - full_energy(s, x, q + delta[j], r, grid)
                + full_energy(s, x, q, r, grid)
            )
            assert e.interactions[i, j] == pytest.approx(pair, abs=1e-8 * scale), (i, j)
    assert np.array_equal(e.interactions, e.interactions.T) and not e.interactions.diagonal().any()


def test_in_a_uniform_dielectric_only_coulomb_remains():
    x, q, r, sites = molecule_with_sites()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, pdie=4.0, sdie=4.0)
    e = site_energies(s, x, q, r, sites)
    assert np.abs(e.born).max() < 1e-12
    lb = pbsolve.bjerrum_length(298.15) * KT / 4.0
    a, b = sites[0], sites[2]
    da = np.asarray(a.charged) - np.asarray(a.neutral)
    db = np.asarray(b.charged) - np.asarray(b.neutral)
    d = np.linalg.norm(x[list(a.atoms)][:, None] - x[list(b.atoms)][None], axis=-1)
    assert e.interactions[0, 2] == pytest.approx(lb * np.sum(np.outer(da, db) / d), rel=1e-12)


def test_sites_are_checked():
    x, q, r, sites = molecule_with_sites()
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.8, fadd=8.0)
    with pytest.raises(ValueError, match="neutral state"):
        site_energies(s, x, q + 0.1, r, sites)
    with pytest.raises(ValueError, match="shares atoms"):
        site_energies(s, x, q, r, [sites[0], Site("dup", [2], [q[2]], [q[2] - 1], 4.0, True)])
    with pytest.raises(ValueError, match="model_atoms"):
        site_energies(s, x, q, r, [Site("m", [0], [q[0]], [q[0] - 1], 4.0, True, [1, 2])])
    with pytest.raises(ValueError, match="unknown method"):
        titrate(energies([4.0], [True], [[0.0]]), method="newton")
