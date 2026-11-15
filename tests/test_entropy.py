import math

import numpy as np
import pytest

import pbsolve
from pbsolve.entropy import SIGMA_LIMIT

KT = pbsolve.kt_kcal_per_mol(298.15)
COULOMB = pbsolve.bjerrum_length(298.15) * KT  # kcal·Å/mol


def _two_level(a, b, k, n):
    """IE and C2 of k energies a and n − k energies b, in closed form."""
    p = k / n
    mean = p * a + (1 - p) * b
    ie = KT * math.log(p * math.exp((a - mean) / KT) + (1 - p) * math.exp((b - mean) / KT))
    return ie, p * (1 - p) * (a - b) ** 2 / (2 * KT)


@pytest.mark.parametrize("a, b, k, n", [(-50.0, -48.0, 3, 10), (-20.0, -26.0, 1, 40), (5.0, 5.5, 20, 21)])
def test_two_level_distribution_is_exact(a, b, k, n):
    e = [a] * k + [b] * (n - k)
    r = pbsolve.interaction_entropy(e, blocks=1)
    ie, c2 = _two_level(a, b, k, n)
    assert r.ie == pytest.approx(ie, rel=1e-12)
    assert r.c2 == pytest.approx(c2, rel=1e-12)
    assert r.sigma == pytest.approx(abs(a - b) * math.sqrt(k / n * (1 - k / n)), rel=1e-12)
    assert r.n == n


def test_gaussian_energies_give_ie_equal_to_c2():
    e = np.random.default_rng(7).normal(-45.0, 1.0, 400_000)
    r = pbsolve.interaction_entropy(e)
    assert r.c2 == pytest.approx(e.var() / (2 * KT), rel=1e-14)
    assert r.c2 == pytest.approx(1.0 / (2 * KT), rel=0.01)  # σ = 1
    assert r.ie == pytest.approx(r.c2, rel=0.02)  # sampling error of the exponential average ~0.6 %
    assert r.reliable


def test_large_fluctuations_are_flagged_and_do_not_overflow():
    e = np.random.default_rng(3).normal(0.0, 200.0, 2000)  # β ΔE up to ~10³: exp overflows naively
    r = pbsolve.interaction_entropy(e)
    assert np.isfinite(r.ie) and np.isfinite(r.c2)
    assert r.ie == pytest.approx(e.max() - e.mean() - KT * math.log(e.size), abs=1.0)  # one snapshot dominates
    assert r.ie < r.c2  # the exponential average is far from converged
    assert not r.reliable
    assert SIGMA_LIMIT * 4.184 == pytest.approx(15.0, rel=1e-15)  # 15 kJ/mol (Ekberg & Ryde 2021)
    assert pbsolve.interaction_entropy([0.0, 2 * SIGMA_LIMIT], blocks=1).reliable  # σ = SIGMA_LIMIT
    assert not pbsolve.interaction_entropy([0.0, 2 * SIGMA_LIMIT * 1.0001], blocks=1).reliable


def test_shift_invariance_positivity_and_temperature():
    e = np.random.default_rng(1).gamma(2.0, 1.5, 500) - 60.0
    r = pbsolve.interaction_entropy(e)
    s = pbsolve.interaction_entropy(e + 123.0)
    assert s.ie == pytest.approx(r.ie, rel=1e-11) and s.c2 == pytest.approx(r.c2, rel=1e-11)
    assert r.ie > 0 and r.c2 > 0  # Jensen: ⟨e^x⟩ ≥ e^⟨x⟩ = 1
    hot = pbsolve.interaction_entropy(e, temperature=310.0)
    assert hot.c2 == pytest.approx(r.c2 * KT / pbsolve.kt_kcal_per_mol(310.0), rel=1e-13)
    assert hot.sigma == r.sigma


def test_blocks():
    e = np.random.default_rng(5).normal(-30.0, 2.0, 103)
    r = pbsolve.interaction_entropy(e, blocks=4)
    assert r.ie_blocks.shape == r.c2_blocks.shape == (4,)
    for x, ie, c2 in zip(np.array_split(e, 4), r.ie_blocks, r.c2_blocks):
        b = pbsolve.interaction_entropy(x, blocks=1)
        assert (ie, c2) == (b.ie, b.c2)
    assert r.c2_sem == pytest.approx(np.std(r.c2_blocks, ddof=1) / 2.0, rel=1e-14)
    assert r.ie_sem == pytest.approx(np.std(r.ie_blocks, ddof=1) / 2.0, rel=1e-14)
    # Each block constant: no fluctuation within blocks although the whole run fluctuates.
    split = pbsolve.interaction_entropy([1.0] * 5 + [4.0] * 5, blocks=2)
    assert list(split.ie_blocks) == list(split.c2_blocks) == [0.0, 0.0]
    assert split.c2 > 0
    assert math.isnan(pbsolve.interaction_entropy([1.0, 2.0], blocks=1).ie_sem)


@pytest.mark.parametrize(
    "energies, blocks", [([1.0], 1), ([1.0, float("nan"), 2.0], 1), (np.zeros(9), 5), (np.zeros(10), 0)]
)
def test_invalid_input(energies, blocks):
    with pytest.raises(ValueError):
        pbsolve.interaction_entropy(energies, blocks=blocks)


def test_interaction_energy_of_two_atoms():
    x = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    elec, lj = pbsolve.interaction_energy(x, [1.0, -0.5], [1.4, 1.6], [0.2, 0.05], [1])
    assert elec == pytest.approx(-0.5 * COULOMB / 3.0, rel=1e-15)
    assert lj == pytest.approx(-0.1, rel=1e-14)  # r = R_i + R_j: the well bottom −√(ε_i ε_j)
    elec4, lj4 = pbsolve.interaction_energy(x, [1.0, -0.5], [1.4, 1.6], [0.2, 0.05], [1], [0], dielectric=4.0)
    assert (elec4, lj4) == (pytest.approx(elec / 4.0, rel=1e-15), lj)
    assert COULOMB == pytest.approx(332.0637133, rel=1e-9)  # e²/4πε0, CODATA 2018


def test_interaction_energy_equals_the_pair_sum():
    rng = np.random.default_rng(11)
    n = 330  # ligand larger than one 256-atom chunk
    x = rng.uniform(0.0, 30.0, (n, 3))
    q = rng.normal(0.0, 0.4, n)
    r = rng.uniform(0.0, 2.0, n)
    eps = rng.uniform(0.0, 0.2, n)
    eps[::7] = 0.0
    lig = list(range(20, 300))
    rec = [i for i in range(n) if i not in lig][::2]  # an explicit, partial receptor
    elec = lj = 0.0
    for i in rec:
        for j in lig:
            d = float(np.linalg.norm(x[i] - x[j]))
            elec += COULOMB * q[i] * q[j] / d
            s6 = ((r[i] + r[j]) / d) ** 6
            lj += math.sqrt(eps[i] * eps[j]) * (s6 * s6 - 2 * s6)
    e, v = pbsolve.interaction_energy(x, q, r, eps, lig, rec)
    assert e == pytest.approx(elec, rel=1e-12) and v == pytest.approx(lj, rel=1e-12)
    rest = pbsolve.interaction_energy(x, q, r, eps, lig)  # default receptor: every other atom
    full = pbsolve.interaction_energy(x, q, r, eps, lig, [i for i in range(n) if i not in lig])
    assert rest == full


@pytest.mark.parametrize("lig, rec", [([0], [0, 1]), ([], [1]), ([0, 1, 2], None)])
def test_interaction_energy_rejects_bad_selections(lig, rec):
    with pytest.raises(ValueError):
        pbsolve.interaction_energy(np.zeros((3, 3)), [0, 0, 0], [1, 1, 1], [0, 0, 0], lig, rec)
    with pytest.raises(ValueError):
        pbsolve.interaction_energy(np.zeros((3, 3)), [0, 0], [1, 1, 1], [0, 0, 0], [0])


def test_interaction_energy_matches_openmm():
    openmm = pytest.importorskip("openmm")
    from openmm import unit

    from pbsolve.openmm_bridge import charges_from_system, lennard_jones_from_system

    rng = np.random.default_rng(2)
    n, lig = 60, list(range(45, 60))
    x = np.concatenate([rng.uniform(0, 12, (45, 3)), rng.uniform(0, 6, (15, 3)) + [15.0, 3.0, 3.0]])
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for i in range(n):
        system.addParticle(12.0)
        nb.addParticle(rng.normal(0, 0.5), rng.uniform(0.15, 0.35), 0.0 if i % 9 == 0 else rng.uniform(0.05, 0.9))
    for i, j in [(0, 1), (1, 2), (10, 30), (45, 46), (50, 59)]:  # intramolecular exceptions (1-4 scaling)
        nb.addException(i, j, 0.1, 0.25, 0.2)
    system.addForce(nb)

    def energy(keep):
        """OpenMM energy with every particle outside `keep` switched off; exceptions stay on."""
        params = [nb.getParticleParameters(i) for i in range(n)]
        for i in range(n):
            if i not in keep:
                nb.setParticleParameters(i, 0.0, params[i][1], 0.0)
        ctx = openmm.Context(system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(x * 0.1)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        for i, p in enumerate(params):
            nb.setParticleParameters(i, *p)
        return e

    # The exceptions and anything bonded survive the switch-off, so E(none) adds them back.
    reference = energy(set(range(n))) - energy(set(range(45))) - energy(set(lig)) + energy(set())
    radii, eps = lennard_jones_from_system(system)
    elec, lj = pbsolve.interaction_energy(x, charges_from_system(system), radii, eps, lig)
    assert abs(elec) > 1.0 and abs(lj) > 0.1
    assert elec + lj == pytest.approx(reference, rel=1e-9)
