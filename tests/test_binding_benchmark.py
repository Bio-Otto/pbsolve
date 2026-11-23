"""Input parsing, energies and statistics of scripts/binding_benchmark.py (ROADMAP 4.19).

The benchmark itself needs OpenMM, AmberTools and the downloaded data; these
tests cover the parts that decide its numbers without them.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "binding_benchmark.py"
    spec = importlib.util.spec_from_file_location("binding_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the benchmark's layout: "additional" ITC blocks before and after the measurement, one entry indented by three
LIGANDS_YML = """\
lig_a:
  measurement:
    additional:
    - error: 0.2
      type: dg
      unit: kilojoules / mole
      value: -31.3
    comment: Figure 1 cmpd 1a
    error: 2.12
    type: ki
    unit: uM
    value: 3.99
  name: lig_a
  smiles: N
lig_b:
  measurement:
   error: 1.0
   type: ic50
   unit: nM
   value: 25.0
   additional:
   - error: 0.2
     type: dg
     unit: kilojoules / mole
     value: -31.3
  name: lig_b
lig_c:
  measurement:
    error: -1
    type: pic50
    unit: ''
    value: 7.5
  name: lig_c
"""

SDF = """\
ammonium
  test

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  3  0  0  0  0  0  0  0  0  0  0
    1.0100   -0.5000    2.2500 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  CHG  1   1   1
M  END
> <i_epik_Tot_Q>
0

$$$$
chloride
  test

  1  0  0  0  0  0  0  0  0  0999 V2000
   -3.5000    4.0000-1000.1250 Cl  0  5  0  0  0  0  0  0  0  0  0  0
M  CHG  1   1  -1
M  END
$$$$
"""


def test_measurements_become_binding_free_energies():
    bb = _script()
    dg = bb.experimental_dg(LIGANDS_YML)
    rt = 8.314462618e-3 / 4.184 * 298.15
    assert dg == pytest.approx({
        "lig_a": rt * math.log(3.99e-6),  # the primary Ki, not the ITC ΔG of "additional"
        "lig_b": rt * math.log(25e-9),
        "lig_c": -rt * math.log(10.0) * 7.5,
    }, rel=1e-12)
    with pytest.raises(ValueError, match="unknown measurement type"):
        bb.experimental_dg(LIGANDS_YML.replace("pic50", "kd"))


def test_sdf_molecules_take_coordinates_elements_and_formal_charges():
    bb = _script()
    mols = bb.sdf_molecules(SDF)
    assert [m["name"] for m in mols] == ["ammonium", "chloride"]
    assert [m["elements"] for m in mols] == [["N", "H"], ["Cl"]]
    assert [m["charge"] for m in mols] == [1, -1]  # M  CHG decides, not the stale Epik property
    np.testing.assert_array_equal(mols[0]["xyz"], [[0.0, 0.0, 0.0], [1.01, -0.5, 2.25]])
    np.testing.assert_array_equal(mols[1]["xyz"], [[-3.5, 4.0, -1000.125]])
    assert mols[1]["block"].startswith("chloride\n") and mols[1]["block"].rstrip().endswith("M  END")
    with pytest.raises(ValueError, match="V2000"):
        bb.sdf_molecules(SDF.replace("V2000", "V3000"))


def test_binding_energies_follow_the_protocol():
    bb = _script()
    terms = {"x": {"dg_exp": -9.0, "coulomb": -10.0, "lj": -5.0, "nonpolar": -2.0, "apolar": -3.0}}
    polar = {"x": {"delta_polar": 8.0, "delta_reference": -2.4}}
    assert bb.binding_energies(terms, polar, "eps4") == {"x": pytest.approx(-10.0 / 4 - 5.0 + 8.0 - 2.0)}
    assert bb.binding_energies(terms, polar, "eps1-fractional") == {"x": pytest.approx(-10.0 - 5.0 + 8.0 - 2.0)}
    assert bb.binding_energies(terms, polar, "eps4", "apolar") == {"x": pytest.approx(-10.0 / 4 - 5.0 + 8.0 - 3.0)}
    # the Gaussian dielectric adds its reaction field to Coulomb/pdie; per-residue dielectrics take the
    # electrostatics of their own reference state on the grid
    assert bb.binding_energies(terms, polar, "gauss") == {"x": pytest.approx(-10.0 - 5.0 + 8.0 - 2.0)}
    assert bb.binding_energies(terms, polar, "residue-eps") == {"x": pytest.approx(-2.4 - 5.0 + 8.0 - 2.0)}


def test_correlations_and_their_bootstrap_intervals():
    pytest.importorskip("scipy")
    bb = _script()
    x = np.arange(12.0)
    exact = bb.correlations(x, 2.0 * x + 1.0)
    for value, lo, hi in exact.values():
        assert value == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    y = x + rng.normal(0.0, 3.0, x.size)
    noisy = bb.correlations(x, y)
    from scipy import stats

    assert noisy["pearson"][0] == pytest.approx(stats.pearsonr(x, y)[0])
    assert noisy["kendall"][0] == pytest.approx(stats.kendalltau(x, y)[0])
    for value, lo, hi in noisy.values():
        assert -1.0 <= lo < value < hi <= 1.0
    assert bb.correlations(x, y) == noisy  # fixed seed


def test_bootstrap_intervals_match_the_normal_theory_interval():
    pytest.importorskip("scipy")
    bb = _script()
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    y = 0.75 * x + rng.normal(size=400)
    r, lo, hi = bb.correlations(x, y)["pearson"]
    # Fisher's z interval; for bivariate normal data the percentile bootstrap agrees within its Monte Carlo
    # error (here 0.002), while the bounds of a 90 % interval would lie 0.012 inside
    half = 1.959964 / math.sqrt(x.size - 3)
    assert lo == pytest.approx(math.tanh(math.atanh(r) - half), abs=0.006)
    assert hi == pytest.approx(math.tanh(math.atanh(r) + half), abs=0.006)


def _molecule(openmm, n, shift):
    """A chain of n charged Lennard-Jones particles with bonds, angles, torsions and 1-2/1-3 exclusions."""
    s = openmm.System()
    bonds, angles, torsions = openmm.HarmonicBondForce(), openmm.HarmonicAngleForce(), openmm.PeriodicTorsionForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(1.2)
    nb.setReactionFieldDielectric(78.54)
    for i in range(n):
        s.addParticle(12.0)
        nb.addParticle(0.3 * (-1) ** i + 0.05 * shift, 0.3, 0.4 + 0.1 * i)
    for i in range(n - 1):
        bonds.addBond(i, i + 1, 0.15, 2.0e5)
    for i in range(n - 2):
        angles.addAngle(i, i + 1, i + 2, 1.9, 300.0)
    for i in range(n - 3):
        torsions.addTorsion(i, i + 1, i + 2, i + 3, 3, 0.2, 1.5)
    nb.createExceptionsFromBonds([(i, i + 1) for i in range(n - 1)], 0.8333, 0.5)
    for f in (bonds, angles, torsions, nb, openmm.CMMotionRemover()):
        s.addForce(f)
    rng = np.random.default_rng(shift)
    xyz = np.cumsum(rng.normal(0.0, 0.1, (n, 3)) + [0.12, 0.0, 0.0], axis=0)
    return s, xyz


def test_merged_systems_add_up_when_far_apart():
    openmm = pytest.importorskip("openmm")
    bb = _script()

    def energy(system, xyz):
        ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(xyz)
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    (a, xa), (b, xb) = _molecule(openmm, 6, 1), _molecule(openmm, 5, 2)
    xb = xb + [5.0, 0.0, 0.0]  # beyond the 1.2 nm cutoff: no interaction left
    merged = bb._merge([a, b])
    assert merged.getNumParticles() == 11
    assert energy(merged, np.vstack([xa, xb])) == pytest.approx(energy(a, xa) + energy(b, xb), rel=1e-12)
    # within the cutoff the two molecules interact
    assert energy(merged, np.vstack([xa, xb - [4.5, 0.0, 0.0]])) != pytest.approx(energy(a, xa) + energy(b, xb))
    a.addForce(openmm.CustomExternalForce("x"))
    with pytest.raises(ValueError, match="CustomExternalForce"):
        bb._merge([a, b])
