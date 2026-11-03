import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from pbsolve.openmm_bridge import charges_from_system, positions_angstrom, radii_from_topology  # noqa: E402


def _toy():
    top = app.Topology()
    res = top.addResidue("LIG", top.addChain())
    el = app.element
    n = top.addAtom("N", el.nitrogen, res)
    hn = top.addAtom("HN", el.hydrogen, res)
    c = top.addAtom("C", el.carbon, res)
    hc = top.addAtom("HC", el.hydrogen, res)
    top.addBond(n, hn)
    top.addBond(n, c)
    top.addBond(c, hc)
    system = openmm.System()
    nb = openmm.NonbondedForce()
    for q in (-0.4, 0.3, 0.2, -0.1):
        system.addParticle(12.0)
        nb.addParticle(q, 0.3, 0.5)
    system.addForce(nb)
    return top, system


def test_radii_schemes():
    top, _ = _toy()
    np.testing.assert_allclose(radii_from_topology(top, "mbondi2"), [1.55, 1.3, 1.7, 1.2])
    np.testing.assert_allclose(radii_from_topology(top, "bondi"), [1.55, 1.2, 1.7, 1.2])
    np.testing.assert_allclose(radii_from_topology(top, "parse"), [1.5, 1.0, 1.7, 1.0])


def test_charges_and_positions():
    _, system = _toy()
    np.testing.assert_allclose(charges_from_system(system), [-0.4, 0.3, 0.2, -0.1])
    pos = unit.Quantity(np.array([[0.1, 0.2, 0.3]]), unit.nanometer)
    np.testing.assert_allclose(positions_angstrom(pos), [[1.0, 2.0, 3.0]])
