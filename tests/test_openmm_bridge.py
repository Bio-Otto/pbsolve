import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import pbsolve  # noqa: E402
from pbsolve.openmm_bridge import (  # noqa: E402
    OpenMMPBSA,
    charges_from_system,
    lennard_jones_from_system,
    positions_angstrom,
    radii_from_topology,
)


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


def test_lennard_jones_parameters():
    _, system = _toy()
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    nb.setParticleParameters(3, -0.1, 1.0, 0.0)  # ε = 0 with a placeholder σ of 1 nm
    radii, eps = lennard_jones_from_system(system)
    np.testing.assert_allclose(radii, [2 ** (1 / 6) * 1.5] * 3 + [0.0], rtol=1e-14)
    np.testing.assert_allclose(eps, [0.5 / 4.184] * 3 + [0.0], rtol=1e-14)


def test_openmm_pbsa_passes_the_atom_dielectrics_on():
    top, system = _toy()
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.3, 0.0], [2.5, 1.5, 0.2]])
    solver = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=8.0)
    eps = np.array([4.0, 4.0, 1.0, 1.0])
    pbsa = OpenMMPBSA(system, top, ligand_atoms=[2, 3], solver=solver, dielectrics=eps)
    radii = radii_from_topology(top, "mbondi2")
    q = charges_from_system(system)
    assert pbsa.frame(xyz) == pbsolve.pbsa_solvation_frame(solver, xyz, q, radii, [2, 3], dielectrics=eps)
    assert pbsa.frame(xyz) != OpenMMPBSA(system, top, ligand_atoms=[2, 3], solver=solver).frame(xyz)
