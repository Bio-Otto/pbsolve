"""Per-atom and per-residue decomposition of the solvation terms."""

import csv
import dataclasses

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli
from pbsolve.apbs import compare, find_apbs

from conftest import synthetic_molecule, write_pqr


def test_atom_energies_sum_to_the_energy():
    x, q, r = synthetic_molecule()
    kw = dict(platform="CPU", grid_spacing=0.6, fadd=10.0)
    cases = (
        dict(ionic_strength=0.15),
        dict(coulomb="analytic", interface="fractional"),
        dict(chgm="spl2"),
        dict(coulomb="analytic", chgm="spl2"),
    )
    # A lone charge of radius 0: its spl2 energy nodes (up to 1.5 h·√3 away) reach beyond the
    # region χ (2 h), where the reaction field is v − G rather than v.
    x = np.vstack([x, x.max(axis=0) + 4.0])
    q, r = np.append(q, 0.3), np.append(r, 0.0)
    for extra in cases:
        res = pbsolve.Solver(**kw, **extra).polar_solvation(x, q, r)
        assert res.atom_energies.shape == (len(q),)
        assert res.atom_energies.sum() == pytest.approx(res.energy, rel=1e-12)
        assert (res.atom_energies[q == 0] == 0).all()
    # The nonlinear energy with ions is a functional, not a sum over the charges; without
    # ions the equation is linear and decomposes as lpbe does.
    assert pbsolve.Solver(**kw, ionic_strength=0.1, equation="npbe").polar_solvation(x, q, r).atom_energies is None
    np.testing.assert_array_equal(
        pbsolve.Solver(**kw, equation="npbe").polar_solvation(x, q, r).atom_energies,
        pbsolve.Solver(**kw).polar_solvation(x, q, r).atom_energies,
    )


def test_binding_terms_per_atom_and_residue():
    x, q, r = synthetic_molecule()
    lig = np.arange(20, 30)
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    f = pbsolve.pbsa_solvation_frame(s, x, q, r, lig)
    polar, nonpolar = f.atom_terms()
    assert polar.sum() == pytest.approx(f.delta_polar, rel=1e-11)
    assert nonpolar.sum() == pytest.approx(f.delta_nonpolar + pbsolve.SITKOFF_BETA, rel=1e-11)  # β belongs to no atom
    labels = ["A"] * 10 + ["B"] * 10 + ["L"] * 10
    res = f.residue_terms(labels)
    assert list(res) == ["A", "B", "L"]
    assert sum(p for p, _ in res.values()) == pytest.approx(f.delta_polar, rel=1e-11)
    assert res["L"][0] == pytest.approx(polar[lig].sum(), rel=1e-12)
    with pytest.raises(ValueError, match="labels"):
        f.residue_terms(labels[:-1])
    # Frames compare by their totals; one built without per-atom terms cannot decompose.
    assert f == pbsolve.pbsa_solvation_frame(s, x, q, r, lig)
    with pytest.raises(ValueError, match="per-atom"):
        dataclasses.replace(f, ligand_atoms=None).atom_terms()


def test_per_residue_cli(tmp_path):
    gro = pbsolve.io.load("tests/data/synthetic.gro")
    rng = np.random.default_rng(3)
    mol = dataclasses.replace(gro, charges=rng.uniform(-0.4, 0.4, gro.n_atoms), radii=np.full(gro.n_atoms, 1.6))
    pqr = tmp_path / "synthetic.pqr"
    mol.write_pqr(pqr)
    args = ["mmpbsa", str(pqr), "tests/data/synthetic.xtc", "--ligand", "resid 2", "--platform", "CPU", "--spacing", "0.7"]
    frames_csv, residues_csv = tmp_path / "frames.csv", tmp_path / "residues.csv"
    assert cli(args + ["--csv", str(frames_csv), "--per-residue", str(residues_csv)]) == 0
    frames = list(csv.DictReader(open(frames_csv)))
    residues = list(csv.DictReader(open(residues_csv)))
    solute = ~mol.solvent_mask()  # the command strips the solvent
    assert len(residues) == len(set(zip(mol.chains[solute], mol.resids[solute]))) == 10
    mean = lambda rows, k: np.mean([float(row[k]) for row in rows])
    assert sum(float(row["delta_polar"]) for row in residues) == pytest.approx(mean(frames, "delta_polar"), rel=1e-10)
    assert sum(float(row["delta_nonpolar"]) for row in residues) == pytest.approx(
        mean(frames, "delta_nonpolar") + pbsolve.SITKOFF_BETA, rel=1e-10
    )
    assert cli(args + ["--per-residue", str(residues_csv), "--equation", "npbe", "--ionic-strength", "0.1"]) == 2


@pytest.mark.skipif(find_apbs() is None, reason="APBS not installed")
@pytest.mark.parametrize("bcfl, tol", [("mdh", 1e-5), ("zero", 1e-8)])
def test_atom_energies_against_apbs(tmp_path, bcfl, tol):
    # APBS `calcenergy comps`: ½ q_i u(r_i) of every atom in each state.
    x, q, r = synthetic_molecule()
    write_pqr(tmp_path / "m.pqr", x, q, r)
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, bcfl=bcfl, dime_policy="apbs")
    rep = compare(tmp_path / "m.pqr", s, workdir=tmp_path / "w", maps=False, inject=False, atom_energies=True)
    assert rep.atom_energies["max_rel_to_max"] < tol and rep.passed()
    with pytest.raises(ValueError, match="spl2"):
        compare(tmp_path / "m.pqr", pbsolve.Solver(chgm="spl2"), workdir=tmp_path / "w2", atom_energies=True)
