"""pbsolve.prep: PDB -> PQR (pdb2pqr --noopt compatible).

Reference PQR files in tests/data were written by pdb2pqr 3.8.0 from 1YCR
(PDB, CC0):  pdb2pqr --ff=FF [--noopt] --drop-water --keep-chain --whitespace
[--titration-state-method propka --with-ph 4].  scripts/validate_prep.py
compares more structures (1TII, 1HPV, 5DK3 with repair and debumping).
"""

import importlib.util
import pathlib

import numpy as np
import pytest

import pbsolve
from pbsolve import prep

DATA = pathlib.Path(__file__).parent / "data"
PDB = DATA / "1ycr.pdb"


@pytest.mark.parametrize("ff", ["amber", "charmm"])
@pytest.mark.parametrize("mode", ["opt", "noopt"])
def test_byte_identical_to_pdb2pqr(ff, mode):
    mol, rep = prep.prepare(PDB, ff, optimize=mode == "opt")
    assert rep.pqr_text == (DATA / f"1ycr_{ff}_{mode}_pdb2pqr.pqr").read_text()
    assert rep.n_atoms == mol.n_atoms == 1670
    assert rep.n_hydrogens_added == 851
    assert rep.repaired == ["VAL A109:OXT"]
    assert rep.net_charge == pytest.approx(3.0, abs=1e-9)
    assert rep.states["HIS A73"] == "HID" and rep.states["GLU A25"] == "NGLU" and rep.states["ASN B29"] == "CASN"
    assert rep.flipped == (["HIS A73", "ASN A79", "ASN A106"] if mode == "opt" else [])


def test_molecule_matches_pqr_text(tmp_path):
    mol, rep = prep.prepare(PDB, "amber")
    p = tmp_path / "x.pqr"
    p.write_text(rep.pqr_text)
    back = pbsolve.io.load(p)
    np.testing.assert_allclose(back.coords, mol.coords, atol=5e-4)
    np.testing.assert_allclose(back.charges, mol.charges, atol=5e-5)
    np.testing.assert_allclose(back.radii, mol.radii, atol=5e-5)
    assert set(mol.elements) == {"C", "H", "N", "O", "S"}


def test_optimisation_moves_only_hydrogens_and_flipped_side_chains():
    opt, _ = prep.prepare(PDB, "amber")
    noopt, _ = prep.prepare(PDB, "amber", optimize=False)
    key = lambda m: {(c, r, n): x for c, r, n, x in zip(m.chains, m.resids, m.names, m.coords)}  # noqa: E731
    a, b = key(opt), key(noopt)
    assert a.keys() == b.keys()
    moved = {k for k in a if np.abs(a[k] - b[k]).max() > 1e-6}
    heavy = {k for k in moved if not k[2].startswith("H")}
    assert {(c, int(r)) for c, r, _ in heavy} <= {("A", 73), ("A", 79), ("A", 106)}
    assert len(moved) == 34


def test_model_pka_rules():
    _, std = prep.prepare(PDB, "amber")
    _, ph7 = prep.prepare(PDB, "amber", ph=7.0)
    assert ph7.pqr_text == std.pqr_text  # pH 7 = standard states
    _, ph4 = prep.prepare(PDB, "amber", ph=4.0)
    v = ph4.states
    assert v["HIS A73"] == "HIP" and v["GLU A95"] == "GLH"  # pKa 6.0 / 4.3 > 4
    assert v["ASP A84"] == "ASP"  # pKa 3.9 < 4
    assert v["GLU A25"] == "NGLU"  # termini are not titrated (AMBER)
    assert ph4.net_charge > std.net_charge


@pytest.mark.skipif(importlib.util.find_spec("propka") is None, reason="PROPKA not installed")
@pytest.mark.parametrize("mode", ["opt", "noopt"])
def test_propka_matches_pdb2pqr(mode):
    _, rep = prep.prepare(PDB, "amber", titration="propka", ph=4.0, optimize=mode == "opt")
    assert rep.pqr_text == (DATA / f"1ycr_amber_{mode}_propka_ph4_pdb2pqr.pqr").read_text()


def test_ligand_from_mol2(tmp_path):
    lig = tmp_path / "lig.mol2"
    lig.write_text(
        "@<TRIPOS>MOLECULE\nLIG\n3 2\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n"
        "1 O1 0.0 0.0 0.0 O.3 1 LIG -0.8\n2 H1 0.96 0.0 0.0 H 1 LIG 0.4\n3 H2 -0.24 0.93 0.0 H 1 LIG 0.4\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n"
    )
    mol, rep = prep.prepare(PDB, "amber", ligand=lig)
    assert mol.n_atoms == 1673
    assert list(mol.resnames[-3:]) == ["LIG"] * 3 and list(mol.chains[-3:]) == ["L"] * 3
    np.testing.assert_allclose(mol.radii[-3:], [1.5, 1.2, 1.2])
    assert rep.net_charge == pytest.approx(3.0)


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="force field"):
        prep.prepare(PDB, "gromos")
    na = tmp_path / "na.pdb"
    na.write_text("ATOM      1  P    DA A   1       0.000   0.000   0.000  1.00  0.00           P\nEND\n")
    with pytest.raises(NotImplementedError, match="nucleic"):
        prep.prepare(na)
    lines = [ln for ln in PDB.read_text().splitlines() if ln.startswith("ATOM")]
    broken = tmp_path / "broken.pdb"
    broken.write_text("\n".join(ln for i, ln in enumerate(lines) if i % 5) + "\nEND\n")  # 20 % of atoms missing
    with pytest.raises(ValueError, match="too incomplete"):
        prep.prepare(broken)


def test_cli(tmp_path):
    out = tmp_path / "o.pqr"
    assert prep.main([str(PDB), str(out), "--ff", "amber", "--quiet"]) == 0
    assert out.read_text() == (DATA / "1ycr_amber_opt_pdb2pqr.pqr").read_text()
    assert prep.main([str(PDB), str(out), "--noopt", "--quiet"]) == 0
    assert out.read_text() == (DATA / "1ycr_amber_noopt_pdb2pqr.pqr").read_text()
    from pbsolve.__main__ import main

    assert main(["prep", str(PDB), str(tmp_path / "p.pqr"), "--quiet"]) == 0
    assert main(["convert", str(PDB), "--ff", "amber", str(tmp_path / "c.pqr")]) == 0
    assert pbsolve.io.load(tmp_path / "c.pqr").n_atoms == 1670
