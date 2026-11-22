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
    # pair terms of the first frames: rows sum to the residues' polar shares, all to delta_polar
    pairs_csv = tmp_path / "pairs.csv"
    short = args + ["--stop", "2", "--csv", str(frames_csv), "--per-residue", str(residues_csv)]
    assert cli(short + ["--pairwise", str(pairs_csv)]) == 0
    frames, residues = list(csv.DictReader(open(frames_csv))), list(csv.DictReader(open(residues_csv)))
    pairs = list(csv.DictReader(open(pairs_csv)))
    assert len(pairs) == 10 * 10
    assert sum(float(p["delta_polar"]) for p in pairs) == pytest.approx(mean(frames, "delta_polar"), rel=1e-8)
    rows = {}
    for p in pairs:
        rows[p["residue"]] = rows.get(p["residue"], 0.0) + float(p["delta_polar"])
    for row in residues:
        assert rows[row["residue"]] == pytest.approx(float(row["delta_polar"]), abs=1e-6)
    assert cli(args + ["--pairwise", str(pairs_csv), "--equation", "npbe", "--ionic-strength", "0.1"]) == 2


def _group_sums(labels, names, values):
    return np.bincount([names.index(lab) for lab in labels], weights=values, minlength=len(names))


def test_residue_pair_terms_are_the_linear_response():
    x, q, r = synthetic_molecule()
    labels = [f"R{i // 5}" for i in range(len(q))]
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    g = s.make_grid(x, r)
    names, e = pbsolve.residue_pair_terms(s, x, q, r, labels, grid=g, workers=2)
    assert names == [f"R{k}" for k in range(6)]
    res = s.polar_solvation(x, q, r, grid=g)
    assert e.sum() == pytest.approx(res.energy, rel=1e-9)
    np.testing.assert_allclose(e.sum(axis=1), _group_sums(labels, names, res.atom_energies), rtol=0, atol=1e-6)

    def alone(*groups):  # ΔG_polar with the charges of these residues only
        keep = np.isin(labels, [names[k] for k in groups])
        return s.polar_solvation(x, np.where(keep, q, 0.0), r, grid=g).energy

    single = [alone(a) for a in range(6)]
    np.testing.assert_allclose(np.diag(e), single, rtol=0, atol=1e-6)
    for a, b in ((0, 1), (2, 5), (3, 4)):
        assert e[a, b] + e[b, a] == pytest.approx(alone(a, b) - single[a] - single[b], abs=1e-6)
    # with focusing and analytic boundary values the halves differ slightly; on one level with
    # zero boundary values the operator is symmetric
    assert 1e-6 < np.abs(e - e.T).max() < 1e-4 * np.abs(e).max()
    one = pbsolve.Solver(platform="CPU", ionic_strength=0.15, bcfl="zero")
    _, e1 = pbsolve.residue_pair_terms(one, x, q, r, labels, grid=pbsolve.GridSpec(g.center, g.dims, g.fine_length))
    np.testing.assert_allclose(e1, e1.T, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="labels"):
        pbsolve.residue_pair_terms(s, x, q, r, labels[:-1])
    npbe = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.1, equation="npbe")
    with pytest.raises(ValueError, match="per-atom"):
        pbsolve.residue_pair_terms(npbe, x, q, r, labels)
    # the energy in a membrane potential is linear in the charges and has no pair terms
    fake = type("R", (), {"atom_energies": np.zeros(len(q)), "membrane_potential_energy": 0.1})()
    with pytest.raises(ValueError, match="membrane potential"):
        pbsolve.mmpbsa._pair_matrix([fake], np.zeros(len(q), dtype=np.int64), [])


def test_binding_pair_terms():
    x, q, r = synthetic_molecule()
    labels = ["A"] * 7 + ["B"] * 8 + ["C"] * 5 + ["L1"] * 4 + ["L2"] * 6
    lig = np.arange(20, 30)
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    names, d = pbsolve.pbsa_pair_terms(s, x, q, r, lig, labels)
    f = pbsolve.pbsa_solvation_frame(s, x, q, r, lig)
    assert names == ["A", "B", "C", "L1", "L2"]
    assert d.sum() == pytest.approx(f.delta_polar, rel=1e-9)
    shares = f.residue_terms(labels)
    np.testing.assert_allclose(d.sum(axis=1), [shares[n][0] for n in names], rtol=0, atol=1e-6)
    # receptor–ligand pairs are the complex's alone; pairs within one partner lose that partner's
    _, c = pbsolve.residue_pair_terms(s, x, q, r, labels, grid=s.make_grid(x, r))
    np.testing.assert_allclose(d[:3, 3:], c[:3, 3:], rtol=0, atol=1e-9)
    _, rec = pbsolve.residue_pair_terms(s, x[:20], q[:20], r[:20], labels[:20], grid=s.make_grid(x, r))
    np.testing.assert_allclose(d[:3, :3], c[:3, :3] - rec, rtol=0, atol=1e-6)


def test_apolar_terms_per_residue():
    x, _, r = synthetic_molecule()
    labels = [f"R{i // 7}" for i in range(len(r))]
    a = pbsolve.apolar_energy(x, r, epsilons=np.full(len(r), 0.1))
    res = a.residue_terms(labels)
    assert list(res) == [f"R{k}" for k in range(5)]
    assert sum(v[0] for v in res.values()) == pytest.approx(a.surface, rel=1e-12)
    assert sum(v[1] for v in res.values()) == pytest.approx(a.dispersion, rel=1e-12)
    assert a.pressure_volume != 0  # ... which belongs to no residue
    no_surface = pbsolve.apolar_energy(x, r, gamma=0.0, epsilons=np.full(len(r), 0.1))
    assert all(v[0] == 0.0 for v in no_surface.residue_terms(labels).values())
    with pytest.raises(ValueError, match="labels"):
        a.residue_terms(labels[:-1])


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
