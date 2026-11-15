"""One interior dielectric per atom (`dielectrics=`, DelPhi-style multiple dielectrics)."""

import dataclasses
import json

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli
from pbsolve.analytic import kirkwood_energy, two_sphere_energy

from conftest import synthetic_molecule, write_pqr

KW = dict(platform="CPU", grid_spacing=0.6, fadd=10.0)

# The spheres of tests/test_analytic.py: one atom of radius A (A2) each, zero-radius charges inside.
A, A2 = 5.0, 4.0
QPOS = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]])
QVAL = np.array([1.0, -1.0, 0.5])
QPOS2 = np.array([[0.0, -1.0, 1.5], [1.2, 0.3, -0.4]])
QVAL2 = np.array([-1.0, 0.7])


def test_dielectrics_equal_to_pdie_change_nothing():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(ionic_strength=0.15, pdie=2.0, **KW)
    eps = np.full(len(q), 2.0)
    a, b = s.polar_solvation(x, q, r), s.polar_solvation(x, q, r, dielectrics=eps)
    assert (a.energy, a.solvated_kt, a.reference_kt) == (b.energy, b.solvated_kt, b.reference_kt)
    assert np.array_equal(a.atom_energies, b.atom_energies)
    for reference in (False, True):
        pa, *_, ea = s.potential(x, q, r, reference=reference)
        pb, *_, eb = s.potential(x, q, r, reference=reference, dielectrics=eps)
        assert np.array_equal(pa, pb) and ea == eb
        ma, mb = s.maps(x, q, r, reference=reference), s.maps(x, q, r, reference=reference, dielectrics=eps)
        assert all(np.array_equal(ma[k], mb[k]) for k in ("eps_x", "eps_y", "eps_z", "kappa_sq"))
    frames = np.stack([x, x + 0.1])
    assert np.array_equal(s.polar_solvation_batch(frames, q, r), s.polar_solvation_batch(frames, q, r, dielectrics=eps))


def test_each_interior_face_takes_its_atoms_dielectric():
    x, q, r = synthetic_molecule()
    eps = 1.0 + np.arange(len(q)) % 7  # 1 … 7
    s = pbsolve.Solver(pdie=1.5, **KW)
    plain = s.maps(x, q, r)
    for reference, outside in ((False, 78.54), (True, 1.5)):
        m = s.maps(x, q, r, reference=reference, dielectrics=eps)
        origin, h = np.asarray(m["origin"]), np.asarray(m["spacing"])
        checked = 0
        for axis, key in enumerate(("eps_x", "eps_y", "eps_z")):
            inside = plain[key] == 1.5
            assert np.all(m[key][~inside] == outside)
            for idx in np.argwhere(inside)[::11]:  # a sample of the interior faces
                p = origin + idx * h
                p[axis] += 0.5 * h[axis]
                power = np.sum((x - p) ** 2, axis=1) - r**2
                assert m[key][tuple(idx)] == eps[np.argmin(power)]
                checked += 1
        assert checked > 300


@pytest.mark.parametrize(
    "eps1, kw, tol",
    [(4.0, dict(), 0.012), (4.0, dict(ionic_strength=0.15, pdie=2.0), 0.012), (20.0, dict(), 0.004)],
)
def test_one_sphere_with_its_own_dielectric_converges_to_kirkwood(eps1, kw, tol):
    """The interior (ε_1) is the same in both states; the reference has pdie outside:
    ΔG = K(ε_1 | sdie, ions) − K(ε_1 | pdie) with K the Kirkwood energy."""
    coords = np.vstack([[0.0, 0.0, 0.0], QPOS])
    charges, radii = np.r_[0.0, QVAL], np.r_[A, 0.0, 0.0, 0.0]
    pdie, ions = kw.get("pdie", 1.0), kw.get("ionic_strength", 0.0)
    exact = kirkwood_energy(QPOS, QVAL, A, pdie=eps1, ionic_strength=ions) - kirkwood_energy(
        QPOS, QVAL, A, pdie=eps1, sdie=pdie
    )
    err = {}
    for h in (0.5, 0.25):
        s = pbsolve.Solver(platform="CPU", grid_spacing=h, **kw)
        err[h] = (s.polar_solvation(coords, charges, radii, dielectrics=np.full(4, eps1)).energy - exact) / abs(exact)
    # measured: −1.54 / −0.92 %, −1.93 / −0.87 %, −0.15 / −0.22 % (docs/VALIDATION.md)
    assert abs(err[0.5]) < 2 * tol and abs(err[0.25]) < tol


@pytest.mark.parametrize("kw", [dict(), dict(ionic_strength=0.15, pdie=2.0)])
def test_two_spheres_with_their_own_dielectrics(kw):
    c = np.array([[0.0, 0.0, 0.0], [15.8, 0.0, 0.0]])
    pos, q = np.vstack([QPOS, QPOS2 + c[1]]), np.r_[QVAL, QVAL2]
    exact = two_sphere_energy(
        pos, q, c, [A, A2], pdie=kw.get("pdie", 1.0), ionic_strength=kw.get("ionic_strength", 0.0), sphere_pdie=(4.0, 10.0)
    )
    coords = np.vstack([c[0], QPOS, c[1], QPOS2 + c[1]])
    charges, radii = np.r_[0.0, QVAL, 0.0, QVAL2], np.r_[A, 0, 0, 0, A2, 0, 0]
    eps = np.r_[4.0, 4.0, 4.0, 4.0, 10.0, 10.0, 10.0]
    e = pbsolve.Solver(platform="CPU", grid_spacing=0.25, **kw).polar_solvation(coords, charges, radii, dielectrics=eps)
    # measured −0.32 % and −0.74 % (−0.83 / −1.56 % at 0.5 Å)
    assert e.energy == pytest.approx(exact, rel=0.01)


def test_two_sphere_reference_with_per_sphere_dielectrics():
    c = np.array([[0.0, 0.0, 0.0], [15.8, 0.0, 0.0]])
    pos, q = np.vstack([QPOS, QPOS2 + c[1]]), np.r_[QVAL, QVAL2]
    for kw in (dict(pdie=2.0), dict(pdie=4.0, ionic_strength=0.1)):
        same = two_sphere_energy(pos, q, c, [A, A2], sphere_pdie=(kw["pdie"], kw["pdie"]), **kw)
        assert same == pytest.approx(two_sphere_energy(pos, q, c, [A, A2], **kw), rel=1e-13)
    # Neutral spheres far apart: two Kirkwood spheres, each with its interior and pdie outside.
    q0, q1 = np.array([0.5, -0.5]), np.array([0.3, -0.3])
    p0, p1 = QPOS[:2], QPOS2
    kirk = lambda p, qq, a, e: kirkwood_energy(p, qq, a, pdie=e) - kirkwood_energy(p, qq, a, pdie=e, sdie=2.0)
    k = kirk(p0, q0, A, 4.0) + kirk(p1, q1, A2, 10.0)
    for d, tol in ((50.0, 3e-4), (300.0, 1e-6)):
        far = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        e = two_sphere_energy(np.vstack([p0, p1 + far[1]]), np.r_[q0, q1], far, [A, A2], pdie=2.0, sphere_pdie=(4.0, 10.0))
        assert e == pytest.approx(k, rel=tol)  # dipole–dipole terms fall off as 1/d³


def test_residue_dielectrics():
    q = np.array([0.6, 0.45, -0.3, 0.3, -0.2, -0.5, -0.49, 0.25, 0.25])
    res = ["A:1", "A:1", "A:2", "A:2", "A:3", "A:3", "B:1", "C:1", "C:1"]
    eps = pbsolve.residue_dielectrics(q, res, charged=4.0, neutral=1.0)
    # A:2 is neutral (−0.3 + 0.3), A:3 has −0.7 e, B:1 only −0.49 e, C:1 exactly 0.5 e
    assert list(eps) == [4.0, 4.0, 1.0, 1.0, 4.0, 4.0, 1.0, 4.0, 4.0]
    loose = pbsolve.residue_dielectrics(q, res, charged=2.7, neutral=1.1, threshold=0.4)
    assert list(loose) == [2.7] * 2 + [1.1] * 2 + [2.7] * 5
    with pytest.raises(ValueError):
        pbsolve.residue_dielectrics(q, res[:3], charged=4.0, neutral=1.0)
    with pytest.raises(ValueError):
        pbsolve.residue_dielectrics(q, res, charged=0.0, neutral=1.0)


def test_mmpbsa_with_dielectrics():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(ionic_strength=0.1, **KW)
    lig = np.arange(len(q) - 6, len(q))
    rec = np.arange(len(q) - 6)
    eps = 1.0 + np.arange(len(q)) % 4
    f = pbsolve.pbsa_solvation_frame(s, x, q, r, lig, dielectrics=eps)
    grid = s.make_grid(x, r)
    direct = [
        s.polar_solvation(x[i], q[i], r[i], grid=grid, dielectrics=eps[i]).energy for i in (np.arange(len(q)), rec, lig)
    ]
    assert [f.complex.polar, f.receptor.polar, f.ligand.polar] == direct
    plain = pbsolve.pbsa_solvation_frame(s, x, q, r, lig)
    assert pbsolve.pbsa_solvation_frame(s, x, q, r, lig, dielectrics=np.ones(len(q))) == plain
    assert f.delta_polar != plain.delta_polar
    with pytest.raises(ValueError):
        pbsolve.pbsa_solvation_frame(s, x, q, r, lig, dielectrics=eps[:-1])


def test_estimate_error_passes_the_dielectrics_on():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(**KW)
    eps = 1.0 + np.arange(len(q)) % 3
    est = pbsolve.estimate_error(s, x, q, r, placements=2, dielectrics=eps)
    assert est.energies[0] == s.polar_solvation(x, q, r, dielectrics=eps).energy
    assert pbsolve.estimate_error(s, x, q, r, placements=2, dielectrics=np.ones(len(q))).energies == (
        pbsolve.estimate_error(s, x, q, r, placements=2).energies
    )


def test_invalid_dielectrics():
    x, q, r = synthetic_molecule()
    n = len(q)
    s = pbsolve.Solver(**KW)
    for bad in (np.ones(n - 1), np.zeros(n), np.full(n, np.nan), -np.ones(n)):
        with pytest.raises(ValueError):
            s.polar_solvation(x, q, r, dielectrics=bad)
    for kw in (dict(surface="spl2"), dict(surface="gauss"), dict(interface="fractional"), dict(coulomb="analytic")):
        with pytest.raises(ValueError, match="atom dielectrics"):
            pbsolve.Solver(**KW, **kw).polar_solvation(x, q, r, dielectrics=np.ones(n))
    with pytest.raises(ValueError, match="2 entries for 1 systems"):
        s.polar_solvation_many([(x, q, r)], dielectrics=[None, None])
    with pytest.raises(ValueError, match="system 1"):
        s.polar_solvation_many([(x, q, r), (x, q, r)], dielectrics=[None, np.ones(3)])
    out = s.polar_solvation_many([(x, q, r), (x, q, r)], dielectrics=[None, np.full(n, 3.0)])
    assert out[0].energy == s.polar_solvation(x, q, r).energy
    assert out[1].energy == s.polar_solvation(x, q, r, dielectrics=np.full(n, 3.0)).energy


def test_cli_residue_dielectric(tmp_path, capsys):
    x, q, r = synthetic_molecule()
    pqr = tmp_path / "m.pqr"
    write_pqr(pqr, x, q, r)
    base = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--json"]
    out = []
    for extra in ([], ["--residue-dielectric", "1", "1"], ["--residue-dielectric", "4", "2"]):
        assert cli(base + extra) == 0
        out.append(json.loads(capsys.readouterr().out))
    assert out[1]["polar_kcal_mol"] == out[0]["polar_kcal_mol"]  # --pdie defaults to 1
    assert out[2]["residue_dielectric"] == [4.0, 2.0]
    mol = pbsolve.io.load(str(pqr))
    labels = [f"{c}:{n}:{i}{ic}" for c, n, i, ic in zip(mol.chains, mol.resnames, mol.resids, mol.icodes)]
    eps = pbsolve.residue_dielectrics(mol.charges, labels, charged=4.0, neutral=2.0)
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6)
    assert out[2]["polar_kcal_mol"] == s.polar_solvation(mol.coords, mol.charges, mol.radii, dielectrics=eps).energy


def test_cli_mmpbsa_residue_dielectric(tmp_path):
    gro = pbsolve.io.load("tests/data/synthetic.gro")
    rng = np.random.default_rng(3)
    mol = dataclasses.replace(gro, charges=rng.uniform(-0.4, 0.4, gro.n_atoms), radii=np.full(gro.n_atoms, 1.6))
    pqr = tmp_path / "synthetic.pqr"
    mol.write_pqr(pqr)
    args = ["mmpbsa", str(pqr), "tests/data/synthetic.xtc", "--ligand", "resid 2", "--platform", "CPU", "--spacing", "0.7"]
    rows = {}
    for name, extra in (("plain", []), ("same", ["--residue-dielectric", "1", "1"]), ("var", ["--residue-dielectric", "4", "2"])):
        assert cli(args + extra + ["--csv", str(tmp_path / f"{name}.csv")]) == 0
        rows[name] = (tmp_path / f"{name}.csv").read_text()
    # No residue of this molecule carries 0.5 e: every atom gets NEUTRAL.
    assert rows["same"] == rows["plain"] and rows["var"] != rows["plain"]


def test_delta_reference_is_the_interaction_in_the_reference_medium():
    x, q, r = synthetic_molecule()
    lig = np.arange(len(q) - 6, len(q))
    zeros = np.zeros(len(q))
    coulomb, _ = pbsolve.interaction_energy(x, q, zeros, zeros, lig)
    kt = pbsolve.kt_kcal_per_mol(298.15)
    for pdie in (1.0, 4.0):
        # coulomb="analytic": the reference energy is the Coulomb sum in pdie, exactly.
        exact = pbsolve.Solver(pdie=pdie, coulomb="analytic", **KW)
        assert pbsolve.pbsa_solvation_frame(exact, x, q, r, lig).delta_reference == pytest.approx(coulomb / pdie, rel=1e-12)
        # On the grid the self-energies cancel on the shared grid (measured −0.03 % for 1YCR at 0.35 Å).
        grid = pbsolve.Solver(pdie=pdie, platform="CPU", grid_spacing=0.35, fadd=10.0)
        assert pbsolve.pbsa_solvation_frame(grid, x, q, r, lig).delta_reference == pytest.approx(coulomb / pdie, rel=5e-3)
    # With atom dielectrics, reference interaction + ΔΔG_polar = the solvated states' difference.
    s = pbsolve.Solver(ionic_strength=0.1, **KW)
    eps = 1.0 + np.arange(len(q)) % 4
    f = pbsolve.pbsa_solvation_frame(s, x, q, r, lig, dielectrics=eps)
    rec = np.arange(len(q) - 6)
    g = s.make_grid(x, r)
    solvated = [
        s.polar_solvation(x[i], q[i], r[i], grid=g, dielectrics=eps[i]).solvated_kt * kt for i in (np.arange(len(q)), rec, lig)
    ]
    assert f.delta_reference + f.delta_polar == pytest.approx(solvated[0] - solvated[1] - solvated[2], rel=1e-12)
    assert f.as_dict()["delta_reference"] == f.delta_reference


def test_varied_dielectrics_reach_every_method():
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(ionic_strength=0.1, **KW)
    eps = 1.0 + np.arange(len(q)) % 5
    one = s.polar_solvation(x, q, r, dielectrics=eps)
    assert one.energy != s.polar_solvation(x, q, r).energy
    for reference, kt in ((False, one.solvated_kt), (True, one.reference_kt)):
        *_, e = s.potential(x, q, r, reference=reference, dielectrics=eps)
        assert e == pytest.approx(kt, rel=1e-12)
    frames = np.stack([x, x + 0.1])
    batch = s.polar_solvation_batch(frames, q, r, dielectrics=eps)
    assert list(batch) == [s.polar_solvation(f, q, r, dielectrics=eps).energy for f in frames]
