"""APBS-compatibility helpers that do not need the APBS executable."""

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs


def test_apbs_dime_reset_rule():
    # APBS 3.4.1 MGparm_check: dime > 65 must be 32c+1, otherwise it is reduced.
    assert pbsolve.apbs_effective_dime(97) == 97
    assert pbsolve.apbs_effective_dime(121) == 97
    assert pbsolve.apbs_effective_dime(81) == 65
    assert pbsolve.apbs_effective_dime(57) == 57
    assert pbsolve.apbs_effective_dime(59) is None


def test_dime_policies(molecule):
    coords, _, radii = molecule
    big = coords * 3.0  # large enough for > 65 nodes per axis at 0.4 Å
    default = pbsolve.Solver(grid_spacing=0.4).make_grid(big, radii)
    assert all((n - 1) % 8 == 0 for n in default.dims)
    if default.apbs_dims != default.dims:
        with pytest.raises(ValueError, match="dime policy 'apbs'"):
            default.to_apbs()
    compat = pbsolve.Solver(grid_spacing=0.4, dime_policy="apbs").make_grid(big, radii)
    assert compat.apbs_dims == compat.dims
    assert max(compat.fine_spacing) <= 0.4 + 1e-12
    assert "dime" in compat.to_apbs()
    with pytest.raises(ValueError):
        pbsolve.Solver(dime_policy="psize")


def test_focusing_levels_follow_apbs():
    g = pbsolve.GridSpec([0.0, 0.0, 0.0], [33, 33, 33], [64.0] * 3, [12.8] * 3)  # ratio 0.2
    levels = g.levels
    assert len(levels) == 3
    np.testing.assert_allclose(levels[0]["spacing"], [2.0] * 3)
    np.testing.assert_allclose(levels[1]["spacing"], [2.0 * np.sqrt(0.2)] * 3)
    np.testing.assert_allclose(levels[2]["spacing"], [0.4] * 3)
    assert len(pbsolve.GridSpec([0.0] * 3, [33] * 3, [20.0] * 3).levels) == 1


def test_to_apbs_is_full_precision(molecule):
    coords, _, radii = molecule
    g = pbsolve.Solver(grid_spacing=0.5, dime_policy="apbs").make_grid(coords + 1.0 / 3.0, radii)
    lines = dict(line.split(" ", 1) for line in g.to_apbs().splitlines())
    assert [float(v) for v in lines["fgcent"].split()] == list(g.center)
    assert [float(v) for v in lines["fglen"].split()] == list(g.fine_length)


def test_input_round_trip(tmp_path, molecule, molecule_pqr):
    coords, _, radii = molecule
    s = pbsolve.Solver(grid_spacing=0.5, ionic_strength=0.1, pdie=2.0, bcfl="sdh", chgm="spl4", dime_policy="apbs")
    grid = s.make_grid(coords, radii)
    solv = apbs.elec_from_solver(s, grid, "solv", writes=[("pot", "p")])
    refr = apbs.elec_from_solver(s, grid, "ref", reference=True)
    path = tmp_path / "in.apbs"
    path.write_text(apbs.write_input([molecule_pqr], [solv, refr], ["solv - ref"]))
    back = apbs.parse_input(path)
    assert [e.name for e in back.elecs] == ["solv", "ref"]
    e = back.elecs[0]
    assert e.grid.dims == grid.dims and e.grid.center == grid.center
    assert e.grid.coarse_length == grid.coarse_length and e.grid.fine_length == grid.fine_length
    assert (e.pdie, e.sdie, e.bcfl, e.chgm, e.srad, e.sdens) == (2.0, 78.54, "sdh", "spl4", 1.4, 10.0)
    assert e.solver().params["chgm"] == "spl4" and back.elecs[1].chgm == "spl4"
    assert e.ions == [(1.0, 0.1, 2.0), (-1.0, 0.1, 2.0)]
    assert back.elecs[1].sdie == 2.0 and back.elecs[1].ions == []
    assert back.prints == [[(1, "solv"), (-1, "ref")]]


def test_run_input_preconditioner_option(tmp_path, molecule, molecule_pqr, capsys):
    """`pbsolve apbs run --no-apbs` honours --preconditioner; all choices give the same energies."""
    coords, _, radii = molecule
    s = pbsolve.Solver(grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, dime_policy="apbs")
    grid = s.make_grid(coords, radii)
    elecs = [apbs.elec_from_solver(s, grid, "solv"), apbs.elec_from_solver(s, grid, "ref", reference=True)]
    path = tmp_path / "user.in"
    path.write_text(apbs.write_input([molecule_pqr.name], elecs, ["solv - ref"]))
    kj = {}
    for pre in (None, "multigrid-f32", "multigrid", "jacobi"):
        rep = apbs.run_input(path, platform="CPU", with_apbs=False, preconditioner=pre)
        kj[pre] = [e["pbsolve_kj"] for e in rep.elecs]
    assert kj[None] == kj["multigrid-f32"]  # the default, bit for bit
    for pre in ("multigrid", "jacobi"):
        assert kj[pre] == pytest.approx(kj[None], rel=1e-7)
    assert apbs.main(["run", str(path), "--platform", "CPU", "--no-apbs", "--preconditioner", "multigrid"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        apbs.main(["run", str(path), "--no-apbs", "--preconditioner", "ilu"])


def test_parse_input_mol_centering_and_reset(tmp_path, molecule, molecule_pqr):
    coords = molecule[0]
    path = tmp_path / "x.in"
    path.write_text(
        f"read\n mol pqr {molecule_pqr}\nend\n"
        "elec name a\n mg-auto\n dime 121 97 65\n cglen 60 60 60\n fglen 40 40 40\n"
        " cgcent mol 1\n fgcent mol 1\n mol 1\n lpbe\n bcfl mdh\n pdie 2\n sdie 78.54\n"
        " srfm mol\n chgm spl0\n sdens 10\n srad 1.4\n swin 0.3\n temp 298.15\n calcenergy total\nend\nquit\n"
    )
    inp = apbs.parse_input(path)
    g = inp.elecs[0].grid
    assert g.dims == [97, 97, 65]  # APBS resets 121 -> 97
    assert any("resets dime" in n for n in inp.notes)
    want = 0.5 * (coords.max(axis=0) + coords.min(axis=0))
    np.testing.assert_allclose(g.center, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "line, match",
    [(" smpbe\n", "size-modified"), (" srfm smol\n", "srfm smol"), (" chgm spl1\n", "chgm spl1"), (" bcfl focus\n", "bcfl")],
)
def test_parse_input_rejects_unsupported(tmp_path, molecule_pqr, line, match):
    path = tmp_path / "x.in"
    path.write_text(f"read\n mol pqr {molecule_pqr}\nend\nelec\n mg-manual\n{line}end\nquit\n")
    with pytest.raises(NotImplementedError, match=match):
        apbs.parse_input(path)


CANNED = """
NOsh:  Bad dime[0]  = 121 (2 nlev)!
NOsh:  Reset dime[0] to 97 and (nlev = 4).
CALCULATION #1 (solv): MULTIGRID
  Grid dimensions: 97 x 97 x 65
  Grid spacings: 0.600 x 0.625 x 0.500
  Grid lengths: 57.600 x 60.000 x 32.000
  Grid center: (1.250, -3.000, 0.125)
  Total electrostatic energy = 1.234567890123E+03 kJ/mol
CALCULATION #2 (solv): MULTIGRID
  Grid dimensions: 97 x 97 x 65
  Grid spacings: 0.400 x 0.417 x 0.333
  Grid lengths: 38.400 x 40.000 x 21.333
  Grid center: (1.250, -3.000, 0.125)
  Total electrostatic energy = -2.000000000000E+01 kJ/mol
PRINT STATEMENTS

print energy 1 (solv) - 2 (ref) end
  Local net energy (PE 0) = -5.263725773362E+03 kJ/mol
  Global net ELEC energy = -5.263725773362E+03 kJ/mol
"""


def test_parse_output():
    out = apbs.parse_output(CANNED)
    assert [c.index for c in out.calcs] == [1, 2]
    c = out.calcs[1]
    assert c.name == "solv" and c.dims == (97, 97, 65)
    assert c.spacing == (0.4, 0.417, 0.333) and c.center == (1.25, -3.0, 0.125)
    assert c.energy_kj == -20.0
    assert out.prints == [("1 (solv) - 2 (ref)", -5263.725773362)]
    assert any("Reset dime" in w for w in out.warnings)


def test_maps_are_consistent(molecule):
    coords, charges, radii = molecule
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5, pdie=2.0, ionic_strength=0.15, fadd=10.0)
    m = s.maps(coords, charges, radii)
    for k in ("eps_x", "eps_y", "eps_z", "kappa_sq", "charge"):
        assert m[k].shape == tuple(m["dims"])
    for k in ("eps_x", "eps_y", "eps_z"):
        assert set(np.unique(m[k])) == {2.0, 78.54}
    assert set(np.unique(m["kappa_sq"])) == {0.0, m["kappa_bar_sq"]}
    assert m["charge"].sum() == pytest.approx(charges.sum(), abs=1e-12)
    # the ion-exclusion region (r + 2 Å) contains the dielectric solute region
    assert np.all(m["kappa_sq"][m["eps_x"] == 2.0] == 0.0)
    ref = s.maps(coords, charges, radii, reference=True)
    assert np.all(ref["eps_y"] == 2.0) and np.all(ref["kappa_sq"] == 0.0)


def test_solve_maps_reproduces_the_solver(molecule):
    coords, charges, radii = molecule
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5, ionic_strength=0.15, fadd=10.0, tol=1e-10)
    grid = s.make_grid(coords, radii)
    grid1 = pbsolve.GridSpec(grid.center, grid.dims, grid.fine_length)  # single level
    pot, origin, spacing, _ = s.potential(coords, charges, radii, grid=grid1)
    m = s.maps(coords, charges, radii, grid=grid1)
    u, iters, resid = s.solve_maps(m["eps_x"], m["eps_y"], m["eps_z"], m["kappa_sq"], m["charge"], pot, origin, spacing)
    assert iters > 0 and resid < 1e-10
    np.testing.assert_allclose(u, pot, rtol=0, atol=1e-8 * np.abs(pot).max())
    with pytest.raises(ValueError):
        s.solve_maps(m["eps_x"][:-1], m["eps_y"], m["eps_z"], m["kappa_sq"], m["charge"], pot, origin, spacing)


def test_unit_factor():
    assert apbs.apbs_unit_factor() == pytest.approx(1.0 + 1.2186e-5, abs=2e-9)
    assert apbs.apbs_unit_factor(310.0) == pytest.approx(apbs.apbs_unit_factor(), rel=1e-12)


def _species(dg_apbs, dg_pbsolve):
    uf = apbs.apbs_unit_factor()
    return apbs.ComparisonReport(
        pqr="x.pqr",
        workdir="w",
        platform="CPU",
        grid={"dims": [65, 65, 65], "fine_spacing": [0.5, 0.5, 0.5], "levels": 2},
        grid_issues=[],
        apbs_warnings=[],
        energies={},
        delta_g={
            "apbs": dg_apbs,
            "pbsolve": dg_pbsolve,
            "diff": dg_pbsolve - dg_apbs,
            "rel": dg_pbsolve / dg_apbs - 1.0,
            "rel_unit_corrected": dg_pbsolve * uf / dg_apbs - 1.0,
        },
        unit_factor=uf,
    )


def _binding(ddg_apbs, ddg_pbsolve):
    """Receptor and ligand agree exactly in APBS units; the complex carries the given ΔΔG values."""
    uf = apbs.apbs_unit_factor()
    rec, lig = -35000.0, -3300.0
    return apbs.BindingReport(
        complex=_species(rec + lig + ddg_apbs, (rec + lig) / uf + ddg_pbsolve),
        receptor=_species(rec, rec / uf),
        ligand=_species(lig, lig / uf),
    )


def test_binding_verdict_is_judged_in_apbs_units():
    uf = apbs.apbs_unit_factor()
    # A large ΔΔG (like-charged partners) on which pbsolve agrees exactly in APBS units: the raw
    # difference is the constant factor alone, 6000 × 1.2e-5 = 0.073 kcal/mol, and must not fail.
    rep = _binding(-6000.0, -6000.0 / uf)
    assert abs(rep.ddg["diff"]) > 0.07
    assert abs(rep.ddg["diff_unit_corrected"]) < 1e-9
    assert rep.passed()
    # Equal raw numbers mean a 0.073 kcal/mol disagreement in APBS units.
    rep = _binding(-6000.0, -6000.0)
    assert abs(rep.ddg["diff"]) < 1e-9
    assert not rep.passed()
    assert rep.passed(ddg_atol=0.08)
    # Small complexes: unchanged verdicts on both sides of the tolerance.
    assert _binding(190.0, 190.0 / uf + 0.04).passed()
    assert not _binding(190.0, 190.0 / uf + 0.06).passed()
    assert "binding verdict: PASS" in _binding(-6000.0, -6000.0 / uf).format()
