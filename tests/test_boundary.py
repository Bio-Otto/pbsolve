"""Boundary conditions: `mdh-fast` (interpolated mdh) against the exact `mdh`.

`mdh` follows APBS operation for operation. `mdh-fast` evaluates the same
Debye-Hueckel sum exactly at Chebyshev points of patches of the box faces and
interpolates in between (error <= 1e-10 of the summed term magnitudes); the
energies must agree far below any physical or numerical significance.
"""

import json

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs

DATA = "tests/data/1ycr_amber_opt_pdb2pqr.pqr"


@pytest.fixture(scope="module")
def complex_1ycr():
    return pbsolve.read_pqr(DATA)


@pytest.mark.parametrize("kw", [dict(ionic_strength=0.15), dict(ionic_strength=0.0, pdie=4.0)])
def test_mdh_fast_energies_equal_mdh(complex_1ycr, kw):
    res = {}
    for bcfl in ("mdh", "mdh-fast"):
        s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, bcfl=bcfl, **kw)
        res[bcfl] = s.polar_solvation(*complex_1ycr)
    a, b = res["mdh"], res["mdh-fast"]
    assert b.energy == pytest.approx(a.energy, rel=1e-10)
    assert b.solvated_kt == pytest.approx(a.solvated_kt, rel=1e-10)
    assert b.reference_kt == pytest.approx(a.reference_kt, rel=1e-10)
    assert b.iterations == a.iterations


def test_mdh_fast_potentials_agree(complex_1ycr):
    """The fine-level potential (its boundary comes from the coarse solution) agrees everywhere."""
    s = {b: pbsolve.Solver(platform="CPU", grid_spacing=0.6, ionic_strength=0.15, bcfl=b) for b in ("mdh", "mdh-fast")}
    grid = s["mdh"].make_grid(complex_1ycr[0], complex_1ycr[2])
    u = {b: s[b].potential(*complex_1ycr, grid=grid)[0] for b in s}
    scale = np.abs(u["mdh"]).max()
    assert np.abs(u["mdh-fast"] - u["mdh"]).max() < 1e-10 * scale


def test_mdh_fast_option_round_trip_and_cli(tmp_path, capsys):
    assert pbsolve.Solver(bcfl="mdh-fast").params["bcfl"] == "mdh-fast"
    assert pbsolve.Solver(bcfl="MDH-FAST").params["bcfl"] == "mdh-fast"
    with pytest.raises(ValueError, match="mdh-fast"):
        pbsolve.Solver(bcfl="mdh-slow")
    from pbsolve.__main__ import main

    mol = pbsolve.io.load(DATA)
    pqr = tmp_path / "peptide.pqr"
    sub = mol.subset(mol.select("chain B"))
    sub.write_pqr(str(pqr), sub.coords)
    out = {}
    for bcfl in ("mdh", "mdh-fast"):
        args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--bcfl", bcfl, "--json"]
        assert main(args) == 0
        out[bcfl] = json.loads(capsys.readouterr().out)
    assert out["mdh-fast"]["polar_kcal_mol"] == pytest.approx(out["mdh"]["polar_kcal_mol"], rel=1e-10)


def test_apbs_input_uses_mdh_for_mdh_fast(complex_1ycr):
    """APBS has no mdh-fast: inputs written for it ask for the exact mdh."""
    s = pbsolve.Solver(grid_spacing=0.6, ionic_strength=0.15, bcfl="mdh-fast", dime_policy="apbs")
    grid = s.make_grid(complex_1ycr[0], complex_1ycr[2])
    assert apbs.elec_from_solver(s, grid, "solv").bcfl == "mdh"
