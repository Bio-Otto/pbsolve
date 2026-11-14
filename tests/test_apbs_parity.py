"""pbsolve vs the real APBS executable (skipped when `apbs` is not installed).

Tolerances: dielectric and ion maps must be identical node for node. The
remaining differences are APBS' physical constants, the 7-digit DX format
(~1e-7) and solver tolerances. `apbs_unit_factor` removes the constants of the
source term and of the kJ/mol conversion exactly. APBS evaluates the analytic
boundary (`mdh`, `sdh`) with yet another constant set (CODATA-1986 SI; ratio
to its CGS source term 1 − 1.23e-5), which cannot be absorbed in one factor:
it leaves up to ~1e-6 in unscreened reference states (pdie, no ions), where
the boundary carries a sizeable part of the energy. With `bcfl zero` all
energies agree to ~1e-10.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs

from conftest import write_pqr

pytestmark = pytest.mark.skipif(apbs.find_apbs() is None, reason="APBS executable not found")


YCR = Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr"

# Relative energy tolerances after the unit correction (see module docstring).
SOLVATED_RTOL = 1e-7
REFERENCE_RTOL = {"zero": 1e-7, "mdh": 5e-6, "mdh-fast": 5e-6, "sdh": 5e-6}


def _check(rep, bcfl):
    assert not rep.grid_issues
    for k in ("eps_x", "eps_y", "eps_z", "kappa"):
        assert rep.maps[k]["mismatch"] == 0, (k, rep.maps[k])
    assert rep.maps["charge"]["max_rel_to_max"] < 1e-5
    for st, inj in rep.injection.items():
        # constants alone shift u by ~6e-6 of its scale (APBS' CGS source term)
        assert inj["potential"]["max_rel_to_max"] < 2e-5, st
        assert abs(inj["energy_rel_unit_corrected"]) < 1e-6, st
    assert abs(rep.energies["solvated"]["rel_unit_corrected"]) < SOLVATED_RTOL
    assert abs(rep.energies["reference"]["rel_unit_corrected"]) < REFERENCE_RTOL[bcfl]
    assert rep.passed()


@pytest.mark.parametrize(
    "kw",
    [
        dict(ionic_strength=0.15, bcfl="mdh"),
        dict(ionic_strength=0.15, bcfl="mdh-fast"),  # APBS runs mdh
        dict(ionic_strength=0.15, bcfl="sdh", pdie=2.0),
        dict(ionic_strength=0.0, bcfl="sdh"),
        dict(ionic_strength=0.05, bcfl="zero", surface="vdw"),
        dict(ionic_strength=0.15, bcfl="mdh", chgm="spl2"),
        dict(ionic_strength=0.0, bcfl="sdh", chgm="spl4"),
    ],
)
def test_layered_parity(tmp_path, molecule_pqr, kw):
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, dime_policy="apbs", **kw)
    rep = apbs.compare(molecule_pqr, s, workdir=tmp_path / "w")
    _check(rep, kw["bcfl"])


def test_three_level_focusing(tmp_path, molecule, molecule_pqr):
    coords, _, radii = molecule
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.1)
    c = list(0.5 * (coords.max(axis=0) + coords.min(axis=0)))
    grid = pbsolve.GridSpec(c, [65, 65, 65], [120.0] * 3, [22.0] * 3)  # ratio 0.18 -> 3 levels
    assert len(grid.levels) == 3
    rep = apbs.compare(molecule_pqr, s, grid=grid, workdir=tmp_path / "w", maps=False, inject=False)
    assert rep.grid["levels"] == 3 and not rep.grid_issues
    assert abs(rep.energies["solvated"]["rel_unit_corrected"]) < SOLVATED_RTOL
    assert abs(rep.energies["reference"]["rel_unit_corrected"]) < REFERENCE_RTOL["mdh"]


def test_born_ion(tmp_path):
    pqr = write_pqr(tmp_path / "ion.pqr", np.array([[0.1, -0.05, 0.02]]), np.array([1.0]), np.array([2.0]))
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.3, fadd=12.0, ionic_strength=0.15, dime_policy="apbs")
    _check(apbs.compare(pqr, s, workdir=tmp_path / "w"), "mdh")


@pytest.mark.parametrize("bcfl, atol", [("zero", 1e-5), ("mdh", 5e-3)])
def test_binding(tmp_path, molecule, bcfl, atol):
    coords, charges, radii = molecule
    pqr = write_pqr(tmp_path / "cx.pqr", coords, charges, radii)
    s = pbsolve.Solver(
        platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, bcfl=bcfl, dime_policy="apbs"
    )
    rep = apbs.compare_binding(pqr, s, np.arange(20, 30), workdir=tmp_path / "w", maps=False, inject=False)
    # kcal/mol; with mdh the residual is APBS' boundary-constant mismatch (~1e-6 of the reference energies)
    assert abs(rep.ddg["diff_unit_corrected"]) < atol
    assert rep.passed()


def test_protein_complex(tmp_path):
    """MDM2–p53 (1YCR: 1670 atoms, two chains, 18 hydrogens of radius 0) on a coarse
    grid: identical maps and energies for complex, receptor and ligand."""
    chains = np.array([ln.split()[4] for ln in YCR.read_text().splitlines() if ln.startswith("ATOM")])
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.8, ionic_strength=0.15, bcfl="mdh", dime_policy="apbs")
    rep = apbs.compare_binding(YCR, s, chains == "B", workdir=tmp_path / "w")
    for r in (rep.complex, rep.receptor, rep.ligand):
        _check(r, "mdh")
    # kcal/mol in APBS units (1.2e-3 measured): APBS' boundary constants in the reference states
    assert abs(rep.ddg["diff_unit_corrected"]) < 5e-3
    assert rep.passed()


def test_protein_complex_cli_zero_boundary(tmp_path):
    """The command line on the same complex with `bcfl zero`: without an analytic
    boundary the unit factor is the only difference, down to the solver tolerance."""
    out = tmp_path / "report.json"
    argv = ["compare", str(YCR), "--platform", "CPU", "--spacing", "0.8", "--ionic-strength", "0.15"]
    argv += ["--bcfl", "zero", "--ligand-chain", "B", "--no-maps", "--workdir", str(tmp_path / "w")]
    assert apbs.main(argv + ["--json", str(out)]) == 0
    rep = json.loads(out.read_text())
    for species in ("complex", "receptor", "ligand"):
        for state in ("solvated", "reference"):
            assert abs(rep[species]["energies"][state]["rel_unit_corrected"]) < 1e-8, (species, state)
    assert abs(rep["ddg"]["diff_unit_corrected"]) < 1e-4  # kcal/mol (5e-6 measured)


def test_run_existing_input(tmp_path, molecule, molecule_pqr):
    coords, _, radii = molecule
    s = pbsolve.Solver(grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, dime_policy="apbs")
    grid = s.make_grid(coords, radii)
    elecs = [apbs.elec_from_solver(s, grid, "solv"), apbs.elec_from_solver(s, grid, "ref", reference=True)]
    path = tmp_path / "user.in"
    path.write_text(apbs.write_input([molecule_pqr.name], elecs, ["solv - ref"]))
    rep = apbs.run_input(path, platform="CPU")
    assert rep.passed(energy_rtol=1e-4)
    assert len(rep.prints) == 1 and abs(rep.prints[0]["rel"]) < 1e-4
    assert apbs.main(["run", str(path), "--platform", "CPU", "--workdir", str(tmp_path / "cli")]) == 0
