"""Multigrid-preconditioned CG versus Jacobi-preconditioned CG.

The preconditioner changes only how fast CG converges, never the discrete
problem: energies must agree to the solver tolerance, the multigrid iteration
count must stay small and roughly independent of the grid spacing.
"""

import numpy as np
import pytest

import pbsolve

DATA = "tests/data/1ycr_amber_opt_pdb2pqr.pqr"


@pytest.fixture(scope="module")
def peptide():
    """p53 peptide (chain B of 1YCR, pdb2pqr AMBER charges and radii)."""
    coords, charges, radii = pbsolve.read_pqr(DATA)
    mol = pbsolve.io.load(DATA)
    keep = mol.select("chain B")
    return coords[keep], charges[keep], radii[keep]


def solve(mol, pre, **kw):
    s = pbsolve.Solver(platform="CPU", preconditioner=pre, fadd=12.0, **kw)
    return s.polar_solvation(*mol)


@pytest.mark.parametrize("kw", [dict(grid_spacing=0.5), dict(grid_spacing=0.35, ionic_strength=0.15, pdie=4.0)])
def test_multigrid_and_jacobi_agree(peptide, kw):
    mg = solve(peptide, "multigrid", tol=1e-10, **kw)
    jac = solve(peptide, "jacobi", tol=1e-10, **kw)
    assert mg.energy == pytest.approx(jac.energy, rel=1e-8)
    assert mg.solvated_kt == pytest.approx(jac.solvated_kt, rel=1e-9)
    assert mg.reference_kt == pytest.approx(jac.reference_kt, rel=1e-9)
    assert set(mg.preconditioners) == {"multigrid"} and min(mg.levels) >= 3
    assert set(jac.preconditioners) == {"jacobi"} and set(jac.levels) == {1}
    assert max(mg.iterations) * 8 < max(jac.iterations)


@pytest.mark.parametrize("kw", [dict(grid_spacing=0.5), dict(grid_spacing=0.35, ionic_strength=0.15, pdie=4.0)])
def test_single_precision_multigrid_agrees_with_double(peptide, kw):
    """The f32 V-cycle is only a preconditioner: the f64 CG converges to the same answer."""
    mg = solve(peptide, "multigrid", **kw)
    mixed = solve(peptide, "multigrid-f32", **kw)
    assert mixed.energy == pytest.approx(mg.energy, rel=1e-9)
    assert mixed.solvated_kt == pytest.approx(mg.solvated_kt, rel=1e-10)
    assert mixed.reference_kt == pytest.approx(mg.reference_kt, rel=1e-10)
    assert set(mixed.preconditioners) == {"multigrid-f32"}
    assert mixed.levels == mg.levels
    assert all(a <= b + 1 for a, b in zip(mixed.iterations, mg.iterations)), (mixed.iterations, mg.iterations)
    assert max(mixed.residuals) < 1e-8


def test_single_precision_multigrid_at_tight_tolerance(peptide):
    """Flexible CG keeps converging far below f32 precision."""
    mixed = solve(peptide, "multigrid-f32", grid_spacing=0.5, tol=1e-12)
    mg = solve(peptide, "multigrid", grid_spacing=0.5, tol=1e-12)
    assert max(mixed.residuals) < 1e-12
    assert mixed.energy == pytest.approx(mg.energy, rel=1e-13)


def test_default_solver_uses_the_single_precision_multigrid(peptide):
    """The default is multigrid-f32; "multigrid" keeps the f64 cycle of earlier versions."""
    default = pbsolve.Solver(platform="CPU", fadd=12.0, ionic_strength=0.15).polar_solvation(*peptide)
    explicit = solve(peptide, "multigrid-f32", ionic_strength=0.15)
    double = solve(peptide, "multigrid", ionic_strength=0.15)
    assert set(default.preconditioners) == {"multigrid-f32"}
    assert default.energy == explicit.energy  # bit-identical: same code path
    assert default.energy == pytest.approx(double.energy, rel=1e-9)
    assert all(a <= b + 1 for a, b in zip(default.iterations, double.iterations))


def test_multigrid_iterations_are_spacing_independent(peptide):
    iters = [max(solve(peptide, "multigrid", grid_spacing=h).iterations) for h in (0.6, 0.45, 0.3)]
    assert max(iters) <= 20, iters
    assert max(iters) - min(iters) <= 4, iters


def test_preconditioner_option_is_validated_and_reported():
    with pytest.raises(ValueError, match="preconditioner"):
        pbsolve.Solver(preconditioner="ilu")
    assert pbsolve.Solver().params["preconditioner"] == "multigrid-f32"
    assert pbsolve.Solver(preconditioner="MG").params["preconditioner"] == "multigrid"
    assert pbsolve.Solver(preconditioner="jacobi").params["preconditioner"] == "jacobi"
    assert pbsolve.Solver(preconditioner="multigrid-f32").params["preconditioner"] == "multigrid-f32"
    assert pbsolve.Solver(preconditioner="MG-F32").params["preconditioner"] == "multigrid-f32"


def test_multigrid_solves_injected_maps(peptide):
    """solve_maps (APBS map injection) uses the same preconditioner."""
    coords, charges, radii = peptide
    res = {}
    for pre in ("multigrid", "jacobi"):
        s = pbsolve.Solver(platform="CPU", grid_spacing=0.6, preconditioner=pre, tol=1e-10, fadd=12.0)
        grid = s.make_grid(coords, radii)
        m = s.maps(coords, charges, radii, grid=grid)
        u, iters, resid = s.solve_maps(
            m["eps_x"], m["eps_y"], m["eps_z"], m["kappa_sq"], m["charge"], np.zeros_like(m["charge"]),
            m["origin"], m["spacing"],
        )
        assert resid < 1e-10
        res[pre] = (u, iters)
    np.testing.assert_allclose(res["multigrid"][0], res["jacobi"][0], rtol=0, atol=1e-8 * np.abs(res["jacobi"][0]).max())
    assert res["multigrid"][1] * 8 < res["jacobi"][1]


def test_cli_preconditioner_option(tmp_path, capsys):
    import json

    from pbsolve.__main__ import main

    mol = pbsolve.io.load(DATA)
    pqr = tmp_path / "peptide.pqr"
    sub = mol.subset(mol.select("chain B"))
    sub.write_pqr(str(pqr), sub.coords)
    out = {}
    for pre in ("multigrid", "multigrid-f32", "jacobi"):
        assert main(["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--preconditioner", pre, "--json"]) == 0
        out[pre] = json.loads(capsys.readouterr().out)
        assert out[pre]["preconditioner"] == [pre]
    assert out["multigrid"]["polar_kcal_mol"] == pytest.approx(out["jacobi"]["polar_kcal_mol"], rel=1e-7)
    assert out["multigrid-f32"]["polar_kcal_mol"] == pytest.approx(out["multigrid"]["polar_kcal_mol"], rel=1e-9)
    assert min(out["multigrid"]["multigrid_levels"]) >= 3
    # Without the option the CLI uses the default, multigrid-f32.
    assert main(["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--json"]) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["preconditioner"] == ["multigrid-f32"]
    assert default["polar_kcal_mol"] == out["multigrid-f32"]["polar_kcal_mol"]
