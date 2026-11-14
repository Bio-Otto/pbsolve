"""The reaction-field decomposition (`coulomb="analytic"`)."""

import json
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs
from pbsolve.__main__ import main
from pbsolve.analytic import kirkwood_energy

A = 5.0
THREE = (np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]]), np.array([1.0, -1.0, 0.5]))
NEAR = (np.array([[4.0, 0.0, 0.0], [0.0, -1.5, 3.5]]), np.array([1.0, 1.0]))  # 1–1.2 Å below the surface
PQR = str(Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")


def sphere(radius, positions, charges):
    coords = np.vstack([[0.0, 0.0, 0.0], positions])
    return coords, np.r_[0.0, charges], np.r_[radius, np.zeros(len(charges))]


def placement_errors(solver, coords, charges, radii, exact, placements=6):
    """Relative errors (%) on the automatic grid moved by fractions of a spacing."""
    grid = solver.make_grid(coords, radii)
    rng = np.random.default_rng(0)
    errors = []
    for k in range(placements):
        x = coords + (rng.uniform(0.0, 0.5, 3) if k else 0.0)
        errors.append(100.0 * (solver.polar_solvation(x, charges, radii, grid=grid).energy - exact) / abs(exact))
    return np.array(errors)


@pytest.mark.parametrize("case", ["three", "near"], ids=["3 charges", "2 charges 1–1.2 Å deep"])
def test_decomposition_narrows_the_placement_spread(case):
    pos, q = {"three": THREE, "near": NEAR}[case]
    exact = kirkwood_energy(pos, q, A, ionic_strength=0.15)
    errors = {
        coulomb: placement_errors(
            pbsolve.Solver(
                platform="CPU", grid_spacing=0.5, ionic_strength=0.15, interface="fractional", coulomb=coulomb
            ),
            *sphere(A, pos, q),
            exact,
        )
        for coulomb in ("grid", "analytic")
    }
    # Measured (6 placements): the spread falls from 0.23 to 0.12 and from 0.67
    # to 0.43 points, the mean stays within 0.01 points.
    assert np.ptp(errors["analytic"]) < 0.75 * np.ptp(errors["grid"])
    assert abs(errors["analytic"].mean() - errors["grid"].mean()) < 0.05
    assert np.all(np.abs(errors["analytic"]) < 0.5)


def test_both_methods_approach_each_other_with_refinement():
    coords, charges, radii = sphere(A, *THREE)
    gaps = []
    for h in (0.6, 0.3):
        grid = pbsolve.Solver(platform="CPU", grid_spacing=h).make_grid(coords, radii)
        e = [
            pbsolve.Solver(platform="CPU", grid_spacing=h, coulomb=c).polar_solvation(coords, charges, radii, grid=grid).energy
            for c in ("grid", "analytic")
        ]
        gaps.append(abs(e[1] - e[0]))
    assert gaps[1] < 0.5 * gaps[0]


def test_result_without_reference_solve():
    coords, charges, radii = sphere(A, *THREE)
    grid_solver = pbsolve.Solver(platform="CPU", grid_spacing=0.6, pdie=2.0)
    grid = grid_solver.make_grid(coords, radii)
    a = grid_solver.polar_solvation(coords, charges, radii, grid=grid)
    b = pbsolve.Solver(platform="CPU", grid_spacing=0.6, pdie=2.0, coulomb="analytic").polar_solvation(
        coords, charges, radii, grid=grid
    )
    # One solve per focusing level instead of two.
    assert len(b.iterations) == len(a.iterations) // 2 == len(grid.levels)
    # The single states without self-energies: the Coulomb energy in pdie, plus ΔG_polar.
    pos, q = THREE
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    coulomb = sum(q[i] * q[j] / np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
    assert b.reference_kt == pytest.approx(pbsolve.bjerrum_length(298.15) / 2.0 * coulomb, rel=1e-12)
    assert b.solvated_kt - b.reference_kt == pytest.approx(b.energy_kt, rel=1e-12)
    assert b.energy == pytest.approx(a.energy, rel=0.02)


def test_coulomb_parameter():
    assert pbsolve.Solver(platform="CPU").params["coulomb"] == "grid"
    s = pbsolve.Solver(platform="CPU", coulomb="analytic")
    assert s.params["coulomb"] == "analytic"
    # Solvers rebuilt from params (as estimate_error does for other spacings) keep it.
    assert pbsolve.estimate._with_spacing(s, 0.7).params["coulomb"] == "analytic"
    with pytest.raises(ValueError, match="coulomb"):
        pbsolve.Solver(platform="CPU", coulomb="exact")
    coords, charges, radii = sphere(A, *THREE)
    with pytest.raises(ValueError, match="no APBS equivalent"):
        apbs.elec_from_solver(s, s.make_grid(coords, radii), "solvated")


def test_charge_on_a_node_next_to_the_boundary_is_reported():
    # A charge of radius 0 in the solvent, exactly on a node of a single grid.
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    charges, radii = np.array([1.0, -1.0]), np.array([2.0, 0.0])
    grid = pbsolve.GridSpec(center=[0.0, 0.0, 0.0], dims=[33, 33, 33], coarse_length=[16.0, 16.0, 16.0])
    s = pbsolve.Solver(platform="CPU", coulomb="analytic")
    with pytest.raises(ValueError, match="coulomb 'analytic'"):
        s.polar_solvation(coords, charges, radii, grid=grid)


def test_command_line(capsys):
    args = ["solvation", PQR, "--platform", "CPU", "--spacing", "0.9", "--json"]
    energies = []
    for coulomb in ("grid", "analytic"):
        assert main(args + ["--interface", "fractional", "--coulomb", coulomb]) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    # The coarse 0.9 Å grid gives −1191.9 (grid) and −1176.6 (analytic).
    assert energies[0] < energies[1] < 0
    assert energies[1] == pytest.approx(energies[0], rel=0.03)
