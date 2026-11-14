"""The fractional dielectric interface (`interface="fractional"`)."""

import json
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve import apbs
from pbsolve.__main__ import main
from pbsolve.analytic import kirkwood_energy

# A 5 Å sphere with three charges (as in test_analytic) and a Born ion.
A = 5.0
QPOS = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, 0.5], [0.5, -1.0, -2.5]])
QVAL = np.array([1.0, -1.0, 0.5])


def sphere(radius, positions, charges):
    coords = np.vstack([[0.0, 0.0, 0.0], positions])
    return coords, np.r_[0.0, charges], np.r_[radius, np.zeros(len(charges))]


def placement_errors(solver, coords, charges, radii, exact, placements=4):
    """Relative errors (%) on the automatic grid moved by fractions of a spacing."""
    grid = solver.make_grid(coords, radii)
    rng = np.random.default_rng(0)
    errors = []
    for k in range(placements):
        x = coords + (rng.uniform(0.0, 0.5, 3) if k else 0.0)
        e = solver.polar_solvation(x, charges, radii, grid=grid).energy
        errors.append(100.0 * (e - exact) / abs(exact))
    return np.array(errors)


@pytest.mark.parametrize(
    "case", ["born", "kirkwood"], ids=["Born ion a = 2 Å", "Kirkwood a = 5 Å, 0.15 M"]
)
def test_fractional_interface_is_accurate_and_placement_independent(case):
    if case == "born":
        system, kw = sphere(2.0, np.zeros((1, 3)), [1.0]), dict(fadd=12.0)
        exact = kirkwood_energy(np.zeros((1, 3)), [1.0], 2.0)
    else:
        system, kw = sphere(A, QPOS, QVAL), dict(ionic_strength=0.15)
        exact = kirkwood_energy(QPOS, QVAL, A, ionic_strength=0.15)
    errors = {
        interface: placement_errors(
            pbsolve.Solver(platform="CPU", grid_spacing=0.5, interface=interface, **kw), *system, exact
        )
        for interface in ("sharp", "fractional")
    }
    # Measured at 0.5 Å (20 placements, docs/VALIDATION.md): sharp −3.2 %
    # (sd 1.0 and 0.36), fractional −0.55 % and −0.23 % (sd 0.02 and 0.08).
    assert np.all(np.abs(errors["fractional"]) < 0.7)
    assert np.all(errors["sharp"] < -1.0)
    assert np.ptp(errors["fractional"]) < np.ptp(errors["sharp"]) / 3


def test_fractional_faces_lie_between_the_sharp_values():
    coords, charges, radii = sphere(A, QPOS, QVAL)
    sharp = pbsolve.Solver(platform="CPU", grid_spacing=0.6, pdie=2.0)
    frac = pbsolve.Solver(platform="CPU", grid_spacing=0.6, pdie=2.0, interface="fractional")
    grid = sharp.make_grid(coords, radii)
    a = sharp.maps(coords, charges, radii, grid=grid)
    b = frac.maps(coords, charges, radii, grid=grid)
    for key in ("eps_x", "eps_y", "eps_z"):
        cut = (b[key] > 2.0) & (b[key] < 78.54)
        # Faces whose edge the surface cuts get intermediate values; all others keep the sharp value.
        assert 500 < cut.sum() < 0.01 * cut.size
        assert np.array_equal(a[key][~cut], b[key][~cut])
    assert np.array_equal(a["kappa_sq"], b["kappa_sq"]) and np.array_equal(a["charge"], b["charge"])
    # The reference state is homogeneous either way.
    ref = frac.maps(coords, charges, radii, grid=grid, reference=True)
    assert np.all(ref["eps_x"] == 2.0)


def test_interface_parameter():
    assert pbsolve.Solver(platform="CPU").params["interface"] == "sharp"
    s = pbsolve.Solver(platform="CPU", interface="fractional")
    assert s.params["interface"] == "fractional"
    # Solvers rebuilt from params (as estimate_error does for other spacings) keep it.
    assert pbsolve.estimate._with_spacing(s, 0.7).params["interface"] == "fractional"
    with pytest.raises(ValueError, match="interface"):
        pbsolve.Solver(platform="CPU", interface="smooth")
    with pytest.raises(ValueError, match="fractional"):
        pbsolve.Solver(platform="CPU", surface="harmonic", interface="fractional")
    coords, charges, radii = sphere(A, QPOS, QVAL)
    with pytest.raises(ValueError, match="no APBS equivalent"):
        apbs.elec_from_solver(s, s.make_grid(coords, radii), "solvated")


def test_command_line(capsys):
    pqr = str(Path(__file__).parent / "data" / "1ycr_amber_opt_pdb2pqr.pqr")
    args = ["solvation", pqr, "--platform", "CPU", "--spacing", "0.9", "--json"]
    energies = []
    for interface in ("sharp", "fractional"):
        assert main(args + ["--interface", interface]) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    # Only the wiring is tested here: the coarse 0.9 Å grid gives −1342.6 (sharp) and −1191.9 (fractional).
    assert energies[0] < energies[1] < 0
