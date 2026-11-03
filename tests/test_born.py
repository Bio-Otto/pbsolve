import numpy as np
import pytest

import pbsolve
from pbsolve.test_installation import born_energy

COORDS = np.array([[0.1, -0.05, 0.02]])
Q = np.array([1.0])
R = np.array([2.0])


@pytest.mark.parametrize("surface", ["mol", "harmonic"])
def test_born_ion_matches_analytic(surface):
    res = pbsolve.Solver(platform="CPU", grid_spacing=0.25, surface=surface, fadd=12.0).polar_solvation(COORDS, Q, R)
    assert res.energy == pytest.approx(born_energy(), rel=0.03)
    assert res.platform == "CPU"
    assert res.energy == pytest.approx(res.energy_kt * 0.5924837, rel=1e-5)


def test_born_converges_with_spacing():
    errs = []
    for h in (0.5, 0.25):
        e = pbsolve.Solver(platform="CPU", grid_spacing=h, fadd=12.0).polar_solvation(COORDS, Q, R).energy
        errs.append(abs(e - born_energy()))
    assert errs[1] < errs[0]


def test_salt_stabilises_a_charge():
    no_salt = pbsolve.Solver(platform="CPU", grid_spacing=0.4).polar_solvation(COORDS, Q, R).energy
    salt = pbsolve.Solver(platform="CPU", grid_spacing=0.4, ionic_strength=0.15).polar_solvation(COORDS, Q, R).energy
    assert salt < no_salt


def test_neutral_system_has_zero_polar_energy():
    res = pbsolve.Solver(platform="CPU", grid_spacing=0.5).polar_solvation(COORDS, np.array([0.0]), R)
    assert res.energy == 0.0
