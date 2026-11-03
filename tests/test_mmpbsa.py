import numpy as np
import pytest

import pbsolve

# Toy "complex": a cation receptor and an anion ligand forming a salt bridge.
COORDS = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.1]])
Q = np.array([1.0, -1.0])
R = np.array([2.0, 2.0])


def test_salt_bridge_desolvation_and_buried_area():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.4)
    frame = pbsolve.pbsa_solvation_frame(s, COORDS, Q, R, ligand_atoms=[1])
    assert frame.delta_polar > 0.0  # desolvation penalty of an ion pair
    assert frame.delta_nonpolar < 0.0  # buried surface
    d = frame.as_dict()
    assert d["delta_solvation"] == pytest.approx(frame.delta_polar + frame.delta_nonpolar)


def test_selections_are_validated():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5)
    with pytest.raises(ValueError):
        pbsolve.pbsa_solvation_frame(s, COORDS, Q, R, ligand_atoms=[0], receptor_atoms=[0])
    with pytest.raises(ValueError):
        pbsolve.pbsa_solvation_frame(s, COORDS, Q, R, ligand_atoms=[0, 1])
