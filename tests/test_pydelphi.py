"""The Gaussian dielectric against pyDelPhi (skipped when pyDelPhi is not installed; ROADMAP R4).

pyDelPhi (AGPL-3.0) runs as a separate program through scripts/compare_pydelphi.py,
which reads only its output files. Point PBSOLVE_PYDELPHI at its
``pydelphi-static`` or put that on PATH. Tolerances: the cube files carry 6
significant digits and the energies 4 decimals (kT).
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

EXE = os.environ.get("PBSOLVE_PYDELPHI") or shutil.which("pydelphi-static")
pytestmark = pytest.mark.skipif(EXE is None, reason="pyDelPhi (pydelphi-static) not installed")


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_pydelphi.py"
    spec = importlib.util.spec_from_file_location("compare_pydelphi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_differences_from_pydelphi_are_its_stated_definitions(salt, tmp_path):
    cp = _script()
    coords, q, radii = cp.SYNTHETIC
    r = cp.compare(coords, q, radii, cp.pqr_text(coords, q, radii), cp.grid_center(coords), 65, 0.5, salt, EXE,
                   str(tmp_path))
    for media in ("water", "vacuum"):
        assert r[media]["formula"] < 1e-9  # pbsolve's maps are the Gaussian formula
    # the reference (vacuum) maps agree as they are; the solvated ones once pyDelPhi's definitions
    # (Gaussian cutoff, density floor 0.02, 2 % smoothing) are applied, face for face
    assert r["vacuum"]["raw_off"] == 0
    assert r["water"]["raw_off"] > 1000
    assert r["water"]["model_off"] == 0 and r["vacuum"]["model_off"] == 0
    # coulombic boundary with pyDelPhi's Debye constant; pbsolve's mdh differs only with ions
    assert r["boundary_rel"] < 1e-5
    assert (r["boundary_mdh_rel"] < 1e-5) == (salt == 0)
    # the solvers agree on the same maps, and pbsolve reproduces pyDelPhi with its definitions
    e = r["energies_kt"]
    assert np.allclose(e["pbsolve on pydelphi's maps"], e["pydelphi"], rtol=0, atol=2e-4)
    assert np.allclose(e["pbsolve, pydelphi's definitions"], e["pydelphi"], rtol=0, atol=2e-4)
    # ... to the last printed digit once pyDelPhi's e²/(4πε0 k) (CODATA 2006) is used
    assert np.allclose(e["  ... and pydelphi's constant"], e["pydelphi"], rtol=0, atol=6e-5)
    # the definitions move ΔG_polar by ~1e-5 relative
    assert abs(e["pbsolve"][2] - e["pydelphi"][2]) < 3e-5 * abs(e["pydelphi"][2])
    assert abs(e["pbsolve"][2] - e["pydelphi"][2]) > 2e-4  # ... but measurably
