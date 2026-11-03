import numpy as np
import pytest

import pbsolve

DIPOLE = np.array([[-1.0, 0.0, 0.0], [1.2, 0.3, 0.0]])
Q2 = np.array([0.5, -0.5])
R2 = np.array([1.7, 1.5])


def test_version_and_platforms():
    assert pbsolve.__version__
    plats = pbsolve.available_platforms()
    assert "CPU" in plats
    assert pbsolve.Solver().platform in plats  # "auto" resolves to an available one


def test_bad_inputs_raise_value_error():
    s = pbsolve.Solver(platform="CPU")
    with pytest.raises(ValueError):
        s.polar_solvation(np.zeros((2, 2)), Q2, R2)
    with pytest.raises(ValueError):
        s.polar_solvation(DIPOLE, Q2[:1], R2)
    with pytest.raises(ValueError):
        pbsolve.Solver(platform="quantum")
    with pytest.raises(ValueError):
        pbsolve.Solver(surface="spl9")


def test_cuda_request_without_gpu_raises():
    if "CUDA" in pbsolve.available_platforms():
        pytest.skip("CUDA is available here")
    with pytest.raises(RuntimeError):
        pbsolve.Solver(platform="CUDA")


def test_results_are_reproducible():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5)
    a = s.polar_solvation(DIPOLE, Q2, R2).energy
    b = s.polar_solvation(DIPOLE, Q2, R2).energy
    assert a == b


def test_batch_matches_single_frames():
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5)
    frames = np.stack([DIPOLE, DIPOLE + 0.3])
    grid = s.make_grid(frames[0], R2)
    batch = s.polar_solvation_batch(frames, Q2, R2, grid=grid)
    single = [s.polar_solvation(f, Q2, R2, grid=grid).energy for f in frames]
    np.testing.assert_allclose(batch, single, rtol=0, atol=0)


def test_gridspec_roundtrip_and_apbs_keywords():
    s = pbsolve.Solver(grid_spacing=0.5)
    g = s.make_grid(DIPOLE, R2)
    assert all((n - 1) % 8 == 0 for n in g.dims)
    assert max(g.fine_spacing) <= 0.5 + 1e-12
    text = g.to_apbs()
    for key in ("dime", "cglen", "fglen", "cgcent", "fgcent"):
        assert key in text
    g2 = pbsolve.GridSpec(g.center, g.dims, g.coarse_length, g.fine_length)
    assert g2.dims == g.dims


def test_sasa_of_isolated_sphere():
    area = pbsolve.sasa(np.zeros((1, 3)), np.array([1.6]), probe=1.4)[0]
    assert area == pytest.approx(4 * np.pi * 3.0**2, rel=1e-9)


def test_read_pqr(tmp_path):
    p = tmp_path / "x.pqr"
    p.write_text(
        "ATOM      1  N   ALA A   1      -1.000   2.000   3.500  0.1414 1.8240\n"
        "ATOM      2  CA  ALA     1       0.000   0.000   0.000 -0.0912 1.9080\nEND\n"
    )
    xyz, q, r = pbsolve.read_pqr(p)
    assert xyz.shape == (2, 3) and q.shape == (2,) and r.shape == (2,)
    np.testing.assert_allclose(xyz[0], [-1.0, 2.0, 3.5])


def test_potential_and_dx_roundtrip(tmp_path):
    s = pbsolve.Solver(platform="CPU", grid_spacing=0.5)
    pot, origin, spacing, energy_kt = s.potential(DIPOLE, Q2, R2)
    assert pot.ndim == 3 and np.isfinite(pot).all()
    path = tmp_path / "pot.dx"
    pbsolve.write_dx(path, pot, origin, spacing)
    back, o2, h2 = pbsolve.read_dx(path)
    np.testing.assert_allclose(back, pot, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(o2, origin, rtol=1e-6)
    np.testing.assert_allclose(h2, spacing, rtol=1e-6)
