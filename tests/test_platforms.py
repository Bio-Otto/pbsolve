import numpy as np
import pytest

import pbsolve

needs_cuda = pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")


@needs_cuda
def test_cuda_matches_cpu():
    coords = np.array([[-1.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.0, 1.5, -0.4]])
    q, r = np.array([0.6, -0.9, 0.3]), np.array([1.7, 1.5, 1.2])
    kw = dict(grid_spacing=0.4, ionic_strength=0.15)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(coords, q, r)
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(coords, q, r)
    assert gpu.platform == "CUDA"
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-8)


def test_cuda_info_is_consistent():
    info = pbsolve.cuda_info()
    assert set(info) >= {"compiled", "cuda_major", "available", "device_count", "reason"}
    assert info["available"] == ("CUDA" in pbsolve.available_platforms())
    if not info["available"]:
        assert info["reason"]


@needs_cuda
@pytest.mark.parametrize("pre", ["multigrid", "jacobi"])
def test_cuda_matches_cpu_on_a_peptide(pre):
    """Multilevel hierarchy (Galerkin operators, transfers, coarsest solve) on the GPU."""
    coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
    sel = slice(1500, None)
    kw = dict(grid_spacing=0.45, ionic_strength=0.15, pdie=2.0, preconditioner=pre, fadd=12.0)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(coords[sel], charges[sel], radii[sel])
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(coords[sel], charges[sel], radii[sel])
    assert gpu.platform == "CUDA"
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-10)
    assert gpu.levels == cpu.levels and gpu.preconditioners == cpu.preconditioners
    assert all(abs(a - b) <= 1 for a, b in zip(gpu.iterations, cpu.iterations))
