import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pbsolve

needs_cuda = pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")


@needs_cuda
@pytest.mark.parametrize("interface", ["sharp", "fractional"])
def test_cuda_matches_cpu(interface):
    coords = np.array([[-1.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.0, 1.5, -0.4]])
    q, r = np.array([0.6, -0.9, 0.3]), np.array([1.7, 1.5, 1.2])
    kw = dict(grid_spacing=0.4, ionic_strength=0.15, interface=interface)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(coords, q, r)
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(coords, q, r)
    assert gpu.platform == "CUDA"
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-8)


def test_cuda_info_is_consistent():
    info = pbsolve.cuda_info()
    assert set(info) >= {
        "compiled", "cuda_major", "available", "device_count", "reason", "driver", "nvrtc", "nvrtc_library",
    }
    assert info["available"] == ("CUDA" in pbsolve.available_platforms())
    assert info["cuda_major"] == pbsolve._core._CUDA_MAJOR
    if not info["available"]:
        assert info["reason"]
    else:
        assert info["driver"] and info["nvrtc"] and info["nvrtc_library"]
        if not os.environ.get("PBSOLVE_NVRTC") and pbsolve._nvrtc_candidates(info["cuda_major"]):
            # This environment ships an NVRTC of the build's major: it is the one used.
            assert int(info["nvrtc"].split(".")[0]) == info["cuda_major"]
        if sys.platform.startswith("linux"):
            assert os.path.isfile(info["nvrtc_library"])


def _run_isolated(code: str, **env: str) -> dict:
    """Runs `code` (which prints one JSON line) in a fresh interpreter."""
    full = {**os.environ, **env}
    out = subprocess.run(
        [sys.executable, "-c", code], env=full, capture_output=True, text=True, check=True, timeout=300
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


_PROBE = """
import json, pbsolve
info = pbsolve.cuda_info()
auto = pbsolve.Solver(platform="auto").platform
try:
    pbsolve.Solver(platform="CUDA")
    cuda_error = None
except RuntimeError as e:
    cuda_error = str(e)
print(json.dumps({"info": info, "auto": auto, "cuda_error": cuda_error}))
"""


@pytest.mark.skipif(not pbsolve.cuda_info()["compiled"], reason="CPU-only build")
def test_unusable_nvrtc_disables_cuda_with_a_reason():
    """A broken NVRTC must never break the package: CUDA reports why, auto uses the CPU."""
    r = _run_isolated(_PROBE, PBSOLVE_NVRTC="/nonexistent/libnvrtc.so.99")
    assert r["info"]["available"] is False
    assert r["auto"] == "CPU"
    assert r["cuda_error"]
    if pbsolve.cuda_info()["driver"]:  # driver present: NVRTC is the reason
        assert "PBSOLVE_NVRTC" in r["info"]["reason"]
        assert "NVRTC" in r["cuda_error"]


@needs_cuda
def test_invalid_code_type_is_reported():
    r = _run_isolated(_PROBE, PBSOLVE_CUDA_CODE="sass")
    assert r["auto"] == "CPU"
    assert "PBSOLVE_CUDA_CODE" in r["cuda_error"]


@needs_cuda
@pytest.mark.parametrize("code", ["cubin", "ptx"])
def test_cuda_code_types_give_identical_energies(code):
    """CUBIN (default) and driver-JIT PTX run the same arithmetic."""
    probe = """
import json, pbsolve
import numpy as np
xyz = np.array([[-1.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.0, 1.5, -0.4]])
q, r = np.array([0.6, -0.9, 0.3]), np.array([1.7, 1.5, 1.2])
res = pbsolve.Solver(platform="CUDA", grid_spacing=0.4, ionic_strength=0.15).polar_solvation(xyz, q, r)
print(json.dumps({"energy": res.energy.hex(), "iterations": res.iterations}))
"""
    try:
        got = _run_isolated(probe, PBSOLVE_CUDA_CODE=code)
    except subprocess.CalledProcessError as e:
        info = pbsolve.cuda_info()
        nvrtc, driver = (tuple(map(int, info[k].split("."))) for k in ("nvrtc", "driver"))
        if code == "ptx" and nvrtc > driver and "UNSUPPORTED_PTX_VERSION" in e.stderr:
            pytest.skip(f"PTX from NVRTC {info['nvrtc']} cannot run on driver {info['driver']}")
        raise
    default = _run_isolated(probe)
    assert got == default


def test_nvrtc_search_covers_conda_and_pip_layouts(tmp_path, monkeypatch):
    if sys.platform.startswith("linux"):
        conda = tmp_path / "env" / "lib" / "libnvrtc.so.13"
        pip13 = tmp_path / "site" / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13"
        pip12 = tmp_path / "site" / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
    elif sys.platform == "win32":
        conda = tmp_path / "env" / "Library" / "bin" / "nvrtc64_130_0.dll"
        pip13 = tmp_path / "site" / "nvidia" / "cu13" / "bin" / "nvrtc64_130_0.dll"
        pip12 = tmp_path / "site" / "nvidia" / "cuda_nvrtc" / "bin" / "nvrtc64_120_0.dll"
    else:
        assert pbsolve._nvrtc_candidates(13, str(tmp_path)) == []
        return
    for f in (conda, pip13, pip12):
        f.parent.mkdir(parents=True)
        f.write_bytes(b"")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    prefix = str(tmp_path / "env")

    def found(major):
        # Ignore real nvidia packages installed next to the test's fake one.
        return [c for c in pbsolve._nvrtc_candidates(major, prefix) if c.startswith(str(tmp_path))]

    # The environment's own library first, then pip's; only the requested major.
    assert pbsolve._nvrtc_candidates(13, prefix)[0] == str(conda)
    assert found(13) == [str(conda), str(pip13)]
    assert found(12) == [str(pip12)]
    assert pbsolve._nvrtc_candidates(0, prefix) == []


def test_preload_can_be_disabled(monkeypatch, tmp_path):
    lib = tmp_path / "lib" / "libnvrtc.so.13"
    lib.parent.mkdir()
    lib.write_bytes(b"not a library")
    for var in ("PBSOLVE_NO_PRELOAD", "PBSOLVE_NVRTC"):
        monkeypatch.setenv(var, "1")
        assert pbsolve._preload_nvrtc(13, str(tmp_path)) is None
        monkeypatch.delenv(var)
    # A file that is not a loadable library is skipped silently.
    assert pbsolve._preload_nvrtc(13, str(tmp_path)) is None


@needs_cuda
@pytest.mark.parametrize("pre", ["multigrid", "multigrid-f32", "jacobi"])
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
