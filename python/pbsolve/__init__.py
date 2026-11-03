"""pbsolve — fast Poisson–Boltzmann electrostatics and solvation energies (CPU/CUDA).

Quick start::

    import pbsolve

    solver = pbsolve.Solver(platform="auto", grid_spacing=0.5, ionic_strength=0.15)
    result = solver.polar_solvation(coords, charges, radii)  # Å, e, Å
    print(result.energy, "kcal/mol on", result.platform)

Units: lengths in Å, charges in e, concentrations in mol/L, energies in
kcal/mol (``*_kt`` attributes are in kT), potentials in kT/e.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys


def _preload_cuda_runtime() -> None:
    """Make the environment's NVRTC visible to the runtime loader.

    The extension loads libcuda (from the NVIDIA driver) and NVRTC (from the
    conda package ``cuda-nvrtc``) lazily. Preloading NVRTC by absolute path
    makes discovery independent of RPATH / LD_LIBRARY_PATH details. A missing
    library is not an error: pbsolve then runs on the CPU.
    Set ``PBSOLVE_NO_PRELOAD=1`` to disable.
    """
    if os.environ.get("PBSOLVE_NO_PRELOAD"):
        return
    if sys.platform == "win32":
        libdir = os.path.join(sys.prefix, "Library", "bin")
        if os.path.isdir(libdir):
            os.add_dll_directory(libdir)
        return
    if not sys.platform.startswith("linux"):
        return
    for libdir in (os.path.join(sys.prefix, "lib"), os.path.join(sys.prefix, "lib64")):
        # Shortest name first: libnvrtc.so.12 before libnvrtc.so.12.8.93.
        for path in sorted(glob.glob(os.path.join(libdir, "libnvrtc.so.*")), key=len):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_preload_cuda_runtime()

from ._core import (  # noqa: E402
    ConvergenceError,
    GridSpec,
    PolarResult,
    Solver,
    __version__,
    apbs_effective_dime,
    available_platforms,
    bjerrum_length,
    cuda_info,
    kt_kcal_per_mol,
    kt_kj_per_mol,
    TrajectoryReader,
    read_pqr,
    read_structure,
    sasa,
)
from . import io, prep, radii  # noqa: E402
from .dx import read_dx, write_dx  # noqa: E402
from .mmpbsa import (  # noqa: E402
    SITKOFF_BETA,
    SITKOFF_GAMMA,
    PBSAFrame,
    SolvationTerms,
    nonpolar_energy,
    pbsa_solvation_frame,
)

__all__ = [
    "ConvergenceError",
    "GridSpec",
    "PBSAFrame",
    "PolarResult",
    "SITKOFF_BETA",
    "SITKOFF_GAMMA",
    "SolvationTerms",
    "Solver",
    "TrajectoryReader",
    "__version__",
    "apbs_effective_dime",
    "available_platforms",
    "bjerrum_length",
    "cuda_info",
    "io",
    "kt_kcal_per_mol",
    "kt_kj_per_mol",
    "nonpolar_energy",
    "pbsa_solvation_frame",
    "prep",
    "radii",
    "read_dx",
    "read_pqr",
    "read_structure",
    "sasa",
    "write_dx",
]
