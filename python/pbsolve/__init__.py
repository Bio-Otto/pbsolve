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
import importlib.util
import os
import sys
from typing import Optional


def _nvrtc_candidates(major: int, prefix: str = sys.prefix) -> list[str]:
    """NVRTC libraries of CUDA ``major`` installed in this Python environment.

    Looks in the environment's library directory (conda ``cuda-nvrtc``) and
    in the ``nvidia`` packages pip installs (``nvidia-cuda-nvrtc-cu12`` puts
    the library in ``nvidia/cuda_nvrtc``, ``nvidia-cuda-nvrtc`` 13 in
    ``nvidia/cu13``), in that order.
    """
    if major <= 0:
        return []
    if sys.platform.startswith("linux"):
        name = f"libnvrtc.so.{major}"
        dirs = [os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")]
        sub = ["cuda_nvrtc/lib", f"cu{major}/lib"]
    elif sys.platform == "win32":
        name = f"nvrtc64_{major}0_0.dll"
        dirs = [os.path.join(prefix, "Library", "bin"), os.path.join(prefix, "Library", "bin", "x64")]
        sub = ["cuda_nvrtc/bin", f"cu{major}/bin", f"cu{major}/bin/x86_64"]
    else:
        return []
    try:
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations or []) if spec else []
    except (ImportError, ValueError):
        roots = []
    dirs += [os.path.join(root, *s.split("/")) for root in roots for s in sub]
    found = []
    for d in dirs:
        path = os.path.join(d, name)
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return found


def _preload_nvrtc(major: int, prefix: str = sys.prefix) -> Optional[str]:
    """Load this environment's NVRTC so that the CUDA backend uses it.

    The backend opens NVRTC by its versioned name (``libnvrtc.so.<major>``,
    ``nvrtc64_<major>0_0.dll``); the dynamic loader returns an already-loaded
    library of that name, so the environment's NVRTC wins over a system CUDA
    toolkit, and pip-installed NVRTC (not on any search path) is found at all.
    A missing library is not an error: pbsolve then runs on the CPU. Returns
    the loaded path. Skipped when ``PBSOLVE_NO_PRELOAD`` is set or
    ``PBSOLVE_NVRTC`` names the library explicitly.
    """
    if os.environ.get("PBSOLVE_NO_PRELOAD") or os.environ.get("PBSOLVE_NVRTC"):
        return None
    if sys.platform == "win32":
        libdir = os.path.join(prefix, "Library", "bin")
        if os.path.isdir(libdir):
            os.add_dll_directory(libdir)
    for path in _nvrtc_candidates(major, prefix):
        try:
            ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return path
        except OSError:
            continue
    return None


from ._core import (  # noqa: E402
    _CUDA_MAJOR,
    ConvergenceError,
    GridSpec,
    PolarForces,
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

# NVRTC is opened lazily (first CUDA query), so preloading after the import works.
_preload_nvrtc(_CUDA_MAJOR)

from . import io, prep, radii, timeseries  # noqa: E402
from .apolar import ApolarEnergy, apolar_energy  # noqa: E402
from .dx import read_dx, write_dx  # noqa: E402
from .entropy import InteractionEntropy, interaction_energy, interaction_entropy  # noqa: E402
from .estimate import ErrorEstimate, estimate_error  # noqa: E402
from .mmpbsa import (  # noqa: E402
    SITKOFF_BETA,
    SITKOFF_GAMMA,
    PBSAFrame,
    SolvationTerms,
    nonpolar_energy,
    pbsa_pair_terms,
    pbsa_solvation_frame,
    pbsa_solvation_frames,
    residue_dielectrics,
    residue_pair_terms,
)
from .timeseries import SeriesStatistics, series_statistics  # noqa: E402

__all__ = [
    "ApolarEnergy",
    "ConvergenceError",
    "ErrorEstimate",
    "GridSpec",
    "InteractionEntropy",
    "PBSAFrame",
    "PolarForces",
    "PolarResult",
    "SITKOFF_BETA",
    "SITKOFF_GAMMA",
    "SeriesStatistics",
    "SolvationTerms",
    "Solver",
    "TrajectoryReader",
    "__version__",
    "apbs_effective_dime",
    "apolar_energy",
    "available_platforms",
    "bjerrum_length",
    "cuda_info",
    "estimate_error",
    "interaction_energy",
    "interaction_entropy",
    "io",
    "kt_kcal_per_mol",
    "kt_kj_per_mol",
    "nonpolar_energy",
    "pbsa_pair_terms",
    "pbsa_solvation_frame",
    "pbsa_solvation_frames",
    "prep",
    "radii",
    "read_dx",
    "read_pqr",
    "read_structure",
    "residue_dielectrics",
    "residue_pair_terms",
    "sasa",
    "series_statistics",
    "timeseries",
    "write_dx",
]
