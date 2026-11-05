"""Type stubs for the compiled extension ``pbsolve._core``."""

from os import PathLike
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__version__: str
_CUDA_MAJOR: int

class ConvergenceError(RuntimeError): ...

class GridSpec:
    """Two-level focusing grid (APBS mg-auto: dime, cglen, fglen, cgcent = fgcent)."""

    def __init__(
        self,
        center: Sequence[float],
        dims: Sequence[int],
        coarse_length: Sequence[float],
        fine_length: Optional[Sequence[float]] = None,
    ) -> None: ...
    @property
    def center(self) -> list[float]: ...
    @property
    def dims(self) -> list[int]: ...
    @property
    def coarse_length(self) -> list[float]: ...
    @property
    def fine_length(self) -> list[float]: ...
    @property
    def fine_spacing(self) -> list[float]: ...
    @property
    def is_focused(self) -> bool: ...
    @property
    def apbs_dims(self) -> Optional[list[int]]:
        """Node counts APBS would actually use (it resets dime > 65 that is not 32c+1)."""
    @property
    def levels(self) -> list[dict[str, list[float] | list[int]]]:
        """Focusing levels (coarse first) as APBS mg-auto sets them up:
        dicts with ``dims``, ``spacing``, ``origin``, ``lengths``, ``center``."""
    def to_apbs(self) -> str:
        """APBS mg-auto keywords; raises ValueError if APBS would change the grid."""

class PolarResult:
    energy: float  # kcal/mol
    energy_kt: float
    solvated_kt: float
    reference_kt: float
    iterations: list[int]
    residuals: list[float]
    preconditioners: list[str]  # per linear solve: "multigrid", "multigrid-f32" or "jacobi"
    levels: list[int]  # multigrid levels per linear solve (1 for Jacobi)
    # seconds per stage: maps, boundary, assembly, setup, solve, energy, total
    timings: dict[str, float]
    grid: GridSpec
    platform: str

class Solver:
    def __init__(
        self,
        platform: str = "auto",
        device: int = 0,
        grid_spacing: float = 0.5,
        pdie: float = 1.0,
        sdie: float = 78.54,
        temperature: float = 298.15,
        ionic_strength: float = 0.0,
        ion_radius: float = 2.0,
        ions: Optional[Sequence[tuple[float, float, float]]] = None,
        srad: float = 1.4,
        sdens: float = 10.0,
        surface: str = "mol",
        bcfl: str = "mdh",
        cfac: float = 1.7,
        fadd: float = 20.0,
        tol: float = 1e-8,
        max_iter: int = 20000,
        dime_policy: str = "pbsolve",
        preconditioner: str = "multigrid-f32",  # f32 V-cycle in f64 CG; or "multigrid" (f64), "jacobi"
    ) -> None: ...
    @property
    def params(self) -> dict[str, object]: ...
    @property
    def platform(self) -> str: ...
    @property
    def device_name(self) -> str: ...
    @property
    def ionic_strength(self) -> float: ...
    def make_grid(self, coords: npt.ArrayLike, radii: npt.ArrayLike) -> GridSpec: ...
    def polar_solvation(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
    ) -> PolarResult: ...
    def polar_solvation_batch(
        self,
        frames: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec | Sequence[GridSpec]] = None,
        workers: Optional[int] = None,
    ) -> npt.NDArray[np.float64]: ...
    def polar_solvation_many(
        self,
        systems: Sequence[tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]],
        grid: Optional[GridSpec | Sequence[GridSpec]] = None,
        workers: Optional[int] = None,
    ) -> list[PolarResult]: ...
    def potential(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
        reference: bool = False,
    ) -> tuple[npt.NDArray[np.float64], list[float], list[float], float]: ...
    def maps(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
        reference: bool = False,
        level: int = -1,
    ) -> dict[str, object]:
        """eps_x/eps_y/eps_z, kappa_sq, charge (C-order arrays) + origin, spacing, dims, kappa_bar_sq."""
    def solve_maps(
        self,
        eps_x: npt.NDArray[np.float64],
        eps_y: npt.NDArray[np.float64],
        eps_z: npt.NDArray[np.float64],
        kappa_sq: npt.NDArray[np.float64],
        charge: npt.NDArray[np.float64],
        boundary: npt.NDArray[np.float64],
        origin: Sequence[float],
        spacing: Sequence[float],
    ) -> tuple[npt.NDArray[np.float64], int, float]: ...

def available_platforms() -> list[str]: ...
def cuda_info() -> dict[str, object]: ...
def sasa(
    coords: npt.ArrayLike, radii: npt.ArrayLike, probe: float = 1.4, n_points: int = 960
) -> npt.NDArray[np.float64]: ...
def kt_kj_per_mol(temperature: float = 298.15) -> float: ...
def kt_kcal_per_mol(temperature: float = 298.15) -> float: ...
def apbs_effective_dime(dime: int) -> Optional[int]: ...
def bjerrum_length(temperature: float = 298.15) -> float: ...
def read_structure(
    path: str | PathLike[str], format: Optional[str] = None, include_dirs: Optional[Sequence[str]] = None
) -> dict[str, object]:
    """Per-atom data, bonds, coords (F, N, 3) Å, cells, times, format, periodic, notes."""

class TrajectoryReader:
    def __init__(
        self,
        path: str | PathLike[str],
        format: Optional[str] = None,
        n_atoms: Optional[int] = None,
        mdcrd_box: Optional[bool] = None,
    ) -> None: ...
    @property
    def n_atoms(self) -> Optional[int]: ...
    @property
    def n_frames(self) -> Optional[int]: ...
    def __iter__(self) -> "TrajectoryReader": ...
    def __next__(
        self,
    ) -> tuple[npt.NDArray[np.float64], Optional[list[float]], Optional[float], Optional[int]]: ...

def read_pqr(
    path: str | PathLike[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
