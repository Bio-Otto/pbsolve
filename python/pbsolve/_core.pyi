"""Type stubs for the compiled extension ``pbsolve._core``.

The docstrings here are the documentation of the compiled module: the
documentation site and editors read them from this file, ``help()`` reads the
same text from the extension (the ``///`` comments in
``crates/pbsolve-py/src/lib.rs``). ``tests/test_docstrings.py`` keeps the two
identical.
"""

from os import PathLike
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__version__: str
_CUDA_MAJOR: int

class ConvergenceError(RuntimeError):
    """The linear solver did not converge.

    Raised when a linear solve does not reach the relative residual ``tol``
    within ``max_iter`` iterations (see :class:`Solver`).
    """

class GridSpec:
    """Focusing grid, the equivalent of APBS ``mg-auto``.

    A coarse box solved with the analytic boundary condition and a fine box
    with the same centre and node counts; intermediate levels are added as
    APBS does when the spacing shrinks by more than a factor of 4 (see
    :attr:`levels`). :meth:`Solver.make_grid` builds one automatically; in
    MM-PBSA, build it once from the complex and reuse it for receptor and
    ligand.

    Parameters
    ----------
    center : sequence of 3 float
        Common centre of all levels in Å (APBS ``cgcent`` = ``fgcent``).
    dims : sequence of 3 int
        Nodes per axis, the same on every level (APBS ``dime``). ``8k+1``
        values suit the multigrid preconditioner.
    coarse_length : sequence of 3 float
        Edge lengths of the coarse box in Å (APBS ``cglen``).
    fine_length : sequence of 3 float, optional
        Edge lengths of the fine box in Å (APBS ``fglen``). Defaults to
        ``coarse_length``: a single grid without focusing.

    Raises
    ------
    ValueError
        If the lengths or node counts do not define valid grids.
    """

    def __init__(
        self,
        center: Sequence[float],
        dims: Sequence[int],
        coarse_length: Sequence[float],
        fine_length: Optional[Sequence[float]] = None,
    ) -> None: ...
    @property
    def center(self) -> list[float]:
        """Common centre of all levels in Å."""
    @property
    def dims(self) -> list[int]:
        """Nodes per axis on every level."""
    @property
    def coarse_length(self) -> list[float]:
        """Edge lengths of the coarse box in Å."""
    @property
    def fine_length(self) -> list[float]:
        """Edge lengths of the fine box in Å."""
    @property
    def fine_spacing(self) -> list[float]:
        """Node spacing of the fine grid in Å."""
    @property
    def is_focused(self) -> bool:
        """Whether the fine box is smaller than the coarse box."""
    @property
    def apbs_dims(self) -> Optional[list[int]]:
        """Node counts APBS would actually use.

        APBS silently resets a ``dime`` above 65 that is not ``32c+1`` to a
        smaller value. ``None`` if APBS would reject the grid.
        """
    @property
    def levels(self) -> list[dict[str, list[float] | list[int]]]:
        """Focusing levels, coarse first, exactly as APBS ``mg-auto`` sets them up.

        A list of dicts with ``dims``, ``spacing``, ``origin``, ``lengths``
        and ``center``.
        """
    def to_apbs(self) -> str:
        """Equivalent APBS ``mg-auto`` keywords, numbers in full precision.

        Raises
        ------
        ValueError
            If APBS would run a different grid (see :attr:`apbs_dims`).
        """

class PolarResult:
    """Result of a polar solvation calculation (:meth:`Solver.polar_solvation`).

    The per-solve lists (:attr:`iterations`, :attr:`residuals`,
    :attr:`preconditioners`, :attr:`levels`) hold the solvated state's levels
    (coarse first), then the reference state's; with ``coulomb="analytic"``
    there is no reference solve.
    """

    energy: float
    """Polar solvation free energy ΔG_polar in kcal/mol."""
    energy_kt: float
    """ΔG_polar in kT."""
    solvated_kt: float
    """Electrostatic energy ½Σqu of the solvated state in kT (with
    ``coulomb="analytic"``: :attr:`reference_kt` + :attr:`energy_kt`)."""
    reference_kt: float
    """Electrostatic energy ½Σqu of the reference state in kT (with
    ``coulomb="analytic"``: the Coulomb energy of the charges in ``pdie``,
    without self-energies)."""
    iterations: list[int]
    """CG iterations of each linear solve."""
    residuals: list[float]
    """Final relative residual of each linear solve."""
    preconditioners: list[str]
    """Preconditioner used by each linear solve ("multigrid-f32", "multigrid" or "jacobi")."""
    levels: list[int]
    """Multigrid levels of each linear solve (1 for Jacobi)."""
    timings: dict[str, float]
    """Wall time per stage in seconds (maps, boundary, assembly, setup, solve, energy and total)."""
    grid: GridSpec
    """The grid of the calculation."""
    platform: str
    """Platform the calculation ran on, "CPU" or "CUDA"."""

class Solver:
    """Linearized Poisson–Boltzmann solver with APBS ``mg-auto`` style focusing.

    Lengths are in Å, charges in e, concentrations in mol/L, potentials in
    kT/e and energies in kcal/mol (attributes ending in ``_kt`` in kT). The
    keyword names follow APBS where it has one.

    Parameters
    ----------
    platform : str
        "auto" (CUDA when available, otherwise the CPU), "CPU" or "CUDA";
        case-insensitive.
    device : int
        CUDA device index.
    grid_spacing : float
        Largest fine-grid spacing in Å of automatic grids
        (:meth:`make_grid`).
    pdie : float
        Solute dielectric constant (APBS ``pdie``).
    sdie : float
        Solvent dielectric constant (APBS ``sdie``).
    temperature : float
        Temperature in K.
    ionic_strength : float
        Ionic strength of a 1:1 salt in mol/L (ions of charge +1 and -1 at
        that concentration). Ignored when ``ions`` is given.
    ion_radius : float
        Radius of both salt ions in Å; it defines the ion-accessibility
        surface.
    ions : sequence of tuple, optional
        Mobile ion species as in APBS ``ion``, each a tuple ``(charge,
        concentration, radius)``: valence in e, concentration in mol/L,
        radius in Å. They must be electroneutral. Replaces
        ``ionic_strength`` and ``ion_radius``.
    srad : float
        Solvent probe radius in Å (APBS ``srad``).
    sdens : float
        Surface sphere density in points per Å² (APBS ``sdens``).
    surface : str
        Dielectric boundary (APBS ``srfm``): "mol" (solvent-excluded
        surface, as APBS ``srfm mol``), "vdw" (union of the atomic spheres)
        or "harmonic" (molecular surface on the nodes with harmonic-mean face
        values; not APBS ``smol``).
    bcfl : str
        Boundary condition of the coarsest grid (APBS ``bcfl``): "mdh" (one
        Debye–Hückel sphere per atom), "mdh-fast" (the same sum interpolated
        on the box faces, error at most 1e-10 of the summed terms; not an
        APBS option), "sdh" (one sphere carrying the total charge) or "zero".
    cfac : float
        Automatic grids: the coarse box is ``cfac`` times the molecular
        extent, and at least the fine box (APBS ``psize``).
    fadd : float
        Automatic grids: the fine box is the molecular extent plus ``fadd``
        Å.
    tol : float
        Relative residual ‖r‖/‖b‖ at which a linear solve stops, in (0, 1).
    max_iter : int
        Largest number of iterations of a linear solve.
    dime_policy : str
        Node counts of automatic grids: "pbsolve" (``8k+1``) or "apbs"
        (``8k+1`` up to 65, ``32c+1`` above, which APBS runs unchanged). Use
        "apbs" to compare with APBS.
    preconditioner : str
        "multigrid-f32" (single-precision multigrid V-cycle inside the f64
        conjugate gradients), "multigrid" (the V-cycle in f64) or "jacobi".
        Energies of the multigrid variants agree to 1e-12 relative.
    chgm : str
        Assignment of the charges to the grid (APBS ``chgm``): "spl0"
        (trilinear, 2 nodes per axis), "spl2" (B-spline of order 3, 3 nodes)
        or "spl4" (B-spline of order 5, 5 nodes). The energy uses the same
        weights with "spl2" and trilinear weights otherwise, as APBS does.
    interface : str
        Dielectric of the grid faces the surface cuts, with ``surface`` "mol"
        or "vdw": "sharp" (``pdie`` or ``sdie`` by the side of the face's
        midpoint, as APBS) or "fractional" (harmonic average of ``pdie`` and
        ``sdie`` weighted by the lengths of the face's grid edge inside and
        outside the solute; close to second order on the analytic references,
        not an APBS option).
    coulomb : str
        How ΔG_polar removes the Coulomb potential of the charges, and with it
        the grid's self-energy: "grid" (solve a reference state with ε =
        ``pdie`` and no ions on the same grid and subtract it, as APBS) or
        "analytic" (reaction-field decomposition: subtract the analytic
        Coulomb potential near the solute and solve only for the reaction
        field, one solve instead of two; not an APBS option). Affects
        :meth:`polar_solvation` and the functions built on it, not
        :meth:`potential`.

    Raises
    ------
    ValueError
        If a parameter is invalid.
    RuntimeError
        If ``platform="CUDA"`` and CUDA is not available
        (:func:`cuda_info` gives the reason).

    Examples
    --------
    >>> solver = pbsolve.Solver(ionic_strength=0.15)
    >>> result = solver.polar_solvation(coords, charges, radii)
    >>> result.energy  # kcal/mol
    """

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
        preconditioner: str = "multigrid-f32",
        chgm: str = "spl0",
        interface: str = "sharp",
        coulomb: str = "grid",
    ) -> None: ...
    @property
    def params(self) -> dict[str, object]:
        """All solver settings as a dict (APBS keyword names where they exist)."""
    @property
    def platform(self) -> str:
        """Platform the solver runs on, "CPU" or "CUDA"."""
    @property
    def device_name(self) -> str:
        """Name of the device the solver runs on."""
    @property
    def device(self) -> int:
        """Index of the CUDA device the solver was created for (``device``)."""
    @property
    def ionic_strength(self) -> float:
        """Ionic strength ½Σcz² of the mobile ions in mol/L."""
    def make_grid(self, coords: npt.ArrayLike, radii: npt.ArrayLike) -> GridSpec:
        """Automatic focusing grid for the given atoms, as APBS ``psize`` sizes it.

        The fine box is the extent of the atoms (radii included) plus
        ``fadd``, the coarse box ``cfac`` times the extent, and the node
        counts give a fine spacing of at most ``grid_spacing``. In MM-PBSA,
        build it once from the complex and reuse it for receptor and ligand.

        Parameters
        ----------
        coords : array_like, shape (N, 3)
            Atom positions in Å.
        radii : array_like, shape (N,)
            Atomic radii in Å.

        Returns
        -------
        GridSpec
        """
    def polar_solvation(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
    ) -> PolarResult:
        """Polar solvation free energy ΔG_polar.

        Solves the solvated state (``sdie``, ions) and the reference state
        (``pdie`` everywhere, no ions) on the same grid; ΔG_polar is
        ½Σq(u_solvated − u_reference). With ``coulomb="analytic"`` it solves
        only for the reaction field u − G (G the Coulomb potential in
        ``pdie``) and ΔG_polar is ½Σq(u − G).

        Parameters
        ----------
        coords : array_like, shape (N, 3)
            Atom positions in Å.
        charges : array_like, shape (N,)
            Atomic charges in e.
        radii : array_like, shape (N,)
            Atomic radii in Å.
        grid : GridSpec, optional
            Grid of the calculation; by default :meth:`make_grid` of the atoms.

        Returns
        -------
        PolarResult

        Raises
        ------
        ValueError
            If the arrays do not match or a charge lies outside the grid.
        ConvergenceError
            If a linear solve does not converge.
        """
    def polar_solvation_batch(
        self,
        frames: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec | Sequence[GridSpec]] = None,
        workers: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """ΔG_polar in kcal/mol for every frame of a trajectory.

        The energies do not depend on ``workers`` and equal those of
        :meth:`polar_solvation` bit for bit.

        Parameters
        ----------
        frames : array_like, shape (F, N, 3)
            Atom positions of every frame in Å.
        charges : array_like, shape (N,)
            Atomic charges in e.
        radii : array_like, shape (N,)
            Atomic radii in Å.
        grid : GridSpec or sequence of GridSpec, optional
            One grid for all frames or one per frame; by default an automatic
            grid per frame.
        workers : int, optional
            Frames solved concurrently. Default: 2 on the CPU, 3 on CUDA; on
            CUDA no more than fit in the free GPU memory (see
            :meth:`batch_workers`).

        Returns
        -------
        numpy.ndarray, shape (F,)
        """
    def polar_solvation_many(
        self,
        systems: Sequence[tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]],
        grid: Optional[GridSpec | Sequence[GridSpec]] = None,
        workers: Optional[int] = None,
    ) -> list[PolarResult]:
        """ΔG_polar for several systems, several of them solved at once.

        The results are in input order and equal, bit for bit, those of
        :meth:`polar_solvation` called on each system in turn.

        Parameters
        ----------
        systems : sequence of tuple
            The systems, each a tuple ``(coords, charges, radii)`` as the
            arguments of :meth:`polar_solvation`.
        grid : GridSpec or sequence of GridSpec, optional
            One grid for all systems or one per system; by default an
            automatic grid per system.
        workers : int, optional
            Systems solved concurrently. Default: 2 on the CPU, 3 on CUDA; on
            CUDA no more than fit in the free GPU memory (see
            :meth:`batch_workers`).

        Returns
        -------
        list of PolarResult
        """
    def batch_workers(
        self,
        grid: GridSpec | Sequence[GridSpec],
        workers: Optional[int] = None,
    ) -> int:
        """Number of calculations the batch methods run at once on ``grid``.

        This is the number :meth:`polar_solvation_many` and
        :meth:`polar_solvation_batch` use for a batch of at least that many
        systems: ``workers`` (default 2 on the CPU, 3 on CUDA), on CUDA
        limited to the calculations that fit in the GPU memory free now while
        1/20 of the device stays free. At least 1.

        Parameters
        ----------
        grid : GridSpec or sequence of GridSpec
            The grid of every calculation, or one per system.
        workers : int, optional
            Upper limit; the default as above.

        Returns
        -------
        int
        """
    def potential(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
        reference: bool = False,
    ) -> tuple[npt.NDArray[np.float64], list[float], list[float], float]:
        """Electrostatic potential on the finest grid.

        Parameters
        ----------
        coords : array_like, shape (N, 3)
            Atom positions in Å.
        charges : array_like, shape (N,)
            Atomic charges in e.
        radii : array_like, shape (N,)
            Atomic radii in Å.
        grid : GridSpec, optional
            Grid of the calculation; by default :meth:`make_grid` of the atoms.
        reference : bool
            Solve the reference state (``pdie`` everywhere, no ions) instead
            of the solvated state.

        Returns
        -------
        potential : numpy.ndarray, shape (nx, ny, nz)
            Potential in kT/e, C order (x is the first index).
        origin : list of 3 float
            Position of node (0, 0, 0) in Å.
        spacing : list of 3 float
            Node spacing in Å.
        energy_kt : float
            Electrostatic energy ½Σqu of that state in kT.
        """
    def maps(
        self,
        coords: npt.ArrayLike,
        charges: npt.ArrayLike,
        radii: npt.ArrayLike,
        grid: Optional[GridSpec] = None,
        reference: bool = False,
        level: int = -1,
    ) -> dict[str, object]:
        """Coefficient maps of one focusing level.

        The maps compare one to one with APBS ``write dielx``, ``diely``,
        ``dielz``, ``kappa`` and ``charge`` (APBS writes kappa as 0/1
        accessibility and the charge as a density in e/Å³).

        Parameters
        ----------
        coords : array_like, shape (N, 3)
            Atom positions in Å.
        charges : array_like, shape (N,)
            Atomic charges in e.
        radii : array_like, shape (N,)
            Atomic radii in Å.
        grid : GridSpec, optional
            Grid of the calculation; by default :meth:`make_grid` of the atoms.
        reference : bool
            Maps of the reference state instead of the solvated state.
        level : int
            Focusing level, counted from the coarsest (0); negative values
            count from the finest (-1).

        Returns
        -------
        dict
            ``eps_x``, ``eps_y``, ``eps_z`` (ε on the faces i+½, j+½, k+½),
            ``kappa_sq`` (κ̄² in Å⁻², 0 where ions are excluded) and
            ``charge`` (e per node, trilinear spreading) as C-order
            (nx, ny, nz) arrays; the level's ``origin``, ``spacing``,
            ``dims``, ``lengths`` and ``center``; and the bulk
            ``kappa_bar_sq``.
        """
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
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        """Solve the LPBE with given coefficient maps.

        Feeding APBS' own maps here tests pbsolve's discretisation and linear
        solver in isolation ("map injection"). Temperature, tolerance,
        preconditioner and platform come from this solver. The arrays are
        C-order (nx, ny, nz), as :meth:`maps` returns them.

        Parameters
        ----------
        eps_x, eps_y, eps_z : numpy.ndarray
            Dielectric constant on the faces i+½, j+½, k+½.
        kappa_sq : numpy.ndarray
            κ̄² in Å⁻².
        charge : numpy.ndarray
            Charge per node in e.
        boundary : numpy.ndarray
            Potential in kT/e; only the boundary nodes are read.
        origin : sequence of 3 float
            Position of node (0, 0, 0) in Å.
        spacing : sequence of 3 float
            Node spacing in Å.

        Returns
        -------
        potential : numpy.ndarray, shape (nx, ny, nz)
            Potential in kT/e.
        iterations : int
            CG iterations.
        residual : float
            Final relative residual.
        """

def available_platforms() -> list[str]:
    """Platforms usable in this process, fastest first (e.g. ``["CUDA", "CPU"]``)."""

def cuda_info() -> dict[str, object]:
    """Build and runtime information about CUDA support.

    Returns
    -------
    dict
        ``compiled`` (built with CUDA support), ``cuda_major`` (CUDA major
        version of the build, 0 without CUDA), ``available``,
        ``device_count``, ``reason`` (why CUDA is unavailable, or ``None``),
        ``driver`` and ``nvrtc`` (versions as "major.minor", or ``None``) and
        ``nvrtc_library`` (the NVRTC library loaded).
    """

def sasa(
    coords: npt.ArrayLike, radii: npt.ArrayLike, probe: float = 1.4, n_points: int = 960
) -> npt.NDArray[np.float64]:
    """Solvent-accessible surface area of each atom (Shrake–Rupley).

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Atom positions in Å.
    radii : array_like, shape (N,)
        Atomic radii in Å.
    probe : float
        Solvent probe radius in Å.
    n_points : int
        Test points per atom sphere.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Area of each atom in Å².
    """

def kt_kj_per_mol(temperature: float = 298.15) -> float:
    """k_B·T in kJ/mol (the unit APBS reports energies in)."""

def kt_kcal_per_mol(temperature: float = 298.15) -> float:
    """k_B·T in kcal/mol."""

def apbs_effective_dime(dime: int) -> Optional[int]:
    """The ``dime`` APBS actually uses for a requested value.

    APBS resets a ``dime`` above 65 that is not ``32c+1`` to a smaller value
    (121 becomes 97). ``None`` if APBS rejects the value.
    """

def bjerrum_length(temperature: float = 298.15) -> float:
    """Vacuum Bjerrum length e²/(4πε₀k_BT) in Å."""

def read_structure(
    path: str | PathLike[str], format: Optional[str] = None, include_dirs: Optional[Sequence[str]] = None
) -> dict[str, object]:
    """Read a structure, topology or coordinate file.

    :func:`pbsolve.io.load` builds a molecule from this; use it unless you
    need the raw arrays.

    Parameters
    ----------
    path : str or path-like
        The file.
    format : str, optional
        File format ("pdb", "pqr", "cif", "gro", "mol2", "prmtop", "psf",
        "gmxtop", "inpcrd", "netcdf", "mdcrd", "dcd", "xtc", "trr");
        detected from the extension by default.
    include_dirs : sequence of str, optional
        Extra directories searched for GROMACS ``#include`` files.

    Returns
    -------
    dict
        Per-atom lists and arrays (``names``, ``resnames``, ``resids``,
        ``icodes``, ``chains``, ``segids``, ``elements``, ``atom_types``,
        ``hetero``, ``charges``, ``radii``, ``masses``; NaN where the format
        has no value), ``bonds`` (B, 2), ``coords`` (F, N, 3) in Å,
        ``cells`` (F, 6), ``times`` (F,), ``format``, ``periodic`` and
        ``notes``.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file is malformed or the format unknown.
    """

class TrajectoryReader:
    """Streams the frames of a trajectory.

    Reads DCD, XTC, TRR, Amber NetCDF and mdcrd, and multi-model PDB, GRO
    and mmCIF files. Iterating yields ``(coords, cell, time, step)``:
    positions (N, 3) in Å, the unit cell (a, b, c in Å, α, β, γ in degrees)
    or ``None``, the time or ``None`` and the step or ``None``.

    Parameters
    ----------
    path : str or path-like
        The trajectory file.
    format : str, optional
        File format; detected from the extension by default.
    n_atoms : int, optional
        Atoms per frame; required for ASCII mdcrd, checked otherwise.
    mdcrd_box : bool, optional
        Whether mdcrd frames carry a box line; detected by default.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the file is malformed or the format unknown.
    """

    def __init__(
        self,
        path: str | PathLike[str],
        format: Optional[str] = None,
        n_atoms: Optional[int] = None,
        mdcrd_box: Optional[bool] = None,
    ) -> None: ...
    @property
    def n_atoms(self) -> Optional[int]:
        """Atoms per frame, if known before reading a frame."""
    @property
    def n_frames(self) -> Optional[int]:
        """Number of frames if known without reading the file (DCD, NetCDF, multi-model files)."""
    def __iter__(self) -> "TrajectoryReader": ...
    def __next__(
        self,
    ) -> tuple[npt.NDArray[np.float64], Optional[list[float]], Optional[float], Optional[int]]: ...

def read_pqr(
    path: str | PathLike[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read a PQR file.

    Returns
    -------
    coords : numpy.ndarray, shape (N, 3)
        Atom positions in Å.
    charges : numpy.ndarray, shape (N,)
        Atomic charges in e.
    radii : numpy.ndarray, shape (N,)
        Atomic radii in Å.
    """
