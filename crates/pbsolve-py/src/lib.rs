//! Python bindings (`pbsolve._core`). Keep this layer thin: argument
//! conversion, platform selection and GIL release only. Physics lives in
//! `pbsolve-core`, ergonomics in the pure-Python package.

use numpy::ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pbsolve_core::batch::{device_workers, polar_solvation_batch as core_batch, worker_device_bytes, Job};
use pbsolve_core::charge::spread_charges;
use pbsolve_core::molecule::{atoms_from_arrays, read_pqr as core_read_pqr};
use pbsolve_core::pb::{electrostatic_energy_kt, FieldSolution};
use pbsolve_core::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol, kt_in_kj_per_mol};
use pbsolve_core::{
    apbs_effective_dime as core_apbs_effective_dime, coefficient_maps, polar_solvation, solve_focused, solve_with_maps,
    Atom, Backend, BoundaryCondition, CoefficientMaps, CpuBackend, DimePolicy, Grid, GridSpec, IonSpecies, Medium,
    PbError, PbParams, PolarSolvation, Preconditioner, SolveStats, SolverOptions, SurfaceMethod,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(
    pbsolve,
    ConvergenceError,
    PyRuntimeError,
    concat!(
        "The linear solver did not converge.\n",
        "\n",
        "Raised when a linear solve does not reach the relative residual ``tol``\n",
        "within ``max_iter`` iterations (see :class:`Solver`).",
    )
);

fn to_py(e: PbError) -> PyErr {
    match e {
        PbError::InvalidInput(_) | PbError::Parse { .. } => PyValueError::new_err(e.to_string()),
        PbError::Io(_) => PyOSError::new_err(e.to_string()),
        PbError::NotConverged { .. } => ConvergenceError::new_err(e.to_string()),
        PbError::Backend(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------- platforms

enum Engine {
    Cpu(CpuBackend),
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    Cuda(Box<pbsolve_cuda::CudaBackend>),
}

macro_rules! dispatch {
    ($engine:expr, $be:ident => $body:expr) => {
        match $engine {
            Engine::Cpu($be) => $body,
            #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
            Engine::Cuda($be) => {
                let $be: &pbsolve_cuda::CudaBackend = $be;
                $body
            }
        }
    };
}

impl Engine {
    fn create(platform: &str, device: usize) -> PyResult<Self> {
        match platform.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Engine::Cpu(CpuBackend::new())),
            "cuda" => Self::cuda(device),
            "auto" => {
                if pbsolve_cuda::is_available() {
                    Self::cuda(device).or_else(|_| Ok(Engine::Cpu(CpuBackend::new())))
                } else {
                    Ok(Engine::Cpu(CpuBackend::new()))
                }
            }
            other => Err(PyValueError::new_err(format!(
                "unknown platform '{other}' (expected 'auto', 'CPU' or 'CUDA')"
            ))),
        }
    }

    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    fn cuda(device: usize) -> PyResult<Self> {
        pbsolve_cuda::CudaBackend::new(device)
            .map(|b| Engine::Cuda(Box::new(b)))
            .map_err(to_py)
    }

    #[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
    fn cuda(_device: usize) -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            pbsolve_cuda::unavailable_reason().unwrap_or_else(|| "CUDA unavailable".into()),
        ))
    }

    fn platform(&self) -> &'static str {
        dispatch!(self, be => be.platform())
    }

    fn device_name(&self) -> String {
        dispatch!(self, be => be.device_name())
    }

    /// Free and total device memory in bytes; `None` on the CPU.
    fn device_memory(&self) -> Result<Option<(u64, u64)>, PbError> {
        match self {
            Engine::Cpu(_) => Ok(None),
            #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
            Engine::Cuda(be) => be.memory_info().map(Some),
        }
    }

    fn polar(&self, atoms: &[Atom], params: &PbParams, spec: &GridSpec) -> Result<PolarSolvation, PbError> {
        dispatch!(self, be => polar_solvation(be, atoms, params, spec))
    }

    fn field(
        &self,
        atoms: &[Atom],
        params: &PbParams,
        spec: &GridSpec,
        medium: Medium,
    ) -> Result<FieldSolution, PbError> {
        dispatch!(self, be => solve_focused(be, spec, atoms, params, medium))
    }

    fn with_maps(
        &self,
        grid: &Grid,
        maps: CoefficientMaps,
        charge: &[f64],
        g_ext: &[f64],
        temperature: f64,
        opts: SolverOptions,
    ) -> Result<(Vec<f64>, SolveStats), PbError> {
        dispatch!(self, be => solve_with_maps(be, grid, maps, charge, g_ext, temperature, opts))
    }
}

/// Platforms usable in this process, fastest first (e.g. ``["CUDA", "CPU"]``).
#[pyfunction]
fn available_platforms() -> Vec<&'static str> {
    let mut v = Vec::new();
    if pbsolve_cuda::is_available() {
        v.push("CUDA");
    }
    v.push("CPU");
    v
}

/// Build and runtime information about CUDA support.
///
/// Returns
/// -------
/// dict
///     ``compiled`` (built with CUDA support), ``cuda_major`` (CUDA major
///     version of the build, 0 without CUDA), ``available``,
///     ``device_count``, ``reason`` (why CUDA is unavailable, or ``None``),
///     ``driver`` and ``nvrtc`` (versions as "major.minor", or ``None``) and
///     ``nvrtc_library`` (the NVRTC library loaded).
#[pyfunction]
fn cuda_info(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("compiled", pbsolve_cuda::compiled_with_cuda())?;
    d.set_item("cuda_major", pbsolve_cuda::cuda_major())?;
    d.set_item("available", pbsolve_cuda::is_available())?;
    d.set_item("device_count", pbsolve_cuda::device_count())?;
    d.set_item("reason", pbsolve_cuda::unavailable_reason())?;
    let version = |(major, minor): (i32, i32)| format!("{major}.{minor}");
    d.set_item("driver", pbsolve_cuda::driver_version().map(version))?;
    let nvrtc = pbsolve_cuda::compiled_with_cuda()
        .then(pbsolve_cuda::nvrtc_info)
        .and_then(|r| r.ok());
    d.set_item("nvrtc", nvrtc.as_ref().map(|n| version(n.version)))?;
    d.set_item(
        "nvrtc_library",
        nvrtc.as_ref().map(|n| n.path.clone().unwrap_or_else(|| n.name.clone())),
    )?;
    Ok(d)
}

// ---------------------------------------------------------------- helpers

fn coords_from(view: ArrayView2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "coordinates must have shape (N, 3), got {:?}",
            view.shape()
        )));
    }
    Ok(view.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn atoms_from(
    coords: &PyReadonlyArray2<'_, f64>,
    charges: &PyReadonlyArray1<'_, f64>,
    radii: &PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<Atom>> {
    let xyz = coords_from(coords.as_array())?;
    let q = charges.as_array().to_vec();
    let r = radii.as_array().to_vec();
    atoms_from_arrays(&xyz, &q, &r).map_err(to_py)
}

/// x-fastest storage -> C-order (nx, ny, nz) numpy array.
fn to_numpy<'py>(py: Python<'py>, dims: [usize; 3], field: &[f64]) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let [nx, ny, nz] = dims;
    let mut v = vec![0.0; field.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                v[(i * ny + j) * nz + k] = field[i + nx * (j + ny * k)];
            }
        }
    }
    PyArray1::from_vec(py, v).reshape([nx, ny, nz])
}

/// C-order (nx, ny, nz) numpy array -> x-fastest storage, checking the shape.
fn from_numpy(
    name: &str,
    arr: &PyReadonlyArray3<'_, f64>,
    dims: Option<[usize; 3]>,
) -> PyResult<(Vec<f64>, [usize; 3])> {
    let a = arr.as_array();
    let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
    if let Some(d) = dims {
        if d != shape {
            return Err(PyValueError::new_err(format!(
                "{name} has shape {shape:?}, expected {d:?}"
            )));
        }
    }
    let [nx, ny, nz] = shape;
    let mut v = vec![0.0; nx * ny * nz];
    for ((i, j, k), x) in a.indexed_iter() {
        v[i + nx * (j + ny * k)] = *x;
    }
    Ok((v, shape))
}

fn grid_dict<'py>(py: Python<'py>, g: &Grid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("dims", g.dims)?;
    d.set_item("spacing", g.spacing)?;
    d.set_item("origin", g.origin)?;
    d.set_item("lengths", g.lengths())?;
    d.set_item("center", g.center())?;
    Ok(d)
}

// ---------------------------------------------------------------- GridSpec

/// Focusing grid, the equivalent of APBS ``mg-auto``.
///
/// A coarse box solved with the analytic boundary condition and a fine box
/// with the same centre and node counts; intermediate levels are added as
/// APBS does when the spacing shrinks by more than a factor of 4 (see
/// :attr:`levels`). :meth:`Solver.make_grid` builds one automatically; in
/// MM-PBSA, build it once from the complex and reuse it for receptor and
/// ligand.
///
/// Parameters
/// ----------
/// center : sequence of 3 float
///     Common centre of all levels in Å (APBS ``cgcent`` = ``fgcent``).
/// dims : sequence of 3 int
///     Nodes per axis, the same on every level (APBS ``dime``). ``8k+1``
///     values suit the multigrid preconditioner.
/// coarse_length : sequence of 3 float
///     Edge lengths of the coarse box in Å (APBS ``cglen``).
/// fine_length : sequence of 3 float, optional
///     Edge lengths of the fine box in Å (APBS ``fglen``). Defaults to
///     ``coarse_length``: a single grid without focusing.
///
/// Raises
/// ------
/// ValueError
///     If the lengths or node counts do not define valid grids.
#[pyclass(name = "GridSpec", module = "pbsolve", frozen, from_py_object)]
#[derive(Clone)]
struct PyGridSpec {
    inner: GridSpec,
}

#[pymethods]
impl PyGridSpec {
    #[new]
    #[pyo3(signature = (center, dims, coarse_length, fine_length=None))]
    fn new(
        center: [f64; 3],
        dims: [usize; 3],
        coarse_length: [f64; 3],
        fine_length: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let inner = GridSpec {
            center,
            dims,
            coarse_length,
            fine_length: fine_length.unwrap_or(coarse_length),
        };
        inner.validate().map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Common centre of all levels in Å.
    #[getter]
    fn center(&self) -> [f64; 3] {
        self.inner.center
    }
    /// Nodes per axis on every level.
    #[getter]
    fn dims(&self) -> [usize; 3] {
        self.inner.dims
    }
    /// Edge lengths of the coarse box in Å.
    #[getter]
    fn coarse_length(&self) -> [f64; 3] {
        self.inner.coarse_length
    }
    /// Edge lengths of the fine box in Å.
    #[getter]
    fn fine_length(&self) -> [f64; 3] {
        self.inner.fine_length
    }
    /// Node spacing of the fine grid in Å.
    #[getter]
    fn fine_spacing(&self) -> [f64; 3] {
        self.inner.fine_spacing()
    }
    /// Whether the fine box is smaller than the coarse box.
    #[getter]
    fn is_focused(&self) -> bool {
        self.inner.is_focused()
    }

    /// Node counts APBS would actually use.
    ///
    /// APBS silently resets a ``dime`` above 65 that is not ``32c+1`` to a
    /// smaller value. ``None`` if APBS would reject the grid.
    #[getter]
    fn apbs_dims(&self) -> Option<[usize; 3]> {
        self.inner.apbs_dims()
    }

    /// Focusing levels, coarse first, exactly as APBS ``mg-auto`` sets them up.
    ///
    /// A list of dicts with ``dims``, ``spacing``, ``origin``, ``lengths``
    /// and ``center``.
    #[getter]
    fn levels<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let levels = self.inner.levels().map_err(to_py)?;
        levels.iter().map(|g| grid_dict(py, g)).collect()
    }

    /// Equivalent APBS ``mg-auto`` keywords, numbers in full precision.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If APBS would run a different grid (see :attr:`apbs_dims`).
    fn to_apbs(&self) -> PyResult<String> {
        self.inner.to_apbs().map_err(to_py)
    }

    fn __repr__(&self) -> String {
        let h = self.inner.fine_spacing();
        format!(
            "GridSpec(dims={:?}, coarse_length={:?}, fine_length={:?}, fine_spacing=[{:.3}, {:.3}, {:.3}])",
            self.inner.dims, self.inner.coarse_length, self.inner.fine_length, h[0], h[1], h[2]
        )
    }
}

// ---------------------------------------------------------------- results

/// Result of a polar solvation calculation (:meth:`Solver.polar_solvation`).
///
/// The per-solve lists (:attr:`iterations`, :attr:`residuals`,
/// :attr:`preconditioners`, :attr:`levels`) hold the solvated state's levels
/// (coarse first), then the reference state's.
#[pyclass(name = "PolarResult", module = "pbsolve", frozen, get_all)]
struct PyPolarResult {
    /// Polar solvation free energy ΔG_polar in kcal/mol.
    energy: f64,
    /// ΔG_polar in kT.
    energy_kt: f64,
    /// Electrostatic energy ½Σqu of the solvated state in kT.
    solvated_kt: f64,
    /// Electrostatic energy ½Σqu of the reference state in kT.
    reference_kt: f64,
    /// CG iterations of each linear solve.
    iterations: Vec<usize>,
    /// Final relative residual of each linear solve.
    residuals: Vec<f64>,
    /// Preconditioner used by each linear solve ("multigrid-f32", "multigrid" or "jacobi").
    preconditioners: Vec<&'static str>,
    /// Multigrid levels of each linear solve (1 for Jacobi).
    levels: Vec<usize>,
    /// Wall time per stage in seconds (maps, boundary, assembly, setup, solve, energy and total).
    timings: std::collections::BTreeMap<&'static str, f64>,
    /// The grid of the calculation.
    grid: PyGridSpec,
    /// Platform the calculation ran on, "CPU" or "CUDA".
    platform: String,
}

#[pymethods]
impl PyPolarResult {
    fn __repr__(&self) -> String {
        format!(
            "PolarResult(energy={:.4} kcal/mol, platform='{}', iterations={:?})",
            self.energy, self.platform, self.iterations
        )
    }
}

fn timings_map(t: &pbsolve_core::Timings) -> std::collections::BTreeMap<&'static str, f64> {
    [
        ("maps", t.maps),
        ("boundary", t.boundary),
        ("assembly", t.assembly),
        ("setup", t.setup),
        ("solve", t.solve),
        ("energy", t.energy),
        ("total", t.total()),
    ]
    .into_iter()
    .collect()
}

impl PyPolarResult {
    fn from_core(r: PolarSolvation, platform: &str) -> Self {
        let stats = r.stats_solvated.iter().chain(&r.stats_reference);
        Self {
            energy: r.energy_kcal,
            energy_kt: r.energy_kt,
            solvated_kt: r.solvated_kt,
            reference_kt: r.reference_kt,
            iterations: stats.clone().map(|s| s.iterations).collect(),
            residuals: stats.clone().map(|s| s.relative_residual).collect(),
            preconditioners: stats.clone().map(|s| s.preconditioner.as_str()).collect(),
            levels: stats.map(|s| s.levels).collect(),
            timings: timings_map(&r.timings),
            grid: PyGridSpec { inner: r.grid },
            platform: platform.to_string(),
        }
    }
}

// ---------------------------------------------------------------- Solver

/// Linearized Poisson–Boltzmann solver with APBS ``mg-auto`` style focusing.
///
/// Lengths are in Å, charges in e, concentrations in mol/L, potentials in
/// kT/e and energies in kcal/mol (attributes ending in ``_kt`` in kT). The
/// keyword names follow APBS where it has one.
///
/// Parameters
/// ----------
/// platform : str
///     "auto" (CUDA when available, otherwise the CPU), "CPU" or "CUDA";
///     case-insensitive.
/// device : int
///     CUDA device index.
/// grid_spacing : float
///     Largest fine-grid spacing in Å of automatic grids
///     (:meth:`make_grid`).
/// pdie : float
///     Solute dielectric constant (APBS ``pdie``).
/// sdie : float
///     Solvent dielectric constant (APBS ``sdie``).
/// temperature : float
///     Temperature in K.
/// ionic_strength : float
///     Ionic strength of a 1:1 salt in mol/L (ions of charge +1 and -1 at
///     that concentration). Ignored when ``ions`` is given.
/// ion_radius : float
///     Radius of both salt ions in Å; it defines the ion-accessibility
///     surface.
/// ions : sequence of tuple, optional
///     Mobile ion species as in APBS ``ion``, each a tuple ``(charge,
///     concentration, radius)``: valence in e, concentration in mol/L,
///     radius in Å. They must be electroneutral. Replaces
///     ``ionic_strength`` and ``ion_radius``.
/// srad : float
///     Solvent probe radius in Å (APBS ``srad``).
/// sdens : float
///     Surface sphere density in points per Å² (APBS ``sdens``).
/// surface : str
///     Dielectric boundary (APBS ``srfm``): "mol" (solvent-excluded
///     surface, as APBS ``srfm mol``), "vdw" (union of the atomic spheres)
///     or "harmonic" (molecular surface on the nodes with harmonic-mean face
///     values; not APBS ``smol``).
/// bcfl : str
///     Boundary condition of the coarsest grid (APBS ``bcfl``): "mdh" (one
///     Debye–Hückel sphere per atom), "mdh-fast" (the same sum interpolated
///     on the box faces, error at most 1e-10 of the summed terms; not an
///     APBS option), "sdh" (one sphere carrying the total charge) or "zero".
/// cfac : float
///     Automatic grids: the coarse box is ``cfac`` times the molecular
///     extent, and at least the fine box (APBS ``psize``).
/// fadd : float
///     Automatic grids: the fine box is the molecular extent plus ``fadd``
///     Å.
/// tol : float
///     Relative residual ‖r‖/‖b‖ at which a linear solve stops, in (0, 1).
/// max_iter : int
///     Largest number of iterations of a linear solve.
/// dime_policy : str
///     Node counts of automatic grids: "pbsolve" (``8k+1``) or "apbs"
///     (``8k+1`` up to 65, ``32c+1`` above, which APBS runs unchanged). Use
///     "apbs" to compare with APBS.
/// preconditioner : str
///     "multigrid-f32" (single-precision multigrid V-cycle inside the f64
///     conjugate gradients), "multigrid" (the V-cycle in f64) or "jacobi".
///     Energies of the multigrid variants agree to 1e-12 relative.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is invalid.
/// RuntimeError
///     If ``platform="CUDA"`` and CUDA is not available
///     (:func:`cuda_info` gives the reason).
///
/// Examples
/// --------
/// >>> solver = pbsolve.Solver(ionic_strength=0.15)
/// >>> result = solver.polar_solvation(coords, charges, radii)
/// >>> result.energy  # kcal/mol
#[pyclass(name = "Solver", module = "pbsolve")]
struct PySolver {
    engine: Engine,
    /// Extra backends for concurrent batch workers (created on demand).
    pool: std::sync::Mutex<Vec<Engine>>,
    device: usize,
    params: PbParams,
    spacing: f64,
    cfac: f64,
    fadd: f64,
    dime_policy: DimePolicy,
}

/// Default number of concurrent batch workers: enough to overlap host work
/// with solver work (measured in docs/VALIDATION.md, *Batched solves*). On
/// CUDA, fewer run where they do not fit in the free device memory.
fn default_workers(platform: &str) -> usize {
    match platform {
        "CUDA" => 3,
        _ => 2,
    }
}

fn job_error(e: PbError, index: usize) -> PbError {
    match e {
        PbError::InvalidInput(m) => PbError::InvalidInput(format!("system {index}: {m}")),
        PbError::Backend(m) => PbError::Backend(format!("system {index}: {m}")),
        other => other,
    }
}

impl PySolver {
    /// Runs the calculations on up to `workers` concurrent backends; results
    /// in input order. Fails with the first failing system's error.
    fn run_batch(
        &self,
        systems: &[Vec<Atom>],
        specs: &[GridSpec],
        workers: usize,
    ) -> Result<Vec<PolarSolvation>, PbError> {
        let workers = self.fitting_workers(specs, workers)?.min(systems.len().max(1));
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        while pool.len() + 1 < workers {
            let extra = Engine::create(self.engine.platform(), self.device)
                .map_err(|e| PbError::Backend(format!("cannot create batch worker {}: {e}", pool.len() + 1)))?;
            pool.push(extra);
        }
        let jobs: Vec<Job> = systems
            .iter()
            .zip(specs)
            .map(|(atoms, spec)| Job { atoms, spec })
            .collect();
        let engines: Vec<&Engine> = std::iter::once(&self.engine).chain(pool.iter()).take(workers).collect();
        let results = match &self.engine {
            Engine::Cpu(_) => {
                let bes: Vec<&CpuBackend> = engines
                    .iter()
                    .map(|e| match e {
                        Engine::Cpu(b) => b,
                        #[allow(unreachable_patterns)]
                        _ => unreachable!("batch workers share the main platform"),
                    })
                    .collect();
                core_batch(&bes, &self.params, &jobs)
            }
            #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
            Engine::Cuda(_) => {
                let bes: Vec<&pbsolve_cuda::CudaBackend> = engines
                    .iter()
                    .map(|e| match e {
                        Engine::Cuda(b) => &**b,
                        _ => unreachable!("batch workers share the main platform"),
                    })
                    .collect();
                core_batch(&bes, &self.params, &jobs)
            }
        };
        results
            .into_iter()
            .enumerate()
            .map(|(i, r)| r.map_err(|e| job_error(e, i)))
            .collect()
    }

    /// `workers` (at least 1), on CUDA limited to the calculations on grids
    /// `specs` that fit in the free device memory at once.
    fn fitting_workers(&self, specs: &[GridSpec], workers: usize) -> Result<usize, PbError> {
        let workers = workers.max(1);
        let Some((free, total)) = self.engine.device_memory()? else {
            return Ok(workers);
        };
        let per_worker = worker_device_bytes(specs.iter().map(|s| s.dims), self.params.preconditioner);
        Ok(device_workers(workers, per_worker, free, total))
    }

    /// Grid of every system: one shared `GridSpec`, one per system, or automatic.
    fn specs_for(&self, systems: &[Vec<Atom>], grid: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<GridSpec>> {
        match grid {
            None => systems.iter().map(|a| self.auto_grid(a).map_err(to_py)).collect(),
            Some(g) => {
                if let Ok(one) = g.extract::<PyGridSpec>() {
                    return Ok(vec![one.inner; systems.len()]);
                }
                let many: Vec<PyGridSpec> = g.extract().map_err(|_| {
                    PyValueError::new_err("grid must be a GridSpec, a sequence of GridSpec (one per system) or None")
                })?;
                if many.len() != systems.len() {
                    return Err(PyValueError::new_err(format!(
                        "{} grids given for {} systems",
                        many.len(),
                        systems.len()
                    )));
                }
                Ok(many.into_iter().map(|g| g.inner).collect())
            }
        }
    }

    fn workers_for(&self, workers: Option<usize>) -> PyResult<usize> {
        match workers {
            Some(0) => Err(PyValueError::new_err("workers must be at least 1")),
            Some(w) => Ok(w),
            None => Ok(default_workers(self.engine.platform())),
        }
    }

    fn auto_grid(&self, atoms: &[Atom]) -> Result<GridSpec, PbError> {
        GridSpec::auto_with(atoms, self.spacing, self.cfac, self.fadd, self.dime_policy)
    }

    fn spec_for(&self, atoms: &[Atom], grid: Option<PyGridSpec>) -> PyResult<GridSpec> {
        match grid {
            Some(g) => Ok(g.inner),
            None => self.auto_grid(atoms).map_err(to_py),
        }
    }
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (
        platform = "auto", device = 0, grid_spacing = 0.5,
        pdie = 1.0, sdie = 78.54, temperature = 298.15,
        ionic_strength = 0.0, ion_radius = 2.0, ions = None,
        srad = 1.4, sdens = 10.0, surface = "mol", bcfl = "mdh",
        cfac = 1.7, fadd = 20.0, tol = 1e-8, max_iter = 20000, dime_policy = "pbsolve",
        preconditioner = "multigrid-f32"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        platform: &str,
        device: usize,
        grid_spacing: f64,
        pdie: f64,
        sdie: f64,
        temperature: f64,
        ionic_strength: f64,
        ion_radius: f64,
        ions: Option<Vec<(f64, f64, f64)>>,
        srad: f64,
        sdens: f64,
        surface: &str,
        bcfl: &str,
        cfac: f64,
        fadd: f64,
        tol: f64,
        max_iter: usize,
        dime_policy: &str,
        preconditioner: &str,
    ) -> PyResult<Self> {
        let mut params = PbParams {
            pdie,
            sdie,
            temperature,
            srad,
            sdens,
            surface: SurfaceMethod::parse(surface).map_err(to_py)?,
            bcfl: BoundaryCondition::parse(bcfl).map_err(to_py)?,
            tol,
            max_iter,
            preconditioner: Preconditioner::parse(preconditioner).map_err(to_py)?,
            ..PbParams::default()
        };
        params = match ions {
            Some(list) => {
                params.ions = list
                    .into_iter()
                    .map(|(charge, concentration, radius)| IonSpecies {
                        charge,
                        concentration,
                        radius,
                    })
                    .collect();
                params
            }
            None => params.with_salt(ionic_strength, ion_radius),
        };
        params.validate().map_err(to_py)?;
        if grid_spacing.is_nan() || grid_spacing <= 0.0 {
            return Err(PyValueError::new_err("grid_spacing must be positive"));
        }
        Ok(Self {
            engine: Engine::create(platform, device)?,
            pool: std::sync::Mutex::new(Vec::new()),
            device,
            params,
            spacing: grid_spacing,
            cfac,
            fadd,
            dime_policy: DimePolicy::parse(dime_policy).map_err(to_py)?,
        })
    }

    /// All solver settings as a dict (APBS keyword names where they exist).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.params;
        let d = PyDict::new(py);
        d.set_item("pdie", p.pdie)?;
        d.set_item("sdie", p.sdie)?;
        d.set_item("temperature", p.temperature)?;
        let ions: Vec<(f64, f64, f64)> = p.ions.iter().map(|i| (i.charge, i.concentration, i.radius)).collect();
        d.set_item("ions", ions)?;
        d.set_item("ionic_strength", p.ionic_strength())?;
        d.set_item("srad", p.srad)?;
        d.set_item("sdens", p.sdens)?;
        d.set_item("surface", p.surface.as_str())?;
        d.set_item("bcfl", p.bcfl.as_str())?;
        d.set_item("tol", p.tol)?;
        d.set_item("max_iter", p.max_iter)?;
        d.set_item("preconditioner", p.preconditioner.as_str())?;
        d.set_item("grid_spacing", self.spacing)?;
        d.set_item("cfac", self.cfac)?;
        d.set_item("fadd", self.fadd)?;
        d.set_item("dime_policy", self.dime_policy.as_str())?;
        Ok(d)
    }

    /// Platform the solver runs on, "CPU" or "CUDA".
    #[getter]
    fn platform(&self) -> &'static str {
        self.engine.platform()
    }

    /// Name of the device the solver runs on.
    #[getter]
    fn device_name(&self) -> String {
        self.engine.device_name()
    }

    /// Ionic strength ½Σcz² of the mobile ions in mol/L.
    #[getter]
    fn ionic_strength(&self) -> f64 {
        self.params.ionic_strength()
    }

    /// Automatic focusing grid for the given atoms, as APBS ``psize`` sizes it.
    ///
    /// The fine box is the extent of the atoms (radii included) plus
    /// ``fadd``, the coarse box ``cfac`` times the extent, and the node
    /// counts give a fine spacing of at most ``grid_spacing``. In MM-PBSA,
    /// build it once from the complex and reuse it for receptor and ligand.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (N, 3)
    ///     Atom positions in Å.
    /// radii : array_like, shape (N,)
    ///     Atomic radii in Å.
    ///
    /// Returns
    /// -------
    /// GridSpec
    fn make_grid(&self, coords: PyReadonlyArray2<'_, f64>, radii: PyReadonlyArray1<'_, f64>) -> PyResult<PyGridSpec> {
        let xyz = coords_from(coords.as_array())?;
        let r = radii.as_array().to_vec();
        let q = vec![0.0; r.len()];
        let atoms = atoms_from_arrays(&xyz, &q, &r).map_err(to_py)?;
        self.auto_grid(&atoms).map(|inner| PyGridSpec { inner }).map_err(to_py)
    }

    /// Polar solvation free energy ΔG_polar.
    ///
    /// Solves the solvated state (``sdie``, ions) and the reference state
    /// (``pdie`` everywhere, no ions) on the same grid; ΔG_polar is
    /// ½Σq(u_solvated − u_reference).
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (N, 3)
    ///     Atom positions in Å.
    /// charges : array_like, shape (N,)
    ///     Atomic charges in e.
    /// radii : array_like, shape (N,)
    ///     Atomic radii in Å.
    /// grid : GridSpec, optional
    ///     Grid of the calculation; by default :meth:`make_grid` of the atoms.
    ///
    /// Returns
    /// -------
    /// PolarResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the arrays do not match or a charge lies outside the grid.
    /// ConvergenceError
    ///     If a linear solve does not converge.
    #[pyo3(signature = (coords, charges, radii, grid = None))]
    fn polar_solvation(
        &self,
        py: Python<'_>,
        coords: PyReadonlyArray2<'_, f64>,
        charges: PyReadonlyArray1<'_, f64>,
        radii: PyReadonlyArray1<'_, f64>,
        grid: Option<PyGridSpec>,
    ) -> PyResult<PyPolarResult> {
        let atoms = atoms_from(&coords, &charges, &radii)?;
        let spec = self.spec_for(&atoms, grid)?;
        let res = py
            .detach(|| self.engine.polar(&atoms, &self.params, &spec))
            .map_err(to_py)?;
        Ok(PyPolarResult::from_core(res, self.engine.platform()))
    }

    /// ΔG_polar in kcal/mol for every frame of a trajectory.
    ///
    /// The energies do not depend on ``workers`` and equal those of
    /// :meth:`polar_solvation` bit for bit.
    ///
    /// Parameters
    /// ----------
    /// frames : array_like, shape (F, N, 3)
    ///     Atom positions of every frame in Å.
    /// charges : array_like, shape (N,)
    ///     Atomic charges in e.
    /// radii : array_like, shape (N,)
    ///     Atomic radii in Å.
    /// grid : GridSpec or sequence of GridSpec, optional
    ///     One grid for all frames or one per frame; by default an automatic
    ///     grid per frame.
    /// workers : int, optional
    ///     Frames solved concurrently. Default: 2 on the CPU, 3 on CUDA; on
    ///     CUDA no more than fit in the free GPU memory (see
    ///     :meth:`batch_workers`).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (F,)
    #[pyo3(signature = (frames, charges, radii, grid = None, workers = None))]
    fn polar_solvation_batch<'py>(
        &self,
        py: Python<'py>,
        frames: PyReadonlyArray3<'py, f64>,
        charges: PyReadonlyArray1<'py, f64>,
        radii: PyReadonlyArray1<'py, f64>,
        grid: Option<Bound<'py, PyAny>>,
        workers: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = frames.as_array();
        if f.shape()[2] != 3 {
            return Err(PyValueError::new_err("frames must have shape (F, N, 3)"));
        }
        let q = charges.as_array().to_vec();
        let r = radii.as_array().to_vec();
        let mut systems = Vec::with_capacity(f.shape()[0]);
        for frame in f.outer_iter() {
            let xyz = coords_from(frame)?;
            systems.push(atoms_from_arrays(&xyz, &q, &r).map_err(to_py)?);
        }
        let specs = self.specs_for(&systems, grid.as_ref())?;
        let workers = self.workers_for(workers)?;
        let out = py.detach(|| self.run_batch(&systems, &specs, workers)).map_err(to_py)?;
        Ok(PyArray1::from_vec(py, out.iter().map(|r| r.energy_kcal).collect()))
    }

    /// ΔG_polar for several systems, several of them solved at once.
    ///
    /// The results are in input order and equal, bit for bit, those of
    /// :meth:`polar_solvation` called on each system in turn.
    ///
    /// Parameters
    /// ----------
    /// systems : sequence of tuple
    ///     The systems, each a tuple ``(coords, charges, radii)`` as the
    ///     arguments of :meth:`polar_solvation`.
    /// grid : GridSpec or sequence of GridSpec, optional
    ///     One grid for all systems or one per system; by default an
    ///     automatic grid per system.
    /// workers : int, optional
    ///     Systems solved concurrently. Default: 2 on the CPU, 3 on CUDA; on
    ///     CUDA no more than fit in the free GPU memory (see
    ///     :meth:`batch_workers`).
    ///
    /// Returns
    /// -------
    /// list of PolarResult
    #[pyo3(signature = (systems, grid = None, workers = None))]
    #[allow(clippy::type_complexity)]
    fn polar_solvation_many<'py>(
        &self,
        py: Python<'py>,
        systems: Vec<(
            PyReadonlyArray2<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
        )>,
        grid: Option<Bound<'py, PyAny>>,
        workers: Option<usize>,
    ) -> PyResult<Vec<PyPolarResult>> {
        let atoms: Vec<Vec<Atom>> = systems
            .iter()
            .enumerate()
            .map(|(i, (c, q, r))| atoms_from(c, q, r).map_err(|e| PyValueError::new_err(format!("system {i}: {e}"))))
            .collect::<PyResult<_>>()?;
        let specs = self.specs_for(&atoms, grid.as_ref())?;
        let workers = self.workers_for(workers)?;
        let out = py.detach(|| self.run_batch(&atoms, &specs, workers)).map_err(to_py)?;
        let platform = self.engine.platform();
        Ok(out.into_iter().map(|r| PyPolarResult::from_core(r, platform)).collect())
    }

    /// Number of calculations the batch methods run at once on ``grid``.
    ///
    /// This is the number :meth:`polar_solvation_many` and
    /// :meth:`polar_solvation_batch` use for a batch of at least that many
    /// systems: ``workers`` (default 2 on the CPU, 3 on CUDA), on CUDA
    /// limited to the calculations that fit in the GPU memory free now while
    /// 1/20 of the device stays free. At least 1.
    ///
    /// Parameters
    /// ----------
    /// grid : GridSpec or sequence of GridSpec
    ///     The grid of every calculation, or one per system.
    /// workers : int, optional
    ///     Upper limit; the default as above.
    ///
    /// Returns
    /// -------
    /// int
    #[pyo3(signature = (grid, workers = None))]
    fn batch_workers(&self, grid: Bound<'_, PyAny>, workers: Option<usize>) -> PyResult<usize> {
        let specs: Vec<GridSpec> = if let Ok(one) = grid.extract::<PyGridSpec>() {
            vec![one.inner]
        } else {
            let many: Vec<PyGridSpec> = grid
                .extract()
                .map_err(|_| PyValueError::new_err("grid must be a GridSpec or a sequence of GridSpec"))?;
            many.into_iter().map(|g| g.inner).collect()
        };
        let workers = self.workers_for(workers)?;
        self.fitting_workers(&specs, workers).map_err(to_py)
    }

    /// Electrostatic potential on the finest grid.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (N, 3)
    ///     Atom positions in Å.
    /// charges : array_like, shape (N,)
    ///     Atomic charges in e.
    /// radii : array_like, shape (N,)
    ///     Atomic radii in Å.
    /// grid : GridSpec, optional
    ///     Grid of the calculation; by default :meth:`make_grid` of the atoms.
    /// reference : bool
    ///     Solve the reference state (``pdie`` everywhere, no ions) instead
    ///     of the solvated state.
    ///
    /// Returns
    /// -------
    /// potential : numpy.ndarray, shape (nx, ny, nz)
    ///     Potential in kT/e, C order (x is the first index).
    /// origin : list of 3 float
    ///     Position of node (0, 0, 0) in Å.
    /// spacing : list of 3 float
    ///     Node spacing in Å.
    /// energy_kt : float
    ///     Electrostatic energy ½Σqu of that state in kT.
    #[pyo3(signature = (coords, charges, radii, grid = None, reference = false))]
    #[allow(clippy::type_complexity)]
    fn potential<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<'py, f64>,
        charges: PyReadonlyArray1<'py, f64>,
        radii: PyReadonlyArray1<'py, f64>,
        grid: Option<PyGridSpec>,
        reference: bool,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, [f64; 3], [f64; 3], f64)> {
        let atoms = atoms_from(&coords, &charges, &radii)?;
        let spec = self.spec_for(&atoms, grid)?;
        let medium = if reference { Medium::Reference } else { Medium::Solvated };
        let (sol, energy) = py
            .detach(|| -> Result<_, PbError> {
                let sol = self.engine.field(&atoms, &self.params, &spec, medium)?;
                let e = electrostatic_energy_kt(&sol, &atoms)?;
                Ok((sol, e))
            })
            .map_err(to_py)?;
        let arr = to_numpy(py, sol.grid.dims, &sol.potential)?;
        Ok((arr, sol.grid.origin, sol.grid.spacing, energy))
    }

    /// Coefficient maps of one focusing level.
    ///
    /// The maps compare one to one with APBS ``write dielx``, ``diely``,
    /// ``dielz``, ``kappa`` and ``charge`` (APBS writes kappa as 0/1
    /// accessibility and the charge as a density in e/Å³).
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (N, 3)
    ///     Atom positions in Å.
    /// charges : array_like, shape (N,)
    ///     Atomic charges in e.
    /// radii : array_like, shape (N,)
    ///     Atomic radii in Å.
    /// grid : GridSpec, optional
    ///     Grid of the calculation; by default :meth:`make_grid` of the atoms.
    /// reference : bool
    ///     Maps of the reference state instead of the solvated state.
    /// level : int
    ///     Focusing level, counted from the coarsest (0); negative values
    ///     count from the finest (-1).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``eps_x``, ``eps_y``, ``eps_z`` (ε on the faces i+½, j+½, k+½),
    ///     ``kappa_sq`` (κ̄² in Å⁻², 0 where ions are excluded) and
    ///     ``charge`` (e per node, trilinear spreading) as C-order
    ///     (nx, ny, nz) arrays; the level's ``origin``, ``spacing``,
    ///     ``dims``, ``lengths`` and ``center``; and the bulk
    ///     ``kappa_bar_sq``.
    #[pyo3(signature = (coords, charges, radii, grid = None, reference = false, level = -1))]
    #[allow(clippy::too_many_arguments)]
    fn maps<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<'py, f64>,
        charges: PyReadonlyArray1<'py, f64>,
        radii: PyReadonlyArray1<'py, f64>,
        grid: Option<PyGridSpec>,
        reference: bool,
        level: isize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let atoms = atoms_from(&coords, &charges, &radii)?;
        let spec = self.spec_for(&atoms, grid)?;
        let mut levels = spec.levels().map_err(to_py)?;
        let n = levels.len() as isize;
        let idx = if level < 0 { n + level } else { level };
        if idx < 0 || idx >= n {
            return Err(PyValueError::new_err(format!(
                "level {level} out of range for {n} focusing levels"
            )));
        }
        let g = levels.swap_remove(idx as usize);
        let medium = if reference { Medium::Reference } else { Medium::Solvated };
        let (maps, q) = py
            .detach(|| -> Result<_, PbError> {
                let q = spread_charges(&g, &atoms)?;
                Ok((coefficient_maps(&g, &atoms, &self.params, medium), q))
            })
            .map_err(to_py)?;
        let d = grid_dict(py, &g)?;
        d.set_item("eps_x", to_numpy(py, g.dims, &maps.eps_x)?)?;
        d.set_item("eps_y", to_numpy(py, g.dims, &maps.eps_y)?)?;
        d.set_item("eps_z", to_numpy(py, g.dims, &maps.eps_z)?)?;
        d.set_item("kappa_sq", to_numpy(py, g.dims, &maps.kappa_sq)?)?;
        d.set_item("charge", to_numpy(py, g.dims, &q)?)?;
        let kbar2 = if reference {
            0.0
        } else {
            kappa_bar_squared(&self.params.ions, self.params.temperature)
        };
        d.set_item("kappa_bar_sq", kbar2)?;
        Ok(d)
    }

    /// Solve the LPBE with given coefficient maps.
    ///
    /// Feeding APBS' own maps here tests pbsolve's discretisation and linear
    /// solver in isolation ("map injection"). Temperature, tolerance,
    /// preconditioner and platform come from this solver. The arrays are
    /// C-order (nx, ny, nz), as :meth:`maps` returns them.
    ///
    /// Parameters
    /// ----------
    /// eps_x, eps_y, eps_z : numpy.ndarray
    ///     Dielectric constant on the faces i+½, j+½, k+½.
    /// kappa_sq : numpy.ndarray
    ///     κ̄² in Å⁻².
    /// charge : numpy.ndarray
    ///     Charge per node in e.
    /// boundary : numpy.ndarray
    ///     Potential in kT/e; only the boundary nodes are read.
    /// origin : sequence of 3 float
    ///     Position of node (0, 0, 0) in Å.
    /// spacing : sequence of 3 float
    ///     Node spacing in Å.
    ///
    /// Returns
    /// -------
    /// potential : numpy.ndarray, shape (nx, ny, nz)
    ///     Potential in kT/e.
    /// iterations : int
    ///     CG iterations.
    /// residual : float
    ///     Final relative residual.
    #[pyo3(signature = (eps_x, eps_y, eps_z, kappa_sq, charge, boundary, origin, spacing))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve_maps<'py>(
        &self,
        py: Python<'py>,
        eps_x: PyReadonlyArray3<'py, f64>,
        eps_y: PyReadonlyArray3<'py, f64>,
        eps_z: PyReadonlyArray3<'py, f64>,
        kappa_sq: PyReadonlyArray3<'py, f64>,
        charge: PyReadonlyArray3<'py, f64>,
        boundary: PyReadonlyArray3<'py, f64>,
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, usize, f64)> {
        let (ex, dims) = from_numpy("eps_x", &eps_x, None)?;
        let (ey, _) = from_numpy("eps_y", &eps_y, Some(dims))?;
        let (ez, _) = from_numpy("eps_z", &eps_z, Some(dims))?;
        let (k2, _) = from_numpy("kappa_sq", &kappa_sq, Some(dims))?;
        let (q, _) = from_numpy("charge", &charge, Some(dims))?;
        let (g, _) = from_numpy("boundary", &boundary, Some(dims))?;
        let center = [0, 1, 2].map(|a| origin[a] + 0.5 * (dims[a] - 1) as f64 * spacing[a]);
        let mut grid = Grid::from_spacing(center, spacing, dims).map_err(to_py)?;
        grid.origin = origin;
        let maps = CoefficientMaps {
            eps_x: ex,
            eps_y: ey,
            eps_z: ez,
            kappa_sq: k2,
        };
        let opts = SolverOptions {
            tol: self.params.tol,
            max_iter: self.params.max_iter,
            preconditioner: self.params.preconditioner,
        };
        let temperature = self.params.temperature;
        let (u, stats) = py
            .detach(|| self.engine.with_maps(&grid, maps, &q, &g, temperature, opts))
            .map_err(to_py)?;
        Ok((to_numpy(py, dims, &u)?, stats.iterations, stats.relative_residual))
    }

    fn __repr__(&self) -> String {
        format!(
            "Solver(platform='{}', grid_spacing={}, pdie={}, sdie={}, ionic_strength={}, surface='{}', bcfl='{}', dime_policy='{}')",
            self.engine.platform(),
            self.spacing,
            self.params.pdie,
            self.params.sdie,
            self.params.ionic_strength(),
            self.params.surface.as_str(),
            self.params.bcfl.as_str(),
            self.dime_policy.as_str()
        )
    }
}

// ---------------------------------------------------------------- functions

/// Solvent-accessible surface area of each atom (Shrake–Rupley).
///
/// Parameters
/// ----------
/// coords : array_like, shape (N, 3)
///     Atom positions in Å.
/// radii : array_like, shape (N,)
///     Atomic radii in Å.
/// probe : float
///     Solvent probe radius in Å.
/// n_points : int
///     Test points per atom sphere.
///
/// Returns
/// -------
/// numpy.ndarray, shape (N,)
///     Area of each atom in Å².
#[pyfunction]
#[pyo3(signature = (coords, radii, probe = 1.4, n_points = 960))]
fn sasa<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    probe: f64,
    n_points: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xyz = coords_from(coords.as_array())?;
    let r = radii.as_array().to_vec();
    let q = vec![0.0; r.len()];
    let atoms = atoms_from_arrays(&xyz, &q, &r).map_err(to_py)?;
    let out = py.detach(|| pbsolve_core::sasa::sasa(&atoms, probe, n_points));
    Ok(PyArray1::from_vec(py, out))
}

/// k_B·T in kJ/mol (the unit APBS reports energies in).
#[pyfunction]
#[pyo3(signature = (temperature = 298.15))]
fn kt_kj_per_mol(temperature: f64) -> f64 {
    kt_in_kj_per_mol(temperature)
}

/// k_B·T in kcal/mol.
#[pyfunction]
#[pyo3(signature = (temperature = 298.15))]
fn kt_kcal_per_mol(temperature: f64) -> f64 {
    kt_in_kcal_per_mol(temperature)
}

/// Vacuum Bjerrum length e²/(4πε₀k_BT) in Å.
#[pyfunction]
#[pyo3(signature = (temperature = 298.15))]
fn bjerrum_length(temperature: f64) -> f64 {
    bjerrum_length_vacuum(temperature)
}

/// The ``dime`` APBS actually uses for a requested value.
///
/// APBS resets a ``dime`` above 65 that is not ``32c+1`` to a smaller value
/// (121 becomes 97). ``None`` if APBS rejects the value.
#[pyfunction]
fn apbs_effective_dime(dime: usize) -> Option<usize> {
    core_apbs_effective_dime(dime)
}

// ---------------------------------------------------------------- file I/O

fn io_err(e: pbsolve_io::IoError) -> PyErr {
    match e {
        pbsolve_io::IoError::Io(err) => PyOSError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn io_format(format: Option<&str>) -> PyResult<Option<pbsolve_io::Format>> {
    format.map(pbsolve_io::Format::parse).transpose().map_err(io_err)
}

fn opt_array(py: Python<'_>, v: impl Iterator<Item = Option<f64>>) -> Bound<'_, PyArray1<f64>> {
    PyArray1::from_vec(py, v.map(|x| x.unwrap_or(f64::NAN)).collect())
}

/// Read a structure, topology or coordinate file.
///
/// :func:`pbsolve.io.load` builds a molecule from this; use it unless you
/// need the raw arrays.
///
/// Parameters
/// ----------
/// path : str or path-like
///     The file.
/// format : str, optional
///     File format ("pdb", "pqr", "cif", "gro", "mol2", "prmtop", "psf",
///     "gmxtop", "inpcrd", "netcdf", "mdcrd", "dcd", "xtc", "trr");
///     detected from the extension by default.
/// include_dirs : sequence of str, optional
///     Extra directories searched for GROMACS ``#include`` files.
///
/// Returns
/// -------
/// dict
///     Per-atom lists and arrays (``names``, ``resnames``, ``resids``,
///     ``icodes``, ``chains``, ``segids``, ``elements``, ``atom_types``,
///     ``hetero``, ``charges``, ``radii``, ``masses``; NaN where the format
///     has no value), ``bonds`` (B, 2), ``coords`` (F, N, 3) in Å,
///     ``cells`` (F, 6), ``times`` (F,), ``format``, ``periodic`` and
///     ``notes``.
///
/// Raises
/// ------
/// OSError
///     If the file cannot be read.
/// ValueError
///     If the file is malformed or the format unknown.
#[pyfunction]
#[pyo3(signature = (path, format = None, include_dirs = None))]
fn read_structure<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    format: Option<&str>,
    include_dirs: Option<Vec<std::path::PathBuf>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fmt = io_format(format)?;
    let opts = pbsolve_io::ReadOptions {
        include_dirs: include_dirs.unwrap_or_default(),
    };
    let s = py.detach(|| pbsolve_io::read_file(&path, fmt, &opts)).map_err(io_err)?;
    let d = PyDict::new(py);
    let a = &s.atoms;
    let strs = |f: fn(&pbsolve_io::AtomInfo) -> &String| a.iter().map(|x| f(x).clone()).collect::<Vec<String>>();
    d.set_item("format", &s.format)?;
    d.set_item("names", strs(|x| &x.name))?;
    d.set_item("resnames", strs(|x| &x.resname))?;
    d.set_item("icodes", strs(|x| &x.icode))?;
    d.set_item("chains", strs(|x| &x.chain))?;
    d.set_item("segids", strs(|x| &x.segid))?;
    d.set_item("elements", strs(|x| &x.element))?;
    d.set_item("atom_types", strs(|x| &x.atom_type))?;
    d.set_item("resids", PyArray1::from_vec(py, a.iter().map(|x| x.resid).collect()))?;
    d.set_item("hetero", PyArray1::from_vec(py, a.iter().map(|x| x.hetero).collect()))?;
    d.set_item("charges", opt_array(py, a.iter().map(|x| x.charge)))?;
    d.set_item("radii", opt_array(py, a.iter().map(|x| x.radius)))?;
    d.set_item("masses", opt_array(py, a.iter().map(|x| x.mass)))?;
    let nb = s.bonds.len();
    let flat: Vec<i64> = s.bonds.iter().flat_map(|&(i, j)| [i as i64, j as i64]).collect();
    d.set_item("bonds", PyArray1::from_vec(py, flat).reshape([nb, 2])?)?;
    let n = a.len();
    let nf = s.frames.len();
    let xyz: Vec<f64> = s
        .frames
        .iter()
        .flat_map(|f| f.coords.iter().flatten().copied())
        .collect();
    d.set_item("coords", PyArray1::from_vec(py, xyz).reshape([nf, n, 3])?)?;
    let cells: Vec<f64> = s.frames.iter().flat_map(|f| f.cell.unwrap_or([f64::NAN; 6])).collect();
    d.set_item("cells", PyArray1::from_vec(py, cells).reshape([nf, 6])?)?;
    d.set_item("times", opt_array(py, s.frames.iter().map(|f| f.time)))?;
    d.set_item("periodic", s.periodic)?;
    d.set_item("notes", s.notes)?;
    Ok(d)
}

/// Streams the frames of a trajectory.
///
/// Reads DCD, XTC, TRR, Amber NetCDF and mdcrd, and multi-model PDB, GRO
/// and mmCIF files. Iterating yields ``(coords, cell, time, step)``:
/// positions (N, 3) in Å, the unit cell (a, b, c in Å, α, β, γ in degrees)
/// or ``None``, the time or ``None`` and the step or ``None``.
///
/// Parameters
/// ----------
/// path : str or path-like
///     The trajectory file.
/// format : str, optional
///     File format; detected from the extension by default.
/// n_atoms : int, optional
///     Atoms per frame; required for ASCII mdcrd, checked otherwise.
/// mdcrd_box : bool, optional
///     Whether mdcrd frames carry a box line; detected by default.
///
/// Raises
/// ------
/// OSError
///     If the file cannot be opened.
/// ValueError
///     If the file is malformed or the format unknown.
#[pyclass(name = "TrajectoryReader", module = "pbsolve")]
struct PyTrajectoryReader {
    inner: std::sync::Mutex<Box<dyn pbsolve_io::FrameReader>>,
}

impl PyTrajectoryReader {
    fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn pbsolve_io::FrameReader>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl PyTrajectoryReader {
    #[new]
    #[pyo3(signature = (path, format = None, n_atoms = None, mdcrd_box = None))]
    fn new(
        path: std::path::PathBuf,
        format: Option<&str>,
        n_atoms: Option<usize>,
        mdcrd_box: Option<bool>,
    ) -> PyResult<Self> {
        let fmt = io_format(format)?;
        let inner = pbsolve_io::open_trajectory(&path, fmt, n_atoms, mdcrd_box).map_err(io_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    /// Atoms per frame, if known before reading a frame.
    #[getter]
    fn n_atoms(&self) -> Option<usize> {
        self.lock().n_atoms()
    }

    /// Number of frames if known without reading the file (DCD, NetCDF, multi-model files).
    #[getter]
    fn n_frames(&self) -> Option<usize> {
        self.lock().n_frames()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[allow(clippy::type_complexity)]
    fn __next__<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Option<[f64; 6]>, Option<f64>, Option<i64>)>> {
        let inner = &self.inner;
        let f = py
            .detach(|| inner.lock().unwrap_or_else(|e| e.into_inner()).next_frame())
            .map_err(io_err)?;
        let Some(f) = f else { return Ok(None) };
        let n = f.coords.len();
        let xyz: Vec<f64> = f.coords.into_iter().flatten().collect();
        Ok(Some((
            PyArray1::from_vec(py, xyz).reshape([n, 3])?,
            f.cell,
            f.time,
            f.step,
        )))
    }
}

/// Read a PQR file.
///
/// Returns
/// -------
/// coords : numpy.ndarray, shape (N, 3)
///     Atom positions in Å.
/// charges : numpy.ndarray, shape (N,)
///     Atomic charges in e.
/// radii : numpy.ndarray, shape (N,)
///     Atomic radii in Å.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn read_pqr<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let atoms = core_read_pqr(&path).map_err(to_py)?;
    let n = atoms.len();
    let xyz: Vec<f64> = atoms.iter().flat_map(|a| a.position).collect();
    let q: Vec<f64> = atoms.iter().map(|a| a.charge).collect();
    let r: Vec<f64> = atoms.iter().map(|a| a.radius).collect();
    let coords = PyArray1::from_vec(py, xyz).reshape([n, 3])?;
    Ok((coords, PyArray1::from_vec(py, q), PyArray1::from_vec(py, r)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // CUDA major version of this build (0 = CPU-only); read by the NVRTC preload.
    m.add("_CUDA_MAJOR", pbsolve_cuda::cuda_major())?;
    m.add("ConvergenceError", m.py().get_type::<ConvergenceError>())?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyGridSpec>()?;
    m.add_class::<PyPolarResult>()?;
    m.add_function(wrap_pyfunction!(available_platforms, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_info, m)?)?;
    m.add_function(wrap_pyfunction!(sasa, m)?)?;
    m.add_function(wrap_pyfunction!(read_pqr, m)?)?;
    m.add_function(wrap_pyfunction!(kt_kj_per_mol, m)?)?;
    m.add_function(wrap_pyfunction!(kt_kcal_per_mol, m)?)?;
    m.add_function(wrap_pyfunction!(apbs_effective_dime, m)?)?;
    m.add_function(wrap_pyfunction!(bjerrum_length, m)?)?;
    m.add_function(wrap_pyfunction!(read_structure, m)?)?;
    m.add_class::<PyTrajectoryReader>()?;
    Ok(())
}
