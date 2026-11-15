# Changelog

## Unreleased

### Added

- Geometric multigrid preconditioner for CG (new module `multigrid` in
  `pbsolve-core`; CPU and CUDA): per-axis vertex coarsening for any grid size,
  trilinear transfers, Galerkin coarse operators, Chebyshev smoothing, banded
  Cholesky on the coarsest level. It is the new default; CG iterations drop
  from 300–700 to ~10–20 and stay constant under grid refinement.
- `Solver(preconditioner="multigrid" | "jacobi")`, `--preconditioner` on the
  CLI, `PolarResult.preconditioners` and `PolarResult.levels`; the JSON output
  of `pbsolve solvation` reports both.
- `cargo run --release -p pbsolve-core --example multigrid`: Jacobi versus
  multigrid study (iterations, setup and solve time).
- `cuda_info()` reports `driver`, `nvrtc` and `nvrtc_library`, and
  `pbsolve test` prints them. New environment variables `PBSOLVE_NVRTC` (choose
  the NVRTC library) and `PBSOLVE_CUDA_CODE=ptx|cubin` (diagnostics).
  `CudaBackend::with_code`, `nvrtc_info()` and `driver_version()` in
  `pbsolve-cuda`.

- Concurrent calculations (module `batch` in `pbsolve-core`):
  - `Solver.polar_solvation_many(systems, grid=None, workers=None)` returns
    results in input order;
  - `workers` on `Solver.polar_solvation_batch`;
  - `pbsolve.pbsa_solvation_frames(...)` for several frames at once;
  - `pbsolve mmpbsa --workers`.

  On 1YCR, MM-PBSA frames/s rise by up to 1.4× on the CPU and 2.2× on the GPU,
  and results are identical, bit for bit, to sequential calls.
- `preconditioner="multigrid-f32"` (CLI `--preconditioner multigrid-f32`)
  runs the multigrid V-cycle in single precision inside the f64 CG, which
  then uses the flexible Polak–Ribière update. Energies agree with
  `multigrid` to ≤ 1e-12 relative, and iteration counts are equal.
- Single-precision twin in the `Backend` trait (`type Single`, `single()`,
  `demote`, `promote`) and a fused `dot2`; CPU and CUDA kernels are generic
  over `Real` (f64/f32). `CpuSingle`, `CudaSingle`.
  `cargo run -p pbsolve-cuda --example precision` measures device throughput.
- `bcfl="mdh-fast"` (`--bcfl mdh-fast`): the `mdh` boundary sum evaluated
  exactly at Chebyshev points of patches of the box faces and interpolated in
  between; error ≤ 1e-10 of the summed term magnitudes per patch, checked up
  to κ = 0.6 Å⁻¹. Energies agree with `mdh` to ≤ 1.1e-13. The boundary of
  5DK3 (20 179 atoms) takes 3.4 → 0.55 s, and a CUDA calculation 1.36× less.
  `mdh` stays the default and still follows APBS; APBS inputs written from
  `mdh-fast` use `mdh`.
- `examples/pbsolve_tutorial.ipynb`: a tutorial from installation to MM-PBSA
  and the comparison with APBS, stored with its outputs. A changelog at its
  top and a date under every section show what changed;
  `tests/test_tutorial.py` checks its structure and saved results and runs it
  when Jupyter is installed.
- Multigrid setup on any backend: `multigrid::Levels` (a hierarchy held by a
  backend, built by `Levels::build`), `MultigridPreconditioner::from_levels`,
  `Stencil27::from_coef`, and the `Backend` methods `galerkin`,
  `gershgorin_lmax`, `coarse_to_host`, `demote_operator` and
  `demote_coarse`. `Hierarchy::build` stays the host reference.
- Assembly on any backend: `surface::maps_on`, `solver::pcg_on`,
  `charge::spread_charges_sparse`, `StencilOperator::from_parts`,
  `StencilOperator::face_coefficients`, `multigrid::coarsest_dims`,
  `multigrid::hierarchy_fits` and `real::zeros_touched`, and the `Backend` methods
  `mark_maps`, `scatter`, `assemble_operator`, `source_rhs`,
  `operator_to_host`, `upload_owned` and `synchronize`. `mark_map`,
  `StencilOperator::new`, `spread_charges` and `assemble_rhs` stay the host
  references. `Levels::build` and `MultigridPreconditioner::from_levels` take
  the fine grid size and inverse diagonal instead of a host operator.
- Device memory planning: `solver::solve_device_bytes` (the backend memory a
  calculation holds at its peak, equal to the CUDA memory pool's peak to the
  byte), `batch::worker_device_bytes`, `batch::device_workers`,
  `batch::POOL_SLACK_BYTES` and `multigrid::coarse_dims` in `pbsolve-core`;
  `CudaBackend::memory_info`, `pool_peak`, `reset_pool_peak` and `PoolPeak`
  in `pbsolve-cuda`; `Solver.batch_workers(grid, workers=None)` in Python.
- Reading values from the backend: `Backend::gather` (the values at a list
  of nodes; the default downloads the vector) and `backend::check_gather`,
  `grid::NodeSet`, `Grid::interpolate_with`, `Grid::boundary_position`, and
  `boundary::{analytic_boundary_values, focus_boundary_values, focus_nodes}`
  (boundary values on the boundary nodes only, in `boundary_indices` order).
- Memory retention on the backend: `Backend::retain_memory` and
  `release_memory` (no-ops by default) and `backend::MemoryHold`;
  `CudaBackend::pool_usage` and `pool_release_threshold` in `pbsolve-cuda`.
- `Backend::runs_on_device` (false by default, true for CUDA): whether the
  kernels leave the host's threads idle, so that the pipeline can do host
  work in the background.
- `PolarResult.timings` gives the seconds spent per stage: maps, boundary,
  assembly, setup, solve and energy. `SolveStats` gains `setup_seconds` and
  `solve_seconds`; `PolarSolvation` and `FieldSolution` gain `timings`.
- Documentation site (not published yet): Sphinx with MyST and furo, built
  from the existing pages (`docs/*.md`, `CHANGELOG.md`, the README), the
  tutorial notebook with its stored outputs, and an API reference read from
  the sources and `_core.pyi` (nothing is executed). Configuration for Read
  the Docs (`.readthedocs.yaml`) and a GitHub Actions workflow that builds it
  with warnings as errors and can publish it to GitHub Pages.
- Docstrings for every public class, function, property and attribute,
  with NumPy-style Parameters, Returns and Raises sections where there are
  any (units, defaults, APBS keyword equivalents). The stubs (`_core.pyi`,
  read by the documentation site and editors) and `help()` on the compiled
  module show the same text; `tests/test_docstrings.py` keeps them
  identical and the stubs complete.
- User guide on the documentation site (`docs/guide`): physical model, grids
  and focusing, linear solver, CPU and GPU, trajectories and MM-PBSA, input
  files and structure preparation, command line, comparison with APBS. Its
  Python examples run in the test suite (`tests/test_guide.py`), which also
  checks the command-line page against the options of every command.
- Comparison with APBS 3.4.1 on three larger proteins: 1HPV (3 137 atoms,
  three settings), 1TII (10 862) and 5DK3 (20 179 atoms, 193×321×321 grid).
  Maps are identical, ΔG_polar agrees to ≤ 4e-6 in APBS units and ΔΔG_polar
  of binding to ≤ 0.042 kcal/mol. With `bcfl zero` everything agrees to
  ~1e-10, so the remainder is APBS' boundary constants (`docs/VALIDATION.md`).
  The test suite now also compares the 1YCR complex with APBS.
- `pbsolve.analytic.kirkwood_energy`: the exact polar solvation energy of
  point charges anywhere inside a dielectric sphere, with or without salt
  beyond a Stern layer [Kirkwood 1934], in pbsolve's constants and model.
  `scripts/validate_analytic.py` reports the error against it and the Born
  ion at several spacings, averaged over grid placements, and the observed
  convergence order: 1.09–1.18, from about 2.3–2.9 % at 0.5 Å to 0.8–1.0 %
  at 0.2 Å (`docs/VALIDATION.md`).
- `pbsolve.estimate_error` and `pbsolve solvation --placements N
  [--extrapolate SPACING]`: ΔG_polar (or, with `parts`, the association
  energy) averaged over random placements of the grid. Reports the
  placements' standard deviation and the standard error of the mean, and
  with two spacings extrapolates to zero spacing. From 0.35 and 0.25 Å the
  extrapolation is within 0.3 % on the analytic references. For 1YCR it
  shows a spacing error of ~3 % in ΔG_polar and ~6 kcal/mol in ΔΔG_polar at
  0.25 Å (`docs/VALIDATION.md`).
- `Solver(interface="fractional")`, `--interface`: faces cut by the `mol` or
  `vdw` surface get the harmonic average of `pdie` and `sdie` weighted by the
  exact lengths of their grid edge inside and outside the solute. Energies
  become close to second order: errors on the analytic references fall from
  2.3–4.7 % to 0.1–0.7 % at 0.5 Å, and for 1YCR the placement spread shrinks
  5–10 times and the ΔΔG_polar bias at 0.5 Å from +13 to +2.7 kcal/mol.
  Opt-in: the default `"sharp"` (as APBS) is bit-identical to before
  (`docs/VALIDATION.md`).
- `Solver(equation="npbe")`, `--equation`: the nonlinear Poisson–Boltzmann
  equation, as APBS `npbe`. An inexact Newton iteration from the linearized
  solution reuses the linear multigrid solver. The energy is APBS'
  free-energy functional, including the coarser focusing levels' outer terms.
  The solvated energies agree with APBS 3.4.1 to ≤ 2.3e-9 in APBS units (1YCR,
  −8/−20/−30 e test systems, 1:1 and 2:1 salts). `pbsolve apbs compare/run`
  now handle `npbe`; `solve_maps` and `potential` follow the equation. A
  calculation costs 1.3–1.4× a linear one on the CPU and 1.7–1.9× on CUDA.
  The Newton iteration runs on the device (new backend operations
  `boltzmann_slope` and `boltzmann_residual`), from a start damped where
  |u| > 1. The default `"lpbe"` is bit-identical to before
  (`docs/VALIDATION.md`).
- `Solver(equation="smpbe", ion_size=a)`, `--equation smpbe --ion-size`:
  the size-modified (lattice-gas) PB equation, with ions that each fill a
  cube of edge a, all of one size. APBS 3.4.1 disables its `smpbe`. It
  matches the exact planar double layer (orders 2.6–3.8), the a³ law of the
  entropic cost [Silalahi et al. 2010], and saturates counterions at one per
  cell where point ions would reach 84 M. It costs about as much as `npbe`,
  which stays bit-identical (`docs/VALIDATION.md`).
- `Solver(surface="spl2" | "spl4", swin=0.3)`, `--surface spl2|spl4 --swin`:
  APBS' spline surfaces (`srfm spl2`, `spl4`), smooth in the atom positions;
  the maps equal APBS' to the 7 digits of its DX files. `pbsolve.apbs`
  writes and reads them.
- `Solver.polar_forces(coords, charges, radii)` and `pbsolve.PolarForces`:
  ΔG_polar and the force on every atom (kcal/mol/Å) with its charge–field,
  dielectric-boundary and ionic-boundary parts, as APBS `calcforce comps`
  (spline surfaces; `lpbe`, `npbe` and `smpbe`). On one grid with
  `bcfl="zero"` they are the exact gradient of the energy (to 4e-9); they
  agree with APBS 3.4.1 to ≤ 4e-7 of the largest force and with Kirkwood's
  force on a charge in a sphere to 0.08 %. `pbsolve apbs compare --forces`
  compares them with APBS (`docs/VALIDATION.md`).
- `Solver(surface="gauss", gauss_sigma=0.93, gauss_contour=20.0)`,
  `--surface gauss --gauss-sigma --gauss-contour`: the Gaussian dielectric
  (Li et al. 2013; not an APBS option). ε is graded from `pdie` to `sdie` by
  a molecular density of atomic Gaussians, and the ions pay a Born penalty
  (Jia et al. 2017). The reference state keeps the Gaussian ε up to the
  contour (Li et al. 2014); a homogeneous reference would leave the charges'
  grid self-energies uncancelled. One Gaussian atom agrees with its exact
  radial solution to 1–3 % at 0.5 Å; ΔΔG_polar of 1YCR changes by less than
  3 kcal/mol between 0.5 and 0.25 Å. Not with `npbe`, `coulomb="analytic"`
  or forces (`docs/VALIDATION.md`).
- `pbsolve.apolar_energy(coords, radii, epsilons)` and `pbsolve.ApolarEnergy`:
  the non-polar solvation energy of APBS `apolar` (Wagoner & Baker 2006),
  γ·SASA + p·SAV + the WCA dispersion integral, with every part per atom.
  It equals APBS' alkane examples to every printed digit and APBS 3.4.1 on a
  1670-atom protein to 1e-13, 17 times faster. `pbsolve.apbs.read_parm`
  reads APBS parameter files, `pbsolve.apbs.apolar_energy` runs APBS on the
  same input, and `pbsolve.openmm_bridge.lennard_jones_from_system` takes the
  Lennard-Jones radii and well depths from an OpenMM system.
- `batch::calculation_device_bytes`: the device memory of a calculation
  with its settings (the nonlinear equation needs 4 more vectors); used to
  reserve memory and to size batch workers.
- `Solver(coulomb="analytic")`, `--coulomb`: reaction-field decomposition.
  The analytic Coulomb potential of the charges is subtracted near the
  solute, and ΔG_polar comes from one solve per focusing level, without a
  reference state or grid self-energy. With the fractional interface the
  placement spread halves, and 1YCR's ΔG_polar error at 0.5 Å falls from
  10 to 6 kcal/mol, with the same zero-spacing limit. The CPU needs 0.6
  times the time. On CUDA it is 7–20 % faster for 1YCR and slower for very
  large proteins (direct Coulomb sum). Opt-in: the default `"grid"` (as
  APBS) is bit-identical to before (`docs/VALIDATION.md`).
- `Solver(chgm="spl2" | "spl4")`, `--chgm`: B-spline charge assignment as
  APBS `chgm spl2` and `spl4` (orders 3 and 5, 3 and 5 nodes per axis), with
  APBS' energy rule. The default `spl0` (trilinear) is unchanged. pbsolve's
  maps and energies agree with APBS 3.4.1 as for `spl0`, and
  `pbsolve apbs run` now accepts inputs that set them. They hardly change
  ΔG_polar or its dependence on the grid placement (`docs/VALIDATION.md`).
- `pbsolve.analytic.two_sphere_energy`: the same for charges in two spheres,
  from a converged multipole solution. The association energy of two spheres
  computed on one grid, as in MM-PBSA, agrees with it to 0.002 kcal/mol with
  `surface="vdw"`. With the default `mol` surface, APBS' sampled probe
  positions add 0.011–0.019 kcal/mol.

### Changed

- The default preconditioner is now `multigrid-f32` (Rust
  `Preconditioner::DEFAULT`, `PbParams`, `SolverOptions`, the Python `Solver`
  and the CLI). Default energies change by ≤ 4.2e-13 relative with identical
  iteration counts (Born ion and four proteins up to 20 179 atoms, 14
  parameter variants, CPU and CUDA; `docs/VALIDATION.md`), and full protein
  calculations are 1.4–1.7× faster on the CPU. `preconditioner="multigrid"`
  reproduces the previous default bit for bit.
- `pbsolve apbs compare` and `pbsolve apbs run` accept `--preconditioner`
  (`run_input(..., preconditioner=...)`).

- Faster host stages with bit-identical results:
  - the molecular surface is built once per focusing sequence;
  - node maps are filled in parallel;
  - `mdh` hoists atom factors and interleaves four boundary nodes;
  - the CPU backend's host transfers run in parallel.

  Gains: 1.06–1.26× per calculation on the CPU and 1.03–1.19× on CUDA.
  Energies, iterations, maps and potentials are identical bit for bit.
- The CUDA backend builds the multigrid hierarchy on the GPU: Galerkin
  operators, inverse diagonals, Gershgorin bounds and the f32 copies used by
  `multigrid-f32` are computed in device memory in the host's operation
  order. Results are identical bit for bit. On a shared RTX 2080 SUPER the
  setup stage is 2.2–3.1× faster and a calculation 1.2–1.3× faster (5DK3:
  13.5 → 10.5 s). The CPU path is unchanged.
- The CUDA backend also builds the dielectric and ion maps, the operator, the
  charges on the grid and the right-hand side in device memory; the host
  operator is no longer built or uploaded. Results are identical bit for bit.
  A calculation is 1.23–1.32× faster on a shared RTX 2080 SUPER (5DK3:
  10.3 → 8.4 s). The CPU runs the same pipeline through the CPU backend and
  is unchanged in speed.

- `polar_solvation` (and everything built on it: batches, MM-PBSA) no
  longer copies potentials from the backend. Each focusing level reads the
  coarser solution only at the nodes its boundary interpolates, and the
  energy reads the finest solution only at the charged nodes, with the same
  arithmetic as before. Results are identical bit for bit. A CUDA
  calculation is 1.06–1.10× faster on a shared RTX 2080 SUPER (5DK3:
  8.9 → 8.2 s) and its process needs 0.5 GB less host memory; the CPU gains
  1–2 %. Boundary values are kept on the boundary nodes only.
  `timings["energy"]` now includes reading the potential at the charges, and
  `timings["solve"]` no longer includes a download. `Solver.potential` and
  `solve_maps` still return the whole finest potential.
- On CUDA a calculation (`polar_solvation`, `solve_focused`, and so batches
  and MM-PBSA) keeps the GPU memory it frees between focusing levels and
  returns it to the driver when it ends, also on an error. Before, the
  device's memory pool returned the solver's buffers after every level and
  reserved them again for the next one (51 ms per level for 5DK3). While a
  calculation runs, the pool's release threshold is raised by what
  `batch::device_workers` sets aside for it; concurrent calculations add up,
  and the previous threshold is restored afterwards. An idle process holds no
  more GPU memory than before. Results are identical bit for bit. The solve
  stage is 1.05–1.10× faster, and a calculation 1.03–1.08× faster where the
  GPU stages dominate (5DK3 with `sdh`: 4.07 → 3.88 s); with `mdh` the saving
  (~0.2 s for 5DK3) is small next to the host boundary. `timings["solve"]` no
  longer includes releasing GPU memory; the single release at the end of a
  calculation (0.05 s for 5DK3) belongs to no stage.
- The exact `mdh` boundary sums (also the points `mdh-fast` sums exactly)
  run on AVX2 where the CPU has it: the same code with four boundary nodes in
  the lanes of a 256-bit register. Results are identical bit for bit. On the
  CPU the boundary stage is 1.18–1.24× faster (5DK3: 3.54 → 2.91 s).
- `polar_solvation` solves the reference state first. On CUDA the host
  builds the solvated state's molecular surface and outermost boundary while
  the reference state is solved on the GPU, in a thread pool of its own.
  The states are independent, so results are identical bit for bit. A CUDA
  calculation is 1.21–1.25× faster with `mdh` (5DK3: 6.60 → 5.28 s),
  1.15–1.16× with `mdh-fast` and 1.08–1.10× with `sdh`. MM-PBSA batches
  are 1.16× faster on one CUDA worker, and the same within noise on three.
  On CUDA, `timings["boundary"]` and `timings["maps"]` count only the wait for
  the background work.
- On CUDA the outermost level of the solvated state (and of `solve_focused`)
  assembles its maps and operator and sets up its multigrid while its
  boundary is still being built in the background. It waits for the boundary
  only when the right-hand side needs it. Results are identical bit for bit.
  A calculation with `mdh` is 1.05–1.06× faster (5DK3: 5.42 → 5.10 s, median
  of 15), and `Solver.potential` 1.03–1.04×. `timings["boundary"]` now counts
  the wait at that point, and `timings["maps"]` the wait for the surface.
- The multigrid smoother runs each of its steps (the residual and the first
  and later Chebyshev updates) as one pass on the CPU and one kernel on CUDA,
  instead of up to four separate operations; the new `Backend` methods
  `residual`, `chebyshev_first` and `chebyshev_next` default to those
  operations. Results are identical bit for bit. A calculation is 1.17–1.25×
  faster on the CPU; on CUDA the solve stage is 1.17–1.26× faster and a
  calculation 1.05–1.11× (measured on a shared GPU).
- The conjugate-gradient iteration makes fewer passes over memory. One pass
  updates x and r, computes ‖r‖² (and rᵀz) and rounds r for the f32 V-cycle;
  another widens the V-cycle's result while computing rᵀz. On the CPU, A p and
  pᵀAp are one pass as well. The new `Backend` methods `cg_update`,
  `promote_dot` and `apply_dot` default to the separate operations, and
  `backend::Stepwise` runs any backend with those defaults (the tests'
  reference). Results are identical bit for bit. A calculation is
  1.03–1.07× faster on the CPU (1.13× with `jacobi`). On CUDA the CG vector
  kernels take 10 % less time, 2–4 % of the solve stage (measured on a
  shared GPU).

### Fixed

- `coulomb="analytic"` with a spline surface assumed ε = `pdie` on every face
  between nodes more than one spacing inside an atom, which the switching
  window reaches when `swin` exceeds about the spacing (from h ≈ 0.2 Å with
  the default 0.3 Å). A charge at the centre of a 3 Å sphere with `swin=1`
  gave −50.3 instead of −75.3 kcal/mol at h = 0.25 Å. The region where the
  decomposition takes ε = `pdie` now shrinks by the window; the result
  agrees with `coulomb="grid"` and with the exact radial integral.
- Batches on CUDA (`polar_solvation_many`, `polar_solvation_batch`,
  `pbsa_solvation_frames`, `pbsolve mmpbsa`) ran three calculations at once
  whatever their size, and large grids ran out of GPU memory (three 5DK3
  calculations on an 8 GB card). They now run at most as many workers as fit
  in the free device memory, also when `workers` is given; small grids keep
  three. Results do not depend on the number of workers, bit for bit.
- The CUDA platform failed with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` when the
  NVRTC in use was newer than the driver. It also picked a system toolkit's
  NVRTC over the environment's, because cudarc tried the unversioned
  `libnvrtc.so` first. pbsolve now loads NVRTC itself by the versioned name of
  its CUDA major and compiles the kernels to a CUBIN for the device.
- A pip-installed NVRTC (`nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-nvrtc` 13)
  was never found; the preload now searches `site-packages/nvidia/`.
- The binding verdict of `pbsolve apbs compare` (`BindingReport.passed`)
  checked the raw ΔΔG_polar difference, which includes APBS' constant factor
  (ΔΔG × 1.2e-5). Above ~4000 kcal/mol it reported FAIL even for exact
  agreement. It now checks the difference in APBS units, like the rest of
  the report.

### Changed

- `Backend` trait: `precondition` replaced by `scale`; new `prepare_coarse`,
  `apply_coarse`, `restrict`, `prolong_add`, `axpby`, `copy`, `read`, `write`.
  `SolverOptions` and `PbParams` gain `preconditioner`; `SolveStats` gains
  `preconditioner` and `levels`.
- Energies change only within the solver tolerance (≤ 3e-8 relative on the
  tested systems); `preconditioner="jacobi"` reproduces v0.0.1 bit for bit.
- The CPU and CUDA kernels were rewritten generically. The f64 results are
  bit-identical to before (checked on 1YCR on both platforms). CUDA kernel
  names gained an `_f64`/`_f32` suffix.
- Each CUDA backend runs on a stream of its own instead of the legacy default
  stream, so several backends can work concurrently. Results are unchanged.
- `pbsa_solvation_frame` is now `pbsa_solvation_frames` with one frame and
  gives the same results as before.

## 0.0.1 — 2026-11-03 (first preview)

### Solver

- Linearized Poisson–Boltzmann equation on finite-difference grids (7-point
  stencil, Jacobi-preconditioned CG) with multi-level focusing and automatic
  grid sizing.
- Dielectric models `mol` (solvent-excluded surface), `vdw` and `harmonic`;
  mobile ions with an ion-exclusion surface.
- Boundary conditions `mdh` (Debye–Hückel spheres), `sdh` (multipole) and `zero`.
- CPU (rayon) and CUDA (cudarc + NVRTC, loaded at runtime) backends behind one
  `Backend` trait; deterministic reductions.
- Python API: `Solver`, `GridSpec`, batch trajectories, potential maps (OpenDX),
  `Solver.maps()` / `Solver.solve_maps()`, unit helpers.
- MM-PBSA helpers (polar + non-polar terms on a shared grid) and an OpenMM bridge.

### Input files

- Crate `pbsolve-io` (no dependencies) and module `pbsolve.io`: PDB
  (multi-model), PQR, mmCIF, GRO, mol2, Amber prmtop/inpcrd/rst7/ncrst,
  CHARMM/NAMD PSF, GROMACS top/itp, and trajectories DCD, XTC, TRR, Amber
  NetCDF and mdcrd.
- `pbsolve.io.load` / `Molecule` (selections, `strip_solvent`, radius schemes,
  `write_pqr`), `pbsolve.io.frames`, `TrajectoryReader`, `read_structure`.

### Structure preparation

- Module `pbsolve.prep` and `pbsolve prep`: hydrogens, termini, heavy-atom
  repair, debumping, hydrogen-bond network optimisation, protonation states
  (standard, model-pKa rules, or optional PROPKA), AMBER/PARSE/CHARMM charges
  and radii, mol2 ligands.

### Command line and validation

- `pbsolve {info,test,solvation,convert,mmpbsa,prep}`.
- Cross-checks against other Poisson–Boltzmann software (`pbsolve apbs`) and
  validation scripts for the readers and the structure preparation; results in
  `docs/VALIDATION.md`.

### Packaging

- conda recipe with CPU and CUDA 12/13 variants. Package license
  `MIT AND BSD-3-Clause AND BSD-2-Clause` (see `THIRD_PARTY_NOTICES.md`).
