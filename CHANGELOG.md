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
- `PolarResult.timings` gives the seconds spent per stage: maps, boundary,
  assembly, setup, solve and energy. `SolveStats` gains `setup_seconds` and
  `solve_seconds`; `PolarSolvation` and `FieldSolution` gain `timings`.

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

### Fixed

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
