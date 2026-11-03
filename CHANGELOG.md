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

### Changed

- `Backend` trait: `precondition` replaced by `scale`; new `prepare_coarse`,
  `apply_coarse`, `restrict`, `prolong_add`, `axpby`, `copy`, `read`, `write`.
  `SolverOptions` and `PbParams` gain `preconditioner`; `SolveStats` gains
  `preconditioner` and `levels`.
- Energies change only within the solver tolerance (≤ 3e-8 relative on the
  tested systems); `preconditioner="jacobi"` reproduces v0.0.1 bit for bit.

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
