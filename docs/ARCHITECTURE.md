# Architecture

```
python/pbsolve/         pure-Python API: io (load/select/frames), prep (PDB -> PQR),
        │               MM-PBSA helpers, OpenMM bridge, APBS parity, DX I/O, CLI
        │  (PyO3, GIL released during solves and file reads)
crates/pbsolve-py       thin bindings: argument conversion + platform selection
        │
        ├── crates/pbsolve-io     readers: PDB/PQR/mmCIF/GRO/mol2, prmtop/PSF/GROMACS top,
        │                         DCD/XTC/TRR/NetCDF/mdcrd (no dependencies)
crates/pbsolve-core     ALL physics + CPU backend (rayon)      crates/pbsolve-cuda
        │   grid, surfaces, charges, boundaries,                CUDA backend: cudarc,
        │   operator, focusing, energies, PCG     ◄── Backend ──  NVRTC kernels
```

Input path: any supported file → `pbsolve.io.Molecule` (charges from the
topology, radii from the file or an element scheme) → `Solver`. PDB files
without charges go through `pbsolve.prep` (pdb2pqr-compatible hydrogens,
protonation states and force-field charges/radii).

## Numerical model

LPBE in APBS' dimensionless form (u in kT/e, lengths in Å):

    -∇·(ε∇u) + κ̄² u = 4π l_B0 Σ z_i δ(r − r_i),   l_B0 = e²/(4πε₀k_BT),  κ̄² = ε_s κ²

* **Discretisation:** cell-centred finite volumes on a regular grid, 7-point
  stencil, ε on the three staggered (face-centred) meshes, κ̄² on nodes. The
  matrix is scaled by the cell volume and is symmetric positive definite.
* **Charges:** trilinear spreading (APBS `chgm spl0`).
* **Dielectric boundary:** `mol` reproduces APBS 3.x `srfm mol` operation for
  operation (`surface.rs`): face nodes inside `r_i + srad` of any atom become
  solute, then nodes within `srad` of an *accessible* SAS point become solvent
  again. SAS points come from one APBS θ/φ reference sphere
  (`⌈4π(r_max + R_clist)²·sdens⌉` requested points, `R_clist = max(srad,
  r_ion) + 0.5`) scaled onto every atom; a point is buried if strictly inside
  `r_j + srad` of another atom, zero-radius atoms included. Node tests use
  APBS' `markSphere` arithmetic, so the maps are identical to APBS' `dielx/y/z`.
  `vdw` is the same with `srad = 0`; `harmonic` evaluates the surface at nodes
  and uses harmonic means on faces (no APBS equivalent).
* **Ions:** κ̄² is zero inside `r_i + r_ion` (largest listed ion radius) of
  every atom with `r_i > 0`, marked the same way.
* **Boundary conditions** (APBS `vpmg.c`): `mdh` sums Debye–Hückel spheres of
  the atom radii; `sdh` uses charge + dipole + traceless quadrupole about the
  coordinate-box centre with the exact exterior solution of a sphere (ε_p
  inside, ε_w and κ outside); `zero`. Applied on the outermost grid only.
* **Focusing** follows APBS `mg-auto`: two levels, or more when
  `h_fine/h_coarse < 0.25` (`ρ = (h_f/h_c)^{1/(n−1)}` per level); every level
  takes its boundary from the previous one by trilinear interpolation. Node
  coordinates use APBS' formula (`xmin = c − (n−1)h/2`).
* **Energy:** ΔG_polar = ½ Σ q_i [u_solv(r_i) − u_ref(r_i)], with the reference
  state (ε = pdie everywhere, no ions) solved on the *same* grids so the grid
  self-energy cancels exactly.
* **Linear solver:** preconditioned CG, written once against `Backend`.
  Preconditioner: one geometric multigrid V-cycle (`multigrid.rs`):
  per-axis vertex coarsening `n → n/2 + 1` (any grid size; axes with < 5 nodes
  are kept), trilinear prolongation `P` and restriction `Pᵀ`, Galerkin coarse
  operators `PᵀAP` (27-point, exactly symmetric), Chebyshev smoothing of
  degree 2 on `[λmax/10, λmax]` of `D⁻¹A` (Gershgorin bound), banded Cholesky
  on the coarsest level (≤ 1000 unknowns). The V-cycle is a fixed SPD
  operator, so plain CG theory applies; CG iterations stay at ~10–20
  independent of the spacing. `preconditioner="jacobi"` keeps the v0.0.1
  solver (bit-identical results). The default, `preconditioner="multigrid-f32"`
  (`Preconditioner::DEFAULT`), evaluates the same V-cycle in single precision.
  The outer CG stays in f64 and uses the flexible (Polak–Ribière) update,
  because the rounded V-cycle is no longer exactly symmetric. The converged
  f64 solution is unchanged and the cycle moves half the data.
  `preconditioner="multigrid"` runs the cycle in f64.

## The Backend contract

A backend provides:
* `prepare`/`prepare_coarse`, which upload the 7-point and 27-point
  operators;
* `upload/zeros/download/read/write`;
* `apply` and `apply_coarse` (stencils);
* `restrict` and `prolong_add` (grid transfers);
* `dot` and the fused `dot2`, both deterministic;
* the element-wise updates `axpy`, `xpay`, `axpby`, `scale` (diagonal
  scaling) and `copy`;
* a single-precision twin (`type Single`, `single()`) on the same device and
  stream, with `demote`/`promote` to convert vectors.

The kernels are written once, generic over `Real` (`f64`, `f32`; `real.rs`).
CUDA instantiates templates as `pb_*_f64` and `pb_*_f32`. That is the entire surface a new
accelerator (HIP, Metal, SYCL) must implement; the multigrid setup (Galerkin
products, smoother bounds, coarsest factorisation) runs once on the host and
is shared. The CUDA kernels are written in the same operation order as the
CPU ones and compiled with `--fmad=false`, so only the reduction order differs.

## Concurrent calculations (trajectories, MM-PBSA)

`batch.rs` runs independent polar-solvation calculations on several workers.
Each worker has its own backend and a private rayon pool holding an equal
share of the threads. A CUDA backend owns a non-blocking stream; buffers never
cross streams, so cudarc's event tracking is switched off. While one worker
builds surfaces, boundary values and the multigrid hierarchy on the host,
another one's kernels run. Jobs are handed out dynamically; results are
returned in job order, so they are identical to sequential calls bit for bit.
The Python `Solver` keeps the extra backends in a pool, created on first use:

* `polar_solvation_many(systems, grid, workers)` runs any list of systems;
* `polar_solvation_batch(frames, …, workers)` runs the frames of a trajectory;
* `pbsa_solvation_frames` runs three jobs per frame;
* `pbsolve mmpbsa --workers` feeds chunks of frames to the latter.

Every result reports per-stage wall times (`PolarResult.timings`).
`docs/VALIDATION.md` records how the default worker counts (CPU 2, CUDA 3)
were chosen.

## CUDA runtime model

* The driver (`libcuda`, through cudarc) and NVRTC (`pbsolve-cuda/src/nvrtc.rs`)
  are **dlopen'ed at runtime**. Building needs no CUDA toolkit; a machine
  without a driver gets `available_platforms() == ["CPU"]` and a readable
  reason in `cuda_info()`.
* NVRTC is opened as `$PBSOLVE_NVRTC`, else by the versioned name of the
  build's CUDA major (`libnvrtc.so.13`, `nvrtc64_130_0.dll`), else as
  `libnvrtc.so`. The dynamic loader returns an already-loaded library of the
  versioned name, so the copy the Python package preloads wins over a system
  toolkit.
* The Python package preloads that library from the environment (conda
  `lib/`, then pip's `nvidia/cuda_nvrtc` or `nvidia/cu13`) right after
  importing the extension; NVRTC is only opened at the first CUDA query.
* Kernels are compiled when a `Solver(platform="CUDA")` is created, to a
  **CUBIN for the device's architecture**: it loads on any driver of the same
  major version, whereas PTX from an NVRTC newer than the driver is rejected.
  PTX (`compute_XY`) is used only for GPUs newer than the NVRTC.
  `PBSOLVE_CUDA_CODE=ptx|cubin` forces either. The image is cached per
  process.
* `cuda_info()` reports the driver and NVRTC versions and the NVRTC file in
  use.

## Roadmap

See `docs/ROADMAP.md` for the plan with literature and acceptance criteria.
In short: multigrid-preconditioned CG and batched GPU solves (v0.1),
second-order interface discretisation and error estimates (v0.2), nonlinear
PB, forces and smooth dielectrics (v0.3), end-point free-energy features
(v0.4), adjoint gradients for ML workflows (v0.5), then further solution
methods (boundary elements, domain decomposition) behind the same API.
