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
* **Linear solver:** Jacobi-preconditioned CG, written once against `Backend`.

## The Backend contract

A backend provides: `prepare` (upload coefficients), `upload/zeros/download`,
`apply` (stencil), `precondition` (Jacobi), `dot` (deterministic), `axpy`,
`xpay`. That is the entire surface a new accelerator (HIP, Metal, SYCL) must
implement. The CUDA stencil is written in the same operation order as the CPU
one and compiled with `--fmad=false`, so only the reduction order differs.

## CUDA runtime model

* The driver (`libcuda`) and NVRTC are **dlopen'ed at runtime** (cudarc
  `dynamic-loading`). Building needs no CUDA toolkit; a machine without a
  driver gets `available_platforms() == ["CPU"]` and a readable reason in
  `cuda_info()`.
* Kernels are compiled by NVRTC when a `Solver(platform="CUDA")` is created,
  for whatever GPU is present.
* The Python package preloads `libnvrtc.so.*` from the environment so the
  conda-provided NVRTC is always found.

## Roadmap (in priority order)

1. Geometric multigrid preconditioner (grids are already 8k+1) — CG iteration
   count currently grows with grid refinement.
2. APBS parity report on a larger public protein/ligand benchmark set
   (`docs/VALIDATION.md`; tooling in `pbsolve.apbs`, first complex done).
3. Exact APBS `smol`/`spl2` surface definitions for drop-in comparisons.
4. Nonlinear PBE (Newton–CG) and forces.
5. Batched multi-frame GPU solves (several small grids per kernel launch).
6. HIP backend for AMD GPUs.
