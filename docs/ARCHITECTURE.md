# Architecture

```
python/pbsolve/         pure-Python API: io (load/select/frames), prep (PDB -> PQR),
        │               MM-PBSA helpers, interaction entropy, OpenMM bridge,
        │               APBS parity, DX I/O, CLI
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
* **Charges:** trilinear spreading (APBS `chgm spl0`, the default) or the
  order-3 and order-5 B-splines of APBS `chgm spl2` and `spl4` (`charge.rs`);
  the energy uses the same weights for `spl2`, trilinear ones otherwise.
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
  `interface="fractional"` (with `mol` or `vdw`; `surface::fractional_faces`,
  host) splits each face's grid edge exactly into its lengths inside and
  outside the solute (chords through the solute spheres minus chords through
  the probe spheres) and gives cut faces the harmonic average weighted by
  them; `mark_map` first classifies the nodes whose edges cannot be cut.
  `spl2` and `spl4` (`spline.rs`, host) are APBS' spline surfaces
  (`fillcoCoefSpline`, `fillcoCoefSpline4`): ε = pdie + (sdie − pdie)·Π S_a
  on the faces and the accessibility K = Π S_a(r + r_ion) on the nodes, with
  each atom's cubic or septic switch over [r − swin, r + swin], in APBS'
  order of atoms, products and guards.
  `gauss` (`gaussian.rs`, host; not an APBS option) has the same form with
  S_a = 1 − exp(−d²/σ²R²), evaluated as the product of per-atom axis factors
  and cut off below 1e-12, and the ions' Born accessibility
  K = exp(−z² l_B/(2 r_ion)·(1/ε − 1/sdie)) on the nodes. Its reference
  state is not homogeneous: `maps_on` without a surface gives the Gaussian ε
  where it is at most `gauss_contour`, `pdie` beyond (`contour_faces`).
  Atom dielectrics (`PbParams::atom_pdie`, Python `dielectrics=`;
  `surface::atom_dielectric_faces`, host; not an APBS option) mark the
  sharp `mol`/`vdw` interior with `mark_map` and give each interior face the
  dielectric of the atom with the smallest power distance |x − c|² − R²
  (cell-list search within the largest solute sphere + 2h); faces outside
  get `sdie`, and in the reference state, which rebuilds the surface,
  `pdie`. Batches carry them per job (`batch::Job::atom_pdie`).
  The implicit membrane (`PbParams::membrane`, `membrane.rs`, host; APBS'
  `draw_membrane2` rule) is drawn into the solvated state's finished maps
  of every level: `maps_on` downloads them, `Membrane::apply` sets the
  non-solute faces (ε > pdie + 0.05) and all nodes of the slab outside the
  exclusion cone to the membrane's ε (uniform, or PCHIP in the depth below
  the nearer surface) and κ̄² = 0, optionally only where a 4-connected path
  in the z plane reaches the box sides (`fill`), and uploads them again. The
  reference state has no membrane.
* **Ions:** κ̄² is zero inside `r_i + r_ion` (largest listed ion radius) of
  every atom with `r_i > 0`, marked the same way.
* **Nonlinear PB** (`nonlinear.rs`): with `equation="npbe"` the solvated
  levels are solved by an inexact Newton iteration from the linearized
  solution, damped where |u| > 1 (Gouy–Chapman). The maps are built as usual
  and read back for the energy. Every step is a linear solve with κ̄²
  replaced by N'(u) on the backend (multigrid CG), damped by halving until
  ‖F‖ decreases. The backend operations `boltzmann_slope` and
  `boltzmann_residual` (CPU loops, CUDA kernels) evaluate N'(u) and
  complete J u into the residual, so the iteration stays on the device; it
  keeps 4 vectors beside the linear solve (`calculation_device_bytes`).
  `equation="smpbe"` divides the ion terms by the lattice denominator
  D = 1 − Σφ + Σφ e^{−qu} of finite-size ions and uses ln D for their energy;
  with φ = 0 (point ions) D is exactly 1 and the code is that of `npbe`.
  The energy is APBS' functional Σqu − E_ε − E_ion plus the coarser levels'
  outer terms (`NonlinearLevel::external_terms`, APBS `Vpmg_extEnergy`).
* **Forces** (`forces.rs`, `pb::polar_forces`): the explicit derivative of
  each state's functional at its converged finest-level potential, read back
  in full: charge–field (derivative of the `chgm` weights), dielectric
  boundary (∂ε_f/∂R = (sdie − pdie) H_f ∇ ln S_a, `spline::LogGradient`) and
  ionic boundary (∂κ̄²/∂R times the ion density ½u², the Boltzmann terms or
  ln D). Solvated minus reference, on the host, parallel over atoms,
  following APBS `qfForceSpline1/2/4`, `Vpmg_dbForce`, `Vpmg_ibForce`; it
  needs a spline surface. Without forces `polar` reads only the charged
  nodes, as before.
* **Non-polar terms** (`apolar.rs`, `pbsolve.apolar_energy`): APBS
  `apolar`'s area (reference-sphere points on r + srad, the surface code's
  `Accessibility` test), volume (trapezoidal rule over APBS' cell-list box)
  and WCA dispersion (per atom, a 0.5 Å-default lattice within 14 Å), with
  APBS' coordinate accumulation and order of sums; host only, parallel over
  atoms or planes. Independent of the PB grid.
* **Coulomb part** (`coulomb.rs`): with `coulomb="analytic"` the finest level
  solves A v = χ·A_p G − A(χG) for v = u − χG, where G is the Coulomb
  potential in `pdie` and χ the interior nodes within max(1 Å, 2h) of an
  atom. `Split` marks χ and the thin layer where the right-hand side can be
  nonzero (with `mark_map`; deep nodes see only `pdie` faces and no ions).
  It reads the maps on that layer back from the backend, sums G directly
  there, and passes the result to the solver as node charges. The coarser
  levels and the operator are those of the solvated state; no reference
  state is solved.
* **Boundary conditions** (APBS `vpmg.c`): `mdh` sums Debye–Hückel spheres of
  the atom radii; `sdh` uses charge + dipole + traceless quadrupole about the
  coordinate-box centre with the exact exterior solution of a sphere (ε_p
  inside, ε_w and κ outside); `zero`. Applied on the outermost grid only.
  `mdh-fast` (pbsolve only) evaluates the `mdh` sum exactly at Chebyshev
  points of face patches and interpolates between them. The degree comes from
  each patch's distance to the atoms and κ, and the error is ≤ 1e-10 of the
  summed term magnitudes. Patches near atoms are summed directly.
  The exact sums interleave four boundary nodes. Where the CPU has AVX2 the
  same code runs with the four nodes in the lanes of a 256-bit register:
  lane-wise IEEE arithmetic rounds like the scalar code, Rust never forms
  FMAs, and `exp` remains the scalar libm call, so the sums are bit-identical.
* **Focusing** follows APBS `mg-auto`: two levels, or more when
  `h_fine/h_coarse < 0.25` (`ρ = (h_f/h_c)^{1/(n−1)}` per level); every level
  takes its boundary from the previous one by trilinear interpolation. Node
  coordinates use APBS' formula (`xmin = c − (n−1)h/2`).
* **Energy:** ΔG_polar = ½ Σ q_i [u_solv(r_i) − u_ref(r_i)], with the reference
  state (ε = pdie everywhere, no ions) solved on the *same* grids so the grid
  self-energy cancels exactly. Each atom's term (`charge::atom_energies`,
  the energy's weights applied to the potential at the charged nodes that
  the energy reads anyway) is kept as `PolarSolvation::atom_energies_kcal`;
  the energy itself is summed as before, so it is unchanged bit for bit.
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
* the multigrid smoother steps `residual` (r ← b − A x, or r ← r − A x),
  `chebyshev_first` and `chebyshev_next`, with defaults that run them as
  sequences of the operations above (`backend::*_by_steps`). The CPU and
  CUDA backends fuse each step into one pass (one kernel) that evaluates
  every element with the same expressions: smoothing and residual move about
  30 % less data (10 instead of 23 kernels per level), and the results stay
  identical bit for bit;
* the CG steps `apply_dot` (q ← A p and p·q), `cg_update` (x ← x + α p,
  r ← r − α q, ‖r‖², for flexible CG also r·z, and the f32 copy of r for the
  mixed-precision V-cycle) and `promote_dot` (z widened from f32 and r·z),
  also defaulted to their sequences. The CPU fuses all three, CUDA the last
  two (`apply_dot` is slower as one kernel there: on the reduction grid the
  stencil loses its L2 reuse). A fused step sums every dot product exactly as
  `dot` does (the CPU's 2¹⁴-element chunks, the GPU's grid of at most 1024
  blocks with the same elements per thread), so the results stay identical
  bit for bit. `backend::Stepwise<B>` runs a backend with every fused step as
  its sequence; tests compare whole solves with it;
* a single-precision twin (`type Single`, `single()`) on the same device and
  stream, with `demote`/`promote` to convert vectors;
* the multigrid setup (f64 backends only): `galerkin` (Galerkin operator and
  inverse diagonal of the next level), `gershgorin_lmax`, `coarse_to_host`
  (for the coarsest factorisation), and `demote_operator`/`demote_coarse`
  (f32 copies for the twin);
* the assembly of the linear system (f64 backends only): `mark_maps`
  (dielectric and ion maps: a fill value, then spheres marked pass by pass),
  `scatter` (sparse charges and boundary values into a zero vector),
  `assemble_operator` (κ̄²V and the inverse diagonal from the maps),
  `source_rhs` (b = 4π l_B q − A g on interior nodes), `operator_to_host`
  (grids too small to coarsen), `gather` (the values at a list of nodes),
  and `upload_owned`/`synchronize`/`gather` with defaults;
* memory: `retain_memory`/`release_memory` (no-ops by default), used through
  `backend::MemoryHold` for the duration of one calculation;
* `runs_on_device` (false by default): whether the kernels leave the host's
  threads idle, which lets the pipeline do host work in the background.

The kernels are written once, generic over `Real` (`f64`, `f32`; `real.rs`).
CUDA instantiates templates as `pb_*_f64` and `pb_*_f32`; the setup and
assembly kernels exist in f64 only. That is the entire surface a new
accelerator (HIP, Metal, SYCL) must implement. `Hierarchy::build` is the host
reference of the setup; `Levels::build` runs the same steps through the
backend, so the CUDA backend builds the coarse operators in device memory and
only the coarsest level (≤ 1000 unknowns) is factorised on the host. Each
Galerkin row is accumulated by one thread in the host's loop order. The CUDA
kernels are written in the same operation order as the CPU ones and compiled
with `--fmad=false`, so the setup is bit-identical to the host and only the
dot-product reduction order differs.

The same holds for the assembly. `surface::maps_on` decides which spheres
mark which value (solute spheres, probe spheres, ion-exclusion spheres) and
calls `mark_maps`; the CPU backend runs `surface::mark_map`, the CUDA backend
marks one sphere per warp with the host's node test. The `harmonic` surface
and the `fractional` interface are built on the host and uploaded. Charges are spread on
the host (`spread_charges_sparse_with`, the same sums as `spread_charges_with`) and
scattered on the device together with the boundary values; `pcg_on` then
solves without any host copy of the operator. The solution stays on the
backend: the next focusing level reads it with `gather` at the trilinear
stencils of its boundary nodes (`boundary::focus_nodes`), and the energy at
the charged nodes, with the same arithmetic as the full potential, so the
results are bit-identical. Only `solve_focused` (potential maps) downloads
the finest potential. The surface itself (SAS points) and the boundary
values are computed on the host.

`polar_solvation` solves the reference state first, then the solvated state;
they are independent, so the order changes no result. On a device backend
(`runs_on_device`) the host builds the solvated state's inputs (its molecular
surface and the analytic boundary of its outermost level, with `mdh` the
largest host stage) while the reference state is solved. The reference
state's own boundary is cheap and is built before, with all threads, while
the device is idle. The background work runs in a rayon pool of its own with
one thread fewer than the caller's pool. In the caller's pool, a thread that
waits for its own tasks would run queued background tasks, and the device
would wait for them.

The inputs come through channels, one value at a time. The outermost level
of the solvated state needs its boundary values only for the right-hand
side. Before waiting for them it assembles its maps and operator and sets up
its multigrid (`solver::pcg_setup`, then `PreparedPcg::solve`, which together
compute what `pcg_on` computes). `solve_focused` does the same for its one
state. `timings` count only the waits: for the surface as map time, for the
boundary as boundary time.

## Concurrent calculations (trajectories, MM-PBSA)

`batch.rs` runs independent polar-solvation calculations on several workers.
Each worker has its own backend and a private rayon pool holding an equal
share of the threads. A CUDA backend owns a non-blocking stream; buffers never
cross streams, so cudarc's event tracking is switched off. While one worker
builds surfaces, maps and boundary values on the host, another one's kernels
run. Jobs are handed out dynamically; results are
returned in job order, so they are identical to sequential calls bit for bit.
The Python `Solver` keeps the extra backends in a pool, created on first use:

* `polar_solvation_many(systems, grid, workers)` runs any list of systems;
* `polar_solvation_batch(frames, …, workers)` runs the frames of a trajectory;
* `pbsa_solvation_frames` runs three jobs per frame;
* `pbsolve mmpbsa --workers` feeds chunks of frames to the latter.

Every result reports per-stage wall times (`PolarResult.timings`).
`docs/VALIDATION.md` records how the default worker counts (CPU 2, CUDA 3)
were chosen.

On a GPU every worker holds the buffers of its own calculation.
`solver::solve_device_bytes` counts them from the grid size and the
preconditioner: the peak is reached in the CG iterations, and each solve
frees its buffers before the next one. `tests/memory.rs` checks this count
against the CUDA memory pool's high watermark, to the byte. Before a batch,
the Python `Solver` reads the free device memory and lets
`batch::device_workers` cap the worker count. Each worker is given the largest
calculation of the batch plus the pool's rounding, and 1/20 of the device
stays free.

All CUDA backends of a process allocate from the device's memory pool. By
default the pool returns freed memory to the driver at the next
synchronisation, so every focusing level would reserve its buffers again.
`polar_solvation` and `solve_focused` therefore hold a `MemoryHold` of
`worker_device_bytes` for their grid. The CUDA backend raises the pool's
release threshold by the bytes of all calculations in progress on the
device (a process-wide registry, so concurrent workers add up). A
calculation that ends lowers the threshold by its bytes and trims the pool
to the new threshold, so after the last one the previous threshold is back
and the memory is returned. An idle process holds no more GPU memory than before, and a worker never keeps
more than it was given by `device_workers`.

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
