# pbsolve development roadmap

Status: draft for v0.1–v1.0, written after v0.0.1 (November 2026).
Each item states **why** (the scientific or engineering gap, with literature),
**what** we build, and **how we will know it works** (acceptance evidence).
Items are implemented from the published equations; see
[Licensing guardrails](#7-licensing-guardrails) before reading other codes.

## 1. Where pbsolve should stand

pbsolve aims to be a fast, accurate and openly licensed Poisson–Boltzmann
engine for **ensembles**: many conformations from simulations, several species
per frame (complex, receptor, ligand), on CPUs and GPUs, with numbers that come
with an error estimate. Four properties set the direction:

1. **Throughput on ensembles.** MM-PBSA and related end-point methods average
   over hundreds to thousands of frames; their statistical error usually
   dominates over the PB discretisation error [Genheden & Ryde 2015; Wang E
   et al. 2019]. Fast batched solves are therefore worth more than a faster single solve.
2. **Controlled accuracy.** Sharp dielectric interfaces make standard
   finite-difference energies oscillate with grid placement (our Born-ion
   errors: −4.7 / −1.1 / −1.3 / −0.9 % at 0.5 / 0.33 / 0.25 / 0.2 Å).
   Interface-aware discretisations reach second-order convergence and
   grid-independent energies [Zhou, Feig & Wei 2008; Wei, Luo & Qi 2019; Wei
   et al. 2019]. pbsolve should offer these and report a discretisation-error estimate.
3. **Several solution methods behind one API.** Finite differences with
   multigrid now; later boundary-element and domain-decomposition solvers
   (Section 5), sharing the molecule, dielectric model, energies and validation suite.
4. **Built for modern workflows.** Direct MD-format input (done), OpenMM
   integration, gradients with respect to coordinates and parameters (forces,
   parameter fitting, machine-learning training), deterministic results.

## 2. Starting point (v0.0.1)

| Area | State | Main limitation |
|---|---|---|
| Discretisation | cell-centred FD, 7-point stencil, trilinear charges, focusing | first-order accuracy at the dielectric interface |
| Linear solver | Jacobi-preconditioned CG, CPU + CUDA | iterations grow with refinement (≈160 → 420 on the Born test) |
| Physics | linearized PB, `mol`/`vdw`/`harmonic` surfaces, `mdh`/`sdh`/`zero` | no nonlinear PB, no forces, no ion-size effects |
| Energies | ΔG_polar (reference state on the same grid), γ·SASA + β | single nonpolar model, no per-residue decomposition |
| Workflows | readers for all major MD formats, PDB preparation, MM-PBSA terms per frame, OpenMM bridge | frames solved one by one |
| Validation | analytic Born ion, map-level cross-checks with an established FD code, reader and preparation validation | no multi-charge analytic tests, no binding benchmark |

## 3. Roadmap overview

| Version | Theme | Key deliverables |
|---|---|---|
| v0.1 | Performance foundation | multigrid-preconditioned CG (CPU + GPU), robust NVRTC loading, batched multi-frame solves, mixed precision |
| v0.2 | Accuracy | reaction-field decomposition, B-spline charges, analytic test suite, second-order interface method, error estimates |
| v0.3 | Physics | nonlinear and size-modified PB, forces, smooth dielectric models |
| v0.4 | End-point free energies | nonpolar models, per-residue decomposition, entropy options, membranes, uncertainty, public benchmark |
| v0.5 | Gradients and ML | adjoint gradients, PyTorch/JAX interop, dataset generation, Δ-learning hooks |
| v0.6+ | New solution methods | boundary-element (FMM/treecode), domain-decomposition LPB, PB-based pKa, AMD/HIP |
| v1.0 | Stable release | frozen API, documentation site, software paper, conda-forge |

The order follows dependencies: multigrid makes fine grids affordable, which
the accuracy work needs; the decomposed formulation from v0.2 is the basis
for forces (v0.3) and gradients (v0.5).

## 4. Detailed plan

### v0.1 — Performance foundation

**4.1 Geometric multigrid preconditioner.** *Why:* multigrid is the optimal
solver for the PB operator on CPUs [Holst & Saied 1993]. On GPUs, single
precision makes robustness a bigger issue than speed for large biomolecules,
and a hybrid of multigrid speed and CG robustness works best [Qi & Luo 2019].
The dielectric jump (ε 1–4 → 78.5) requires operator-dependent interpolation
or Galerkin coarse operators; plain rediscretisation degrades convergence
[Alcouffe et al. 1981].
*What:* vertex-based coarsening of the existing 8k+1 grids (n → (n+1)/2),
Galerkin (RAP) coarse operators with operator-dependent prolongation,
red–black Gauss–Seidel smoothing applied symmetrically so the V-cycle is SPD,
used as the `precondition` step of the existing PCG. Written once in
`pbsolve-core`; the CUDA backend adds restriction, prolongation and smoother
kernels with the same operation order (invariant 5).
*Evidence:* iteration count independent of spacing (0.5 → 0.2 Å); ≥ 5× fewer
operator applications than Jacobi-PCG at 0.25 Å; energies unchanged to the
solver tolerance; CPU/CUDA agreement < 1e-8; determinism preserved.
*Status: done (unreleased).* Chebyshev smoothing [Adams et al. 2003] instead
of Gauss–Seidel (no colouring, same kernels for 7- and 27-point operators),
per-axis coarsening for any grid size. Measured: 10–19 CG iterations
independent of spacing (Jacobi 155–688), 5.4–7.7× faster full ΔG_polar on the
CPU and 6.8–10× on the GPU for 1YCR, energies within 3e-8 of Jacobi, CPU vs
CUDA ≤ 3e-14, thread-count independent, APBS parity unchanged
(`docs/VALIDATION.md`). Operator-dependent prolongation was not needed.

**4.2 GPU robustness.** Load the NVRTC library the package preloads rather
than whichever `libnvrtc.so` the loader finds first, and compile to CUBIN for
the detected architecture, so a newer NVRTC works with an older driver (open
issue in `docs/VALIDATION.md`). Run the CUDA 12 build.
*Status: done (unreleased).*
* NVRTC is loaded by pbsolve itself: `$PBSOLVE_NVRTC`, then the versioned
  name of the build's major, then `libnvrtc.so`. The Python preload also finds
  pip's `nvidia-cuda-nvrtc*`.
* Kernels are compiled to a CUBIN for the device; PTX is used only for GPUs
  newer than the NVRTC.
* Verified with NVRTC 12.9 / 13.2 / 13.3 / 13.4 on a 13.2 driver: energies are
  bit-identical across all of them. The `cuda-12` build has been run on the
  GPU. Three setups that failed before now work (`docs/VALIDATION.md`, *GPU
  runtime*).

**4.3 Batched multi-frame solves.** Solve several frames or species at once,
either as one batched kernel launch over small grids or concurrently in streams.
*Evidence:* frames per second on a public complex for 1, 8 and 64 frames per
batch; identical energies to single-frame solves.
*Status: done (unreleased).*
* Concurrent workers (`batch.rs`), each with its own backend, its own CUDA
  stream and a private share of the CPU threads.
* `polar_solvation_many`, `pbsa_solvation_frames` and `pbsolve mmpbsa
  --workers`; per-stage timings in every result.
* MM-PBSA on 1YCR: 0.27 → 0.60 frames/s on the GPU (2.2×) and 0.093 → 0.130 on
  the CPU (1.4×). Results are bit-identical for every batch size and worker
  count.
* At 3 CUDA workers the GPU kernels are the limit, so the next gains come from
  faster kernels (fused updates, 4.4).
* One batched kernel launch over many small grids was not needed for this
  workload. It remains an option for small ligand-only grids.
* CUDA workers from the free device memory (done): the device memory of a
  calculation is counted exactly (`solve_device_bytes`, equal to the memory
  pool's peak), and a batch runs at most as many workers as fit. Three 5DK3
  calculations now run on two workers instead of running out of memory on an
  8 GB card. On the shared GPU, two 5DK3 workers were no faster than one.

**4.4 Mixed precision.** FP32 multigrid cycles inside an FP64 outer CG or
iterative refinement, the arrangement analysed for multigrid by McCormick,
Benzaken and Tamstorf [2021a, b]. Optional, off by default until the
benchmark shows equal energies (|ΔG difference| < 1e-6 relative).
*Status: done (unreleased); the default since the robustness sweep below.*
* `preconditioner="multigrid-f32"`: an f32 V-cycle inside f64 flexible CG.
  The kernels are generic over f64/f32, and the f64 results are unchanged bit
  for bit.
* ΔG_polar agrees with f64 to ≤ 7.9e-13 relative, with equal iteration
  counts.
* 1YCR is 1.42–1.55× faster on the CPU, and 2.2× faster together with 4.3.
* On the shared GPU the V-cycle is 2× faster on the device, but a full
  calculation gains only 1.03–1.12×. CG synchronisations (inner products and
  the host coarsest solve) now dominate there.
* Made the default after a robustness sweep: the Born ion and four proteins
  (up to 20 179 atoms) with 14 parameter variants on CPU and CUDA, with
  energies within 4.2e-13 and equal iteration counts (`docs/VALIDATION.md`,
  *Mixed precision*). `preconditioner="multigrid"` keeps the f64 cycle.
* On an idle GPU the host stages (maps, Debye–Hückel boundary, right-hand
  side, multigrid setup) take 75–78 % of a calculation; the CG solve is
  the smaller part.

**Host stages** (follow-up of 4.4). *Status: done (unreleased).*
* Faster maps, `mdh` sums, operator setup and host transfers, with results
  identical bit for bit: 1.06–1.26× per calculation (CPU), 1.03–1.19× (CUDA).
* Optional `bcfl="mdh-fast"`: the `mdh` sum is evaluated exactly at Chebyshev
  points of face patches and interpolated in between, as in kernel-independent
  treecodes [Wang, Krasny & Tlupova 2020; Trefethen 2019].
  * Error ≤ 1e-10 of the summed term magnitudes, checked up to κ = 0.6 Å⁻¹.
  * Energies within 1.1e-13 of `mdh`.
  * The 5DK3 boundary is 6× faster.
  * `mdh` stays the APBS-compatible default.
* An order-preserving rewrite of the Galerkin product brought nothing: it is
  arithmetic-bound, as measured in `docs/VALIDATION.md`.
* Multigrid setup on the GPU (done): Galerkin operators, inverse diagonals,
  Gershgorin bounds and f32 copies are built in device memory, bit-identical
  to the host. CUDA setup 2.2–3.1× faster, a calculation 1.2–1.3× faster.
* Maps and right-hand side on the GPU (done): the dielectric and ion maps,
  the operator, the node charges and the right-hand side are assembled in
  device memory through the backend, bit-identical to the host; the host
  operator is no longer built or uploaded. A CUDA calculation is 1.23–1.32×
  faster (5DK3 10.3 → 8.4 s on a shared GPU).
* Potentials read only where needed (done): the solution stays on the
  backend; the next focusing level reads it at the stencil nodes of its
  boundary and the energy at the charged nodes (`Backend::gather`),
  bit-identical. A CUDA calculation is 1.06–1.10× faster, and 5DK3 needs
  0.5 GB less host memory.
* GPU memory kept during a calculation (done): the memory pool keeps the
  buffers freed between focusing levels (release threshold raised by the
  calculation's share) and returns them when the calculation ends. Solve
  stage 1.05–1.10× faster, results bit-identical.
* `mdh` boundary vectorised and overlapped (done): the exact sums run on
  AVX2 (CPU boundary 1.18–1.24× faster). On CUDA the solvated state's surface
  and outermost boundary are built while the reference state is solved. A
  CUDA calculation is 1.21–1.25× faster (5DK3 6.60 → 5.28 s), and results
  are bit-identical. A GPU `mdh` kernel would be slower than the host on
  consumer GPUs (f64 at 1/32–1/64 rate, estimated ~4 s for 5DK3) and would
  change the last bits (device `exp`). It remains an option for GPUs with
  full-rate f64.
* Solvated setup before its boundary (done): on CUDA the solvated state's
  outermost level builds its maps, operator and multigrid while its boundary
  is still being built (`pcg_setup`, `PreparedPcg::solve`), bit-identical.
  `mdh` calculations are 1.05–1.06× faster.
* Fused multigrid smoother (done): the V-cycle's residual and Chebyshev
  updates run as one pass (CPU) or kernel (CUDA) each, bit-identical; CPU
  calculations 1.17–1.25×, the CUDA solve stage 1.17–1.26×.
* Fused CG steps (done): the update of x and r with ‖r‖², rᵀz and the f32
  copy of r, the widening of the V-cycle's result with rᵀz and, on the CPU,
  A p with pᵀAp run as one pass (CPU) or kernel (CUDA) each, with the
  summation order of `dot`, bit-identical; CPU calculations 1.03–1.07×
  (1.13× with Jacobi), the CUDA CG vector kernels 1.11×. A p with pᵀAp stays
  two kernels on CUDA: on the reduction grid the stencil loses its L2 reuse.
* *Next:* measure both fused steps on an idle GPU. The host `mdh` sum (~3 s
  for 5DK3) still bounds CUDA calculations with the default boundary.

### v0.2 — Accuracy

**4.5 Reaction-field decomposition.** Split the potential into the singular
Coulomb part (analytic) and a smooth reaction-field part that is solved on
the grid [Chern, Liu & Wang 2003; Geng, Yu & Wei 2007]. This removes the
grid self-energy, so the reference solve is no longer needed, and accuracy
improves at coarse spacing. It is also the formulation that interface methods
and forces build on. The current "same grid" reference-state method stays
available and is the default until the new path is validated.

**4.6 Charge assignment.** Cubic and quintic B-spline spreading (smoother
source, less grid-position dependence) next to trilinear, selectable per run.

**4.7 Analytic and semi-analytic test suite.** Kirkwood's solution for
off-centre charges in a sphere, with and without salt [Kirkwood 1934]; two
interacting spheres; the planar interface. Every discretisation must report
its **observed convergence order** on these, not only the error at one spacing.

**4.8 Second-order interface discretisation.** *Why:* with a sharp interface,
the 7-point stencil is first order and solvation energies depend on how the
surface cuts the grid. Matched interface and boundary (MIB) [Zhou, Feig &
Wei 2008; Chen et al. 2011], immersed interface [Wei, Luo & Qi 2019] and
improved harmonic-average schemes [Wei et al. 2019] reach second order, the
last at the cost and convergence behaviour of the harmonic-average method.
*Plan:* implement the improved harmonic-average scheme first (reported to
keep the convergence behaviour of the harmonic-average method, so CG and
multigrid should stay applicable; to be confirmed on our operator), then
evaluate an IIM/MIB variant, whose non-symmetric stencils need a different
Krylov solver. Interface geometry (normals, curvature,
intersections) comes from an analytic or level-set SES.
*Evidence:* second-order convergence of the potential on 4.7; Born and
Kirkwood energy errors < 0.5 % at 0.5 Å; grid-shift spread of ΔG_polar on a
protein reduced at least 3× relative to v0.0.1.

**4.9 Error estimates.** Every energy optionally comes with a numerical
uncertainty from grid-shift averaging and two-spacing extrapolation, so users
can see whether the discretisation error matters next to the statistical
error over frames.

### v0.3 — Physics

**4.10 Nonlinear PB.** Inexact Newton with the multigrid-preconditioned
linear solver, with damping for the exponential term [Holst & Saied 1995].
Energy functional and convergence tests against the linearized equation at
low potential.

**4.11 Size-modified PB.** Finite ion size via the lattice-gas (Bikerman /
Borukhov) free energy [Borukhov, Andelman & Orland 1997], uniform ion size
first. Relevant for highly charged systems (nucleic acids, membranes).

**4.12 Forces.** Reaction-field, dielectric-boundary and ionic boundary
forces on atoms [Im, Beglov & Roux 1998], using the accurate
dielectric-boundary formulation for finite-difference solvers [Cai et al.
2011] on top of the decomposition from 4.5. Energy conservation in PB
dynamics depends on this accuracy [Wang J et al. 2009].
*Evidence:* forces agree with finite differences of the energy; momentum
conservation; short NVE runs with OpenMM show bounded drift.

**4.13 Smooth dielectric models.** Gaussian-based dielectric functions,
which give a few-unit protein interior and a graded interface without an
explicit surface [Grant, Pickup & Nicholls 2001; Li et al. 2013]. They are
differentiable everywhere (good for forces and gradients) and are the basis
of Gaussian-dielectric pKa methods (4.21).

### v0.4 — End-point free energies (MM-PBSA and related)

**4.14 Nonpolar solvation.** Besides γ·SASA + β: a solvent-accessible-volume
cavity term plus an integral dispersion term, which models repulsive and
attractive parts separately [Wagoner & Baker 2006; Tan, Tan & Luo 2007]. The
coupling of polar and nonpolar terms [Dzubiella, Swanson & McCammon 2006]
stays a research topic. Parameters fitted on openly licensed data only (e.g. FreeSolv [Mobley & Guthrie 2014]).

**4.15 Per-residue and pairwise decomposition** of polar and nonpolar terms
(standard in MM-PBSA analysis tools [Miller et al. 2012; Valdés-Tresanco et al. 2021]).

**4.16 Entropy and dielectric options.** Interaction entropy from the
trajectory [Duan, Liu & Zhang 2016], which large benchmarks (summarised in
[Wang E et al. 2019]) recommend over truncated normal modes for MD-based
MM/PB(GB)SA, and
optional formula-based entropy estimates [ChemRxiv 2024 preprint]. Variable
or residue-dependent interior dielectric as an option, not a default.

**4.17 Membranes.** Implicit membrane slab with a depth-dependent dielectric,
which correlates better with experiment than a uniform slab [Greene et al.
2019], and periodic lateral boundaries to avoid edge effects. Benchmarks in
[Wang et al. 2022] show that membrane dielectric parameters must be tuned per system.

**4.18 Uncertainty and convergence over frames.** Block averaging, bootstrap
confidence intervals and correlation-time estimates in `pbsolve mmpbsa`, so a
report states how many frames were enough.

**4.19 Public binding benchmark.** Follow the community guidelines for
protein–ligand affinity benchmarks [Hahn et al. 2022] on openly licensed
sets. Report correlations with experiment and the effect of each option
(4.8, 4.13, 4.14, 4.16), plus timing. Recent work shows MM/PBSA can beat
MM/GBSA for some classes (e.g. cyclic peptides) after protocol tuning [Brief.
Bioinform. 2025], and that PBSA solvation can be combined with machine-learned
potentials for the gas-phase term [AIQM-PBSA, ChemRxiv 2026]: pbsolve should
be the solvation engine in such workflows, not a fixed protocol.

### v0.5 — Gradients and machine learning

**4.20 Adjoint gradients and ML interop.** The discrete operator is
symmetric, so the gradient of any energy with respect to charges, radii,
dielectric parameters or coordinates needs one extra solve (adjoint method).
Expose this through Python, with zero-copy GPU arrays (DLPack) for PyTorch and JAX.
*Uses:* radius and dielectric parameter fitting; training and checking ML
surrogates; physics-based loss terms.
*Context:* ML is entering PB modelling in three ways: surrogates for
energies or reaction fields [PBML, Biophys. J. 2024; PBNeF, arXiv 2024],
neural-network or physics-informed PDE solvers [Achondo, Chaudhry & Cooper
2024; Huang et al. J. Comput. Phys. 2025], and corrections that bring
continuum models closer to explicit solvent [Katzberger & Riniker 2023, 2024;
J. Chem. Phys. 164, 054107 (2026)]. Each needs a fast, deterministic,
differentiable PB reference. Machine-learned level-set surfaces [Wei, Zhao &
Luo 2021] show how surface computation can become GPU-friendly and
differentiable. pbsolve provides the solver and data pipeline; trained
models are separate, optional packages.

### v0.6+ — Further solution methods

These live behind the same Python API (`method=...`) and share the molecule,
dielectric definition, energies and validation suite.

**4.21 PB-based pKa and titration.** Site–site interaction energies from PB
[Bashford & Karplus 1990] with Monte Carlo over protonation states, with the
Gaussian dielectric (4.13) as in DelPhiPKa [Wang, Li & Alexov 2015] and in
the PB-based PypKa workflow [Reis et al. 2020]. Benchmarks show no single
fast method dominates [Wei, Hogues & Sulea 2023], so this is a research
track, compared with PROPKA, which stays an optional external tool.

**4.22 Boundary-element solver.** Surface-only unknowns, exact far field,
no grid: treecode/FMM-accelerated BEM reaches virus-scale systems [Wilson,
Geng & Krasny 2022; Wang, Cooper, Betcke & Barba 2021], and FEM–BEM coupling
is efficient for small molecules [Bosy et al. 2024]. Needs a robust SES
triangulation [Decherchi & Rocchia 2013 for the method].

**4.23 Domain-decomposition linear PB (ddLPB).** Schwarz decomposition over
atomic balls with spectral accuracy and no 3D grid [Quan, Stamm & Maday
2019], with analytic forces in linear scaling via FMM [Jha et al. 2023].
Well suited to forces in implicit-solvent dynamics.

**4.24 Adaptive grids.** Locally refined FD or octree discretisations reduce
the unknown count around the solute (cf. the adaptive octree FEM of NextGenPB
[Di Florio et al. 2025]); a later option for very large systems.

**4.25 AMD GPUs (HIP)** through the same `Backend` trait, after multigrid
fixes the kernel set.

### v1.0 — Stable release

Frozen public API with deprecation policy; documentation site with tutorials
(MM-PBSA from Amber/GROMACS/OpenMM, electrostatic maps, membranes);
conda-forge feedstock with CPU and CUDA variants; software paper with the
validation and benchmark results; versioned archive with a DOI.

## 5. Architecture changes

To host several methods without duplicating physics:

```
Problem         molecule (coords, charges, radii), dielectric model, ions, temperature
  └─ Method     FD (+ multigrid, interface scheme) | BEM | ddLPB | ...
       └─ Backend   CPU | CUDA | HIP (FD); FMM/treecode kernels (BEM)
Observables     energies, potential at points / on grids, forces, adjoint gradients
```

* `Solver(method="fd", ...)` stays the default; new methods are added behind
  the same `polar_solvation` / `potential` / `forces` calls.
* The dielectric model (sharp SES, vdW, smooth Gaussian, membrane) becomes a
  separate component that every method queries: geometric intersections for
  interface schemes, triangulations for BEM, ball sets for ddLPB.
* Validation stays method-independent: the analytic suite (4.7) and the
  benchmark sets run for every method and platform in CI (CPU) and on the GPU runner.

## 6. Validation programme

| Level | Tests | Run |
|---|---|---|
| Analytic | Born, Kirkwood (with salt), two spheres, planar interface; observed convergence order | every PR |
| Cross-code | map- and energy-level comparison with established FD and BEM codes | release |
| Physical | forces vs energy differences, energy conservation, nonlinear → linear limit | every PR (small), release (full) |
| Applied | small-molecule solvation (FreeSolv), protein–ligand binding benchmark, membrane systems | release |
| Performance | iterations vs spacing, frames per second, GPU vs CPU, memory | release, tracked over versions |

## 7. Licensing guardrails

* pbsolve stays MIT (plus the BSD notices in `THIRD_PARTY_NOTICES.md`).
  Implement methods from **papers**. Do not copy or translate code from
  copyleft or non-OSI projects; for example NextGenPB is GPL-3.0 and PROPKA
  is LGPL-2.1. Check the license of any other code before reading it.
* Permissive libraries (MIT/BSD/Apache, e.g. Bempp-cl, Exafmm) may be optional
  dependencies; copyleft libraries (e.g. FEniCSx) only as optional, separately
  installed runtime dependencies.
* Benchmark data: openly licensed sets only; check each dataset's licence
  before redistributing it in `tests/` or benchmark repositories.
* Naming: method names describe the method (e.g. "improved harmonic average",
  "MIB-type interface"), not other software. Cross-code comparisons are
  factual statements in the validation documents.

## 8. Decisions needed from the maintainer

1. **Priority between throughput (v0.1) and accuracy (v0.2):** proposed order
   is throughput first, because multigrid also makes the accuracy work cheaper.
2. **Forces and dynamics scope:** forces for minimisation and analysis only,
   or PB-based implicit-solvent MD through an OpenMM plugin (larger effort).
3. **ML direction:** gradients and data interfaces only (proposed), or also
   shipping trained surrogate models.
4. **Which second method first:** boundary elements (large systems, exact far
   field) or ddLPB (fast forces for MD).

## References

* Achondo MAP, Chaudhry JH, Cooper CD. An investigation of physics informed neural networks to solve the Poisson–Boltzmann equation in molecular electrostatics. arXiv:2410.12810 (2024).
* AIQM-PBSA: integrating machine learning interatomic potentials with MMPBSA for protein–ligand binding free energies. ChemRxiv, doi:10.26434/chemrxiv-2026-h0kn1 (2026, preprint).
* Adams M, Brezina M, Hu J, Tuminaro R. Parallel multigrid smoothing: polynomial versus Gauss–Seidel. J. Comput. Phys. 188, 593 (2003).
* Alcouffe RE, Brandt A, Dendy JE, Painter JW. The multi-grid method for the diffusion equation with strongly discontinuous coefficients. SIAM J. Sci. Stat. Comput. 2, 430 (1981).
* Bashford D, Karplus M. pKa's of ionizable groups in proteins: atomic detail from a continuum electrostatic model. Biochemistry 29, 10219 (1990).
* Borukhov I, Andelman D, Orland H. Steric effects in electrolytes: a modified Poisson–Boltzmann equation. Phys. Rev. Lett. 79, 435 (1997).
* Bosy M, Scroggs MW, Betcke T, Burman E, Cooper CD. Coupling finite and boundary element methods to solve the Poisson–Boltzmann equation for electrostatics in molecular solvation. J. Comput. Chem. (2024).
* Cai Q, Ye X, Wang J, Luo R. Dielectric boundary forces in numerical Poisson–Boltzmann methods: theory and numerical strategies. Chem. Phys. Lett. 514, 368 (2011).
* Chen D, Chen Z, Chen C, Geng W, Wei GW. MIBPB: a software package for electrostatic analysis. J. Comput. Chem. 32, 756 (2011).
* Chen et al. Poisson–Boltzmann-based machine learning model for electrostatic analysis. Biophys. J. (2024).
* Chern IL, Liu JG, Wang WC. Accurate evaluation of electrostatics for macromolecules in solution. Methods Appl. Anal. 10, 309 (2003).
* Decherchi S, Rocchia W. A general and robust ray-casting-based algorithm for triangulating surfaces at the nanoscale. PLoS One 8, e59744 (2013).
* Di Florio V, Ansalone P, Siryk SV, Decherchi S, de Falco C, Rocchia W. NextGenPB: an analytically-enhanced finite element solver for the linearized Poisson–Boltzmann equation. Comput. Phys. Commun. 317, 109816 (2025).
* Duan L, Liu X, Zhang JZH. Interaction entropy: a new paradigm for highly efficient and reliable computation of protein–ligand binding free energy. J. Am. Chem. Soc. 138, 5722 (2016).
* Dzubiella J, Swanson JMJ, McCammon JA. Coupling hydrophobicity, dispersion, and electrostatics in continuum solvent models. Phys. Rev. Lett. 96, 087802 (2006).
* Geng W, Yu S, Wei GW. Treatment of charge singularities in implicit solvent models. J. Chem. Phys. 127, 114106 (2007).
* Genheden S, Ryde U. The MM/PBSA and MM/GBSA methods to estimate ligand-binding affinities. Expert Opin. Drug Discov. 10, 449 (2015).
* Golub GH, Ye Q. Inexact preconditioned conjugate gradient method with inner-outer iteration. SIAM J. Sci. Comput. 21, 1305 (1999).
* Grant JA, Pickup BT, Nicholls A. A smooth permittivity function for Poisson–Boltzmann solvation methods. J. Comput. Chem. 22, 608 (2001).
* Greene D, Qi R, Nguyen R, Qiu T, Luo R. Heterogeneous dielectric implicit membrane model for the calculation of MMPBSA binding free energies. J. Chem. Inf. Model. 59, 3041 (2019).
* Hahn DF, et al. Best practices for constructing, preparing, and evaluating protein–ligand binding affinity benchmarks. Living J. Comp. Mol. Sci. 4, 1497 (2022).
* Holst M, Saied F. Multigrid solution of the Poisson–Boltzmann equation. J. Comput. Chem. 14, 105 (1993).
* Holst MJ, Saied F. Numerical solution of the nonlinear Poisson–Boltzmann equation: developing more robust and efficient methods. J. Comput. Chem. 16, 337 (1995).
* Huang et al. Neural network Poisson–Boltzmann electrostatics for biomolecular interactions. J. Comput. Phys. (2025); preprint HKU IMR2025-01.
* Im W, Beglov D, Roux B. Continuum solvation model: computation of electrostatic forces from numerical solutions to the Poisson–Boltzmann equation. Comput. Phys. Commun. 111, 59 (1998).
* Jha A, Nottoli M, Mikhalev A, Quan C, Stamm B. Linear scaling computation of forces for the domain-decomposition linear Poisson–Boltzmann method. J. Chem. Phys. 158, 104105 (2023).
* Katzberger P, Riniker S. Implicit solvent approach based on generalized Born and transferable graph neural networks for molecular dynamics simulations. J. Chem. Phys. 158, 204101 (2023); A general graph neural network based implicit solvation model for organic molecules in water. Chem. Sci. (2024).
* Kirkwood JG. Theory of solutions of molecules containing widely separated charges with special application to zwitterions. J. Chem. Phys. 2, 351 (1934).
* Li L, Li C, Zhang Z, Alexov E. On the dielectric "constant" of proteins: smooth dielectric function for macromolecular modeling and its implementation in DelPhi. J. Chem. Theory Comput. 9, 2126 (2013).
* McCormick SF, Benzaken J, Tamstorf R. Algebraic error analysis for mixed-precision multigrid solvers. SIAM J. Sci. Comput. 43, S392 (2021a).
* Miller BR, McGee TD, Swails JM, Homeyer N, Gohlke H, Roitberg AE. MMPBSA.py: an efficient program for end-state free energy calculations. J. Chem. Theory Comput. 8, 3314 (2012).
* Mobley DL, Guthrie JP. FreeSolv: a database of experimental and calculated hydration free energies, with input files. J. Comput. Aided Mol. Des. 28, 711 (2014).
* Notay Y. Flexible conjugate gradients. SIAM J. Sci. Comput. 22, 1444 (2000).
* Qi R, Botello-Smith WM, Luo R. Acceleration of linear finite-difference Poisson–Boltzmann methods on graphics processing units. J. Chem. Theory Comput. 13, 3378 (2017).
* Qi R, Luo R. Robustness and efficiency of Poisson–Boltzmann modeling on graphics processing units. J. Chem. Inf. Model. 59, 409 (2019).
* Quan C, Stamm B, Maday Y. A domain decomposition method for the Poisson–Boltzmann solvation models. SIAM J. Sci. Comput. 41, B320 (2019).
* Reis PBPS, Vila-Viçosa D, Rocchia W, Machuqueiro M. PypKa: a flexible Python module for Poisson–Boltzmann-based pKa calculations. J. Chem. Inf. Model. 60, 4442 (2020).
* Tamstorf R, Benzaken J, McCormick SF. Discretization-error-accurate mixed-precision multigrid solvers. SIAM J. Sci. Comput. 43, S420 (2021b).
* Tan C, Tan YH, Luo R. Implicit nonpolar solvent models. J. Phys. Chem. B 111, 12263 (2007).
* Tatebe O. The multigrid preconditioned conjugate gradient method. Proc. 6th Copper Mountain Conference on Multigrid Methods, NASA CP-3224, 621 (1993).
* Trefethen LN. Approximation Theory and Approximation Practice, extended edition. SIAM (2019).
* Valdés-Tresanco MS, Valdés-Tresanco ME, Valiente PA, Moreno E. gmx_MMPBSA: a new tool to perform end-state free energy calculations with GROMACS. J. Chem. Theory Comput. 17, 6281 (2021).
* Wagoner JA, Baker NA. Assessing implicit models for nonpolar mean solvation forces: the importance of dispersion and volume terms. Proc. Natl. Acad. Sci. USA 103, 8331 (2006).
* Wang E, Sun H, Wang J, Wang Z, Liu H, Zhang JZH, Hou T. End-point binding free energy calculation with MM/PBSA and MM/GBSA: strategies and applications in drug design. Chem. Rev. 119, 9478 (2019).
* Wang J, Cai Q, Li ZL, Zhao HK, Luo R. Achieving energy conservation in Poisson–Boltzmann molecular dynamics: accuracy and precision with finite-difference algorithms. Chem. Phys. Lett. 468, 112 (2009).
* Wang L, Li L, Alexov E. pKa predictions for proteins, RNAs, and DNAs with the Gaussian dielectric function using DelPhi pKa. Proteins 83, 2186 (2015).
* Wang L, Krasny R, Tlupova S. A kernel-independent treecode based on barycentric Lagrange interpolation. Commun. Comput. Phys. 28, 1415 (2020).
* Wang T, Cooper CD, Betcke T, Barba LA. High-productivity, high-performance workflow for virus-scale electrostatic simulations with Bempp-Exafmm. arXiv:2103.01048 (2021).
* Wang et al. MM/PB(GB)SA benchmarks on soluble proteins and membrane proteins. Front. Pharmacol. 13, 1018351 (2022).
* Wei H, Luo A, Qiu T, Luo R, Qi R. Improved Poisson–Boltzmann methods for high-performance computing. J. Chem. Theory Comput. 15, 6190 (2019).
* Wei H, Luo R, Qi R. An efficient second-order Poisson–Boltzmann method. J. Comput. Chem. 40, 1257 (2019).
* Wei H, Zhao Z, Luo R. Machine-learned molecular surface and its application to implicit solvent simulations. J. Chem. Theory Comput. 17, 6214 (2021).
* Wei W, Hogues H, Sulea T. Comparative performance of high-throughput methods for protein pKa predictions. J. Chem. Inf. Model. (2023), doi:10.1021/acs.jcim.3c00165.
* Wilson L, Geng W, Krasny R. TABI-PB 2.0: an improved version of the treecode-accelerated boundary integral Poisson–Boltzmann solver. J. Phys. Chem. B 126, 7104 (2022).
* Wu et al. End-to-end reaction field energy modeling via deep learning based voxel-to-voxel transform (PBNeF). arXiv:2410.03927 (2024).
* Zhou YC, Feig M, Wei GW. Highly accurate biomolecular electrostatics in continuum dielectric environments. J. Comput. Chem. 29, 87 (2008).
* Achieving all-atom molecular dynamics accuracy from the Poisson–Boltzmann method through machine learning. J. Chem. Phys. 164, 054107 (2026).
* Improving the predictive performance of binding affinities and poses for protein–cyclic peptide complexes through fine-tuned MM/PBSA(GBSA)-based methods. Brief. Bioinform. 26, bbaf632 (2025).
* Enhancing MM/P(G)BSA methods: integration of formulaic entropy. ChemRxiv, doi:10.26434/chemrxiv-2024-9308m (2024, preprint).

This list was compiled from abstracts and bibliographic indexes (November
2026); volume and page numbers have not been checked
against the publisher records. Verify each entry before citing it in a paper.
