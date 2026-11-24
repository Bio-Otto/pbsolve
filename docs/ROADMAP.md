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
| First public release | Release | public repository and documentation site, PyPI wheels and the conda-forge recipe, speed on an idle machine and several GPUs, cross-check of the Gaussian dielectric, the logo |
| v0.4 | End-point free energies | nonpolar models, per-residue decomposition, entropy options, membranes, uncertainty, public benchmark |
| v0.5 | Gradients and ML | adjoint gradients, PyTorch/JAX interop, dataset generation, Δ-learning hooks |
| v0.6+ | New solution methods | boundary-element (FMM/treecode), domain-decomposition LPB, PB-based pKa, AMD/HIP |
| v1.0 | Stable release | frozen API, documentation site, software paper, conda-forge |

The order follows dependencies: multigrid makes fine grids affordable, which
the accuracy work needs; the decomposed formulation from v0.2 is the basis
for forces (v0.3) and gradients (v0.5). v0.1–v0.3 are done; the first public
release is the next milestone besides v0.4 (its timing is decision 5).

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
*Status: done (0.0.2).* Chebyshev smoothing [Adams et al. 2003] instead
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
*Status: done (0.0.2).*
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
*Status: done (0.0.2).*
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
*Status: done (0.0.2); the default since the robustness sweep below.*
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

**Host stages** (follow-up of 4.4). *Status: done (0.0.2).*
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
* Both fusions measured on an idle GPU (done): solve stage 1.21–1.26×,
  calculation 1.09–1.12×, energies bit-identical. With the default `mdh`
  boundary the host's sum (5DK3 ~2.2 s of 4.2 s) is now the largest stage
  of a CUDA calculation; `mdh-fast` avoids it.

### v0.2 — Accuracy

**4.5 Reaction-field decomposition.** Split the potential into the singular
Coulomb part (analytic) and a smooth reaction-field part that is solved on
the grid [Chern, Liu & Wang 2003; Geng, Yu & Wei 2007]. This removes the
grid self-energy, so the reference solve is no longer needed, and accuracy
improves at coarse spacing. It is also the formulation that interface methods
and forces build on. The current "same grid" reference-state method stays
available and is the default until the new path is validated.
*Status: done (0.0.2) as `coulomb="analytic"`.*
* u = v + χG with G the Coulomb potential in `pdie` and χ the nodes within
  max(1 Å, 2h) of an atom. The substitution is exact; the one approximation
  replaces the charges inside χ by the discrete source of G. The right-hand
  side then lives on a thin layer, and the solvated operator and solvers are
  reused unchanged. One solve per focusing level, only the finest level
  decomposed.
* Related work: the reference calculation was removed by Luty et al. (1992)
  and Zhou et al. (1996). Chern et al. (2003), Chen et al. (2007), Geng et
  al. (2007), Cai et al. (2009) and Holst et al. (2012) split the potential
  into a singular and a regular part. Their regular part starts at the
  dielectric boundary; here χ extends beyond the atoms, because an edge
  close to a charge costs accuracy (docs/VALIDATION.md).
* With the fractional interface the placement spread halves (analytic
  references and 1YCR), and 1YCR's ΔG_polar at 0.5 Å moves toward its limit.
  With the sharp interface nothing changes beyond 0.16 points.
* Cost: 1.7× faster on the CPU, 7–20 % faster on CUDA for 1YCR. G is a
  direct sum, so 5DK3 on CUDA takes 5.75 instead of 4.30 s; a treecode is
  the follow-up.

**4.6 Charge assignment.** B-spline spreading (smoother source, less
grid-position dependence) next to trilinear, selectable per run.
*Status: done (0.0.2); the expected gain did not appear.*
* `chgm="spl2"` and `"spl4"`: the order-3 and order-5 B-splines (3 and 5
  nodes per axis) that APBS calls cubic and quintic, with APBS' weights and
  energy rule. Maps and energies agree with APBS 3.4.1 as for `spl0`.
* They change ΔG_polar by 0.2–0.3 %, the analytic errors by ≤ 0.3 points, and
  the placement spread of 1YCR by ≤ 5 % (sd 4.1 → 3.9 kcal/mol at 0.5 Å).
  The error comes from the sharp dielectric boundary, and the spread
  charge's self-energy cancels between the two states. Grid-position
  dependence is therefore left to 4.8 and 4.9.

**4.7 Analytic and semi-analytic test suite.** Kirkwood's solution for
off-centre charges in a sphere, with and without salt [Kirkwood 1934]; two
interacting spheres; the planar interface. Every discretisation must report
its **observed convergence order** on these, not only the error at one spacing.
*Status: Kirkwood sphere and two spheres done (0.0.2); the planar
interface waits for a geometric surface input.*
* `pbsolve.analytic.kirkwood_energy`: charges anywhere in a sphere, with salt
  beyond a Stern layer; checked against closed forms (Born, Born–Debye–Hückel,
  grounded-sphere images) and the interface conditions to ≤ 1e-12.
* `scripts/validate_analytic.py`: error at 0.5–0.2 Å, averaged over grid
  placements, and the observed order: 1.09–1.18 for the current
  discretisation (first order). Charges 1 Å below the surface scatter by up
  to 11 % per placement at 0.5 Å, the baseline for 4.8 and 4.9.
* `pbsolve.analytic.two_sphere_energy`: two spheres coupled by multipole
  re-expansion, with salt, checked against exact limits. The complex is first
  order like one sphere. The association energy on one grid is within
  0.002 kcal/mol with `vdw`; with `mol`, APBS' sampled probe positions add
  0.011–0.019 kcal/mol at every spacing.
* Planar interface: pbsolve's input is atoms, so a dielectric half-space can
  only be set up through hand-made maps (`Solver.solve_maps`). That would
  test the operator but not the surface construction. Large-sphere charges
  near the surface already probe the flat limit. The test follows when a
  surface can be given as geometry (level set, 4.8) or as a membrane slab
  (v0.4).

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
*Status: done (0.0.2) as `interface="fractional"`; the other schemes
were not needed.*
* Faces cut by the surface get the harmonic average of `pdie` and `sdie`
  weighted by the exact lengths of their grid edge inside and outside the
  solute [Davis & McCammon 1991; Wang et al. 2012], computed from the
  spheres of the surface. The operator stays symmetric and the multigrid CG
  unchanged; opt-in, the APBS-compatible `sharp` stays the default.
* Analytic references with `vdw`: observed orders 1.6–2.5 on the energies
  (sharp 1.0–1.2), errors at 0.5 Å −0.14 to −0.71 % (sharp −2.3 to
  −4.7 %); the Born ion (−0.51 %) and charges 1 Å below the surface
  (−0.71 %) narrowly miss the 0.5 % target. With `mol` the energies level
  off 0.15 % above the spheres' because of APBS' sampled probe positions.
* 1YCR: placement sd 5–10 times smaller, the same zero-spacing limit as
  sharp, ΔΔG_bind bias at 0.5 Å +2.7 instead of +13 kcal/mol. Calculations
  take 1.5 times longer on CUDA (maps on the host), the same on the CPU.
* Not done: the potential's order, a smoothed ion map, and the
  improved harmonic-average, IIM and MIB schemes.

**4.9 Error estimates.** Every energy optionally comes with a numerical
uncertainty from grid-shift averaging and two-spacing extrapolation, so users
can see whether the discretisation error matters next to the statistical
error over frames.
*Status: done (0.0.2).*
* `pbsolve.estimate_error` and `pbsolve solvation --placements
  --extrapolate`: mean, sd and sem over grid placements, and the means at
  two spacings extrapolated to zero spacing at first order. `parts` covers
  MM-PBSA association energies.
* On the analytic references, extrapolating from 0.35 and 0.25 Å leaves
  ≤ 0.3 %, and the estimated error is within 30 % of the true one. From
  0.5/0.35 Å it is off by up to 2×.
* 1YCR: ΔG_polar changes by 3.6 % from 0.48 to 0.25 Å (estimated error at
  0.25 Å: 3 %). ΔΔG_polar of binding carries ~+6 kcal/mol at 0.25 Å and
  ~+13 at 0.5 Å, far more than its placement noise. That is the case for
  4.8.

### v0.3 — Physics

**4.10 Nonlinear PB.** Inexact Newton with the multigrid-preconditioned
linear solver, with damping for the exponential term [Holst & Saied 1995].
Energy functional and convergence tests against the linearized equation at
low potential.
*Status: done (0.0.2) as `equation="npbe"`, APBS `npbe`.*
* Newton from the linearized solution; each step is the linear solve with
  κ̄² replaced by N'(u), inexact [Dembo et al. 1982; Holst et al. 1994]
  and damped by halving until ‖F‖ decreases. Default tolerances: 15 instead
  of 12 CG iterations per level on 1YCR.
* Energy: APBS' functional [Sharp & Honig 1990] with the coarser levels'
  outer terms. APBS 3.4.1 `npbe` is matched to ≤ 2.3e-9 (solvated energies
  in APBS units, up to a −30 e sphere, 1:1 and 2:1 salts, 1–3 levels).
* Low-potential test: the potential equals the linearized one to O(q²); the
  energies differ by the functional's truncation at the boxes (field energy
  beyond them), which APBS shares. The Sharp–Honig volume form converges
  faster with the box and could be offered as an option.
* Cost 1.3–1.4× (CPU), 1.7–1.9× (CUDA) a linear calculation. The iteration
  runs on the device and starts from a Gouy–Chapman-damped linearized
  potential (5DK3 on CUDA: 12.7 → 7.2 s). Rebuilding the multigrid
  hierarchy every step remains.

**4.11 Size-modified PB.** Finite ion size via the lattice-gas (Bikerman /
Borukhov) free energy [Borukhov, Andelman & Orland 1997], uniform ion size
first. Relevant for highly charged systems (nucleic acids, membranes).
*Status: done (0.0.2) as `equation="smpbe"` with `ion_size`, uniform
size.*
* The lattice denominator D = 1 − Σφ + Σφ e^{−qu} enters the ion term, its
  slope and the energy (ln D / N_A a³), on the `npbe` Newton path and its
  device kernels. For a = 0 the code is `npbe` bit for bit.
* APBS 3.4.1 disables `smpbe` ("numerical stability issues"), so the
  references are analytic and from the literature:
  * the planar double layer's exact profile (Gouy–Chapman for point ions,
    the lattice-gas first integral [Kilic et al. 2007]): orders 2.6–3.8;
  * Silalahi et al. (2010)'s a³ law for the entropic cost: slope 3.00;
  * saturation at one ion per cell where point ions reach 84 M.
* Different sizes per species [Chu et al. 2007; Boschitsch & Danilov 2012]
  remain.

**4.12 Forces.** Reaction-field, dielectric-boundary and ionic boundary
forces on atoms [Im, Beglov & Roux 1998], using the accurate
dielectric-boundary formulation for finite-difference solvers [Cai et al.
2011] on top of the decomposition from 4.5. Energy conservation in PB
dynamics depends on this accuracy [Wang J et al. 2009].
*Evidence:* forces agree with finite differences of the energy; momentum
conservation; short NVE runs with OpenMM show bounded drift.
*Status: done APBS-style (0.0.2) as `Solver.polar_forces`, with APBS'
spline surfaces `surface="spl2"`/`"spl4"`; the rest remains.*
* The spline surfaces [Im, Beglov & Roux 1998; Schnieders et al. 2007] give
  maps equal to APBS' to the 7 digits of its DX files.
* The forces are the explicit derivative of the free-energy functional at the
  converged finest-level potential [Gilson et al. 1993], as APBS `calcforce`:
  * charge–field, dielectric-boundary and ionic-boundary parts, for `lpbe`,
    `npbe` and `smpbe`;
  * on one grid with zero boundary values the exact gradient of the discrete
    energy (finite differences to 4e-9);
  * APBS 3.4.1 to ≤ 4e-7 of the largest force.
* Momentum: a charge in a dielectric sphere is pushed with Kirkwood's force
  (0.08 %, at the surface's Born radius) and the sphere back with it (1 %).
* Remaining:
  * the dielectric-boundary force is noisy over grid placements (5–10 %),
    which the formulation of Cai et al. (2011) addresses;
  * forces with the decomposition of 4.5;
  * NVE runs with OpenMM.

**4.13 Smooth dielectric models.** Gaussian-based dielectric functions,
which give a few-unit protein interior and a graded interface without an
explicit surface [Grant, Pickup & Nicholls 2001; Li et al. 2013]. They are
differentiable everywhere (good for forces and gradients) and are the basis
of Gaussian-dielectric pKa methods (4.21).
*Status: done (0.0.2) as `surface="gauss"` with `gauss_sigma` and
`gauss_contour`; forces remain.*
* The dielectric of Li et al. (2013) and the ions' Born penalty of Jia et al.
  (2017); maps equal the formulas to 1e-9.
* The reference state keeps the Gaussian ε inside a contour (ε ≤ 20) [Li, Li
  & Alexov 2014]. A homogeneous `pdie` reference leaves the charges' grid
  self-energies uncancelled (1YCR: −39 000 kcal/mol at 0.5 Å, −61 000 at
  0.25 Å); with the contour, one Gaussian atom is within 1–3 % of its exact
  radial solution at 0.5 Å and ΔΔG_polar of 1YCR within 3 kcal/mol between
  0.5 and 0.25 Å.
* Remaining: forces (the sharp contour of the reference state makes them
  undefined; a smoothed contour would be a new model), and `npbe`, whose
  Boltzmann density diverges at the charges under the bounded Born penalty.

### First public release

v0.1–v0.3 are done; the repository is public since 2026-11-15, and the latest
tag is v0.1.9. The first release makes the work usable and checkable by
others. Releases are vX.Y.0 tags that pass CI (decision 5, section 8).

**R1 Public repository and documentation site.** The repository is public
(done, 2026-11-15), and Read the Docs builds the site from every push and tag
(https://pbsolve.readthedocs.io, `latest` and `stable`). With v0.1.0: link
it from the README, add install instructions and versioned documentation;
switch the site's theme from Furo to the Wagtail theme
(`sphinx-wagtail-theme`, MIT; sample site
https://sphinx-themes.org/sample-sites/sphinx-wagtail-theme/), pinned like
the other documentation tools; tag the release and archive it with a DOI.
*Evidence:* the site builds from the tag with `-W`, and every guide example
runs.
*Status (v0.1.0):* done except the DOI. The README links the site; a
release installs with pip from its tag (checked in clean environments, CPU
and CUDA builds); the site has a version per release (`stable` = the latest
release); the theme is `sphinx-wagtail-theme` 6.6.0 with the pbsolve logo;
CI passes on Linux, macOS and Windows. The DOI needs the repository enabled
on Zenodo before a release is created.

**R2 Packages.** PyPI wheels besides the conda-forge recipe
(`docs/PACKAGING.md`): Linux x86_64 first, a CPU build and CUDA builds,
which load the driver and NVRTC at runtime and run on the CPU without them.
How pip users choose the CUDA major (separate package names or extras) is
part of the work.
*Evidence:* `pip install` in a clean environment passes `pbsolve test` on a
machine without a GPU and on one with a GPU.
*Status (0.1.1): the wheels are done; the first upload waits for the
maintainer and v0.2.0.*
* One `pbsolve` wheel per CPython version (3.10–3.14), for Linux x86_64 and
  manylinux2014, built with `cuda-12`; plus the sdist. The `cuda12` extra
  adds pip's NVRTC.
* Why not separate package names: one name keeps a single dependency for
  other packages. A `cuda-12` build runs on CUDA 12 and 13 drivers and
  covers sm_50–sm_121 (NVRTC 13 starts at sm_75).
* Checked in clean environments: with the GPU, with no visible device, with
  the driver library hidden, and from the sdist. The energies are
  bit-identical to the development build.
* CI builds the wheels in the manylinux container and tests them on a
  GPU-less runner. A vX.Y.0 release attaches them and, once the maintainer
  has registered the trusted publisher, uploads them to PyPI
  (`docs/PACKAGING.md`).
* *0.1.2:* wheels for Linux aarch64 and Windows x86_64 (`cuda-12`) and for
  macOS arm64 and x86_64 (CPU), each built on its own runner and installed
  on a GPU-less runner of the same platform by CI.
* *0.1.3:* the conda recipe builds, checked locally with conda-forge's pinning
  and linter: CPU, CUDA 12.9 and CUDA 13.4 variants pass their tests, one
  `conda install` picks the CUDA variant with a driver and the CPU variant
  without, and the energies are bit-identical to the wheels. `conda-forge.yml`
  makes the version bot follow GitHub releases (vX.Y.0) only.
* *0.1.4:* the recipe moved to the v1 format (`recipe.yaml`, rattler-build),
  which staged-recipes asks for. Its three Linux builds pass locally from the
  v0.1.0 tarball, and v0.1.0 is submitted to conda-forge/staged-recipes
  (#35780). Its CI passes on linux-64 (CPU, CUDA 12.9, 13), osx-64 and win-64
  for Python 3.11–3.14, once the recipe declares the `pbsolve` entry point
  (Windows); the PR waits for review.
* Remaining:
  * the first upload (v0.2.0, by the release cadence);
  * GPU runs of the aarch64 and Windows wheels (no such GPU at hand);
  * the staged-recipes PR (the maintainer's account) and the feedstock.

**R3 Speed on reference hardware.** A first measurement on a shared machine
(2026-11-15; GPU 94 % busy with other jobs): 1YCR, 0.15 M, 129³ with two
focusing levels takes 8.96 s with APBS 3.4.1, 2.52 s with pbsolve on 18 CPU
threads and 0.48 s on an RTX 2080 SUPER. Repeat this on an idle machine for
several system sizes, with matched and unmatched CPU thread counts, and on at
least two further NVIDIA GPU generations (consumer and data-centre).
*Evidence:* tables in `docs/VALIDATION.md` with hardware, driver and load;
CPU = CUDA energies on every GPU.

**R4 Cross-check of the Gaussian dielectric (4.13) with pyDelPhi.** pyDelPhi
implements DelPhi's Gaussian dielectric and salt model; it is AGPL-3.0, so
it runs only as a separate program and only its outputs are compared
(section 7). Compare dielectric and ion maps and solvated-state energies on
shared grids with the same σ, interior dielectric and radii, and solvation
energies with pdie = 1, where pbsolve's contour reference coincides with Li
et al.'s (vacuum beyond the contour).
*Evidence:* every difference in the maps explained to the last digit or by a
stated difference in definition (e.g. DelPhi's cutoff at 3σ, the sign of the
Born penalty).
*Status (0.1.5): done.* `scripts/compare_pydelphi.py` runs pyDelPhi 0.2.0
[Panday, Zhao & Alexov 2026] as a separate program on shared grids
(synthetic molecule, 1YCR; pdie 1; 0 and 0.15 M); `tests/test_pydelphi.py`
repeats it when pyDelPhi is installed.
* Every face of both runs equals pyDelPhi's maps to their 6 digits once its
  four definitions are applied. The definitions:
  * the Gaussians are cut at max(3σR_max, R_max + r_ion) + √3/2·h (at a
    face, also from the node below);
  * faces of density < 0.02 take the outer ε;
  * the six faces around a node are averaged where they differ by < 2 %;
  * the vacuum contour is drawn on the density.
* Ions: pyDelPhi uses the Born penalty with pbsolve's sign (keeping ions
  out). The difference is its Debye constant (+0.082 % length).
* Boundary and constants: a screened-Coulomb boundary without (1+κa), and
  CODATA 2006 constants (1.83e-7).
* Energies: with all of these, every pyDelPhi energy to its last printed
  digit. Without them, ΔG_polar of 1YCR differs by 5e-6 and its salt effect
  by 1.0 % (docs/VALIDATION.md, "Cross-check with pyDelPhi").

**R5 Logo.** One logo, after the maintainer's design, everywhere a logo
appears: the README banner on GitHub, the documentation site's header and
its icon. A dark indigo banner shows the protein (blue, with its ribbon
cartoon in white) in a shell of ions and water (orange) and the ligand
(orange, two aromatic rings in white) in its own shell (blue), next to the
name "PBsolve" and the tagline "MMPBSA Analysis Suite". Vector graphics with the text as outlines, so that
it looks the same in every browser.
*Evidence:* it renders at full size, at phone width and as a 16 px icon, on
GitHub's light and dark themes and in the site's header.
*Status (0.1.7): done.* `docs/_static/pbsolve-logo.svg` (banner, 930 × 217),
`pbsolve-logo-header.svg` (transparent, for the header, shown 52 px high)
and `pbsolve-mark.svg` (icon). The text is Liberation Sans (SIL Open Font
License 1.1) converted to outlines. After 0.1.7 the cartoon and the rings
became white: drawn dark on the blue and orange discs, they did not read in
the site's header and in the icon.

### v0.4 — End-point free energies (MM-PBSA and related)

**4.14 Nonpolar solvation.** Besides γ·SASA + β: a solvent-accessible-volume
cavity term plus an integral dispersion term, which models repulsive and
attractive parts separately [Wagoner & Baker 2006; Tan, Tan & Luo 2007]. The
coupling of polar and nonpolar terms [Dzubiella, Swanson & McCammon 2006]
stays a research topic. Parameters fitted on openly licensed data only (e.g. FreeSolv [Mobley & Guthrie 2014]).
*Status: the model of Wagoner & Baker done (0.0.2) as
`pbsolve.apolar_energy`, as APBS `apolar`; the rest remains.*
* Surface area, volume and the WCA dispersion integral follow APBS 3.4.1
  operation for operation: APBS' 11 alkane examples to every printed digit
  (13), APBS on a protein (1YCR) to 1e-13; 17 times faster than APBS there.
* Lennard-Jones parameters from OpenMM systems (and so Amber prmtop files)
  and APBS parameter files.
* Remaining: the dispersion model of Tan, Tan & Luo (2007), the non-polar
  model in `pbsolve mmpbsa` and the trajectory functions, and parameters
  fitted for proteins and ligands (FreeSolv).

**4.15 Per-residue and pairwise decomposition** of polar and nonpolar terms
(standard in MM-PBSA analysis tools [Miller et al. 2012; Valdés-Tresanco et al. 2021]).
*Status: done (per-atom and per-residue 0.0.3, pairwise 0.1.6).*
* Every polar energy comes per atom, ½ q_i φ_rf(r_i) with the energy's charge
  weights (`PolarResult.atom_energies`), and sums to ΔG_polar; the binding
  terms per atom and residue (`PBSAFrame.atom_terms`, `residue_terms`,
  `pbsolve mmpbsa --per-residue`) [Gohlke, Kiel & Case 2003].
* The shares are the linear response: E_i of all charges equals E_i of
  charge i plus its pair terms (single and pair solves, to 1e-8), and with one
  level and zero boundary values the pair terms are symmetric. APBS
  `calcenergy comps` agrees to 3e-8 (`bcfl zero`) on 1YCR.
* *0.1.6:* pairwise residue terms, `residue_pair_terms` and
  `pbsa_pair_terms` (`pbsolve mmpbsa --pairwise`): one calculation per
  charged residue with its charges reversed gives that residue's field by
  linearity; the matrix sums to ΔG_polar and its rows to the residue shares,
  and it equals single and pair calculations to 4e-8 kcal/mol. The non-polar
  terms of `apolar_energy` per residue: `ApolarEnergy.residue_terms` (p V
  belongs to no residue).

**4.16 Entropy and dielectric options.** Interaction entropy from the
trajectory [Duan, Liu & Zhang 2016], which large benchmarks (summarised in
[Wang E et al. 2019]) recommend over truncated normal modes for MD-based
MM/PB(GB)SA, and
optional formula-based entropy estimates [Dong, Li & Wang 2025]. Variable
or residue-dependent interior dielectric as an option, not a default.
*Status: done. Interaction entropy (0.0.4), atom and residue dielectrics
(0.0.5); the formula-based estimate and PB-fitted residue dielectrics were
dropped (last point).*
* `pbsolve.interaction_entropy` gives IE and its second-order cumulant C2,
  σ, standard errors over blocks of frames and `reliable` (σ ≤ 15 kJ/mol), as
  [Ekberg & Ryde 2021] recommend; `pbsolve.interaction_energy` the Coulomb +
  Lorentz–Berthelot Lennard-Jones energy between receptor and ligand (equal to
  OpenMM to 1.7e-12 on ff14SB peptides); `examples/mmpbsa_openmm.py` reports
  both. Closed forms and the normal-energy limits in docs/VALIDATION.md.
* `dielectrics=` gives every atom its own interior dielectric (DelPhi's
  multiple dielectrics [Rocchia, Alexov & Honig 2001]; amino-acid specific
  values [Wang, Zhang, Rocchia & Alexov 2013]): interior faces take the
  value of the atom whose power cell holds them, the reference state keeps
  the interior and has `pdie` outside. `pbsolve.residue_dielectrics` sets
  values for charged and other residues (fitted for MM/GBSA: 2.7 and 1.1
  [Liu, Peng & Zhang 2019]), `pbsolve mmpbsa --residue-dielectric`, and
  `PBSAFrame.delta_reference` gives the matching receptor–ligand
  interaction. Uniform values are bit-identical to `pdie`; one and two
  spheres with their own interiors agree with the exact solutions (two
  spheres 0.3–0.7 % at 0.25 Å).
* Dropped: the formulaic entropy of [Dong, Li & Wang 2025]. The authors'
  scripts (github.com/LinaDongXMU/MM_S, MIT) compute −0.1152 ΔSASA_apolar +
  0.0304 ΔSASA_polar − 1.76 N_rb + 0.414 N_atoms, but they take both ΔSASA
  terms as P2 − P2dl − P2dp of deltaVinaXGB's buried-area features. P2 is
  the sum of the other two (deltaVinaXGB's own example: 484.96 − 312.62 −
  172.33 = 0.01), so with those scripts the entropy is the rotatable-bond
  and atom-count terms only. A term like that is better fitted and checked
  against experiment in the binding benchmark (4.19). I found no residue
  dielectrics fitted for PB. The "two-valued" interior dielectric of
  [Akkus et al. 2023] is ε = 1 (or 2) for the Coulomb term and 2–4 for the
  polar term, which an MM-PBSA run with pbsolve already does (ΔE_MM from the
  force field, `pdie` for PB).

**4.17 Membranes.** Implicit membrane slab with a depth-dependent dielectric,
which correlates better with experiment than a uniform slab [Greene et al.
2019], and periodic lateral boundaries to avoid edge effects. Benchmarks in
[Wang et al. 2022] show that membrane dielectric parameters must be tuned per system.
*Status: done. Membrane maps (0.0.6), periodic lateral boundaries (0.0.7),
the membrane potential and a membrane-aware non-polar term (0.0.9).*
* `zmem`, `lmem`, `mdie`: the slab of APBS' `draw_membrane2` [Callenberg et
  al. 2010] (non-solute faces get the membrane's ε, nodes lose their ions,
  optional exclusion cone `membrane_exclusion`); the maps are identical to
  `draw_membrane2`'s output. `membrane_profile="greene2019"` or user (depth,
  ε) points (PCHIP [Fritsch & Carlson 1980]); `membrane_fill` keeps pores
  enclosed by the protein as water. A charge in the membrane agrees with the
  exact layered solution (images [Parsegian 1969] and a Hankel-transform
  reference for the profile) to 0.2 %.
* `period=(Lx, Ly)`: the solvated state is a periodic array in x and y
  (as Amber recommends for membranes [Xiao et al. 2017]; lateral periodic
  model [Ji et al. 2018]). The outermost grid spans one period and is
  periodic, its z faces take the periodic Debye–Hückel sum (Fourier
  series), finer grids focus, the reference state stays one molecule.
  Operator, multigrid (Galerkin in unwrapped coordinates) and CUDA kernels
  wrap around the seam; CUDA equals the CPU bit for bit, the default path
  is unchanged. A periodic cell equals the limit of explicit supercells
  (4e-9 in water) and the exact lattice sum of a charge in a slab (3e-4);
  the automatic plain box was 2 % off for the APBS helix (edge effect).
* `membrane_potential=V` (mV): the bath below the membrane at V [Roux
  1997], as APBSmem's `memv`: u = u_q + φ_mp, φ_mp solved without the
  charges with the source κ̄²vΘ (lower bath) and the exact column solution
  on the outer faces; ΔG gains Σ q_i φ_mp(r_i) (the capacitive ½CV² is left
  out, negligible after Roux). Linear equation only. The field matches the
  exact uniform slab (V/2 in the middle to 2e-9 V, baths 1e-4 V, interior
  0.5 % from the cell placement of the faces); CPU = CUDA, also periodic.
* `sasa`/`nonpolar_energy(zmem=, lmem=, membrane_switch=)`,
  `pbsa_solvation_frame(s)(nonpolar_membrane=)`, `--nonpolar-membrane`:
  the SASA points weighted by GBSW's membrane switch [Im, Feig & Brooks
  2003] (0 in the slab, 1 in water, cubic over ±w, w = 0.3 Å), so the area
  buried in the membrane no longer counts. The weighted area of a sphere
  equals 2πR∫H dz; a membrane away from the molecule is bit-identical to
  none. Depth-dependent surface tensions (HDGB) and membrane-aware
  volume/dispersion terms are not implemented.

**4.18 Uncertainty and convergence over frames.** Block averaging, bootstrap
confidence intervals and correlation-time estimates in `pbsolve mmpbsa`, so a
report states how many frames were enough.
*Status: done (0.0.8).*
* `pbsolve.series_statistics`: the statistical inefficiency g [Chodera et
  al. 2007] (equal to pymbar), correlation time, effective frames, the
  corrected standard error with a Student-t interval [Abramowitz & Stegun
  1964], block averaging [Flyvbjerg & Petersen 1989] as a check and the
  frames a target error needs; `pbsolve.timeseries.block_bootstrap`
  [Künsch 1989] for other statistics. `pbsolve mmpbsa` reports all of it
  (`--target-sem`). Calibrated on AR(1) series with exact answers (t
  interval 94–98 % from ~20 effective frames; the bootstrap and an
  automated blocking test after [Jonsson 2018] cover less) and shown on a
  5 ns MD of 1YCR: the per-frame total's error is 5.3 × SD/√N, and short
  prefixes underestimate the correlation time [Genheden & Ryde 2010;
  Grossfield et al. 2018].
* Remaining: equilibration detection (discarding a start), statistics over
  independent runs as a function.

**4.19 Public binding benchmark.** Follow the community guidelines for
protein–ligand affinity benchmarks [Hahn et al. 2022] on openly licensed
sets. Report correlations with experiment and the effect of each option
(4.8, 4.13, 4.14, 4.16), plus timing. Recent work shows MM/PBSA can beat
MM/GBSA for some classes (e.g. cyclic peptides) after protocol tuning [Brief.
Bioinform. 2025], and that PBSA solvation can be combined with machine-learned
potentials for the gas-phase term [AIQM-PBSA, ChemRxiv 2026]: pbsolve should
be the solvation engine in such workflows, not a fixed protocol.
*Status (0.1.7): done for single structures; an MD-averaged protocol and GPU
timings remain.*
* `scripts/binding_benchmark.py` runs the Open Force Field protein–ligand
  benchmark (commit fd88824; 14 targets, 359 ligands): Amber ff14SB and
  GAFF2/AM1-BCC, a restrained minimisation, one grid per target; Pearson R,
  Spearman ρ and Kendall τ per target with bootstrap intervals
  (docs/VALIDATION.md, "Binding benchmark against experiment").
* Mean R 0.60 at interior dielectric 2 (0.28 without the minimisation);
  per-residue dielectrics 0.59, fractional interface 0.59, Gaussian
  dielectric 0.48, dispersion term 0.37. APBS gives the same ΔΔG_polar on
  mcl1 (0.0002 kcal/mol).
* Remaining: frames from MD with the interaction entropy; cdk2 (its
  phosphothreonine); GPU timings on an idle machine (R3); an entropy or
  ligand-size term fitted against experiment (from 4.16).

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
*Status (0.1.9): charges, dielectric constants, positions and radii done;
the array interfaces remain.*
* `Solver.polar_gradient` (`gradient.rs`): dΔG_polar/dq of every atom and
  dΔG_polar/d pdie, d sdie, exact for the discrete energy through every
  focusing level and the outer boundary condition: one adjoint solve per
  level and state, the boundary cotangent carried to the coarser level by the
  transposed focusing. The energy equals `polar_solvation`'s bit for bit.
* Positions and radii on the spline surfaces (0.1.9, `wrt=("positions",
  "radii")`): the force code's boundary sums with the adjoint as a second
  field, on every level, plus the `mdh` kernel's derivatives; the exact
  counterpart of `polar_forces`, which takes the finest level only.
* 1YCR: central differences to 4e-9 of the largest charge derivative, 2e-8
  for the dielectric constants and 2e-7 of the largest position derivative;
  the summed position derivatives equal a rigid shift's derivative to 2e-6;
  Σ q ∂ΔG/∂q = 2ΔG to 4e-13; 2.1–2.2× the energy's time on the CPU
  (docs/VALIDATION.md, "Exact gradients").
* Next: DLPack arrays and PyTorch/JAX wrappers (`torch.autograd.Function`,
  `jax.custom_vjp`); the nonlinear equations last.

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
| Cross-code | map- and energy-level comparison with established FD and BEM codes (APBS done; pyDelPhi for the Gaussian dielectric, R4) | release |
| Physical | forces vs energy differences, energy conservation, nonlinear → linear limit | every PR (small), release (full) |
| Applied | small-molecule solvation (FreeSolv), protein–ligand binding benchmark, membrane systems | release |
| Performance | iterations vs spacing, frames per second, GPU vs CPU, memory; on an idle machine and several GPU generations (R3) | release, tracked over versions |

## 7. Licensing guardrails

* pbsolve stays MIT (plus the BSD notices in `THIRD_PARTY_NOTICES.md`).
  Implement methods from **papers**. Do not copy or translate code from
  copyleft or non-OSI projects; for example NextGenPB is GPL-3.0, pyDelPhi
  AGPL-3.0 and PROPKA LGPL-2.1. Check the license of any other code before
  reading it. Copyleft programs may be run separately to compare outputs.
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
5. **Timing and version number of the first public release.** *Decided
   (2026-11-15):* versions count v0.0.1, v0.0.2, … and the tenth is v0.1.0
   instead of v0.0.10; CI runs only for vX.Y.0 tags (and on demand), which
   are released when it passes. The repository is public since 2026-11-15.

## References

* Abramowitz M, Stegun IA (eds.). Handbook of Mathematical Functions, eq. 26.7.5. National Bureau of Standards (1964).
* Achondo MAP, Chaudhry JH, Cooper CD. An investigation of physics informed neural networks to solve the Poisson–Boltzmann equation in molecular electrostatics. arXiv:2410.12810 (2024).
* AIQM-PBSA: integrating machine learning interatomic potentials with MMPBSA for protein–ligand binding free energies. ChemRxiv, doi:10.26434/chemrxiv-2026-h0kn1 (2026, preprint).
* Adams M, Brezina M, Hu J, Tuminaro R. Parallel multigrid smoothing: polynomial versus Gauss–Seidel. J. Comput. Phys. 188, 593 (2003).
* Akkus E, Tayfuroglu O, Yildiz M, Kocak A. Revisiting MMPBSA by adoption of MC-based surface area/volume, ANI-ML potentials, and two-valued interior dielectric constant. J. Phys. Chem. B 127, 4415 (2023).
* Alcouffe RE, Brandt A, Dendy JE, Painter JW. The multi-grid method for the diffusion equation with strongly discontinuous coefficients. SIAM J. Sci. Stat. Comput. 2, 430 (1981).
* Bashford D, Karplus M. pKa's of ionizable groups in proteins: atomic detail from a continuum electrostatic model. Biochemistry 29, 10219 (1990).
* Borukhov I, Andelman D, Orland H. Steric effects in electrolytes: a modified Poisson–Boltzmann equation. Phys. Rev. Lett. 79, 435 (1997).
* Borukhov I, Andelman D, Orland H. Adsorption of large ions from an electrolyte solution: a modified Poisson–Boltzmann equation. Electrochim. Acta 46, 221 (2000).
* Boschitsch AH, Danilov PV. Formulation of a new and simple nonuniform size-modified Poisson–Boltzmann description. J. Comput. Chem. 33, 1152 (2012).
* Bosy M, Scroggs MW, Betcke T, Burman E, Cooper CD. Coupling finite and boundary element methods to solve the Poisson–Boltzmann equation for electrostatics in molecular solvation. J. Comput. Chem. (2024).
* Cai Q, Wang J, Zhao HK, Luo R. On removal of charge singularity in Poisson–Boltzmann equation. J. Chem. Phys. 130, 145101 (2009).
* Cai Q, Ye X, Wang J, Luo R. Dielectric boundary forces in numerical Poisson–Boltzmann methods: theory and numerical strategies. Chem. Phys. Lett. 514, 368 (2011).
* Callenberg KM, Choudhary OP, de Forest GL, Gohara DW, Baker NA, Grabe M. APBSmem: a graphical interface for electrostatic calculations at the membrane. PLoS ONE 5, e12722 (2010).
* Chodera JD, Swope WC, Pitera JW, Seok C, Dill KA. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theory Comput. 3, 26–41 (2007).
* Chen D, Chen Z, Chen C, Geng W, Wei GW. MIBPB: a software package for electrostatic analysis. J. Comput. Chem. 32, 756 (2011).
* Chen et al. Poisson–Boltzmann-based machine learning model for electrostatic analysis. Biophys. J. (2024).
* Chen L, Holst MJ, Xu J. The finite element approximation of the nonlinear Poisson–Boltzmann equation. SIAM J. Numer. Anal. 45, 2298 (2007).
* Chern IL, Liu JG, Wang WC. Accurate evaluation of electrostatics for macromolecules in solution. Methods Appl. Anal. 10, 309 (2003).
* Chu VB, Bai Y, Lipfert J, Herschlag D, Doniach S. Evaluation of ion binding to DNA duplexes using a size-modified Poisson–Boltzmann theory. Biophys. J. 93, 3202 (2007).
* Davis ME, McCammon JA. Dielectric boundary smoothing in finite difference solutions of the Poisson equation: an approach to improve accuracy and convergence. J. Comput. Chem. 12, 909 (1991).
* Decherchi S, Rocchia W. A general and robust ray-casting-based algorithm for triangulating surfaces at the nanoscale. PLoS One 8, e59744 (2013).
* Dembo RS, Eisenstat SC, Steihaug T. Inexact Newton methods. SIAM J. Numer. Anal. 19, 400 (1982).
* Di Florio V, Ansalone P, Siryk SV, Decherchi S, de Falco C, Rocchia W. NextGenPB: an analytically-enhanced finite element solver for the linearized Poisson–Boltzmann equation. Comput. Phys. Commun. 317, 109816 (2025).
* Dong L, Li P, Wang B. Enhancing MM/P(G)BSA methods: integration of formulaic entropy for improved binding free energy calculations. J. Comput. Chem. 46, e70093 (2025); preprint ChemRxiv, doi:10.26434/chemrxiv-2024-9308m.
* Duan L, Liu X, Zhang JZH. Interaction entropy: a new paradigm for highly efficient and reliable computation of protein–ligand binding free energy. J. Am. Chem. Soc. 138, 5722 (2016).
* Dzubiella J, Swanson JMJ, McCammon JA. Coupling hydrophobicity, dispersion, and electrostatics in continuum solvent models. Phys. Rev. Lett. 96, 087802 (2006).
* Ekberg V, Ryde U. On the use of interaction entropy and related methods to estimate binding entropies. J. Chem. Theory Comput. 17, 5379 (2021).
* Flyvbjerg H, Petersen HG. Error estimates on averages of correlated data. J. Chem. Phys. 91, 461–466 (1989).
* Fritsch FN, Carlson RE. Monotone piecewise cubic interpolation. SIAM J. Numer. Anal. 17, 238 (1980).
* Geng W, Yu S, Wei GW. Treatment of charge singularities in implicit solvent models. J. Chem. Phys. 127, 114106 (2007).
* Genheden S, Ryde U. How to obtain statistically converged MM/GBSA results. J. Comput. Chem. 31, 837–846 (2010).
* Genheden S, Ryde U. The MM/PBSA and MM/GBSA methods to estimate ligand-binding affinities. Expert Opin. Drug Discov. 10, 449 (2015).
* Gilson MK, Davis ME, Luty BA, McCammon JA. Computation of electrostatic forces on solvated molecules using the Poisson–Boltzmann equation. J. Phys. Chem. 97, 3591 (1993).
* Gohlke H, Kiel C, Case DA. Insights into protein–protein binding by binding free energy calculation and free energy decomposition for the Ras–Raf and Ras–RalGDS complexes. J. Mol. Biol. 330, 891 (2003).
* Golub GH, Ye Q. Inexact preconditioned conjugate gradient method with inner-outer iteration. SIAM J. Sci. Comput. 21, 1305 (1999).
* Grant JA, Pickup BT, Nicholls A. A smooth permittivity function for Poisson–Boltzmann solvation methods. J. Comput. Chem. 22, 608 (2001).
* Greene D, Qi R, Nguyen R, Qiu T, Luo R. Heterogeneous dielectric implicit membrane model for the calculation of MMPBSA binding free energies. J. Chem. Inf. Model. 59, 3041 (2019).
* Grossfield A, Patrone PN, Roe DR, Schultz AJ, Siderius DW, Zuckerman DM. Best practices for quantification of uncertainty and sampling quality in molecular simulations. Living J. Comput. Mol. Sci. 1, 5067 (2018).
* Hahn DF, et al. Best practices for constructing, preparing, and evaluating protein–ligand binding affinity benchmarks. Living J. Comp. Mol. Sci. 4, 1497 (2022).
* Holst M, Kozack RE, Saied F, Subramaniam S. Protein electrostatics: rapid multigrid-based Newton algorithm for solution of the full nonlinear Poisson–Boltzmann equation. J. Biomol. Struct. Dyn. 11, 1437 (1994).
* Holst M, McCammon JA, Yu Z, Zhou YC, Zhu Y. Adaptive finite element modeling techniques for the Poisson–Boltzmann equation. Commun. Comput. Phys. 11, 179 (2012).
* Holst M, Saied F. Multigrid solution of the Poisson–Boltzmann equation. J. Comput. Chem. 14, 105 (1993).
* Holst MJ, Saied F. Numerical solution of the nonlinear Poisson–Boltzmann equation: developing more robust and efficient methods. J. Comput. Chem. 16, 337 (1995).
* Huang et al. Neural network Poisson–Boltzmann electrostatics for biomolecular interactions. J. Comput. Phys. (2025); preprint HKU IMR2025-01.
* Im W, Feig M, Brooks CL III. An implicit membrane generalized Born theory for the study of structure, stability, and interactions of membrane proteins. Biophys. J. 85, 2900 (2003).
* Im W, Beglov D, Roux B. Continuum solvation model: computation of electrostatic forces from numerical solutions to the Poisson–Boltzmann equation. Comput. Phys. Commun. 111, 59 (1998).
* Jha A, Nottoli M, Mikhalev A, Quan C, Stamm B. Linear scaling computation of forces for the domain-decomposition linear Poisson–Boltzmann method. J. Chem. Phys. 158, 104105 (2023).
* Ji N, Liu T, Xu J, Shen LQ, Lu B. A finite element solution of lateral periodic Poisson–Boltzmann model for membrane channel proteins. Int. J. Mol. Sci. 19, 695 (2018).
* Jia Z, Li L, Chakravorty A, Alexov E. Treating ion distribution with Gaussian-based smooth dielectric function in DelPhi. J. Comput. Chem. 38, 1974 (2017).
* Jonsson M. Standard error estimation by an automated blocking method. Phys. Rev. E 98, 043304 (2018).
* Katzberger P, Riniker S. Implicit solvent approach based on generalized Born and transferable graph neural networks for molecular dynamics simulations. J. Chem. Phys. 158, 204101 (2023); A general graph neural network based implicit solvation model for organic molecules in water. Chem. Sci. (2024).
* Kilic MS, Bazant MZ, Ajdari A. Steric effects in the dynamics of electrolytes at large applied voltages. I. Double-layer charging. Phys. Rev. E 75, 021502 (2007).
* Kirkwood JG. Theory of solutions of molecules containing widely separated charges with special application to zwitterions. J. Chem. Phys. 2, 351 (1934).
* Künsch HR. The jackknife and the bootstrap for general stationary observations. Ann. Stat. 17, 1217–1241 (1989).
* Li L, Li C, Alexov E. On the modeling of polar component of solvation energy using smooth Gaussian-based dielectric function. J. Theor. Comput. Chem. 13, 1440002 (2014).
* Li L, Li C, Zhang Z, Alexov E. On the dielectric "constant" of proteins: smooth dielectric function for macromolecular modeling and its implementation in DelPhi. J. Chem. Theory Comput. 9, 2126 (2013).
* Liu X, Peng L, Zhang JZH. Accurate and efficient calculation of protein–protein binding free energy—interaction entropy with residue type-specific dielectric constants. J. Chem. Inf. Model. 59, 272 (2019).
* Luty BA, Davis ME, McCammon JA. Electrostatic energy calculations by a finite-difference method: rapid calculation of charge–solvent interaction energies. J. Comput. Chem. 13, 768 (1992).
* McCormick SF, Benzaken J, Tamstorf R. Algebraic error analysis for mixed-precision multigrid solvers. SIAM J. Sci. Comput. 43, S392 (2021a).
* Miller BR, McGee TD, Swails JM, Homeyer N, Gohlke H, Roitberg AE. MMPBSA.py: an efficient program for end-state free energy calculations. J. Chem. Theory Comput. 8, 3314 (2012).
* Mobley DL, Guthrie JP. FreeSolv: a database of experimental and calculated hydration free energies, with input files. J. Comput. Aided Mol. Des. 28, 711 (2014).
* Nina M, Im W, Roux B. Optimized atomic radii for protein continuum electrostatics solvation forces. Biophys. Chem. 78, 89 (1999).
* Notay Y. Flexible conjugate gradients. SIAM J. Sci. Comput. 22, 1444 (2000).
* Qi R, Botello-Smith WM, Luo R. Acceleration of linear finite-difference Poisson–Boltzmann methods on graphics processing units. J. Chem. Theory Comput. 13, 3378 (2017).
* Qi R, Luo R. Robustness and efficiency of Poisson–Boltzmann modeling on graphics processing units. J. Chem. Inf. Model. 59, 409 (2019).
* Panday SK, Zhao S, Alexov E. Accurate and scalable continuum electrostatics for large biomolecular systems: the pyDelPhi Poisson–Boltzmann framework. J. Chem. Inf. Model. 66, 488 (2026).
* Parsegian VA. Energy of an ion crossing a low dielectric membrane: solutions to four relevant electrostatic problems. Nature 221, 844 (1969).
* Quan C, Stamm B, Maday Y. A domain decomposition method for the Poisson–Boltzmann solvation models. SIAM J. Sci. Comput. 41, B320 (2019).
* Reis PBPS, Vila-Viçosa D, Rocchia W, Machuqueiro M. PypKa: a flexible Python module for Poisson–Boltzmann-based pKa calculations. J. Chem. Inf. Model. 60, 4442 (2020).
* Rocchia W, Alexov E, Honig B. Extending the applicability of the nonlinear Poisson–Boltzmann equation: multiple dielectric constants and multivalent ions. J. Phys. Chem. B 105, 6507 (2001).
* Roux B. Influence of the membrane potential on the free energy of an intrinsic protein. Biophys. J. 73, 2980 (1997).
* Schnieders MJ, Baker NA, Ren P, Ponder JW. Polarizable atomic multipole solutes in a Poisson–Boltzmann continuum. J. Chem. Phys. 126, 124114 (2007).
* Sharp KA, Honig B. Calculating total electrostatic energies with the nonlinear Poisson–Boltzmann equation. J. Phys. Chem. 94, 7684 (1990).
* Silalahi ARJ, Boschitsch AH, Harris RC, Fenley MO. Comparing the predictions of the nonlinear Poisson–Boltzmann equation and the ion size-modified Poisson–Boltzmann equation for a low-dielectric charged spherical cavity in an aqueous salt solution. J. Chem. Theory Comput. 6, 3631 (2010).
* Tamstorf R, Benzaken J, McCormick SF. Discretization-error-accurate mixed-precision multigrid solvers. SIAM J. Sci. Comput. 43, S420 (2021b).
* Tan C, Tan YH, Luo R. Implicit nonpolar solvent models. J. Phys. Chem. B 111, 12263 (2007).
* Tatebe O. The multigrid preconditioned conjugate gradient method. Proc. 6th Copper Mountain Conference on Multigrid Methods, NASA CP-3224, 621 (1993).
* Trefethen LN. Approximation Theory and Approximation Practice, extended edition. SIAM (2019).
* Valdés-Tresanco MS, Valdés-Tresanco ME, Valiente PA, Moreno E. gmx_MMPBSA: a new tool to perform end-state free energy calculations with GROMACS. J. Chem. Theory Comput. 17, 6281 (2021).
* Wagoner JA, Baker NA. Assessing implicit models for nonpolar mean solvation forces: the importance of dispersion and volume terms. Proc. Natl. Acad. Sci. USA 103, 8331 (2006); correction 104, 1732 (2007).
* Wang E, Sun H, Wang J, Wang Z, Liu H, Zhang JZH, Hou T. End-point binding free energy calculation with MM/PBSA and MM/GBSA: strategies and applications in drug design. Chem. Rev. 119, 9478 (2019).
* Wang J, Cai Q, Li ZL, Zhao HK, Luo R. Achieving energy conservation in Poisson–Boltzmann molecular dynamics: accuracy and precision with finite-difference algorithms. Chem. Phys. Lett. 468, 112 (2009).
* Wang J, Cai Q, Xiang Y, Luo R. Reducing grid dependence in finite-difference Poisson–Boltzmann calculations. J. Chem. Theory Comput. 8, 2741 (2012).
* Wang L, Li L, Alexov E. pKa predictions for proteins, RNAs, and DNAs with the Gaussian dielectric function using DelPhi pKa. Proteins 83, 2186 (2015).
* Wang L, Zhang Z, Rocchia W, Alexov E. Using DelPhi capabilities to mimic protein's conformational reorganization with amino acid specific dielectric constants. Commun. Comput. Phys. 13, 13 (2013).
* Wang L, Krasny R, Tlupova S. A kernel-independent treecode based on barycentric Lagrange interpolation. Commun. Comput. Phys. 28, 1415 (2020).
* Wang T, Cooper CD, Betcke T, Barba LA. High-productivity, high-performance workflow for virus-scale electrostatic simulations with Bempp-Exafmm. arXiv:2103.01048 (2021).
* Wang et al. MM/PB(GB)SA benchmarks on soluble proteins and membrane proteins. Front. Pharmacol. 13, 1018351 (2022).
* Wei H, Luo A, Qiu T, Luo R, Qi R. Improved Poisson–Boltzmann methods for high-performance computing. J. Chem. Theory Comput. 15, 6190 (2019).
* Wei H, Luo R, Qi R. An efficient second-order Poisson–Boltzmann method. J. Comput. Chem. 40, 1257 (2019).
* Wei H, Zhao Z, Luo R. Machine-learned molecular surface and its application to implicit solvent simulations. J. Chem. Theory Comput. 17, 6214 (2021).
* Wei W, Hogues H, Sulea T. Comparative performance of high-throughput methods for protein pKa predictions. J. Chem. Inf. Model. (2023), doi:10.1021/acs.jcim.3c00165.
* Wilson L, Geng W, Krasny R. TABI-PB 2.0: an improved version of the treecode-accelerated boundary integral Poisson–Boltzmann solver. J. Phys. Chem. B 126, 7104 (2022).
* Wu et al. End-to-end reaction field energy modeling via deep learning based voxel-to-voxel transform (PBNeF). arXiv:2410.03927 (2024).
* Xiao L, Diao J, Greene D, Wang J, Luo R. A continuum Poisson–Boltzmann model for membrane channel proteins. J. Chem. Theory Comput. 13, 3398–3412 (2017).
* Zhou YC, Feig M, Wei GW. Highly accurate biomolecular electrostatics in continuum dielectric environments. J. Comput. Chem. 29, 87 (2008).
* Zhou Z, Payne P, Vasquez M, Kuhn N, Levitt M. Finite-difference solution of the Poisson–Boltzmann equation: complete elimination of self-energy. J. Comput. Chem. 17, 1344 (1996).
* Achieving all-atom molecular dynamics accuracy from the Poisson–Boltzmann method through machine learning. J. Chem. Phys. 164, 054107 (2026).
* Improving the predictive performance of binding affinities and poses for protein–cyclic peptide complexes through fine-tuned MM/PBSA(GBSA)-based methods. Brief. Bioinform. 26, bbaf632 (2025).

This list was compiled from abstracts and bibliographic indexes (November
2026); volume and page numbers have not been checked
against the publisher records. Verify each entry before citing it in a paper.
