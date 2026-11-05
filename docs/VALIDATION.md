# Validation

A PB code is only as trustworthy as its validation record. Every release
should update the status table below.

## Protocol

1. **Analytic references.** Born ion (unit charge, a = 2 Å, ε 1 → 78.54) and,
   planned, Kirkwood sphere with off-centre charges and salt.
2. **Grid convergence.** The error must shrink overall as the spacing decreases
   (0.5 → 0.33 → 0.25 → 0.2 Å). With a sharp dielectric boundary the error
   oscillates by a few tenths of a percent depending on how the surface cuts the
   grid, so judge the trend, and average over small grid shifts when comparing codes.
3. **APBS parity** (`pbsolve apbs compare`, module `pbsolve.apbs`). pbsolve and
   APBS 3.4.1 run the same PQR with identical `pdie/sdie/srad/sdens/ion/bcfl/temp`
   and the identical mg-auto grid; `surface="mol"` vs `srfm mol`. The report is
   layered so that any disagreement can be located:
   1. *grids* — the dims/spacings/lengths/centres APBS prints for every focusing
      level must equal `GridSpec.levels` (APBS silently resets `dime` > 65 that
      is not `32c+1`; build grids with `Solver(dime_policy="apbs")`);
   2. *maps* — APBS `dielx/diely/dielz/kappa/charge` vs `Solver.maps()` on the
      finest level: dielectric and ion maps must match node for node;
   3. *injection* — `Solver.solve_maps()` solves APBS' own maps, charges and
      boundary values: tests discretisation + linear solver alone;
   4. *energies* — solvated, reference, ΔG_polar; for complexes also receptor,
      ligand and ΔΔG_bind on the complex grid (`--ligand-chain`/`--ligand-atoms`).

   Existing APBS inputs can be replayed with `pbsolve apbs run apbs.in`.
   Acceptance: identical maps; |ΔG_pbsolve − ΔG_APBS| / |ΔG_APBS| < 1e-3
   (`passed()`); ΔΔG_bind differences ≪ 0.1 kcal/mol.
   **Expected residuals** come only from physical constants: APBS scales the
   source with CGS constants and converts kT → kJ/mol with CODATA-1986 values
   (pbsolve: CODATA 2018), so all energies differ by `apbs_unit_factor()` =
   1 + 1.2186e-5; reports also give differences "in APBS units" with this factor
   removed. APBS' analytic boundary uses a third constant set (CODATA-1986 SI),
   which leaves ≲ 1e-6 in unscreened reference states; with `bcfl zero` all
   energies agree to ~1e-10.
4. **Platform agreement.** CPU vs CUDA relative difference < 1e-8
   (`tests/test_platforms.py`, `python -m pbsolve test`).
5. **MM-PBSA benchmark.** Correlation of ΔG_bind with experiment on a public
   protein–ligand set, compared with the same protocol using APBS.

## Status (v0.0.1 + unreleased changes)

| Check | Result |
|---|---|
| Born ion, `mol`, h = 0.500 / 0.333 / 0.250 / 0.200 Å | −4.73 % / −1.08 % / −1.27 % / −0.94 % vs analytic (−81.96 kcal/mol); converging overall, not monotonically (before the APBS-exact surface: −4.73 / −1.09 / −1.23 / −0.71 %) |
| Born ion, `harmonic`, same spacings | −4.16 % / −1.08 % / −1.65 % / −1.09 % (unchanged) |
| Operator symmetry, charge conservation, SES crevice, SASA, APBS reference-sphere count, marking = brute force | pass (unit tests) |
| Multigrid preconditioner (unreleased): Pᵀ = R, linear interpolation, Galerkin = RAP, exact symmetry/SPD, coarsest solve, SPD V-cycle, contraction 0.23 per V-cycle, thread-count independence, multigrid = Jacobi energies | pass (`multigrid.rs`, `pb.rs`, `tests/test_multigrid.py`); CG iterations 10–19 independent of spacing vs 155–688 for Jacobi; Born table unchanged; see [Linear solver](#linear-solver-multigrid-versus-jacobi) |
| APBS 3.4.1 parity, synthetic 30-atom molecule (crevices, zero-radius atoms), `mdh`/`sdh`/`zero`, `mol`/`vdw`, 0–0.15 M, 2- and 3-level focusing, Born ion | all maps identical; energies within 1e-7 (solvated) / 5e-6 (reference, `mdh`/`sdh`) in APBS units — `tests/test_apbs_parity.py` |
| APBS 3.4.1 parity, MDM2–p53 peptide complex (PDB 1YCR, pdb2pqr 3 `--ff=AMBER`, 1670 atoms, 18 zero-radius H; ligand = chain B) | see table below: 0 / 2 146 689 differing nodes in every map; ΔG_polar +0.0012 % raw, ≤ 1.2e-6 in APBS units; ΔΔG_bind within 0.004 kcal/mol |
| Mixed precision (unreleased, the default) | `multigrid-f32`: ΔG_polar within 7.9e-13 of f64 (Born, 1YCR), equal iteration counts; 1YCR 1.42–1.55× faster on the CPU, 1.03–1.12× on the shared GPU (V-cycle 2× on the device). Default after a sweep of 28 cases (5 systems up to 20 179 atoms, 14 parameter variants, CPU and CUDA): ≤ 4.2e-13, identical iterations, CPU 1.4–1.7× on proteins; Born table and 1YCR APBS comparison unchanged; see *Mixed precision* |
| Batched solves (unreleased) | MM-PBSA frames/s on 1YCR: 1.4× (CPU, 2 workers) and 2.2× (CUDA, 3 workers), results bit-identical to sequential calls (see *Batched solves*) |
| CUDA backend | **first GPU run** (RTX 2080 SUPER, cc 7.5, driver 595.99 = CUDA 13.2, `--features cuda-13`, conda NVRTC 13.2): `pbsolve test` and `tests/test_platforms.py` pass, CPU vs CUDA 9.8e-15; APBS comparison on CUDA identical to CPU. Multigrid on the GPU: CPU vs CUDA ≤ 2.8e-14 (identical iteration counts, `tests/test_platforms.py`). NVRTC selection and CUBIN output (unreleased): the GPU now works with NVRTC 12.9, 13.2, 13.3 and 13.4 on the 13.2 driver, with energies bit-identical across all of them and to the earlier runs; CUDA 12 build run on the GPU (see *GPU runtime* below) |
| MM-PBSA benchmark | **not yet run** |

### APBS parity: 1YCR (h ≈ 0.46/0.46/0.38 Å, dime 129³, 2 focusing levels, `cfac 1.7`, `fadd 20`)

Energies in kcal/mol. "APBS units" = pbsolve × `apbs_unit_factor()`.

| Setting | Species | ΔG_polar APBS | ΔG_polar pbsolve | rel. diff | in APBS units |
|---|---|---|---|---|---|
| pdie 1, 0.15 M, `mdh` | complex | −1221.2814 | −1221.2671 | +1.17e-5 | −4.7e-7 |
| | receptor | −1234.0045 | −1233.9910 | +1.10e-5 | −1.2e-6 |
| | ligand | −410.1748 | −410.1701 | +1.15e-5 | −6.5e-7 |
| | **ΔΔG_bind** | 422.8979 | 422.8939 | −0.0039 kcal/mol | +0.0012 kcal/mol |
| pdie 2, 0 M, `sdh` | complex | −591.5911 | −591.5842 | +1.17e-5 | −4.4e-7 |
| | receptor | −598.4228 | −598.4162 | +1.10e-5 | −1.2e-6 |
| | ligand | −200.0468 | −200.0445 | +1.15e-5 | −6.4e-7 |
| | **ΔΔG_bind** | 206.8785 | 206.8766 | −0.0019 kcal/mol | +0.0006 kcal/mol |

Per-state energies agree to ≤ 3.4e-11 (solvated) and ≤ 2.2e-8 (reference) in
APBS units; the injected solves reproduce APBS' energies to ≤ 1.9e-8. Maximum
potential differences are ~6e-6 of the potential scale, i.e. the CGS/2018
source-constant ratio.

For comparison, the previous `scripts/compare_apbs.py` reported +3.1 % on the
same complex: APBS had silently reset the requested dime (121, 121, 105) to
(97, 97, 97), so the two codes solved on different grids.

Reproduce:

```bash
pdb2pqr --ff=AMBER --keep-chain --whitespace 1YCR.pdb 1ycr.pqr
pbsolve apbs compare 1ycr.pqr --spacing 0.5 --ionic-strength 0.15 --bcfl mdh --ligand-chain B
pbsolve apbs compare 1ycr.pqr --spacing 0.5 --pdie 2 --bcfl sdh --ligand-chain B
```

## Linear solver: multigrid versus Jacobi

The default preconditioner is a geometric multigrid V-cycle (see
`docs/ARCHITECTURE.md`); `preconditioner="jacobi"` keeps the v0.0.1 solver and
reproduces its energies bit for bit (checked on all benchmark cases below).
Both solve the same discrete system to ‖r‖/‖b‖ < 1e-8, so energies differ
only within that tolerance.

Component tests (`crates/pbsolve-core/src/multigrid.rs`): restriction is the
exact transpose of prolongation (odd, even and non-coarsened axes);
prolongation reproduces linear functions; the Galerkin operator equals
`Pᵀ(A(Px))` to 1e-12 on two levels of a sharp-boundary, salt-containing
operator; it is bitwise symmetric and positive definite, and preserves
constants of the Laplacian; the banded Cholesky solves 7- and 27-point
operators to 1e-11; the V-cycle is a symmetric positive definite operator
(1e-11) that reduces the residual by 0.23 per cycle as a stationary iteration.
Solver tests (`pb.rs`, `tests/test_multigrid.py`): multigrid and Jacobi
energies agree to ≤ 1e-8 relative (tol 1e-10), results are bitwise identical
with 1 and 4 threads, and the same preconditioner is used for map injection.

**Iterations and time, synthetic 800-atom globule** (`cargo run --release -p
pbsolve-core --example multigrid`; ε 2/78.54, 0.15 M, one solve, 18 threads):

| Grid | Levels | Jacobi CG iter. | Jacobi s | Multigrid CG iter. | Setup s | Multigrid s | ‖Δu‖/‖u‖ |
|---|---|---|---|---|---|---|---|
| 65³ (h 0.69 Å) | 4 | 286 | 0.25 | 18 | 0.01 | 0.16 | 6.9e-9 |
| 97³ (h 0.46 Å) | 5 | 420 | 2.32 | 18 | 0.03 | 0.74 | 1.3e-8 |
| 129³ (h 0.34 Å) | 5 | 553 | 10.26 | 18 | 0.07 | 1.97 | 2.0e-8 |
| 161³ (h 0.28 Å) | 5 | 688 | 24.83 | 18 | 0.13 | 3.88 | 2.4e-8 |

Smoother choice: Chebyshev degree 1–4 with interval ratio 3–15 was scanned on
this case; degree 1 was ~10 % faster on the CPU but needed ~40 % more CG
iterations (26–27), degree 3–4 saved iterations but not time. Default: degree
2 on [λmax/10, λmax].

**Full ΔG_polar calculation** (solvated + reference, 2 focusing levels,
including surface maps; 1YCR complex, 1670 atoms, 0.15 M; Born ion a = 2 Å).
GPU: RTX 2080 SUPER, shared with another job at ~95 % utilisation during the
measurement. Times cover `polar_solvation` only; creating the `Solver` (CUDA
context and kernel compilation, ≈ 0.3 s once per process) is not included:

| System | h (Å) | Fine dims | CG iter. Jacobi | CG iter. multigrid | CPU Jacobi s | CPU MG s | CPU speed-up | GPU Jacobi s | GPU MG s | GPU speed-up | ΔE MG vs Jacobi | ΔE GPU vs CPU (MG) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Born | 0.5 | 33³ | 80, 80 | 8, 8 | 0.07 | 0.05 | 1.6× | 1.03 | 0.14 | 7.6× | 5.3e-9 | 1.5e-15 |
| Born | 0.333 | 57³ | 163, 137 | 9, 8 | 0.24 | 0.12 | 2.0× | 2.38 | 0.24 | 9.9× | 1.4e-8 | 0 |
| Born | 0.25 | 65³ | 170, 155 | 10, 8 | 0.33 | 0.20 | 1.6× | 2.47 | 0.28 | 8.8× | 5.6e-9 | 6.5e-15 |
| Born | 0.2 | 81³ | 213, 187 | 9, 8 | 0.85 | 0.42 | 2.0× | 3.14 | 0.31 | 10.0× | 1.8e-8 | 6.5e-15 |
| 1YCR | 0.5 | 121×121×105 | 324–426 | 8–12 | 19.13 | 3.53 | 5.4× | 13.43 | 1.33 | 10.1× | 2.2e-8 | 1.4e-14 |
| 1YCR | 0.35 | 169×177×145 | 446–593 | 8–13 | 79.82 | 10.35 | 7.7× | 22.45 | 3.31 | 6.8× | 3.0e-8 | 2.8e-14 |

Iterations are per linear solve (solvated coarse, solvated fine, reference
coarse, reference fine). After surface construction, the remaining CPU time at
0.35 Å is dominated by the multigrid cycles themselves (map construction:
0.78 s); fused kernels are the next optimisation.

**APBS parity with multigrid:** both `pbsolve apbs compare` runs on 1YCR
(table above) were repeated with the multigrid default: all maps identical,
ΔG_polar unchanged within 1e-4 kcal/mol (sdh receptor −598.4163 instead of
−598.4162), ΔΔG_bind 422.8939 / 206.8766 kcal/mol as before; per-state
energies agree with APBS to ≤ 1.1e-9 (solvated) and ≤ 2.2e-8 (reference) in
APBS units.

## Batched solves: concurrent MM-PBSA frames

**Setup.** Single-trajectory MM-PBSA on 1YCR (public PDB entry; pdb2pqr AMBER,
1670 atoms; ligand = p53 peptide).
* Every frame needs three ΔG_polar calculations on one fixed grid:
  complex, receptor and ligand.
* The grid is 121×121×105 at h = 0.5 Å, with 0.15 M salt and default
  settings.
* Frames are the crystal structure with Gaussian noise (σ = 0.3 Å, seed 1).
* Machine: 18 CPU threads, and an RTX 2080 SUPER shared with another job
  (74–99 % utilisation).

**Where the time goes** (one calculation, sequential, `PolarResult.timings`,
seconds):

| Platform | Species | Maps | Boundary | Assembly | MG setup | CG solve | Total |
|---|---|---|---|---|---|---|---|
| CPU | complex | 0.139 | 0.090 | 0.098 | 0.250 | 3.083 | 3.66 |
| CPU | ligand | 0.053 | 0.021 | 0.098 | 0.249 | 3.058 | 3.48 |
| CUDA | complex | 0.141 | 0.081 | 0.099 | 0.365 | 0.591 | 1.28 |
| CUDA | ligand | 0.059 | 0.021 | 0.101 | 0.366 | 0.579 | 1.13 |

On the GPU, host-side work (maps, boundary, assembly, multigrid setup) takes
0.55–0.69 s per calculation, longer than the 0.58 s of GPU solving. Running
several calculations at once overlaps the two.

**Design choices, measured** (4 frames = 12 calculations, frames/s):

| Workers | CPU, one shared rayon pool | CPU, private pool per worker | CUDA, private pools |
|---|---|---|---|
| 1 | 0.093 | 0.093 | 0.275 |
| 2 | 0.094 | 0.128 | 0.462 |
| 3 | 0.094 | 0.134 | 0.600 |
| 4 | 0.094 | — | 0.592 |
| 6 | — | — | 0.681 |

* **One shared rayon pool gives nothing.** Every operation is still split as
  finely as for one calculation. Private pools with an equal share of the
  threads match what separate processes achieved:
  * 2 processes × 9 threads: 0.124 frames/s;
  * 3 processes × 6 threads: 0.131 frames/s.
* **Each CUDA backend needs its own stream.** The legacy default stream would
  serialise the workers.
* **Default worker counts.** 2 on the CPU, where a third worker adds 4 %. 3 on
  CUDA, where more workers add little and multiply the device memory in use.
  At 3 CUDA workers the time per calculation (≈ 0.55 s) is about the GPU solve
  time, so the GPU is now the limit.

**Evidence (ROADMAP 4.3):** `pbsa_solvation_frames` with 1, 8 and 64 frames
per call, frames/s:

| Frames per call | CPU, 1 worker | CPU, 2 workers | Speed-up | CUDA, 1 worker | CUDA, 3 workers | Speed-up |
|---|---|---|---|---|---|---|
| 1 | 0.093 | 0.121 | 1.30× | 0.266 | 0.510 | 1.92× |
| 8 | 0.093 | 0.128 | 1.39× | 0.272 | 0.599 | 2.21× |
| 64 | 0.092 | 0.130 | 1.41× | 0.269 | 0.603 | 2.24× |

Every frame's complex, receptor and ligand energies (polar and non-polar) are
identical, bit for bit:
* with 1 worker and with the default workers;
* between the 1-, 8- and 64-frame calls;
* for the first 4 frames, with the sequential `polar_solvation` calls of the
  profile above (CPU and CUDA).

The CUDA results also equal those from before the stream change. With one
worker, throughput does not depend on the batch size: there is no per-call
overhead to amortise. The gain comes from overlapping calculations. A single
frame already has three calculations to overlap.

**Tests.**
* `batch.rs`, 3 unit tests:
  * batch = sequential, bit for bit, for 1–3 workers;
  * a failing job leaves the others intact;
  * more workers than jobs, and empty batches.
* `tests/test_batch.py`, 10 tests (CPU and CUDA):
  * `polar_solvation_many` = sequential calls bit for bit, for 1, 2, 3 and 7
    workers;
  * `polar_solvation_batch` with workers = single frames;
  * shared, per-system and automatic grids;
  * argument errors;
  * the index of a failing system is reported, and the solver stays usable;
  * the timings add up;
  * `pbsa_solvation_frames` = `pbsa_solvation_frame`;
  * `pbsolve mmpbsa --workers 1` and `--workers 3` write identical CSV files.

## Mixed precision: f32 multigrid inside f64 CG

**Method.** `preconditioner="multigrid-f32"` runs the multigrid V-cycle
(hierarchy, smoother, transfers) on f32 copies of the operators and vectors.
* The Galerkin products and the coarsest Cholesky factorisation stay in f64.
* The CG iteration, the operator it applies and every energy stay in f64.
* Rounding makes the preconditioner slightly non-symmetric, so CG uses the
  flexible Polak–Ribière update [Golub & Ye 1999; Notay 2000]. Its extra
  inner product is fused with ‖r‖² (`dot2`), so it adds no synchronisation.
* This is the arrangement whose error McCormick, Benzaken & Tamstorf (2021)
  analyse: the low precision only limits how much one cycle reduces the
  error, never the accuracy the outer f64 iteration reaches.

**Correctness.**
* The f32 kernels round identically on the CPU and the GPU (same expression
  order, no FMA contraction): bit-identical outputs in
  `single_precision_kernels_match_the_cpu_bit_for_bit`.
* One f32 V-cycle differs from the f64 V-cycle by 8.8e-8 relative.
* Rewriting the kernels generically left the f64 path bit-identical: 1YCR
  energies of every MM-PBSA species on CPU and CUDA were compared before and
  after.

**Energies and iterations** (CPU unless noted; 0.15 M; Born ion with
`fadd 12`):

| System | h (Å) | Iterations f64 / f32 | Relative difference in ΔG_polar |
|---|---|---|---|
| Born | 0.5 / 0.333 / 0.25 / 0.2 | equal | 2.4e-14 / 7.1e-14 / 1.0e-13 / 7.9e-13 |
| 1YCR | 0.5 | 12, 12, 8, 8 both | 4.9e-14 (CUDA 6.3e-14) |
| 1YCR | 0.35 | 12, 13, 8, 8 both | 2.9e-14 (CUDA 5.7e-14) |

**Tolerance study.** 1YCR at h = 0.5 Å, tolerances 1e-8, 1e-10 and 1e-12.
* Both precisions need the same iteration counts at every tolerance.
* Both reach true residuals of 4–5e-13, so there is no stagnation at f32
  precision.
* At tolerances of 1e-10 and below, the f32 and f64 energies are identical to
  the last bit. The energy error of CG is quadratic in the solution error,
  since bᵀ(u − x_k) = ‖u − x_k‖²_A.
* Plain PCG with the f32 cycle converged identically too. The flexible update
  costs 1–2 % of solve time on the CPU and is kept because it remains correct
  for a non-symmetric preconditioner.

**Speed: full ΔG_polar calculation** (seconds, `PolarResult.timings`).
CUDA ran on the RTX 2080 SUPER shared with three GROMACS jobs (60–99 %
utilisation):

| Platform | System | h (Å) | CG solve f64 → f32 | Setup f64 → f32 | Total f64 → f32 | Speed-up |
|---|---|---|---|---|---|---|
| CPU | 1YCR | 0.5 | 3.02 → 1.73 | 0.24 → 0.28 | 3.58 → 2.32 | 1.55× |
| CPU | 1YCR | 0.35 | 8.88 → 5.65 | 0.76 → 0.88 | 10.66 → 7.50 | 1.42× |
| CPU | Born | 0.35 | 0.072 → 0.082 | 0.011 → 0.011 | 0.093 → 0.102 | 0.91× |
| CUDA | 1YCR | 0.5 | 0.62 → 0.55 | 0.36 → 0.40 | 1.31 → 1.27 | 1.03× |
| CUDA | 1YCR | 0.35 | 1.44 → 0.99 | 0.98 → 1.04 | 3.29 → 2.93 | 1.12× |

Setup grows by 10–15 % for the f32 copies of the operators. Small grids gain
nothing.

**GPU device throughput** (`cargo run --release -p pbsolve-cuda --features
cuda-13 --example precision`; 800-atom globule; operations queued back to
back; milliseconds):

| Grid | Fine operator f64 / f32 | V-cycle f64 / f32 | Full CG solve f64-MG / f32-MG |
|---|---|---|---|
| 97³ | 0.35 / 0.22 (1.6×) | 7.8 / 3.9 (2.0×) | 237 / 226 (1.05×) |
| 129³ | 0.71 / 0.38 (1.9×) | 13.4 / 8.1 (1.65×) | 404 / 267 (1.52×) |
| 161³ | 1.52 / 0.53 (2.9×) | 23.6 / 11.2 (2.1×) | 601 / 437 (1.37×) |
| 193³ | 2.35 / 0.86 (2.7×) | 38.4 / 18.8 (2.0×) | 873 / 540 (1.62×) |

The V-cycle is twice as fast in f32 on the device. A full solve gains less,
because each CG iteration also waits 9–10 ms on synchronisations (inner
products and the host-side coarsest solve) on this shared GPU. On an idle GPU
these waits are small, and the host stages dominate instead (see *Where the
time goes on the GPU* below).

**MM-PBSA throughput** (1YCR, 8 frames per call, frames/s; Δ polar terms
identical to 6 decimals):

| Platform | Workers | multigrid | multigrid-f32 | Gain |
|---|---|---|---|---|
| CPU | 1 | 0.096 | 0.145 | 1.51× |
| CPU | 2 | 0.135 | 0.209 | 1.55× |
| CUDA | 1 | 0.265 | 0.272 | 1.03× |
| CUDA | 3 | 0.566 | 0.647 | 1.14× |

Concurrent workers (4.3) and the f32 cycle combine: on the CPU, 0.096 →
0.209 frames/s (2.2×) against the sequential f64 run of the same frames.

**Making it the default.** Before `multigrid-f32` replaced `multigrid` as
the default, both were run on a wider set. Each row is one ΔG_polar
calculation with both preconditioners (0.15 M, h = 0.5 Å, `mdh`, `mol`,
tol 1e-8, `fadd 20` unless noted):
* Systems: Born ion (`fadd 12`), 1YCR (1 670 atoms), and pdb2pqr pH 4 models
  of 1HPV (3 137), 1TII (10 862) and 5DK3 (20 179 atoms, 193×313×321 grid).
* 1YCR variants: h = 0.35 Å; I = 0 and 1 M; pdie 2, 4, 20; sdie 2;
  `vdw` and `harmonic` surfaces; `sdh` and `zero` boundaries; APBS grid
  sizes; tol 1e-6 and 1e-12.

| Platform | Cases | Iterations f64 = f32 | Max relative ΔG difference | Total time f64 → f32 |
|---|---|---|---|---|
| CPU | Born, 2 spacings | all | 6.1e-14 | 1.00–1.14× |
| CPU | 1YCR, default + 14 variants | all | 4.2e-13 (tol 1e-6); ≤ 1.0e-13 otherwise | 1.41–1.67× |
| CPU | 1HPV / 1TII / 5DK3 | all | 7.7e-14 | 5.64 → 3.77 s, 22.4 → 15.0 s, 56.7 → 40.4 s (1.40–1.50×) |
| CUDA | Born, 1YCR (4 variants), 1HPV, 1TII, 5DK3 | all | 6.7e-14 | 1.04–1.24× |

CUDA ran on the RTX 2080 SUPER with no other jobs. Also unchanged by the
switch:
* the Born-ion table (`examples/born.rs`): every energy, error and
  iteration count; only the times change;
* the 1YCR APBS comparison (`pbsolve apbs compare … --ligand-chain B`): every
  printed energy and potential difference is identical for `multigrid` and
  `multigrid-f32`, and all three species and ΔΔG pass.

`preconditioner="multigrid"` still reproduces the previous default bit for
bit. `pbsolve apbs compare/run` accept `--preconditioner`.

**Where the time goes on the GPU.** On an otherwise idle GPU, the CG solve is
a small part of a calculation. Seconds per ΔG_polar (`PolarResult.timings`,
f64 multigrid, 18 CPU threads for the host stages):

| System | Platform | Maps | Boundary | Assembly | Setup | Solve | Total |
|---|---|---|---|---|---|---|---|
| 1YCR | CUDA | 0.14 | 0.07 | 0.09 | 0.26 | 0.19 | 0.76 |
| 1TII | CUDA | 0.69 | 1.05 | 0.49 | 1.21 | 1.25 | 4.72 |
| 5DK3 | CUDA | 1.58 | 4.02 | 1.30 | 3.39 | 3.01 | 13.47 |
| 5DK3 | CPU | 1.58 | 4.06 | 1.53 | 2.80 | 45.19 | 55.31 |

* The host stages are 75–78 % of a CUDA calculation: the dielectric and ion
  maps, the Debye–Hückel boundary (a sum over all atoms for every boundary
  node), the right-hand side, and the multigrid setup (Galerkin products,
  factorisation, uploads).
* The 9–10 ms per CG iteration measured earlier came from sharing the GPU
  with three GROMACS jobs. On the idle GPU, a 1YCR solve takes 0.19 s for
  12 iterations on two levels.

**Tests.**
* `real.rs`: conversions.
* `multigrid.rs`, 2 tests:
  * the f32 kernels equal the f64 ones to rounding;
  * the f32 V-cycle approximates the f64 one and is symmetric to 1e-5.
* `solver.rs`, 4 tests:
  * f64 accuracy (true residual ≤ 2·tol) down to tol 1e-13, with no extra
    iterations;
  * thread-count independence;
  * `dot2` = two dots bit for bit;
  * preconditioner names.
* `pb.rs`: f32 energies = f64 energies to 1e-10.
* `crates/pbsolve-cuda/tests/gpu.rs`:
  * CUDA = CPU with f32 (|Δu|/|u| = 2.2e-16);
  * CUBIN = PTX bit for bit, including f32;
  * the fused `dot2` in both precisions;
  * the f32 kernels match the CPU bit for bit.
* `tests/test_multigrid.py`: agreement at 0.5 and 0.35 Å, tol 1e-12, the
  option, the CLI, and the default (`multigrid-f32`, bit-identical to the
  explicit option).
* `pb.rs`: the default parameters use the f32 cycle; thread-count
  independence for both multigrid variants.
* `tests/test_apbs_tools.py`: `--preconditioner` of `pbsolve apbs run`.
* `tests/test_platforms.py`: CUDA = CPU on the peptide with
  `multigrid-f32`.

## GPU runtime: NVRTC selection and CUBIN kernels

**Problem.** cudarc opened the unversioned `libnvrtc.so` first. Only full
toolkits ship that name, so on a machine with `/usr/local/cuda` the system
NVRTC replaced the one in the Python environment. Kernels were compiled to PTX,
and PTX from an NVRTC newer than the driver is rejected
(`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). A pip-installed NVRTC
(`nvidia-cuda-nvrtc-cu12`) was never found, because `site-packages/nvidia/…` is
on no loader path.

**Fix.**
* `pbsolve-cuda` loads NVRTC itself, in this order:
  1. `$PBSOLVE_NVRTC`, if set;
  2. the versioned name of the build's CUDA major (`libnvrtc.so.13`);
  3. the unversioned `libnvrtc.so`, as a last resort.
* The kernels are compiled to a CUBIN for the device. A CUBIN loads on any
  driver of the same major, unlike PTX (CUDA minor-version compatibility).
  PTX is used only for GPUs newer than the NVRTC.
* The Python package preloads the NVRTC of the build's major. It looks in the
  environment's own library directory first (conda), then in pip's
  `nvidia/cuda_nvrtc` and `nvidia/cu13` directories.
* `cuda_info()` reports the `driver`, `nvrtc` and `nvrtc_library` in use.
* `PBSOLVE_CUDA_CODE=ptx|cubin` forces the code type, for diagnostics.

**Before and after.** Same machine (RTX 2080 SUPER, driver CUDA 13.2, system
toolkit 13.3 in `/usr/local/cuda`). "Before" is the wheel of commit `e876bfe`.
All rows use the `cuda-13` build unless stated.

| Setup | Before | After |
|---|---|---|
| conda env with `cuda-nvrtc` 13.2 | fails: system 13.3 loaded, `UNSUPPORTED_PTX_VERSION` | works: NVRTC 13.2, CUBIN sm_75 |
| OpenMM env (`pbsolve-mdtools`), conda NVRTC 13.4, newer than the driver | fails: `UNSUPPORTED_PTX_VERSION` | works: NVRTC 13.4, CUBIN sm_75 |
| conda env, `PBSOLVE_NO_PRELOAD=1` | fails: `UNSUPPORTED_PTX_VERSION` | works: NVRTC 13.2 (the versioned name resolves through Python's RPATH) |
| pip `nvidia-cuda-nvrtc` 13.4, preload from `nvidia/cu13` | not measured | works: NVRTC 13.4, CUBIN |
| plain Rust binary, system NVRTC 13.3 | — | works: CUBIN; forced PTX is rejected by the driver, as expected |
| `cuda-12` build, pip `nvidia-cuda-nvrtc-cu12` 12.9 | not run before | works: `pbsolve test`, `test_platforms.py` and `test_multigrid.py` all pass |
| `cuda-12` build, no NVRTC 12 installed | — | falls back to `libnvrtc.so` (13.3), CUBIN, works |
| `PBSOLVE_NVRTC=/nonexistent` | — | CUDA unavailable with the reason; `platform="auto"` uses the CPU |

**Bit-for-bit checks.** The multigrid benchmark (Born ion at 4 spacings and
1YCR at 0.5/0.35 Å, on CUDA) was run with each of these:
* conda NVRTC 13.2;
* pip NVRTC 13.4;
* system NVRTC 13.3;
* the `cuda-12` build with NVRTC 12.9.

Every energy and every iteration count is bit-identical to the run made before
the change (PTX through the `LD_LIBRARY_PATH` workaround). Solve times are
unchanged. With NVRTC 13.2 and 12.9 the forced-PTX kernels also give
bit-identical solutions to the CUBIN ones. The APBS parity results above
therefore carry over unchanged.

**Cost.** CUBIN compilation adds ≈ 30 ms over PTX: the first
`Solver(platform="CUDA")` takes 0.34 s instead of 0.32 s. Most of that is CUDA
context creation, which repeats whenever the last CUDA solver of a process is
dropped. The compiled image is cached per process: a further solver created
while one is alive takes 1–3 ms.

**Tests.**
* `crates/pbsolve-cuda/src/nvrtc.rs`, 8 unit tests that need no GPU:
  * target selection: a CUBIN for supported devices, PTX for newer ones, an
    error for older ones, forced code types;
  * parsing of `PBSOLVE_CUDA_CODE`;
  * the order of library names.
* `crates/pbsolve-cuda/tests/gpu.rs`, 5 GPU tests (they skip without a
  device):
  * the build's NVRTC major is preferred;
  * the kernels are a CUBIN for the device;
  * CUDA = CPU for both preconditioners (|Δu|/|u| ≤ 4.3e-16, equal iteration
    counts);
  * CUBIN = PTX bit for bit;
  * the image cache is reused.
* `tests/test_platforms.py`, 7 new or extended tests:
  * `cuda_info` consistency;
  * an unusable NVRTC disables CUDA with a reason;
  * an invalid code type is reported;
  * CUBIN and PTX give identical energies;
  * conda and pip directory layouts are searched;
  * the preload opt-outs work.
* CI runs the unit tests for both CUDA majors. The GPU job also runs the Rust
  GPU tests against the pip NVRTC.

## Input readers (`pbsolve-io`)

`scripts/make_io_fixtures.sh` builds a solvated p53 peptide (1YCR chain B)
with tleap + OpenMM + cpptraj + ParmEd (4681 atoms) and with GROMACS 2026
pdb2gmx/solvate/genion/mdrun (7790 atoms); `scripts/validate_io.py` compares
pbsolve's readers with MDAnalysis 2.11 / ParmEd 4.3 / netCDF4. Status: **49/49
checks pass**.

| File (writer) | Checked | Max deviation |
|---|---|---|
| prmtop (tleap) | names, residues, charges, masses, RADII, elements, 4676 bonds, IFBOX | charge 7e-8 e (÷18.2223 rounding of the reference) |
| DCD (OpenMM), NetCDF (cpptraj) | 10 frames | 0 (bit-identical) |
| mdcrd, inpcrd, rst7, ncrst (cpptraj/tleap) | coordinates | ≤ 1.9e-6 Å (reference is float32) |
| PSF (ParmEd) | names, segids, charges, masses, bonds; elements = prmtop's | 3e-8 e |
| mol2 (ParmEd), PDB | names, charges, bonds, coordinates | ≤ 1.4e-6 Å |
| GROMACS top (`#include` resolved) and `grompp -pp` output | names, residues, charges, masses, 5262 bonds vs the tpr | 3e-8 e, 8e-7 amu |
| XTC, TRR (mdrun), GRO with velocities, 11-model PDB (trjconv) | coordinates | ≤ 4.3e-6 Å (float32 reference) |
| mmCIF vs PDB of 1YCR | atoms, chains, residues, coordinates | 0 |

Consistency: ΔG_polar of the peptide from prmtop + rst7 (Amber mbondi2 radii)
and from PSF + rst7 with `--radii mbondi2` agree to 1e-9.

## Structure preparation (`pbsolve.prep`) vs pdb2pqr 3.8.0

`scripts/validate_prep.py` compares `pbsolve.prep.prepare(pdb, ff, optimize=…)`
with `pdb2pqr --ff=FF [--noopt] --drop-water --keep-chain --whitespace` line by
line. **48/48 identical** (8 structures × 3 force fields × with/without the
hydrogen-bond optimisation):

| Structure | Atoms out | Features exercised |
|---|---|---|
| 1YCR | 1670 | termini, OXT rebuilt; optimisation flips HIS A73, ASN A79, ASN A106 and rotates 19 OH/SH hydrogens (34 atoms move) |
| 1YCR variants | 1667–1670 | altlocs + insertion codes; blank chain without TER; two segments with the same chain ID split by OXT |
| 1YCR with ASH/GLH/HIE/HIP/LYN residue names | 1648–1669 | input-named states, Carboxylic optimisation |
| 1TII (waters stripped) | 10811 | multi-chain, disulfides |
| 1HPV (waters stripped) | 3128 | ligand atoms omitted |
| 5DK3 (waters stripped) | 20090 | 194 heavy atoms rebuilt, ~35 residues debumped, glycans omitted |

PROPKA titration (`--titration-state-method propka --with-ph X`), identical:
1YCR, 1TII, 1HPV, 5DK3 at pH 4 (opt and noopt, AMBER), 1TII at pH 9 and 11
(AMBER, PARSE, CHARMM; LYN, TYM, AR0, HIE). PROPKA is run on the same model
text as pdb2pqr (hydrogens removed, repaired atoms included).

Speed: 5DK3 1.24 s vs 3.37 s for pdb2pqr; 1YCR 0.09 s vs 0.28 s (pure Python both).

Known pdb2pqr difference: with 5-digit serial numbers fused to `HETATM`
(e.g. `HETATM10273`), pdb2pqr's `--drop-water` keeps those waters and
protonates them; pbsolve reads the fixed columns and drops them.

## Sensitivity of the sharp dielectric boundary

With `surface="mol"` (APBS `srfm mol`) a coordinate change of 1e-4 Å can flip
a few boundary faces next to a charge: on the 221-atom peptide, rounding the
coordinates to 4 decimals flipped 3 of 2.7 million faces and moved ΔG_polar by
0.4 kcal/mol (0.1 %). APBS behaves identically (same maps). Average over MD
frames, or use the smoothed surfaces planned in the roadmap, when such jumps matter.

