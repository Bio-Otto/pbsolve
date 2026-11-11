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
   (`passed()`); ΔΔG_bind differences ≪ 0.1 kcal/mol (the binding verdict:
   ≤ 0.05 kcal/mol in APBS units).
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
| APBS 3.4.1 parity, larger proteins: 1HPV (3 137 atoms; `mdh`, `sdh`, `zero`), 1TII (10 862) and 5DK3 (20 179 atoms, 193×321×321 grid), complex/receptor/ligand each | every map identical (1HPV, 1TII; 5DK3 compared without maps); ΔG_polar ≤ 4.0e-6 in APBS units; ΔΔG_bind within 0.042 kcal/mol in APBS units (\|ΔΔG\| up to 6 232 kcal/mol); with `zero` ≤ 1.6e-10 per state and ΔΔG 7.6e-7 kcal/mol, so the remainder is APBS' boundary constants; the binding verdict now uses APBS units (the raw difference contains ΔΔG × 1.2e-5); 1YCR in the test suite; see *APBS parity: larger proteins* |
| Mixed precision (unreleased, the default) | `multigrid-f32`: ΔG_polar within 7.9e-13 of f64 (Born, 1YCR), equal iteration counts; 1YCR 1.42–1.55× faster on the CPU, 1.03–1.12× on the shared GPU (V-cycle 2× on the device). Default after a sweep of 28 cases (5 systems up to 20 179 atoms, 14 parameter variants, CPU and CUDA): ≤ 4.2e-13, identical iterations, CPU 1.4–1.7× on proteins; Born table and 1YCR APBS comparison unchanged; see *Mixed precision* |
| Host stages (unreleased) | Faster maps, `mdh` boundary, operator setup and host transfers; every energy, iteration count, residual, map and potential bit-identical to the previous commit on CPU and CUDA (13 cases + MM-PBSA species); 1.06–1.26× (CPU), 1.03–1.19× (CUDA) per calculation; see *Host stages* |
| `bcfl="mdh-fast"` (unreleased, opt-in) | Interpolated `mdh`: ΔG_polar within 1.1e-13 of `mdh` (Born, 1YCR ×4, 1HPV, 1TII, 5DK3; CPU and CUDA), equal iterations; boundary 5DK3 3.39 → 0.55 s; 1YCR APBS comparison prints identical energies; see *mdh-fast* |
| Multigrid setup on the GPU (unreleased) | Galerkin operators, 1/diag, Gershgorin bounds and f32 copies built on the device: bit-identical to the host hierarchy (4 operators incl. semi-coarsening, all levels) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); CUDA setup 2.2–3.1×, calculation 1.2–1.3× faster on a shared GPU; see *Multigrid setup on the GPU* |
| Maps and right-hand side on the GPU (unreleased) | Dielectric and ion maps, operator, node charges and right-hand side assembled through the backend: bit-identical to the host (sphere marking incl. constructed boundary and summation-order cases, all surfaces and media, random maps, whole solves) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); APBS 1YCR on CUDA unchanged; CUDA calculation 1.23–1.32× faster on a shared GPU, CPU unchanged; see *Maps and right-hand side on the GPU* |
| CUDA workers from free memory (unreleased) | The device memory of a calculation is counted exactly (estimate = memory-pool peak to the byte on every measured grid up to 19.4 M nodes, 3 preconditioners); batches run as many CUDA workers as fit in the free memory (at most 3 by default): 3 × 5DK3 now runs on 2 workers instead of failing with out-of-memory on an 8 GB card; 1YCR and 1TII keep 3 workers; every result bit-identical to the previous commit (CPU and CUDA); see *CUDA workers from the free device memory* |
| Potentials read where needed (unreleased) | The solution stays on the backend: focusing reads it at the stencil nodes of the next level's boundary, the energy at the charged nodes; energies and potentials bit-identical to the full-potential path and to a host reference of the old pipeline (7 core cases, 4 GPU cases) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); CUDA calculation 1.06–1.10× faster on a shared GPU, 0.5 GB less host memory for 5DK3; see *Potentials read only where they are needed* |
| GPU memory kept during a calculation (unreleased) | The CUDA memory pool keeps the buffers a calculation frees between focusing levels and returns them when it ends (release threshold raised by the calculation's `worker_device_bytes`, restored and trimmed at the end, also on errors): exactly one hold per calculation (core test with a recording backend, CI), threshold and pool usage back to idle after single, failing and concurrent calculations (GPU test), 10/10 mutations caught; every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); APBS 1YCR on CUDA unchanged; solve stage 1.05–1.10× faster, a calculation 1.03–1.08× with `sdh` on a shared GPU; see *GPU memory kept during a calculation* |
| `mdh` boundary vectorised and overlapped (unreleased) | Exact `mdh` sums on AVX2 (bit-identical to the scalar sums and the direct `dh` sum) and, on CUDA, the solvated state's surface and outermost boundary built while the reference state is solved: every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds), also from a recording device backend with 1 and 3 threads (CI); 6/6 mutations caught; APBS 1YCR on CUDA unchanged; CUDA calculation 1.21–1.25× faster with `mdh` (5DK3 6.60 → 5.28 s), CPU boundary 1.18–1.24×; a GPU `mdh` kernel ruled out on consumer GPUs; see *The `mdh` boundary: vectorised and overlapped* |
| Solvated setup before its boundary (unreleased) | On CUDA the solvated state's outermost level builds maps, operator and multigrid before waiting for its boundary from the background (`pcg_setup` + `PreparedPcg::solve` = `pcg_on` bit for bit, also for b = 0): a test backend that delivers the boundary only during the setup proves the order; every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 5/5 mutations caught; APBS 1YCR on CUDA unchanged; CUDA `mdh` 1.05–1.06× (5DK3 5.42 → 5.10 s); see *Solvated setup before its boundary* |
| Fused multigrid smoother (unreleased) | The V-cycle's residuals and Chebyshev updates run as one pass each (`Backend::residual`, `chebyshev_first`, `chebyshev_next`; CPU loops and CUDA kernels with the separate operations' expressions): equal bit for bit to the step-by-step smoother (f64 and f32, degrees 1–3, semi-coarsening; CPU = CUDA for every step and a whole V-cycle) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 6/6 mutations caught; Born table and APBS 1YCR on CUDA unchanged; CPU 1.17–1.25× per calculation, CUDA solve stage 1.17–1.26× (calculation 1.05–1.11×) on a shared GPU; see *Fused multigrid smoother* |
| Fused CG steps (unreleased) | The CG update (x, r, ‖r‖², rᵀz, the f32 copy of r), the widening of the V-cycle's result with rᵀz and, on the CPU, A p with pᵀAp run as one pass each (`Backend::cg_update`, `promote_dot`, `apply_dot`), with the summation order of `dot`: the new CG loop equals the old one (kept in the tests) and a solve through `backend::Stepwise` bit for bit (3 preconditioners, 5 grids, CPU and CUDA), and so does every step (CPU and CUDA, both precisions); every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 12/12 mutations caught; Born table and APBS 1YCR on CUDA unchanged; CPU 1.03–1.07× per calculation (1.13× with `jacobi`), CUDA f64 CG kernels 1.11× (2–4 % of the solve stage) on a shared GPU; see *Fused CG steps* |
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

### APBS parity: larger proteins (1HPV, 1TII, 5DK3)

Setup:
* Three pdb2pqr 3.8.0 models: AMBER, `--noopt`, PROPKA titration at pH 4,
  waters removed. These are the models of the mixed-precision sweep.
* h = 0.5 Å, two focusing levels, `cfac 1.7`, `fadd 20`; pbsolve on the CPU.
* Each run compares complex, receptor and ligand on the complex's grid.

| System | Atoms | Ligand (leaves the complex) | dime |
|---|---|---|---|
| 1HPV, HIV-1 protease dimer (inhibitor not parametrised) | 3 137 | chain B (1 570 atoms) | 161×129×161 |
| 1TII, AB5 enterotoxin | 10 862 | A subunit, chains A + C (atoms 1–3484) | 225×193×225 |
| 5DK3, IgG4 antibody (pembrolizumab) | 20 179 | light chain A (3 317 atoms) | 193×321×321 |

Energies in kcal/mol. "APBS units" = pbsolve × `apbs_unit_factor()`.

| System, setting | Species | ΔG_polar APBS | ΔG_polar pbsolve | rel. diff | in APBS units |
|---|---|---|---|---|---|
| 1HPV, pdie 1, 0.15 M, `mdh` | complex | −2893.0061 | −2892.9784 | +9.57e-6 | −2.6e-6 |
| | receptor | −1337.1970 | −1337.1821 | +1.12e-5 | −9.8e-7 |
| | ligand | −1746.0495 | −1746.0315 | +1.03e-5 | −1.9e-6 |
| | **ΔΔG_bind** | 190.2405 | 190.2352 | −0.0053 kcal/mol | −0.0030 kcal/mol |
| 1HPV, pdie 2, 0 M, `sdh` | complex | −1406.2784 | −1406.2649 | +9.58e-6 | −2.6e-6 |
| | receptor | −648.7114 | −648.7041 | +1.12e-5 | −9.7e-7 |
| | ligand | −849.6027 | −849.5939 | +1.03e-5 | −1.9e-6 |
| | **ΔΔG_bind** | 92.0357 | 92.0331 | −0.0026 kcal/mol | −0.0015 kcal/mol |
| 1HPV, pdie 1, 0.15 M, `zero` | complex | −2267.0530 | −2267.0254 | +1.22e-5 | −1.3e-8 |
| | receptor | −1228.5596 | −1228.5446 | +1.22e-5 | −8.7e-9 |
| | ligand | −1475.4042 | −1475.3863 | +1.22e-5 | −1.2e-8 |
| | **ΔΔG_bind** | 436.9108 | 436.9055 | −0.0053 kcal/mol | −7.6e-7 kcal/mol |
| 1TII, pdie 1, 0.15 M, `mdh` | complex | −18323.0869 | −18322.9238 | +8.90e-6 | −3.3e-6 |
| | receptor | −8180.5848 | −8180.5011 | +1.02e-5 | −2.0e-6 |
| | ligand | −6722.1693 | −6722.1035 | +9.79e-6 | −2.4e-6 |
| | **ΔΔG_bind** | −3420.3327 | −3420.3192 | +0.0135 kcal/mol | −0.0282 kcal/mol |
| 5DK3, pdie 1, 0.15 M, `mdh` | complex | −44557.4193 | −44557.0534 | +8.21e-6 | −4.0e-6 |
| | receptor | −35008.7745 | −35008.4786 | +8.45e-6 | −3.7e-6 |
| | ligand | −3316.3615 | −3316.3251 | +1.10e-5 | −1.2e-6 |
| | **ΔΔG_bind** | −6232.2833 | −6232.2496 | +0.0337 kcal/mol | −0.0423 kcal/mol |

Every run passes. In detail:
* **Maps.** The dielectric and ion maps are identical for 1HPV (all three
  settings) and 1TII: 0 of 2.4 × 10⁸ node values differ (four maps, three
  species per run).
  * Charges differ by ≤ 8e-8 of the maximum (the DX format).
  * Potentials differ by ≤ 6.6e-6 of their scale, the CGS/2018 ratio of the
    source constants, as for 1YCR.
  * Solves on APBS' own maps reproduce APBS' energies to ≤ 7.3e-8 in APBS
    units.
  * 5DK3 was compared without maps (`--no-maps`): the comparison holds both
    programs' maps in memory (3.0 GB for 1TII, ~6 GB estimated for 5DK3),
    more than the shared machine had free.
* **Where the remainder comes from.** In APBS units the solvated states agree
  to ≤ 1.5e-9 and the reference states (no salt, pdie everywhere) to
  ≤ 2.0e-7.
  * The reference remainder is APBS' analytic boundary, which uses a third
    set of constants.
  * ΔG_polar is the difference of two energies 20–50 times larger than
    itself, so this remainder appears as 1–4e-6 in ΔG_polar.
  * It grows with the molecule: 0.042 kcal/mol in the ΔΔG of 5DK3, 7e-6 of
    its value.
  * Without an analytic boundary (`bcfl zero`) every state of 1HPV agrees to
    ≤ 1.6e-10, and ΔΔG_bind to 7.6e-7 kcal/mol.
* **Binding verdict fixed.** The raw ΔΔG difference also contains the
  constant factor, ΔΔG × 1.2e-5: 0.076 kcal/mol for 5DK3.
  * `BindingReport.passed` compared the raw difference with 0.05 kcal/mol.
    It would therefore report FAIL above ~4000 kcal/mol even for exact
    agreement; 5DK3 passed only because the boundary remainder had the
    opposite sign.
  * The verdict now uses the difference in APBS units, like the rest of the
    report (`test_binding_verdict_is_judged_in_apbs_units`).
* **Tests on a real complex.** `tests/test_apbs_parity.py` now runs 1YCR
  through APBS on a coarse grid (h = 0.8 Å):
  * `mdh` with maps and injection for each species, ΔΔG ≤ 5e-3 kcal/mol
    (1.2e-3 measured);
  * the command line with `--ligand-chain B` and `bcfl zero`, ≤ 1e-8 per
    state and ΔΔG ≤ 1e-4 kcal/mol (5e-6 measured).
  * Mutations caught: 4/4 (raw ΔΔG verdict, chain field, unit factor off by
    1e-7, reference state written as solvated).
* **Cost.** One run (three species, both programs, maps) takes 73–83 s for
  1HPV and 7.7 min for 1TII; 5DK3 without maps takes 8.5 min. APBS peaks at
  1.1 / 2.2 / 4.1 GB.

Reproduce (waters removed from the PDB first: pdb2pqr's `--drop-water`
keeps some waters of 5DK3, see *Structure preparation*):

```bash
grep -v HOH 1HPV.pdb > 1hpv_nowat.pdb   # likewise 1TII, 5DK3
pdb2pqr --ff=AMBER --noopt --keep-chain --whitespace \
    --titration-state-method propka --with-ph 4 1hpv_nowat.pdb 1hpv.pqr
pbsolve apbs compare 1hpv.pqr --spacing 0.5 --ionic-strength 0.15 --bcfl mdh --ligand-chain B
pbsolve apbs compare 1hpv.pqr --spacing 0.5 --pdie 2 --bcfl sdh --ligand-chain B
pbsolve apbs compare 1hpv.pqr --spacing 0.5 --ionic-strength 0.15 --bcfl zero --ligand-chain B
pbsolve apbs compare 1tii.pqr --spacing 0.5 --ionic-strength 0.15 --bcfl mdh --ligand-atoms 1-3484
pbsolve apbs compare 5dk3.pqr --spacing 0.5 --ionic-strength 0.15 --bcfl mdh --ligand-chain A --no-maps
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
  CUDA, where more workers add little and multiply the device memory in use
  (since *CUDA workers from the free device memory*, fewer where they do not
  fit).
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
* `batch.rs`, 5 unit tests:
  * batch = sequential, bit for bit, for 1–3 workers;
  * a failing job leaves the others intact;
  * more workers than jobs, and empty batches;
  * worker counts from device memory (two tests).
* `tests/test_batch.py`, 12 tests (CPU and CUDA):
  * `polar_solvation_many` = sequential calls bit for bit, for 1, 2, 3 and 7
    workers;
  * `polar_solvation_batch` with workers = single frames;
  * shared, per-system and automatic grids;
  * argument errors;
  * the index of a failing system is reported, and the solver stays usable;
  * the timings add up;
  * `pbsa_solvation_frames` = `pbsa_solvation_frame`;
  * `pbsolve mmpbsa --workers 1` and `--workers 3` write identical CSV files;
  * `batch_workers`.

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

## Host stages: bit-identical speed-ups

On an idle GPU the host stages took 75–78 % of a calculation (see *Where the
time goes on the GPU*). These changes make them faster without changing any
floating-point operation:
* The molecular surface (SAS points, probe spheres) depends only on the atoms.
  It is now built once per focusing sequence instead of once per level.
* Node maps (`mark_map`, homogeneous reference maps) are filled per z-plane in
  parallel, so their pages are touched by all threads.
* Map validation, κ²·V, `u += g` and the CPU backend's host transfers
  (`upload`, `download`, `read`, `write`) run in parallel. They are
  element-wise.
* `mdh` hoists the atom-only factors (l_B0 q/ε, 1 + κa) out of the node loop
  and sums four boundary nodes per pass over the atoms. Each node keeps its
  own sum in atom order, starting from the neutral element of `f64: Sum`, so
  every term and every sum is bit-identical. The test
  `mdh_equals_the_direct_sum_bit_for_bit` checks this.

**Check.** Energies, solvated and reference energies, iteration counts and
residuals (as `float.hex`) were compared before and after, together with
SHA-256 digests of all maps and both potentials of 1YCR and the MM-PBSA
species energies. The 13 cases cover Born, 1YCR with `multigrid`, `jacobi`,
`vdw`, `harmonic`, no salt, `sdh`, `zero`, pdie 4 with 0.5 M and APBS grids,
1TII and 5DK3. All are identical on CPU and CUDA.

**Time per ΔG_polar** (s, before → after; the load of other jobs rose from 2
to 12 during the last CPU cases):

| System | Platform | Maps | Boundary | Assembly | Setup | Total |
|---|---|---|---|---|---|---|
| 1YCR | CPU | 0.134 → 0.100 | 0.073 → 0.066 | 0.089 → 0.077 | 0.263 → 0.237 | 2.13 → 1.81 (1.18×) |
| 1TII | CPU | 0.75 → 0.39 | 1.16 → 0.96 | 0.71 → 0.54 | 1.31 → 1.09 | 16.5 → 14.7 (1.12×) |
| 5DK3 | CPU | 1.71 → 0.81 | 4.05 → 3.48 | 1.66 → 1.22 | 3.56 → 2.93 | 39.9 → 36.7 (1.09×) |
| 1YCR | CUDA | 0.139 → 0.095 | 0.072 → 0.067 | 0.087 → 0.086 | 0.30 → 0.30 | 0.72 → 0.67 (1.08×) |
| 1TII | CUDA | 0.74 → 0.38 | 1.11 → 0.94 | 0.53 → 0.37 | 1.43 → 1.43 | 4.63 → 3.95 (1.17×) |
| 5DK3 | CUDA | 1.64 → 0.80 | 4.05 → 3.41 | 1.34 → 0.91 | 3.84 → 3.73 | 12.89 → 10.87 (1.19×) |

**Galerkin products: a negative result.** The multigrid setup (`Setup`) is now
the largest remaining host stage. Two order-preserving rewrites of
`galerkin` were measured against the original:
* per-axis index tables instead of recomputing transfers per stencil entry;
* a row-major symmetrisation instead of 27 strided passes.

Both gave bit-identical operators and no speed-up: 37.6–38.0 s against
40.2–40.7 s of CPU time for three 5DK3 hierarchies. The product is bound by
its arithmetic. The fine level needs about 640 multiply-adds per coarse row
(27 fine rows × 7 entries × 3.4 interpolation targets), and their order
cannot change without changing the result. Re-associating the product axis by
axis (sum factorisation) would save only 1.3–2× of these operations, so it
was not done. The changes were reverted. Moving the product to the GPU, where
it is cheap and can keep the same operation order, is the better route.

## `mdh-fast`: interpolated Debye–Hückel boundary

**Method.** `bcfl="mdh-fast"` (not an APBS option) evaluates the `mdh` sum
exactly only at the Chebyshev points of rectangular patches of the box faces.
The other boundary nodes get tensor-product barycentric Lagrange
interpolation, as in kernel-independent treecodes (Wang, Krasny & Tlupova
2020).
* The six faces partition the boundary nodes. Each face is split recursively
  into patches.
* A patch is interpolated when an a-priori degree bound meets the tolerance
  with at most half as many exact sums as it has nodes. Otherwise it is split
  or summed exactly.
* The bound uses the Bernstein ellipse of the nearest atom,
  ρ = d/L + sqrt((d/L)² + 1), and the growth of e^(−κr) off the real axis.
* Tolerance: the error on a patch is at most 1e-10 of the largest summed term
  magnitude Σ_i |dh_i| on that patch. This is the scale of the potential
  before charges cancel.
* The default `mdh` is unchanged and still follows APBS operation for
  operation. Inputs written for APBS with `mdh-fast` ask for `mdh`.

**Calibration** (`mdh_fast_is_within_its_tolerance_of_mdh`, maximum error per
patch relative to the patch scale):

| Geometry | κ = 0 | 0.125 | 0.33 | 0.6 Å⁻¹ | Exact sums at 0.125 |
|---|---|---|---|---|---|
| 400 atoms centred, 80 Å box | 1.1e-15 | 1.2e-14 | 1.1e-13 | (all exact) | 51 % |
| 300 atoms next to a face | 4.0e-15 | 7.8e-14 | 3.0e-13 | 2.0e-13 | 39 % |
| one ion near a corner | 2.5e-14 | 2.6e-13 | 3.1e-13 | 6.2e-13 | 25 % |
| 800 atoms, 170 Å box | 7.9e-16 | 9.7e-14 | 1.2e-12 | 1.8e-12 | 13 % |

* The weight of the screening term, 0.5, is the smallest of 1, 0.5, 0.25 and
  0.125 that keeps every case below 1e-10. With 0.25 the error reached
  2.9e-10 at κ = 0.6.
* Ignoring screening altogether gave errors of 1e-7 at κ = 0.6.

**Energies** (ΔG_polar, `mdh-fast` against `mdh`, same grids):

| System | Platform | Relative ΔG difference | Iterations | Boundary s | Total s |
|---|---|---|---|---|---|
| Born, h 0.35 | CPU / CUDA | 1.6e-14 / 1.3e-14 | equal | 0.001 → 0.001 | 0.06 → 0.07 |
| 1YCR, 0.15 M | CPU / CUDA | 1.1e-13 / 7.0e-14 | equal | 0.062 → 0.042 | CUDA 0.65 → 0.64 |
| 1YCR, no salt | CPU / CUDA | 5.6e-14 / 0 | equal | 0.037 → 0.027 | |
| 1YCR, 1 M, pdie 4 | CPU / CUDA | 5.2e-14 / 2.2e-14 | equal | 0.064 → 0.042 | |
| 1YCR, h 0.35 | CPU / CUDA | 0 / 2.9e-14 | equal | 0.149 → 0.071 | CUDA 1.80 → 1.71 |
| 1HPV | CPU / CUDA | 3.6e-14 / 3.0e-14 | equal | 0.158 → 0.066 | CUDA 1.21 → 1.11 |
| 1TII | CPU / CUDA | 7.5e-15 / 7.5e-15 | equal | 0.956 → 0.246 | CUDA 3.99 → 3.26 |
| 5DK3 | CPU / CUDA | 3.1e-15 / 0 | equal | 3.38 → 0.55 | CUDA 10.94 → 8.05 |

* Solvated and reference energies agree to ≤ 1.1e-15.
* Coarse-grid boundary alone: 1YCR 1.6×, 1TII 4.1–4.6×, 5DK3 7.6×, with a
  maximum error of 1.7e-13 of max |φ|.
* The 1YCR APBS comparison with `--bcfl mdh-fast` prints the same energies as
  with `mdh`, and all species and ΔΔG pass.
* With step 1, a 5DK3 calculation on the GPU takes 12.89 → 8.05 s (1.6×).

**Tests.**
* `boundary.rs`:
  * the tolerance over 4 geometries × 4 salt levels;
  * the assembled boundary equals its patches;
  * independence of the thread count;
  * the faces partition the boundary.
* `tests/test_boundary.py`: energies and potentials against `mdh`, the
  option, the CLI, and the APBS input mapping.
* `tests/test_apbs_parity.py`: `mdh-fast` against APBS `mdh`.

## Multigrid setup on the GPU

**Problem.** On the CUDA platform the host built the multigrid hierarchy:
the Galerkin products (0.74 s per 5DK3 solve), the Gershgorin bounds, and
the f32 copies of `multigrid-f32`, converted on the host and uploaded. The
setup was the largest host stage (4.5 of 13.5 s for a 5DK3 ΔG_polar on a
shared GPU).

**Change.** `Levels::build` runs the steps of `Hierarchy::build` through new
`Backend` methods; on the CUDA backend they are f64 kernels:
* `pb_galerkin7` and `pb_galerkin27`: one thread per coarse row, which
  accumulates in the host's loop order (fine rows, stencil entries,
  interpolation targets);
* `pb_symmetrise27`: symmetrisation in place, one thread per off-diagonal
  pair; 0.5 (a + b) equals the host's 0.5 (b + a) exactly;
* `pb_diagonal27`: diagonal, inverse diagonal and a flag for a NaN or
  negative diagonal. The host then reports the first one with the same
  message as `galerkin`;
* `pb_gershgorin7` and `pb_gershgorin27`: block maxima of Σ|a|/a_diag; a
  maximum is exact in any order.

The coarsest level (≤ 1000 unknowns) is downloaded and factorised on the
host. The f32 copies are rounded on the device (`demote_coarse`,
`demote_operator`) as the host conversion rounds them. `pcg` now releases the
host operator right after the setup; it used to be released inside the
upload.

**Checks.**
* `device_multigrid_setup_equals_the_host_bit_for_bit` (CUDA): every coarse
  operator, inverse diagonal and Gershgorin bound, the coarsest factor and
  all f32 copies equal the host hierarchy bit for bit. Operators: 60 random
  atoms on 49³; 66×41×6 (semi-coarsening); 50×37×29 without salt; 97³ at
  1 M.
* `vcycle_of_device_levels_equals_the_host_vcycle`: the V-cycle output is
  identical in f64 and f32.
* `device_setup_reports_a_negative_diagonal_like_the_host`: the same error
  message; the f32 twin refuses to build a hierarchy.
* Core tests: the levels built through the CPU backend equal
  `Hierarchy::build` (3 grids, f64 and f32), the same negative-diagonal
  error, and the f32 twin refuses.
* The 13 fingerprint cases of *Host stages* (energies, iterations and
  residuals as `float.hex`, SHA-256 of maps and potentials, MM-PBSA species)
  are bit-identical to commit 7f2f939 on CPU and CUDA.
* The `cuda-12` build passes the GPU tests with NVRTC 12.9.
* The 1YCR APBS comparison on CUDA (with map injection) passes with the
  same energies as before: ΔG_polar −1221.2671 / −1233.9910 / −410.1701
  kcal/mol for complex, receptor and ligand, ΔΔG 422.8939 kcal/mol.

**Time per ΔG_polar** on CUDA (s, A/B: the previous and the new build run
alternately, best of 3). The RTX 2080 SUPER was shared with two GROMACS
processes (96–97 % utilisation), so the solve stage is about twice as slow
as on an idle GPU; the setup gain does not depend on that.

| System | Setup | Solve | Assembly | Total |
|---|---|---|---|---|
| 1YCR (1 670 atoms) | 0.353 → 0.163 | 0.318 → 0.334 | 0.083 → 0.079 | 0.926 → 0.772 (1.20×) |
| 1TII (10 862 atoms) | 1.645 → 0.577 | 1.393 → 1.353 | 0.396 → 0.400 | 4.918 → 3.784 (1.30×) |
| 5DK3 (20 179 atoms) | 4.495 → 1.470 | 3.389 → 3.415 | 1.039 → 1.037 | 13.49 → 10.48 (1.29×) |

On the CPU the times are unchanged (1YCR 2.21 → 2.27 s, 1TII 15.83 →
15.87 s under a load of 14–22 from other jobs). One 5DK3 setup on the GPU
now takes 0.31 s instead of 1.00 s (0.08 s upload, 0.74 s host hierarchy,
0.18 s f32 conversions and uploads):

| Step | Seconds |
|---|---|
| upload of the f64 operator | 0.080 |
| `Levels::build` on the device (incl. coarsest factorisation) | 0.169 |
| f32 copies of the levels | 0.018 |
| fine inverse diagonal (host → f32) | 0.024 |
| f32 copy of the fine operator | 0.005 |

The f64 operator upload is now the largest item; building the maps and the
right-hand side on the device would remove most of it.

**GPU memory** (peak above the baseline, 5DK3 solved by
`polar_solvation_many` with 1–3 workers; ~1 GB of the 8 GB card was in use by
other processes):

| Workers | Previous | New |
|---|---|---|
| 1 | 2 378 MiB | 2 458 MiB |
| 2 | 4 832 MiB | 5 024 MiB |
| 3 | out of memory in this run | out of memory in one run, 5 856 MiB in another |

The f64 copy of the first coarse level exists briefly during the setup
(+80 MiB per calculation for 5DK3). Three concurrent 5DK3 calculations were
already at the limit of an 8 GB card before this change. The number of
CUDA workers should follow the free device memory (see `docs/ROADMAP.md`).

## Maps and right-hand side on the GPU

**Problem.** After the multigrid setup moved to the GPU, a CUDA calculation
still built the dielectric and ion maps, the operator (κ̄²V, inverse
diagonal), the node charges and the right-hand side on the host, and then
uploaded the operator (four f64 grids) and the right-hand side. For 5DK3
(19.4 M nodes per grid, four solves) these host steps and transfers took
about 2.1 s of a 10.3 s calculation: marking 0.66 s, operator and
right-hand side 0.58 s, uploads 0.51 s, the dense boundary vector 0.18 s,
map checks and freeing the host operator 0.19 s (shared GPU, four solves).

**Change.** The pipeline (`pb.rs`) assembles the system through new
`Backend` methods. `surface::maps_on` decides which spheres mark which value
and calls `mark_maps`; the CPU backend runs `surface::mark_map`, the CUDA
backend runs f64 kernels:
* `pb_mark_spheres`: one warp per sphere. The `mark_range` bounds of the
  three axes are computed by six lanes (the same divisions, `ceil`, `floor`,
  `fmax` and `fmin`), the squared axis distances dx², dy², dz² once per
  sphere into shared memory, and each lane tests whole rows with the host's
  sum (dz² + dy²) + dx² ≤ r². The host skips nodes with dx² + dy² > r² first;
  because rounding is monotone and the squares are non-negative, such nodes
  (and rows with dz² + dy² > r²) fail the full test as well, so the GPU tests
  only the sum. Writes of the same value by several spheres need no order.
* `pb_fill`, `pb_operator7` (κ̄²V in place and 1/diag), `pb_scatter`
  (charges and boundary values from sparse host lists) and `pb_source_rhs`
  (b = 4π l_B q − A g on interior nodes, with `apply` for A g).

The node charges are spread on the host as a sparse list
(`spread_charges_sparse`: every node sums its contributions in the order of
`spread_charges`, starting from 0.0). `pcg_on` solves with the device
operator; `pcg` keeps its signature and uploads a host operator. The final
u = x + g adds the boundary values on the host as before (interior nodes add
0.0). The surface (SAS points), the boundary values, the potential download
and the energies stay on the host. Two small host changes keep the solve
stage as fast as before: the CUDA download writes its target buffer in
parallel first (a fresh 155 MB buffer was otherwise faulted in page by page
during the copy: 113 → 80 ms per 5DK3 download), and `pcg_on` waits for the
queued f32 copies before it reports the setup time. With `multigrid-f32`
the inverse diagonal is rounded to f32 before the hierarchy is built, so the
f64 copy is freed early (`multigrid::hierarchy_fits` tells from the grid
size alone whether a hierarchy can be built; otherwise Jacobi keeps the f64
copy). `Grid::boundary_indices` no longer visits interior nodes (same
indices, same order).

**Checks.**
* `device_sphere_marking_equals_the_host_bit_for_bit` (CUDA): random
  spheres inside, partly and fully outside the grid, radius 0, spheres on
  nodes; three passes on four shifted grids.
* `device_sphere_marking_follows_the_host_arithmetic_at_the_boundary`:
  200 spheres constructed so that the two summation orders disagree at a
  node, and nodes exactly on spheres (dyadic grid). Kernel mutations (`<`
  instead of `≤`, another summation order) fail this test; other mutations
  (diagonal order, κ̄²V, right-hand-side sign, scatter values) fail the
  operator and assembly tests. Dropping the host's dx² + dy² pre-test
  passes, as the argument above says.
* `device_coefficient_maps_equal_the_host_maps`: `mol` with and without
  salt, `vdw` and `harmonic`, solvated and reference states.
* `device_operator_of_random_maps_equals_the_host` and
  `device_assembly_equals_the_host_assembly_bit_for_bit`: operator, inverse
  diagonal, charges, boundary values and right-hand side equal the host's;
  a whole solve from device-built maps equals the solve of the uploaded host
  system bit for bit (potential, iterations, residual) for all three
  preconditioners.
* `device_assembly_rejects_bad_arguments`: index and length checks on both
  backends; the f32 twins refuse to assemble.
* Core: `sparse_spreading_equals_dense_spreading_bit_for_bit`,
  `boundary_indices_are_the_boundary_nodes_in_order`; the levels test also
  covers a grid too small to coarsen (`operator_to_host`).
* The 13 fingerprint cases (energies, iterations and residuals as
  `float.hex`, SHA-256 of maps and potentials, MM-PBSA species) are
  bit-identical to the previous commit on CPU and CUDA.
* The `cuda-12` build passes the GPU tests with NVRTC 12.9.
* The 1YCR APBS comparison on CUDA passes with the same energies as before:
  ΔG_polar −1221.2671 / −1233.9910 / −410.1701 kcal/mol for complex,
  receptor and ligand, ΔΔG 422.8939 kcal/mol. The Born table is unchanged.

**Time per ΔG_polar** on CUDA (s, A/B: the previous and the new build run
alternately, best of 3; the RTX 2080 SUPER was shared with two GROMACS
processes at 91–97 % utilisation):

| System | Maps | Assembly | Setup | Solve | Total |
|---|---|---|---|---|---|
| 1YCR (1 670 atoms) | 0.107 → 0.063 | 0.081 → 0.022 | 0.138 → 0.088 | 0.295 → 0.288 | 0.713 → 0.539 (1.32×) |
| 1TII (10 862 atoms) | 0.427 → 0.285 | 0.386 → 0.055 | 0.544 → 0.289 | 1.281 → 1.347 | 3.777 → 3.079 (1.23×) |
| 5DK3 (20 179 atoms) | 0.878 → 0.609 | 0.981 → 0.150 | 1.401 → 0.745 | 3.228 → 3.219 | 10.29 → 8.38 (1.23×) |

The maps column includes the SAS points (0.20 s for 5DK3, host). Marking
the three face maps of one 5DK3 grid (19.4 M nodes, 19 772 solute and
1 345 195 probe spheres) takes 0.22 s on the shared GPU and 0.37 s on the 18
CPU threads; the first kernel version (one block per sphere, every node
computing its own distances) took 0.80 s. The CPU backend runs the same
pipeline and its times are unchanged within the noise between two builds
(1YCR 2.25 → 2.29 s, 1TII 16.18 → 16.16 s, 5DK3 39.65 → 39.39 s, load 16–24
from other jobs). In one process, the host-assembled system (`pcg`) and the
backend-assembled one (`pcg_on`) solve equally fast on the CPU (1YCR fine
grid, 6 repetitions: 0.588 and 0.594 s per solve, spread ±0.02 s).

The largest stages on CUDA are now the `mdh` boundary (3.5 s for 5DK3,
host; 0.55 s with `mdh-fast`) and the CG solve; the four potential downloads
of a 5DK3 calculation take 0.3 s.

**GPU memory** (peak above the baseline, 5DK3 by `polar_solvation_many`):

| Workers | Previous | New |
|---|---|---|
| 1 | 2 464 MiB | 2 496 MiB |
| 2 | 4 992 and 5 088 MiB (two runs) | 4 960 MiB (two runs) |

Without the early f32 rounding of the inverse diagonal the new peak was
2 560 MiB for one worker.

## CUDA workers from the free device memory

**Problem.** Batches (`polar_solvation_many`, `polar_solvation_batch`,
`pbsa_solvation_frames`, `pbsolve mmpbsa`) ran three calculations at once on
CUDA by default, whatever the grid. Each worker holds the buffers of its own
calculation, about 131 bytes per grid node with the default preconditioner.
Three 5DK3 calculations (19.4 M nodes) need 7.1 GiB, and they failed with
`CUDA_ERROR_OUT_OF_MEMORY` on the 8 GB card (6.3 GiB free).

**Change.**
* `solver::solve_device_bytes(dims, preconditioner)` counts the backend
  vectors that a solve holds at its peak, in the CG iterations:
  * f64 values per fine node: the operator (ε_x, ε_y, ε_z, κ̄²V) and the CG
    vectors x, r, z, Ap and p (9), plus 1/diag for Jacobi;
  * `multigrid`: 4 more f64 per fine node (1/diag and the smoother's r, d,
    t), and 33 f64 per coarse node (27 coefficients, 1/diag, r, d, t, x, b);
  * `multigrid-f32`: the same levels in f32, plus the f32 fine operator and
    the rounded residual and correction (10 f32 per fine node).

  `polar_solvation` frees the buffers of each solve before the next one, so
  this is also the peak of a whole calculation.
* `batch::worker_device_bytes` sets aside this amount for the largest grid of
  the batch, plus 1/8 and 64 MiB (`POOL_SLACK_BYTES`) for the rounding of the
  device's memory pool.
* `batch::device_workers` keeps 1/20 of the device free and runs as many
  workers as fit in the rest of the free memory (`cuMemGetInfo`, which
  counts other processes). It returns at least 1, and at most the requested
  or default count.
* The Python `Solver` applies this limit on CUDA to the default and to an
  explicit `workers`. `Solver.batch_workers(grid, workers=None)` returns the
  number it will use. The CPU is unchanged.
* `CudaBackend` gains `memory_info`, `pool_peak` and `reset_pool_peak` (the
  high watermarks of the device's memory pool).
* `multigrid` now checks `hierarchy_fits` before building the hierarchy, as
  `multigrid-f32` already did. The results are the same, and no hierarchy is
  built just to be discarded. (With the current coarsening the coarsest
  level has at most 1000 unknowns, so every grid fits.)

**Estimate and measured peak** (one `polar_solvation`, CUDA). The pool's
high watermark of used memory equals the estimate plus the 16 KiB reduction
scratch of each precision in use, to the byte, on every grid below. Memory
depends on the grid and the preconditioner only, so the rows are labelled by
grid size.

| Grid (nodes) | Preconditioner | Estimate = measured (MiB) | Bytes/node | Reserved by the pool (MiB) | Set aside per worker (MiB) |
|---|---|---|---|---|---|
| 121×121×105 (1YCR) | multigrid-f32 | 193 | 131.5 | 224 (1.16×) | 281 |
| 129³ | multigrid-f32 | 269 | 131.4 | 320 (1.19×) | 367 |
| 201×177×201 (1TII) | multigrid-f32 | 895 | 131.3 | 992 (1.11×) | 1 071 |
| 201×177×201 (1TII) | multigrid | 972 | 142.5 | 1 056 (1.09×) | 1 157 |
| 201×177×201 (1TII) | jacobi | 546 | 80.0 | 576 (1.06×) | 678 |
| 225³ | multigrid-f32 | 1 425 | 131.2 | 1 536 (1.08×) | 1 667 |
| 193×313×321 (5DK3) | multigrid-f32 | 2 425 | 131.1 | 2 528 (1.04×) | 2 792 |

The pool reserves 32 MiB chunks, so it rounds up by 4–19 % (at most
115 MiB). Two concurrent workers on the 1TII grid reserved 2 016 MiB, 1.13×
their buffers. Building an extra backend did not change the free device
memory measurably.

**Batches before and after** (`polar_solvation_many` of N copies on one
grid, default workers, two runs each; previous build = 7bad270; RTX 2080
SUPER shared with two GROMACS processes at ~95 %, 6.3 GiB free; GPU memory
by `nvidia-smi` above the baseline):

| System | N | Previous: workers, time | New: workers, time | GPU memory, new |
|---|---|---|---|---|
| 5DK3 | 3 | 3: out of memory (both runs, +6.2 GiB at failure) | 2: 25.8 / 26.0 s | +4.9 GiB |
| 1TII | 6 | 3: 12.0 / 13.2 s | 3: 13.2 / 12.8 s | +2.7 GiB |
| 1YCR | 12 | 3: 3.28 / 3.62 s | 3: 3.65 / 3.58 s | +0.6 GiB |

* The energies are identical (`float.hex`) between the two builds, across
  all copies and for every worker count.
* An explicit `workers=3` on 5DK3 now runs 2 workers (26.8 / 27.2 s).
* For comparison, the previous build with `workers=2` took 27.0 / 25.7 s,
  and the new one with `workers=1` took 25.9 / 26.1 s.
* On this shared machine, several workers therefore give nothing for a
  grid as large as 5DK3's. The gain of concurrency is on small grids
  (1YCR 2.2×, see *Batched solves*). Whether large grids gain on an idle GPU
  is not measured.

**Checks.**
* Core `device_bytes_match_the_measured_peaks`: the estimate equals the
  pool peaks measured above (three preconditioners, three grids), so the
  formula is checked on machines without a GPU too.
* Core `workers_are_limited_by_the_free_device_memory` and
  `device_workers_boundaries`:
  * the benchmark grids on an 8 GB card give 3 workers for 1YCR and 1TII
    and 2 for 5DK3;
  * the count is at least 1 and at most the limit;
  * the largest grid decides;
  * a worker is added exactly when it fits next to the 1/20 reserve.
* Core: the levels test checks that `coarse_dims` equals the levels of the
  host hierarchy.
* `tests/memory.rs` (CUDA) runs in a process of its own, because the pool's
  counters cover the whole process.
  * For four grids up to 7.2 M nodes, the pool peak equals
    `solve_device_bytes` plus the scratch (three preconditioners).
  * The reserved memory stays within `worker_device_bytes`.
  * Three concurrent workers stay within three times that.
* `test_batch_workers` (CPU and CUDA): the default, an explicit count, at
  most 1 on CUDA for a 1025³ grid (~140 GB), and argument errors.
* Mutations, all caught:
  * fewer f32 values per node in the estimate;
  * no 1/20 reserve;
  * the smallest grid deciding;
  * no 1/8 slack (caught by the 7.2 M-node grid);
  * an extra CG vector that the estimate does not count;
  * half the memory per worker;
  * the binding ignoring device memory.
* The 13 fingerprint cases are bit-identical to the previous commit on CPU
  and CUDA, and single-calculation times are unchanged.

## Potentials read only where they are needed

**Problem.** After every solve the pipeline downloaded the whole solution
(155 MB per 5DK3 grid), added the boundary values on the host, interpolated
the next level's boundary from that array and summed the energy over a
charge array of full grid length. On CUDA (5DK3, four solves) this took
0.68 s of an 8.8 s calculation: the downloads 0.34 s, adding the boundary
values 0.05 s, the two focusing boundaries 0.10 s and the energies 0.19 s.
About 0.19 s of the download time was in fact a wait for the CUDA memory
pool, which returns the solver's freed buffers to the driver at the next
synchronisation; that wait remains (see below).

**Change.**
* `Backend::gather(v, idx)` returns the values of a backend vector at a
  list of nodes: on CUDA the kernel `pb_gather` and a download of just those
  values, on the CPU a copy. The default implementation downloads `v`.
* The solution of a level stays on the backend (`LevelSolution`: x and the
  boundary values g). u = x + g is computed as before (x + g on boundary
  nodes, x + 0.0 inside), but only at the nodes that are needed:
  * focusing: `boundary::focus_nodes` collects the coarse nodes of the
    trilinear stencils of the fine boundary nodes in a bitmap
    (`grid::NodeSet`, which also gives a node's position in the gathered
    list), and `focus_boundary_values` interpolates with the arithmetic of
    `focus_boundary` (`Grid::interpolate_with`). For 5DK3 that is 428 556 of
    19.4 M coarse nodes;
  * energy: ½ Σ q u over the charged nodes of `spread_charges_sparse`, in
    increasing order (157 656 nodes for 5DK3). The full-length sum adds the
    same products in the same order plus exact zeros, which leave a sum
    with any nonzero term unchanged, so the energy is identical. Only a
    total of exactly zero (no charges) could differ in its sign.
* Boundary values are kept on the boundary nodes only
  (`analytic_boundary_values`, `focus_boundary_values`,
  `Grid::boundary_position`); `analytic_boundary` and `focus_boundary` keep
  their full-length results and are built from these.
* `solve_focused` (`Solver.potential`, maps) and `solve_with_maps` still
  download the finest potential; their coarser levels read only the
  focusing nodes.
* Timings: the energy stage now contains reading u at the charges. The
  solve stage no longer contains a download, but it ends with a
  synchronisation, so the memory pool's release of the solver's buffers
  (~47 ms per 5DK3 solve) still counts there. Freeing host memory belongs to
  no stage, as before.

**Checks.**
* `energies_from_the_charged_nodes_equal_the_full_potential_energies`
  (core, CPU): 7 cases with 1, 2 and 3 focusing levels, `mdh`, `sdh`,
  `mdh-fast` and `zero`, all three preconditioners, and a level whose y faces
  lie on the coarse level's faces (focusing then reads coarse boundary
  nodes). `polar_solvation`'s energies equal `solve_focused` +
  `electrostatic_energy_kt` bit for bit, and `solve_focused`'s potential
  equals a host reference of the old pipeline (full potentials,
  `analytic_boundary`, `focus_boundary`, host maps and charges,
  `solve_with_maps`) bit for bit. An uncharged molecule gives 0. Ignoring g
  at gathered boundary nodes fails the test (the flush case catches it).
* `boundary_values_are_the_boundary_nodes_of_the_full_vectors` (all four
  boundary conditions and focusing), `focus_nodes_are_the_nodes_focusing_reads`
  (also the error for a fine grid outside the coarse one),
  `boundary_values_are_added_on_the_boundary_nodes_only` (an interior −0.0
  still becomes +0.0), `node_set_lists_and_ranks_its_members`,
  `interpolation_through_a_closure_equals_the_slice_form`, and
  `boundary_position` against `boundary_indices`.
* GPU: `device_gather_reads_the_requested_values` (repetitions, any order,
  −0.0 and a subnormal, an empty list, index errors, the f32 twin refuses) and
  `device_energies_from_the_charged_nodes_equal_the_full_potential_energies`
  (1–3 levels, the flush level, both multigrid precisions). The memory test
  still finds the pool's peak equal to `solve_device_bytes` to the byte.
* The 13 fingerprint cases (energies, iterations, residuals, maps,
  potentials, MM-PBSA species) are bit-identical to the previous commit on
  CPU and CUDA; the `cuda-12` build passes the GPU tests with NVRTC 12.9.
* The 1YCR APBS comparison on CUDA passes with the same energies as before
  (ΔΔG 422.8939 kcal/mol). The Born table is unchanged.

**Time per ΔG_polar** (s; A/B: the previous and the new build run
alternately, median of 6 (1YCR, CUDA), 3 (CPU 1YCR, CUDA 1TII and 5DK3) or 1
calculation per run, two runs each. The RTX 2080 SUPER was shared with
GROMACS at 92–93 % utilisation; the 18 CPU threads had a load of 12–15 from
other jobs. Peak RSS of the Python process.)

| System | Platform | Previous | New | Energy stage | Peak RSS (MiB) |
|---|---|---|---|---|---|
| 1YCR | CUDA | 0.629 / 0.616 | 0.577 / 0.583 (1.07×) | 0.004 → 0.003 | 315 → 272 |
| 1TII | CUDA | 3.228 / 3.134 | 2.940 / 2.966 (1.08×) | 0.070 → 0.003 | 514 → 332 |
| 5DK3 | CUDA | 8.928 / 8.839 | 8.177 / 8.321 (1.08×) | 0.189 → 0.005 | 936 → 430 |
| 1YCR | CPU | 2.345 / 2.317 | 2.321 / 2.284 (1.01×) | 0.005 → 0.000 | 307 → 286 |
| 1TII | CPU | 16.38 / 16.38 | 16.13 / 16.13 (1.02×) | 0.06 → 0.002 | 1 196 → 1 057 |
| 5DK3 | CPU | 39.89 / 40.05 | 39.54 / 39.53 (1.01×) | 0.19 → 0.01 | 3 136 → 2 741 |

The 5DK3 CPU row was measured with an intermediate build that freed the
finest solution inside the energy stage; it is otherwise the same code.
On CUDA the 5DK3 solve stage went from 3.44 / 3.39 to 3.28 / 3.25 s and
assembly from 0.16 to 0.11 s (no 155 MB host buffers to fault in).

The largest stages of a CUDA calculation are now the `mdh` boundary (host,
3.4 s for 5DK3; 0.55 s with `mdh-fast`) and the CG solve. The memory pool
returns each level's buffers to the driver at the next synchronisation
(release threshold 0) and allocates them again for the next level; the next
section keeps them reserved during a calculation.

## GPU memory kept during a calculation

**Problem.** The CUDA backend allocates from the device's memory pool
(cudarc uses `cuMemAllocAsync`). The pool's release threshold is 0 by
default, so at every synchronisation it returns all freed memory to the
driver. Each focusing level freed its solver buffers; at the next
synchronisation the pool released them, and the next level reserved new
memory. A probe at the synchronisation after each solve measured 51 ms per
level for 5DK3 (2.5 GB reserved), 19 ms for 1TII and 3.4 ms for 1YCR, four
times per calculation (two levels in each of the two media). With the
threshold raised the wait was 0.

**Change.**
* `Backend::retain_memory(bytes)` and `release_memory(bytes)` (no-ops by
  default), used through `backend::MemoryHold`, which retains on creation and
  releases on `release()` or when dropped (so also when a calculation fails).
* `polar_solvation` (both media) and `solve_focused` hold
  `worker_device_bytes([dims], preconditioner)`, what `device_workers` sets
  aside for one calculation, from start to end.
* CUDA: a process-wide registry per device sums the bytes of the
  calculations in progress, and the pool's release threshold is the previous
  threshold plus that sum. When a calculation ends, the backend synchronises
  its stream (so its frees have run), lowers the threshold and trims the pool
  to it (`cuMemPoolTrimTo`): the memory is returned at once, not at some
  later synchronisation. After the last calculation the previous threshold is
  back (0, unless another library of the process set one). An idle process
  holds no more GPU memory than before, and a worker keeps at most its share.
* Timings: the solve stage still ends with a synchronisation but no longer
  contains a release. The single release at the end of a calculation (54 ms
  for 5DK3) belongs to no stage.
* `CudaBackend::pool_usage` (current used and reserved memory) and
  `pool_release_threshold`, for diagnostics and tests.

**Checks.**
* Core `tests/pipeline.rs` (CPU, runs in CI; first named `retention.rs`)
  wraps the CPU backend in one that records the calls:
  * `polar_solvation` and `solve_focused` (3 grids, all preconditioners) make
    one retain of `worker_device_bytes` before their first backend call and one
    release after their last, with results bit-identical to the plain CPU
    backend;
  * a calculation that fails at its fine level still releases;
  * in a batch on 2 workers every calculation holds on its own worker (4 holds
    of the right sizes), with energies equal to sequential calls;
  * a `MemoryHold` ends exactly once (`release` then drop, nested holds).
* GPU `tests/retention.rs` (a test binary of its own, because the pool is
  process-wide):
  * the default pool returns freed memory at the next synchronisation;
  * holds raise the threshold by their bytes, add up (also through the f32
    twin) and keep freed memory;
  * after the last hold the threshold is 0 and the reserved memory is back to
    idle without a further synchronisation;
  * after `polar_solvation` (`multigrid-f32`, Jacobi), `solve_focused`, a
    failing calculation and a batch on 3 workers, threshold and pool usage are
    back to idle;
  * a threshold set by another library (48 MiB) is restored;
  * the pool's reserved peak stays within `worker_device_bytes` (three times
    that for the batch).
* Ten mutations, all caught:
  * no hold in `polar_solvation`, or none in `solve_focused`;
  * a release between the two media;
  * the estimate without slack;
  * a double release;
  * no trim;
  * a release that does not subtract its bytes;
  * a retain that does nothing;
  * the threshold restored to 0 instead of its previous value;
  * a trim before the stream's frees ran.
* `tests/memory.rs` passes unchanged: the pool's peak of used memory still
  equals `solve_device_bytes` to the byte.
* The 13 fingerprint cases are bit-identical to the previous commit on CPU
  and CUDA. The `cuda-12` build passes the GPU, memory and retention tests
  with NVRTC 12.9. The 1YCR APBS comparison on CUDA passes with the same
  energies (ΔΔG 422.8939 kcal/mol), and the Born table is unchanged.

**Time per ΔG_polar** (s). A/B: the previous and the new build ran
alternately, two runs each, with the median of 6 (1YCR) or 3 calculations
per run. The RTX 2080 SUPER was shared with GROMACS at 26–91 % utilisation,
and the 18 CPU threads had a load of 23–35 from other jobs. Energies are
bit-identical in every row.

| System | `bcfl` | Previous | New | Solve stage | Setup + assembly |
|---|---|---|---|---|---|
| 1YCR | `mdh` | 0.537 / 0.550 | 0.527 / 0.513 | 0.163 / 0.161 → 0.155 / 0.147 | 0.065 / 0.065 → 0.064 / 0.062 |
| 1TII | `mdh` | 3.62 / 3.90 | 3.43 / 3.53 | 0.955 / 0.938 → 0.892 / 0.868 | 0.250 / 0.259 → 0.244 / 0.234 |
| 5DK3 | `mdh` | 10.47 / 10.27 | 10.74 / 9.42 | 2.319 / 2.323 → 2.112 / 2.103 | 0.708 / 0.662 → 0.644 / 0.614 |
| 1YCR | `sdh` | 0.351 / 0.353 | 0.325 / 0.341 | 0.162 / 0.158 → 0.146 / 0.145 | 0.065 / 0.064 → 0.061 / 0.063 |
| 1TII | `sdh` | 1.737 / 1.728 | 1.660 / 1.670 | 0.948 / 0.960 → 0.867 / 0.879 | 0.252 / 0.248 → 0.243 / 0.242 |
| 5DK3 | `sdh` | 4.055 / 4.082 | 3.896 / 3.860 | 2.302 / 2.312 → 2.125 / 2.095 | 0.667 / 0.685 → 0.641 / 0.641 |

* With `mdh` the wall time is dominated by the host boundary. Under this load
  it took 5.9–7.0 s for 5DK3 instead of 3.4 s, and its noise hides the
  ~0.2 s saved.
* With `sdh` the GPU stages dominate, and a calculation is 1.03–1.08× faster.
* The release at the end of a calculation (wall time minus the stages) takes
  0.004 s for 1YCR, 0.025 s for 1TII and 0.068 s for 5DK3.
* MM-PBSA batches (1YCR, 6 frames, h = 0.5 Å) are within noise:
  7.88 / 7.81 → 7.84 / 7.75 s with one worker and 5.08 / 5.17 → 4.84 / 5.26 s
  with three. These batches are limited by the host stages.
* The CPU path is unchanged: the hold is a no-op there. CPU timings under
  this load (1YCR 5–10 s instead of 2.3 s) cannot show a difference.

## The `mdh` boundary: vectorised and overlapped

**Problem.** With the default `bcfl="mdh"` the boundary of the outermost
level is the sum of the Debye–Hückel potentials of all atoms at every
boundary node, on the host. For 5DK3 (20 179 atoms, coarse grid
193 × 313 × 321, 442 370 boundary nodes) that is 8.9e9 terms per state. It
took 3.45 s of a 6.6 s CUDA calculation (2.48 s solvated, 0.98 s reference,
18 threads), while the GPU waited.

**A GPU kernel was ruled out.** The RTX 2080 SUPER runs f64 at 1/32 of its
f32 rate: 96 FP64 units at 1.8 GHz, about 1.7e11 f64 instructions/s. One
term needs a square root, a division and (with salt) an `exp`, roughly 50
f64 instructions with `exp` and 30 without. The 1.8e10 terms of 5DK3 would
take an estimated ~2.6 s (solvated) + ~1.6 s (reference), longer than the
host. Current consumer GPUs run f64 at 1/64. CUDA's `exp` (1 ulp) also
differs from the host's libm in the last bit, so the results would change. A
device kernel pays off only on GPUs with full-rate f64 (A100, H100, ~30× this
card), and only as an option.

**Change.** Both parts keep every result bit for bit.
* AVX2: `DhSources::potentials` already summed four boundary nodes at once,
  each in atom order. Where the CPU has AVX2 (checked at run time) the same
  code is compiled with `#[target_feature(enable = "avx2")]`, and the four
  nodes fill the lanes of a 256-bit register. Lane-wise subtraction,
  multiplication, division and square root are IEEE operations that round
  like the scalar ones, Rust never contracts them into FMAs, and `exp` stays
  the scalar libm call per node. Without salt (the reference state) the term
  has no `exp` and vectorises fully. `mdh-fast` uses the same sums at its
  Chebyshev points.
* Overlap: `polar_solvation` now solves the reference state first; the
  states are independent, so the order changes no result. When the backend
  runs on a device (`Backend::runs_on_device`, true for CUDA), the host
  builds the solvated state's inputs (`StateInputs`: the molecular surface,
  then the analytic boundary of the outermost level) while the reference
  state is solved. This work runs in a rayon pool of its own with one thread
  fewer than the caller's pool, created per calculation (0.35 ms median,
  1.7 ms at most, for 17 threads).
  * The first version spawned it into the caller's pool. The calculation
    became only 1.13× faster, and the solve stage went from 2.0 to 4.4 s. A rayon thread that waits for its own tasks runs
    queued ones meanwhile, so the thread driving the GPU picked up boundary
    work.
  * With a pool of its own the solve stage is unchanged.
* Timings: the stage times stay sequential pieces of the wall time. Only the
  wait for the background work after the reference state counts: as boundary
  time up to what the boundary took (it is built last), beyond that as map
  time.
* On the CPU backend the calculation runs as before, with the reference state
  first: its threads are busy anyway.
* A probe on 5DK3 measured the reference state's GPU work at 0.92–0.98 s,
  against 2.55 s for the solvated inputs in the background (0.21 s surface,
  2.33 s boundary), so 1.6 s of waiting remains. With this structure the
  estimated lower bound is ~4.9 s (measured 5.28 s). Overlapping the solvated state's
  maps, operator and multigrid setup with its boundary would need the
  right-hand side after the setup, a change to `pcg_on`.

**Checks.**
* `vectorised_mdh_terms_equal_the_scalar_ones_bit_for_bit`: 500 atoms (one
  without radius, one without charge) at 4000 points from inside the solute to
  400 Å away, for κ = 0.125 and 0.6 Å⁻¹, no salt and κ = 1e-300; the AVX2 sums
  equal the scalar sums bit for bit. The run-time dispatch is checked too.
  `mdh_equals_the_direct_sum_bit_for_bit` (against `dh` term by term) passes
  through the AVX2 path.
* Core `tests/pipeline.rs` (CI): a recording backend that claims to run on a
  device gives the CPU backend's energies, residuals and iteration counts bit
  for bit for `mdh`, `mdh-fast`, `sdh` and `zero`, in pools of 1 and 3
  threads. Its stage times are non-negative and sum to less than the wall
  time. The memory-hold tests pass with that backend too: a failing
  calculation still releases, and batches on a device worker hold memory.
* The GPU energy test asserts that the CUDA backend reports `runs_on_device`.
* Six mutations, all caught:
  * the AVX2 path with another κ;
  * a reordered term (`/ w / d`);
  * swapped lanes in the dispatch;
  * the solvated inputs built for the reference medium, on every backend or
    only on a device;
  * the boundary of the background pool's threads built for the wrong medium.
* The 13 fingerprint cases (energies, iterations, residuals), maps and MM-PBSA
  species are bit-identical to the previous commit on CPU and CUDA. The
  `cuda-12` build passes the GPU, memory and retention tests. The 1YCR APBS
  comparison on CUDA passes with the same energies (ΔΔG 422.8939 kcal/mol).
  The Born table is unchanged, and pytest passes (109, 2 skipped).

**Time per ΔG_polar** (s). A/B: the previous and the new build ran
alternately, two runs each, each the median of 6 (1YCR), 3 or 1 (5DK3, CPU)
calculations. The RTX 2080 SUPER was shared at 81–99 % utilisation, and the
18 CPU threads had a load of 8–22 from other jobs. Energies are bit-identical
in every row.

| Platform | System | `bcfl` | Previous | New | Speed-up | Boundary stage |
|---|---|---|---|---|---|---|
| CUDA | 1YCR | `mdh` | 0.304 / 0.301 | 0.250 / 0.251 | 1.21× | 0.063 / 0.064 → 0.031 / 0.035 |
| CUDA | 1TII | `mdh` | 2.223 / 2.233 | 1.874 / 1.765 | 1.22× | 0.929 / 0.939 → 0.665 / 0.565 |
| CUDA | 5DK3 | `mdh` | 6.578 / 6.629 | 5.282 / 5.275 | 1.25× | 3.454 / 3.485 → 2.382 / 2.360 |
| CUDA | 1TII | `mdh-fast` | 1.491 / 1.495 | 1.298 / 1.293 | 1.15× | 0.204 / 0.197 → 0.122 / 0.119 |
| CUDA | 5DK3 | `mdh-fast` | 3.546 / 3.543 | 3.037 / 3.055 | 1.16× | 0.413 / 0.437 → 0.182 / 0.181 |
| CUDA | 1TII | `sdh` | 1.306 / 1.321 | 1.187 / 1.199 | 1.10× | 0.018 / 0.020 → 0.017 / 0.016 |
| CUDA | 5DK3 | `sdh` | 3.142 / 3.151 | 2.903 / 2.901 | 1.08× | 0.035 / 0.042 → 0.033 / 0.031 |
| CPU | 1YCR | `mdh` | 2.095 / 2.108 | 2.055 / 2.049 | 1.02× | 0.062 / 0.060 → 0.053 / 0.056 |
| CPU | 1TII | `mdh` | 15.96 / 16.00 | 15.83 / 15.94 | 1.01× | 0.933 / 0.937 → 0.748 / 0.760 |
| CPU | 5DK3 | `mdh` | 37.93 / 38.13 | 37.44 / 37.51 | 1.01× | 3.295 / 3.463 → 2.801 / 2.882 |

* On CUDA the map stage also shrinks (5DK3 0.53 → 0.29 s), because the
  solvated state's surface is built in the background. That is the whole gain
  with `sdh`.
* CPU boundary stage alone (median of 3, 2 and 1 calculations), `mdh` /
  `mdh-fast`: 1YCR 0.062 / 0.041 → 0.052 / 0.038 s, 1TII 0.911 / 0.196 →
  0.773 / 0.181 s, 5DK3 3.535 / 0.432 → 2.905 / 0.370 s. The solvated sum is
  bound by the scalar `exp` (1.15×); the reference sum, without `exp`, gains
  1.7×. A scratch benchmark of the sums alone, 5DK3: solvated 2.48 → 2.17 s,
  reference 0.98 → 0.57 s.
* MM-PBSA batches (1YCR, 6 frames, h = 0.5 Å, identical energies): CUDA,
  1 worker 4.87 / 4.92 → 4.22 / 4.20 s (1.16×); 3 workers 3.32 / 3.29 →
  3.28 / 3.18 s, within noise (the other workers already fill the GPU while
  one computes its boundary). CPU 1 worker 36.7 / 37.1 → 37.3 / 36.9 s and
  2 workers 27.4 / 27.3 → 27.4 / 26.7 s, unchanged.

## Solvated setup before its boundary

**Problem.** After the previous step a 5DK3 calculation on CUDA still waited
about 1.6 s for the solvated state's outermost boundary, which the host
builds in the background. Only the right-hand side of that level needs the
boundary values. The level's maps, operator and multigrid setup (about
0.25 s for 5DK3) waited as well.

**Change.**
* `solver::pcg_setup` sets up the preconditioner before the right-hand side
  is known; `PreparedPcg::solve` then runs the iterations. Together they
  compute what `pcg_on` computes, including the b = 0 case (no iterations, one
  level reported); `pcg_on` itself is unchanged.
* The pipeline passes host inputs as `Pending` values: ready, or still being
  built in the background and delivered through a channel. A level whose
  boundary is pending runs maps → operator → multigrid setup → wait → right-hand
  side → iterations. The setup consumes the inverse diagonal, which the
  right-hand side needs too, so a copy is kept. The copy takes the place b
  held during the setup, and `tests/memory.rs` confirms that the memory peak
  is unchanged (to the byte).
* The background sends the solvated surface, then its outermost boundary.
  If the calculation fails while the surface is being built, the boundary is
  skipped. `solve_focused` on CUDA uses the same path for its single state.
* A first version also built the reference state's boundary in the
  background, to hide the reference setup as well. Measured on 5DK3, the
  background chain grew from 3.1 to 3.4 s (17 threads instead of 18, and
  competition with the host work of the reference state), and the calculation
  became no faster. The reference boundary is therefore built first, with all
  threads, while the GPU has nothing to do yet.
* Timings: the wait at the level counts as boundary time, the wait for the
  surface as map time.

**Checks.**
* `pending_boundary_values_are_waited_for_after_the_setup`: a test backend
  sends the boundary values from inside its first Galerkin product, so they
  arrive during the setup. A wait before the setup would never end; the test
  times out after 120 s if it does. The solution equals the one from ready
  values bit for bit (`multigrid-f32` and `multigrid`).
* `a_prepared_solve_equals_pcg_on`: three preconditioners, with b ≠ 0 and
  b = 0, the same solution, iterations, residual and levels.
* Core `tests/pipeline.rs`:
  * the recording backend checks the call order. On a device, the solvated
    outermost level (and that of `solve_focused`) runs its Galerkin products
    before `source_rhs`. The reference state, the focusing levels and every
    level on the CPU build the right-hand side first;
  * a molecule without charges now runs through the device path too.
* Five mutations, all caught:
  * no copy of the inverse diagonal;
  * the wait moved before the setup;
  * no b = 0 branch in `PreparedPcg::solve`;
  * the background built for the reference medium;
  * the background boundary built on the wrong level.
* The 13 fingerprint cases, maps and MM-PBSA species are bit-identical to the
  previous commit on CPU and CUDA. The `cuda-12` build passes the GPU, memory
  and retention tests. The 1YCR APBS comparison on CUDA passes with the same
  energies (ΔΔG 422.8939 kcal/mol). The Born table is unchanged, and pytest
  passes (109, 2 skipped).

**Time** (s). A/B against the previous commit: alternating builds, two runs
each, each the median of 6 (1YCR) or 3 calculations. The GPU was shared at
81–98 % utilisation, and the CPU load was 7–19. Energies and potentials are
bit-identical in every row.

| Calculation | System | Previous | New | Boundary stage |
|---|---|---|---|---|
| CUDA, `mdh` | 1YCR | 0.254 / 0.251 | 0.248 / 0.249 | 0.031 / 0.033 → 0.036 / 0.027 |
| CUDA, `mdh` | 1TII | 1.761 / 1.746 | 1.665 / 1.690 | 0.577 / 0.539 → 0.507 / 0.501 |
| CUDA, `mdh` | 5DK3 | 5.425 / 5.344 | 5.041 / 5.086 | 2.542 / 2.474 → 2.088 / 2.227 |
| CUDA, `Solver.potential` | 1TII | 1.637 / 1.645 | 1.557 / 1.590 | |
| CUDA, `Solver.potential` | 5DK3 | 4.656 / 4.881 | 4.641 / 4.531 | |

* A longer run on 5DK3 (5 alternations, 15 calculations per build) gave
  medians of 5.42 → 5.10 s (1.06×) and boundary waits of 2.47 → 2.19 s.
  Setup and maps of one level of 5DK3 take about 0.25 s, so most of the
  hidden setup appears in the wall time.
* Unchanged within noise: `sdh` and `mdh-fast` on CUDA (their boundaries take
  0.03–0.19 s, so little waiting is left to hide); the CPU (1YCR 2.07 /
  2.07 → 2.08 / 2.04 s, 1TII 15.75 / 15.81 → 15.79 / 15.78 s); MM-PBSA
  batches of 1YCR on CUDA (1 worker 4.17 / 4.17 → 4.19 / 4.24 s, 3 workers
  3.21 / 3.27 → 3.21 / 3.34 s) and on the CPU (2 workers 26.8 / 26.6 →
  27.0 / 26.6 s).

## Fused multigrid smoother

**Problem.** A profile of a 1TII calculation on CUDA (nsys) showed that the
f32 V-cycle spends about as much time in its element-wise updates (`axpy`,
`scale`, `axpby` and device-to-device copies: ~250 ms) as in its stencil
applications (~195 ms). Each smoothing step of a level ran as separate
kernels, every one reading and writing whole vectors: per level and V-cycle
23 kernels and copies (restriction and prolongation included), 4 of them
stencils; smoothing and residual moved about 284 bytes per node of a 7-point
level in f32. The CPU loops have the same structure.

**Change.**
* Three smoother steps in the `Backend` trait, with defaults that run their
  operations one by one (`backend::residual_by_steps`,
  `chebyshev_first_by_steps`, `chebyshev_next_by_steps`):
  * `residual`: r ← b + (−1)·(A x), or r ← r + (−1)·(A x) in place (`apply`,
    `copy`, `axpy`);
  * `chebyshev_first`: d ← (s·w)·r, then x ← d or x ← x + 1·d (`scale`,
    `copy` or `axpy`);
  * `chebyshev_next`: d ← 1·((c1·w)·r) + c2·d, x ← x + 1·d (`scale`,
    `axpby`, `axpy`).
* The CPU backend evaluates each step in one parallel loop, the CUDA backend
  in one kernel (`pb_residual*`, `pb_cheb_first`, `pb_cheb_next`), with the
  separate operations' expressions in the same order, so every element
  rounds as before. The stencils are shared: `apply7_with`/`apply27_with` on
  the CPU and `stencil7_at`/`stencil27_at` in CUDA, which `apply` also uses.
* The smoother runs one residual update and one vector pass per degree; with
  a zero initial guess the residual starts as b itself, which the old code
  copied first. A level now runs 10 kernels instead of 23, and smoothing
  and residual move about 196 bytes per node instead of 284. The scratch vectors stay allocated, so
  the device memory of a solve is unchanged (`tests/memory.rs` passes to the
  byte).

**Checks.**
* `the_fused_smoother_equals_the_step_by_step_smoother`: one V-cycle (applied
  twice) with the fused smoother and with the old step-by-step code, kept in
  the tests: equal bit for bit for f64 and f32, Chebyshev degrees 1, 2 and 3,
  on 41³, 66×41×6 (semi-coarsening) and 50×37×29 grids.
* `fused_smoother_steps_equal_their_sequences` (CPU) and
  `fused_smoother_steps_on_the_gpu_equal_their_sequences` (CUDA): every step,
  on the 7-point and a 27-point operator, both residual forms and both first
  steps, equals its sequence bit for bit; on the GPU also equal to the CPU.
* `gpu_vcycle_equals_the_cpu_vcycle_bit_for_bit`: a whole V-cycle, both
  precisions.
* Six mutations, all caught:
  * c1·(w·r) instead of (c1·w)·r;
  * a residual from r instead of b;
  * the first step ignoring `overwrite`;
  * the smoother using b at every degree (caught by the degree-3 case);
  * the CUDA update of x reading d before it is updated;
  * the 27-point sum in reverse order on the GPU.

  A first version of the CUDA mutation read `d[c]` after the kernel had
  already stored the new value there, so it changed nothing and nothing failed;
  the corrected mutation is caught.
* The 13 fingerprint cases, maps and MM-PBSA species are bit-identical to the
  previous commit on CPU and CUDA. The GPU, memory and retention tests pass
  with the `cuda-12` and `cuda-13` builds. The Born table is unchanged, and the
  1YCR APBS comparison on CUDA passes with the same energies (ΔΔG 422.8939
  kcal/mol).

**Time** (s). A/B against the previous commit: alternating builds, two runs
each, each the median of 6 (1YCR CUDA), 5 (1YCR CPU), 3 or 2 (5DK3 CPU)
calculations. Energies are bit-identical in every row. The CPU load from
other work was about 1; the GPU was shared at 79–97 % utilisation.

| Calculation | System | Previous | New | Solve stage |
|---|---|---|---|---|
| CPU, `multigrid-f32` | 1YCR | 2.067 / 2.060 | 1.765 / 1.767 | 1.517 / 1.501 → 1.218 / 1.219 |
| CPU, `multigrid-f32` | 1TII | 15.96 / 15.79 | 13.16 / 13.17 | 12.93 / 12.75 → 10.16 / 10.13 |
| CPU, `multigrid` | 1TII | 22.67 / 22.66 | 18.16 / 18.20 | 19.87 / 19.80 → 15.38 / 15.40 |
| CPU, `multigrid-f32` | 5DK3 | 37.66 / 37.91 | 32.22 / 32.47 | 29.25 / 29.40 → 24.05 / 24.06 |
| CUDA, `mdh` | 1YCR | 0.248 / 0.254 | 0.238 / 0.240 | 0.127 / 0.139 → 0.117 / 0.117 |
| CUDA, `mdh` | 1TII | 1.684 / 1.721 | 1.582 / 1.659 | 0.857 / 0.839 → 0.687 / 0.717 |
| CUDA, `mdh`, `multigrid` | 1TII | 2.094 / 2.140 | 1.904 / 1.933 | 1.389 / 1.405 → 1.111 / 1.099 |
| CUDA, `mdh` | 5DK3 | 4.957 / 5.101 | 4.667 / 4.616 | 1.940 / 1.930 → 1.683 / 1.639 |
| CUDA, `mdh-fast` | 5DK3 | 3.026 / 3.039 | 2.727 / 2.741 | 1.960 / 1.968 → 1.630 / 1.645 |

* The CPU gains 1.17–1.25× per calculation. Its solve stage is most of the
  time there.
* On CUDA the solve stage gains 1.17–1.26×. A calculation with `mdh` gains
  less (1.05–1.08×), because the reference state now finishes sooner and the
  GPU waits longer for the solvated state's boundary, which the host builds
  (1TII: 0.47–0.53 → 0.57 s). With `mdh-fast`, whose boundary takes 0.2 s,
  5DK3 gains 1.11×.
* The CUDA numbers come from a shared GPU and should be repeated on an idle
  one.

## Fused CG steps

**Problem.** After the smoother (previous section), the f64 vector work of
the CG iteration was the next cost. Each iteration ran nine passes over the
fine grid: `apply` and `dot(p, q)`, two `axpy`, `dot` (or `dot2` for ‖r‖² and
rᵀz in flexible CG), the f32 copy of r for the mixed-precision V-cycle, the
f64 copy of its result, `dot(r, z)` and `xpay`. On CUDA these kernels took
about 220 ms of a 1TII calculation (nsys).

**Change.**
* Three CG steps in the `Backend` trait, with defaults that run their
  operations one by one (`backend::apply_dot_by_steps`, `cg_update_by_steps`,
  `promote_dot_by_steps`):
  * `apply_dot`: q ← A p; returns p·q (`apply`, `dot`);
  * `cg_update`: x ← x + α·p and r ← r + (−α)·q; returns ‖r‖² and, for
    flexible CG, rᵀz; writes the f32 copy of r when the V-cycle runs in f32
    (`axpy`, `axpy`, `dot` or `dot2`, `demote`);
  * `promote_dot`: z ← z₃₂ widened to f64; returns rᵀz (`promote`, `dot`).
* The CG loop calls them. The f32 copy of the residual is now written by the
  update rather than at the start of the V-cycle (`Precond::rounded_residual`,
  `Precond::apply_dot` in `solver.rs`).
* Every dot product keeps its summation order:
  * **CPU.** Each 2¹⁴-element chunk of `dot` is updated, then summed by the
    function `dot` itself uses (`chunk_dot`), while it is still in cache.
    A p is computed chunk by chunk (`operator::apply7_into`, which shares the
    stencil expression with `apply7_with`).
  * **CUDA.** `pb_cg_update` and `pb_promote_dot` run on the grid of `dot`
    (at most 1024 blocks), and each thread adds the same elements in the
    same order. The block sums use `block_sum`, as `pb_dot_partial` does.
* **CUDA keeps `apply` followed by `dot` for `apply_dot`.** A fused kernel has to
  run on the dot's grid, whose blocks run in waves. The stencil then no longer
  finds the neighbours one plane away in the L2 cache. On a 165³ grid, A p
  alone takes 1.07 ms on that grid instead of 0.73 ms with one thread per
  node, and the fused kernel (1.07–1.10 ms) was slower than the pair
  (0.95 ms). Letting each launched block hold several of the dot's blocks,
  so that all blocks sweep memory together, gave no reliable gain on the
  shared GPU.
* Per iteration the CPU makes 4 passes instead of 9, and CUDA runs 5 kernels
  instead of 9. No vector is added; `tests/memory.rs` passes to the byte.

**Checks.**
* `the_fused_cg_iteration_equals_the_step_by_step_iteration`: three solves
  give the same solution, iteration count and residual bit for bit:
  * the CG loop as it was before the change, kept in the tests
    (`iterate_by_steps`);
  * the new loop;
  * a solve through `backend::Stepwise`, a wrapper that runs every fused step
    of a backend and of its f32 twin, smoother included, as its sequence.

  Covered: all three preconditioners, converged (tol 1e-10) and stopped after
  3 iterations, on 17³, 41³, 66×41×6, 50×37×29 and 9×11×8 grids (less than
  one to four CPU chunks, chunk boundaries inside rows, the Jacobi fallback).
* `fused_cg_steps_equal_their_sequences` (CPU, f64 and the f32 twin) and
  `fused_cg_steps_on_the_gpu_equal_their_sequences` (CUDA, both
  precisions): every step equals its sequence bit for bit. Covered: with and
  without z and the f32 copy, both signs of α, from 60 to 4.2 million nodes
  (CPU lengths around the chunk size; on CUDA from one block up to 16
  elements per thread of the reduction grid). A sum of −0.0 products keeps
  its sign. On the GPU the vectors also equal the CPU's.
* `gpu_cg_solves_equal_the_stepwise_solves_bit_for_bit`: whole CUDA solves
  equal the `Stepwise` solves (three preconditioners, 49³ and 81³, converged
  and stopped).
* Twelve mutations, all caught:
  * CPU:
    * another summation split in `apply_dot`;
    * the f32 copy of r taken from the old residual;
    * rᵀz taken from the old residual;
    * A p computed on the last column of each row;
    * `promote_dot` summing the old z;
    * no f32 residual passed to the V-cycle;
    * no initial f32 copy.
  * CUDA:
    * ‖r‖² of the old residual;
    * `promote_dot` on 512 blocks;
    * the rᵀz block sums in reverse order;
    * x updated with −α;
    * the two sums swapped.

  Only the new tests catch the mutations that change nothing but the order
  of a sum.
* The 13 fingerprint cases, maps and MM-PBSA species are bit-identical to the
  previous commit on CPU and CUDA. The GPU, memory and retention tests pass
  with the `cuda-12` and `cuda-13` builds. The Born table is unchanged, and
  the 1YCR APBS comparison on CUDA passes with the same energies.

**Time.** Each step against its sequence, on a 165³ grid (µs, best of 6×20):

| Step | CPU fused | CPU by steps | CUDA fused | CUDA by steps |
|---|---|---|---|---|
| `apply_dot` | 10 950–11 090 | 13 090–13 220 | (not fused) | 910–960 |
| `cg_update` (with z and the f32 copy) | 8 940–9 090 | 12 620–13 170 | 719–742 | 982–1011 |
| `promote_dot` | 2 780–2 820 | 3 070–3 130 | 270–275 | 348–352 |

A/B against the previous commit: alternating builds, two runs each, each the
median of 6 (1YCR CUDA), 5 (1YCR CPU), 4, 3 or 2 calculations. Energies are
bit-identical in every row. The CPU load from other work was about 1. The GPU
was shared at 75–98 % utilisation.

| Calculation | System | Previous | New | Solve stage |
|---|---|---|---|---|
| CPU, `multigrid-f32` | 1YCR | 1.827 / 1.787 | 1.734 / 1.702 | 1.261 / 1.216 → 1.168 / 1.152 |
| CPU, `multigrid` | 1YCR | 2.545 / 2.578 | 2.465 / 2.470 | 2.046 / 2.061 → 1.962 / 1.964 |
| CPU, `jacobi` | 1YCR | 17.36 / 17.40 | 15.44 / 15.39 | 17.10 / 17.14 → 15.17 / 15.13 |
| CPU, `multigrid-f32` | 1TII | 13.18 / 13.24 | 12.33 / 12.29 | 10.15 / 10.17 → 9.28 / 9.23 |
| CPU, `multigrid` | 1TII | 18.22 / 18.27 | 17.66 / 17.70 | 15.43 / 15.47 → 14.89 / 14.95 |
| CPU, `multigrid-f32` | 5DK3 | 32.34 / 32.39 | 30.23 / 30.24 | 24.04 / 24.05 → 21.92 / 21.80 |
| CUDA, `jacobi` | 1YCR | 1.505 / 1.515 | 1.458 / 1.440 | 1.444 / 1.459 → 1.397 / 1.388 |
| CUDA, `mdh` | 1TII | 1.601 / 1.536 | 1.592 / 1.642 | 0.715 / 0.707 → 0.712 / 0.682 |
| CUDA, `mdh`, `multigrid` | 1TII | 1.890 / 1.925 | 1.919 / 1.902 | 1.107 / 1.117 → 1.087 / 1.096 |
| CUDA, `mdh` | 5DK3 | 4.862 / 4.794 | 4.798 / 4.711 | 1.604 / 1.655 → 1.590 / 1.599 |
| CUDA, `mdh-fast` | 5DK3 | 2.750 / 2.760 | 2.678 / 2.686 | 1.639 / 1.641 → 1.583 / 1.573 |

* The CPU gains 1.03–1.07× per calculation with multigrid. With Jacobi it
  gains 1.13×, because Jacobi's many iterations are mostly vector work. The
  solve stage gains up to 1.10× (1TII, 5DK3).
* On CUDA the f64 CG kernels of a 1TII calculation take 198 ms instead of
  219 ms (nsys, all kernels 878 → 859 ms). That is 2–4 % of the solve stage
  and within the noise of a shared GPU for most rows; `jacobi` and
  `mdh-fast` show 1.03–1.04×. The CUDA numbers should be repeated on an
  idle GPU.

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

