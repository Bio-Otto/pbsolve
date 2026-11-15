# Validation

A PB code is only as trustworthy as its validation record. Every release
should update the status table below.

## Protocol

1. **Analytic references.** Born ion (unit charge, a = 2 Å, ε 1 → 78.54) and
   Kirkwood sphere with off-centre charges and salt
   (`pbsolve.analytic.kirkwood_energy`); each with its observed convergence
   order, averaged over grid placements (`scripts/validate_analytic.py`).
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

## Status (0.0.4)

| Check | Result |
|---|---|
| Born ion, `mol`, h = 0.500 / 0.333 / 0.250 / 0.200 Å | −4.73 % / −1.08 % / −1.27 % / −0.94 % vs analytic (−81.96 kcal/mol); converging overall, not monotonically (before the APBS-exact surface: −4.73 / −1.09 / −1.23 / −0.71 %) |
| Born ion, `harmonic`, same spacings | −4.16 % / −1.08 % / −1.65 % / −1.09 % (unchanged) |
| Spline surfaces and forces (0.0.2): `surface="spl2"`/`"spl4"`, `Solver.polar_forces` | APBS 3.4.1 spline maps equal to the 7 digits of its DX files, energies ≤ 1e-9 (solvated) and 3e-10 (ΔG, `bcfl zero`) in APBS units; forces (qf, db, ib; `lpbe`, `npbe`, `smpbe`) = the exact gradient of the discrete energy on one level with `bcfl zero` (2e-9 to 4e-9), APBS `calcforce comps` to ≤ 4e-7 of the largest force (≤ 6e-10 with `zero`); a charge in a dielectric sphere: Kirkwood's force at the surface's Born radius to 0.08 %, momentum balanced to 1 %; db scatters 5–10 % over grid placements; spline surfaces are a different model (Born −11…−16 %, 1YCR −2609 vs −1231 kcal/mol with `mol`); 14/14 mutations; default path bit-identical; `coulomb="analytic"` on spline surfaces fixed (now equal to `coulomb="grid"` within 1e-4 on an exact radial case); see *Spline surfaces and forces* |
| Gaussian dielectric (0.0.2): `surface="gauss"`, `gauss_sigma`, `gauss_contour` | maps = the formulas to 1e-9 (ε and the ions' Born accessibility); reference state with the Gaussian ε inside the ε ≤ 20 contour (Li et al. 2014): one Gaussian atom against its exact radial solution within 1.0–3.2 % at 0.5 Å and 0.45–1.4 % at 0.18 Å, salt effect to 0.07 %; with a homogeneous reference 17–63 % off (charges' grid self-energies); 1YCR ΔΔG_polar 385.9 / 383.3 / 383.5 kcal/mol at 0.49 / 0.35 / 0.25 Å (`mol` 434.9 / 429.0 / 427.6); `npbe` refused (Born penalty bounded, density diverges at charges); CPU = CUDA; 9/9 mutations; default path bit-identical; see *Gaussian dielectric* |
| Non-polar terms (0.0.2): `apolar_energy` (γ·SASA + p·SAV + WCA dispersion, as APBS `apolar`) | APBS' 11 alkane examples to every printed digit (totals ≤ 2.2e-13); APBS 3.4.1 live on neopentane, a synthetic molecule with radius-0 atoms and 1YCR (1670 atoms) ≤ 1.3e-12, volumes identical, per atom ≤ 5e-13; one sphere, two spheres and one atom's dispersion against closed forms (0.2–0.5 %); Lennard-Jones from OpenMM = APBS' AMBER parameters to 1e-9; 10/10 mutations; 1YCR 0.75 s on 18 threads (APBS 12.7 s); see *Non-polar terms* |
| Per-atom and per-residue decomposition (0.0.3): `PolarResult.atom_energies`, `PBSAFrame.atom_terms`/`residue_terms`, `pbsolve mmpbsa --per-residue` | shares sum to ΔG_polar (1e-12; `spl0`, `spl2`, `spl4`, `coulomb="analytic"`, forces path identical); linear response: E_i of all charges = E_i alone + pair terms from single and pair solves (1e-8, focusing and `mdh`), pair terms symmetric with one level and `zero` (1e-8); APBS 3.4.1 `calcenergy comps` per atom ≤ 2.8e-6 of the largest share with `mdh`, 1e-9 (synthetic) and 3.2e-8 (1YCR) with `zero`; 9/9 mutations; energies bit-identical (CPU, CUDA), the shares add ≤ 50 ms to 5DK3; see *Per-atom and per-residue decomposition* |
| Interaction entropy (0.0.4): `interaction_entropy`, `interaction_energy` | IE and C2 of two-level energy distributions = closed form (1e-12); normal energies: C2 = σ²/2kT, IE → C2, and the low bias of IE at large σ reproduces Ekberg & Ryde 2021 (σ = 3 kcal/mol: IE 7.0 ± 0.4 vs exact 7.60 with 10⁵ frames, C2 7.59 ± 0.03); no overflow at σ = 200 kcal/mol; E_int = OpenMM 8.6 (Reference platform, ff14SB peptides from tleap, bonds and 1-4 exceptions inside the molecules) to 1.7e-12, and = an explicit pair sum to 1e-12; 16/16 mutations; see *Interaction entropy* |
| Atom dielectrics (0.0.5): `dielectrics=`, `residue_dielectrics`, `--residue-dielectric`, `PBSAFrame.delta_reference` | equal values reproduce `pdie` bit for bit (maps of `mol` and `vdw`, both states, `lpbe` and `npbe`, every Python method; fingerprint of the default path unchanged); every interior face owned by the atom of smallest power distance (brute force); a central charge sees only the outer boundary for any interior ε (0.25 Å: −1.50/−0.14/+0.64 % for ε 1/4/20); one sphere against Kirkwood with its own interior ≤ 0.92 % at 0.25 Å; two spheres with interiors 4 and 10 against the exact solution (`two_sphere_energy(sphere_pdie=)`) −0.32 % and −0.74 % at 0.25 Å (one `pdie`: −1.68 and −1.58 %); `delta_reference` = Coulomb/`pdie` exactly with `coulomb="analytic"`, to 0.03 % for 1YCR at 0.35 Å on the grid; CPU = CUDA ≤ 8e-14; 30/30 mutations; CPU +6 %, CUDA 2.2× (host maps); see *Atom dielectrics* |
| Size-modified PB (0.0.2): `equation="smpbe"`, `ion_size` | lattice-gas ions of one size (APBS 3.4.1 disables `smpbe`); planar double layer against the exact profile: errors 2e-4 (npbe) to 4e-7 at 0.125 Å, orders 2.6–3.8; charged cavity [Silalahi et al. 2010]: entropic cost ∝ a³ (slope 3.00), counterions saturate at 1/(N_A a³) where point ions reach 84 M; functional stationary; a → 0 gives npbe; `npbe` bit-identical; CPU = CUDA; 9/9 mutations; cost as npbe; see *Size-modified PB* |
| Nonlinear PB (0.0.2): `equation="npbe"` | inexact Newton on the multigrid linear solver, APBS' functional with the coarser levels' outer terms; APBS 3.4.1 `npbe` (1YCR, −8/−20/−30 e test systems, 1:1 and 2:1 salts, 1–3 levels): identical maps, solvated energies ≤ 2.3e-9 and nonlinear corrections to 1e-4 kcal/mol in APBS units; functional stationary at the solution; weak potentials give the linearized potential; 15/15 mutations caught; CPU = CUDA ≤ 1e-10; default `lpbe` bit-identical; Newton on the device from a damped start, cost 1.3–1.4× (CPU), 1.7–1.9× (CUDA), 4 extra device vectors (asserted); see *Nonlinear Poisson–Boltzmann* |
| Reaction-field decomposition (0.0.2): `coulomb="analytic"` | one solve per level, no reference state; sparse right-hand side = dense χ·A_pG − A(χG) to 1e-11 and zero off the layer; Rust = Python prototype to 4e-15; default `grid` bit-identical (CPU, CUDA); CPU–CUDA ≤ 1e-10; 11/11 mutations caught; fractional interface: placement spread halves (analytic references and 1YCR), same limits as `grid`, 1YCR ΔG_polar error at 0.5 Å −5.6 instead of −10.1 kcal/mol; sharp interface: ≤ 0.16 points; 1.7× faster on the CPU, 7–20 % on CUDA (1YCR), slower for 5DK3 on CUDA (direct Coulomb sum); see *Reaction-field decomposition* |
| Fractional dielectric interface (0.0.2): `interface="fractional"` | faces split exactly against an independent computation (≤ 1e-12); default `sharp` bit-identical (CPU, CUDA); device maps = host maps; 10/11 mutations caught; analytic references with `vdw`: observed orders 1.6–2.5 (sharp 1.0–1.2), errors at 0.5 Å −0.14 … −0.71 % (sharp −2.3 … −4.7 %); with `mol` a +0.13–0.18 % plateau from APBS' sampled probe positions (gone with `sdens=100`); 1YCR placement sd 5–10× smaller, same extrapolated limit as sharp, ΔΔG_bind bias at 0.5 Å +2.7 instead of +13 kcal/mol; calculation 1.5× slower on CUDA, same on CPU; see *Fractional dielectric interface* |
| Error estimates (0.0.2): `estimate_error`, `pbsolve solvation --placements --extrapolate` | first placement = ordinary calculation bit for bit; on the analytic references, extrapolation from 0.35 and 0.25 Å leaves ≤ 0.3 % (means 1.1–1.8 % off), estimated errors within 30 % of the true ones; from 0.5/0.35 Å off by up to 2×; 1YCR: ΔG_polar −1234.7 / −1205.6 / −1190.5 kcal/mol at 0.48 / 0.34 / 0.25 Å, extrapolated −1152, ΔΔG_bind bias ~+6 kcal/mol at 0.25 Å; see *Error estimates* |
| B-spline charges (0.0.2): `chgm` `spl2`, `spl4` as APBS | weights: charge and centre to 1e-12, variance h²/4 and 5h²/12, exact on-node values; energies at the charged nodes = full-potential energies bit for bit (CPU, CUDA); `spl0` bit-identical to the previous commit; APBS 3.4.1: synthetic molecule and 1YCR complex with maps identical and ΔG ≤ 1.3e-6, ΔΔG +0.0012 kcal/mol in APBS units, as `spl0`; 6/6 mutations caught; accuracy: analytic errors and orders unchanged within 0.3 points, 1YCR placement sd 4.09 → 4.07 / 3.88 kcal/mol; cost on 5DK3 +1 % / +4 % (CUDA); see *Charge assignment: B-splines* |
| Two spheres (0.0.2): 5 and 4 Å spheres 15.8 Å apart, 0 and 0.15 M, complex and association ΔΔG on the complex's grid | reference: exact limits (no dielectric jump 7e-15, conductors 3e-12, far apart with salt 3e-13, charge–dipole asymptote) and converged to 1e-11; pbsolve: complex first order (1.11–1.17 `mol`, 1.02–1.06 `vdw`), association ΔΔG of 1.8 kcal/mol within 0.002 kcal/mol on average at every spacing with `vdw`; with `mol` 0.011–0.019 kcal/mol from APBS' sampled molecular surface; see *Analytic references: two spheres* |
| Kirkwood sphere (0.0.2): charges off the centre of a 5 Å sphere, 0 and 0.15 M, pdie 1 and 4; Born ion | reference = closed forms to ≤ 1e-13 (Born, Born–Debye–Hückel with Stern layer, grounded-sphere images for several charges) and the interface conditions to 1e-12; pbsolve, averaged over 20 grid placements: −2.3 to −2.9 % at 0.5 Å, −0.8 to −1.0 % at 0.2 Å, observed order 1.09–1.18 (first order, as expected for the sharp boundary); charges 1 Å below the surface scatter from −11 to 0 % per placement at 0.5 Å; see *Analytic references: Kirkwood sphere* |
| Operator symmetry, charge conservation, SES crevice, SASA, APBS reference-sphere count, marking = brute force | pass (unit tests) |
| Multigrid preconditioner (0.0.2): Pᵀ = R, linear interpolation, Galerkin = RAP, exact symmetry/SPD, coarsest solve, SPD V-cycle, contraction 0.23 per V-cycle, thread-count independence, multigrid = Jacobi energies | pass (`multigrid.rs`, `pb.rs`, `tests/test_multigrid.py`); CG iterations 10–19 independent of spacing vs 155–688 for Jacobi; Born table unchanged; see [Linear solver](#linear-solver-multigrid-versus-jacobi) |
| APBS 3.4.1 parity, synthetic 30-atom molecule (crevices, zero-radius atoms), `mdh`/`sdh`/`zero`, `mol`/`vdw`, 0–0.15 M, 2- and 3-level focusing, Born ion | all maps identical; energies within 1e-7 (solvated) / 5e-6 (reference, `mdh`/`sdh`) in APBS units — `tests/test_apbs_parity.py` |
| APBS 3.4.1 parity, MDM2–p53 peptide complex (PDB 1YCR, pdb2pqr 3 `--ff=AMBER`, 1670 atoms, 18 zero-radius H; ligand = chain B) | see table below: 0 / 2 146 689 differing nodes in every map; ΔG_polar +0.0012 % raw, ≤ 1.2e-6 in APBS units; ΔΔG_bind within 0.004 kcal/mol |
| APBS 3.4.1 parity, larger proteins: 1HPV (3 137 atoms; `mdh`, `sdh`, `zero`), 1TII (10 862) and 5DK3 (20 179 atoms, 193×321×321 grid), complex/receptor/ligand each | every map identical (1HPV, 1TII; 5DK3 compared without maps); ΔG_polar ≤ 4.0e-6 in APBS units; ΔΔG_bind within 0.042 kcal/mol in APBS units (\|ΔΔG\| up to 6 232 kcal/mol); with `zero` ≤ 1.6e-10 per state and ΔΔG 7.6e-7 kcal/mol, so the remainder is APBS' boundary constants; the binding verdict now uses APBS units (the raw difference contains ΔΔG × 1.2e-5); 1YCR in the test suite; see *APBS parity: larger proteins* |
| Mixed precision (0.0.2, the default) | `multigrid-f32`: ΔG_polar within 7.9e-13 of f64 (Born, 1YCR), equal iteration counts; 1YCR 1.42–1.55× faster on the CPU, 1.03–1.12× on the shared GPU (V-cycle 2× on the device). Default after a sweep of 28 cases (5 systems up to 20 179 atoms, 14 parameter variants, CPU and CUDA): ≤ 4.2e-13, identical iterations, CPU 1.4–1.7× on proteins; Born table and 1YCR APBS comparison unchanged; see *Mixed precision* |
| Host stages (0.0.2) | Faster maps, `mdh` boundary, operator setup and host transfers; every energy, iteration count, residual, map and potential bit-identical to the previous commit on CPU and CUDA (13 cases + MM-PBSA species); 1.06–1.26× (CPU), 1.03–1.19× (CUDA) per calculation; see *Host stages* |
| `bcfl="mdh-fast"` (0.0.2, opt-in) | Interpolated `mdh`: ΔG_polar within 1.1e-13 of `mdh` (Born, 1YCR ×4, 1HPV, 1TII, 5DK3; CPU and CUDA), equal iterations; boundary 5DK3 3.39 → 0.55 s; 1YCR APBS comparison prints identical energies; see *mdh-fast* |
| Multigrid setup on the GPU (0.0.2) | Galerkin operators, 1/diag, Gershgorin bounds and f32 copies built on the device: bit-identical to the host hierarchy (4 operators incl. semi-coarsening, all levels) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); CUDA setup 2.2–3.1×, calculation 1.2–1.3× faster on a shared GPU; see *Multigrid setup on the GPU* |
| Maps and right-hand side on the GPU (0.0.2) | Dielectric and ion maps, operator, node charges and right-hand side assembled through the backend: bit-identical to the host (sphere marking incl. constructed boundary and summation-order cases, all surfaces and media, random maps, whole solves) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); APBS 1YCR on CUDA unchanged; CUDA calculation 1.23–1.32× faster on a shared GPU, CPU unchanged; see *Maps and right-hand side on the GPU* |
| CUDA workers from free memory (0.0.2) | The device memory of a calculation is counted exactly (estimate = memory-pool peak to the byte on every measured grid up to 19.4 M nodes, 3 preconditioners); batches run as many CUDA workers as fit in the free memory (at most 3 by default): 3 × 5DK3 now runs on 2 workers instead of failing with out-of-memory on an 8 GB card; 1YCR and 1TII keep 3 workers; every result bit-identical to the previous commit (CPU and CUDA); see *CUDA workers from the free device memory* |
| Potentials read where needed (0.0.2) | The solution stays on the backend: focusing reads it at the stencil nodes of the next level's boundary, the energy at the charged nodes; energies and potentials bit-identical to the full-potential path and to a host reference of the old pipeline (7 core cases, 4 GPU cases) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); CUDA calculation 1.06–1.10× faster on a shared GPU, 0.5 GB less host memory for 5DK3; see *Potentials read only where they are needed* |
| GPU memory kept during a calculation (0.0.2) | The CUDA memory pool keeps the buffers a calculation frees between focusing levels and returns them when it ends (release threshold raised by the calculation's `worker_device_bytes`, restored and trimmed at the end, also on errors): exactly one hold per calculation (core test with a recording backend, CI), threshold and pool usage back to idle after single, failing and concurrent calculations (GPU test), 10/10 mutations caught; every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); APBS 1YCR on CUDA unchanged; solve stage 1.05–1.10× faster, a calculation 1.03–1.08× with `sdh` on a shared GPU; see *GPU memory kept during a calculation* |
| `mdh` boundary vectorised and overlapped (0.0.2) | Exact `mdh` sums on AVX2 (bit-identical to the scalar sums and the direct `dh` sum) and, on CUDA, the solvated state's surface and outermost boundary built while the reference state is solved: every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds), also from a recording device backend with 1 and 3 threads (CI); 6/6 mutations caught; APBS 1YCR on CUDA unchanged; CUDA calculation 1.21–1.25× faster with `mdh` (5DK3 6.60 → 5.28 s), CPU boundary 1.18–1.24×; a GPU `mdh` kernel ruled out on consumer GPUs; see *The `mdh` boundary: vectorised and overlapped* |
| Solvated setup before its boundary (0.0.2) | On CUDA the solvated state's outermost level builds maps, operator and multigrid before waiting for its boundary from the background (`pcg_setup` + `PreparedPcg::solve` = `pcg_on` bit for bit, also for b = 0): a test backend that delivers the boundary only during the setup proves the order; every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 5/5 mutations caught; APBS 1YCR on CUDA unchanged; CUDA `mdh` 1.05–1.06× (5DK3 5.42 → 5.10 s); see *Solvated setup before its boundary* |
| Fused multigrid smoother (0.0.2) | The V-cycle's residuals and Chebyshev updates run as one pass each (`Backend::residual`, `chebyshev_first`, `chebyshev_next`; CPU loops and CUDA kernels with the separate operations' expressions): equal bit for bit to the step-by-step smoother (f64 and f32, degrees 1–3, semi-coarsening; CPU = CUDA for every step and a whole V-cycle) and every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 6/6 mutations caught; Born table and APBS 1YCR on CUDA unchanged; CPU 1.17–1.25× per calculation, CUDA solve stage 1.17–1.26× (calculation 1.05–1.11×) on a shared GPU and 1.16–1.26× on an idle one; see *Fused multigrid smoother* |
| Fused CG steps (0.0.2) | The CG update (x, r, ‖r‖², rᵀz, the f32 copy of r), the widening of the V-cycle's result with rᵀz and, on the CPU, A p with pᵀAp run as one pass each (`Backend::cg_update`, `promote_dot`, `apply_dot`), with the summation order of `dot`: the new CG loop equals the old one (kept in the tests) and a solve through `backend::Stepwise` bit for bit (3 preconditioners, 5 grids, CPU and CUDA), and so does every step (CPU and CUDA, both precisions); every result bit-identical to the previous commit (13 cases + maps, potentials, MM-PBSA species; CPU and CUDA; cuda-12 and cuda-13 builds); 12/12 mutations caught; Born table and APBS 1YCR on CUDA unchanged; CPU 1.03–1.07× per calculation (1.13× with `jacobi`), CUDA f64 CG kernels 1.11× (2–4 % of the solve stage) on a shared GPU, solve stage 1.00–1.06× on an idle one; see *Fused CG steps* |
| Batched solves (0.0.2) | MM-PBSA frames/s on 1YCR: 1.4× (CPU, 2 workers) and 2.2× (CUDA, 3 workers), results bit-identical to sequential calls (see *Batched solves*) |
| CUDA backend | **first GPU run** (RTX 2080 SUPER, cc 7.5, driver 595.99 = CUDA 13.2, `--features cuda-13`, conda NVRTC 13.2): `pbsolve test` and `tests/test_platforms.py` pass, CPU vs CUDA 9.8e-15; APBS comparison on CUDA identical to CPU. Multigrid on the GPU: CPU vs CUDA ≤ 2.8e-14 (identical iteration counts, `tests/test_platforms.py`). NVRTC selection and CUBIN output (0.0.2): the GPU now works with NVRTC 12.9, 13.2, 13.3 and 13.4 on the 13.2 driver, with energies bit-identical across all of them and to the earlier runs; CUDA 12 build run on the GPU (see *GPU runtime* below) |
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
* The CUDA numbers come from a shared GPU. On the idle GPU (no other job,
  4 % utilisation before the run; two rounds against `54a0f8b`, energies
  bit-identical) the solve stage gains 1.16–1.26× as well:

  | Calculation | Solve stage before → after (s) | Gain |
  |---|---|---|
  | 1YCR `multigrid-f32` | 0.105 → 0.089 | 1.18× |
  | 1YCR `multigrid` | 0.167 → 0.133 | 1.26× |
  | 1YCR `jacobi` (no V-cycle) | 1.137 → 1.141 | 1.00× |
  | 1TII `multigrid-f32` | 0.710 → 0.587 | 1.21× |
  | 5DK3 `multigrid-f32`, `mdh` / `mdh-fast` | 1.603 → 1.379 / 1.591 → 1.339 | 1.16× / 1.19× |

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
  `mdh-fast` show 1.03–1.04×.
* On the idle GPU (same runs as in *Fused multigrid smoother*, against
  `6ae134a`, energies bit-identical) the solve stage gains 1.00–1.06×:
  1YCR `multigrid-f32` 0.089 → 0.084 s, `jacobi` 1.141 → 1.095 s, 5DK3
  1.379 → 1.329 s (`mdh`) and 1.339 → 1.302 s (`mdh-fast`); 1YCR `multigrid`
  and 1TII are unchanged. Both fusions together: solve stage 1.21–1.26× with
  multigrid, a calculation 1.09–1.12× (1YCR 0.209 → 0.192 s, 1TII
  1.571 → 1.419 s, 5DK3 4.60 → 4.19 s with `mdh`, 2.52 → 2.26 s with
  `mdh-fast`). With `mdh` the host's boundary sum (5DK3 2.1–2.3 s) is now
  half of a large calculation.

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

## Analytic references: Kirkwood sphere

**Model.** Point charges inside a sphere of radius a: interior `pdie`,
exterior `sdie`, ions beyond the Stern radius b = a + `ion_radius`
[Kirkwood 1934]. In pbsolve the sphere is one uncharged atom of radius a and
the charges are atoms of radius 0 inside it; the ion map then excludes ions
within exactly b (checked on the maps), and the reference state is `pdie`
without ions in both.

**Reference** (`pbsolve.analytic.kirkwood_energy`). The reaction field is a
sum over multipoles n; each coefficient comes from the interface conditions
at a and b in closed form. For salt, kₙ(κb) itself overflows at large n, so
only its logarithmic derivative is used, from a stable recurrence of kₙ₋₁/kₙ.
The series is cut where (ρ_max/a)²ⁿ < 1e-14 (~800 terms at ρ/a = 0.98).
Checks (`tests/test_analytic.py`):
* the Born energy, and Born with Debye–Hückel screening beyond a Stern
  layer, to ≤ 1e-13;
* sdie → ∞ or κ → ∞ makes the sphere a grounded conductor: image charges for
  three charges (cross terms included), and for one charge at ρ/a = 0.98, to
  1e-10 (1e-5 at finite κa ≈ 1.6·10⁴);
* each coefficient against the full 4×4 interface conditions written with
  the explicit kₙ polynomial, to 1e-12 (n ≤ 12). That polynomial matches
  scipy's `spherical_kn` to 1e-15, and the recurrence matches it to 2e-16
  up to n = 59;
* rotation invariance, and rejection of charges on or outside the sphere.
* Mutations caught: 6/6 (Legendre recurrence, Stern radius, start of the
  kₙ recurrence, series cut too early, κ without sdie, powers of ρ).

**pbsolve against it.** `scripts/validate_analytic.py`: CPU, `mol`, `mdh`,
`fadd 20` (Born: 12). Every spacing is the mean over 20 placements of the
molecule on a fixed grid (the first unshifted; range in brackets).
Error = (ΔG − ΔG_analytic)/|ΔG_analytic|. The three charges are +1, −1 and
+0.5 at 2.0, 2.3 and 2.7 Å from the centre; the near-surface pair is two +1
charges 1.0 and 1.2 Å below the surface.

| Case | ΔG analytic (kcal/mol) | h = 0.5 | h = 0.35 | h = 0.25 | h = 0.2 | order |
|---|---|---|---|---|---|---|
| Born ion, a = 2 Å | −81.9589 | −2.94 % (−4.73 … −1.95) | −1.59 % (−2.14 … −1.00) | −1.23 % (−1.42 … −1.03) | −0.96 % (−1.15 … −0.73) | 1.18 |
| Kirkwood, 3 charges, a = 5 Å | −37.2679 | −2.77 % (−3.29 … −2.06) | −1.82 % (−2.16 … −1.64) | −1.27 % (−1.39 … −1.15) | −0.98 % (−1.08 … −0.89) | 1.12 |
| same, 0.15 M | −37.3369 | −2.69 % (−3.20 … −2.07) | −1.85 % (−2.18 … −1.61) | −1.33 % (−1.47 … −1.21) | −0.97 % (−1.06 … −0.89) | 1.09 |
| same, 0.15 M, pdie 4 | −8.8892 | −2.34 % (−2.71 … −1.92) | −1.60 % (−1.87 … −1.35) | −1.09 % (−1.24 … −0.92) | −0.84 % (−0.93 … −0.73) | 1.11 |
| 2 charges 1.0–1.2 Å below the surface, 0.15 M | −224.9608 | −4.68 % (−11.42 … +0.11) | −1.99 % (−5.94 … +0.02) | −1.24 % (−3.36 … −0.22) | −1.10 % (−2.38 … −0.32) | 1.59 |

* **Observed order ≈ 1.1–1.2.** The 7-point scheme with a sharp dielectric
  boundary is first order, as the roadmap states (4.8). Salt and pdie change
  the constant, not the order. On average pbsolve's |ΔG| is too large in
  every case.
* **Placement matters most near the surface.** For charges 1 Å below it a
  single calculation at 0.5 Å can be off by 11 %, and the fitted order (1.59)
  is dominated by that scatter. Between 0.35 and 0.2 Å the mean error falls
  roughly in proportion to h. This is the case the second-order interface
  methods (4.8) and the error estimates (4.9) address.
* With 6 placements instead of 20 the orders were 1.14–1.21 and 2.13 for the
  near-surface pair, so read that row as "noisy", not "second order".
* Tests: the 3-charge sphere at 0.5 and 0.25 Å with 0 M / pdie 1 and
  0.15 M / pdie 4 (|error| < 1.5 % at 0.25 Å and less than 0.6 times the
  error at 0.5 Å; 1 s each).

Reproduce: `python scripts/validate_analytic.py --placements 20` (the script
also runs the two spheres below; 20 min on 18 threads, 4.5 min for these
rows).

## Analytic references: two spheres

**Model.** Two spheres with charges inside, as for Kirkwood, far enough
apart that their ion-free shells do not overlap. pbsolve's maps for a 5 and
a 4 Å sphere 15.8 Å apart (0.4 Å grid):
* The ion map is the union of the two balls of radius a + `ion_radius`,
  node for node.
* The dielectric map is the union of the spheres, face for face, with `vdw`.
* `mol` (APBS' molecular surface) samples probe positions at `sdens` points
  per Å². Between them it leaves cusps: 25 of 0.8 million x-faces lie
  0.0004–0.0044 Å outside the spheres and are marked inside. This differs
  from the exact spheres by a fixed amount at every spacing.

**Reference** (`pbsolve.analytic.two_sphere_energy`).
* Each sphere has a multipole expansion: regular inside, the ion-free shell,
  and outgoing kₙ(κr) (r⁻⁽ⁿ⁺¹⁾ without salt) outside.
* The spheres are coupled by re-expanding each one's outgoing field about the
  other centre. With z along the line between the centres, each azimuthal
  order m is a separate linear system.
* The re-expansion coefficients come from Gauss–Legendre projection on the
  other sphere's ion boundary. Each sphere's response per order comes from
  its four interface conditions, scaled to stay well conditioned.
* The order is chosen from the geometry for ~1e-12 (at most 150); one call
  takes ~0.03 s.

Checks (`tests/test_analytic.py`):
* pdie = sdie without salt: the re-expanded fields must rebuild the Coulomb
  interaction exactly; |ΔG| = 7e-15 kcal/mol.
* sdie → ∞: each sphere is a grounded conductor on its own (image charges),
  minus the reference state's Coulomb interaction: 3e-12.
* Far apart without salt: ΔG − Kirkwood₁ − Kirkwood₂ against the charge and
  dipole interaction through the solvent (a dielectric sphere passes on
  3/(pdie + 2 sdie) of a dipole's field). The remainder is 2.5e-2 / 5.8e-3 /
  5.6e-5 at 50 / 100 / 1000 Å and falls as 1/d², as the next order should.
* Far apart with salt (300 Å): only the reference state's Coulomb
  interaction is left, to 3e-13.
* Swapping the spheres, rotating and translating everything, and raising
  the order to 80 change nothing beyond 1e-11.
* The Legendre functions are orthonormal; the logarithmic derivatives of iₙ
  and kₙ match the power series and the explicit polynomial to 1e-13.
* Mutations caught: 9/9 (direction of the re-expansion, multipole
  normalisation, the ±m pairs, the reference Coulomb term, the Stern shell,
  the iₙ and kₙ recurrences, the Legendre recurrence, pdie/sdie in the
  interface conditions).

**pbsolve against it** (`scripts/validate_analytic.py --placements 20`,
`mol` and `--surface vdw`). Charges as in the Kirkwood table (5 Å sphere)
plus −1 and +0.7 in the 4 Å sphere. The association energy ΔΔG = complex −
sphere 1 − sphere 2 is computed on the complex's grid, as in MM-PBSA; its
errors are in kcal/mol.

| Case | surface | exact | h = 0.5 | h = 0.35 | h = 0.25 | h = 0.2 | order |
|---|---|---|---|---|---|---|---|
| two spheres, 0 M | `mol` | −55.3714 | −3.35 % (−3.93 … −2.76) | −2.15 % (−2.43 … −1.91) | −1.48 % (−1.69 … −1.34) | −1.13 % (−1.22 … −0.99) | 1.17 |
| | `vdw` | | −3.57 % (−4.02 … −2.96) | −2.39 % (−2.65 … −2.11) | −1.70 % (−1.89 … −1.51) | −1.35 % (−1.44 … −1.24) | 1.06 |
| two spheres, 0.15 M | `mol` | −55.4476 | −3.25 % (−4.17 … −2.40) | −2.15 % (−2.55 … −1.96) | −1.55 % (−1.74 … −1.40) | −1.16 % (−1.25 … −1.08) | 1.11 |
| | `vdw` | | −3.43 % (−4.38 … −2.49) | −2.34 % (−2.78 … −2.10) | −1.72 % (−1.92 … −1.59) | −1.33 % (−1.43 … −1.20) | 1.02 |
| association, 0 M | `mol` | 1.7878 | −0.015 (−0.105 … +0.079) | −0.018 (−0.073 … +0.056) | −0.011 (−0.034 … +0.019) | −0.016 (−0.043 … +0.005) | – |
| | `vdw` | | −0.002 (−0.004 … +0.000) | −0.002 (−0.003 … −0.001) | −0.002 (−0.002 … −0.001) | −0.002 (−0.002 … −0.001) | – |
| association, 0.15 M | `mol` | 1.8061 | −0.014 (−0.091 … +0.011) | −0.019 (−0.079 … +0.016) | −0.006 (−0.050 … +0.024) | −0.012 (−0.028 … +0.006) | – |
| | `vdw` | | −0.001 (−0.002 … +0.001) | −0.000 (−0.001 … +0.001) | −0.000 (−0.001 … +0.001) | −0.000 (−0.000 … +0.000) | – |

* **The complex is first order**, as one sphere is: orders 1.11–1.17 with
  `mol` and 1.02–1.06 with `vdw`. With `vdw` the Kirkwood and Born rows give
  0.96–0.99 and 1.10, with errors 0.1–0.25 points larger than with `mol`.
  The cusps of `mol` enlarge the cavity slightly and offset part of the
  discretisation's overestimate of |ΔG|.
* **The association energy cancels on one grid.** With `vdw` its mean is
  within 0.002 kcal/mol (0.1 %) at every spacing, single placements within
  0.004; with salt within 0.001 and 0.002. The complex itself is off by
  0.7–2 kcal/mol.
  * Without salt the remaining 0.002 kcal/mol comes from the coarsest box's
    `mdh` values, which treat the spheres as water: with a coarse box 2–3
    times larger (`cfac 5`, `fadd 60`) it is +0.0001 kcal/mol (h = 0.35 Å,
    4 placements).
* **With `mol` the association is off by 0.011–0.019 kcal/mol on average
  (0.6–1 %) at every spacing.** Single placements scatter by up to
  ±0.1 kcal/mol at 0.5 Å. This is the cusps above, not the discretisation:
  at 0.25 Å (10 placements) `vdw` gives −0.0016 ± 0.0001 and `mol`
  −0.019 ± 0.007 kcal/mol, and a larger box does not change `mol`. It is a
  property of the APBS-compatible surface, small next to the statistical
  error of MM-PBSA.
* Tests: the 0 M pair with `vdw` at 0.5 and 0.35 Å; the complex is within
  2.5 % and improves, and ΔΔG is within 0.006 kcal/mol (measured −0.0031 /
  −0.0007); 2.4 s.

Reproduce: `python scripts/validate_analytic.py --placements 20` and the same
with `--surface vdw` (20 min each on 18 threads).

## Charge assignment: B-splines (`chgm spl2`, `spl4`)

**Change.** `Solver(chgm=...)` and `--chgm` choose how point charges are
spread onto the grid, with APBS 3.4.1's weights, node ranges and energy rule:

| `chgm` | Weights | Nodes per axis | Energy ½ Σ q u |
|---|---|---|---|
| `spl0` (default, unchanged) | trilinear | 2 | trilinear at the atoms |
| `spl2` | B-spline of order 3 (piecewise quadratic) | 3 | the same weights (`Vpmg_qfEnergyVolume`) |
| `spl4` | B-spline of order 5 (piecewise quartic) | 5 | trilinear at the atoms (`Vpmg_qfEnergyPoint`) |

* APBS documents `spl2` and `spl4` as cubic and quintic; its functions
  (`bspline2`, `bspline4`) are the order-3 and order-5 B-splines above.
* The host spreads the charges for every backend (the node charges reach
  the GPU with `scatter`), so CPU and CUDA get the same source.
* A spline needs room to the outer faces: 2 (`spl2`) and 3 (`spl4`) node
  spacings, otherwise `ValueError`.
* APBS inputs with `chgm spl2` or `spl4` are now accepted by
  `pbsolve apbs run` and written by `pbsolve apbs compare`.

**Checks.**
* Weights (`charge.rs`): every spread keeps the charge and its centre to
  1e-12, with variance h²/4 (`spl2`) and 5h²/12 (`spl4`) per axis (50 random
  positions). The exact weights on a node are 1/8, 3/4 and 1/384, 19/96,
  115/192. Sparse and dense spreading agree bit for bit for all three
  methods, and each method refuses charges without room to the boundary.
* Energies taken at the charged nodes equal the energies of the full
  potentials bit for bit, and the potentials equal a host reference of the
  pipeline (`spl2` on two levels, `spl4` on three). On the GPU the same
  holds, and CPU and CUDA agree to < 1e-12 (cuda-12 and cuda-13 builds).
* `spl0` results are bit-identical to the previous commit (13 cases + maps,
  potentials, MM-PBSA species; CPU and CUDA).
* APBS 3.4.1, synthetic molecule (`spl2`, `mdh`, 0.15 M; `spl4`, `sdh`, 0 M)
  in `tests/test_apbs_parity.py`: all maps identical and energies within the
  `spl0` tolerances.
* APBS 3.4.1, 1YCR complex (0.15 M, `mdh`, ligand chain B, maps compared):

  | `chgm` | ΔG_polar complex APBS / pbsolve | in APBS units | ΔΔG_bind APBS / pbsolve | in APBS units |
  |---|---|---|---|---|
  | `spl0` | −1221.2814 / −1221.2671 | −4.7e-7 | 422.8979 / 422.8939 | +0.0012 kcal/mol |
  | `spl2` | −1218.9705 / −1218.9562 | −4.9e-7 | 422.6106 / 422.6066 | +0.0012 kcal/mol |
  | `spl4` | −1219.0214 / −1219.0072 | −4.9e-7 | 422.4409 / 422.4369 | +0.0012 kcal/mol |

  Dielectric and ion maps: 0 nodes differ; charge maps within 2.3e-7 of
  their maximum (the DX format); per state ≤ 2.5e-10 (solvated) and
  ≤ 3.7e-8 (reference) in APBS units, as with `spl0`. The comparison now
  takes the `spl4` energy of the solve on APBS' maps at the atoms, as APBS
  does (it was compared with the `spl4` weights and showed −0.3).
* Mutations caught: 6/6 (node offset, `spl4` interior bound, `spl4` energy
  with its own weights, a `bspline4` coefficient, energy from the source
  charges, APBS input always `spl0`).

**Accuracy.** The B-splines hardly change ΔG_polar:
* Analytic references (`scripts/validate_analytic.py --chgm`, `mol`, 20
  placements): mean errors at 0.5 / 0.2 Å, `spl0` → `spl2` → `spl4`:

  | Case | `spl0` | `spl2` | `spl4` |
  |---|---|---|---|
  | Born ion | −2.94 / −0.96 % | −2.89 / −0.96 % | −2.91 / −0.96 % |
  | Kirkwood, 3 charges | −2.77 / −0.98 % | −2.80 / −0.98 % | −2.78 / −0.98 % |
  | Kirkwood, 2 charges below the surface | −4.68 / −1.10 % (−11.42 … +0.11 at 0.5 Å) | −4.41 / −1.11 % (−10.64 … −0.16) | −4.44 / −1.11 % (−10.77 … −0.21) |
  | Two spheres, 0 M | −3.35 / −1.13 % | −3.40 / −1.13 % | −3.37 / −1.13 % |
  | Association ΔΔG, 0 M | −0.015 / −0.016 kcal/mol | −0.015 / −0.016 | −0.015 / −0.016 |

  Observed orders: 1.09–1.18, 1.10–1.19 and 1.09–1.18.
* 1YCR (0.15 M, CUDA, h = 0.5 Å, one grid, 16 placements of the molecule):

  | `chgm` | ΔG_polar mean, sd, range (kcal/mol) | ΔΔG_bind mean, sd, range |
  |---|---|---|
  | `spl0` | −1233.34, 4.09, 15.4 | 425.65, 1.57, 6.0 |
  | `spl2` | −1230.63, 4.07, 14.9 | 425.20, 1.81, 6.9 |
  | `spl4` | −1229.37, 3.88, 14.4 | 424.89, 1.62, 6.0 |

* So the placement spread falls by at most 5 %, against the 3× the roadmap
  hoped for. The error comes from the sharp dielectric boundary. The
  self-energy of the spread charge, which the spline changes, cancels
  between the solvated and the reference state.
* The methods differ from each other by 0.2–0.3 % in ΔG_polar (1YCR,
  5DK3), less than the spread over placements.
* The placement spread of 1YCR (sd 4 kcal/mol, 0.3 %, at 0.5 Å) is the
  baseline for the interface methods (4.8) and the error estimates (4.9).

**Cost** (5DK3, 0.15 M, idle GPU): the host spreads 125 (`spl4`) or 27
(`spl2`) contributions per charge instead of 8. The assembly stage takes
0.05 → 0.10 → 0.44 s on CUDA (a calculation 4.17 → 4.21 → 4.34 s). On the
CPU they added at most 4 % to a 28.5 s calculation (measured before the
zero weights were skipped). Nodes whose weight is
exactly 0 (one end node per axis) are skipped; that halved the `spl4`
assembly (0.83 → 0.44 s) without changing the energy.

## Error estimates (`estimate_error`)

**Change.** `pbsolve.estimate_error(solver, coords, charges, radii,
placements=8, spacings=None, parts=None)` and `pbsolve solvation
--placements N [--extrapolate H]`:
* The automatic grid is moved by uniform random fractions (−½ … ½) of its
  fine spacing along each axis; the first placement is the unmoved grid.
  The placements are solved with `polar_solvation_many`.
* It reports the mean, the standard deviation over the placements and the
  standard error of the mean.
* Given two spacings or more, it extrapolates the means at the two finest
  to zero spacing at first order. `discretisation_error` is mean −
  extrapolated.
* With `parts` (masks or indices) the quantity is ΔG(all) − Σ ΔG(part) on
  the grid of the whole molecule (MM-PBSA association energy).
* Nothing changes unless it is called; `polar_kcal_mol` of
  `pbsolve solvation` stays the ordinary single calculation.

**Checks** (`tests/test_estimate.py`, 5 tests, 6 s):
* Every placement equals `polar_solvation` on the automatic grid moved by
  its reported offset (at most half a spacing per axis; the first is zero),
  bit for bit.
* Mean, sd and sem follow their definitions, and the same seed gives the
  same placements.
* The extrapolation follows its formula with the actual fine spacings
  (grids round the spacing down: 0.469 / 0.341 Å for 0.5 / 0.35).
* Masks and index arrays give the same association energies, equal to three
  calculations on the complex's grid; bad arguments raise `ValueError`.
* Against Kirkwood's solution the extrapolation removes most of the error.
* The command line reports mean, sd, sem and the extrapolation.
* Mutations caught: 7/7 (extrapolation formula, sem, sd, offsets in Å
  instead of spacings, first placement moved, parts added, spacings in the
  wrong order).

**Accuracy of the estimates** (CUDA, 12 placements, the cases of
`scripts/validate_analytic.py`; errors relative to the exact value, in %;
association energies in kcal/mol):

| Case | spacings (Å) | error of the mean | sd | error after extrapolation | estimated error of the mean | true / estimated |
|---|---|---|---|---|---|---|
| Born ion, a = 2 Å | 0.5, 0.35 | −1.63 | 0.35 | +1.87 | −3.49 | 0.47 |
| | 0.35, 0.25 | −1.29 | 0.12 | −0.30 | −1.00 | 1.30 |
| Kirkwood, 3 charges | 0.5, 0.35 | −1.79 | 0.13 | +0.76 | −2.55 | 0.70 |
| | 0.35, 0.25 | −1.34 | 0.09 | −0.09 | −1.25 | 1.07 |
| same, 0.15 M | 0.35, 0.25 | −1.37 | 0.09 | −0.11 | −1.26 | 1.09 |
| same, 0.15 M, pdie 4 | 0.35, 0.25 | −1.14 | 0.09 | −0.14 | −1.01 | 1.14 |
| 2 charges 1–1.2 Å below the surface | 0.5, 0.35 | −2.38 | 1.58 | −1.14 | −1.24 | 1.92 |
| | 0.35, 0.25 | −1.75 | 1.24 | −0.02 | −1.73 | 1.01 |
| Two spheres, 0 M | 0.5, 0.35 | −2.17 | 0.21 | +0.54 | −2.72 | 0.80 |
| | 0.35, 0.25 | −1.53 | 0.14 | +0.19 | −1.72 | 0.89 |
| Two spheres, 0.15 M | 0.35, 0.25 | −1.54 | 0.15 | +0.25 | −1.79 | 0.86 |
| Association ΔΔG, 0 M (kcal/mol) | 0.35, 0.25 | −0.017 | 0.019 | −0.022 | +0.005 | – |
| Association ΔΔG, 0.15 M (kcal/mol) | 0.35, 0.25 | −0.012 | 0.013 | −0.006 | −0.007 | – |

* From 0.35 and 0.25 Å the extrapolated energies are within 0.3 % of the
  exact ones (the means are off by 1.1–1.8 %). The estimated error of the
  mean is within 30 % of the true error, even for charges just below the
  surface.
* From 0.5 and 0.35 Å the correction is off by up to a factor of 2 in both
  directions: the first-order regime starts below ~0.35 Å. The documentation
  recommends spacings of 0.35 Å and below.
* Association energies on one grid are already within the placement noise;
  their extrapolation carries no information. Use their sd and sem.

**A protein** (1YCR, 0.15 M, CUDA, 8 placements per spacing; 12 s for
ΔG_polar, 34 s with receptor and ligand):

| Fine spacing (Å) | 0.484 | 0.341 | 0.245 | extrapolated (0.341, 0.245) |
|---|---|---|---|---|
| ΔG_polar, mean (kcal/mol) | −1234.7 | −1205.6 | −1190.5 (sd 1.23, sem 0.44) | −1152.3 |
| ΔΔG_bind, mean (kcal/mol) | 426.6 | 421.6 | 419.3 (sd 0.43, sem 0.15) | 413.5 |

* The protein depends on the spacing far more than the spheres do: ΔG_polar
  changes by 3.6 % from 0.48 to 0.25 Å, and the estimated error at 0.25 Å
  is 38 kcal/mol (3 %).
* The estimate itself has an uncertainty: extrapolating from 0.48 and
  0.34 Å instead gives −1136 kcal/mol. On the spheres the finest pair was
  right to 30 % of the correction.
* ΔΔG_polar of binding carries a bias of ~+6 kcal/mol at 0.25 Å and ~+13 at
  0.5 Å. That is far above its placement noise (sd 0.4 at 0.25 Å, 1.6 at
  0.5 Å, see *Charge assignment*) and enters absolute MM-PBSA binding
  energies directly. How much of it cancels between related ligands is
  not measured here.
* Placement sd: 4.1 kcal/mol at 0.5 Å (16 placements, *Charge assignment*),
  1.2 at 0.25 Å.

## Fractional dielectric interface (`interface="fractional"`)

**Change.** `Solver(interface="fractional")` and `--interface fractional`,
for the surfaces `mol` and `vdw`:
* A face whose grid edge (the segment between its two nodes) the surface
  cuts gets ε = h / (ℓ_in/pdie + ℓ_out/sdie). Here ℓ_in and ℓ_out are the
  lengths of the edge inside and outside the solute. This is the exact
  effective dielectric of that edge in one dimension: harmonic averaging of
  boundary dielectrics [Davis & McCammon 1991; Wang et al. 2012], weighted by
  the exact cut of each edge. Wang et al. (2009, Eq. 3) define the same
  edge-weighted harmonic average ("HA") with the interface point on the
  edge.
* ℓ_in is computed exactly from the spheres of the surface: the edge's
  chords through the solute spheres (radius r + `srad`) minus its chords
  through the probe spheres. Faces the surface does not cut keep `pdie` or
  `sdie`, as with the sharp marking.
* Only the dielectric maps change. The 7-point operator stays symmetric, so
  CG and multigrid run unchanged (the Born ion takes 17 CG iterations, as
  before). The ion map and the charges are unchanged.
* The maps are built on the host for every backend and uploaded.
  `interface="sharp"` stays the default and is bit-identical to the previous
  commit. `fractional` has no APBS equivalent; `pbsolve.apbs` refuses it.

**Choice of scheme.** A Python prototype on `Solver.solve_maps` compared
three face rules on the Born and Kirkwood cases (6 placements, 0.5 / 0.35 /
0.25 Å, `vdw`):
* the edge rule above: errors −0.53 / −0.18 % at 0.5 Å (Born / Kirkwood),
  orders 1.2–2.3;
* the harmonic average with the fill fraction of the face's dual cell:
  1.7–2.8 times larger errors;
* an anisotropic rule n_a²·harmonic + (1 − n_a²)·arithmetic: −5 to −13 %,
  worse than sharp.

The Rust implementation equals the prototype to ≤ 6.5e-15 relative. The
improved harmonic-average, IIM and MIB schemes of the roadmap were not
evaluated: the edge rule already meets the roadmap's targets, needs no
interface normals and keeps the symmetric operator.

**Checks:**
* `fractional_faces_split_every_edge_exactly` (Rust): every face of three
  atoms with probes (`srad` 1.4 and 0) on an anisotropic grid
  (0.5 / 0.477 / 0.45 Å) is checked against an independent split of its edge
  at every sphere crossing, to ≤ 1e-12. Faces whose edge the surface does not
  cut equal the sharp marking.
* `tests/test_interface.py` (5 tests, 3 s):
  * the Born ion and the Kirkwood sphere at 0.5 Å over 4 placements: every
    error within 0.7 %, a range 3 times narrower than sharp;
  * the maps differ from sharp only on cut faces, whose values lie strictly
    between `pdie` and `sdie`; ions, charges and the reference state are
    unchanged;
  * unknown values, `surface="harmonic"` and APBS inputs are refused;
  * solvers rebuilt from `params` keep the setting;
  * the command line.
* GPU: device maps equal the host maps bit for bit
  (`device_coefficient_maps_equal_the_host_maps`). CUDA against CPU energies
  (`test_cuda_matches_cpu`) agree to ≤ 4e-14 on the spheres.
* Default path: the fingerprint set (energies, iterations, residuals) is
  bit-identical on CPU and CUDA, and so are the sharp rows of the Born table.
* The two shortcuts leave the maps bit-identical (1YCR, `mol` and `vdw`,
  0.5 and 0.25 Å, both focusing levels):
  * nodes whose edges cannot meet the surface are classified first with
    `mark_map`;
  * a probe that covers all of the solute part ends the search.
* Mutations caught: 10/11 (arithmetic instead of harmonic mean, chord not
  clipped at either end, interval merge and overlap, both node
  classifications, both early exits, spacing of the wrong axis). Missed:
  dropping the h/2 margin of the probe search radius, which the cell list's
  rounding to whole cells covers in practice.

**Born ion** (`cargo run --release -p pbsolve-core --example born`, one
placement; the `mol` and `harmonic` rows are unchanged):

| h (Å) | 0.5 | 0.333 | 0.25 | 0.2 |
|---|---|---|---|---|
| `mol`, sharp | −4.73 % | −1.08 % | −1.27 % | −0.94 % |
| `mol`, fractional | −0.39 % | −0.06 % | +0.02 % | +0.07 % |

**Analytic references** (`scripts/validate_analytic.py --placements 20
--interface fractional`, CUDA; mean error over 20 placements and its range,
as in the tables above):

| Case | surface | h = 0.5 | h = 0.35 | h = 0.25 | h = 0.2 | order |
|---|---|---|---|---|---|---|
| Born ion, a = 2 Å | `vdw` | −0.51 % (−0.55 … −0.49) | −0.21 % (−0.21 … −0.20) | −0.11 % (−0.11 … −0.11) | −0.07 % (−0.07 … −0.07) | 2.15 |
| | `mol` | −0.38 % (−0.40 … −0.34) | −0.07 % (−0.10 … −0.04) | +0.02 % (+0.01 … +0.03) | +0.06 % (+0.05 … +0.07) | – |
| Kirkwood, 3 charges | `vdw` | −0.17 % (−0.37 … −0.05) | −0.09 % (−0.15 … −0.05) | −0.06 % (−0.06 … −0.05) | −0.04 % (−0.06 … −0.02) | 1.62 |
| | `mol` | +0.04 % (−0.13 … +0.18) | +0.12 % (+0.06 … +0.17) | +0.16 % (+0.15 … +0.17) | +0.18 % (+0.13 … +0.22) | – |
| same, 0.15 M | `vdw` | −0.16 % (−0.23 … −0.06) | −0.09 % (−0.15 … −0.04) | −0.05 % (−0.05 … −0.04) | −0.03 % (−0.05 … −0.01) | 1.88 |
| same, 0.15 M, pdie 4 | `vdw` | −0.14 % (−0.37 … +0.00) | −0.07 % (−0.12 … −0.01) | −0.04 % (−0.04 … −0.03) | −0.03 % (−0.04 … −0.01) | 1.84 |
| 2 charges 1.0–1.2 Å below the surface, 0.15 M | `vdw` | −0.71 % (−1.77 … +0.35) | −0.28 % (−0.69 … +0.05) | −0.13 % (−0.26 … −0.04) | −0.07 % (−0.20 … +0.07) | 2.51 |
| | `mol` | −0.50 % (−1.49 … +0.53) | −0.07 % (−0.45 … +0.27) | +0.08 % (−0.05 … +0.17) | +0.15 % (+0.01 … +0.25) | – |
| two spheres, 0 M | `vdw` | −0.23 % (−0.33 … −0.08) | −0.12 % (−0.21 … −0.06) | −0.06 % (−0.08 … −0.04) | −0.04 % (−0.06 … −0.03) | 1.84 |
| | `mol` | −0.01 % (−0.10 … +0.15) | +0.09 % (+0.01 … +0.16) | +0.15 % (+0.12 … +0.18) | +0.17 % (+0.14 … +0.20) | – |
| two spheres, 0.15 M | `vdw` | −0.27 % (−0.39 … −0.15) | −0.11 % (−0.19 … −0.04) | −0.06 % (−0.11 … −0.02) | −0.04 % (−0.05 … −0.01) | 2.14 |
| association, 0 M (kcal/mol) | `vdw` | −0.001 (−0.004 … +0.000) | −0.002 (−0.003 … −0.001) | −0.001 (−0.002 … −0.001) | −0.002 (−0.002 … −0.001) | – |
| | `mol` | −0.015 (−0.026 … −0.007) | −0.016 (−0.023 … −0.007) | −0.015 (−0.019 … −0.010) | −0.016 (−0.020 … −0.011) | – |
| association, 0.15 M (kcal/mol) | `vdw` | −0.001 (−0.002 … +0.001) | −0.000 (−0.001 … +0.001) | −0.000 (−0.001 … +0.001) | −0.000 (−0.000 … +0.000) | – |
| | `mol` | −0.012 (−0.018 … −0.003) | −0.012 (−0.017 … −0.007) | −0.010 (−0.015 … −0.005) | −0.011 (−0.013 … −0.009) | – |

* **Second order with `vdw`.** The errors fall to zero with observed orders
  1.6–2.5, against 1.0–1.2 for sharp (1.6 for the scattered near-surface
  pair). At 0.5 Å they are −0.14 to −0.27 %
  for charges 2–3 Å inside (sharp, `mol`: −2.3 to −3.4 %). They are
  −0.51 % for the Born ion and −0.71 % for charges 1 Å below the surface
  (sharp: −2.9 and −4.7 %).
* **Placement scatter shrinks.** At 0.5 Å the range over the placements
  (`mol`) falls 4 times for the Kirkwood charges, 5 times for the two
  spheres, 6 times for the near-surface pair (11.5 → 2.0 points) and 46
  times for the Born ion (2.8 → 0.06).
* **With `mol` the error levels off at +0.13 … +0.18 %.** This is APBS'
  surface, not the discretisation. The cusps between its sampled probe
  positions (see *two spheres*) enlarge the cavity slightly, and the
  fractional faces now resolve them. With `sdens=100` instead of 10 (cusps
  ~10 times smaller; 6 placements) the Kirkwood error at 0.25 / 0.2 Å goes
  from +0.16 / +0.19 % to −0.03 / −0.02 %. The two spheres go from +0.15 /
  +0.17 % to −0.04 / −0.02 %, close to the `vdw` values. The orders of the
  `mol` rows are therefore not meaningful.
* **Association energies** keep their floors: 0.002 kcal/mol with `vdw` and
  the cusps' 0.010–0.016 with `mol`. The single placements with `mol` now
  scatter ±0.01 instead of ±0.1 kcal/mol at 0.5 Å.
* Targets of roadmap 4.8: Born and Kirkwood errors below 0.5 % at 0.5 Å
  are met, except the Born ion with `vdw` (−0.51 %) and the near-surface
  pair (−0.50 % `mol`, −0.71 % `vdw`). The order was measured on energies,
  not on the potential.

**A protein** (1YCR, 0.15 M, CUDA, `pbsolve.estimate_error` with 8
placements per spacing; ΔΔG_bind with receptor and ligand on the complex's
grid):

| h (Å) | ΔG_polar sharp: mean (sd) | ΔG_polar fractional: mean (sd) | ΔΔG_bind sharp: mean (sd) | ΔΔG_bind fractional: mean (sd) |
|---|---|---|---|---|
| 0.5 | −1234.72 (4.60) | −1165.02 (0.90) | 426.61 (2.66) | 416.16 (0.71) |
| 0.35 | −1205.56 (2.50) | −1159.24 (0.26) | 421.56 (1.08) | 414.91 (0.22) |
| 0.25 | −1190.53 (1.23) | −1157.00 (0.23) | 419.30 (0.43) | 414.49 (0.10) |
| extrapolated from 0.341 / 0.245 Å | −1152.2 | −1151.3 | 413.5 | 413.4 |

* **Placement sd shrinks 5–10 times** for ΔG_polar and 3.7–4.9 times for
  ΔΔG_bind. The roadmap's target was 3 times.
* **Both schemes extrapolate to the same limit** (−1152 and 413.4 kcal/mol),
  so the fractional faces converge to the same solution, only faster. (This
  extrapolation is linear; fitted with its observed order of about 2.9 the
  fractional sequence gives −1155.6 and 414.3, see *Reaction-field
  decomposition*.)
  Between 0.5 and 0.25 Å ΔG_polar changes by 0.7 % instead of 3.6 %.
* **ΔΔG_bind bias** at 0.5 Å: +2.7 instead of +13 kcal/mol; at 0.25 Å +1.1
  instead of +5.8. A fractional calculation at 0.5 Å is closer to the limit
  than a sharp one at 0.25 Å and costs a third of it on CUDA (an eighth on
  the CPU).

**Cost** (1YCR, 0.15 M, best of 3, seconds; the maps of both focusing
levels are built on the host):

| | CUDA 0.5 Å | CUDA 0.25 Å | CPU 0.5 Å | CPU 0.25 Å |
|---|---|---|---|---|
| maps, sharp | 0.023 | 0.075 | 0.10 | 0.31 |
| maps, fractional | 0.12 | 0.53 | 0.13 | 0.49 |
| calculation, sharp | 0.19 | 1.00 | 1.52 | 13.0 |
| calculation, fractional | 0.29 | 1.43 | 1.52 | 12.8 |

At 0.25 Å, per level, 0.06 s go to the node classification and 0.16 s to
splitting the 1.6–1.9 million edges near the surface (241 × 241 × 201 nodes,
125 000 probe spheres). Without the two shortcuts the maps took 0.96 s
(CUDA). On the CPU the solve dominates and the total is unchanged.

**Not done:** the ion-accessibility map stays sharp, so with salt a small
first-order part remains (the 0.15 M rows are as accurate as 0 M). Forces
and nonlinear PB do not exist yet.

Reproduce: `python scripts/validate_analytic.py --placements 20 --interface
fractional [--surface vdw] --platform CUDA` (about 10 min for both).

## Reaction-field decomposition (`coulomb="analytic"`)

**Change.** `Solver(coulomb="analytic")` and `--coulomb analytic`:
* ΔG_polar comes from one solve per focusing level instead of two. The
  reference state (ε = `pdie`, no ions) is not solved: the analytic Coulomb
  potential G of the charges is subtracted near the solute, and the finest
  level solves for the reaction field. The coarser levels solve the solvated
  state as before; they only supply boundary values.
* The grid self-energy never enters, so nothing has to cancel. Removing it
  analytically instead of with a second finite-difference calculation is an
  old idea [Luty et al. 1992; Zhou et al. 1996], and so is splitting the
  potential into a singular Coulomb part and a regular reaction field
  [Chern et al. 2003; Chen et al. 2007; Geng et al. 2007; Cai et al. 2009;
  Holst et al. 2012]. The discrete form below is pbsolve's own.
* `coulomb="grid"` stays the default and is bit-identical to the previous
  commit. `analytic` has no APBS equivalent; `pbsolve.apbs` refuses it.
  `Solver.potential` is not affected.
* `reference_kt` becomes the Coulomb energy of the charges in `pdie`, without
  self-energies; `solvated_kt` is that plus ΔG_polar. `stats_reference` (the
  reference solves' iterations) is empty.

**Method.** The discrete solvated state is A u = f. A is the volume-scaled
7-point operator with the face dielectrics ε_f and κ̄²; f = 4π l_B q are the
spread charges. The default computes

ΔG_polar = ½ qᵀ (A⁻¹ f − A_p⁻¹ f),

with A_p the operator of the reference state (ε = `pdie`, no ions). The
spread charges' grid self-energy ½ qᵀ A_p⁻¹ f is in both terms and cancels.

*The decomposition.* G = l_B Σ_j q_j / (pdie |r − r_j|) is the Coulomb
potential in a homogeneous `pdie` medium; χ is a set of interior nodes that
contains the charges. The substitution u = v + χG is exact for any χ:

A v = f − A(χG).

Inside χ the charges are the sources of G (−pdie ∇²G = 4π l_B ρ). The
decomposition replaces f there by the discrete source of G, A_p G, and
outside χ by zero, the continuous source:

A v = χ·A_p G − A(χG) =: r.     (1)

That is the only approximation. Written out per node (c_a the face
coefficient h_b h_c / h_a, V the cell volume, nb the six neighbours across
the faces f):

* for c in χ: r_c = Σ_f c_a [(pdie − ε_f)(G_c − G_nb) − ε_f (1 − χ_nb) G_nb] − κ̄²_c V G_c;
* for c outside χ: r_c = Σ_f c_a ε_f χ_nb G_nb.

r vanishes at a node that lies in χ together with its neighbours, sees only
`pdie` faces and no ions. It lives on a thin layer: the edge of χ, plus the
dielectric and ionic boundaries inside χ. G is evaluated only there.
Equation (1) has the operator and the solvers of the solvated state, so CG,
multigrid and the CUDA kernels are unchanged. v is the reaction field u − G
inside χ and the total potential outside, so

ΔG_polar = ½ Σ_i q_i (u − G)(r_i) = ½ Σ_nodes q [v − (1 − χ) G],

with the energy's charge weights (`chgm`).

*Where the accuracy comes from.* With χ = every node, (1) becomes
A w = −(A − A_p) G for w = u − G. The default satisfies exactly the same
equation with G replaced by the discrete potential of the spread charges,
u_ref = A_p⁻¹ f, because A⁻¹f − A_p⁻¹f = −A⁻¹(A − A_p) A_p⁻¹f. A − A_p is
nonzero only on the dielectric and ionic boundaries. The two methods
therefore differ only in the Coulomb field those boundaries see: G itself, or
the discrete field of the spread charges, which deviates from G within a few
node spacings of a charge. Wang et al. (2009) describe the same effect: the
finite-volume charge distribution "does distort the otherwise singular
Coulombic potential especially when it is close to a grid charge".

*The region χ.* The prototype (below) settled this:
* χ = every node needs G at every node with ions, that is most of the grid.
  Holst et al. (2012) also show that this two-term splitting is unstable for
  the potential in the solvent: there G is much larger than u.
* χ = the nodes whose six faces are `pdie` puts the edge just inside the
  dielectric boundary. Then the boundary sees the discrete field of the
  charges again: the Born ion's placement scatter grows 5 to 20 times.
* χ = the nodes within r_i + D of an atom. For any D ≥ 0 the spheres give the
  same energies, to 0.003 points. For D < 0 the error grows as the edge
  approaches a charge: for 2 charges 1.0–1.2 Å deep at 0.5 Å, D = −0.25,
  −0.5 and −1 Å give −0.25, +0.94 and −181 %, against −0.59 % for D ≥ 0.
  In a protein, D = 0 is not enough: 109 of 1YCR's 1670 charges, hydrogens
  of radius 0, lie only 0.72–0.8 Å inside the union of the spheres. At 0.5 Å
  D = 0 differs from D = 0.5–2.5 Å by 1.8 kcal/mol, while those agree to
  0.02. At 0.25 Å, D = 0.5–1.5 Å agree to 0.002 and D = 0.25 Å is off by 0.04.

pbsolve uses D = max(1 Å, 2h): the edge stays at least 1.7 Å and two node
spacings from every charge. In the papers we could read the regular part
starts at the dielectric boundary [Cai et al. 2009; Wang et al. 2009; Geng
et al. 2007; Holst et al. 2012], where it needs interface jump conditions.
Here χ is a set of nodes beyond the atoms and (1) follows from the exact
substitution; we did not find this discrete form, or the margin, in the
literature we could read. The full texts of Zhou et al. (1996) and Cai et
al. (2009) were not accessible; their methods are cited from their
abstracts and from Wang et al. (2009).

**Prototype.** A Python prototype on `Solver.maps` and `Solver.solve_maps`:
* reproduced the default to 1e-13;
* compared the choices of χ above, on the analytic references and 1YCR;
* decomposing every focusing level or only the finest gave the same 1YCR
  energies to 0.001 kcal/mol (0.5 and 0.25 Å).

The Rust implementation equals the prototype to ≤ 4e-15 relative: 5
spheres × 3 settings and 1YCR, with both interfaces.

**Checks:**
* `source_equals_the_dense_formula_and_vanishes_off_the_layer` (Rust): the
  sparse right-hand side equals χ·A_p G − A(χG) evaluated at every node to
  1e-11 of its largest value, and vanishes off the layer. It covers `mol`,
  `vdw` and `harmonic`, both interfaces, with and without ions, and two
  spheres 0.1 Å apart whose gap lies between two nodes.
* `a_fine_grid_through_the_solute`: a fine grid that cuts an atom (focusing
  on part of a molecule) agrees with the reference method as closely as one
  that holds it.
* Further Rust tests: the Born ion; the result fields; outside χ the
  reaction field is v − G; a charge on a node where G is needed gives an
  error, not a NaN.
* Mutations caught: 11/11. They cover:
  * each term of r, the scale 1/(4π l_B), and the Coulomb energy;
  * a layer that misses the nodes outside χ, a deep region that is too large;
  * χ on the grid boundary, a zero margin, the energy outside χ.
* `tests/test_coulomb.py` (7 tests, 3 s):
  * the placement spread at 0.5 Å (fractional) narrows by more than a
    quarter, and the means stay within 0.05 points;
  * the two methods approach each other with refinement;
  * one solve per level, and the Coulomb energy in `reference_kt`;
  * parameter, APBS refusal, error message and the command line.
* GPU: `analytic_coulomb_matches_the_cpu` (one to three levels, both
  interfaces, with and without ions): ≤ 1e-10 relative, the Coulomb energy
  bit for bit. `test_cuda_matches_cpu` for all four combinations. On 1YCR
  and 5DK3, CPU and CUDA agree to ≤ 4e-11 and 8e-9 kcal/mol.
* Default path: the fingerprint set is bit-identical on CPU and CUDA.

**Analytic references**, fractional interface (`scripts/validate_analytic.py
--placements 10 --interface fractional --coulomb grid|analytic`, CUDA; mean
error and range over the same 10 placements):

| Case | surface | 0.5 Å, grid | 0.5 Å, analytic | 0.25 Å, grid | 0.25 Å, analytic | order grid / analytic |
|---|---|---|---|---|---|---|
| Born ion, a = 2 Å | `vdw` | −0.52 (−0.55 … −0.49) | −0.47 (−0.50 … −0.45) | −0.11 (−0.11 … −0.11) | −0.11 (−0.11 … −0.11) | 2.17 / 2.06 |
|  | `mol` | −0.38 (−0.40 … −0.35) | −0.33 (−0.35 … −0.30) | +0.01 (+0.01 … +0.03) | +0.02 (+0.01 … +0.03) | – |
| Kirkwood, 3 charges | `vdw` | −0.16 (−0.24 … −0.06) | −0.16 (−0.20 … −0.11) | −0.06 (−0.06 … −0.05) | −0.05 (−0.06 … −0.05) | 1.58 / 1.64 |
|  | `mol` | +0.05 (−0.03 … +0.16) | +0.05 (+0.00 … +0.11) | +0.16 (+0.15 … +0.17) | +0.16 (+0.15 … +0.17) | – |
| same, 0.15 M | `vdw` | −0.19 (−0.36 … −0.06) | −0.17 (−0.26 … −0.10) | −0.05 (−0.05 … −0.05) | −0.05 (−0.05 … −0.04) | 1.93 / 1.93 |
|  | `mol` | −0.01 (−0.18 … +0.13) | +0.01 (−0.08 … +0.08) | +0.13 (+0.13 … +0.14) | +0.14 (+0.13 … +0.14) | – |
| same, 0.15 M, pdie 4 | `vdw` | −0.18 (−0.29 … +0.00) | −0.15 (−0.21 … −0.06) | −0.04 (−0.04 … −0.03) | −0.03 (−0.04 … −0.03) | 2.12 / 2.05 |
|  | `mol` | +0.01 (−0.12 … +0.18) | +0.03 (−0.04 … +0.12) | +0.14 (+0.14 … +0.15) | +0.15 (+0.14 … +0.15) | – |
| 2 charges 1.0–1.2 Å below the surface, 0.15 M | `vdw` | −0.65 (−1.54 … −0.11) | −0.51 (−0.97 … −0.17) | −0.13 (−0.23 … +0.03) | −0.12 (−0.17 … −0.03) | 2.17 / 2.03 |
|  | `mol` | −0.42 (−1.27 … +0.13) | −0.28 (−0.71 … +0.06) | +0.08 (−0.01 … +0.25) | +0.09 (+0.05 … +0.19) | – |
| two spheres, 0 M | `vdw` | −0.28 (−0.46 … −0.16) | −0.25 (−0.33 … −0.19) | −0.07 (−0.10 … −0.04) | −0.06 (−0.08 … −0.05) | 2.11 / 2.00 |
|  | `mol` | −0.07 (−0.25 … +0.04) | −0.03 (−0.12 … +0.03) | +0.14 (+0.12 … +0.17) | +0.15 (+0.13 … +0.16) | – |
| two spheres, 0.15 M | `vdw` | −0.24 (−0.31 … −0.18) | −0.22 (−0.25 … −0.20) | −0.06 (−0.10 … −0.03) | −0.05 (−0.07 … −0.04) | 2.13 / 2.12 |
|  | `mol` | −0.06 (−0.12 … +0.00) | −0.04 (−0.07 … −0.01) | +0.12 (+0.08 … +0.15) | +0.13 (+0.11 … +0.14) | – |
| association, 0 M (kcal/mol) | `vdw` | −0.002 (−0.003 … +0.000) | −0.001 (−0.002 … −0.000) | −0.002 (−0.002 … −0.001) | −0.001 (−0.002 … −0.001) | – |
|  | `mol` | −0.017 (−0.027 … −0.007) | −0.016 (−0.026 … −0.007) | −0.016 (−0.019 … −0.011) | −0.015 (−0.018 … −0.011) | – |
| association, 0.15 M (kcal/mol) | `vdw` | −0.000 (−0.002 … +0.001) | −0.000 (−0.001 … +0.001) | −0.000 (−0.001 … +0.001) | +0.000 (−0.000 … +0.000) | – |
|  | `mol` | −0.011 (−0.017 … −0.001) | −0.011 (−0.017 … −0.002) | −0.011 (−0.016 … −0.005) | −0.011 (−0.015 … −0.005) | – |

* **The placement spread halves.** At 0.5 Å the range falls 1.8 times for
  the charges 1.0–1.2 Å deep (`vdw`: 1.43 → 0.80 points), 2 times for the
  Kirkwood charges and 2 times for the two spheres. At 0.25 Å the ranges
  are already small (≤ 0.26 points) and fall up to 2 times.
* **The means move a little toward zero** at 0.5 Å, most for the
  near-surface pair (−0.65 → −0.51 % `vdw`, −0.42 → −0.28 % `mol`) and the
  Born ion (−0.52 → −0.47 %). The orders are unchanged (about 2). The `mol`
  plateau of +0.13–0.16 % is APBS' surface and stays.
* **With the sharp interface** nothing changes: the means move by at most
  0.16 points and the ranges stay the same. There the error of the boundary
  itself (first order, several %) dominates.

**A protein** (1YCR, 0.15 M, CUDA, 8 placements per spacing on a fixed grid,
the same for both methods; ΔΔG_bind with receptor and ligand on the complex's
grid; mean (sd) in kcal/mol):

| h (Å) | interface | ΔG_polar, grid | ΔG_polar, analytic | ΔΔG_bind, grid | ΔΔG_bind, analytic |
|---|---|---|---|---|---|
| 0.5 | sharp | −1230.27 (4.99) | −1224.06 (4.32) | 425.20 (2.16) | 424.31 (1.79) |
| 0.35 | sharp | −1206.19 (1.80) | −1204.67 (1.68) | 421.45 (0.59) | 421.17 (0.54) |
| 0.25 | sharp | −1190.83 (0.78) | −1190.48 (0.77) | 419.46 (0.42) | 419.36 (0.41) |
| 0.5 | fractional | −1165.69 (1.08) | −1161.38 (0.49) | 415.92 (0.59) | 415.32 (0.37) |
| 0.35 | fractional | −1159.18 (0.50) | −1157.90 (0.25) | 414.87 (0.22) | 414.68 (0.09) |
| 0.25 | fractional | −1156.93 (0.15) | −1156.64 (0.07) | 414.53 (0.08) | 414.45 (0.04) |

* **Fractional: the same limit, reached sooner.** Each sequence is fitted
  with its own observed order: 2.9 / 2.8 for ΔG_polar, 3.1 / 2.8 for
  ΔΔG_bind. The limits agree: ΔG_polar −1155.6 / −1155.8 and ΔΔG_bind
  414.3 / 414.3 kcal/mol (grid / analytic). The errors against these limits:

  | | ΔG_polar, grid | ΔG_polar, analytic | ΔΔG_bind, grid | ΔΔG_bind, analytic |
  |---|---|---|---|---|
  | 0.5 Å | −10.1 | −5.6 | +1.6 | +1.0 |
  | 0.25 Å | −1.4 | −0.8 | +0.19 | +0.15 |

  The placement sd halves at every spacing. The fractional section above
  extrapolates linearly to −1151.3, which overshoots for energies that
  converge this fast.
* **Sharp: a small gain.** The two methods differ by 6.2, 1.5 and
  0.35 kcal/mol at 0.5, 0.35 and 0.25 Å, and the sd falls by up to 13 %.
  The boundary's first-order error (tens of kcal/mol) dominates.

**Cost** (0.15 M, best of 2 on CUDA, 1 on the CPU, seconds per calculation):

| | CUDA, grid | CUDA, analytic | CPU, grid | CPU, analytic |
|---|---|---|---|---|
| 1YCR 0.5 Å, sharp | 0.20 | 0.16 | 1.57 | 0.97 |
| 1YCR 0.5 Å, fractional | 0.29 | 0.27 | 1.58 | 0.96 |
| 1YCR 0.25 Å, sharp | 1.04 | 0.86 | 13.3 | 7.7 |
| 1YCR 0.25 Å, fractional | 1.49 | 1.35 | 13.0 | 7.4 |
| 5DK3 0.5 Å (20 179 atoms), sharp | 4.30 | 5.75 | 30.0 | 21.6 |

* **Speed-up.** Dropping the reference solves saves 38–43 % on the CPU
  for 1YCR (28 % for 5DK3) and 7–20 % on CUDA.
* **Cost of G.** G is summed directly over all atoms at the layer's nodes.
  The layer's right-hand side takes 0.03 s for 1YCR at 0.5 Å, 0.18 s at
  0.25 Å and 2.1 s for 5DK3 (CUDA runs). For large proteins on a GPU this
  costs more than the reference solve it replaces.
* **Follow-up.** A treecode or multilevel summation would make G cheap.

**Limits.**
* A charge exactly on a node where G is needed (next to the dielectric
  boundary, or a charge in the solvent) cannot be decomposed. It raises an
  error that names the node.
* The coarser focusing levels are not decomposed; they only provide
  boundary values.
* Forces, which the roadmap builds on this decomposition (4.12), do not
  exist yet.

Reproduce: `python scripts/validate_analytic.py --placements 10 --interface
fractional --coulomb analytic [--surface vdw] --platform CUDA`.

## Nonlinear Poisson–Boltzmann (`equation="npbe"`)

**Change.** `Solver(equation="npbe")` and `--equation npbe` solve the
nonlinear PB equation for the solvated state, as APBS `npbe`:
* The reference state (`pdie`, no ions) stays linear. Without ions `npbe` is
  `lpbe` bit for bit.
* `pbsolve apbs compare` and `run` now write, read and compare `npbe`, so
  APBS inputs with `npbe` are no longer refused.
* `Solver.potential` returns the nonlinear potential and, as its energy,
  the free-energy functional.
* `Solver.solve_maps` solves the nonlinear equation for a solver with
  `equation="npbe"`.
* `equation="lpbe"` stays the default and is bit-identical to the previous
  commit. `npbe` needs `coulomb="grid"`.

**Method.** On the interior nodes of a level the discrete equation is
(APBS `Vc_vecpmg`, with the volume scaling of the linear operator):

F(u) = A_ε u + V N(u) − 4π l_B q = 0,
N(u) = κ̄²_c Σ_j −(c_j q_j / 2I) e^{−q_j u}.

* **Terms.** A_ε is the 7-point operator without its ion term, κ̄²_c the ion
  map, and c_j, q_j and I the ion concentrations, valences and ionic
  strength. For a 1:1 salt N = κ̄² sinh u. The exponents are clamped to ±85
  as in APBS.
* **Newton steps.** Newton's method solves J δ = −F with
  J = A_ε + V N'(u) and N'(u) = κ̄²_c Σ_j (c_j q_j² / 2I) e^{−q_j u}. J is
  the linear operator with κ̄² replaced by N'(u), so every step is an
  ordinary linear solve: multigrid CG on the CPU or the GPU, unchanged. The
  iteration runs on the backend: two element-wise operations
  (`boltzmann_slope`, `boltzmann_residual`; CUDA kernels with the CPU's
  expressions) give N'(u) and complete J u into the residual
  F = J u + V (N(u) − N'_J u) − f, so the line search needs no further
  operator. The host only damps the start and keeps the solution for the
  energy.
* **Damping and accuracy.** Steps are halved until ‖F‖ decreases (Armijo),
  which is the damping the roadmap asks for [Holst & Saied 1995]. The
  iteration starts from the linearized solution solved to 1e-3. It is
  inexact [Dembo et al. 1982; Holst et al. 1994]: each step's linear solve
  stops at η = min(0.1, ‖F‖/‖b‖), and no tighter than the final target
  ‖F‖ ≤ `tol`·‖b‖ needs. This cut the CG iterations of 1YCR from 36 to 15
  per level, with the same energy to 1e-6 kcal/mol.
* **Damped start.** Where the ions are, the linearized potential
  overshoots once |u| > 1, and Newton then crawls down the exponential
  (5DK3: residual 0.24–0.55 at the start, 8–9 steps). The start uses the
  planar Gouy–Chapman relation between the linear and the nonlinear
  potential, u = 2 asinh(u_lin/2) (Grahame), shifted to leave |u| ≤ 1
  unchanged. Newton steps (5DK3 / helix / −30 e sphere / 1YCR): 9 / 5 / 18
  / 2 → 4–5 / 4 / 5 / 2. The converged energies are unchanged.

**Energy.** APBS' free-energy functional (`Vpmg_energy` for `npbe` with
ions) is the energy-density form of Sharp & Honig (1990):

G = Σ q u − ½ Σ_faces ε (Δu)² c_a / 4π l_B − V Σ_nodes κ̄²_c Σ_j (c_j / 2I)(e^{−q_j u} − 1) / 4π l_B.

* **Stationary.** Its gradient is −F/4π l_B, so G is stationary at the
  discrete solution (tested).
* **Outer terms.** APBS adds, for every coarser focusing level, the
  dielectric and ion terms of its nodes outside the next finer box
  (`Vpmg_extEnergy`). This is invisible with `lpbe`, whose energy is
  ½ Σ q u, but with salt it is a sizeable part of the functional: 17 kT of
  ion energy and 6 kT of field energy for a −30 e sphere in a small fine
  box.
* **APBS quirks.** pbsolve reproduces the coarse nodes' weights
  (`Vpmg_setPart`: a node with any share of the finer box counts there)
  and APBS' face loop.

**Choices, with the evidence:**
* **Prototype first.** A Python prototype on `Solver.maps` and
  `solve_maps` first matched APBS 3.4.1's potential (6.5e-6, its
  constants). The energies differed by 1.9e-5. With APBS' source constant
  emulated, Σ q u agreed to 7e-9 and the dielectric term to 1e-7, but the
  ion term was 0.34 % high. APBS' log showed the outer terms; adding them
  gave 7e-9 and 5e-6. The Rust implementation equals the prototype to
  5e-15.
* **Box dependence.** At weak potentials the potential is the linearized
  one, to 1.6e-10 of its maximum (and ∝ q²). The `npbe` and `lpbe` energies
  nevertheless differ by up to 0.09 % on small boxes, independently of the
  charge scale. The difference shrinks with the box (9e-4 → 5e-5 for fadd
  12 → 30 Å), so it is the field energy the functional misses beyond the
  boxes, not a nonlinear effect. APBS has the same property. The
  Sharp–Honig volume form ½ Σ q u − ∫[½ ρ_ion u + kT Σ c_j (e^{−q_j u} − 1)]
  has no boundary term and converges faster with the box:

  | npbe − lpbe (kcal/mol) | coarse box | APBS functional | volume form |
  |---|---|---|---|
  | −8 e sphere, 0.15 M | 28 Å (default) | −1.173 | −1.423 |
  | | 38 Å | −1.542 | −1.594 |
  | | 48 Å | −1.632 | −1.643 |
  | | 68 Å | −1.654 | −1.655 |
  | −20 e helix, 0.15 M | 42 Å (default) | −2.195 | −2.539 |
  | | 88 Å | −2.879 | −2.880 |
  | | 132 Å | −2.875 | −2.875 |

  pbsolve keeps APBS' functional, which matches APBS. For strongly
  charged molecules the default boxes miss 20–30 % of the nonlinear
  correction; enlarge `cfac` and `fadd`. The volume form is a possible
  opt-in.

**APBS 3.4.1** (`pbsolve apbs compare --equation npbe`, 0.5 Å, `etol`
1e-10, CUDA; the maps were identical in every case):

| System | levels | ΔG_polar npbe, pbsolve / APBS (kcal/mol) | rel. (APBS units) | solvated energy rel. (APBS units) | npbe − lpbe, pbsolve / APBS |
|---|---|---|---|---|---|
| 1YCR, 0.15 M | 2 | −1221.2952 / −1221.3095 | −4.8e-7 | −5.0e-11 | — |
| −8 e sphere (a = 4 Å), 0.15 M | 1 | −2673.1706 / −2673.1952 | −3.0e-6 | −8.3e-11 | −1.1728 / −1.1729 |
| −20 e helix (20 phosphates), 0.15 M | 2 | −5752.7325 / −5752.7723 | −5.3e-6 | −2.3e-9 | −2.2046 / −2.2047 |
| −30 e sphere, 0.1 M | 1 | −37657.0556 / −37657.4025 | −3.0e-6 | +1.6e-10 | −97.50 / — |
| same, small fine box (cfac 3, fadd 8) | 2 | −37670.9226 / −37671.2508 | −3.5e-6 | +9.7e-12 | — |

* **The solvated nonlinear energies agree to ≤ 2.3e-9 in APBS units.**
  The remaining 3–5e-6 of ΔG_polar is the reference state's `mdh` boundary
  constants (see *Comparing with APBS*), the same as with `lpbe`.
* **The nonlinear correction agrees to 1e-4 kcal/mol** (npbe − lpbe, both
  programs).
* **The potentials** differ by 6.3–6.5e-6 of their maximum, APBS' source
  constant; injected into pbsolve's solver, APBS' own maps give the same.
* **Strong nonlinearity.** At the −30 e sphere the linearized potential
  at the ions is −21.9 kT/e, the nonlinear one −7.5. Newton converges in
  32 CG iterations in total.

**Checks:**
* Rust (`nonlinear.rs`, 6 tests):
  * 1:1 salt terms are sinh and cosh, clamped at 85; any electroneutral
    salt has N(0) = 0 and N'(0) = 1;
  * `Vpmg_setPart` shares;
  * the functional is stationary at the solution (halving a perturbation
    quarters the change, ratio 4.00 ± 0.05);
  * weak potentials give the linearized potential to O(q²), with an energy
    gap that does not depend on the charge scale;
  * strong charges converge, with a more negative ΔG_polar;
  * the outer terms equal an independent sum over the nodes beyond the
    finer box.
* Mutations caught: 15/15. In `nonlinear.rs`, 12:
  * signs, powers and clamp of the Boltzmann terms;
  * the energy's factors and charges;
  * the Newton direction and the residual's volume;
  * the outer weights and shares, the face weights.

  In `pb.rs`, 3: the outer terms dropped or added with the wrong sign,
  caught by the APBS tests after the functional was computed in one place
  for `polar_solvation` and `potential`.

  The device iteration, 10/10:
  * the residual's sign, mask and interior test;
  * the slope's power;
  * the Newton direction, ‖b‖, and a Jacobian without its ion term;
  * the same three in the CUDA kernels, caught by the GPU test.
* `tests/test_nonlinear.py` (6 tests, 2 s):
  * weak and strong potentials;
  * the energies and `potential()` on a two-level grid;
  * `solve_maps`;
  * the parameter and the APBS keyword;
  * the command line.
* `tests/test_apbs_parity.py`, with APBS:
  * `npbe` on the synthetic molecule with a 1:1 salt (`mdh`), a 2:1 salt
    (`sdh`, `pdie` 2) and `zero`/`vdw`/`spl2`;
  * three focusing levels;
  * the −30 e sphere;
  * an existing APBS input with `npbe` (`pbsolve apbs run`).
* GPU: `nonlinear_pb_matches_the_cpu` (one to three levels, 1:1 and 2:1
  salts): ≤ 1e-10. The field solve's functional equals
  `polar_solvation`'s.
* Default path: the fingerprint set is bit-identical on CPU and CUDA.

**Cost** (default `tol`, seconds per calculation, best of 2 on CUDA; the
reference state is the same). "First version" is the Newton iteration on
the host, with the linearized start:

| | lpbe | npbe, first version | npbe | CG iterations per level, lpbe / npbe |
|---|---|---|---|---|
| 1YCR 0.5 Å, CUDA | 0.20 | 0.44 | 0.34 | 12 / 14 |
| 1YCR 0.5 Å, CPU | 1.58 | 2.16 | 2.13 | 12 / 14 |
| 1YCR 0.25 Å, CUDA | 1.04 | 2.67 | 1.96 | 13 / 15 |
| 1YCR 0.25 Å, CPU | 13.2 | 18.1 | 17.5 | 13 / 15 |
| 5DK3 0.5 Å, CUDA | 4.28 | 12.7 | 7.20 | 19–20 / 29–32 |
| −20 e helix, CUDA | 0.10 | 0.36 | 0.25 | 13 / 23 |
| −30 e sphere, CUDA | 0.01 | 0.12 | 0.04 | 9 / 21 |

* **Ratio.** `npbe` now costs 1.3–1.4 times `lpbe` on the CPU and 1.7–1.9
  times on CUDA (5DK3 3.0 → 1.7).
* **Where it went** (5DK3, per level): the profile of the first version
  showed 8–9 Newton steps, each with uploads of the maps (0.07 s), host
  residuals and line searches (0.1 s) and transfers of u and δ.
* **What remains.** Every step still sets up a new multigrid hierarchy
  (about 0.1 s for 5DK3), and the dielectric maps are uploaded again for
  each Jacobian.
* **Not done: reusing the hierarchy across steps.** Its Chebyshev smoother
  is tuned to the old diagonal, and the ion term of the new Jacobian
  (κ̄² cosh u) can far exceed it.

**Device memory.** The iteration keeps 4 vectors (u, the charges, κ̄²,
N'(u)) beside the linear solve's buffers. A GPU test asserts that the
pool's peak equals `solve_device_bytes` plus exactly these, for 0.11–7.2
million nodes. `calculation_device_bytes` adds them to the memory a
calculation, or a batch worker, sets aside. The first version needed 10:
* the device maps lived until the level ended;
* the residual was copied for the right-hand side (now J δ = F is solved
  and u ← u − λδ);
* a trial vector was kept (the line search now updates u in place);
* the diagonal was copied as a mask (the residual kernel now tests the
  node index).

**Not done:** the Sharp–Honig volume form as an option; ion-size effects
(size-modified PB, roadmap 4.11); forces.

Reproduce: `pbsolve apbs compare complex.pqr --ionic-strength 0.15
--equation npbe --spacing 0.5`; `pytest tests/test_apbs_parity.py -k
nonlinear`.

## Spline surfaces and forces (`surface="spl2"`/`"spl4"`, `Solver.polar_forces`)

**Change.**
* **Surfaces.** `surface="spl2"` and `"spl4"` with `swin` (Å, default 0.3):
  APBS' spline surfaces (`srfm spl2`, `spl4`), smooth in the atom positions.
* **Forces.** `Solver.polar_forces(coords, charges, radii)` returns a
  `PolarForces`: ΔG_polar, the force on every atom (kcal/mol/Å) and its
  charge–field (`qf`), dielectric-boundary (`db`) and ionic-boundary (`ib`)
  parts, as APBS `calcforce comps`.
* **APBS.** `pbsolve apbs compare --forces` compares the forces; `--surface
  spl2|spl4` and `--swin` the surfaces; APBS inputs with `srfm spl2/spl4` are
  read.
* **Defaults unchanged.** `mol` stays the default; the default path is
  bit-identical, and `polar_forces` gives `polar_solvation`'s energy bit for
  bit.

### The spline surfaces

**Method.** Every atom of radius r has a switching function S(d) of the
distance d to its centre that rises from 0 at r − w to 1 at r + w (w =
`swin`). The solvent fraction is H = Π_atoms S, the face dielectric ε = pdie
+ (sdie − pdie) H, and the ion accessibility K the same product with
r + r_ion. The smooth boundary is what makes the energy differentiable in the
atom positions [Im, Beglov & Roux 1998]. `spl2` uses the cubic ¾ s²/w² − ¼
s³/w³ of s = d − r + w (C¹). `spl4` uses the septic smoothstep with three
vanishing derivatives, which APBS added for multipole force fields
[Schnieders et al. 2007]. The maps follow APBS' `fillcoCoefSpline` and
`fillcoCoefSpline4` operation for operation:
* the atoms in order;
* the guards that stop multiplying once a value is ≤ 1e-12;
* atoms off the box skipped;
* `spl4`'s monomial evaluation, which squares the distance differently for
  the x/y and the z faces.

**APBS' rounding in `spl4`.** APBS evaluates the septic in monomials of d
with coefficients up to 2e5 (atoms) and 1e7 (atom plus ion radius). It
therefore loses up to 7 digits: 5e-10 for r = 1.7 Å, 2e-7 for the ion map
with r = 3.7 Å (measured against the closed form 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷).
pbsolve reproduces this to keep APBS' maps.

**APBS 3.4.1** (synthetic 30-atom molecule, 0.6 Å, `pbsolve apbs compare`):

| case | partial faces | max \|Δε\| | max \|ΔK\| | solvated energy (APBS units) | ΔG_polar (APBS units) |
|---|---|---|---|---|---|
| `spl2`, 0.15 M, `mdh` | 1773 | 5.0e-6 | 5.0e-8 | 1.0e-9 | −6.8e-7 |
| `spl4`, 0.15 M, `mdh` | 1759 | 5.0e-6 | 5.0e-8 | 6.0e-10 | −7.1e-7 |
| `spl2`, `swin` 0.5, pdie 2, 0.1 M, `zero` | 2933 | 5.0e-6 | 5.0e-8 | −3.9e-10 | −1.3e-9 |
| `spl4`, `chgm spl2`, 0.1 M, `zero` | 1759 | 5.0e-6 | 5.0e-8 | 8.4e-12 | −3.1e-10 |

* **Maps.** No face or node differs by more than half of the last digit APBS
  writes to its DX files (7 significant digits).
* **Energies.** With `bcfl zero` ΔG_polar agrees to 3e-10. With `mdh` the
  remaining 7e-7 is the reference state's boundary constants, as for `mol`.
  `npbe` on `spl2` agrees as well.

**A different model.** The smooth boundary moves the dielectric jump inwards:
ε rises fast once S > 0, so the effective sharp boundary lies near r − 0.7w.

| Born ion (a = 2 Å) | 0.5 Å | 0.333 Å | 0.25 Å | 0.2 Å |
|---|---|---|---|---|
| `spl2` | −16.49 % | −14.56 % | −13.73 % | −14.07 % |
| `spl4` | −13.49 % | −12.22 % | −11.01 % | −11.09 % |

(`mol`, `harmonic` and `mol+frac` unchanged.) A 6 Å sphere has the Born
radius 5.77 Å (`spl2`) and 5.82 Å (`spl4`) at 0.35 Å.
* **Proteins.** With van der Waals spheres and no probe, crevices count as
  solvent. 1YCR has ΔG_polar −1231.3 (`mol`), −1754.1 (`vdw`) and −2609.0
  kcal/mol (`spl2`, 0.5 Å).
* **Radii.** As in APBS, spline surfaces need radii fitted for them (for
  example [Nina, Im & Roux 1999] for a smoothed boundary); with `mol` radii
  they are not an accuracy option.
* **Exposed charges.** A charged atom whose own position has H > 0 (a radius-0
  atom near the surface) feels a grid-dependent self-energy. The synthetic
  molecule's atom 5 (radius 0, H = 0.36) has forces of 40, 80, 128, 134 and
  63 kcal/mol/Å in changing directions on five grids, and ΔG_polar runs from
  −412 to −485 kcal/mol as h → 0. 1YCR's 18 radius-0 hydrogens are buried (H
  = 0).

**`coulomb="analytic"` (fixed).** The decomposition takes ε = `pdie` on the
faces between nodes that lie one spacing inside an atom ("deep" nodes) and
computes its source only off them. The spline window reaches such faces
when `swin` exceeds about the spacing; the region is now shrunk by `swin`.
A unit charge at the centre of a 3 Å `spl2` sphere (pdie 1) has an exact
radial solution, ΔG = ½ q² l_B ∫ (1/ε(r) − 1/pdie) dr/r² (kcal/mol):

| `swin`, h | exact | `coulomb="grid"` | `"analytic"` before | `"analytic"` now |
|---|---|---|---|---|
| 0.3, 0.2 Å | −59.465 | −59.520 | −59.487 | −59.520 |
| 0.6, 0.2 Å | −65.293 | −65.280 | −53.245 | −65.280 |
| 1.0, 0.25 Å | −75.265 | −75.307 | −50.265 | −75.305 |
| 1.0, 0.2 Å | −75.265 | −75.263 | −44.982 | −75.262 |

The analytic and grid results now agree within 1e-4 at every spacing and
window. The sparse source equals the dense formula on a 0.2 Å grid with the
default and with a 0.6 Å window (`coulomb.rs`), and
`tests/test_forces.py` checks the 1.0 Å case against the integral; without
the fix it gives −50.26.

### Forces

**Method.** pbsolve's free energy is the functional of the discrete
potential ([`nonlinear.rs`](../crates/pbsolve-core/src/nonlinear.rs); for the
linearized equation with the ion density ½u²):

G(u; R) = Σ_n q_n(R) u_n − ½ Σ_faces c_f ε_f(R) (Δu_f)² / 4π l_B − V Σ_n κ̄²_n(R) ρ(u_n) / 4π l_B.

It is stationary in u at the solution, so −dG/dR is the explicit derivative
at fixed u [Gilson et al. 1993; Im, Beglov & Roux 1998]:
* **qf** = −q ∇_R Σ_n W_n(R) u_n, the derivative of the charge-assignment
  weights (`spl0`: trilinear; `spl2`, `spl4`: the B-splines and their
  derivatives);
* **db** = ½ Σ_faces c_f ∂ε_f/∂R (Δu_f)² / 4π l_B, with ∂H_f/∂R_a = H_f ∇ ln
  S_a;
* **ib** = V Σ_n ∂κ̄²_n/∂R ρ(u_n) / 4π l_B: ρ = ½u² (`lpbe`), the Boltzmann
  terms (`npbe`), ln D / N_A a³ (`smpbe`).

**Scope.** The forces come from each state's finest level, solvated minus
reference (APBS `print force 1 - 2`). The coarser levels and the boundary
values' dependence on the atoms are left out, as in APBS. The loops, node
ranges and sums follow `qfForceSpline1/2/4`, `Vpmg_dbForce` and
`Vpmg_ibForce`, with two deliberate differences:
* Where APBS sets an atom's force to zero because the atom is too close to
  the box edge, pbsolve fails and asks for a larger grid.
* APBS' choice to set a trilinear qf component to zero when the atom sits
  exactly on a grid line is kept.

**The exact gradient.** On one level with `bcfl zero` the forces are the
derivative of the discrete energy. Central differences of `polar_solvation`,
Richardson-extrapolated, agree to 1.7e-9 to 4e-9 of the largest force
(`forces.rs` tests, 4 atoms, a non-cubic 41³ grid):
* `lpbe` with `spl0` and `spl2` charges;
* `npbe` and `smpbe` with charges × 6;
* every term contributes.

On `spl4` the agreement is 6e-7 (APBS' map rounding). With `mdh` the
neglected boundary term is 7e-5 (fadd 10 Å) and 5e-6 (fadd 20 Å) of the
largest force (synthetic molecule).

**APBS 3.4.1** (`calcforce comps`, `print force solvated - reference`;
synthetic molecule, 0.6 Å, maximum difference per term over all atoms
relative to the largest force, APBS units):

| case | largest \|F\| (kJ/mol/Å) | qf | db | ib | total |
|---|---|---|---|---|---|
| `spl2`, `chgm spl2`, 0.15 M, `mdh` | 242.3 | 2.4e-7 | 1.8e-9 | 7.2e-10 | 2.4e-7 |
| `spl2`, `chgm spl0`, 0.15 M, `zero` | 792.1 | 5.7e-10 | 2.3e-10 | 1.4e-10 | 4.3e-10 |
| `spl4`, `chgm spl4`, `swin` 0.4, pdie 2, 0.1 M, `mdh` | 74.7 | 3.8e-7 | 1.9e-9 | 1.9e-9 | 3.8e-7 |
| `spl2`, `chgm spl2`, `npbe`, 0.15 M, `mdh` | 242.3 | 2.4e-7 | 4.9e-10 | 6.5e-10 | 2.4e-7 |

With `bcfl zero` every term agrees to ≤ 6e-10. With `mdh` qf carries the
reference state's boundary constants (as the energies do); db and ib agree
to ~1e-9. APBS also computes forces on the finest levels only (one force
calculation per state).

**A charge in a dielectric sphere** (Kirkwood; analytic). A unit charge
2.5 Å off the centre of an uncharged sphere (radius 6 Å, pdie 2, sdie 78.54;
0.35 Å, mean of 4 grid placements). The charge is pushed outwards, the sphere
the other way:

| surface | force on the charge | Kirkwood at the Born radius | Kirkwood at 6 Å | force on the sphere | net force |
|---|---|---|---|---|---|
| `spl2` | 3.1416 ± 0.034 | 3.1394 (5.773 Å) | 2.7037 | −3.1081 | +0.034 |
| `spl4` | 3.0452 ± 0.041 | 3.0428 (5.820 Å) | 2.7037 | −3.0650 | −0.020 |

(kcal/mol/Å; the Born radius is that of the same sphere's Born energy on
the same grids.)
* **qf against Kirkwood.** The reaction-field force agrees with Kirkwood's to
  0.07–0.08 % once the boundary is placed where the surface puts it.
* **Momentum.** The sphere's dielectric-boundary force balances it to 1 %.
  The forces across the axis stay below 0.06.

**Translation invariance.** The net force on an isolated solute vanishes in
the continuum. On a grid it measures the discretisation error (the same
sphere and charge, 8 placements):

| `swin` | h (Å) | force on the charge | its sd | net force | its sd |
|---|---|---|---|---|---|
| 0.3 | 0.5 | 3.175 | 0.092 | −0.072 | 0.319 |
| 0.3 | 0.35 | 3.134 | 0.032 | +0.099 | 0.155 |
| 0.3 | 0.25 | 3.122 | 0.042 | −0.056 | 0.163 |
| 0.6 | 0.5 | 3.700 | 0.111 | +0.088 | 0.463 |
| 0.6 | 0.35 | 3.653 | 0.062 | −0.004 | 0.240 |
| 0.6 | 0.25 | 3.641 | 0.052 | −0.031 | 0.122 |

* **qf is accurate** to a few per cent per placement.
* **db is noisy.** It is the small difference of large pressures over the
  whole surface, so it carries the derivative of the grid's placement noise
  in the energy: 5–10 % of the sphere's force per placement. It shrinks
  with h only once h is well below 2w. This is the inaccuracy of
  finite-difference dielectric-boundary forces that Cai et al. (2011) address
  with a different discretisation of the boundary force (not done).

**Checks:**
* Rust:
  * `spline.rs` (3 tests): the switches against their closed forms, ∇ ln S
    against differences, the maps against an independent product;
  * `forces.rs` (3 tests): the exact gradient above, and qf in a linear
    potential (−q∇u exactly for every `chgm`; zero on a grid line, as APBS);
  * the B-spline derivatives against differences.
* `tests/test_forces.py` (8 tests, 10 s):
  * the API and the energy equality;
  * the errors (sharp surfaces, `coulomb="analytic"`, atoms near the grid
    edge);
  * the gradient in kcal/mol/Å;
  * Kirkwood and momentum for `spl2` and `spl4`;
  * the spline maps and parameters;
  * the command line;
  * CUDA = CPU to 1e-9 (forces of 1YCR on both platforms differ by 5.6e-12
    kcal/mol/Å).
* `tests/test_apbs_parity.py`:
  * maps and energies for `spl2`/`spl4` (`lpbe`, `npbe`) and an APBS input
    with `srfm spl4`;
  * forces in 4 cases and through the command line.
* Mutations caught: 14/14:
  * db (factor, face, a dropped face, loop range);
  * ib (factor, ion radius, nonlinear density);
  * qf (sign, spacing);
  * ∇ ln S (sign, window);
  * the ion map's switch;
  * the reference state's forces added instead of subtracted.
* Default path: the fingerprint set is bit-identical on CPU and CUDA; the
  existing rows of the Born table are unchanged.

**Cost** (0.15 M, 0.5 Å, `chgm spl2` with `spl2`; seconds per calculation,
minimum of 3). GROMACS shared the GPU and the CPU during these runs, so the
absolute times are high; the differences between the columns are what matter:

| | `mol` | `spl2` | `spl2` + forces | energy and force stage |
|---|---|---|---|---|
| 1YCR, CPU | 1.82 | 1.79 | 1.82 | 0.07 |
| 1YCR, CUDA | 0.44 | 0.49 | 0.49 | 0.05 |
| 5DK3, CUDA | 5.53 | 5.02 | 5.54 | 0.72 |

* **Forces.** They add the download of both finest potentials and a host
  pass over every atom's stencils: 2 % (1YCR) to 10 % (5DK3).
* **Spline maps.** They are built on the host and cost about as much as the
  molecular surface on the device.

**Not done.**
* The accurate dielectric-boundary force of Cai et al. (2011) and forces of
  the reaction-field decomposition (4.5); forces of sharp surfaces (APBS
  refuses them too).
* The coarser levels' and the boundary's share of the force.
* Forces along trajectories and molecular dynamics through OpenMM.

Reproduce:
* `cargo test -p pbsolve-core forces spline`;
* `pytest tests/test_forces.py tests/test_apbs_parity.py -k "forces or spl"`;
* `pbsolve apbs compare molecule.pqr --surface spl2 --chgm spl2 --forces`.

## Gaussian dielectric (`surface="gauss"`)

**Change.** `surface="gauss"` with `gauss_sigma` (default 0.93) and
`gauss_contour` (default 20); CLI `--surface gauss --gauss-sigma
--gauss-contour`. Not an APBS option, so `pbsolve apbs compare` refuses it.
The default path is bit-identical (CPU and CUDA fingerprints).

**Method.** An atom of radius R > 0 contributes the density
g = exp(−d²/σ²R²) at distance d; the molecular density is ρ = 1 − Π(1 − g)
and the face dielectric ε = ρ·pdie + (1 − ρ)·sdie [Grant, Pickup & Nicholls
2001; Li et al. 2013]. This is the form of the spline surfaces,
ε = pdie + (sdie − pdie)·Π S, with S = 1 − g. The mobile ions pay the Born
energy of leaving the bulk solvent [Jia et al. 2017], so the node
accessibility is K = exp(−z² l_B/(2 r_ion)·(1/ε − 1/sdie)) and κ̄² =
κ̄²_bulk·K. Jia et al. (2017, eqs. 1–2) and Chakravorty et al. (2018, eqs.
4–5) print this term with the opposite overall sign, which would draw ions
into the low dielectric; their text says it keeps them out, and the Born
transfer energy (≥ 0) does that. g is the product of three axis factors
precomputed per atom, cut off where g ≤ 1e-12 (5.26 σR); the maps are built
on the host.

**Semi-analytic reference.** A unit charge at the centre of one Gaussian
atom: ε(r) and K(r) are radial, so (ε r² u′)′ = κ̄² K r² u. pbsolve's test
solves it by shooting (DOP853, rtol 1e-12) in the variables
w = u − l_B q/(pdie r) and ε r² u′, which removes the singularity at the
charge. Without ions it reduces to ΔG = ½ q² l_B ∫ (1/ε − 1/pdie) dr/r²; the
shooting reproduces that quadrature to 2e-7 and 3e-8 kcal/mol (relative 2e-10) for pdie 1 and 4, and
does not depend on its outer radius (25–50 Å) in any printed digit.

**Which reference state.** ΔG_polar subtracts a reference state on the same
grid, so that each charge's grid self-energy cancels. That needs the same ε
around every charge in both states. With a homogeneous `pdie` reference this
fails: next to a nucleus ε − pdie ≈ (sdie − pdie)·P·d²/σ²R² (P: the other
atoms' solvent fraction there), and the trilinear charge, spread over one
spacing, sees a different dielectric in the two states. The error is first
order in h with a large coefficient. It falls only once h resolves r* =
σR·√(pdie/(sdie − pdie)), 0.21 Å for R = 2 Å and pdie 1. Mean over four grid
placements, relative to the exact solution:

| homogeneous reference | exact (kcal/mol) | 0.5 Å | 0.33–0.35 Å | 0.25 Å | 0.18 Å |
|---|---|---|---|---|---|
| R 2 Å, pdie 1 | −1230.34 | +63.4 % | +51.0 % | +41.9 % | +31.8 % |
| R 2 Å, pdie 4 | −148.97 | +41.7 % | +30.1 % | +23.3 % | +16.8 % |
| R 4 Å, pdie 4 | −74.48 | +23.3 % | +16.7 % | +12.0 % | +8.4 % |

The placement spread reaches 43 kcal/mol (R 2 Å, pdie 1). As Li et al.
(2014), the reference state therefore keeps the Gaussian ε wherever it is at
most `gauss_contour` and puts `pdie` beyond (they put vacuum there; the same
with pdie 1), without ions. Every charge of an atom with R > 0 sits where
ε = pdie, inside the contour, and its self-energy cancels:

| contour reference (ε ≤ 20) | exact (kcal/mol) | 0.5 Å | 0.33–0.35 Å | 0.24–0.25 Å | 0.18 Å |
|---|---|---|---|---|---|
| R 2 Å, pdie 1 | −164.45 | +3.18 % | +2.43 % | +1.99 % | +1.37 % |
| R 2 Å, pdie 1, 0.15 M | −164.67 | +3.18 % | +2.43 % | +1.98 % | +1.37 % |
| R 2 Å, pdie 4 | −41.03 | +1.78 % | +0.95 % | +1.31 % | +1.00 % |
| R 1.5 Å, pdie 4, 0.15 M | −54.94 | +1.00 % | +2.05 % | +1.19 % | +1.34 % |
| R 4 Å, pdie 4, 0.15 M | −20.70 | +1.30 % | +0.94 % | +0.68 % | +0.45 % |

* First order (0.9–1.2) when the contour lies well away from the charge
  (R 4 Å: 1.83 Å); erratic within about 1 % when it comes within 0.7–0.9 Å
  (R 1.5–2 Å), as a sharp boundary near a charge does.
* **Salt.** The salt effect (0.15 M) matches the exact one to 0.07 %:
  −0.18304 to −0.18309 against −0.18297 kcal/mol (R 4 Å), −0.21494 against
  −0.21495 (R 2 Å, 0.5 Å). Without the Born penalty it would be −3.92 and
  −14.65 kcal/mol.

**1YCR** (complex and chain B on the complex's grid, 0.15 M, CUDA, mean
(sd) over two grid placements; mbondi2 radii, so no charge has radius 0):

| ΔG_polar / ΔΔG_polar (kcal/mol) | 0.49 Å | 0.35 Å | 0.25 Å |
|---|---|---|---|
| `mol`, pdie 1 | −1286.0 (2.4) / 434.9 (1.7) | −1256.4 (1.2) / 429.0 (0.3) | −1240.6 (0.5) / 427.6 (0.3) |
| `gauss`, pdie 1, homogeneous reference | −38 992 (558) / 438.8 (2.3) | −51 011 (1.3) / 477.8 (10.8) | −61 427 (310) / 529.6 (4.7) |
| `gauss`, pdie 1 | −1239.9 (4.9) / 385.9 (2.2) | −1213.7 (0.9) / 383.3 (0.8) | −1200.0 (0.7) / 383.5 (0.2) |
| `mol`, pdie 4 | −300.3 (0.5) / 105.2 (0.3) | −294.5 (0.3) / 104.0 (0.1) | −291.3 (0.1) / 103.7 (0.1) |
| `gauss`, pdie 4, homogeneous reference | −5075 (56) / 111.5 (0.6) | −6101 (1.0) / 116.7 (1.4) | −6879 (24) / 122.1 (0.5) |
| `gauss`, pdie 4 | −550.7 (2.4) / 103.4 (0.2) | −544.1 (0.0) / 103.8 (0.4) | −542.2 (0.6) / 103.5 (0.3) |

* With the contour, ΔG_polar converges as with `mol`, and ΔΔG_polar changes
  by less than 3 kcal/mol between 0.49 and 0.25 Å (`mol`: 7). The model
  differs: dielectric up to 20 inside the molecule counts as solute. Li et
  al. (2014) report scale errors below 0.8 % at 0.5 Å relative to 0.167 Å
  for 91 proteins (DelPhi, pdie 4, vacuum beyond the contour); here pdie 4
  changes by 1.6 % between 0.49 and 0.25 Å, with other radii and charge
  assignment.
* **pdb2pqr's AMBER radii.** The 18 charged hydrogens of radius 0 carry no
  density; they sit where the Gaussian ε is 9–15 (pdie 1) or 12–17 (pdie 4),
  inside the contour, and are accepted. ΔG_polar −990.1 / −971.0 / −966.4,
  ΔΔG_polar 404.1 / 403.8 / 402.0 (pdie 1); with pdie 4 ΔΔG_polar 117.5 /
  118.4 / 119.6, with a placement spread up to 3.6 kcal/mol from the
  hydrogens near the contour. With σ = 0.85 one of them lies at ε = 20.5,
  outside, and pbsolve refuses the input: its self-energy would diverge as
  h → 0.

**`npbe` refused.** The Born penalty is bounded, z² l_B/(2 r_ion)·(1/pdie −
1/sdie) = 138 kT for pdie 1 and 33 kT for pdie 4 (r_ion 2 Å), but the
potential next to a point charge is not, so the Boltzmann density K e^{|u|}
diverges at the charges inside the molecule. With radius-0 atoms given 1 Å,
the synthetic molecule's ΔG_polar is −226.2 instead of −170.3 kcal/mol
(`lpbe`, pdie 1) and −154.4 instead of −71.8 (pdie 4); 1YCR does not
converge (pdie 1) or gives −7171.8 instead of −551.9 (pdie 4). `smpbe` bounds
the density: −170.4, −71.9, −1241.0 and −552.3, close to `lpbe`.

**Checks.**
* Maps against the formulas with every atom and no cutoff, on a non-cubic
  grid with an atom of radius 0 and one outside the grid: ε and K within
  1e-9; the reference keeps exactly the faces with ε ≤ `gauss_contour`.
* With the contour above `sdie`, no ions and `bcfl zero` the two states are
  one system and ΔG_polar is exactly 0.
* CPU = CUDA (`smpbe`, synthetic molecule) within 1e-9.
* 9/9 mutations caught (contour comparison, Born radius, face position,
  cutoff, homogeneous reference, the radius-0 check, the penalty, σ, the
  valence check).

**Cost** (1YCR, 0.25 Å, 0.15 M; shared CPU, load ~10, and GPU):

| | maps (s) | total (s) |
|---|---|---|
| CPU `mol` | 0.36 | 14.9 |
| CPU `gauss` | 0.96 | 14.4 |
| CUDA `mol` | 0.16 | 1.92 |
| CUDA `gauss` | 1.22 | 2.93 |

The Gaussian maps are built on the host for both states (the reference
needs them for its contour). One `exp` per atom and point took 2.05 s (CPU)
and 2.29 s (CUDA); the per-axis factors halved that, and restricting each row
to the cutoff sphere saved nothing measurable.

## Per-atom and per-residue decomposition

**Change.**
* `PolarResult.atom_energies`: every atom's ΔG_polar in kcal/mol, ½ q_i
  [u_solvated − u_reference] at the atom with the energy's charge weights
  (`chgm`'s energy method: trilinear for `spl0` and `spl4`, the `spl2`
  weights for `spl2`); with `coulomb="analytic"`, ½ q_i times the reaction
  field. `None` for `npbe` and `smpbe` with ions, whose energy is a
  functional rather than a sum over the charges.
* `SolvationTerms.atom_polar` / `atom_nonpolar` (γ·SASA_i; β belongs to no
  atom), `PBSAFrame.atom_terms()` (every complex atom's share of the binding
  terms: the complex's minus the receptor's or ligand's),
  `PBSAFrame.residue_terms(labels)` and `pbsolve mmpbsa --per-residue FILE`
  (mean over frames, standard error of the total) [Gohlke, Kiel & Case 2003].
* `pbsolve apbs compare --atom-energies` compares the shares with APBS
  `calcenergy comps`.
* The shares come from the potential at the charged nodes that the energy
  reads anyway; the energy is summed as before and stays bit-identical.

**Why the shares are well defined.** The polar energy of a linear state is
½ Σ_ij q_i R_ij q_j with the discrete reaction-field operator R; atom i's
share ½ q_i Σ_j R_ij q_j gives each charge its self term and half of each
pair term. The shares therefore add up exactly and superpose linearly.

**Checks** (`pb.rs`, `tests/test_decomposition.py`):
* the shares sum to ΔG_polar to 1e-12 for `spl0`, `spl2`, `spl4` and
  `coulomb="analytic"` with fractional faces; the forces path gives the same
  shares bit for bit; uncharged atoms have none;
* linear response, four charged atoms: E_i of all charges equals E_i of
  charge i alone plus its pair terms E_i^(ij) − E_i^(i), from 4 single and
  6 pair solves on the same grid and dielectric, to 1e-8 of the largest share
  (two focusing levels, `mdh`); with one level and zero boundary values the
  pair terms are symmetric to 1e-8;
* the binding shares per atom sum to ΔΔG_polar (1e-11) and, with β added, to
  ΔΔG_nonpolar; the `--per-residue` columns sum to the mean terms over the
  frames.

**APBS 3.4.1** (`calcenergy comps`: ½ q_i u(r_i) per atom in each state;
largest difference relative to the largest share, APBS units):

| system | `mdh` | `zero` |
|---|---|---|
| synthetic molecule, 0.15 M (`spl0`; `spl4`; `spl2` surface) | 2.2e-6; 2.2e-6; 1.7e-6 | 1.0e-9 (0.1 M) |
| 1YCR, 0.15 M, 129³, two levels | 2.8e-6 | 3.2e-8 |

As for the energies, the remainder with `mdh` is APBS' boundary constants.
APBS takes the `spl2` energy with the `spl2` weights but its per-atom
energies with trilinear ones, so they do not add up to its energy; the
comparison refuses `chgm="spl2"`.

**1YCR, binding shares** (pdb2pqr AMBER radii, 0.15 M, default grid;
ΔΔG_polar = +423.7 kcal/mol). Most favourable: B:LYS:24 −37.8, the
C-terminal A:VAL:109 −30.6, A:GLU:52 −24.2, A:GLU:95 −22.4, A:ASP:46 −17.1;
most unfavourable: B:ASN:29 +89.2, B:GLU:28 +69.0, A:LYS:51 +55.6 (polar,
kcal/mol). These are solvation shares; the Coulomb interaction in ΔE_MM
offsets much of them.

**Mutations.** 9/9 caught: the factor ½, the potential, the reference
state, the energy's weights (`spl4`), the reaction field of the
decomposition (only beyond χ: caught by a lone radius-0 charge whose `spl2`
energy nodes reach out of it), the ligand's sign in `atom_terms`, the
residues' non-polar sums, the CLI's columns and the reference state in the
APBS comparison.

**Cost.** The default-path fingerprints are bit-identical (CPU and CUDA).
The shares reuse the potential at the charged nodes; the energy stage of
5DK3 (20 179 atoms) grows from 3 to 10 ms on the CPU and from 5 to 53 ms on
CUDA, under 1 % of the calculation (shared machine).

## Non-polar terms: surface, volume and dispersion (`apolar_energy`)

**Change.**
* `pbsolve.apolar_energy(coords, radii, epsilons)` computes the non-polar
  model of APBS `apolar` [Wagoner & Baker 2006], G_np = γA + pV + ρ̄ Σ_i ∫
  u_i^att θ dy, and returns `ApolarEnergy` with every part, also per atom.
* `pbsolve.apbs.read_parm` reads APBS parameter files, and
  `pbsolve.apbs.apolar_energy` runs APBS on the same input.
  `pbsolve.openmm_bridge.lennard_jones_from_system` takes R_min/2 and ε from
  an OpenMM system.
* The defaults are Wagoner & Baker's alkane protocol (APBS'
  `examples/alkanes`) in kcal/mol, with a 0.5 Å grid instead of 0.3 Å.
* `nonpolar_energy` (γ·SASA + β) is unchanged; so is the default path (the
  surface code now shares its accessibility test with the new module; the
  fingerprints are bit-identical).

**Method** (APBS 3.4.1 operation for operation):
* **Area.** APBS' reference sphere of ⌈4π(r_max + srad + 2 dpos)² sdens⌉
  points (the cell-list radius of an `apolar` calculation is srad + 2 dpos)
  is scaled to every atom's r + srad. The points outside every other atom's
  sphere count; atoms of radius 0 have no area but bury points within srad.
  A_i = 4π(r_i + srad)² n_i / n_ref.
* **Volume.** The trapezoidal rule over APBS' cell-list box (the atoms'
  extent ± 1.42 (r_max + srad + 2 dpos)), with the coordinates accumulated
  in steps of `grid` as APBS accumulates them.
* **Dispersion.** Per atom, a lattice from (int)(x − 14) to (int)(x + 14)
  (truncated towards zero, as C converts) in steps of `grid`. Each
  accessible point within 14 Å adds ρ̄ ε_iw[(σ/r)¹² − 2(σ/r)⁶] for r ≥ σ and
  −ρ̄ ε_iw inside, with σ = R_i + R_w and ε_iw = √(ε_i ε_w).
* pbsolve tests the accessibility only within 14 Å; APBS tests every point
  of the cube, but beyond 14 Å the term is 0 either way. The atoms run in
  parallel, and every sum keeps APBS' order.

**APBS' alkane examples** (grid 0.3 Å, srad 0.65 Å, sdens 100, γ 0.0085
kJ/(mol·Å²), p 0.2394 kJ/(mol·Å³), ρ̄ 0.033428 Å⁻³, AMBER C and H, TIP3P
oxygen; `tests/data/apbs_alkanes`). All 11 alkanes reproduce the SASA, SAV
and WCA energy APBS prints (6 digits) and the total non-polar energy (13
digits, ≤ 2.2e-13 relative).

**APBS 3.4.1, live** (`pbsolve.apbs.apolar_energy`; largest relative
difference of area, volume, dispersion and total):

| case | atoms | time pbsolve / APBS | terms | per atom |
|---|---|---|---|---|
| neopentane, alkane protocol | 17 | 0.025 / 0.38 s | 1.1e-13 (volume identical) | 2.5e-13 |
| synthetic molecule (2 atoms of radius 0), srad 1.4, sdens 10, grid 0.5 | 30 | 0.013 / 0.18 s | 4.4e-14 (volume identical) | 3.8e-13 |
| the same with srad 0, dpos 0, sdens 20, grid 0.35 | 30 | 0.030 / 0.40 s | 1.3e-12 (total, after cancellation) | 3.9e-13 |
| 1YCR, AMBER parameters, alkane protocol, grid 0.5 | 1670 | 0.75 / 12.7 s | 1.2e-13 (volume identical) | 5.0e-13 |

For 1YCR the radii and well depths come from APBS' `vparam-amber-parm94.dat`
by residue and atom name, with HIS, OXT and the N-terminal H2/H3 mapped to
HIE/HID/HIP, O and H (pdb2pqr's AMBER names are not all in that file; APBS
itself would reject this PQR with that file). pbsolve runs on 18 threads,
APBS on one.

**1YCR's terms** (kcal/mol, 0.5 Å grid): area 5957.7 Å², volume 17 507 Å³;
γA = 12.1, pV = 1001.7, dispersion −432.0, total 581.8 (581.6 with grid 0.3).
The volume and dispersion terms are large and cancel; what matters in
practice are differences such as binding.

**Closed forms** (`apolar.rs`):
* one sphere: area 4π(r + srad)² to 1e-12; volume within 0.3 % at 0.1 Å;
* two spheres of radius 2 Å at 3 Å: union volume within 0.2 % at 0.05 Å, each
  area (sphere minus the 0.5 Å cap) within 0.5 % with 200 points per Å²;
* one atom's dispersion equals the radial integral
  ρ̄ ∫ u 4π t² dt from r + srad to 14 Å within 0.3 % at 0.1 Å.

**Lennard-Jones from OpenMM.** For a peptide built by tleap with ff14SB and
read through OpenMM, `lennard_jones_from_system` gives the R_min/2 and ε of
APBS' AMBER parameter file to 1e-9 for all 60 atoms listed there. Atoms with
ε = 0 (serine HG) get radius 0, as in that file; OpenMM may give them a
placeholder σ.

**Mutations.** 10/10 caught:
* the box inflation (1.41) and the box padding;
* edge weights of 1;
* no water radius in σ;
* ε_i ε_w without the square root;
* an atom burying its own surface;
* the reference sphere without 2 dpos;
* floor instead of truncation;
* no constant inside σ;
* γ applied to the area as p.

The 13-digit alkane comparison catches the five the closed forms miss.

**Cost** (1YCR, 18 threads, shared machine): area 0.07 s (sdens 100), volume
0.02 s, dispersion 0.72 s at 0.5 Å and 2.99 s at 0.3 Å (1.08 and 4.73 s
before the accessibility tests were restricted to 14 Å).

## Interaction entropy

**Change.**
* `pbsolve.interaction_entropy(energies, temperature, blocks)`: −TΔS of
  binding from the gas-phase receptor–ligand interaction energy of every
  frame, IE = kT ln⟨exp(β(E_int − ⟨E_int⟩))⟩ [Duan, Liu & Zhang 2016] and the
  second-order cumulant C2 = σ²/(2kT), with σ, both per block of consecutive
  frames, their standard errors and `reliable` = σ ≤ 15 kJ/mol, as
  recommended by [Ekberg & Ryde 2021] (they found IE impossible to converge
  above that σ and recommend C2 where the two differ). The exponential
  average is evaluated as a shifted log-sum-exp.
* `pbsolve.interaction_energy(coords, charges, lj_radii, lj_epsilons,
  ligand_atoms)`: Coulomb (e²/4πε0 = 332.0637133 kcal·Å/mol, CODATA 2018,
  the same constants as OpenMM) and Lennard-Jones ε_ij[(R_ij/r)¹² − 2(R_ij/r)⁶]
  with R_ij = R_i + R_j, ε_ij = √(ε_i ε_j) over every receptor–ligand pair, no
  cutoff.

**Closed forms** (`tests/test_entropy.py`). For energies that take two
values a and b with frequencies p and 1 − p, IE = kT ln(p e^{β(a−m)} + (1 − p)
e^{β(b−m)}) and C2 = p(1 − p)(a − b)²/2kT: three cases agree to 1e-12. IE and
C2 are invariant under a constant shift, positive (Jensen), and C2 scales as
1/kT.

**Normal energies** (exact −TΔS = σ²/2kT for IE and C2; mean ± sd over 20
random samples, kcal/mol, 298.15 K):

| σ | exact | IE, 10³ frames | IE, 10⁴ | IE, 10⁵ | C2, 10³ | C2, 10⁵ |
|---|---|---|---|---|---|---|
| 0.5 | 0.211 | 0.208 ± 0.010 | 0.211 ± 0.003 | 0.211 ± 0.001 | 0.208 ± 0.008 | 0.211 ± 0.001 |
| 1 | 0.844 | 0.824 ± 0.060 | 0.844 ± 0.017 | 0.842 ± 0.005 | 0.833 ± 0.033 | 0.843 ± 0.003 |
| 2 | 3.376 | 3.04 ± 0.38 | 3.34 ± 0.25 | 3.34 ± 0.10 | 3.33 ± 0.13 | 3.372 ± 0.012 |
| 3 | 7.595 | 5.87 ± 0.81 | 6.80 ± 0.67 | 7.05 ± 0.44 | 7.50 ± 0.30 | 7.588 ± 0.028 |
| 4 | 13.50 | 8.86 ± 1.23 | 10.53 ± 1.05 | 11.19 ± 0.81 | 13.33 ± 0.53 | 13.49 ± 0.05 |
| 6 | 30.38 | 14.99 ± 2.02 | 18.19 ± 1.73 | 19.79 ± 1.46 | 29.98 ± 1.18 | 30.35 ± 0.11 |

IE converges from below and needs exponentially many frames as σ grows;
C2 is unbiased up to its 1/N term. This is the behaviour Ekberg & Ryde
describe, and the reason the result carries σ, both estimates and a
`reliable` flag. With σ = 200 kcal/mol (βΔE ~ 10³) IE stays finite and equals
the largest fluctuation minus kT ln N, as it must when one frame dominates.

**Interaction energy against OpenMM 8.6.** tleap/ff14SB peptides
ACE-ALA-ARG-LYS-TRP-NME (92 atoms, +2 e) and ACE-GLU-ASP-PHE-NME (59 atoms,
−2 e), the ligand shifted along y; OpenMM `NoCutoff`, Reference platform,
E_int = E(all) − E(receptor only) − E(ligand only) + E(none), with the
other group's charges and ε switched off (bonded terms and the 1-4
exceptions stay inside the molecules); pbsolve parameters from
`charges_from_system` and `lennard_jones_from_system`:

| shift (Å) | closest pair (Å) | OpenMM (kcal/mol) | pbsolve | Coulomb | LJ | rel. diff |
|---|---|---|---|---|---|---|
| 11 | 1.72 | 5.54656389 | 5.54656389 | −106.72 | 112.26 | 1.7e-12 |
| 13 | 3.10 | −96.03719065 | −96.03719065 | −94.94 | −1.10 | 2.4e-13 |
| 20 | 9.52 | −62.90649119 | −62.90649119 | −62.88 | −0.026 | 5.3e-13 |

Overlapping placements (closest pair 0.16–1.5 Å, E_int up to 3e15 kcal/mol)
agree to 5e-14. A synthetic system with exceptions inside each group is
the unit test; an explicit double loop over the pairs (ligand larger than
one 256-atom chunk, 7000 pairs) agrees to 1e-12.

**Mutations.** 16/16 seeded faults caught (log-sum-exp shift, mean vs sum,
ddof of σ and of the block errors, the 2 in C2 and in the LJ term, the σ
limit and its comparison, the block-size check, combining rules, the chunk
stride, the dielectric, the default receptor).

## Atom dielectrics

**Change.**
* `dielectrics=` (one value per atom) on `Solver.polar_solvation`, the batch
  methods, `potential`, `maps`, `pbsa_solvation_frame(s)`, `OpenMMPBSA` and
  `estimate_error`; core `PbParams::atom_pdie`, `surface::atom_dielectric_faces`.
  The multiple dielectrics of DelPhi [Rocchia, Alexov & Honig 2001] with
  amino-acid specific values [Wang, Zhang, Rocchia & Alexov 2013]: a face
  inside the sharp `mol`/`vdw` surface takes the dielectric of the atom with
  the smallest power distance |x − c|² − R² to its midpoint (ties: lower
  index); faces outside get `sdie`. The reference state keeps the interior
  and has `pdie` outside, without ions, so the faces around every charge are
  the same in both states. Not an APBS option.
* `pbsolve.residue_dielectrics(charges, residues, charged=, neutral=)`: one
  value for residues of net charge ≥ 0.5 e, one for the rest (MM/GBSA fit
  2.7/1.1 [Liu, Peng & Zhang 2019]; no defaults, nothing is fitted for PB);
  `pbsolve solvation`/`mmpbsa --residue-dielectric CHARGED NEUTRAL`.
* `SolvationTerms.reference`, `PBSAFrame.delta_reference`: the reference
  state's energy and the receptor–ligand interaction in it (complex −
  receptor − ligand on the complex's grid).
* `two_sphere_energy(sphere_pdie=(ε_0, ε_1))`: the reference solution with
  a dielectric per sphere; with equal values it matches the old code path to
  4e-16, and two neutral spheres far apart approach two Kirkwood spheres as
  1/d³ (8e-5 at 50 Å, 4e-7 at 300 Å).

**Default path.** Equal values reproduce `pdie` bit for bit. This holds for the
solvated and reference maps of `mol` and `vdw` and for `lpbe` and `npbe` energies
(Rust), and for `polar_solvation`, `potential`, `maps`, the batches,
`pbsa_solvation_frame` and `estimate_error` (Python). The default path does not
touch the new code, and the fingerprint of the default path did not change.

**Ownership.** Every interior face of a six-atom cluster with a radius-0 atom
(both states, all three axes) equals the brute-force argmin of the power
distance; on the test molecule of the Python suite, every 11th interior face
of both states.

**One sphere** (radius-0 charges inside a neutral atom of radius a with
interior ε_1). A central charge sees only the outer boundary:
ΔG = −(q²/2a)(1/`pdie` − 1/`sdie`) for every ε_1. At 0.25 Å (`vdw`, a = 2 Å,
exact −81.96 kcal/mol) the errors are:

| ε_1 | 1 (= plain, bit for bit) | 4 | 20 |
|---|---|---|---|
| error | −1.50 % | −0.14 % | +0.64 % |

Off-centre charges (the Kirkwood sphere of *Analytic references*, a = 5 Å)
against K(ε_1 | `sdie`, ions) − K(ε_1 | `pdie`):

| ε_1, `pdie`, salt | 0.5 Å | 0.25 Å |
|---|---|---|
| 4, 1, none | −1.54 % | −0.92 % |
| 4, 2, 0.15 M | −1.93 % | −0.87 % |
| 20, 1, none | −0.15 % | −0.22 % |

**Two spheres** with interiors 4 and 10 (radii 5 and 4 Å, centres 15.8 Å
apart, five charges) against `two_sphere_energy(sphere_pdie=(4, 10))`:

| `pdie`, salt | exact (kcal/mol) | 0.5 Å | 0.25 Å | one `pdie` for both: 0.5 / 0.25 Å |
|---|---|---|---|---|
| 1, none | −25.706 | −0.83 % | −0.32 % | −3.69 / −1.68 % |
| 2, 0.15 M | −17.366 | −1.56 % | −0.74 % | −3.44 / −1.58 % |

**1YCR** (MDM2–p53, AMBER charges and radii, 0.15 M, chain B as ligand;
`residue_dielectrics` puts 468 atoms of 25 charged residues at the charged
value). On CUDA, kcal/mol:

| interior | spacing | ΔG_polar complex | ΔE_ref (`delta_reference`) | ΔΔG_polar | sum |
|---|---|---|---|---|---|
| `pdie` 1 | 0.5 / 0.35 / 0.25 | −1231.3 / −1205.9 / −1190.7 | −361.4 / −361.0 / −361.0 | 423.7 / 421.8 / 419.2 | 62.3 / 60.8 / 58.2 |
| `pdie` 4 | 0.5 / 0.35 / 0.25 | −286.8 / −282.0 / −279.0 | −90.4 / −90.3 / −90.3 | 102.4 / 102.0 / 101.5 | 12.1 / 11.8 / 11.3 |
| charged 4, other 1 | 0.5 / 0.35 / 0.25 | −937.6 / −923.1 / −916.2 | −386.0 / −382.9 / −383.0 | 421.5 / 420.7 / 418.5 | 35.5 / 37.7 / 35.5 |
| charged 2.7, other 1.1 | 0.5 / 0.35 / 0.25 | −996.1 / −981.1 / −973.2 | −378.6 / −376.4 / −376.4 | 417.7 / 416.6 / 414.5 | 39.1 / 40.2 / 38.1 |

With one `pdie`, ΔE_ref equals the analytic Coulomb interaction over `pdie`
(−360.92 kcal/mol in vacuum) to −0.14 % at 0.5 Å and −0.03 % at 0.35 and
0.25 Å. The grid self-energies cancel on the shared grid. With
`coulomb="analytic"` the match is exact (1e-12, test). The residue dielectrics
hardly change ΔΔG_polar, because the reference medium outside the protein is
still vacuum. They change the interaction in the reference state (charged
residues of ε 4 surrounded by vacuum) and the sum. The sum equals the
solvated states' difference (test, 1e-12) and does not depend on the
reference. With atom dielectrics, `delta_reference` therefore takes the place
of the force field's Coulomb term. CPU and CUDA agree to every digit for the
complex and to 8e-14 for the binding terms.

**Mutations.** 30/30 seeded faults caught:
* Rust (13): the power distance without R² or with +R², the comparison, the
  midpoint shift, the search radius, the outside value of either state, the
  probes, the length check, validation of values and of surface, interface
  and `coulomb`, and the batch job's dielectrics.
* Python (12): the per-species subsets, the residue threshold and net charge,
  the kT of the reference energy, `delta_reference`, the forwarding in
  `pbsa_solvation_frames`, `estimate_error` and the CLI, the CHARGED/NEUTRAL
  order, and the reference and per-sphere values of `two_sphere_energy`.
* Bindings (5): `polar_solvation`, `polar_solvation_many`, `maps`,
  `potential` and `polar_solvation_batch` dropping the values.

**Cost** (1YCR, 0.5 Å, 0.15 M; shared machine, GPU at 83 % with GROMACS,
medians of 5 alternating runs). CPU (18 threads): 1.79 → 1.89 s per
calculation. CUDA: 0.24 → 0.53 s. The maps are built on the host (map stage
0.03 → 0.30 s), and the reference state rebuilds the molecular surface on
every level (about 0.06 s each; a shortcut that sharing it with the solvated
state would remove). The ownership search takes about 0.01 s per map.

## Size-modified PB (`equation="smpbe"`)

**Change.** `Solver(equation="smpbe", ion_size=a)` and `--equation smpbe
--ion-size a`: the nonlinear equation for ions of finite size.
* **Model.** Every ion fills one cube of edge a (lattice gas [Borukhov et al.
  1997, 2000]), all species the same size. In APBS
  terms this is `smpbe vol a size 1`. APBS 3.4.1 refuses `smpbe` ("Sorry,
  due to numerical stability issues SMPBE is currently disabled"), so there
  is no APBS reference; `pbsolve.apbs` refuses it too.
* **Defaults unchanged.** `lpbe` and `npbe` are unchanged: the default path
  is bit-identical, and `npbe` is bit-identical to the previous commit on
  the CPU (energies and iterations).
* **Validation.** `ion_size` must be > 0 for `smpbe` and is refused
  otherwise. The ions must leave room in the bulk, Σ c_j N_A a³ < 1.

**Method.** With φ_j = c_j N_A a³ the bulk volume fraction of species j and
e_j = e^{−q_j u}, the local concentrations are c_j e_j / D with the lattice
denominator

D(u) = 1 − Σ φ_j + Σ φ_j e_j,

so no species exceeds one ion per cell. The ion term, its slope and the
energy density follow from the lattice-gas free energy:
* N(u) = −κ̄² Σ_j a_j q_j e_j / D (a_j = c_j/2I);
* N'(u) = κ̄² (S2/D − S1 P1/D²), with S1 = Σ a q e, S2 = Σ a q² e,
  P1 = Σ φ q e;
* energy density ln D / (N_A a³) in place of Σ_j c_j (e_j − 1) (kT per
  volume).

These are the k = 1 branch of APBS' `Vc_vecsmpbe` and
`Vpmg_qmEnergySMPBE` [Chu et al. 2007], for any number of species. For
a = 0, D = 1 exactly, and the code is the `npbe` path operation for
operation, so `npbe` keeps its results bit for bit. The Newton iteration,
its device kernels, the outer terms of the coarser levels and the memory
are those of `npbe`.

**Planar double layer** (exact reference). For a 1:1 salt the planar
equation has the first integral u'² = (2κ²/φ0) ln(1 − φ0 + φ0 cosh u)
[Borukhov et al. 1997; Kilic et al. 2007]. For φ0 → 0 this is
Gouy–Chapman, 4κ² sinh²(u/2). Integrated by quadrature, it gives the exact
profile from u(0) = 6 kT/e at 0.15 M. pbsolve solves a slab whose boundary
carries that profile, with `solve_maps`:

| h (Å) | npbe (point ions) | smpbe, a = 3 Å | smpbe, a = 7 Å |
|---|---|---|---|
| 0.5 | 1.1e-2 | 3.4e-3 | 5.5e-5 |
| 0.25 | 1.8e-3 | 3.7e-4 | 5.0e-6 |
| 0.125 | 2.0e-4 | 3.0e-5 | 3.6e-7 |
| observed order | 2.6, 3.1 | 3.2, 3.6 | 3.4, 3.8 |

(maximum error of u in kT/e). These are the first analytic checks of the
nonlinear operator itself, for `npbe` too, independent of APBS. With 7 Å
ions D reaches 13 at the surface: the counterions are crowded, and the
profile is smoother than Gouy–Chapman's.

**A charged cavity** [Silalahi et al. 2010]: a sphere of radius 20 Å, pdie
4, sdie 78.5, central charge −50 e, 0.1 M 1:1 salt, no Stern layer
(`ion_radius=0`, `surface="vdw"`), 0.5 Å, CUDA. ΔG_polar: lpbe −5115.564,
npbe −5131.975 kcal/mol.

| a (Å) | smpbe − npbe (kcal/mol) | log slope against a | largest counterion concentration (M) | 1/(N_A a³) (M) |
|---|---|---|---|---|
| 0 (npbe) | — | — | 5.26 | — |
| 0.5 | +0.0013 | — | 5.26 | 13 284 |
| 1 | +0.0101 | 3.00 | 5.26 | 1 661 |
| 2 | +0.0811 | 3.00 | 5.19 | 208 |
| 4 | +0.6485 | 3.00 | 4.70 | 25.9 |
| 8 | +5.0908 | 2.97 | 2.48 | 3.24 |

* **Entropic cost.** Finite size always costs free energy (placing ions
  costs entropy). While the ions are dilute (ξ ≪ 1), the cost grows exactly
  as a³: this is Silalahi et al.'s eq. 11, log(F_SMPBE − F_NLPBE) =
  3 log a + const.
* **Saturation.** At 8 Å the counterions approach one per cell. At a
  smaller sphere (radius 10 Å, −50 e, 0.6 Å) point ions reach 84 M at the
  surface, an impossible concentration. With 8 Å ions the counterions stop
  at 3.2432 M = 1/(N_A a³): the saturation layer of Borukhov et al.

**Checks:**
* Rust (`nonlinear.rs`, 7 tests, 2 new or extended):
  * the functional is stationary at the solution for `smpbe` too (ratio
    4.00 ± 0.05);
  * no species exceeds one ion per cell for u from −40 to 40;
  * a = 0.01 Å gives the `npbe` energy to 1e-9;
  * the cost of finite size is positive and grows 8.0-fold from 1 to 2 Å.
* GPU: `nonlinear_pb_matches_the_cpu` adds an `smpbe` case (≤ 1e-10).
* `tests/test_size_modified.py` (7 tests, 2 s):
  * the planar double layer for `npbe` and `smpbe` with 3 and 7 Å ions
    (second order or better);
  * the a³ cost on a −20 e cavity;
  * saturation at a −50 e cavity;
  * parameter checks and the APBS refusal;
  * the command line.
* Mutations caught: 9/9:
  * D's Boltzmann factor and its 1 − Σφ;
  * the slope without dD/du, the residual without D;
  * the energy density's factor;
  * a² instead of a³, half the volume fraction;
  * two of them in the CUDA kernels.
* Default path: the fingerprint set is bit-identical on CPU and CUDA.

**Cost:** the same as `npbe` (1YCR 0.5 Å on CUDA: 0.33 s for smpbe with 6 Å
ions, 0.34 s for npbe, 0.19 s for lpbe). For 5DK3 `smpbe` is cheaper
(6.35 s, against 7.25 s for npbe): the crowded ions soften the
nonlinearity, and Newton needs fewer steps. On 1YCR 6 Å ions raise
ΔG_polar by 0.007 kcal/mol; on 5DK3 by 2.6 kcal/mol.

**Notes.**
* **Size of an ion.** There is no single right value. Chu et al. (2007)
  fitted sizes to ion-counting data on DNA; Silalahi et al. used a = 2–16 Å.
* **Stern layer.** pbsolve keeps its Stern layer (`ion_radius`) with
  `smpbe`. Silalahi et al. argue that combining both counts the excluded
  volume twice; set `ion_radius=0` to follow them.
* **Not done.** Species of different sizes (APBS' `size` ≠ 1 and the
  non-uniform model of Boschitsch & Danilov 2012) and ion correlations.

Reproduce: `pytest tests/test_size_modified.py`; the cavity table with
`Solver(equation="smpbe", ion_size=a, pdie=4.0, sdie=78.5,
ionic_strength=0.1, ion_radius=0.0, surface="vdw")`.

## Sensitivity of the sharp dielectric boundary

With `surface="mol"` (APBS `srfm mol`) a coordinate change of 1e-4 Å can flip
a few boundary faces next to a charge: on the 221-atom peptide, rounding the
coordinates to 4 decimals flipped 3 of 2.7 million faces and moved ΔG_polar by
0.4 kcal/mol (0.1 %). APBS behaves identically (same maps). Average over MD
frames, or use the smoothed surfaces planned in the roadmap, when such jumps matter.

