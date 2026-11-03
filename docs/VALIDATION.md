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

## Status (v0.0.1)

| Check | Result |
|---|---|
| Born ion, `mol`, h = 0.500 / 0.333 / 0.250 / 0.200 Å | −4.73 % / −1.08 % / −1.27 % / −0.94 % vs analytic (−81.96 kcal/mol); converging overall, not monotonically (before the APBS-exact surface: −4.73 / −1.09 / −1.23 / −0.71 %) |
| Born ion, `harmonic`, same spacings | −4.16 % / −1.08 % / −1.65 % / −1.09 % (unchanged) |
| Operator symmetry, charge conservation, SES crevice, SASA, APBS reference-sphere count, marking = brute force | pass (unit tests) |
| APBS 3.4.1 parity, synthetic 30-atom molecule (crevices, zero-radius atoms), `mdh`/`sdh`/`zero`, `mol`/`vdw`, 0–0.15 M, 2- and 3-level focusing, Born ion | all maps identical; energies within 1e-7 (solvated) / 5e-6 (reference, `mdh`/`sdh`) in APBS units — `tests/test_apbs_parity.py` |
| APBS 3.4.1 parity, MDM2–p53 peptide complex (PDB 1YCR, pdb2pqr 3 `--ff=AMBER`, 1670 atoms, 18 zero-radius H; ligand = chain B) | see table below: 0 / 2 146 689 differing nodes in every map; ΔG_polar +0.0012 % raw, ≤ 1.2e-6 in APBS units; ΔΔG_bind within 0.004 kcal/mol |
| CUDA backend | **first GPU run** (RTX 2080 SUPER, cc 7.5, driver 595.99 = CUDA 13.2, `--features cuda-13`, conda NVRTC 13.2): `pbsolve test` and `tests/test_platforms.py` pass, CPU vs CUDA 9.8e-15; APBS comparison on CUDA identical to CPU. **Open issue:** by default cudarc loaded the system NVRTC 13.3 (`/usr/local/cuda`) instead of the preloaded conda one, whose PTX the 13.2 driver rejects (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`); verified with `LD_LIBRARY_PATH` pointing `libnvrtc.so` at the conda library. CUDA 12 not run |
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

