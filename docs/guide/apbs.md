# Comparing with APBS

With `surface="mol"` and `bcfl` `"mdh"`, `"sdh"` or `"zero"`, pbsolve follows
APBS 3.4.1 (`mg-auto`, `lpbe`, `srfm mol`, `chgm spl0`) operation for
operation: its focusing grids, dielectric and ion maps are identical to
APBS', and its energies agree to a few parts per million. `pbsolve.apbs`
runs both programs on the same problem and shows where they differ, if they
do. It needs APBS 3.4.1 on the `PATH`
(`conda install -c conda-forge apbs=3.4.1`).

## Comparing a structure

```bash
pbsolve apbs compare complex.pqr --spacing 0.5 --ionic-strength 0.15 --ligand-chain B
```

pbsolve writes an APBS input with its own settings, runs both programs and
compares, in layers, so that a disagreement can be located:

1. **grids**: the dimensions, spacings and centres APBS reports for every
   focusing level must equal pbsolve's;
2. **maps**: APBS' dielectric (three face grids), ion-accessibility and
   charge maps against pbsolve's, node for node;
3. **injection**: pbsolve solves APBS' own maps and boundary values, which
   tests the discretisation and the linear solver alone;
4. **energies**: solvated, reference and ΔG_polar; with `--ligand-chain` or
   `--ligand-atoms` also receptor, ligand and ΔΔG_polar of binding on the
   complex's grid, as in MM-PBSA.

The verdict passes with identical grids, matching maps, the injected solution
within 1e-4 of the potential scale and ΔG_polar within `--energy-rtol`
(default 1e-3 relative); a binding comparison also requires ΔΔG_polar within
0.05 kcal/mol in APBS units (see [below](#what-can-differ-and-why)).
[The command line](cli.md#pbsolve-apbs-compare) lists all options.

The same from Python, with {py:func}`pbsolve.apbs.compare` and
{py:func}`pbsolve.apbs.compare_binding`:

```{code-block} python
import numpy as np
import pbsolve
from pbsolve import apbs

solver = pbsolve.Solver(ionic_strength=0.15, dime_policy="apbs")
report = apbs.compare("complex.pqr", solver, workdir="apbs_compare")
print(report.format(), report.passed())

ligand = np.nonzero(pbsolve.io.load("complex.pqr").select("chain B"))[0]
binding = apbs.compare_binding("complex.pqr", solver, ligand)
print(binding.format())
```

## Replaying an APBS input

```bash
pbsolve apbs run apbs.in
```

evaluates the `mg-auto` and `mg-manual` calculations and `print` statements
of an existing APBS input with pbsolve, runs APBS on the same file (unless
`--no-apbs`) and prints the energies side by side. Inputs with features
pbsolve lacks (nonlinear PB, external maps, `mg-para`) are refused with a
message naming the keyword.

## What can differ, and why

* **Grids.** APBS silently replaces a `dime` above 65 that is not `32c+1` by a
  smaller value. Build grids with `Solver(dime_policy="apbs")` (`pbsolve apbs`
  does this itself); pbsolve refuses to compare when APBS ran different
  grids. See [grids](grids.md#node-counts-and-apbs).
* **Physical constants.** APBS scales the source term with CGS constants and
  converts to kJ/mol with CODATA-1986 values, while pbsolve uses CODATA 2018.
  All energies therefore differ by a factor
  {py:func}`pbsolve.apbs.apbs_unit_factor` ≈ 1 + 1.2e-5; reports also give
  the differences in "APBS units", with the factor removed. APBS' analytic
  boundary uses a third set of constants, which leaves up to a few 1e-7 in
  unscreened reference states. ΔG_polar is a small difference of two such
  energies, so this becomes a few 1e-6 there and, for large complexes,
  a few 0.01 kcal/mol in ΔΔG_polar. With `bcfl zero` the energies agree to
  ~1e-10.
* **Options without an APBS equivalent.** `bcfl="mdh-fast"` is written as
  `mdh` (energies agree to ~1e-13); `surface="harmonic"` cannot be compared.

## Results

MDM2 with the p53 peptide (PDB 1YCR, pdb2pqr `--ff=AMBER`, 1670 atoms, 18 of
them hydrogens with radius 0; dime 129³, two focusing levels): every map is
identical (0 of 2 146 689 nodes differ), ΔG_polar agrees to 1.2e-6 in APBS
units, and ΔΔG_polar of binding to 0.004 kcal/mol. A synthetic 30-atom
molecule with crevices and zero-radius atoms, run with all three boundary
conditions, both surfaces, 0 to 0.15 M salt and two or three focusing levels,
gives identical maps as well (`tests/test_apbs_parity.py`).

Three larger proteins give the same picture:
* **1HPV**, the HIV-1 protease dimer (3 137 atoms);
* **1TII**, an AB5 toxin (10 862 atoms, the A subunit leaving the B pentamer);
* **5DK3**, an IgG4 antibody (20 179 atoms, one light chain leaving).

The grids go up to 193×321×321 nodes. The dielectric and ion maps of 1HPV
and 1TII are identical. ΔG_polar agrees to ≤ 4e-6 in APBS units, and
ΔΔG_polar of binding to ≤ 0.042 kcal/mol for binding energies of up to
6 232 kcal/mol. With `bcfl zero` the energies agree to ~1e-10 and ΔΔG_polar to
< 1e-6 kcal/mol. The small remainder with `mdh` and `sdh` therefore comes
from APBS' boundary constants. The [validation record](../VALIDATION) has the
full tables.
