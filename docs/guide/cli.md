# Command line

The `pbsolve` command covers the common jobs without writing Python.
`pbsolve <command> --help` lists the options of a command; this page explains
them. Settings without an option use the {py:class}`pbsolve.Solver`
defaults.

| Command | Purpose |
|---|---|
| [`pbsolve info`](#pbsolve-info) | build and platform information |
| [`pbsolve test`](#pbsolve-test) | installation check |
| [`pbsolve solvation`](#pbsolve-solvation) | polar and non-polar solvation energy of one structure |
| [`pbsolve convert`](#pbsolve-convert) | write a PQR file from any supported input |
| [`pbsolve mmpbsa`](#pbsolve-mmpbsa) | MM-PBSA solvation terms along a trajectory |
| [`pbsolve prep`](#pbsolve-prep) | prepare a PDB file (hydrogens, protonation, charges, radii) |
| [`pbsolve apbs compare`](#pbsolve-apbs-compare) | run pbsolve and APBS on the same structure and compare |
| [`pbsolve apbs run`](#pbsolve-apbs-run) | evaluate an existing APBS input file with pbsolve (and APBS) |

## Input options

Commands that read a structure take these options.

| Option | Meaning |
|---|---|
| `--coords` | coordinate file for a topology input (inpcrd, rst7, gro, pdb, ..., or a trajectory: its first frame) |
| `--format` | format of the input if its extension is ambiguous |
| `--radii` | `file` (default: the radii in a PQR or Amber topology), or a scheme: `bondi`, `mbondi`, `mbondi2`, `parse` ([radii](files.md#radii)) |
| `--keep-solvent` | keep water and monovalent ions (removed by default) |
| `-I`, `--include` | GROMACS include directory (repeatable) |
| `--ff` | prepare a PDB or mmCIF file without charges first, with this force field: `amber`, `parse` or `charmm` (as [`pbsolve prep`](#pbsolve-prep)) |

## Solver options

Commands that solve take these options; they are the
[`Solver` arguments](model.md) of the same name.

| Option | Default | Meaning |
|---|---|---|
| `--platform` | `auto` | `auto`, `CPU` or `CUDA` |
| `--spacing` | 0.5 | largest fine-grid spacing (Å) |
| `--pdie` | 1.0 | solute dielectric constant |
| `--sdie` | 78.54 | solvent dielectric constant |
| `--temperature` | 298.15 | temperature (K) |
| `--ionic-strength` | 0.0 | 1:1 salt (mol/L) |
| `--ion-radius` | 2.0 | ion radius (Å) |
| `--srad` | 1.4 | solvent probe radius (Å) |
| `--surface` | `mol` | `mol`, `vdw` or `harmonic` |
| `--interface` | `sharp` | dielectric of the faces the surface cuts: `sharp` (as APBS) or `fractional` (edge-weighted harmonic average, see [the model](model.md#dielectric-boundary)) |
| `--equation` | `lpbe` | Poisson–Boltzmann equation of the solvated state: `lpbe` (linearized), `npbe` (nonlinear, as APBS) or `smpbe` (nonlinear with ions of finite size), see [the model](model.md#nonlinear-equation) |
| `--ion-size` | 0 | with `--equation smpbe`: edge of the lattice cell one ion fills (Å) |
| `--coulomb` | `grid` | how ΔG_polar removes the Coulomb potential: `grid` (reference state on the same grid, as APBS) or `analytic` (reaction-field decomposition, one solve instead of two, see [the model](model.md#solvation-energy)) |
| `--bcfl` | `mdh` | `mdh`, `mdh-fast`, `sdh` or `zero` |
| `--dime-policy` | `pbsolve` | grid node counts: `pbsolve` (8k+1) or `apbs` (values APBS keeps) |
| `--preconditioner` | `multigrid-f32` | `multigrid-f32`, `multigrid` or `jacobi` |
| `--chgm` | `spl0` | charge assignment: `spl0` (trilinear), `spl2` or `spl4` (B-splines of order 3 and 5), as APBS `chgm` |

## `pbsolve info`

Prints the version, the usable platforms and the CUDA status: whether
support is compiled in and usable (and if not, why), the number of GPUs, the
driver and NVRTC versions and the NVRTC library. No options.

## `pbsolve test`

Solves a Born ion on every available platform and compares the energy with
the analytic value and between platforms.

| Option | Meaning |
|---|---|
| `--platform` | test only this platform (repeatable) |
| `--spacing` | grid spacing (Å, default 0.25) |

## `pbsolve solvation`

```text
pbsolve solvation INPUT [options]
```

Prints ΔG_polar, the non-polar term (γ·SASA + β, Sitkoff constants), the
grid and the solver statistics. Takes the [input options](#input-options)
and the [solver options](#solver-options), and:

| Option | Meaning |
|---|---|
| `--dx` | also write the solvated potential to this OpenDX file |
| `--apbs-grid` | print the equivalent APBS grid keywords (implies `--dime-policy apbs`) |
| `--placements` | also solve on this many grid placements (at least 2) and report the mean, sd and sem of ΔG_polar ([error estimates](solver.md#error-estimates)) |
| `--extrapolate` | with `--placements`: repeat at this finer spacing and extrapolate the mean to zero spacing |
| `--json` | print the results as JSON |

```bash
pbsolve solvation tests/data/1ycr_amber_opt_pdb2pqr.pqr --ionic-strength 0.15 --json
```

## `pbsolve convert`

```text
pbsolve convert INPUT OUTPUT.pqr [options]
```

Writes a PQR file with charges and radii from any input pbsolve reads. Takes
the [input options](#input-options), and:

| Option | Meaning |
|---|---|
| `--select` | atoms to keep, a [selection](files.md#simulation-files) such as `"not resname LIG"` |
| `--frame` | model or frame of a multi-frame input or `--coords` file (default 0) |

## `pbsolve mmpbsa`

```text
pbsolve mmpbsa TOPOLOGY TRAJECTORY --ligand SELECTION [options]
```

MM-PBSA solvation terms of complex, receptor and ligand for every frame
([Trajectories and MM-PBSA](many.md)). The topology must carry charges (prmtop,
psf, top, pqr, mol2, ...); the trajectory can be DCD, XTC, TRR, NetCDF, mdcrd
or a multi-model PDB/GRO file. Takes the [solver options](#solver-options),
and:

| Option | Meaning |
|---|---|
| `--ligand` | ligand selection, such as `"resname LIG"` or `"chain B"` (required) |
| `--receptor` | receptor selection (default: all other solute atoms) |
| `--format` | format of the topology if its extension is ambiguous |
| `--radii` | as in the [input options](#input-options) |
| `--keep-solvent` | keep water and monovalent ions |
| `-I`, `--include` | GROMACS include directory (repeatable) |
| `--start`, `--stop`, `--step` | frames to use (0-based, as a Python slice) |
| `--csv` | write the per-frame terms to this CSV file |
| `--workers` | calculations solved at once (default 2 on the CPU, 3 on CUDA, fewer if the GPU memory is short); results do not depend on it |

## `pbsolve prep`

```text
pbsolve prep INPUT.pdb OUTPUT.pqr [options]
```

Adds hydrogens, assigns protonation states, optimises the hydrogen-bond
network and assigns charges and radii, like pdb2pqr
([preparing a PDB file](files.md#preparing-a-pdb-file)).

| Option | Meaning |
|---|---|
| `--ff` | force field: `amber` (default), `parse` or `charmm` |
| `--ph` | apply model-compound pKa values at this pH |
| `--titration` | `none` (default) or `propka` (PROPKA 3 pKa values; optional package) |
| `--ligand` | mol2 file with partial charges to append |
| `--ligand-radii` | radius scheme of the ligand: `bondi`, `mbondi`, `mbondi2` (default) or `parse` |
| `--noopt` | skip the hydrogen-bond optimisation (`pdb2pqr --noopt`) |
| `--quiet` | do not print the report |

## `pbsolve apbs compare`

```text
pbsolve apbs compare INPUT [options]
```

Runs pbsolve and APBS with identical settings and compares grids, maps and
energies ([Comparing with APBS](apbs.md)). The input is a PQR file, or any file
pbsolve reads (converted to PQR first).

| Option | Meaning |
|---|---|
| `--coords`, `--radii`, `--keep-solvent`, `-I`, `--include`, `--ff` | as in the [input options](#input-options) |
| `--platform`, `--spacing`, `--pdie`, `--sdie`, `--temperature`, `--ionic-strength`, `--ion-radius`, `--srad` | as in the [solver options](#solver-options) |
| `--sdens` | surface point density (points per Å², default 10) |
| `--surface` | `mol` (default) or `vdw`; `harmonic` has no APBS equivalent |
| `--bcfl` | `mdh` (default), `mdh-fast` (pbsolve's interpolated `mdh`; APBS runs `mdh`), `sdh` or `zero` |
| `--chgm` | charge assignment, `spl0` (default), `spl2` or `spl4`, written as APBS `chgm` |
| `--equation` | `lpbe` (default) or `npbe`, written as the APBS keyword |
| `--cfac`, `--fadd` | automatic grid sizing ([grids](grids.md)) |
| `--tol` | pbsolve's CG tolerance, also written as APBS `etol` |
| `--preconditioner` | pbsolve's preconditioner |
| `--ligand-chain` | also compare receptor, ligand and ΔΔG of binding; the ligand is this PQR chain |
| `--ligand-atoms` | the ligand as 1-based atom ranges instead, such as `1450-1670` |
| `--workdir` | where to run APBS and keep its files (default `apbs_compare`) |
| `--no-maps` | skip the map comparison and the map injection (faster) |
| `--apbs` | APBS executable (default: `$APBS`, or `apbs` on the `PATH`) |
| `--json` | also write the report to this JSON file |
| `--energy-rtol` | relative energy tolerance of the verdict (default 1e-3) |

The grid always uses node counts APBS keeps (`dime_policy="apbs"`).

## `pbsolve apbs run`

```text
pbsolve apbs run INPUT.in [options]
```

Evaluates an existing APBS input file (`mg-auto` and `mg-manual`
calculations and `print` statements) with pbsolve and, unless `--no-apbs`, with APBS, side by side.

| Option | Meaning |
|---|---|
| `--platform` | pbsolve's platform |
| `--no-apbs` | only run pbsolve |
| `--apbs` | APBS executable |
| `--workdir` | where to run APBS (default: the input's directory) |
| `--preconditioner` | pbsolve's preconditioner |
| `--json` | also write the report to this JSON file |
| `--energy-rtol` | relative energy tolerance of the verdict (default 1e-3) |
