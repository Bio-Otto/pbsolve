# Input files and structure preparation

A PB calculation needs three numbers per atom: its position (Å), its partial
charge (e) and its radius (Å). pbsolve reads them from PQR files, from
simulation topologies, or prepares them from a PDB file.

## PQR files

{py:func}`pbsolve.read_pqr` returns the three arrays:

```python
import numpy as np
import pbsolve

coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
print(coords.shape, charges.sum(), radii.max())
```

## Simulation files

{py:func}`pbsolve.io.load` reads structures and topologies into a
{py:class}`~pbsolve.io.Molecule` with atom names, residues, chains, charges,
radii and coordinates; a second file can supply the coordinates:

| Kind | Formats |
|---|---|
| structures | PDB (multi-model), PQR, mmCIF, GRO, mol2 |
| topologies with charges | Amber `prmtop`, CHARMM/NAMD `psf`, GROMACS `top`/`itp` |
| coordinates | Amber `inpcrd`/`rst7`/`ncrst` |
| trajectories | DCD, XTC, TRR, Amber NetCDF and mdcrd, multi-model PDB/GRO/mmCIF |

The format comes from the file extension (`format=` overrides it). Readers
convert to Å and e, and fail with an error on malformed input; they never
invent charges or radii.

```{code-block} python
# Amber: charges and radii from the topology
mol = pbsolve.io.load("complex.prmtop", "complex.inpcrd")
# GROMACS: charges from the topology, radii by scheme
mol = pbsolve.io.load("topol.top", "md.gro", radii="mbondi2")
```

Molecules can be filtered with selections and written as PQR:

```python
mol = pbsolve.io.load("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
print(mol)
peptide = mol.subset("chain B")              # a new Molecule
mask = mol.select("protein and not hydrogen")  # a boolean mask
peptide.write_pqr("peptide.pqr")
```

Selections combine `all`, `none`, `protein`, `water`, `ions`, `hydrogen`,
`hetero`, `resname`, `name`, `element`, `chain`, `segid`, `resid` (ranges
like `1-10`), `index` (0-based) and `serial` (1-based) with `and`, `or`, `not`
and parentheses; names accept `*` and `?` wildcards.
`mol.strip_solvent()` removes water and monovalent ions (PB treats the
solvent as a continuum) and keeps divalent and structural ions.
`mol.require()` raises a clear error if charges, radii or coordinates are
missing. {py:func}`pbsolve.io.frames` iterates over the frames of a
trajectory ([Trajectories and MM-PBSA](many.md)).

## Radii

PB radii define the dielectric boundary, and energies depend on them. Amber
topologies and PQR files carry radii; GROMACS and CHARMM files do not, so
assign a scheme by element, as Amber's `set default PBRadii` does:

| Scheme | Radii |
|---|---|
| `bondi` | Bondi (1964) van der Waals radii |
| `mbondi` | Bondi; H bonded to C or N 1.3 Å, H bonded to O or S 0.8 Å |
| `mbondi2` | Bondi; H bonded to N 1.3 Å (Onufriev, Bashford and Case 2004) |
| `parse` | PARSE radii (Sitkoff, Sharp and Honig 1994) for H, C, N, O, S |

Pass `radii=` to `load`, or call `mol.with_radii("mbondi2")`.

## Preparing a PDB file

A PDB file has no hydrogens, charges or radii. {py:func}`pbsolve.prep.prepare`
adds hydrogens, rebuilds missing heavy atoms, assigns protonation states,
optimises the hydrogen-bond network and assigns AMBER, PARSE or CHARMM
charges and radii. Its output is identical to that of `pdb2pqr --drop-water
--keep-chain --whitespace` (checked on 1YCR, 1TII, 1HPV and 5DK3). Waters
are dropped.

```python
prepared, report = pbsolve.prep.prepare("tests/data/1ycr.pdb", "amber")
print(report.summary())
prepared.write_pqr("1ycr.pqr")
```

`ph=` applies model-compound pKa values at that pH, and
`titration="propka"` uses PROPKA 3 pKa values (the optional `propka`
package). `ligand=` appends a ligand from a mol2 file with partial charges.
`optimize=False` skips the hydrogen-bond optimisation (`pdb2pqr --noopt`).

## Potential maps

{py:meth}`~pbsolve.Solver.potential` returns the potential (kT/e) on the
finest grid. {py:func}`pbsolve.write_dx` writes it in the OpenDX format that
PyMOL, VMD and Chimera read, and {py:func}`pbsolve.read_dx` reads OpenDX files
from pbsolve or APBS:

```python
solver = pbsolve.Solver(ionic_strength=0.15)
potential, origin, spacing, energy_kt = solver.potential(coords, charges, radii)
pbsolve.write_dx("potential.dx", potential, origin, spacing)
data, origin, spacing = pbsolve.read_dx("potential.dx")
```

## From the command line

`pbsolve convert` writes a PQR from any input `pbsolve` reads (with
`--select`, `--radii`, and `--ff` to prepare a PDB first); `pbsolve prep`
prepares a PDB file:

```bash
pbsolve convert complex.prmtop complex.pqr --coords complex.inpcrd
pbsolve prep 1ycr.pdb 1ycr.pqr --ff amber --ph 7.0
```
