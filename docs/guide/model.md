# Physical model

pbsolve solves the linearized Poisson–Boltzmann equation (LPBE) the way APBS
does, in APBS' dimensionless form:

$$
-\nabla\cdot\big(\varepsilon(\mathbf r)\,\nabla u(\mathbf r)\big)
+ \bar\kappa^2(\mathbf r)\,u(\mathbf r)
= 4\pi\, l_B \sum_i z_i\,\delta(\mathbf r - \mathbf r_i)
$$

with the potential $u$ in kT/e, lengths in Å, the vacuum Bjerrum length
$l_B = e^2/(4\pi\varepsilon_0 k_B T)$ ({py:func}`pbsolve.bjerrum_length`)
and $\bar\kappa^2 = \varepsilon_s\kappa^2$, where $\kappa$ is the inverse
Debye length of the mobile ions. All settings are keyword arguments of
{py:class}`pbsolve.Solver`; their names follow the APBS keywords.

```python
import pbsolve

# positions (Å), charges (e) and radii (Å)
coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")

solver = pbsolve.Solver(pdie=2.0, sdie=78.54, ionic_strength=0.15, temperature=298.15)
result = solver.polar_solvation(coords, charges, radii)
print(f"ΔG_polar = {result.energy:.2f} kcal/mol")
```

## Solvation energy

{py:meth}`~pbsolve.Solver.polar_solvation` solves two states on the same
grid:

* the **solvated** state: dielectric `pdie` inside the solute, `sdie`
  outside, mobile ions outside the ion-exclusion region;
* the **reference** state: `pdie` everywhere, no ions.

The polar solvation free energy is

$$
\Delta G_\text{polar} = \tfrac12 \sum_i q_i\,\big[u_\text{solv}(\mathbf r_i) - u_\text{ref}(\mathbf r_i)\big].
$$

Because both states use the same grid, the self-energy of each charge on the
grid cancels exactly. The energies of the single states
(`result.solvated_kt`, `result.reference_kt`) contain that self-energy and
depend on the grid; only their difference is physical. `result.energy` is in
kcal/mol, the attributes ending in `_kt` are in kT.

## Dielectric boundary

`pdie` and `sdie` are the dielectric constants of solute and solvent. The
boundary between them is sharp and is evaluated on the faces between grid
nodes (one map per axis). `surface` defines it:

| `surface` | Boundary | APBS |
|---|---|---|
| `"mol"` (default) | solvent-excluded surface: the region a probe sphere of radius `srad` (Å) cannot reach, built from surface points at density `sdens` (points per Å²) | `srfm mol`; the maps are identical node for node |
| `"vdw"` | union of the atomic spheres (`"mol"` with a zero probe) | `srfm mol` with `srad 0` |
| `"harmonic"` | molecular surface evaluated on the nodes; the face values are harmonic means of the neighbouring nodes (smoother) | none; this is **not** APBS `srfm smol` |

The defaults `srad=1.4` and `sdens=10.0` are APBS' usual values. Charges are
spread onto the grid trilinearly (APBS `chgm spl0`).

## Mobile ions

`ionic_strength` (mol/L) adds a 1:1 salt: ions of charge +1 and −1, each at
that concentration, with radius `ion_radius` (Å, default 2.0). Ions are
excluded within `r_i + ion_radius` of every atom with a radius above 0 (the
Stern layer); atoms of radius 0, such as some AMBER hydrogens, carry their
charge but do not exclude ions.

For other electrolytes, list the species as `(charge, concentration, radius)`,
as with APBS `ion`. They must be electroneutral, and the exclusion region uses
the largest radius:

```python
# 0.1 M CaCl2: Ca2+ at 0.1 M and Cl- at 0.2 M
salt = pbsolve.Solver(ions=[(2.0, 0.1, 2.0), (-1.0, 0.2, 1.8)])
print(salt.ionic_strength)  # ½ Σ c z² = 0.3 mol/L
```

`temperature` (K, default 298.15) enters $k_BT$, $l_B$ and $\kappa$.

## Boundary condition

The potential on the faces of the outermost (coarsest) grid comes from an
analytic approximation, `bcfl`. Every finer level takes its boundary values
from the coarser solution ([focusing](grids.md)).

| `bcfl` | Boundary values | Notes |
|---|---|---|
| `"mdh"` (default) | sum of the Debye–Hückel potentials of all atoms at every boundary node, as APBS `bcfl mdh` | the most accurate; its cost grows with atoms × boundary nodes |
| `"mdh-fast"` | the same sum, exact at Chebyshev points of patches of the box faces and interpolated in between; error at most 1e-10 of the summed terms | not an APBS option; energies agree with `"mdh"` to ~1e-13, the boundary is several times faster on large grids |
| `"sdh"` | one sphere carrying the total charge, dipole and quadrupole, as APBS `bcfl sdh` | cheaper; changes the result |
| `"zero"` | u = 0 | needs a coarse box much larger than the molecule |

```python
fast = pbsolve.Solver(ionic_strength=0.15, bcfl="mdh-fast")
print(fast.polar_solvation(coords, charges, radii).energy)
```

## Non-polar term

MM-PBSA adds a non-polar solvation term γ·SASA + β.
{py:func}`pbsolve.sasa` gives the solvent-accessible surface area of every atom
(Shrake–Rupley) and {py:func}`pbsolve.nonpolar_energy` the term itself, by
default with Sitkoff's γ = 0.00542 kcal/(mol·Å²) and β = 0.92 kcal/mol and a
1.4 Å probe:

```python
area = pbsolve.sasa(coords, radii)  # Å² per atom
nonpolar = pbsolve.nonpolar_energy(coords, radii)  # kcal/mol
print(f"SASA {area.sum():.0f} Å², non-polar term {nonpolar:.2f} kcal/mol")
```

[Trajectories and MM-PBSA](many.md) combines both terms for complex, receptor and
ligand.
