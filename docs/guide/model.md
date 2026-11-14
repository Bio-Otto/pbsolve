# Physical model

pbsolve solves the linearized Poisson–Boltzmann equation (LPBE) the way APBS
does, in APBS' dimensionless form (the nonlinear equation is
[below](#nonlinear-equation)):

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

`coulomb="analytic"` replaces the reference state by the analytic Coulomb
potential $G = l_B \sum_j q_j / (\varepsilon_p |\mathbf r - \mathbf r_j|)$
(reaction-field decomposition). It writes $u = v + \chi G$, where $\chi$
marks the grid nodes within 1 Å (at least two spacings) of an atomic
sphere. Inside $\chi$ the charges are replaced by the discrete source of $G$,
so only the reaction field $v$ is solved for, once per focusing level:

$$
\Delta G_\text{polar} = \tfrac12 \sum_i q_i\,\big[u(\mathbf r_i) - G(\mathbf r_i)\big].
$$

No grid self-energy appears. `result.reference_kt` is then the Coulomb energy
of the charges in `pdie` and `result.solvated_kt` that plus ΔG_polar.

| `coulomb` | Coulomb part | Accuracy and cost |
|---|---|---|
| `"grid"` (default) | reference state solved on the same grid | as APBS |
| `"analytic"` | G subtracted near the solute | not an APBS option. With `interface="fractional"` the spread over grid placements halves, and the error of a protein (1YCR) at 0.5 Å shrinks by about 40 %; with the sharp interface it changes little. The CPU needs about 0.6 times the time. On a GPU it is faster for proteins of a few thousand atoms but slower for very large ones, because G is summed over all atoms. |

```python
decomposed = pbsolve.Solver(ionic_strength=0.15, interface="fractional", coulomb="analytic")
print(f"{decomposed.polar_solvation(coords, charges, radii).energy:.2f} kcal/mol")
```

The derivation, the choice of $\chi$ and the related literature are in the
[validation record](../VALIDATION).

## Dielectric boundary

`pdie` and `sdie` are the dielectric constants of solute and solvent. The
boundary between them is sharp and is evaluated on the faces between grid
nodes (one map per axis). `surface` defines it:

| `surface` | Boundary | APBS |
|---|---|---|
| `"mol"` (default) | solvent-excluded surface: the region a probe sphere of radius `srad` (Å) cannot reach, built from surface points at density `sdens` (points per Å²) | `srfm mol`; the maps are identical node for node |
| `"vdw"` | union of the atomic spheres (`"mol"` with a zero probe) | `srfm mol` with `srad 0` |
| `"harmonic"` | molecular surface evaluated on the nodes; the face values are harmonic means of the neighbouring nodes (smoother) | none; this is **not** APBS `srfm smol` |

The defaults `srad=1.4` and `sdens=10.0` are APBS' usual values.

With `"mol"` and `"vdw"`, `interface` sets the dielectric of the faces the
surface cuts:

| `interface` | Value of a cut face | Accuracy |
|---|---|---|
| `"sharp"` (default) | `pdie` or `sdie`, by the side of the surface the face's midpoint lies on | first order; as APBS |
| `"fractional"` | h / (ℓ_in/`pdie` + ℓ_out/`sdie`), where ℓ_in and ℓ_out are the lengths of the face's grid edge inside and outside the solute, computed exactly from the atomic and probe spheres | close to second order; not an APBS option |

`"fractional"` makes energies far less dependent on the spacing and on where
the molecule sits on the grid. For 1YCR at 0.5 Å the spread over grid
placements falls 5 times, and ΔΔG_polar of binding lies within 3 kcal/mol of
its zero-spacing limit instead of 13 ([validation](../VALIDATION)). The
solver itself is unchanged. The maps take extra time on the host: a
calculation runs about 1.5 times as long on a GPU and as long on the CPU.
Keep `"sharp"` to reproduce APBS.

```python
accurate = pbsolve.Solver(ionic_strength=0.15, interface="fractional")
print(f"{accurate.polar_solvation(coords, charges, radii).energy:.2f} kcal/mol")
```

## Charges

Point charges are spread onto the grid nodes; `chgm` chooses how, with
APBS' names and weights:

| `chgm` | Weights | Nodes per axis |
|---|---|---|
| `"spl0"` (default) | trilinear | 2 |
| `"spl2"` | B-spline of order 3 (piecewise quadratic; APBS calls it cubic) | 3 |
| `"spl4"` | B-spline of order 5 (piecewise quartic; APBS calls it quintic) | 5 |

The energy ½ Σ q u is taken with the same weights for `"spl2"` and with
trilinear weights for `"spl0"` and `"spl4"`, as APBS does. The wider
splines need room to the grid's outer faces (2 and 3 node spacings).
They hardly change ΔG_polar: its discretisation error comes from the sharp
dielectric boundary, and the self-energy of the spread charge cancels
between the solvated and the reference state ([validation](../VALIDATION)).
Use them to reproduce APBS inputs that set them.

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

## Nonlinear equation

The LPBE assumes that the ions respond linearly to the potential, which holds
for $|u| \ll 1$ kT/e. Near strongly charged molecules (nucleic acids,
membranes, highly charged proteins) the ions follow the Boltzmann
distribution, and `equation="npbe"` (APBS `npbe`) solves

$$
-\nabla\cdot\big(\varepsilon\nabla u\big)
+ \bar\kappa^2 \sum_j \frac{c_j z_j}{2I}\,\big(-e^{-z_j u}\big)
= 4\pi\, l_B \sum_i z_i\,\delta(\mathbf r - \mathbf r_i),
$$

which for a 1:1 salt is $\bar\kappa^2 \sinh u$ instead of $\bar\kappa^2 u$.
Newton's method solves it with the linear solver of the LPBE, starting from the
linearized solution. The solvated energy is APBS' free-energy functional
[Sharp & Honig 1990], so ΔG_polar matches APBS `npbe`
([validation](../VALIDATION)); the reference state stays linear.

```python
nonlinear = pbsolve.Solver(ionic_strength=0.15, equation="npbe")
print(f"{nonlinear.polar_solvation(coords, charges, radii).energy:.2f} kcal/mol")
```

For a typical protein such as 1YCR the two equations differ by 0.03 kcal/mol.
For strongly charged molecules the difference grows to kcal/mol and more: about
100 kcal/mol for a −30 e sphere, where the linearized potential at the ions is
−22 kT/e and the nonlinear one −7.5. The functional misses the field energy
beyond the outermost box. For strongly charged molecules, enlarge `cfac` and
`fadd`: the default boxes miss 20–30 % of the nonlinear correction there. A
nonlinear calculation costs about 1.3–1.4 times a linear one on the CPU and
1.7–1.9 times on a GPU. Without ions both equations are the same.

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
