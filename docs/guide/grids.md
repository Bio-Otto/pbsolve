# Grids and focusing

pbsolve uses APBS' `mg-auto` scheme: a coarse grid large enough for the
analytic [boundary condition](model.md#boundary-condition), then a finer grid
around the molecule that takes its boundary values from the coarse solution
("focusing"). Both levels have the same number of nodes. When the spacing
shrinks by more than a factor of 4, APBS inserts intermediate levels, and so
does pbsolve. Energies and potentials come from the finest level.

A {py:class}`pbsolve.GridSpec` describes this set of grids.

## Automatic grids

Unless you pass `grid=...`, every calculation builds its grid from the atoms,
as APBS' `psize` helper does:

* the **fine box** is the extent of the atoms (radii included) plus `fadd`
  (Å, default 20);
* the **coarse box** is `cfac` times the extent (default 1.7), and at least
  the fine box;
* the **node count** per axis is the smallest `8k+1` that gives a fine
  spacing of at most `grid_spacing` (Å, default 0.5).

```python
import numpy as np
import pbsolve

coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")

# grid_spacing, cfac and fadd at their defaults
solver = pbsolve.Solver(grid_spacing=0.5, cfac=1.7, fadd=20.0, ionic_strength=0.15)
grid = solver.make_grid(coords, radii)
print(grid.dims, grid.fine_spacing)
for level in grid.levels:  # coarse first
    print(level["dims"], level["spacing"], level["lengths"])
```

{py:meth}`~pbsolve.Solver.make_grid` returns the grid a calculation without
`grid=` would use, so you can inspect it, reuse it or change it.

## Choosing the spacing

The spacing sets the accuracy and the cost: halving it multiplies the number
of nodes, the memory and roughly the time by eight. 0.5 Å is the usual
choice for proteins. With a sharp dielectric boundary the error does not
shrink monotonically, because the surface cuts the grid differently at every
spacing; the Born ion (a = 2 Å) for example:

| Fine spacing (Å) | 0.5 | 0.333 | 0.25 | 0.2 |
|---|---|---|---|---|
| Error vs the analytic energy | −4.7 % | −1.1 % | −1.3 % | −0.9 % |

So compare energies only between calculations with the same grid settings,
and judge convergence from the trend over several spacings
([validation](../VALIDATION)). Averaged over placements of the molecule on
the grid, the error falls in proportion to the spacing (first order), from
about 3 % at 0.5 Å to 1 % at 0.2 Å. This holds for the Born ion and for
charges anywhere inside a sphere. {py:func}`pbsolve.analytic.kirkwood_energy`
gives the exact energy of such a sphere [Kirkwood 1934], with or without
salt, to check settings or new methods against;
{py:func}`pbsolve.analytic.two_sphere_energy` does the same for two spheres.
`scripts/validate_analytic.py` measures the order. With
`interface="fractional"` ([dielectric boundary](model.md#dielectric-boundary))
the errors at 0.5 Å are at most 0.7 % and fall about as h², and the placement
spread shrinks 4–46 times. Energy differences
computed on one grid, such as the association energy of the two spheres,
are far more accurate than the energies themselves: within 0.002 kcal/mol
with `surface="vdw"` and ~0.015 kcal/mol with the default `mol`, where the
complex is off by 1–2 kcal/mol.

## The same grid for related calculations

Discretisation errors cancel best when related calculations share a grid:

* **MM-PBSA**: complex, receptor and ligand are solved on the complex's grid;
  {py:func}`pbsolve.pbsa_solvation_frame` does this for you.
* **Several conformations or mutants**: build one grid, for example from the
  largest structure, and pass it to every calculation.

```python
peptide = slice(1449, None)  # chain B of 1YCR: the last 221 atoms
complex_dg = solver.polar_solvation(coords, charges, radii, grid=grid).energy
peptide_dg = solver.polar_solvation(
    coords[peptide], charges[peptide], radii[peptide], grid=grid
).energy
```

Every charge must lie inside the grid, away from its outer faces (at least
1, 2 or 3 node spacings with `chgm` `"spl0"`, `"spl2"` or `"spl4"`, see
[charges](model.md#charges)); otherwise the calculation raises `ValueError`.

## Grids by hand

A `GridSpec` takes the common centre, the node counts and the box lengths,
the equivalents of APBS `cgcent`/`fgcent`, `dime`, `cglen` and `fglen`.
Without `fine_length` it is a single grid without focusing.

```python
finer = pbsolve.GridSpec(
    center=grid.center,
    dims=[161, 161, 129],  # 8k+1 suits the multigrid preconditioner
    coarse_length=grid.coarse_length,
    fine_length=grid.fine_length,
)
print(finer.fine_spacing, len(finer.levels), "levels")
```

## Node counts and APBS

pbsolve's automatic grids use `8k+1` nodes per axis, which gives a fine choice
of sizes and suits the multigrid preconditioner. APBS silently replaces a
`dime` above 65 that is not `32c+1` by a smaller value (121 becomes 97), so a
comparison with APBS needs grids it keeps. `dime_policy="apbs"` builds those
(`8k+1` up to 65, `32c+1` above), and {py:meth}`~pbsolve.GridSpec.to_apbs`
writes the matching APBS keywords:

```python
apbs_grid = pbsolve.Solver(dime_policy="apbs").make_grid(coords, radii)
print(pbsolve.apbs_effective_dime(121), grid.apbs_dims, apbs_grid.dims)
print(apbs_grid.to_apbs())
```

`to_apbs()` raises `ValueError` for a grid APBS would change. [Comparing with
APBS](apbs.md) shows the whole workflow.
