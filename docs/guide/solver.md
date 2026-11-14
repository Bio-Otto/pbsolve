# Linear solver

Each grid level is a sparse, symmetric positive definite linear system.
pbsolve solves it with preconditioned conjugate gradients (CG) until the
relative residual ‖r‖/‖b‖ falls below `tol` (default 1e-8). The
preconditioner decides how many iterations that takes; it does not change
the answer beyond that tolerance.

## Preconditioners

| `preconditioner` | Method | CG iterations per level |
|---|---|---|
| `"multigrid-f32"` (default) | one geometric multigrid V-cycle (Galerkin coarse operators, Chebyshev smoothing) evaluated in single precision, inside the double-precision CG | ~10–20, independent of the spacing |
| `"multigrid"` | the same V-cycle in double precision | the same |
| `"jacobi"` | diagonal scaling | 150–700, growing as the grid is refined |

The single-precision cycle moves half the data and gives the same energies as
`"multigrid"` to ~1e-12 relative (at most 4.2e-13 over 28 measured cases) with
the same iteration counts; the CG itself always runs in double precision.
A grid too small to coarsen falls back to Jacobi; `result.preconditioners`
reports what each solve used.

```python
import pbsolve

coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")

for name in ["multigrid-f32", "multigrid", "jacobi"]:
    solver = pbsolve.Solver(ionic_strength=0.15, preconditioner=name)
    result = solver.polar_solvation(coords, charges, radii)
    print(f"{name:14s} {result.energy:.8f} kcal/mol  iterations {result.iterations}")
```

## What a result reports

Besides the energies, a {py:class}`pbsolve.PolarResult` describes every linear
solve: first the solvated state's levels (coarse first), then the reference
state's.

```python
solver = pbsolve.Solver(ionic_strength=0.15)
result = solver.polar_solvation(coords, charges, radii)
print(result.iterations)       # CG iterations per solve
print(result.residuals)        # final relative residuals
print(result.preconditioners)  # e.g. ['multigrid-f32', ...]
print(result.levels)           # multigrid levels per solve
print({stage: round(seconds, 3) for stage, seconds in result.timings.items()})
```

`timings` splits the wall time into stages:

| Key | Stage |
|---|---|
| `maps` | dielectric and ion maps (molecular surface) |
| `boundary` | boundary values: analytic on the coarsest level, then focusing |
| `assembly` | charges on the grid, operator, right-hand side |
| `setup` | multigrid hierarchy and uploads |
| `solve` | CG iterations |
| `energy` | energy evaluation |
| `total` | the sum |

On CUDA the solvated state's surface and analytic boundary are built on the
CPU while the GPU works; there, `maps` and `boundary` count only the time the
GPU had to wait for them.

## When CG does not converge

A solve that does not reach `tol` within `max_iter` iterations (default
20000) raises {py:class}`pbsolve.ConvergenceError`, a `RuntimeError`; pbsolve
never returns an unconverged result. With the multigrid preconditioners CG
normally needs 10–20 iterations, so reaching the limit points to a problem
with the input or the settings rather than to a small `max_iter`.

```python
try:
    impatient = pbsolve.Solver(preconditioner="jacobi", max_iter=20)
    impatient.polar_solvation(coords, charges, radii)
except pbsolve.ConvergenceError as err:
    print("not converged:", err)
```

## Error estimates

With a sharp dielectric boundary, ΔG_polar depends on the spacing and on
where the surface falls between the nodes. {py:func}`pbsolve.estimate_error`
measures both:

* It solves on several placements of the grid (the automatic grid moved by
  random fractions of a spacing) and reports their mean, the standard
  deviation (the spread of a single calculation) and the standard error of
  the mean.
* Given two spacings or more, it also extrapolates the means at the two
  finest to zero spacing (first order, as measured on the analytic
  references). `discretisation_error` is the estimated error of the mean.
* With `parts` (receptor and ligand masks) the quantity is the association
  energy ΔG(complex) − ΔG(receptor) − ΔG(ligand) on the complex's grid, as in
  MM-PBSA.

```python
import pbsolve

coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
solver = pbsolve.Solver(ionic_strength=0.15)
est = pbsolve.estimate_error(solver, coords, charges, radii, placements=4)
print(est.format())
print(est.energies[0])  # the first placement is the ordinary calculation
```

Each placement costs one calculation, so the estimate costs `placements`
times as much per spacing; {py:meth}`~pbsolve.Solver.polar_solvation_many`
runs them concurrently. For the extrapolation, use spacings of 0.35 Å and
below: from 0.35 and 0.25 Å the extrapolated energies of the analytic
references are within 0.3 % of the exact ones, and the estimated error is
within 30 % of the true one. From 0.5 and 0.35 Å the correction can be off
by a factor of 2 ([validation](../VALIDATION)).

```{code-block} python
est = pbsolve.estimate_error(solver, coords, charges, radii, placements=8, spacings=[0.35, 0.25])
print(est.extrapolated, est.discretisation_error)
```

For 1YCR at 0.15 M (8 placements), the mean ΔG_polar is −1234.7, −1205.6
and −1190.5 kcal/mol at 0.48, 0.34 and 0.25 Å, and extrapolates to
−1152 kcal/mol. Placements spread by 4 kcal/mol (sd) at 0.5 Å and by
1.2 kcal/mol at 0.25 Å. On the command line, `pbsolve solvation` takes
`--placements` and `--extrapolate` ([command line](cli.md#pbsolve-solvation)).

## Reproducibility

Sums run in a fixed order. The same input on the same platform therefore
gives the same result bit for bit, whatever the number of CPU threads and
however many calculations run at once ([batches](many.md)). CPU and CUDA agree
to rounding (1e-13 relative or better); see [CPU and GPU](platforms.md).

## Solving given maps

{py:meth}`~pbsolve.Solver.maps` returns the coefficient maps of one focusing
level (dielectric on the three face grids, κ̄², charges), and
{py:meth}`~pbsolve.Solver.solve_maps` solves the LPBE for maps and boundary
values you supply. Feeding APBS' own maps into `solve_maps` tests pbsolve's
discretisation and solver in isolation, which is how the
[APBS comparison](apbs.md) locates any difference.

```python
maps = solver.maps(coords, charges, radii)  # the finest level
print(maps["dims"], maps["spacing"], maps["eps_x"].shape, maps["kappa_bar_sq"])
```
