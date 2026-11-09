# User guide

These pages explain pbsolve topic by topic: what each setting means, what it
changes, and which tool to use for a job. The [tutorial](../tutorial) covers
the same ground once, in order and with outputs; the
[API reference](../reference) lists every class, function and parameter.

The examples run as they are from the root of the repository. They use the
structures in `tests/data`, mainly MDM2 with a p53 peptide (PDB 1YCR,
prepared by pdb2pqr with AMBER charges and radii; the peptide is chain B).
The test suite runs every example (`tests/test_guide.py`).

```{toctree}
:maxdepth: 1

model
grids
solver
platforms
many
files
cli
apbs
```
