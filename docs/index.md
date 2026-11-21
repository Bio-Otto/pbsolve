# pbsolve

**Fast Poisson–Boltzmann electrostatics and solvation energies: Rust core,
CPU and CUDA backends, Python API.**

```{include} ../README.md
:start-after: <!-- docs-intro-start -->
:end-before: <!-- docs-intro-end -->
```

## Quick start

```python
import pbsolve

coords, charges, radii = pbsolve.read_pqr("complex.pqr")   # Å, e, Å
solver = pbsolve.Solver(platform="auto", ionic_strength=0.15)
result = solver.polar_solvation(coords, charges, radii)
print(result.energy, "kcal/mol on", result.platform)
```

The [tutorial](tutorial) goes from installation to MM-PBSA and a comparison
with APBS; the [user guide](guide/index) explains the settings and tools by
topic.

This site exists for every release: `stable` is the latest release, `latest`
follows the main branch, and each release vX.Y.0 keeps its own version (the
version menu of Read the Docs switches between them; the title shows the
version you are reading).

```{toctree}
:maxdepth: 1
:caption: Using pbsolve

installation
tutorial
guide/index
reference
```

```{toctree}
:maxdepth: 1
:caption: Background

VALIDATION
ARCHITECTURE
ROADMAP
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
PACKAGING
license
```
