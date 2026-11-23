<h1 align="center"><img src="https://raw.githubusercontent.com/Bio-Otto/pbsolve/main/docs/_static/pbsolve-logo.svg" alt="pbsolve: MMPBSA Analysis Suite" width="720"></h1>

**Fast Poisson–Boltzmann electrostatics and solvation energies — Rust core, CPU/CUDA backends, Python API.**

**Documentation:** <https://pbsolve.readthedocs.io> (tutorial, user guide, API
reference, validation). `stable` is the latest release, `latest` the main
branch, and every release vX.Y.0 keeps its own version.

<!-- docs-intro-start -->
pbsolve computes electrostatic potentials and polar solvation energies of
biomolecules from the Poisson–Boltzmann equation, on the CPU or on NVIDIA GPUs,
and plugs directly into simulation and MM-PBSA workflows. The current release
solves the linearized and the nonlinear equation with finite differences and
grid focusing; the
design keeps the physics separate from the compute backends so that further
solution methods and integrations can be added.

* **One algorithm, many devices** — physics is written once in `pbsolve-core`;
  CPU (rayon) and CUDA backends implement a small `Backend` trait.
* **Multigrid solver** — CG preconditioned by a geometric multigrid V-cycle:
  ~10–20 iterations independent of the grid spacing, 5–10× faster than
  diagonal preconditioning on protein grids (`docs/VALIDATION.md`). By default
  the V-cycle runs in single precision inside the double-precision CG
  (`preconditioner="multigrid-f32"`): same energies, ~1.5× faster on the CPU;
  `preconditioner="multigrid"` keeps the cycle in f64.
* **APBS-compatible by default, faster on request** — `bcfl="mdh"` follows
  APBS operation for operation. `bcfl="mdh-fast"` interpolates the same sum
  (energies within ~1e-13) and makes the boundary of large proteins ~6×
  cheaper.
* **Automatic GPU** — the CUDA build loads the driver and NVRTC at runtime;
  without a GPU it transparently runs on the CPU (`platform="auto"`).
* **Forces on the atoms** — `Solver.polar_forces` gives −dΔG_polar/dr on APBS'
  smooth spline surfaces (`surface="spl2"`/`"spl4"`), split into charge–field,
  dielectric-boundary and ionic-boundary parts as APBS `calcforce`.
* **Gaussian dielectric** — `surface="gauss"` grades ε from the atoms'
  Gaussian densities instead of a surface (Li et al. 2013), with ions kept
  out by their Born energy; the reference state keeps the molecule's interior
  dielectric (Li et al. 2014), which makes the energies converge. Checked
  against pyDelPhi (DelPhi's Gaussian model) face for face.
* **Membranes** — `zmem`/`lmem` add an implicit membrane slab (APBS'
  `draw_membrane2` rule, maps identical), uniform or with the depth profile
  of Greene et al. (2019), with automatic pore filling, and periodic
  lateral boundaries (`period=(Lx, Ly)`): the solute repeats in x and y as
  in the simulation box, so the slab does not end at the box faces. A
  membrane potential (`membrane_potential=` in mV, Roux 1997) adds the
  charges' energy in the field across the membrane (gating charges), and
  the non-polar term can leave out the area buried in the membrane (GBSW's
  switch, `nonpolar_membrane=`).
* **A dielectric per atom** — `dielectrics=` gives every atom (e.g. charged
  and other residues) its own interior dielectric, as DelPhi's multiple
  dielectrics; equal values reproduce one `pdie` bit for bit.
* **Honest error bars** — `series_statistics` corrects the standard error of
  an average over correlated frames (statistical inefficiency, as pymbar),
  gives intervals, the correlation time and how many frames a target error
  needs; `pbsolve mmpbsa` prints them.
* **MM-PBSA ready** — polar (PB) + non-polar (γ·SASA + β) terms, shared grids for
  complex/receptor/ligand, OpenMM bridge, OpenDX maps. Trajectory frames are
  solved several at a time; results do not depend on the number of workers.
  Per-residue and residue-pair shares, and −TΔS from the interaction entropy of
  the frames.
* **Reads simulation output directly** — PDB, mmCIF, Amber, CHARMM/NAMD, GROMACS
  and common trajectory formats, plus built-in preparation of PDB files.
* **Validated and reproducible** — checked against analytic results and
  established tools (`docs/VALIDATION.md`); deterministic on every platform.
<!-- docs-intro-end -->

## Install

<!-- docs-install-start -->
pip builds a release from its GitHub tag; this needs Python ≥ 3.10 and a
Rust toolchain ≥ 1.85 ([rustup](https://rustup.rs)), but no CUDA toolkit:

```bash
pip install "pbsolve @ git+https://github.com/Bio-Otto/pbsolve@v0.1.0"      # CPU
MATURIN_PEP517_ARGS="--features cuda-12" \
  pip install "pbsolve @ git+https://github.com/Bio-Otto/pbsolve@v0.1.0"    # CPU + CUDA 12 (cuda-13 for CUDA 13)
```

The CUDA build loads the driver and NVRTC when it runs and falls back to the
CPU without them; NVRTC comes from conda (`cuda-nvrtc`) or pip
(`nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-nvrtc` for 13). On Windows set the
variable with `set MATURIN_PEP517_ARGS=--features cuda-12` (cmd) or
`$env:MATURIN_PEP517_ARGS="--features cuda-12"` (PowerShell).

From v0.2.0 on, every release carries ready-made wheels for Python 3.10–3.14,
which need no Rust: Linux (x86_64, aarch64) and Windows (x86_64), each one
wheel for the CPU and CUDA 12+, and macOS (Apple silicon, Intel), CPU only.
`pip install pbsolve`; `pip install "pbsolve[cuda12]"` adds NVRTC for the GPU.
They are attached to the GitHub release and go to PyPI once publishing is
enabled; conda-forge is planned (`docs/ROADMAP.md`, R2, and
`docs/PACKAGING.md`).

Check the installation (like `python -m openmm.testInstallation`):

```bash
pbsolve test        # Born ion on every available platform + cross-platform agreement
pbsolve info        # build / CUDA status and the reason if CUDA is unavailable
```
<!-- docs-install-end -->

## Input files

pbsolve reads simulation output directly — no conversion step:

| kind | formats |
|---|---|
| structures | PDB (multi-model), PQR, mmCIF, GRO, mol2 |
| topologies (charges) | Amber prmtop, CHARMM/NAMD PSF, GROMACS top/itp (`#include` via `GMXLIB` or `-I`) |
| coordinates | Amber inpcrd/rst7/ncrst |
| trajectories | DCD (CHARMM/NAMD/OpenMM), XTC, TRR, Amber NetCDF, mdcrd, multi-model PDB/GRO |

```python
mol = pbsolve.io.load("complex.prmtop", "complex.inpcrd")            # Amber charges + radii
mol = pbsolve.io.load("topol.top", "md.gro", radii="mbondi2")        # GROMACS, radii by scheme
solute = mol.strip_solvent()                                          # water + monovalent ions
lig = solute.select("resname LIG")                                    # boolean mask
for xyz in pbsolve.io.frames(mol, "md.xtc", step=10, atoms=~mol.solvent_mask()):
    ...
```

```bash
pbsolve solvation complex.prmtop --coords complex.inpcrd
pbsolve convert topol.top --coords md.gro --radii mbondi2 complex.pqr
pbsolve mmpbsa complex.prmtop prod.nc --ligand "resname LIG" --step 10 --csv terms.csv
```

Radii come from the file when it has them (PQR, prmtop `RADII`), otherwise
from an element scheme (`bondi`, `mbondi`, `mbondi2`, `parse`). Selections:
`resname`, `name`, `resid`, `chain`, `segid`, `element`, `index`, `serial`,
`protein`, `water`, `ions`, `hydrogen` with `and`/`or`/`not` and parentheses.
The readers are checked against MDAnalysis/ParmEd on files written by tleap,
OpenMM, cpptraj and GROMACS (`scripts/make_io_fixtures.sh`, `scripts/validate_io.py`).

## Preparing PDB files

`pbsolve.prep` adds hydrogens, repairs missing heavy atoms, optimises the
hydrogen-bond network (HIS/ASN/GLN flips, hydroxyl and carboxyl hydrogens),
assigns protonation states and AMBER/PARSE/CHARMM charges and radii. It follows
the conventions of pdb2pqr and is validated against it (`docs/VALIDATION.md`).

```bash
pbsolve prep protein.pdb protein.pqr --ff amber              # add --noopt to skip the optimisation
pbsolve prep protein.pdb protein.pqr --ph 5.0                # model-compound pKa rules
pbsolve prep protein.pdb protein.pqr --titration propka      # PROPKA 3, if installed (optional package)
pbsolve prep complex.pdb complex.pqr --ligand lig.mol2       # append a ligand with its own charges
pbsolve solvation protein.pdb --ff amber                     # prepare + solve in one step
```

```python
mol, report = pbsolve.prep.prepare("protein.pdb", forcefield="amber", ph=7.0)
print(report.summary())
```

Waters are dropped; nucleic acids are not implemented yet.

## Python API

A step-by-step tutorial, from installation to MM-PBSA and the comparison
with APBS, is in `examples/pbsolve_tutorial.ipynb`.

```python
import numpy as np
import pbsolve

solver = pbsolve.Solver(platform="auto", grid_spacing=0.5, ionic_strength=0.15)
coords, charges, radii = pbsolve.read_pqr("protein.pqr")     # Å, e, Å

res = solver.polar_solvation(coords, charges, radii)
print(res.energy, "kcal/mol on", res.platform, res.iterations)

# MM-PBSA solvation terms for one frame (shared grid for all three species)
frame = pbsolve.pbsa_solvation_frame(solver, coords, charges, radii, ligand_atoms=lig_idx)
print(frame.delta_polar, frame.delta_nonpolar)

# ... or for many frames, solved concurrently (default: 2 workers on the CPU, 3 on CUDA,
# fewer on CUDA if they do not fit in the free GPU memory: see solver.batch_workers(grid))
frames = pbsolve.pbsa_solvation_frames(solver, frame_coords, charges, radii, ligand_atoms=lig_idx)
results = solver.polar_solvation_many([(x, charges, radii) for x in frame_coords], workers=3)
print(results[0].timings)   # seconds per stage: maps, boundary, assembly, setup, solve, energy

# Electrostatic potential map for visualisation
pot, origin, spacing, _ = solver.potential(coords, charges, radii)
pbsolve.write_dx("potential.dx", pot, origin, spacing)
```

With OpenMM (see `examples/mmpbsa_openmm.py`):

```python
from pbsolve.openmm_bridge import OpenMMPBSA
pbsa = OpenMMPBSA(system, topology, ligand_atoms=lig_idx, solver=solver, radii="mbondi2")
terms = pbsa.frame(positions)            # OpenMM Quantity or Å array
dG = delta_E_mm + terms.delta_solvation  # ΔE_MM from your OpenMM workflow
```

## Command line

```bash
pbsolve solvation protein.pqr --spacing 0.5 --ionic-strength 0.15 --dx pot.dx
```

## Validation

`docs/VALIDATION.md` records the current checks: analytic references, grid
convergence, cross-checks against other Poisson–Boltzmann software (the
`pbsolve apbs` subcommand runs such comparisons when APBS is installed), and
tests of the readers and the structure preparation against reference tools.

## Status

pbsolve is an early preview (0.0.x); the API may still change. Validation numbers and
the GPU status are in `docs/VALIDATION.md`, the design in `docs/ARCHITECTURE.md`,
the development plan in `docs/ROADMAP.md`, packaging in `docs/PACKAGING.md`.

## Layout

```
crates/pbsolve-core   physics + CPU backend (pure Rust, no Python)
crates/pbsolve-cuda   CUDA backend (cudarc + NVRTC kernels in kernels/pb_kernels.cu)
crates/pbsolve-io     structure, topology and trajectory readers
crates/pbsolve-py     PyO3 bindings -> pbsolve._core
python/pbsolve        Python API, structure preparation, MM-PBSA helpers, OpenMM bridge, CLI
recipe/               conda recipe (CPU + CUDA 12/13 variants)
scripts/              validation and maintenance scripts
examples/             tutorial notebook, OpenMM MM-PBSA example
tests/                pytest suite;  `cargo test -p pbsolve-core` for Rust tests
```

## References

* Baker NA, Sept D, Joseph S, Holst MJ, McCammon JA. Electrostatics of
  nanosystems: application to microtubules and the ribosome. *PNAS* 98, 10037 (2001).
* Jurrus E, et al. Improvements to the APBS biomolecular solvation software
  suite. *Protein Sci.* 27, 112 (2018).
* Sitkoff D, Sharp KA, Honig B. Accurate calculation of hydration free
  energies using macroscopic solvent models. *J. Phys. Chem.* 98, 1978 (1994).

pbsolve is an independent project and is not affiliated with or endorsed by the
authors of the software cited above.

## License

MIT — see `LICENSE`. A few files contain code or data derived from APBS and
pdb2pqr (BSD-3-Clause) and xdrfile (BSD-2-Clause), all compatible with MIT;
see `THIRD_PARTY_NOTICES.md`, which must accompany redistributions.
