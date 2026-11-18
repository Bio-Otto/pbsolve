# Trajectories and MM-PBSA

Trajectory analysis and MM-PBSA need many PB calculations. pbsolve runs
several of them at once, each on its own share of the CPU threads or on its
own CUDA stream, which keeps the CPU and the GPU busy. The results are
identical, bit for bit, to running the calculations one after another.

## Many calculations at once

{py:meth}`~pbsolve.Solver.polar_solvation_batch` takes the frames of a
trajectory (one atom set, many coordinate sets) and returns ΔG_polar per
frame; {py:meth}`~pbsolve.Solver.polar_solvation_many` takes any list of
systems and returns a `PolarResult` per system, in input order.

```python
import numpy as np
import pbsolve

mol = pbsolve.io.load("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
coords, charges, radii = mol.coords, mol.charges, mol.radii
solver = pbsolve.Solver(ionic_strength=0.15)

# two "frames": the structure and a slightly perturbed copy
rng = np.random.default_rng(1)
frames = np.stack([coords, coords + rng.normal(scale=0.05, size=coords.shape)])
energies = solver.polar_solvation_batch(frames, charges, radii)  # kcal/mol per frame

peptide = mol.select("chain B")
grid = solver.make_grid(coords, radii)
results = solver.polar_solvation_many(
    [(coords, charges, radii), (coords[peptide], charges[peptide], radii[peptide])],
    grid=grid,  # one grid for all; or a list with one grid per system; or None
)
print(energies, [r.energy for r in results])
```

`workers` sets how many calculations run at once: by default 2 on the CPU
and 3 on CUDA, where pbsolve starts no more than fit in the free GPU memory
({py:meth}`~pbsolve.Solver.batch_workers`). It changes the speed, never the
results. With `grid=None` every frame gets its own automatic grid.

## MM-PBSA solvation terms

In the single-trajectory protocol, receptor and ligand are cut out of each
complex frame. {py:func}`pbsolve.pbsa_solvation_frames` computes the polar
(PB) and non-polar (γ·SASA + β, see [the model](model.md#non-polar-term))
solvation energies of complex, receptor and ligand, all three on the
complex's grid so that discretisation errors cancel in the difference:

```python
ligand = np.nonzero(mol.select("chain B"))[0]  # atom indices
results = pbsolve.pbsa_solvation_frames(solver, frames, charges, radii, ligand)

terms = results[0]  # a PBSAFrame
print(f"ΔG_polar {terms.delta_polar:.2f}  ΔG_nonpolar {terms.delta_nonpolar:.2f}  "
      f"ΔG_solv {terms.delta_solvation:.2f} kcal/mol")
print(terms.complex.polar, terms.receptor.nonpolar, terms.ligand.total)
print(terms.as_dict())  # flat dict, e.g. for a CSV row
```

The receptor defaults to all atoms that are not ligand (`receptor_atoms`
overrides it). {py:func}`pbsolve.pbsa_solvation_frame` does the same for a
single frame. pbsolve provides the solvation terms; the binding free energy
adds the gas-phase energy ΔE_MM from your force field and, optionally, the
entropy term (see [Binding entropy](#binding-entropy)):
ΔG_bind ≈ ΔE_MM + ΔG_solv − TΔS.

A call solves all the frames it is given together, so pass a long trajectory
in chunks of a few frames per worker, as `pbsolve mmpbsa` does. With files
from a simulation (charges and radii from an Amber topology; water and
monovalent ions removed):

```{code-block} python
mol = pbsolve.io.load("complex.prmtop")
solute = ~mol.solvent_mask()
sub = mol.subset(solute)
ligand = np.nonzero(sub.select("resname LIG"))[0]

chunk = []
for xyz in pbsolve.io.frames(mol, "md.nc", step=10, atoms=solute):
    chunk.append(xyz)
    if len(chunk) == 12:
        results = pbsolve.pbsa_solvation_frames(
            solver, chunk, sub.charges, sub.radii, ligand
        )
        print([terms.delta_solvation for terms in results])
        chunk.clear()
# ... and the last, shorter chunk
```

### Membrane proteins

For a membrane protein, give the solver an [implicit membrane](model.md#implicit-membrane)
(`zmem`, `lmem`, and `membrane_profile="greene2019"` for the depth profile
of Greene et al. 2019). Complex, receptor and ligand then all sit in the same
membrane on the complex's grid. Align the protein with z and centre it, as
the membrane is a slab normal to z, and pass the frames in that frame of
reference. `membrane_fill=True` keeps the water in a channel the protein
encloses. The non-polar term γ·SASA still counts the area buried in the
membrane as water-exposed.

The automatic grid is too narrow in x and y for a membrane: the slab ends
at its faces (2 % of ΔG_polar for the APBS helix example). Either pass a
grid whose coarse box is about 100 Å wide or more, or give the solver the
lateral size of the simulation box as `period=(Lx, Ly)` ([periodic lateral
boundaries](model.md#periodic-lateral-boundaries), needs ions): the complex,
receptor and ligand then each sit in a periodic array, as in the
simulation. With an NPT trajectory the box changes from frame to frame;
one period (for example the mean box) keeps the grids and the images the
same for every frame.

```{code-block} python
solver = pbsolve.Solver(ionic_strength=0.15, pdie=2.0, zmem=-20.0, lmem=40.0,
                        membrane_profile="greene2019", membrane_fill=True,
                        period=(92.0, 92.0))
terms = pbsolve.pbsa_solvation_frame(solver, frame, charges, radii, ligand)
```

### Residue-specific interior dielectrics

`dielectrics` gives every complex atom its own interior dielectric (see
[the model](model.md#interior-dielectric-per-atom)); receptor and ligand keep
their atoms' values. {py:func}`pbsolve.residue_dielectrics` sets one value for
the atoms of charged residues and another for the rest:

```python
labels = [f"{c}:{n}:{i}" for c, n, i in zip(mol.chains, mol.resnames, mol.resids)]
eps = pbsolve.residue_dielectrics(mol.charges, labels, charged=4.0, neutral=1.0)
varied = pbsolve.pbsa_solvation_frame(solver, frames[0], charges, radii, ligand, dielectrics=eps)
print(f"ΔG_polar {varied.delta_polar:.2f} kcal/mol (one pdie: {results[0].delta_polar:.2f})")
```

With atom dielectrics the force field's Coulomb energy no longer fits the
solvation terms. {py:attr}`pbsolve.PBSAFrame.delta_reference` is the
receptor–ligand interaction in the reference state, computed on the shared
grid. With one `pdie` it is the Coulomb energy divided by `pdie` (1YCR:
within 0.03 % at 0.35 Å). Use it in place of the Coulomb term:
`delta_reference + delta_polar` is the electrostatic binding energy in water,
and it does not depend on the choice of reference.

```python
print(f"ΔE_ref {varied.delta_reference:.2f}  ΔE_ref + ΔΔG_polar {varied.delta_reference + varied.delta_polar:.2f} kcal/mol")
```

`pbsolve mmpbsa --residue-dielectric CHARGED NEUTRAL` does the same and
prints `delta_reference`. The CSV file always has a column for it.

### Per-atom and per-residue shares

Every term also comes per atom, so the binding terms can be split over the
residues [Gohlke, Kiel & Case 2003]. The polar energy of a state is a sum
over the charges, ½ Σ_i q_i φ_rf(r_i), with φ_rf the reaction field (the
solvated minus the reference potential) taken with the energy's charge
weights; atom i's share is its term. The shares sum to ΔG_polar exactly. An
interaction between two charges, q_i R_ij q_j, is split equally between
them. The non-polar share of an atom is γ·SASA_i; β belongs to no atom.
{py:attr}`pbsolve.PolarResult.atom_energies` holds the polar shares of one
calculation, {py:meth}`pbsolve.PBSAFrame.atom_terms` every complex atom's
share of ΔG_polar and ΔG_nonpolar of binding (complex minus receptor or
ligand), and {py:meth}`pbsolve.PBSAFrame.residue_terms` their sums per
residue:

```python
polar, nonpolar = terms.atom_terms()  # kcal/mol per complex atom
labels = [f"{c}:{n}:{i}" for c, n, i in zip(mol.chains, mol.resnames, mol.resids)]
shares = terms.residue_terms(labels)  # label -> (polar, non-polar)
for label, (p, n) in sorted(shares.items(), key=lambda kv: sum(kv[1]))[:5]:
    print(f"{label:12s} {p:8.2f} {n:7.2f} kcal/mol")
```

These are solvation shares: add each residue's share of ΔE_MM from your
force field for a full decomposition. With `equation="npbe"` or `"smpbe"` and
ions the energy is a functional, not a sum over the charges, and there are
no per-atom terms.

### Binding entropy

The single-trajectory protocol leaves out the entropy of binding, −TΔS. The
interaction entropy [Duan, Liu & Zhang 2016] takes it from the fluctuations
of the gas-phase receptor–ligand interaction energy E_int over the same
frames,

    −TΔS = kT ln⟨exp(β (E_int − ⟨E_int⟩))⟩,

and its second-order cumulant (C2) approximation is σ²/(2kT), with σ the
standard deviation of E_int; both are ≥ 0, and they are equal when E_int is
normally distributed. The exponential average is dominated by the rare frames
with the highest energy: it converges slowly and comes out too low when σ is
large. Ekberg & Ryde (2021) therefore advise reporting σ, comparing IE and C2
over blocks of the trajectory, not using IE above σ = 15 kJ/mol (3.6 kcal/mol),
and trusting C2 where the two differ. {py:func}`pbsolve.interaction_entropy`
returns both, with their standard errors over consecutive blocks:

```python
# a stand-in for E_int of 1000 frames (kcal/mol); see below for your own frames
e_int = np.random.default_rng(0).normal(-62.0, 1.5, 1000)
s = pbsolve.interaction_entropy(e_int, temperature=298.15, blocks=5)
print(f"-TΔS  IE {s.ie:.2f} ± {s.ie_sem:.2f}  C2 {s.c2:.2f} ± {s.c2_sem:.2f} kcal/mol, "
      f"σ {s.sigma:.2f} ({'IE usable' if s.reliable else 'use C2'})")
```

For normally distributed energies with σ = 3 kcal/mol, 10⁵ frames still give IE
7.0 ± 0.4 against the exact 7.60, while C2 gives 7.59 ± 0.03 (see the
[validation](../VALIDATION.md#interaction-entropy)).

E_int is the force field's Coulomb plus Lennard-Jones energy between receptor
and ligand atoms, without solvent and cutoff. {py:func}`pbsolve.interaction_energy`
computes it for force fields with Lorentz–Berthelot combining rules (AMBER,
CHARMM without NBFIX), for example with the parameters of an OpenMM `System`:

```{code-block} python
from pbsolve.openmm_bridge import charges_from_system, lennard_jones_from_system

q = charges_from_system(system)
lj_radii, lj_eps = lennard_jones_from_system(system)
e_int = []
for xyz in frames_in_angstrom:
    coulomb, vdw = pbsolve.interaction_energy(xyz, q, lj_radii, lj_eps, ligand_indices)
    e_int.append(coulomb + vdw)
entropy = pbsolve.interaction_entropy(e_int)
minus_t_ds = entropy.ie if entropy.reliable else entropy.c2
```

## From the command line

`pbsolve mmpbsa` does all of this for a topology and a trajectory, and writes
the per-frame terms to a CSV file:

```bash
pbsolve mmpbsa complex.prmtop md.nc --ligand "resname LIG" --step 10 \
    --ionic-strength 0.15 --csv terms.csv --per-residue residues.csv
```

`--per-residue` writes every residue's share of the binding terms, averaged
over the frames, with the standard error of its total.

See [the command line](cli.md#pbsolve-mmpbsa) for every option.

## With OpenMM

{py:class}`pbsolve.openmm_bridge.OpenMMPBSA` takes the charges from an OpenMM
`System` (one `NonbondedForce`) and radii by element scheme (default
`mbondi2`) from its `Topology`. The System must describe the solute only,
without explicit water or ions.

```{code-block} python
import pbsolve
from pbsolve.openmm_bridge import OpenMMPBSA

pbsa = OpenMMPBSA(system, topology, ligand_atoms=ligand_indices,
                  solver=pbsolve.Solver(ionic_strength=0.15))
for positions in trajectory_positions:   # OpenMM Quantity, or an array in Å
    terms = pbsa.frame(positions)        # PBSAFrame
    dg_bind = delta_e_mm + terms.delta_solvation
```

The same `System` provides the Lennard-Jones parameters for
{py:func}`pbsolve.interaction_energy` (`lennard_jones_from_system`), and so
the interaction entropy of the trajectory (see [Binding entropy](#binding-entropy)).
`examples/mmpbsa_openmm.py` in the repository is a complete script that
reports both.
