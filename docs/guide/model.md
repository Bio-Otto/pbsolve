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
boundary between them is evaluated on the faces between grid nodes (one map
per axis). `surface` defines it:

| `surface` | Boundary | APBS |
|---|---|---|
| `"mol"` (default) | solvent-excluded surface: the region a probe sphere of radius `srad` (Å) cannot reach, built from surface points at density `sdens` (points per Å²) | `srfm mol`; the maps are identical node for node |
| `"vdw"` | union of the atomic spheres (`"mol"` with a zero probe) | `srfm mol` with `srad 0` |
| `"harmonic"` | molecular surface evaluated on the nodes; the face values are harmonic means of the neighbouring nodes (smoother) | none; this is **not** APBS `srfm smol` |
| `"spl2"`, `"spl4"` | atomic spheres smoothed over ±`swin` (Å, default 0.3) by a cubic or septic switch: ε rises continuously from `pdie` to `sdie`, and the ion accessibility from 0 to 1 | `srfm spl2`, `spl4`; the maps are identical to the 7 digits APBS writes |
| `"gauss"` | no surface: ε from a molecular density built of atomic Gaussians, graded from `pdie` in densely packed regions to `sdie` in the solvent ([below](#gaussian-dielectric)) | none |

The defaults `srad=1.4` and `sdens=10.0` are APBS' usual values.

The spline surfaces make the energy a smooth function of the atom positions,
which [forces](#forces) need. They are a different model: no probe fills the
crevices, and the smoothing moves the effective boundary about 0.2 Å inwards.
1YCR has ΔG_polar −1231 kcal/mol with `"mol"`, −1754 with `"vdw"` and
−2609 with `"spl2"`. Use radii fitted for a smooth boundary with them. Every
charged atom should lie inside the solute: a charge with radius 0 near the
surface feels a grid-dependent self-energy.

### Gaussian dielectric

`surface="gauss"` replaces the surface by a molecular density [Grant, Pickup
& Nicholls 2001; Li et al. 2013]. An atom of radius R contributes
g = exp(−d²/σ²R²) at distance d (σ = `gauss_sigma`, default 0.93), the
molecule has the density ρ = 1 − Π(1 − g), and

$$
\varepsilon = \rho\,\varepsilon_\text{pdie} + (1 - \rho)\,\varepsilon_\text{sdie}.
$$

Densely packed regions get nearly `pdie`, loose packing and grooves
intermediate values, the solvent `sdie`. Li et al. (2013) fitted σ = 0.93
with `pdie=4` to pKa values. The mobile ions pay the Born energy of leaving
the bulk solvent for the local dielectric [Jia et al. 2017], which scales κ̄²
by

$$
K = \exp\left[-\frac{z^2 l_B}{2\,r_\text{ion}}\left(\frac{1}{\varepsilon} - \frac{1}{\varepsilon_\text{sdie}}\right)\right]
$$

with r_ion = `ion_radius` (l_B: Bjerrum length in vacuum). Jia et al. print
this penalty with the opposite overall sign; the sign here keeps the ions
out of the molecule, as their text describes.

The **reference state** of ΔG_polar keeps the Gaussian ε wherever it is at
most `gauss_contour` (default 20) and puts `pdie` beyond, without ions, as Li
et al. (2014): ΔG_polar is the transfer of the molecule with its internal
dielectric into the solvent. (Li et al. put vacuum beyond the contour; with
`pdie=1` this is the same.) A homogeneous `pdie` reference does not work for
this model: ε rises from `pdie` within a few tenths of an Å of every nucleus,
so the grid self-energies of the charges would not cancel, and ΔG_polar of
1YCR would be −39 000 kcal/mol at 0.5 Å and −61 000 at 0.25 Å
([validation](../VALIDATION)). With the contour, a single Gaussian atom is
within 1–3 % of the exact solution at 0.5 Å, and ΔΔG_polar of binding for
1YCR changes by less than 3 kcal/mol between 0.5 and 0.25 Å. This has
consequences:

* A charged atom of radius 0 must lie inside the contour; otherwise its
  charge sees different dielectrics in the two states and pbsolve refuses
  it. Give it a radius (for example `Molecule.with_radii("mbondi2")`) or
  raise `gauss_contour`. Charges just inside the contour make results
  noisier.
* `coulomb="analytic"` and [forces](#forces) are not available: the
  reference state is neither homogeneous nor smooth.
* `equation="npbe"` is refused. The Born penalty is bounded (138 kT with
  `pdie=1`), the potential next to a point charge is not, so the Boltzmann
  ion density diverges at the charges inside the molecule. `"smpbe"` bounds
  the density and works.
* All ion species need one valence and one radius (one penalty).

A calculation takes about 1.5 times as long as with `"mol"` on a GPU (the
maps are built on the host) and as long on the CPU.

```python
gauss = pbsolve.Solver(ionic_strength=0.15, surface="gauss", pdie=4.0)
print(f"{gauss.polar_solvation(coords, charges, radii).energy:.2f} kcal/mol")
```

### Cut faces

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

### Interior dielectric per atom

`pdie` is one dielectric for the whole solute. With `dielectrics`, an array
with one value per atom, every atom gets its own, as with DelPhi's multiple
dielectrics [Rocchia, Alexov & Honig 2001]. {py:meth}`pbsolve.Solver.polar_solvation`,
the batch methods, `potential`, `maps`, the MM-PBSA functions and
{py:func}`pbsolve.estimate_error` accept it. Residue-specific values stand for
the larger reorganisation of charged side chains around a charge [Wang,
Zhang, Rocchia & Alexov 2013]. For MM/GBSA, Liu, Peng & Zhang (2019) fitted
2.7 for charged and 1.1 for the other residues to protein–protein binding;
no values have been fitted for PB. {py:func}`pbsolve.residue_dielectrics`
gives two values by the net charge of each residue.

A face inside the molecular surface (the sharp `"mol"` or `"vdw"` marking)
takes the dielectric of the atom whose power cell contains its midpoint x:
the atom i with the smallest

$$
\pi_i(\mathbf x) = |\mathbf x - \mathbf c_i|^2 - R_i^2
$$

(ties go to the lower index). These cells divide two overlapping spheres at
the plane through the circle where they intersect. A face in a crevice the
probe cannot reach goes to the nearest sphere in this sense. Atoms of radius
0 take part with R = 0 and own only faces outside the other spheres. Faces
outside the surface get `sdie` in the solvated state.

The **reference state** keeps this interior map and has `pdie` outside the
surface, without ions. ΔG_polar is then the transfer of the molecule, with
its interior dielectrics, from a medium of `pdie` (vacuum for `pdie=1`) into
the solvent. The faces around every charge are the same in both states, so
the grid self-energies cancel as they do with one `pdie`. With every value
equal to `pdie` the maps and energies are those of `pdie` bit for bit.

The interior dielectric acts through the charges' distances from the
surface and on each other:

* A charge at the centre of one sphere sees only the outer boundary:
  ΔG = −(q²/2a)(1/`pdie` − 1/`sdie`) for any interior value.
* Off-centre charges follow the Kirkwood solution for the sphere's interior
  value.
* For two spheres with interiors 4 and 10, pbsolve lies within 0.3–0.7 % of
  the exact solution at 0.25 Å ([validation](../VALIDATION)).

For binding energies the force field's Coulomb term has to be replaced by
the interaction in the reference state ({py:attr}`pbsolve.PBSAFrame.delta_reference`,
[MM-PBSA](many.md#residue-specific-interior-dielectrics)).

Only `"mol"` and `"vdw"` with `interface="sharp"` and `coulomb="grid"` are
supported ([forces](#forces) need a spline surface). APBS has one `pdie`, so
there is nothing to compare there. The maps are built on the host, and the
reference state rebuilds the surface on every level. For 1YCR a calculation
takes 6 % longer on the CPU and about twice as long on a GPU.

```python
mol = pbsolve.io.load("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
labels = [f"{c}:{n}:{i}" for c, n, i in zip(mol.chains, mol.resnames, mol.resids)]
eps = pbsolve.residue_dielectrics(mol.charges, labels, charged=4.0, neutral=1.0)
variable = pbsolve.Solver(ionic_strength=0.15, pdie=1.0)
print(f"{variable.polar_solvation(coords, charges, radii, dielectrics=eps).energy:.2f} kcal/mol")
```

### Implicit membrane

`zmem` adds a lipid membrane to the solvated state as a slab normal to z,
from `zmem` to `zmem + lmem` (Å), without mobile ions [Callenberg et al.
2010]. The rule is that of APBS' `draw_membrane2` (APBSmem): every
dielectric face in the slab that is not solute gets the membrane's
dielectric, and every node in the slab loses its ions. APBS itself has no
keyword for it; `draw_membrane2` edits APBS' maps. pbsolve's maps are
identical to its output node for node (`mol` and `spl2`, with and without
an exclusion cone; [validation](../VALIDATION)).

| Argument | Default | Meaning |
|---|---|---|
| `zmem` | none | z of the lower membrane surface (Å); none: no membrane |
| `lmem` | 40 | thickness (Å) |
| `mdie` | 2.0 | dielectric of a uniform membrane |
| `membrane_profile` | none | a dielectric that varies with depth instead of `mdie` (below) |
| `membrane_exclusion` | (0, 0) | radii (Å) at the bottom and the top of a cone around the vertical axis that stays water, for channel pores |
| `membrane_fill` | off | keep water and ions where a membrane plane is not connected to the sides of the box |

The axis of the cone runs through the grid's centre shifted by half a
spacing in x and y, as in `draw_membrane2`, so the protein has to be
centred and aligned with z (the automatic grid is centred on it).
`membrane_fill=True` finds pores without a cone: in every z plane of the
slab, only the parts connected to the sides of the box become membrane, so
a pore or cavity that the protein encloses keeps its water and ions.

**Depth profile.** The low dielectric of the hydrocarbon core rises
through the ester and headgroup regions to that of water. Greene et al.
(2019) fitted this profile to decharging free energies of amino-acid side
chains in an explicit DMPC membrane. It is ε = 1, 2, 4, 20 and 80 at 20,
15, 10, 5 and 0 Å below the membrane surface, interpolated by PCHIP. For
the P2Y12 receptor it raised the correlation of MM-PBSA with experiment
from R = 0.59 (uniform ε = 4) to 0.92. `membrane_profile="greene2019"`
uses it, and a sequence of (depth, ε) points defines another profile. The
depth is measured from the nearer surface, and the value of the last point
continues to the centre; a membrane thicker than 40 Å therefore has a
wider ε = 1 core. Amber implements the same data (`membraneopt=2`), but
its interpolation of a thicker membrane's core is not documented, so the
two need not agree to the last digit.

The **reference state** has no membrane: ΔG_polar is the transfer of the
solute from a medium of `pdie` into water with the membrane. With
`pbsa_solvation_frame` the complex, receptor and ligand all see the same
membrane on the complex's grid. The non-polar term γ·SASA still counts the
area buried in the membrane as exposed to water.

**Boundaries.** The slab runs into the faces of the box, where the
boundary condition (`mdh`, `zero`, ...) assumes water around the molecule.
For a charge in the middle of a 30 Å slab against the exact layered
solution, the error is below 0.2 % with a 150 Å coarse box and 0.25 Å
spacing. A larger coarse box does not help at 0.5 Å (−2.3 % with 600 Å):
its coarse levels resolve the slab badly. Amber uses periodic lateral
boundaries for membranes; pbsolve does not have them yet.

Not available with `surface="gauss"`, atom dielectrics, `coulomb="analytic"`
or [forces](#forces). With the GPU, the membrane is drawn into the maps on
the host: a calculation of the APBS helix example took 0.15 instead of
0.11 s.

```{code-block} python
# A membrane protein aligned with z and centred at the origin; a 40 Å bilayer.
membrane = pbsolve.Solver(ionic_strength=0.1, pdie=4.0, zmem=-20.0, lmem=40.0,
                          membrane_profile="greene2019", membrane_fill=True)
result = membrane.polar_solvation(coords, charges, radii)
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

Point ions can crowd without limit: near a −50 e cavity the nonlinear equation
puts 84 M of counterions at the surface. `equation="smpbe"` gives every ion a
cube of edge `ion_size` (Å) on a lattice [Borukhov et al. 1997]. The local
concentrations become $c_j e^{-z_j u} / D$ with
$D = 1 - \sum_k \phi_k + \sum_k \phi_k e^{-z_k u}$, where
$\phi_k = c_k N_A a^3$, so no species exceeds one ion per cell. The ion
energy becomes $\ln D / (N_A a^3)$ per volume.

```python
crowded = pbsolve.Solver(ionic_strength=0.15, equation="smpbe", ion_size=6.0)
print(f"{crowded.polar_solvation(coords, charges, radii).energy:.2f} kcal/mol")
```

* **Cost.** Finite size costs free energy, about a³ while the ions are
  dilute: 0.007 kcal/mol for 1YCR with 6 Å ions, 2.6 kcal/mol for the
  antibody 5DK3.
* **Choice of size.** All ions have the same size, and there is no single
  right value: fits to ion-counting data on DNA [Chu et al. 2007] and
  model studies [Silalahi et al. 2010] use 2–16 Å. The Stern layer
  (`ion_radius`) still applies; set it to 0 to let the lattice alone keep
  the ions away.
* **APBS.** APBS 3.4.1 disables its `smpbe`, so this cannot be compared
  with APBS; the [validation record](../VALIDATION) checks it against exact
  planar solutions and published scaling laws.

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

### Surface, volume and dispersion

{py:func}`pbsolve.apolar_energy` computes the non-polar model of APBS `apolar`
[Wagoner & Baker 2006]:

$$
G_\text{np} = \gamma A + p V + \bar\rho \sum_i \int u_i^\text{att}(y)\,\theta(y)\,\mathrm{d}y .
$$

A is the solvent-accessible surface area and V the solvent-accessible volume
(spheres of the atoms' radii plus `srad`). The integral is the attractive
part of each atom's Lennard-Jones interaction with water, ε_iw[(σ/r)¹² −
2(σ/r)⁶] beyond σ = R_i + R_w and −ε_iw inside (ε_iw = √(ε_i ε_w), the
Weeks–Chandler–Andersen split), over the solvent within 14 Å; ρ̄ is the
number density of water. The volume term models the cost of the cavity, the
dispersion term the solvent's attraction, which the area alone cannot
reproduce [Wagoner & Baker 2006].

The function needs Lennard-Jones radii R_min/2 and well depths per atom
(they also define A and V, as in APBS).
{py:func}`pbsolve.openmm_bridge.lennard_jones_from_system` takes them from an
OpenMM system, which also covers Amber prmtop files;
{py:func}`pbsolve.apbs.read_parm` reads APBS parameter files. The defaults are
the alkane protocol of Wagoner & Baker that APBS ships as an example (γ =
0.0085 kJ/(mol·Å²), p = 0.2394 kJ/(mol·Å³), ρ̄ = 0.033428 Å⁻³, `srad` 0.65 Å,
TIP3P water), converted to kcal/mol, except the integration grid: 0.5 Å
instead of 0.3 Å, which changes 1YCR's energy by 0.05 % and is four times
faster. These parameters were fitted to alkanes, not to proteins.

```python
# Ethane as two united atoms 1.53 Å apart, with AMBER's CT radius and well depth
g = pbsolve.apolar_energy([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]], [1.908, 1.908], [0.1094, 0.1094])
print(f"{g.energy:.2f} kcal/mol = {g.surface:.2f} + {g.pressure_volume:.2f} {g.dispersion:+.2f}")
```

The three terms equal APBS' to the digits it prints, on APBS' own alkane
examples and on a protein ([validation](../VALIDATION)), and
{py:func}`pbsolve.apbs.apolar_energy` runs APBS on the same input. Each term
is computed only when its coefficient is not zero; for 1YCR (1670 atoms) the
calculation takes 0.75 s on 18 CPU threads (APBS: 13 s).

## Forces

{py:meth}`~pbsolve.Solver.polar_forces` computes ΔG_polar and the force
−dΔG_polar/dr on every atom (kcal/mol/Å), as APBS `calcforce`. It needs a
spline surface:

```python
smooth = pbsolve.Solver(ionic_strength=0.15, surface="spl2", chgm="spl2")
f = smooth.polar_forces(coords, charges, radii)
print(f"ΔG_polar {f.energy:.2f} kcal/mol, largest force {abs(f.forces).max():.2f} kcal/mol/Å")
```

The free energy is a functional of the potential that is stationary at the
solution, so the force is its derivative at the converged potential, in three
parts [Gilson et al. 1993; Im, Beglov & Roux 1998]:

| part | attribute | origin |
|---|---|---|
| charge–field | `f.qf` | the reaction field at the charges (the derivative of the charge-assignment weights) |
| dielectric boundary | `f.db` | the field's pressure on the moving dielectric boundary |
| ionic boundary | `f.ib` | the ions' osmotic pressure on the moving ion-accessibility boundary |

`f.forces` is their sum, `f.result` the energy calculation.

* **Exactness.** Each part is the solvated state's minus the reference
  state's, from the finest grid level. As in APBS the coarser levels and the
  boundary values do not contribute. On one grid with `bcfl="zero"` the
  forces are the exact gradient of the discrete energy (to 4e-9); with
  `mdh` the neglected boundary term is ~1e-5 of the largest force.
* **APBS.** The forces agree with APBS 3.4.1 to ≤ 4e-7 of the largest force
  (≤ 6e-10 with `bcfl zero`), for `lpbe` and `npbe`; `smpbe` works as well.
* **Accuracy.** The charge–field part is accurate to a few per cent. The
  dielectric-boundary part carries the grid's noise: the net force on an
  isolated molecule, zero in the continuum, scatters by 5–10 % of the
  boundary force over grid placements. A spacing well below 2·`swin`
  reduces it.
* **Grid.** Every atom's boundary force needs the finest grid to extend
  radius + `swin` + `srad` and one node beyond the atom; pbsolve fails
  rather than dropping the force, as APBS does.

The [validation record](../VALIDATION) has the derivation and the checks
(finite differences, APBS, a charge in a dielectric sphere).
