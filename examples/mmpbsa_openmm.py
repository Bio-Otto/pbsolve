"""Add PB solvation to an OpenMM-based MM-GBSA pipeline (single-trajectory MM-PBSA).

Requirements: pbsolve, openmm, mdtraj (only for reading the trajectory).
Inputs: a solute-only Amber topology (no water/ions) and a stripped trajectory.
"""

import mdtraj as md
import numpy as np
import openmm.app as app

import pbsolve
from pbsolve.openmm_bridge import OpenMMPBSA, charges_from_system, lennard_jones_from_system

prmtop = app.AmberPrmtopFile("complex.prmtop")
system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)
ligand = [a.index for a in prmtop.topology.atoms() if a.residue.name == "LIG"]

solver = pbsolve.Solver(platform="auto", grid_spacing=0.5, ionic_strength=0.15)
print(f"PB solver running on {solver.platform}: {solver.device_name}")

pbsa = OpenMMPBSA(system, prmtop.topology, ligand_atoms=ligand, solver=solver, radii="mbondi2")
# Coulomb + Lennard-Jones parameters for the receptor–ligand interaction energy (interaction entropy)
charges = charges_from_system(system)
lj_radii, lj_epsilons = lennard_jones_from_system(system)

traj = md.load("complex.dcd", top="complex.prmtop")
rows = []
e_int = []
for xyz_nm in traj.xyz:
    xyz = xyz_nm * 10.0  # nm -> Å
    terms = pbsa.frame(xyz)
    coulomb, vdw = pbsolve.interaction_energy(xyz, charges, lj_radii, lj_epsilons, ligand)
    e_int.append(coulomb + vdw)
    # Your existing tool supplies the gas-phase term from OpenMM:
    # dG = delta_E_MM(frame) + terms.delta_solvation  (− TΔS below)
    rows.append((terms.delta_polar, terms.delta_nonpolar))

rows = np.array(rows)
print(f"<ΔG_polar>    = {rows[:, 0].mean():8.2f} ± {rows[:, 0].std(ddof=1) / np.sqrt(len(rows)):.2f} kcal/mol")
print(f"<ΔG_nonpolar> = {rows[:, 1].mean():8.2f} ± {rows[:, 1].std(ddof=1) / np.sqrt(len(rows)):.2f} kcal/mol")

# −TΔS from the fluctuations of the interaction energy (Duan et al. 2016). Report σ and both
# estimates; above σ = 15 kJ/mol IE does not converge and C2 is the one to use (Ekberg & Ryde 2021).
s = pbsolve.interaction_entropy(e_int, temperature=298.15)
print(f"<E_int>       = {np.mean(e_int):8.2f} kcal/mol, σ = {s.sigma:.2f}")
print(f"-TΔS          = IE {s.ie:.2f} ± {s.ie_sem:.2f}, C2 {s.c2:.2f} ± {s.c2_sem:.2f} kcal/mol"
      + ("" if s.reliable else " (σ too large for IE: use C2)"))
