"""Add PB solvation to an OpenMM-based MM-GBSA pipeline (single-trajectory MM-PBSA).

Requirements: pbsolve, openmm, mdtraj (only for reading the trajectory).
Inputs: a solute-only Amber topology (no water/ions) and a stripped trajectory.
"""

import mdtraj as md
import numpy as np
import openmm.app as app

import pbsolve
from pbsolve.openmm_bridge import OpenMMPBSA

prmtop = app.AmberPrmtopFile("complex.prmtop")
system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)
ligand = [a.index for a in prmtop.topology.atoms() if a.residue.name == "LIG"]

solver = pbsolve.Solver(platform="auto", grid_spacing=0.5, ionic_strength=0.15)
print(f"PB solver running on {solver.platform}: {solver.device_name}")

pbsa = OpenMMPBSA(system, prmtop.topology, ligand_atoms=ligand, solver=solver, radii="mbondi2")

traj = md.load("complex.dcd", top="complex.prmtop")
rows = []
for xyz_nm in traj.xyz:
    terms = pbsa.frame(xyz_nm * 10.0)  # nm -> Å
    # Your existing tool supplies the gas-phase term from OpenMM:
    # dG = delta_E_MM(frame) + terms.delta_solvation
    rows.append((terms.delta_polar, terms.delta_nonpolar))

rows = np.array(rows)
print(f"<ΔG_polar>    = {rows[:, 0].mean():8.2f} ± {rows[:, 0].std(ddof=1) / np.sqrt(len(rows)):.2f} kcal/mol")
print(f"<ΔG_nonpolar> = {rows[:, 1].mean():8.2f} ± {rows[:, 1].std(ddof=1) / np.sqrt(len(rows)):.2f} kcal/mol")
