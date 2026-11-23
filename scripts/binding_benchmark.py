#!/usr/bin/env python
"""Protein–ligand binding benchmark for pbsolve (ROADMAP 4.19).

Single-structure MM-PBSA on the poses of the Open Force Field
protein-ligand-benchmark (github.com/openforcefield/protein-ligand-benchmark;
code MIT, data CC BY 4.0) at a fixed commit, compared with the experimental
affinities that come with it. The statistics follow Hahn et al. (2022): per
target, Pearson R, Spearman ρ and Kendall τ with bootstrap 95 % intervals.

    python scripts/binding_benchmark.py fetch DATA
    python scripts/binding_benchmark.py prepare DATA WORK [--targets tyk2 p38] [--jobs 12]
    python scripts/binding_benchmark.py run WORK [--protocols eps1 eps4] [--platform cpu]
    python scripts/binding_benchmark.py apbs WORK mcl1 [--protocol eps2]
    python scripts/binding_benchmark.py report WORK

``prepare`` needs OpenMM and AmberTools (antechamber, parmchk2, tleap), ``run``
needs pbsolve, ``apbs`` pbsolve and APBS 3.4.1, ``report`` SciPy. Every stage skips work whose
output already exists, so an interrupted run continues where it stopped.

Protocol. Receptor: ``protein.pdb`` as provided (hydrogens kept, crystal waters
removed), Amber ff14SB, metal ions with OpenMM's amber14 ion parameters,
cofactors GAFF2 with AM1-BCC charges. Ligands: GAFF2 with AM1-BCC charges
(antechamber/sqm), net charge from the SDF. No minimisation, no entropy term:
ΔG = ΔE_MM + ΔΔG_polar + ΔΔG_nonpolar on one structure, where ΔE_MM is the
receptor–ligand Coulomb energy divided by the interior dielectric plus the
Lennard-Jones energy (bonded terms cancel). Dielectric radii mbondi2, ionic
strength 0.15 M. All PB calculations of one target share one grid, built
around the receptor and every ligand, so the receptor is solved once per
protocol. cdk2 is left out: its phosphothreonine has no template in OpenMM's
ff14SB files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

COMMIT = "fd88824f9114244f95a14b485e6d6c96c1de716d"
RAW = "https://raw.githubusercontent.com/openforcefield/protein-ligand-benchmark/{commit}/data/{target}/{path}"
TARGETS = (
    "cdk8", "cmet", "eg5", "hif2a", "mcl1", "p38", "pde2",
    "pfkfb3", "ptp1b", "shp2", "syk", "thrombin", "tnks2", "tyk2",
)
FILES = ("00_data/ligands.yml", "00_data/target.yml", "01_protein/crd/protein.pdb", "02_ligands/ligands.sdf")
COFACTORS = "01_protein/crd/cofactor.sdf"

GAS_CONSTANT = 8.314462618e-3 / 4.184  # kcal/(mol K), CODATA 2018
TEMPERATURE = 298.15
CONCENTRATION_UNITS = {"nM": 1e-9, "uM": 1e-6}

IONIC_STRENGTH = 0.15
SOLVENT_DIELECTRIC = 78.54
MINIMISATION_CUTOFF = 12.0  # Å
MINIMISATION_RESTRAINT = 10.0  # kcal/(mol Å²) on heavy atoms, E = ½ k d²
STRUCTURES = ("minimized", "raw")
RESIDUE_DIELECTRICS = (2.7, 1.1)  # charged, other residues: Liu, Peng & Zhang 2019 (fitted for GB)
PROTOCOLS = {
    "eps1": {"pdie": 1.0},
    "eps2": {"pdie": 2.0},
    "eps4": {"pdie": 4.0},
    "eps1-fractional": {"pdie": 1.0, "interface": "fractional"},
    "eps2-fractional": {"pdie": 2.0, "interface": "fractional"},
    "eps4-fractional": {"pdie": 4.0, "interface": "fractional"},
    "gauss": {"surface": "gauss", "pdie": 1.0},
    "residue-eps": {"pdie": 1.0},
}
# Protocols whose electrostatic binding energy is Δ(reference) + ΔΔG_polar from the grid, not Coulomb/pdie:
# atom dielectrics, where every charge keeps the faces around it in all three species. Not the Gaussian
# dielectric: the neighbours' Gaussians change ε next to each charge, so the grid self-energies do not
# cancel in Δ(reference) (only in ΔΔG_polar); its binding energy is Coulomb/pdie + ΔΔG_polar (Li et al.).
GRID_COULOMB = ("residue-eps",)


# ---------------------------------------------------------------- input files

def experimental_dg(text: str) -> dict[str, float]:
    """ΔG_exp (kcal/mol) of every ligand in ``ligands.yml`` from its primary measurement.

    Ki and IC50 are both taken as dissociation constants, ΔG = RT ln(K / 1 M),
    as the benchmark itself does; pIC50 gives −RT ln(10) pIC50.
    """
    out: dict[str, float] = {}
    name, fields, inside, indent = None, {}, False, None

    def finish() -> None:
        if name is None:
            return
        kind, value = fields.get("type"), float(fields["value"])
        rt = GAS_CONSTANT * TEMPERATURE
        if kind in ("ki", "ic50"):
            out[name] = rt * math.log(value * CONCENTRATION_UNITS[fields["unit"]])
        elif kind == "pic50":
            out[name] = -rt * math.log(10.0) * value
        else:
            raise ValueError(f"{name}: unknown measurement type {kind!r}")

    for line in text.splitlines():
        stripped = line.lstrip(" ")
        depth = len(line) - len(stripped)
        if not stripped:
            continue
        if depth == 0:
            finish()
            name, fields, inside, indent = line.rstrip(":"), {}, False, None
        elif depth == 2:
            inside, indent = stripped == "measurement:", None
        elif inside:
            indent = depth if indent is None else indent
            # keys of the measurement itself; "additional:" lists (dg, dh, tds) are deeper or start with "- "
            if depth == indent and not stripped.startswith("- ") and ":" in stripped:
                key, value = stripped.split(":", 1)
                fields[key] = value.strip().strip("'\"")
    finish()
    return out


def sdf_molecules(text: str) -> list[dict]:
    """Name, mol block, coordinates (Å), elements and net charge of every molecule of a V2000 SDF file."""
    out = []
    for block in text.split("$$$$"):
        block = block[1:] if block.startswith("\n") else block
        if not block.strip():
            continue
        lines = block.splitlines()
        if "V2000" not in lines[3]:
            raise ValueError(f"{lines[0]!r}: only V2000 mol blocks are supported")
        n = int(lines[3][0:3])
        atoms = lines[4:4 + n]
        xyz = np.array([[float(a[0:10]), float(a[10:20]), float(a[20:30])] for a in atoms])
        elements = [a[31:34].strip() for a in atoms]
        charge = 0
        for line in lines:
            if line.startswith("M  CHG"):
                f = line.split()[3:]
                charge += sum(int(f[i + 1]) for i in range(0, len(f), 2))
        # the structure's formal charges decide; the files' Epik charge property is stale for 25 ligands
        out.append({"name": lines[0].strip(), "block": block, "xyz": xyz, "elements": elements, "charge": charge})
    return out


def fetch(data: str) -> None:
    """Download the files the benchmark needs at the pinned commit."""
    for target in TARGETS:
        for path in FILES + (COFACTORS,):
            dest = os.path.join(data, target, path)
            if os.path.exists(dest):
                continue
            try:
                with urllib.request.urlopen(RAW.format(commit=COMMIT, target=target, path=path), timeout=60) as r:
                    body = r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404 and path == COFACTORS:
                    continue
                raise
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(body)
        print(f"{target}: done")


# ---------------------------------------------------------------- parameters

def _run(cmd: list[str], cwd: str) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}:\n{(r.stdout + r.stderr)[-2000:]}")


def gaff(directory: str, mol: dict, resname: str) -> str:
    """GAFF2/AM1-BCC Amber topology of one molecule (cached); returns the prmtop path."""
    prmtop = os.path.join(directory, "mol.prmtop")
    if os.path.exists(prmtop):
        return prmtop
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "mol.sdf"), "w") as f:
        f.write(mol["block"] + "$$$$\n")
    _run(["antechamber", "-i", "mol.sdf", "-fi", "mdl", "-o", "mol.mol2", "-fo", "mol2", "-c", "bcc",
          "-nc", str(mol["charge"]), "-at", "gaff2", "-rn", resname, "-s", "0", "-pf", "y"], directory)
    _run(["parmchk2", "-i", "mol.mol2", "-f", "mol2", "-o", "mol.frcmod", "-s", "gaff2"], directory)
    with open(os.path.join(directory, "leap.in"), "w") as f:
        f.write("source leaprc.gaff2\nm = loadmol2 mol.mol2\nloadamberparams mol.frcmod\n"
                "saveamberparm m mol.tmp.prmtop mol.inpcrd\nquit\n")
    _run(["tleap", "-f", "leap.in"], directory)
    os.replace(os.path.join(directory, "mol.tmp.prmtop"), prmtop)
    return prmtop


def _arrays(system, topology) -> dict:
    from pbsolve.openmm_bridge import charges_from_system, lennard_jones_from_system, radii_from_topology

    ljr, lje = lennard_jones_from_system(system)
    return {"q": charges_from_system(system), "r": radii_from_topology(topology, "mbondi2"), "ljr": ljr, "lje": lje}


def molecule_arrays(prmtop: str, mol: dict) -> dict:
    import openmm.app as app

    top = app.AmberPrmtopFile(prmtop)
    elements = [a.element.symbol for a in top.topology.atoms()]
    if elements != mol["elements"]:
        raise ValueError(f"{prmtop}: atoms {elements} do not match the SDF {mol['elements']}")
    out = _arrays(top.createSystem(nonbondedMethod=app.NoCutoff), top.topology)
    excess = out["q"].sum() - mol["charge"]
    if abs(excess) > 0.01:
        raise ValueError(f"{prmtop}: charges add up to {out['q'].sum():.4f}, expected {mol['charge']}")
    out["q"] = out["q"] - excess / len(out["q"])  # antechamber's rounding leaves up to ~1e-3 e
    out["xyz"] = mol["xyz"]
    return out


def _receptor_model(pdb_path: str):
    import openmm.app as app

    pdb = app.PDBFile(pdb_path)
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    return model, app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")


def receptor_arrays(pdb_path: str) -> dict:
    import openmm.app as app
    from pbsolve.openmm_bridge import positions_angstrom

    model, ff = _receptor_model(pdb_path)
    out = _arrays(ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff), model.topology)
    out["xyz"] = positions_angstrom(model.positions)
    out["labels"] = [f"{a.residue.chain.id}:{a.residue.name}:{a.residue.id}" for a in model.topology.atoms()]
    return out


def prepare(data: str, work: str, targets: list[str], jobs: int) -> None:
    inputs, tasks = {}, []
    for target in targets:
        out = os.path.join(work, target, "system.npz")
        if os.path.exists(out):
            continue
        d = os.path.join(data, target)
        with open(os.path.join(d, "02_ligands", "ligands.sdf")) as f:
            ligands = sdf_molecules(f.read())
        cofactor_path = os.path.join(d, COFACTORS)
        cofactors = []
        if os.path.exists(cofactor_path):
            with open(cofactor_path) as f:
                cofactors = sdf_molecules(f.read())
        with open(os.path.join(d, "00_data", "ligands.yml")) as f:
            dg = experimental_dg(f.read())
        missing = [m["name"] for m in ligands if m["name"] not in dg]
        if missing:
            raise ValueError(f"{target}: no measurement for {missing}")
        inputs[target] = (ligands, cofactors, dg)
        for i, mol in enumerate(cofactors):
            tasks.append((os.path.join(work, target, "cofactors", str(i)), mol, f"CF{i}"))
        for mol in ligands:
            tasks.append((os.path.join(work, target, "ligands", mol["name"]), mol, "LIG"))
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for path in pool.map(lambda t: gaff(*t), tasks):
            print(f"  {os.path.relpath(path, work)}", flush=True)

    for target, (ligands, cofactors, dg) in inputs.items():
        rec = receptor_arrays(os.path.join(data, target, "01_protein", "crd", "protein.pdb"))
        for i, mol in enumerate(cofactors):
            c = molecule_arrays(os.path.join(work, target, "cofactors", str(i), "mol.prmtop"), mol)
            for k in ("xyz", "q", "r", "ljr", "lje"):
                rec[k] = np.concatenate([rec[k], c[k]])
            rec["labels"] += [f"X:CF{i}:{i + 1}"] * len(c["q"])
        ligs = [molecule_arrays(os.path.join(work, target, "ligands", m["name"], "mol.prmtop"), m) for m in ligands]
        offsets = np.cumsum([0] + [len(lig["q"]) for lig in ligs])
        np.savez(
            os.path.join(work, target, "system.npz"),
            rec_xyz=rec["xyz"], rec_q=rec["q"], rec_r=rec["r"], rec_ljr=rec["ljr"], rec_lje=rec["lje"],
            rec_labels=np.array(rec["labels"]),
            lig_names=np.array([m["name"] for m in ligands]), lig_offsets=offsets,
            lig_xyz=np.concatenate([lig["xyz"] for lig in ligs]),
            **{f"lig_{k}": np.concatenate([lig[k] for lig in ligs]) for k in ("q", "r", "ljr", "lje")},
            dg_exp=np.array([dg[m["name"]] for m in ligands]),
        )
        print(f"{target}: receptor {len(rec['q'])} atoms (charge {rec['q'].sum():+.2f}), {len(ligs)} ligands")


# ---------------------------------------------------------------- minimisation

def _merge(systems: list):
    """One OpenMM system from several (bonds, angles, torsions, non-bonded terms with their exceptions)
    with a 12 Å cutoff and a reaction field of the solvent's dielectric constant."""
    import openmm

    out = openmm.System()
    bonds, angles, torsions = openmm.HarmonicBondForce(), openmm.HarmonicAngleForce(), openmm.PeriodicTorsionForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(MINIMISATION_CUTOFF / 10.0)
    nb.setReactionFieldDielectric(SOLVENT_DIELECTRIC)
    offset = 0
    for s in systems:
        for i in range(s.getNumParticles()):
            out.addParticle(s.getParticleMass(i))
        for f in s.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                for k in range(f.getNumBonds()):
                    i, j, r0, kb = f.getBondParameters(k)
                    bonds.addBond(i + offset, j + offset, r0, kb)
            elif isinstance(f, openmm.HarmonicAngleForce):
                for k in range(f.getNumAngles()):
                    i, j, m, a0, ka = f.getAngleParameters(k)
                    angles.addAngle(i + offset, j + offset, m + offset, a0, ka)
            elif isinstance(f, openmm.PeriodicTorsionForce):
                for k in range(f.getNumTorsions()):
                    i, j, m, n, periodicity, phase, kt = f.getTorsionParameters(k)
                    torsions.addTorsion(i + offset, j + offset, m + offset, n + offset, periodicity, phase, kt)
            elif isinstance(f, openmm.NonbondedForce):
                for k in range(f.getNumParticles()):
                    nb.addParticle(*f.getParticleParameters(k))
                for k in range(f.getNumExceptions()):
                    i, j, qq, sigma, eps = f.getExceptionParameters(k)
                    nb.addException(i + offset, j + offset, qq, sigma, eps)
            elif not isinstance(f, openmm.CMMotionRemover):
                raise ValueError(f"cannot merge a {type(f).__name__}")
        offset += s.getNumParticles()
    for f in (bonds, angles, torsions, nb):
        out.addForce(f)
    return out


def minimize(data: str, work: str, targets: list[str], threads: int) -> None:
    """Restrained minimisation of every complex; writes ``minimized.npz`` next to ``system.npz``."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    method = {"nonbondedMethod": app.CutoffNonPeriodic, "nonbondedCutoff": MINIMISATION_CUTOFF * unit.angstrom}
    for target in targets:
        out = os.path.join(work, target, "minimized.npz")
        if os.path.exists(out) or not os.path.exists(os.path.join(work, target, "system.npz")):
            continue
        rec, ligs, names, _ = _system(work, target)
        model, ff = _receptor_model(os.path.join(data, target, "01_protein", "crd", "protein.pdb"))
        parts = [ff.createSystem(model.topology, **method)]
        heavy = [a.element is not None and a.element.symbol != "H" for a in model.topology.atoms()]
        i = 0
        while os.path.exists(os.path.join(work, target, "cofactors", str(i), "mol.prmtop")):
            top = app.AmberPrmtopFile(os.path.join(work, target, "cofactors", str(i), "mol.prmtop"))
            parts.append(top.createSystem(**method))
            heavy += [a.element.symbol != "H" for a in top.topology.atoms()]
            i += 1
        if len(heavy) != len(rec["q"]):
            raise ValueError(f"{target}: {len(heavy)} receptor atoms in the topologies, {len(rec['q'])} in system.npz")
        k = MINIMISATION_RESTRAINT * 4.184 * 100.0  # kcal/(mol Å²) -> kJ/(mol nm²)
        xyz, stats = [], []
        t0 = time.perf_counter()
        for name, lig in zip(names, ligs):
            top = app.AmberPrmtopFile(os.path.join(work, target, "ligands", name, "mol.prmtop"))
            system = _merge(parts + [top.createSystem(**method)])
            mask = np.array(heavy + [a.element.symbol != "H" for a in top.topology.atoms()])
            x0 = np.concatenate([rec["xyz"], lig["xyz"]])
            restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            restraint.addGlobalParameter("k", k)
            for p in ("x0", "y0", "z0"):
                restraint.addPerParticleParameter(p)
            for j in np.nonzero(mask)[0]:
                restraint.addParticle(int(j), (x0[j] / 10.0).tolist())
            system.addForce(restraint)
            context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                     openmm.Platform.getPlatformByName("CPU"), {"Threads": str(threads)})
            context.setPositions(x0 / 10.0)
            before = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            openmm.LocalEnergyMinimizer.minimize(context, 10.0, 0)
            state = context.getState(getEnergy=True, getPositions=True)
            x = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            after = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            d = np.linalg.norm(x - x0, axis=1)
            stats.append([before, after, float(np.sqrt(np.mean(d[mask] ** 2))), float(d[mask].max()), float(d.max())])
            xyz.append(x)
        np.savez(out, cpx_xyz=np.concatenate(xyz), cpx_offsets=np.cumsum([0] + [len(x) for x in xyz]),
                 stats=np.array(stats))
        s = np.array(stats)
        print(f"{target}: {len(names)} complexes in {time.perf_counter() - t0:.0f} s; heavy-atom RMSD median "
              f"{np.median(s[:, 2]):.3f} Å (max {s[:, 2].max():.3f}), largest heavy-atom shift {s[:, 3].max():.2f} Å",
              flush=True)


# ---------------------------------------------------------------- MM-PBSA

def _system(work: str, target: str):
    s = np.load(os.path.join(work, target, "system.npz"))
    rec = {k: s[f"rec_{k}"] for k in ("xyz", "q", "r", "ljr", "lje", "labels")}
    o = s["lig_offsets"]
    ligs = [{k: s[f"lig_{k}"][o[i]:o[i + 1]] for k in ("xyz", "q", "r", "ljr", "lje")} for i in range(len(o) - 1)]
    return rec, ligs, [str(n) for n in s["lig_names"]], s["dg_exp"]


def complexes(work: str, target: str, structures: str) -> list[np.ndarray]:
    """Coordinates of every complex (receptor first): the poses as given, or after :func:`minimize`."""
    rec, ligs, _, _ = _system(work, target)
    if structures == "raw":
        return [np.concatenate([rec["xyz"], lig["xyz"]]) for lig in ligs]
    s = np.load(os.path.join(work, target, "minimized.npz"))
    o = s["cpx_offsets"]
    return [s["cpx_xyz"][o[i]:o[i + 1]] for i in range(len(o) - 1)]


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def _complex(rec: dict, lig: dict, key: str) -> np.ndarray:
    return np.concatenate([rec[key], lig[key]])


def run(work: str, targets: list[str], protocols: list[str], platform: str, spacing: float,
        workers: int | None, structures: str) -> None:
    import pbsolve

    for target in targets:
        needed = "system.npz" if structures == "raw" else "minimized.npz"
        if not os.path.exists(os.path.join(work, target, needed)):
            print(f"{target}: no {needed}, skipped")
            continue
        rec, ligs, names, dg_exp = _system(work, target)
        cpx = complexes(work, target, structures)
        n_rec = len(rec["q"])
        # the poses as given share one receptor structure: its terms are computed once
        shared = structures == "raw"
        outdir = os.path.join(work, target, structures)
        os.makedirs(outdir, exist_ok=True)
        terms = os.path.join(outdir, "terms.csv")
        if not os.path.exists(terms):
            np_rec = pbsolve.nonpolar_energy(rec["xyz"], rec["r"]) if shared else None
            ap_rec = pbsolve.apolar_energy(rec["xyz"], rec["ljr"], rec["lje"]).energy if shared else None
            rows = []
            for name, dg, lig, xyz in zip(names, dg_exp, ligs, cpx):
                q, r = _complex(rec, lig, "q"), _complex(rec, lig, "r")
                ljr, lje = _complex(rec, lig, "ljr"), _complex(rec, lig, "lje")
                coulomb, lj = pbsolve.interaction_energy(xyz, q, ljr, lje, range(n_rec, len(xyz)))
                rx, lx = xyz[:n_rec], xyz[n_rec:]
                nonpolar = (pbsolve.nonpolar_energy(xyz, r) - pbsolve.nonpolar_energy(lx, lig["r"])
                            - (np_rec if shared else pbsolve.nonpolar_energy(rx, rec["r"])))
                apolar = (pbsolve.apolar_energy(xyz, ljr, lje).energy
                          - pbsolve.apolar_energy(lx, lig["ljr"], lig["lje"]).energy
                          - (ap_rec if shared else pbsolve.apolar_energy(rx, rec["ljr"], rec["lje"]).energy))
                rows.append([name, f"{dg:.6f}", f"{coulomb:.6f}", f"{lj:.6f}", f"{nonpolar:.6f}", f"{apolar:.6f}"])
            _write_csv(terms, ["ligand", "dg_exp", "coulomb", "lj", "nonpolar", "apolar"], rows)

        xyz_all = np.concatenate(cpx)
        r_all = np.concatenate([_complex(rec, lig, "r") for lig in ligs])
        for protocol in protocols:
            out = os.path.join(outdir, f"{protocol}.csv")
            if os.path.exists(out):
                continue
            solver = pbsolve.Solver(platform=platform, grid_spacing=spacing, ionic_strength=IONIC_STRENGTH,
                                    **PROTOCOLS[protocol])
            grid = solver.make_grid(xyz_all, r_all)
            eps_rec = eps_cpx = None
            if protocol == "residue-eps":
                charged, neutral = RESIDUE_DIELECTRICS
                eps_rec = pbsolve.residue_dielectrics(rec["q"], list(rec["labels"]), charged=charged, neutral=neutral)
            systems, dielectrics = [], []
            if shared:
                systems.append((rec["xyz"], rec["q"], rec["r"]))
                dielectrics.append(eps_rec)
            for lig, xyz in zip(ligs, cpx):
                q, r = _complex(rec, lig, "q"), _complex(rec, lig, "r")
                if protocol == "residue-eps":
                    labels = list(rec["labels"]) + ["L:LIG:1"] * len(lig["q"])
                    eps_cpx = pbsolve.residue_dielectrics(q, labels, charged=charged, neutral=neutral)
                systems.append((xyz, q, r))
                dielectrics.append(eps_cpx)
                if not shared:
                    systems.append((xyz[:n_rec], rec["q"], rec["r"]))
                    dielectrics.append(eps_rec)
                systems.append((xyz[n_rec:], lig["q"], lig["r"]))
                dielectrics.append(None if eps_cpx is None else eps_cpx[n_rec:])
            t0 = time.perf_counter()
            res = solver.polar_solvation_many(systems, grid=grid, workers=workers,
                                              dielectrics=dielectrics if protocol == "residue-eps" else None)
            seconds = time.perf_counter() - t0
            kt = pbsolve.kt_kcal_per_mol(solver.params["temperature"])
            rows = []
            for i, name in enumerate(names):
                c, rr, lg = (res[1 + 2 * i], res[0], res[2 + 2 * i]) if shared else res[3 * i:3 * i + 3]
                polar = c.energy - rr.energy - lg.energy
                reference = (c.reference_kt - rr.reference_kt - lg.reference_kt) * kt
                rows.append([name, f"{polar:.6f}", f"{reference:.6f}"])
            with open(os.path.join(outdir, f"{protocol}.time"), "w") as f:
                f.write(f"{seconds:.3f} {len(systems)} {solver.platform} {'x'.join(map(str, grid.dims))}\n")
            _write_csv(out, ["ligand", "delta_polar", "delta_reference"], rows)
            print(f"{target} {structures} {protocol}: {len(systems)} calculations in {seconds:.1f} s "
                  f"on {solver.platform}", flush=True)


def apbs(work: str, target: str, protocol: str, spacing: float, structures: str) -> None:
    """ΔΔG_polar of every complex from APBS 3.4.1 and pbsolve, each complex on its own APBS grid."""
    import shutil

    import pbsolve
    from pbsolve.apbs import compare_binding

    out = os.path.join(work, target, structures, f"apbs-{protocol}.csv")
    if os.path.exists(out):
        return
    rec, ligs, names, _ = _system(work, target)
    solver = pbsolve.Solver(platform="cpu", grid_spacing=spacing, ionic_strength=IONIC_STRENGTH,
                            dime_policy="apbs", **PROTOCOLS[protocol])
    rows = []
    for name, lig, xyz in zip(names, ligs, complexes(work, target, structures)):
        d = os.path.join(work, target, "apbs", name)
        os.makedirs(d, exist_ok=True)
        q, r = _complex(rec, lig, "q"), _complex(rec, lig, "r")
        pqr = os.path.join(d, "complex.pqr")
        with open(pqr, "w") as f:
            for i, (p, c, s) in enumerate(zip(xyz, q, r), 1):
                f.write(f"ATOM {i} X {'LIG' if i > len(rec['q']) else 'REC'} 1 "
                        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c:.6f} {s:.4f}\n")
        rep = compare_binding(pqr, solver, np.arange(len(rec["q"]), len(q)), workdir=d, maps=False, inject=False)
        ddg = rep.ddg
        rows.append([name, f"{ddg['apbs']:.6f}", f"{ddg['pbsolve']:.6f}", f"{ddg['diff_unit_corrected']:.6f}",
                     int(rep.passed())])
        shutil.rmtree(d)
        print(f"{target} {name}: APBS {ddg['apbs']:.4f}, pbsolve {ddg['pbsolve']:.4f} kcal/mol", flush=True)
    _write_csv(out, ["ligand", "ddg_apbs", "ddg_pbsolve", "diff_apbs_units", "passed"], rows)


# ---------------------------------------------------------------- statistics

def correlations(x: np.ndarray, y: np.ndarray, *, resamples: int = 1000, seed: int = 0) -> dict:
    """Pearson R, Spearman ρ, Kendall τ and their bootstrap 95 % percentile intervals."""
    from scipy import stats

    def three(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        return (stats.pearsonr(a, b)[0], stats.spearmanr(a, b)[0], stats.kendalltau(a, b)[0])

    value = three(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(resamples):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boot.append(three(x[i], y[i]))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    return {k: (value[j], lo[j], hi[j]) for j, k in enumerate(("pearson", "spearman", "kendall"))}


def _read(path: str) -> dict[str, dict[str, float]]:
    with open(path) as f:
        return {r["ligand"]: {k: float(v) for k, v in r.items() if k != "ligand"} for r in csv.DictReader(f)}


def binding_energies(terms: dict, polar: dict, protocol: str, nonpolar: str = "nonpolar") -> dict[str, float]:
    """ΔG_calc of every ligand: electrostatics + Lennard-Jones + ΔΔG_polar + non-polar term."""
    out = {}
    for name, t in terms.items():
        p = polar[name]
        if protocol in GRID_COULOMB:
            elec = p["delta_reference"]
        else:
            elec = t["coulomb"] / PROTOCOLS[protocol]["pdie"]
        out[name] = elec + t["lj"] + p["delta_polar"] + t[nonpolar]
    return out


def report(work: str, targets: list[str], structures: str) -> None:
    rows, timing, consistency, apbs_rows = [], [], [], []
    for target in targets:
        base = os.path.join(work, target, structures)
        terms_path = os.path.join(base, "terms.csv")
        if not os.path.exists(terms_path):
            continue
        terms = _read(terms_path)
        names = list(terms)
        exp = np.array([terms[n]["dg_exp"] for n in names])
        for protocol in PROTOCOLS:
            path = os.path.join(base, f"{protocol}.csv")
            if not os.path.exists(path):
                continue
            polar = _read(path)
            with open(os.path.join(base, f"{protocol}.time")) as f:
                seconds, calcs = f.read().split()[:2]
            timing.append((target, protocol, float(seconds), int(calcs)))
            if protocol not in GRID_COULOMB and "surface" not in PROTOCOLS[protocol]:
                # the grid's reference-state difference is the Coulomb energy / pdie up to discretisation
                ref = np.array([polar[n]["delta_reference"] for n in names])
                coul = np.array([terms[n]["coulomb"] for n in names]) / PROTOCOLS[protocol]["pdie"]
                consistency.append((target, protocol, float(np.max(np.abs(ref - coul)))))
            variants = [(protocol, "nonpolar")]
            if protocol in ("eps1", "eps2", "eps4"):
                variants.append((f"{protocol}+apolar", "apolar"))
            for label, nonpolar in variants:
                calc = binding_energies(terms, polar, protocol, nonpolar)
                y = np.array([calc[n] for n in names])
                rows.append((target, label, len(names), correlations(y, exp)))
            path = os.path.join(base, f"apbs-{protocol}.csv")
            if os.path.exists(path):
                # the same protocol with each complex on its own APBS grid, solved by APBS and by pbsolve
                own = _read(path)
                apbs_rows.append((target, protocol, max(abs(v["diff_apbs_units"]) for v in own.values()),
                                  sum(int(v["passed"]) for v in own.values()), len(own)))
                for code in ("apbs", "pbsolve"):
                    alt = {n: {"delta_polar": own[n][f"ddg_{code}"], "delta_reference": 0.0} for n in names}
                    y = np.array([binding_energies(terms, alt, protocol)[n] for n in names])
                    rows.append((target, f"{protocol} ({code}, own grids)", len(names), correlations(y, exp)))

    labels = list(dict.fromkeys(r[1] for r in rows))
    lines = [
        "| Protocol | " + " | ".join(dict.fromkeys(r[0] for r in rows)) + " | mean R | mean ρ | mean τ |",
    ]
    lines.append("|" + "---|" * (len(lines[0].split("|")) - 2))
    for label in labels:
        sel = [r for r in rows if r[1] == label]
        cells = [f"{r[3]['pearson'][0]:.2f}" for r in sel]
        means = [np.mean([r[3][k][0] for r in sel]) for k in ("pearson", "spearman", "kendall")]
        lines.append(f"| {label} | " + " | ".join(cells) + " | " + " | ".join(f"{m:.2f}" for m in means) + " |")
    text = "Pearson R per target (n ligands: " + ", ".join(
        f"{t} {n}" for t, n in dict.fromkeys((r[0], r[2]) for r in rows)) + ")\n\n" + "\n".join(lines)
    text += "\n\nPer target with bootstrap 95 % intervals:\n\n| Target | Protocol | n | R | ρ | τ |\n|---|---|---|---|---|---|\n"
    for target, label, n, c in rows:
        cells = " | ".join(f"{c[k][0]:.2f} [{c[k][1]:.2f}, {c[k][2]:.2f}]" for k in ("pearson", "spearman", "kendall"))
        text += f"| {target} | {label} | {n} | {cells} |\n"
    text += "\nTiming (PB calculations per target and protocol):\n\n| Target | Protocol | calculations | s | s per calculation |\n|---|---|---|---|---|\n"
    for target, protocol, seconds, calcs in timing:
        text += f"| {target} | {protocol} | {calcs} | {seconds:.1f} | {seconds / calcs:.2f} |\n"
    if consistency:
        worst = max(consistency, key=lambda c: c[2])
        text += f"\nLargest |Δ reference − Coulomb/pdie|: {worst[2]:.3f} kcal/mol ({worst[0]}, {worst[1]})\n"
    for target, protocol, diff, passed, n in apbs_rows:
        text += (f"\nAPBS 3.4.1, {target} {protocol} on each complex's own grid: ΔΔG_polar within {diff:.4f} kcal/mol "
                 f"of pbsolve in APBS units, binding verdict passed {passed}/{n}\n")
    text = f"Structures: {structures}\n\n" + text
    with open(os.path.join(work, f"report-{structures}.md"), "w") as f:
        f.write(text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the benchmark files at the pinned commit")
    f.add_argument("data")
    pr = sub.add_parser("prepare", help="force-field parameters for receptors, cofactors and ligands")
    pr.add_argument("data")
    pr.add_argument("work")
    pr.add_argument("--targets", nargs="+", default=list(TARGETS))
    pr.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    m = sub.add_parser("minimize", help="restrained minimisation of every complex")
    m.add_argument("data")
    m.add_argument("work")
    m.add_argument("--targets", nargs="+", default=list(TARGETS))
    m.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r = sub.add_parser("run", help="MM-PBSA for every protocol")
    r.add_argument("work")
    r.add_argument("--targets", nargs="+", default=list(TARGETS))
    r.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    r.add_argument("--platform", default="cpu")
    r.add_argument("--spacing", type=float, default=0.5)
    r.add_argument("--workers", type=int)
    r.add_argument("--structures", default="minimized", choices=STRUCTURES)
    ap = sub.add_parser("apbs", help="the same protocol in APBS 3.4.1, each complex on its own grid")
    ap.add_argument("work")
    ap.add_argument("target")
    ap.add_argument("--protocol", default="eps2",
                    choices=[k for k in PROTOCOLS if k not in GRID_COULOMB and "surface" not in PROTOCOLS[k]])
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--structures", default="minimized", choices=STRUCTURES)
    rp = sub.add_parser("report", help="correlations with experiment")
    rp.add_argument("work")
    rp.add_argument("--targets", nargs="+", default=list(TARGETS))
    rp.add_argument("--structures", default="minimized", choices=STRUCTURES)
    a = p.parse_args(argv)
    if a.command == "fetch":
        fetch(a.data)
    elif a.command == "prepare":
        prepare(a.data, a.work, a.targets, a.jobs)
    elif a.command == "minimize":
        minimize(a.data, a.work, a.targets, a.threads)
    elif a.command == "run":
        run(a.work, a.targets, a.protocols, a.platform, a.spacing, a.workers, a.structures)
    elif a.command == "apbs":
        apbs(a.work, a.target, a.protocol, a.spacing, a.structures)
    else:
        report(a.work, a.targets, a.structures)


if __name__ == "__main__":
    sys.exit(main())
