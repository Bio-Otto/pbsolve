#!/usr/bin/env python3
"""Validate pbsolve's readers against MDAnalysis/ParmEd on real simulation files.

    scripts/make_io_fixtures.sh OUTDIR        # builds files + references (MD tools env)
    python scripts/validate_io.py OUTDIR      # compares (pbsolve env); exit 1 on failure

Reference values come from MDAnalysis (float32 coordinates, hence ~1e-6 Å
tolerances for ASCII formats) and ParmEd/netCDF4.
"""
import argparse, os, pathlib, sys
import numpy as np


def dump_references(F):
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda, parmed as pmd
    out = {}
    def frames(u):
        return np.array([u.atoms.positions.astype(np.float64) for _ in u.trajectory])
    def topo(key, u):
        a = u.atoms
        out[key+".names"] = np.array(a.names); out[key+".resnames"] = np.array(a.resnames); out[key+".resids"] = a.resids
        if hasattr(a, "charges"): out[key+".charges"] = a.charges.astype(np.float64)
        if hasattr(a, "masses"): out[key+".masses"] = a.masses.astype(np.float64)
    A = f"{F}/amber"; G = f"{F}/gmx"
    u = mda.Universe(f"{A}/pep.prmtop", f"{A}/pep.dcd"); topo("prmtop", u); out["dcd"] = frames(u)
    p = pmd.load_file(f"{A}/pep.prmtop"); out["prmtop.radii"] = np.array([x.solvent_radius for x in p.atoms]); out["prmtop.z"] = np.array([x.atomic_number for x in p.atoms]); out["prmtop.nbonds"] = len(p.bonds)
    out["nc"] = frames(mda.Universe(f"{A}/pep.prmtop", f"{A}/pep.nc"))
    out["mdcrd"] = frames(mda.Universe(f"{A}/pep.prmtop", f"{A}/pep.mdcrd", format="TRJ"))
    out["inpcrd"] = frames(mda.Universe(f"{A}/pep.prmtop", f"{A}/pep.inpcrd", format="INPCRD"))
    out["rst7"] = frames(mda.Universe(f"{A}/pep.prmtop", f"{A}/pep_last.rst7", format="INPCRD"))
    import netCDF4
    with netCDF4.Dataset(f"{A}/pep_last.ncrst") as ds: out["ncrst"] = np.array(ds["coordinates"][:], dtype=np.float64)[None]
    u = mda.Universe(f"{A}/pep.psf"); topo("psf", u); out["psf.segids"] = np.array(u.atoms.segids); out["psf.nbonds"] = len(u.bonds)
    u = mda.Universe(f"{A}/pep_solute.mol2"); topo("mol2", u); out["mol2"] = frames(u); out["mol2.nbonds"] = len(u.bonds)
    u = mda.Universe(f"{A}/pep_solute.pdb"); out["pdb"] = frames(u); out["pdb.names"] = np.array(u.atoms.names)
    u = mda.Universe(f"{G}/md.tpr", f"{G}/md.xtc"); topo("tpr", u); out["xtc"] = frames(u); out["tpr.nbonds"] = len(u.bonds)
    out["trr"] = frames(mda.Universe(f"{G}/md.tpr", f"{G}/md.trr"))
    out["gro"] = frames(mda.Universe(f"{G}/md.gro"))
    out["mdpdb"] = frames(mda.Universe(f"{G}/md_frames.pdb"))
    u = mda.Universe(f"{F}/1YCR.cif") if False else None
    np.savez(f"{F}/refs.npz", **{k: v for k, v in out.items() if v is not None})
    print("references:", len(out))


def validate(F, GMXTOP):
    import pbsolve._core as c
    A = f"{F}/amber"; G = f"{F}/gmx"; R = np.load(f"{F}/refs.npz", allow_pickle=True)  # written by dump_references
    global fails
    fails = 0
    def check(label, ok, detail=""):
        global fails
        fails += (not ok)
        print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    def traj(path, **kw):
        r = c.TrajectoryReader(path, **kw)
        return np.array([f[0] for f in r]), r
    def cmp_frames(label, ours, ref, tol):
        if ours.shape != ref.shape:
            return check(label, False, f"shape {ours.shape} vs {ref.shape}")
        d = np.abs(ours - ref).max()
        check(label, d <= tol, f"{ours.shape[0]} frames, max |Δ| = {d:.2e} Å")
    def cmp_topo(label, d, key, charge_tol=1e-6):
        check(f"{label} names", list(d["names"]) == list(R[key+".names"]))
        check(f"{label} resnames", list(d["resnames"]) == list(R[key+".resnames"]))
        if key == "tpr":  # MDAnalysis renumbers tpr residues from 1; compare with the GRO numbering instead
            g = c.read_structure(f"{G}/md.gro"); prot = np.array([s.startswith("Protein") for s in d["segids"]])
            check(f"{label} resids (protein, vs GRO)", np.array_equal(d["resids"][prot], g["resids"][prot]))
        else:
            check(f"{label} resids", np.array_equal(d["resids"], R[key+".resids"]))
        if key+".charges" in R:
            dq = np.abs(d["charges"] - R[key+".charges"]).max(); check(f"{label} charges", dq <= charge_tol, f"max |Δq| = {dq:.1e}")
        if key+".masses" in R and not np.isnan(d["masses"]).all():
            dm = np.abs(d["masses"] - R[key+".masses"]).max(); check(f"{label} masses", dm <= 1e-4, f"max |Δm| = {dm:.1e}")

    d = c.read_structure(f"{A}/pep.prmtop"); cmp_topo("prmtop", d, "prmtop")
    check("prmtop radii", np.abs(d["radii"] - R["prmtop.radii"]).max() < 1e-12)
    z = {"H":1,"C":6,"N":7,"O":8,"S":16,"Na":11,"Cl":17}; check("prmtop elements", [z[e] for e in d["elements"]] == list(R["prmtop.z"]))
    check("prmtop bonds", len(d["bonds"]) == int(R["prmtop.nbonds"]), f"{len(d['bonds'])}"); check("prmtop periodic", d["periodic"] is True)
    n = len(d["names"])
    cmp_frames("dcd", traj(f"{A}/pep.dcd")[0], R["dcd"], 1e-5)
    x, r = traj(f"{A}/pep.nc"); cmp_frames("netcdf", x, R["nc"], 1e-5)
    cmp_frames("mdcrd (ref is float32)", traj(f"{A}/pep.mdcrd", n_atoms=n, mdcrd_box=None)[0], R["mdcrd"], 5e-6)
    cmp_frames("inpcrd (ref is float32)", c.read_structure(f"{A}/pep.inpcrd")["coords"], R["inpcrd"], 5e-6)
    cmp_frames("rst7", c.read_structure(f"{A}/pep_last.rst7")["coords"], R["rst7"], 1e-7)
    cmp_frames("ncrst", c.read_structure(f"{A}/pep_last.ncrst")["coords"], R["ncrst"], 1e-9)
    d = c.read_structure(f"{A}/pep.psf"); cmp_topo("psf", d, "psf"); check("psf segids", list(d["segids"]) == list(R["psf.segids"])); check("psf bonds", len(d["bonds"]) == int(R["psf.nbonds"]))
    dp = c.read_structure(f"{A}/pep.prmtop"); check("psf elements == prmtop elements", list(d["elements"]) == list(dp["elements"]))
    d = c.read_structure(f"{A}/pep_solute.mol2"); cmp_topo("mol2", d, "mol2"); cmp_frames("mol2 coords", d["coords"], R["mol2"], 1e-4); check("mol2 bonds", len(d["bonds"]) == int(R["mol2.nbonds"]))
    d = c.read_structure(f"{A}/pep_solute.pdb"); cmp_frames("pdb coords", d["coords"], R["pdb"], 1e-6); check("pdb names", list(d["names"]) == list(R["pdb.names"]))
    for top in ("processed.top", "topol.top"):
        d = c.read_structure(f"{G}/{top}", include_dirs=[GMXTOP]); cmp_topo(top, d, "tpr")
        check(f"{top} bonds", len(d["bonds"]) >= int(R["tpr.nbonds"]) - 1, f"{len(d['bonds'])} vs tpr {int(R['tpr.nbonds'])}")
    cmp_frames("xtc", traj(f"{G}/md.xtc")[0], R["xtc"], 2e-4)
    cmp_frames("trr", traj(f"{G}/md.trr")[0], R["trr"], 1e-5)
    cmp_frames("gro (with velocities)", c.read_structure(f"{G}/md.gro")["coords"], R["gro"], 1e-5)  # ref is float32
    cmp_frames("multi-model pdb", c.read_structure(f"{G}/md_frames.pdb")["coords"], R["mdpdb"], 1e-5)  # ref is float32
    a = c.read_structure(f"{F}/1YCR.cif"); b = c.read_structure(f"{F}/1YCR.pdb")
    check("mmCIF vs PDB atoms", list(a["names"]) == list(b["names"]) and list(a["chains"]) == list(b["chains"]) and np.array_equal(a["resids"], b["resids"]))
    cmp_frames("mmCIF vs PDB coords", a["coords"], b["coords"], 1e-9)
    print("FAILURES:", fails)
    return fails


fails = 0
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--dump-references", action="store_true", help="(MD tools env) write refs.npz")
    ap.add_argument("--gmxtop", default=os.path.join(os.environ.get("GMXDATA", ""), "top"), help="GROMACS share/top")
    a = ap.parse_args()
    F = str(pathlib.Path(a.outdir).resolve())
    if a.dump_references:
        dump_references(F)
        sys.exit(0)
    sys.exit(1 if validate(F, a.gmxtop) else 0)
