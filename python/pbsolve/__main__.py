"""Command-line interface: ``pbsolve {info,test,solvation,convert,mmpbsa,prep,apbs}``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional, Sequence

import numpy as np

from . import Solver, __version__, available_platforms, cuda_info, write_dx
from .io import Molecule, frames, load
from .mmpbsa import nonpolar_energy, pbsa_solvation_frame


# ----------------------------------------------------------------- shared options
def _add_input(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="structure or topology (pqr, pdb, cif, gro, mol2, prmtop, psf, top, ...)")
    p.add_argument("--coords", help="coordinates (inpcrd/rst7/gro/pdb/... or a trajectory: first frame)")
    p.add_argument("--format", help="format of INPUT if the extension is ambiguous")
    p.add_argument(
        "--radii",
        choices=["file", "bondi", "mbondi", "mbondi2", "parse"],
        default="file",
        help="radii from the file (PQR, prmtop; default) or by element scheme",
    )
    p.add_argument(
        "--keep-solvent", action="store_true", help="keep water and monovalent ions (stripped by default)"
    )
    p.add_argument("-I", "--include", action="append", default=[], help="GROMACS include directory")
    p.add_argument(
        "--ff",
        choices=["amber", "parse", "charmm"],
        help="prepare a PDB/mmCIF without charges first (hydrogens, protonation, charges/radii; see 'pbsolve prep')",
    )


def _add_solver(p: argparse.ArgumentParser) -> None:
    p.add_argument("--platform", default="auto")
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--pdie", type=float, default=1.0)
    p.add_argument("--sdie", type=float, default=78.54)
    p.add_argument("--temperature", type=float, default=298.15)
    p.add_argument("--ionic-strength", type=float, default=0.0)
    p.add_argument("--ion-radius", type=float, default=2.0)
    p.add_argument("--srad", type=float, default=1.4)
    p.add_argument("--surface", default="mol", choices=["mol", "vdw", "harmonic"])
    p.add_argument("--bcfl", default="mdh", choices=["mdh", "sdh", "zero"])
    p.add_argument(
        "--dime-policy",
        default="pbsolve",
        choices=["pbsolve", "apbs"],
        help="grid node counts: 8k+1 (pbsolve) or values APBS accepts unchanged (apbs)",
    )


def _solver(args: argparse.Namespace, dime_policy: Optional[str] = None) -> Solver:
    return Solver(
        platform=args.platform,
        grid_spacing=args.spacing,
        pdie=args.pdie,
        sdie=args.sdie,
        temperature=args.temperature,
        ionic_strength=args.ionic_strength,
        ion_radius=args.ion_radius,
        srad=args.srad,
        surface=args.surface,
        bcfl=args.bcfl,
        dime_policy=dime_policy or args.dime_policy,
    )


def _load(args: argparse.Namespace, need_coords: bool = True) -> Molecule:
    if getattr(args, "ff", None):
        from .prep import prepare

        mol, rep = prepare(args.input, args.ff)
        print("# " + rep.summary().replace("\n", "\n# "), file=sys.stderr)
        if args.radii != "file":
            mol = mol.with_radii(args.radii)
        return mol
    mol = load(args.input, args.coords, format=args.format, include_dirs=args.include or None)
    if not args.keep_solvent:
        n = mol.n_atoms
        mol = mol.strip_solvent()
        if mol.n_atoms < n:
            print(f"# stripped {n - mol.n_atoms} water/ion atoms (use --keep-solvent to keep them)", file=sys.stderr)
    if args.radii != "file":
        mol = mol.with_radii(args.radii)
    elif not mol.has_radii:
        raise ValueError(
            f"{args.input} has no radii; choose a scheme with --radii (e.g. mbondi2) or use a PQR/prmtop file"
        )
    mol.require(coords=need_coords)
    return mol


# ----------------------------------------------------------------- commands
def _solvation(args: argparse.Namespace) -> int:
    mol = _load(args)
    coords, charges, radii = mol.coords, mol.charges, mol.radii
    solver = _solver(args, "apbs" if args.apbs_grid else None)
    grid = solver.make_grid(coords, radii)
    res = solver.polar_solvation(coords, charges, radii, grid=grid)
    out = {
        "file": args.input,
        "atoms": int(len(charges)),
        "net_charge": float(np.sum(charges)),
        "polar_kcal_mol": res.energy,
        "nonpolar_kcal_mol": nonpolar_energy(coords, radii),
        "platform": res.platform,
        "device": solver.device_name,
        "cg_iterations": res.iterations,
        "dime": grid.dims,
        "fine_spacing": grid.fine_spacing,
    }
    if args.dx:
        pot, origin, spacing, _ = solver.potential(coords, charges, radii, grid=grid)
        write_dx(args.dx, pot, origin, spacing, comment=f"pbsolve {__version__} potential (kT/e) for {args.input}")
        out["dx"] = args.dx
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:>18}: {v}")
    if args.apbs_grid:
        print("\n# equivalent APBS mg-auto grid\n" + grid.to_apbs())
    return 0


def _convert(args: argparse.Namespace) -> int:
    mol = _load(args)
    if args.select:
        mol = mol.subset(args.select)
    xyz = mol.coords
    if args.frame:
        if mol.models is None or args.frame >= mol.models.shape[0]:
            raise ValueError(f"frame {args.frame} not available")
        xyz = mol.models[args.frame]
    mol.write_pqr(args.output, xyz)
    print(f"wrote {args.output}: {mol.n_atoms} atoms, net charge {np.sum(mol.charges):+.4f} e")
    return 0


def _mmpbsa(args: argparse.Namespace) -> int:
    full = load(args.input, None, format=args.format, include_dirs=args.include or None)
    keep = np.ones(full.n_atoms, dtype=bool) if args.keep_solvent else ~full.solvent_mask()
    mol = full.subset(keep)
    if args.radii != "file":
        mol = mol.with_radii(args.radii)
    elif not mol.has_radii:
        raise ValueError(f"{args.input} has no radii; choose a scheme with --radii (e.g. mbondi2)")
    mol.require(coords=False)
    lig = np.nonzero(mol.select(args.ligand))[0]
    rec = None if args.receptor is None else np.nonzero(mol.select(args.receptor))[0]
    if lig.size == 0:
        raise ValueError(f"ligand selection '{args.ligand}' is empty")
    solver = _solver(args)
    rows = []
    for k, xyz in enumerate(frames(full, args.trajectory, start=args.start, stop=args.stop, step=args.step, atoms=keep)):
        f = pbsa_solvation_frame(solver, xyz, mol.charges, mol.radii, lig, rec)
        row = {"frame": args.start + k * args.step, **f.as_dict()}
        rows.append(row)
        print(
            f"frame {row['frame']:>6}: ΔG_polar {row['delta_polar']:10.4f}  ΔG_nonpolar {row['delta_nonpolar']:8.4f}"
            f"  ΔG_solv {row['delta_solvation']:10.4f} kcal/mol",
            flush=True,
        )
    if not rows:
        raise ValueError("no frames selected")
    keys = [k for k in rows[0] if k != "frame"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    sd = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in keys}
    print(f"\n{len(rows)} frames (solvation terms only; add ΔE_MM from your MM engine), kcal/mol:")
    for k in ("delta_polar", "delta_nonpolar", "delta_solvation"):
        print(f"  {k:<16} {mean[k]:10.4f} ± {sd[k] / math.sqrt(len(rows)):.4f} (SEM), SD {sd[k]:.4f}")
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(",".join(["frame", *keys]) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in ["frame", *keys]) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ._console import utf8_stdio

    utf8_stdio()
    ap = argparse.ArgumentParser(
        prog="pbsolve", description="Poisson–Boltzmann electrostatics and solvation energies (CPU/CUDA)"
    )
    ap.add_argument("--version", action="version", version=f"pbsolve {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show build and platform information")

    t = sub.add_parser("test", help="run the installation check")
    t.add_argument("--platform", action="append")
    t.add_argument("--spacing", type=float, default=0.25)

    s = sub.add_parser("solvation", help="polar + non-polar solvation energy of one structure")
    _add_input(s)
    _add_solver(s)
    s.add_argument("--dx", help="write the solvated potential to this OpenDX file")
    s.add_argument(
        "--apbs-grid", action="store_true", help="print the equivalent APBS grid keywords (implies --dime-policy apbs)"
    )
    s.add_argument("--json", action="store_true")

    c = sub.add_parser("convert", help="write a PQR file from any supported input (charges + radii)")
    _add_input(c)
    c.add_argument("output", help="output PQR file")
    c.add_argument("--select", help="atoms to keep, e.g. 'not resname LIG'")
    c.add_argument("--frame", type=int, default=0, help="model/frame of a multi-frame INPUT/--coords file")

    m = sub.add_parser("mmpbsa", help="MM-PBSA solvation terms along a trajectory (single-trajectory protocol)")
    m.add_argument("input", help="topology with charges (prmtop, psf, top, pqr, mol2, ...)")
    m.add_argument("trajectory", help="dcd, xtc, trr, nc, mdcrd, multi-model pdb/gro")
    m.add_argument("--ligand", required=True, help="ligand selection, e.g. 'resname LIG' or 'chain B'")
    m.add_argument("--receptor", help="receptor selection (default: all other solute atoms)")
    m.add_argument("--format", help="format of INPUT if the extension is ambiguous")
    m.add_argument("--radii", choices=["file", "bondi", "mbondi", "mbondi2", "parse"], default="file")
    m.add_argument("--keep-solvent", action="store_true")
    m.add_argument("-I", "--include", action="append", default=[], help="GROMACS include directory")
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--stop", type=int)
    m.add_argument("--step", type=int, default=1)
    m.add_argument("--csv", help="write per-frame terms to this CSV file")
    _add_solver(m)

    sub.add_parser("prep", help="PDB/mmCIF -> PQR: hydrogens, protonation, force field (see 'pbsolve prep -h')", add_help=False)
    sub.add_parser("apbs", help="reciprocal validation against APBS (see 'pbsolve apbs -h')", add_help=False)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["apbs"]:
        from .apbs import main as apbs_main

        return apbs_main(argv[1:])
    if argv[:1] == ["prep"]:
        from .prep import main as prep_main

        return prep_main(argv[1:])
    args = ap.parse_args(argv)
    if args.cmd == "info":
        print(f"pbsolve {__version__}")
        print(f"platforms: {', '.join(available_platforms())}")
        for k, v in cuda_info().items():
            print(f"cuda.{k}: {v}")
        return 0
    if args.cmd == "test":
        from .test_installation import main as test_main

        targv = [f"--spacing={args.spacing}"] + [f"--platform={p}" for p in (args.platform or [])]
        return test_main(targv)
    try:
        return {"solvation": _solvation, "convert": _convert, "mmpbsa": _mmpbsa}[args.cmd](args)
    except (ValueError, OSError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
