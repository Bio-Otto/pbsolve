"""Command-line interface: ``pbsolve {info,test,solvation,convert,mmpbsa,prep,apbs}``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional, Sequence

import numpy as np

from . import Solver, __version__, available_platforms, cuda_info, estimate_error, write_dx
from .io import Molecule, frames, load
from .mmpbsa import nonpolar_energy, pbsa_solvation_frames


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
    p.add_argument("--surface", default="mol", choices=["mol", "vdw", "harmonic", "spl2", "spl4", "gauss"])
    p.add_argument("--swin", type=float, default=0.3, help="spl2/spl4 surfaces: half-width of the switching window (Å)")
    p.add_argument("--gauss-sigma", type=float, default=0.93, help="gauss: width of the atomic Gaussians relative to the radii")
    p.add_argument("--gauss-contour", type=float, default=20.0, help="gauss: the reference state keeps ε up to this value")
    p.add_argument(
        "--interface",
        default="sharp",
        choices=["sharp", "fractional"],
        help="dielectric of the grid faces the surface cuts (default sharp: as APBS; fractional: harmonic average "
        "weighted by the edge length inside the solute, close to second order; mol and vdw surfaces)",
    )
    p.add_argument(
        "--equation",
        default="lpbe",
        choices=["lpbe", "npbe", "smpbe"],
        help="Poisson-Boltzmann equation of the solvated state: lpbe (linearized, default), npbe (nonlinear, "
        "Boltzmann-distributed ions; as APBS npbe) or smpbe (nonlinear with ions of finite size, see --ion-size)",
    )
    p.add_argument(
        "--ion-size",
        type=float,
        default=0.0,
        help="with --equation smpbe: edge of the lattice cell one ion fills, in Å (APBS smpbe vol)",
    )
    p.add_argument(
        "--coulomb",
        default="grid",
        choices=["grid", "analytic"],
        help="how dG_polar removes the Coulomb potential (default grid: reference state solved on the same grid, "
        "as APBS; analytic: reaction-field decomposition, one solve instead of two)",
    )
    p.add_argument(
        "--bcfl",
        default="mdh",
        choices=["mdh", "mdh-fast", "sdh", "zero"],
        help="coarse-grid boundary condition (default mdh, as APBS; mdh-fast interpolates the same sum, "
        "error <= 1e-10 of the summed terms, several times faster on large grids)",
    )
    p.add_argument(
        "--dime-policy",
        default="pbsolve",
        choices=["pbsolve", "apbs"],
        help="grid node counts: 8k+1 (pbsolve) or values APBS accepts unchanged (apbs)",
    )
    p.add_argument(
        "--preconditioner",
        default="multigrid-f32",
        choices=["multigrid-f32", "multigrid", "jacobi"],
        help="preconditioner of the CG solver (default: multigrid-f32, the multigrid V-cycle in single precision "
        "inside the f64 CG; multigrid: the f64 V-cycle, bit-identical to earlier versions)",
    )
    p.add_argument(
        "--chgm",
        default="spl0",
        choices=["spl0", "spl2", "spl4"],
        help="charge assignment to the grid, as APBS chgm (default spl0: trilinear; spl2, spl4: B-splines of "
        "order 3 and 5 over 3 and 5 nodes per axis)",
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
        interface=args.interface,
        coulomb=args.coulomb,
        equation=args.equation,
        ion_size=args.ion_size,
        swin=args.swin,
        gauss_sigma=args.gauss_sigma,
        gauss_contour=args.gauss_contour,
        bcfl=args.bcfl,
        dime_policy=dime_policy or args.dime_policy,
        preconditioner=args.preconditioner,
        chgm=args.chgm,
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
    if args.extrapolate is not None and args.placements < 2:
        print("error: --extrapolate needs --placements (at least 2)", file=sys.stderr)
        return 2
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
        "preconditioner": sorted(set(res.preconditioners)),
        "multigrid_levels": res.levels,
        "dime": grid.dims,
        "fine_spacing": grid.fine_spacing,
    }
    if args.placements:
        spacings = [args.spacing, args.extrapolate] if args.extrapolate is not None else None
        est = estimate_error(solver, coords, charges, radii, placements=args.placements, spacings=spacings)
        out["placements"] = est.placements
        out["polar_mean_kcal_mol"] = est.energy
        out["polar_sd_kcal_mol"] = est.sd
        out["polar_sem_kcal_mol"] = est.sem
        if est.extrapolated is not None:
            out["polar_extrapolated_kcal_mol"] = est.extrapolated
            out["extrapolation_spacings"] = est.spacings
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
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    solver = _solver(args)
    if args.per_residue and solver.params["equation"] != "lpbe" and solver.params["ionic_strength"] > 0:
        raise ValueError("--per-residue needs equation 'lpbe' (or no ions): the nonlinear energy is not a sum over atoms")
    labels = [f"{c}:{n}:{i}{ic}" for c, n, i, ic in zip(mol.chains, mol.resnames, mol.resids, mol.icodes)]
    residues: dict[str, list[tuple[float, float]]] = {}
    rows = []
    # Frames are solved in chunks so that `workers` calculations overlap
    # while memory stays bounded; output order is the frame order.
    chunk_size = 4 * (args.workers or 3)
    trajectory = frames(full, args.trajectory, start=args.start, stop=args.stop, step=args.step, atoms=keep)
    chunk: list = []

    def flush() -> None:
        results = pbsa_solvation_frames(solver, chunk, mol.charges, mol.radii, lig, rec, workers=args.workers)
        for f in results:
            row = {"frame": args.start + len(rows) * args.step, **f.as_dict()}
            rows.append(row)
            if args.per_residue:
                for label, terms in f.residue_terms(labels).items():
                    residues.setdefault(label, []).append(terms)
            print(
                f"frame {row['frame']:>6}: ΔG_polar {row['delta_polar']:10.4f}  ΔG_nonpolar {row['delta_nonpolar']:8.4f}"
                f"  ΔG_solv {row['delta_solvation']:10.4f} kcal/mol",
                flush=True,
            )
        chunk.clear()

    for xyz in trajectory:
        chunk.append(np.array(xyz, dtype=np.float64))
        if len(chunk) == chunk_size:
            flush()
    if chunk:
        flush()
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
    if args.per_residue:
        # β belongs to no residue: the residues' non-polar shares sum to delta_nonpolar + β.
        with open(args.per_residue, "w") as fh:
            fh.write("residue,delta_polar,delta_nonpolar,delta_solvation,sem_solvation\n")
            for label, terms in residues.items():
                t = np.array(terms)
                total = t.sum(axis=1)
                sem = float(np.std(total, ddof=1)) / math.sqrt(len(total)) if len(total) > 1 else 0.0
                fh.write(f"{label},{t[:, 0].mean()},{t[:, 1].mean()},{total.mean()},{sem}\n")
        print(f"per-residue shares of {len(residues)} residues written to {args.per_residue}")
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
    s.add_argument(
        "--placements",
        type=int,
        default=0,
        help="also solve on this many grid placements (>= 2) and report their mean, sd and sem",
    )
    s.add_argument(
        "--extrapolate",
        type=float,
        metavar="SPACING",
        help="with --placements: repeat at this finer spacing and extrapolate the mean to zero spacing",
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
    m.add_argument(
        "--per-residue",
        metavar="CSV",
        help="write every residue's share of the binding terms (mean over frames) to this CSV file",
    )
    m.add_argument(
        "--workers",
        type=int,
        help="PB calculations solved concurrently (default: 2 on the CPU, 3 on CUDA; on CUDA at most as many as "
        "fit in the free GPU memory); results do not depend on it",
    )
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
