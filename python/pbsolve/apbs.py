"""Reciprocal validation against APBS.

pbsolve and APBS (multigrid, ``lpbe`` or ``npbe``, ``srfm mol/spl2/spl4``, ``chgm spl0/spl2/spl4``) can be run
on the same problem in both directions:

* :func:`compare` — pbsolve settings → APBS input → run both → layered report;
* :func:`compare_binding` — the same for complex, receptor and ligand on the
  complex's grid (MM-PBSA ΔΔG_polar);
* :func:`run_input` — an existing APBS input file → run it with pbsolve
  (and APBS) → per-calculation and ``print`` energies side by side.

The comparison is layered so that a disagreement can be located:

1. **grid** — APBS' effective grids, parsed from its output, must equal
   pbsolve's focusing levels (APBS silently resets ``dime`` > 65 that is not
   ``32c+1``; use ``Solver(dime_policy="apbs")``);
2. **maps** — fine-level dielectric (x/y/z faces), ion accessibility and
   charge maps written by APBS vs pbsolve's own;
3. **injection** — pbsolve solves with APBS' maps and boundary values, which
   tests the discretisation and linear solver alone;
4. **energies** — solvated, reference and ΔG_polar.

Known irreducible differences: APBS mixes CGS and CODATA-1986 constants
while pbsolve uses CODATA 2018 (energies differ by a factor
:func:`apbs_unit_factor` ≈ 1 + 1.2e-5, which reports also remove); DX maps
carry 7 significant digits; both linear solvers stop at a finite tolerance.

Command line::

    pbsolve apbs compare complex.pqr --spacing 0.5 --ionic-strength 0.15 [--ligand-chain B]
    pbsolve apbs run apbs.in
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from . import apolar as _apolar
from ._core import GridSpec, Solver, bjerrum_length, kt_kcal_per_mol, kt_kj_per_mol, read_pqr
from .dx import read_dx

PathLike = Union[str, "os.PathLike[str]"]

__all__ = [
    "ApbsCalcOutput",
    "ApbsError",
    "ApbsInput",
    "ApbsOutput",
    "BindingReport",
    "ComparisonReport",
    "ElecCalc",
    "GridMismatchError",
    "RunReport",
    "apbs_unit_factor",
    "apolar_energy",
    "compare",
    "compare_binding",
    "elec_from_solver",
    "find_apbs",
    "main",
    "parse_input",
    "parse_output",
    "read_parm",
    "run_apbs",
    "run_input",
    "write_input",
]

#: Output precision of APBS' "Grid spacings/lengths/center" lines (%.3f).
_GRID_PRINT_TOL = 1.01e-3

# APBS 3.4.1 constants, used ONLY to quantify the expected unit difference.
# Source term (src/generic/vpbe.c, CGS): zmagic = 4π e_c²/(k_B T)·1e8.
_APBS_EC_ESU = 4.803242384e-10
_APBS_KB_ERG = 1.380662000e-16
# kT -> kJ/mol (src/generic/vunit.h).
_APBS_KB = 1.3806581e-23
_APBS_NA = 6.0221367e23


def apbs_unit_factor(temperature: float = 298.15) -> float:
    """Expected ratio E_APBS / E_pbsolve (kJ/mol) caused by constants alone.

    APBS scales the source with a CGS Bjerrum length and converts kT to kJ/mol
    with CODATA-1986 values; pbsolve uses CODATA 2018 throughout. The product
    is ≈ 1 + 1.2186e-5 (temperature independent).
    """
    lb_apbs = _APBS_EC_ESU**2 / (_APBS_KB_ERG * temperature) * 1e8  # Å
    kj_apbs = _APBS_KB * _APBS_NA * temperature / 1000.0
    return (lb_apbs / bjerrum_length(temperature)) * (kj_apbs / kt_kj_per_mol(temperature))


class ApbsError(RuntimeError):
    """APBS failed, or its input/output could not be handled."""


class GridMismatchError(ApbsError):
    """APBS solved on a different grid than pbsolve (the comparison would be meaningless)."""


# ============================================================== input files


@dataclass
class ElecCalc:
    """One APBS ``elec`` block in the subset pbsolve can reproduce."""

    name: str
    grid: GridSpec
    pdie: float = 1.0
    sdie: float = 78.54
    temperature: float = 298.15
    ions: list[tuple[float, float, float]] = field(default_factory=list)  # (charge, conc M, radius Å)
    srad: float = 1.4
    sdens: float = 10.0
    srfm: str = "mol"  # or "spl2", "spl4"
    swin: float = 0.3
    bcfl: str = "mdh"
    chgm: str = "spl0"
    method: str = "mg-auto"  # or "mg-manual"
    pbe: str = "lpbe"  # or "npbe"
    mol: int = 1
    etol: Optional[float] = None
    calcenergy: str = "total"
    calcforce: str = "no"  # or "comps" (per-atom forces)
    writes: list[tuple[str, str]] = field(default_factory=list)  # (APBS type, file stem), DX format
    requested_dims: Optional[tuple[int, int, int]] = None  # dims in the file, before APBS' reset

    def solver(
        self,
        platform: str = "auto",
        tol: Optional[float] = None,
        max_iter: int = 20000,
        preconditioner: Optional[str] = None,
    ) -> Solver:
        """A pbsolve Solver with exactly these physical settings
        (``preconditioner=None``: the Solver default)."""
        return Solver(
            platform=platform,
            pdie=self.pdie,
            sdie=self.sdie,
            temperature=self.temperature,
            ions=list(self.ions) if self.ions else None,
            srad=self.srad,
            sdens=self.sdens,
            surface=self.srfm,
            swin=self.swin,
            bcfl=self.bcfl,
            chgm=self.chgm,
            equation=self.pbe,
            tol=tol if tol is not None else min(self.etol or 1e-8, 1e-8),
            max_iter=max_iter,
            **_preconditioner(preconditioner),
        )

    @property
    def apbs_calc_count(self) -> int:
        """Number of CALCULATION blocks APBS prints for this elec."""
        if self.method == "mg-manual":
            return 1
        return max(2, len(self.grid.levels))


def _preconditioner(name: Optional[str]) -> dict[str, str]:
    """Solver keyword for an optional preconditioner (None keeps the default)."""
    return {} if name is None else {"preconditioner": name}


def elec_from_solver(
    solver: Solver,
    grid: GridSpec,
    name: str,
    *,
    reference: bool = False,
    etol: Optional[float] = None,
    writes: Sequence[tuple[str, str]] = (),
) -> ElecCalc:
    """The APBS ``elec`` equivalent of ``solver`` on ``grid``.

    ``reference=True`` gives the homogeneous reference state (``sdie = pdie``,
    no ions), the second calculation of a ΔG_polar run.
    """
    p = solver.params
    if p["surface"] in ("harmonic", "gauss"):
        raise ValueError(f"surface={p['surface']!r} has no APBS equivalent; compare 'mol' (srfm mol) or 'vdw' (srad 0)")
    if p["interface"] != "sharp":
        raise ValueError(f"interface={p['interface']!r} has no APBS equivalent; compare interface='sharp'")
    if p["coulomb"] != "grid":
        raise ValueError(f"coulomb={p['coulomb']!r} has no APBS equivalent; compare coulomb='grid'")
    if p.get("zmem") is not None:
        raise ValueError("APBS has no keyword for a membrane (draw_membrane2 edits its maps); compare without zmem")
    if p.get("period") is not None:
        raise ValueError("APBS mg-auto has no periodic boundaries; compare without a period")
    if p["equation"] == "smpbe":
        raise ValueError("equation='smpbe' cannot be compared: APBS 3.4.1 disables smpbe")
    srad = 0.0 if p["surface"] == "vdw" else p["srad"]
    srfm = p["surface"] if p["surface"] in ("spl2", "spl4") else "mol"
    grid.to_apbs()  # raises ValueError if APBS would change the grid
    return ElecCalc(
        name=name,
        grid=grid,
        pdie=p["pdie"],
        sdie=p["pdie"] if reference else p["sdie"],
        temperature=p["temperature"],
        ions=[] if reference else [tuple(i) for i in p["ions"]],
        srad=srad,
        sdens=p["sdens"],
        srfm=srfm,
        swin=p["swin"],
        # mdh-fast is pbsolve's interpolated evaluation of APBS' mdh.
        bcfl="mdh" if p["bcfl"] == "mdh-fast" else p["bcfl"],
        chgm=p["chgm"],
        method="mg-auto" if grid.is_focused else "mg-manual",
        pbe=p["equation"],
        etol=etol if etol is not None else p["tol"],
        writes=list(writes),
    )


def _fmt(x: float) -> str:
    return repr(float(x))


def write_input(
    molecules: Sequence[PathLike],
    elecs: Sequence[ElecCalc],
    prints: Sequence[str] = (),
    force_prints: Sequence[str] = (),
) -> str:
    """APBS input text. ``prints`` (energies) and ``force_prints`` (forces) are
    expressions like ``"solvated - reference"``."""
    out = ["read"]
    out += [f"    mol pqr {pathlib.Path(m)}" for m in molecules]
    out.append("end")
    for e in elecs:
        out.append(f"elec name {e.name}")
        out.append(f"    {e.method}")
        if e.method == "mg-auto":
            out += ["    " + line for line in e.grid.to_apbs().splitlines()]
        else:
            g = e.grid
            out.append("    dime {} {} {}".format(*g.dims))
            out.append("    glen " + " ".join(_fmt(v) for v in g.coarse_length))
            out.append("    gcent " + " ".join(_fmt(v) for v in g.center))
            g.to_apbs()  # dims check
        out.append(f"    mol {e.mol}")
        out.append(f"    {e.pbe}")
        out.append(f"    bcfl {e.bcfl}")
        for z, c, r in e.ions:
            out.append(f"    ion charge {_fmt(z)} conc {_fmt(c)} radius {_fmt(r)}")
        out.append(f"    pdie {_fmt(e.pdie)}")
        out.append(f"    sdie {_fmt(e.sdie)}")
        out.append(f"    srfm {e.srfm}")
        out.append(f"    chgm {e.chgm}")
        out.append(f"    sdens {_fmt(e.sdens)}")
        out.append(f"    srad {_fmt(e.srad)}")
        out.append(f"    swin {_fmt(e.swin)}")
        out.append(f"    temp {_fmt(e.temperature)}")
        if e.etol is not None:
            out.append(f"    etol {e.etol:g}")
        out.append(f"    calcenergy {e.calcenergy}")
        out.append(f"    calcforce {e.calcforce}")
        for kind, stem in e.writes:
            out.append(f"    write {kind} dx {stem}")
        out.append("end")
    for expr in prints:
        out.append(f"print elecEnergy {expr} end")
    for expr in force_prints:
        out.append(f"print force {expr} end")
    out.append("quit")
    return "\n".join(out) + "\n"


@dataclass
class ApbsInput:
    """A parsed APBS input file (the subset pbsolve reproduces)."""

    path: Optional[pathlib.Path]
    molecules: list[pathlib.Path]
    elecs: list[ElecCalc]
    prints: list[list[tuple[int, str]]]  # each: [(sign, elec name or 1-based id), ...]
    notes: list[str] = field(default_factory=list)


_UNSUPPORTED = {
    "nrpbe": "nonlinear regularized PBE",
    "lrpbe": "regularized PBE",
    "smpbe": "size-modified PBE",
    "mg-para": "parallel focusing",
    "mg-dummy": "dummy calculations",
    "fe-manual": "finite elements",
    "geoflow-auto": "GeoFlow",
    "bem-manual": "boundary elements",
    "usemap": "external maps",
    "usemesh": "external meshes",
    "zmem": "membrane",
    "lmem": "membrane",
    "mdie": "membrane",
    "memv": "membrane",
}


def _tokens(text: str) -> list[str]:
    toks: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        toks += line.split()
    return toks


def _mol_center(coords: npt.NDArray[np.float64]) -> list[float]:
    """APBS ``Valist`` centre: midpoint of the coordinate bounding box (no radii)."""
    return [0.5 * (float(coords[:, a].max()) + float(coords[:, a].min())) for a in range(3)]


def parse_input(path: PathLike) -> ApbsInput:
    """Parse an APBS input file. Unsupported features raise ``NotImplementedError``."""
    from ._core import apbs_effective_dime

    path = pathlib.Path(path)
    base = path.resolve().parent
    toks = _tokens(path.read_text())
    pos = 0

    def take() -> str:
        nonlocal pos
        if pos >= len(toks):
            raise ApbsError(f"{path}: unexpected end of input")
        pos += 1
        return toks[pos - 1]

    def take_floats(n: int) -> list[float]:
        return [float(take()) for _ in range(n)]

    molecules: list[pathlib.Path] = []
    raw_elecs: list[dict[str, Any]] = []
    prints: list[list[tuple[int, str]]] = []
    notes: list[str] = []
    while pos < len(toks):
        kw = take().lower()
        if kw == "read":
            while True:
                t = take().lower()
                if t == "end":
                    break
                if t == "mol":
                    fmt = take().lower()
                    if fmt != "pqr":
                        raise NotImplementedError(f"read mol {fmt}: only PQR molecules are supported")
                    p = pathlib.Path(take())
                    molecules.append(p if p.is_absolute() else base / p)
                elif t in ("parm", "diel", "kappa", "charge", "pot", "mesh"):
                    raise NotImplementedError(f"read {t}: external maps/parameters are not supported")
        elif kw == "elec":
            e: dict[str, Any] = {"writes": [], "ions": []}
            while True:
                t = take()
                tl = t.lower()
                if tl == "end":
                    break
                if tl in _UNSUPPORTED:
                    raise NotImplementedError(f"elec keyword '{t}' ({_UNSUPPORTED[tl]}) is not supported by pbsolve")
                if tl == "name":
                    e["name"] = take()
                elif tl in ("mg-auto", "mg-manual"):
                    e["method"] = tl
                elif tl == "dime":
                    e["dime"] = [int(take()) for _ in range(3)]
                elif tl in ("cglen", "fglen", "glen"):
                    e[tl] = take_floats(3)
                elif tl == "grid":
                    e["grid"] = take_floats(3)
                elif tl in ("cgcent", "fgcent", "gcent"):
                    t1 = take()
                    e[tl] = ("mol", int(take())) if t1.lower() == "mol" else [float(t1)] + take_floats(2)
                elif tl == "mol":
                    e["mol"] = int(take())
                elif tl in ("lpbe", "npbe"):
                    e["pbe"] = tl
                elif tl == "bcfl":
                    b = take().lower()
                    if b not in ("zero", "sdh", "mdh"):
                        raise NotImplementedError(f"bcfl {b} is not supported (zero, sdh, mdh)")
                    e["bcfl"] = b
                elif tl == "ion":
                    vals: dict[str, float] = {}
                    t1 = take()
                    if t1.lower() in ("charge", "conc", "radius"):
                        key = t1.lower()
                        while True:
                            vals[key] = float(take())
                            if pos < len(toks) and toks[pos].lower() in ("charge", "conc", "radius"):
                                key = take().lower()
                            else:
                                break
                    else:  # legacy: ion <charge> <conc> <radius>
                        vals = {"charge": float(t1), "conc": float(take()), "radius": float(take())}
                    e["ions"].append((vals["charge"], vals["conc"], vals["radius"]))
                elif tl in ("pdie", "sdie", "srad", "sdens", "temp", "swin", "etol", "gamma"):
                    e[tl] = float(take())
                elif tl == "srfm":
                    s = take().lower()
                    if s not in ("mol", "spl2", "spl4"):
                        raise NotImplementedError(f"srfm {s}: pbsolve reproduces 'srfm mol', 'spl2' and 'spl4'")
                    e["srfm"] = s
                elif tl == "chgm":
                    c = take().lower()
                    if c not in ("spl0", "spl2", "spl4"):
                        raise NotImplementedError(f"chgm {c}: only spl0, spl2 and spl4 are supported")
                    e["chgm"] = c
                elif tl == "calcenergy":
                    e["calcenergy"] = take().lower()
                elif tl == "calcforce":
                    f = take().lower()
                    if f != "no":
                        notes.append(f"calcforce {f} ignored here; compare forces with `pbsolve apbs compare --forces`")
                elif tl == "write":
                    kind, fmt, stem = take().lower(), take().lower(), take()
                    e["writes"].append((kind, stem))
                    if fmt != "dx":
                        notes.append(f"write {kind} {fmt}: only DX output is compared")
                elif tl == "nlev":
                    take()
                elif tl == "async":
                    raise NotImplementedError("async (mg-para) is not supported")
                else:
                    raise NotImplementedError(f"elec keyword '{t}' is not supported by pbsolve")
            raw_elecs.append(e)
        elif kw == "print":
            what = take().lower()
            if what not in ("elecenergy", "energy"):
                raise NotImplementedError(f"print {what}: only 'print elecEnergy' is supported")
            terms: list[tuple[int, str]] = []
            sign = 1
            while True:
                t = take()
                if t.lower() == "end":
                    break
                if t in ("+", "-"):
                    sign = 1 if t == "+" else -1
                else:
                    terms.append((sign, t))
                    sign = 1
            prints.append(terms)
        elif kw == "quit":
            break
        elif kw == "apolar":
            raise NotImplementedError("apolar blocks are not supported (use pbsolve.nonpolar_energy)")
        else:
            raise ApbsError(f"{path}: unknown top-level keyword '{kw}'")

    coords_cache: dict[int, npt.NDArray[np.float64]] = {}

    def center_of(spec: Any) -> list[float]:
        if isinstance(spec, tuple):
            mid = spec[1]
            if mid not in coords_cache:
                coords_cache[mid] = read_pqr(molecules[mid - 1])[0]
            return _mol_center(coords_cache[mid])
        return list(spec)

    elecs: list[ElecCalc] = []
    for n, e in enumerate(raw_elecs, start=1):
        name = e.get("name", str(n))
        method = e.get("method")
        if method is None:
            raise ApbsError(f"elec {name}: missing mg-auto/mg-manual")
        if e.get("pbe") not in ("lpbe", "npbe"):
            raise NotImplementedError(f"elec {name}: only 'lpbe' and 'npbe' are supported")
        if "dime" not in e:
            raise ApbsError(f"elec {name}: missing dime")
        req = tuple(e["dime"])
        eff = [apbs_effective_dime(d) for d in req]
        if any(d is None for d in eff):
            raise ApbsError(f"elec {name}: APBS rejects dime {req}")
        dims = [int(d) for d in eff if d is not None]
        if tuple(dims) != req:
            notes.append(f"elec {name}: APBS resets dime {list(req)} -> {dims}; pbsolve uses the reset grid")
        if method == "mg-auto":
            cc, fc = center_of(e.get("cgcent", ("mol", e.get("mol", 1)))), center_of(
                e.get("fgcent", ("mol", e.get("mol", 1)))
            )
            if any(abs(a - b) > 1e-9 for a, b in zip(cc, fc)):
                raise NotImplementedError(f"elec {name}: cgcent != fgcent (pbsolve focusing is concentric)")
            grid = GridSpec(fc, dims, e["cglen"], e["fglen"])
        else:
            if "glen" in e:
                glen = e["glen"]
            elif "grid" in e:
                glen = [(d - 1) * h for d, h in zip(dims, e["grid"])]
            else:
                raise ApbsError(f"elec {name}: mg-manual needs glen or grid")
            grid = GridSpec(center_of(e.get("gcent", ("mol", e.get("mol", 1)))), dims, glen)
        spline = e.get("srfm") in ("spl2", "spl4")
        missing = [k for k in ("bcfl", "pdie", "sdie", "srfm", "temp") if k not in e]
        if spline and "swin" not in e:
            missing.append("swin")
        if not spline and "srad" not in e:
            missing.append("srad")
        if not spline and "sdens" not in e and e.get("srad", 0.0) > 1e-9:
            missing.append("sdens")
        if missing:
            raise ApbsError(f"elec {name}: APBS requires {', '.join(missing)}")
        elecs.append(
            ElecCalc(
                name=name,
                grid=grid,
                pdie=e["pdie"],
                sdie=e["sdie"],
                temperature=e["temp"],
                ions=e["ions"],
                srad=e.get("srad", 1.4),
                sdens=e.get("sdens", 10.0),
                srfm=e["srfm"],
                swin=e.get("swin", 0.3),
                bcfl=e["bcfl"],
                chgm=e.get("chgm", "spl0"),
                method=method,
                pbe=e["pbe"],
                mol=e.get("mol", 1),
                etol=e.get("etol"),
                calcenergy=e.get("calcenergy", "no"),
                writes=e["writes"],
                requested_dims=req,  # type: ignore[arg-type]
            )
        )
    return ApbsInput(path=path, molecules=molecules, elecs=elecs, prints=prints, notes=notes)


# ============================================================== running APBS


@dataclass
class ApbsCalcOutput:
    """One calculation (ELEC statement) of an APBS run: its finest grid and energy in kJ/mol."""

    index: int
    name: str
    dims: tuple[int, int, int]
    spacing: tuple[float, float, float]
    lengths: tuple[float, float, float]
    center: tuple[float, float, float]
    energy_kj: Optional[float]
    # ½ q_i u(r_i) of every atom (kJ/mol), with ``calcenergy comps``.
    atom_energies_kj: Optional[list[float]] = None


@dataclass
class ApbsOutput:
    """What :func:`parse_output` reads from APBS stdout: calculations, PRINT results (kJ/mol), warnings."""

    calcs: list[ApbsCalcOutput]
    prints: list[tuple[str, float]]  # (expression as echoed by APBS, kJ/mol)
    warnings: list[str]
    text: str
    # The per-atom forces of the first `print force` (kJ/mol/Å): "qf", "ib", "db", "tot" -> {atom: (x, y, z)}.
    forces: dict[str, dict[int, tuple[float, float, float]]] = field(default_factory=dict)


_RE_CALC = re.compile(r"^CALCULATION #(\d+) \((.*)\): MULTIGRID", re.M)
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_output(text: str) -> ApbsOutput:
    """Parse APBS stdout: per-calculation grids and energies, print results, warnings."""
    calcs: list[ApbsCalcOutput] = []
    starts = list(_RE_CALC.finditer(text))
    tail = text.find("PRINT STATEMENTS")
    for n, m in enumerate(starts):
        end = starts[n + 1].start() if n + 1 < len(starts) else (tail if tail >= 0 else len(text))
        block = text[m.start() : end]

        def triple(label: str, conv: Any, block: str = block) -> tuple[Any, Any, Any]:
            mm = re.search(rf"{label}:\s*\(?\s*({_NUM})\s*[x,]\s*({_NUM})\s*[x,]\s*({_NUM})", block)
            if mm is None:
                raise ApbsError(f"cannot find '{label}' for calculation #{m.group(1)} in APBS output")
            return (conv(mm.group(1)), conv(mm.group(2)), conv(mm.group(3)))

        em = re.search(rf"Total electrostatic energy = ({_NUM}) kJ/mol", block)
        atom_lines = re.findall(rf"^\s+Atom (\d+):\s+({_NUM}) kJ/mol\s*$", block, re.M)
        calcs.append(
            ApbsCalcOutput(
                index=int(m.group(1)),
                name=m.group(2),
                dims=triple("Grid dimensions", int),
                spacing=triple("Grid spacings", float),
                lengths=triple("Grid lengths", float),
                center=triple("Grid center", float),
                energy_kj=float(em.group(1)) if em else None,
                atom_energies_kj=[float(v) for _, v in atom_lines] or None,
            )
        )
    prints = [
        (m.group(1).strip(), float(m.group(2)))
        for m in re.finditer(rf"print energy (.*?) end\s*\n(?:.*\n)*?\s*Global net ELEC energy = ({_NUM}) kJ/mol", text)
    ]
    warnings = sorted(set(re.findall(r"^(NOsh:.*(?:Bad|Reset).*|.*off the mesh.*)$", text, re.M)))
    forces: dict[str, dict[int, tuple[float, float, float]]] = {}
    at = text.find("Printing per-atom forces (kJ/mol/A).")
    if at >= 0:
        for m in re.finditer(rf"^\s+(qf|ib|db|tot)\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$", text[at:], re.M):
            forces.setdefault(m.group(1), {})[int(m.group(2))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
    return ApbsOutput(calcs=calcs, prints=prints, warnings=warnings, text=text, forces=forces)


def find_apbs(exe: Optional[str] = None) -> Optional[str]:
    """Path of the APBS executable (``exe``, ``$APBS`` or ``apbs`` on PATH)."""
    return shutil.which(exe or os.environ.get("APBS", "apbs"))


def run_apbs(input_text: str, workdir: PathLike, exe: Optional[str] = None, name: str = "apbs") -> ApbsOutput:
    """Write ``<name>.in`` into ``workdir``, run APBS there and parse its output."""
    work = pathlib.Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.in").write_text(input_text)
    path = find_apbs(exe)
    if path is None:
        raise ApbsError("APBS executable not found (install 'apbs' from conda-forge or set $APBS)")
    proc = subprocess.run([path, f"{name}.in"], cwd=work, capture_output=True, text=True, check=False)
    text = proc.stdout + proc.stderr
    (work / f"{name}.out").write_text(text)
    if proc.returncode != 0:
        raise ApbsError(f"APBS exited with code {proc.returncode}; see {work / (name + '.out')}")
    return parse_output(text)


def _check_grids(elec: ElecCalc, calcs: Sequence[ApbsCalcOutput]) -> list[str]:
    """Differences between APBS' grids and pbsolve's levels for one elec."""
    levels = elec.grid.levels
    if len(levels) == 1 and len(calcs) == 2:  # unfocused mg-auto: APBS repeats the grid
        levels = levels * 2
    if len(calcs) != len(levels):
        return [f"{elec.name}: APBS ran {len(calcs)} focusing levels, pbsolve {len(levels)}"]
    issues = []
    for n, (c, lv) in enumerate(zip(calcs, levels)):
        if tuple(c.dims) != tuple(lv["dims"]):
            issues.append(f"{elec.name} level {n}: dims APBS {c.dims} vs pbsolve {tuple(lv['dims'])}")
        for key in ("spacing", "lengths", "center"):
            theirs, ours = getattr(c, key), lv[key]
            if any(abs(a - b) > _GRID_PRINT_TOL for a, b in zip(theirs, ours)):
                issues.append(f"{elec.name} level {n}: {key} APBS {theirs} vs pbsolve {[round(v, 4) for v in ours]}")
    return issues


def _split_calcs(elecs: Sequence[ElecCalc], out: ApbsOutput) -> list[list[ApbsCalcOutput]]:
    want = sum(e.apbs_calc_count for e in elecs)
    if len(out.calcs) != want:
        raise GridMismatchError(f"APBS ran {len(out.calcs)} calculations, expected {want}")
    groups, i = [], 0
    for e in elecs:
        groups.append(out.calcs[i : i + e.apbs_calc_count])
        i += e.apbs_calc_count
    return groups


# ============================================================== comparisons


def _kj_per_kcal(temperature: float) -> float:
    return kt_kj_per_mol(temperature) / kt_kcal_per_mol(temperature)


def _rel(a: float, b: float) -> float:
    return (a - b) / abs(b) if b != 0 else (0.0 if a == 0 else math.inf)


def _interp(field_: npt.NDArray[np.float64], origin: Sequence[float], h: Sequence[float], p: npt.NDArray[np.float64]):
    f = (p - np.asarray(origin)) / np.asarray(h)
    i0 = np.clip(np.floor(f).astype(int), 0, np.array(field_.shape) - 2)
    t = f - i0
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (t[:, 0] if dx else 1 - t[:, 0])
                    * (t[:, 1] if dy else 1 - t[:, 1])
                    * (t[:, 2] if dz else 1 - t[:, 2])
                )
                out += w * field_[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def _field_diff(ours: npt.NDArray[np.float64], theirs: npt.NDArray[np.float64]) -> dict[str, float]:
    d = ours - theirs
    scale = float(np.max(np.abs(theirs))) or 1.0
    return {
        "max_abs": float(np.max(np.abs(d))),
        "rms": float(np.sqrt(np.mean(d * d))),
        "max_rel_to_max": float(np.max(np.abs(d)) / scale),
        "scale": scale,
    }


@dataclass
class ComparisonReport:
    """Layered pbsolve-vs-APBS comparison of one molecule (see module docstring)."""

    pqr: str
    workdir: str
    platform: str
    grid: dict[str, Any]
    grid_issues: list[str]
    apbs_warnings: list[str]
    energies: dict[str, dict[str, float]]  # state -> {apbs_kj, pbsolve_kj, rel, rel_unit_corrected}
    delta_g: dict[str, float]  # apbs/pbsolve in kcal/mol, diff, rel, rel_unit_corrected
    maps: dict[str, Any] = field(default_factory=dict)
    injection: dict[str, Any] = field(default_factory=dict)
    unit_factor: float = 1.0  # apbs_unit_factor(T): expected E_APBS / E_pbsolve
    # Per-atom forces (if compared): term -> {max_abs, max_rel_to_max} in kJ/mol/Å, APBS units.
    forces: dict[str, Any] = field(default_factory=dict)
    # Per-atom ΔG_polar (if compared): {largest, max_abs, max_rel_to_max} in kcal/mol, APBS units.
    atom_energies: dict[str, Any] = field(default_factory=dict)

    def passed(self, energy_rtol: float = 1e-3) -> bool:
        """Grids identical, maps (if compared) identical up to ties, injection
        potential within 1e-4 of the potential scale, forces (if compared)
        within 1e-5 of the largest force, per-atom ΔG_polar (if compared)
        within 1e-5 of the largest, and ``|ΔG|`` relative difference ≤
        ``energy_rtol``."""
        if self.grid_issues:
            return False
        m = self.maps
        if m:
            if any(m[k]["mismatch_fraction"] > 1e-5 for k in ("eps_x", "eps_y", "eps_z", "kappa")):
                return False
            if m["charge"]["max_rel_to_max"] > 1e-5:
                return False
        for inj in self.injection.values():
            if inj["potential"]["max_rel_to_max"] > 1e-4:
                return False
        if any(v["max_rel_to_max"] > 1e-5 for k, v in self.forces.items() if k != "largest"):
            return False
        if self.atom_energies.get("max_rel_to_max", 0.0) > 1e-5:
            return False
        return abs(self.delta_g["rel"]) <= energy_rtol

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data (JSON-serialisable)."""
        return dataclasses.asdict(self)

    def format(self) -> str:
        """The report as text, as ``pbsolve apbs compare`` prints it."""
        lines = [f"pbsolve vs APBS — {self.pqr}  (pbsolve on {self.platform}, files in {self.workdir})"]
        g = self.grid
        lines.append(
            f"  grid: dime {g['dims']}, fine h = {[round(v, 4) for v in g['fine_spacing']]} Å, "
            f"{g['levels']} level(s): {'IDENTICAL' if not self.grid_issues else 'MISMATCH'}"
        )
        lines += [f"    ! {s}" for s in self.grid_issues]
        lines += [f"    APBS: {w}" for w in self.apbs_warnings]
        if self.maps:
            lines.append("  maps (finest level):")
            for k in ("eps_x", "eps_y", "eps_z", "kappa"):
                m = self.maps[k]
                lines.append(f"    {k:<8} {m['mismatch']:>8d} / {m['total']} nodes differ ({m['mismatch_fraction']:.2e})")
            c = self.maps["charge"]
            lines.append(f"    charge   max |Δq| = {c['max_abs']:.2e} e  ({c['max_rel_to_max']:.1e} of max)")
            for st, p in self.maps.get("potential", {}).items():
                lines.append(
                    f"    u_{st:<10} max |Δu| = {p['max_abs']:.3e} kT/e ({p['max_rel_to_max']:.1e} of max), "
                    f"at atoms {p['atoms_max_abs']:.3e}"
                )
        for st, inj in self.injection.items():
            p = inj["potential"]
            energy = (
                f"energy {inj['energy_rel']:+.2e} rel ({inj['energy_rel_unit_corrected']:+.2e} in APBS units)"
                if "energy_rel" in inj
                else "energy not compared (npbe: APBS adds the coarser levels' terms)"
            )
            lines.append(
                f"  injection ({st}): pbsolve solver on APBS maps -> max |Δu| = {p['max_abs']:.2e} kT/e "
                f"({p['max_rel_to_max']:.1e} of max); {energy}"
            )
        lines.append("  energies (kJ/mol):          APBS         pbsolve     rel. diff   (APBS units)")
        for st, e in self.energies.items():
            lines.append(
                f"    {st:<16} {e['apbs_kj']:>14.4f} {e['pbsolve_kj']:>14.4f}   {e['rel']:+.2e}"
                f"   {e['rel_unit_corrected']:+.2e}"
            )
        d = self.delta_g
        lines.append(
            f"  ΔG_polar (kcal/mol): APBS {d['apbs']:.4f}   pbsolve {d['pbsolve']:.4f}   "
            f"diff {d['diff']:+.4f} ({100 * d['rel']:+.4f} %; {100 * d['rel_unit_corrected']:+.4f} % in APBS units)"
        )
        if self.forces:
            lines.append(
                f"  forces (kJ/mol/Å, APBS units; largest |F| {self.forces['largest']:.4g}): "
                + ", ".join(
                    f"{k} max |ΔF| {v['max_abs']:.2e} ({v['max_rel_to_max']:.1e})"
                    for k, v in self.forces.items()
                    if k != "largest"
                )
            )
        a = self.atom_energies
        if a:
            lines.append(
                f"  per-atom ΔG_polar (kcal/mol, APBS units; largest {a['largest']:.4g}): "
                f"max |Δ| {a['max_abs']:.2e} ({a['max_rel_to_max']:.1e})"
            )
        lines.append(f"  verdict: {'PASS' if self.passed() else 'FAIL'}")
        return "\n".join(lines)


def compare(
    pqr: PathLike,
    solver: Solver,
    *,
    grid: Optional[GridSpec] = None,
    workdir: PathLike = "apbs_compare",
    maps: bool = True,
    inject: bool = True,
    exe: Optional[str] = None,
    etol: Optional[float] = None,
    atoms: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None,
    forces: bool = False,
    atom_energies: bool = False,
) -> ComparisonReport:
    """Run pbsolve and APBS on ``pqr`` with ``solver``'s settings and compare.

    ``forces=True`` also compares the per-atom forces (APBS ``calcforce
    comps``, ``print force``) with :meth:`Solver.polar_forces`; it needs
    ``surface="spl2"`` or ``"spl4"``. ``atom_energies=True`` compares every
    atom's ΔG_polar (APBS ``calcenergy comps``, solvated minus reference) with
    :attr:`PolarResult.atom_energies`; not with ``chgm="spl2"``, whose energy
    APBS takes with other weights than its per-atom energies.

    ``grid`` defaults to ``solver.make_grid`` — build the solver with
    ``dime_policy="apbs"`` so APBS accepts it. Raises :class:`GridMismatchError`
    if APBS ran different grids. ``atoms`` (coords, charges, radii) overrides
    reading ``pqr`` for pbsolve (values must match the file).
    """
    if atom_energies and solver.params["chgm"] == "spl2":
        raise ValueError("atom_energies: APBS takes the spl2 energy with other weights than its per-atom energies")
    pqr = pathlib.Path(pqr).resolve()
    work = pathlib.Path(workdir)
    coords, charges, radii = atoms if atoms is not None else read_pqr(pqr)
    if grid is None:
        grid = solver.make_grid(coords, radii)
    try:
        grid.to_apbs()
    except ValueError as exc:
        raise GridMismatchError(f"{exc} (e.g. Solver(..., dime_policy='apbs'))") from exc
    T = solver.params["temperature"]
    writes = [("pot", "solvated_pot")]
    if maps or inject:
        writes += [(k, f"solvated_{k}") for k in ("dielx", "diely", "dielz", "kappa", "charge")]
    solv = elec_from_solver(solver, grid, "solvated", etol=etol, writes=writes)
    refr = elec_from_solver(solver, grid, "reference", reference=True, etol=etol, writes=[("pot", "reference_pot")])
    if forces:
        solv.calcforce = refr.calcforce = "comps"
    if atom_energies:
        solv.calcenergy = refr.calcenergy = "comps"
    force_prints = ["solvated - reference"] if forces else []
    out = run_apbs(write_input([pqr], [solv, refr], ["solvated - reference"], force_prints), work, exe)
    groups = _split_calcs([solv, refr], out)
    issues = _check_grids(solv, groups[0]) + _check_grids(refr, groups[1])
    if issues:
        raise GridMismatchError("APBS used different grids:\n  " + "\n  ".join(issues))

    kj = kt_kj_per_mol(T)
    uf = apbs_unit_factor(T)
    pot_s, origin, h, e_s = solver.potential(coords, charges, radii, grid=grid)
    pot_r, _, _, e_r = solver.potential(coords, charges, radii, grid=grid, reference=True)
    energies: dict[str, dict[str, float]] = {}
    for state, ours_kt, grp in (("solvated", e_s, groups[0]), ("reference", e_r, groups[1])):
        theirs = grp[-1].energy_kj
        if theirs is None:
            raise ApbsError(f"APBS printed no energy for '{state}'")
        energies[state] = {
            "apbs_kj": theirs,
            "pbsolve_kj": ours_kt * kj,
            "rel": _rel(ours_kt * kj, theirs),
            "rel_unit_corrected": _rel(ours_kt * kj * uf, theirs),
        }
    to_kcal = 1.0 / _kj_per_kcal(T)
    dg_a = (energies["solvated"]["apbs_kj"] - energies["reference"]["apbs_kj"]) * to_kcal
    dg_p = (e_s - e_r) * kt_kcal_per_mol(T)
    report = ComparisonReport(
        pqr=str(pqr),
        workdir=str(work),
        platform=solver.platform,
        grid={
            "dims": list(grid.dims),
            "fine_spacing": list(grid.fine_spacing),
            "levels": len(grid.levels),
            "center": list(grid.center),
        },
        grid_issues=issues,
        apbs_warnings=out.warnings,
        energies=energies,
        delta_g={
            "apbs": dg_a,
            "pbsolve": dg_p,
            "diff": dg_p - dg_a,
            "rel": _rel(dg_p, dg_a),
            "rel_unit_corrected": _rel(dg_p * uf, dg_a),
        },
        unit_factor=uf,
    )
    if forces:
        report.forces = _compare_forces(solver, (coords, charges, radii), grid, out, T)
    if atom_energies:
        s_at, r_at = groups[0][-1].atom_energies_kj, groups[1][-1].atom_energies_kj
        if s_at is None or r_at is None or len(s_at) != len(charges) or len(r_at) != len(charges):
            raise ApbsError("APBS printed no per-atom energy for every atom")
        theirs_at = (np.array(s_at) - np.array(r_at)) * to_kcal
        ours_at = solver.polar_solvation(coords, charges, radii, grid=grid).atom_energies
        if ours_at is None:
            raise ValueError("atom_energies: the nonlinear equation with ions has no per-atom energies")
        diff = np.abs(ours_at * uf - theirs_at)
        largest = float(np.max(np.abs(theirs_at)))
        report.atom_energies = {
            "largest": largest,
            "max_abs": float(diff.max()),
            "max_rel_to_max": float(diff.max() / largest) if largest > 0 else 0.0,
        }

    if maps or inject:
        spline = solver.params["surface"] in ("spl2", "spl4")
        theirs_maps = {k: read_dx(work / f"solvated_{k}.dx") for k in ("dielx", "diely", "dielz", "kappa", "charge")}
        ours = solver.maps(coords, charges, radii, grid=grid)
        vol = float(np.prod(ours["spacing"]))
        apbs_q = theirs_maps["charge"][0] * vol
        apbs_pot = {st: read_dx(work / f"{st}_pot.dx")[0] for st in ("solvated", "reference")}
        if maps:
            res: dict[str, Any] = {}
            pdie_sdie = max(solver.params["pdie"], solver.params["sdie"])
            for ok, ak in (("eps_x", "dielx"), ("eps_y", "diely"), ("eps_z", "dielz")):
                bad = np.abs(ours[ok] - theirs_maps[ak][0]) > 1e-6 * pdie_sdie
                res[ok] = {"mismatch": int(bad.sum()), "total": int(bad.size), "mismatch_fraction": float(bad.mean())}
            if ours["kappa_bar_sq"] == 0:
                bad = np.zeros(theirs_maps["kappa"][0].shape, dtype=bool)
            elif spline:
                # A continuous accessibility, which APBS writes with 6 significant digits.
                bad = np.abs(ours["kappa_sq"] / ours["kappa_bar_sq"] - theirs_maps["kappa"][0]) > 1e-6
            else:
                bad = (ours["kappa_sq"] > 0) != (theirs_maps["kappa"][0] > 0.5)
            res["kappa"] = {"mismatch": int(bad.sum()), "total": int(bad.size), "mismatch_fraction": float(bad.mean())}
            res["charge"] = _field_diff(ours["charge"], apbs_q)
            res["potential"] = {}
            for st, ours_pot in (("solvated", pot_s), ("reference", pot_r)):
                d = _field_diff(ours_pot, apbs_pot[st])
                at = _interp(ours_pot, origin, h, coords) - _interp(apbs_pot[st], origin, h, coords)
                d["atoms_max_abs"] = float(np.max(np.abs(at)))
                res["potential"][st] = d
            report.maps = res
        if inject:
            eps = {"solvated": [theirs_maps[k][0] for k in ("dielx", "diely", "dielz")]}
            pdie = solver.params["pdie"]
            eps["reference"] = [np.full_like(apbs_q, pdie)] * 3
            accessible = theirs_maps["kappa"][0] if spline else theirs_maps["kappa"][0] > 0.5
            kap = {
                "solvated": accessible * ours["kappa_bar_sq"],
                "reference": np.zeros_like(apbs_q),
            }
            for st in ("solvated", "reference"):
                u, iters, resid = solver.solve_maps(*eps[st], kap[st], apbs_q, apbs_pot[st], ours["origin"], ours["spacing"])
                inj: dict[str, Any] = {"potential": _field_diff(u, apbs_pot[st]), "iterations": iters}
                if st == "solvated" and solver.params["equation"] == "npbe" and ours["kappa_bar_sq"] > 0:
                    # APBS' nonlinear energy adds terms of the coarser levels, which one level lacks.
                    report.injection[st] = inj
                    continue
                if solver.params["chgm"] == "spl4":
                    # APBS takes the spl4 energy at the atoms with trilinear weights (Vpmg_qfEnergy).
                    e_inj = 0.5 * float(np.sum(charges * _interp(u, ours["origin"], ours["spacing"], coords))) * kj
                else:
                    e_inj = 0.5 * float(np.sum(apbs_q * u)) * kj
                inj.update(
                    energy_kj=e_inj,
                    energy_rel=_rel(e_inj, energies[st]["apbs_kj"]),
                    energy_rel_unit_corrected=_rel(e_inj * uf, energies[st]["apbs_kj"]),
                )
                report.injection[st] = inj
    (work / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report


def _compare_forces(
    solver: Solver,
    atoms: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    grid: GridSpec,
    out: ApbsOutput,
    temperature: float,
) -> dict[str, Any]:
    """pbsolve's forces against APBS' ``print force`` (kJ/mol/Å, in APBS units)."""
    if not out.forces:
        raise ApbsError("APBS printed no per-atom forces")
    ours = solver.polar_forces(*atoms, grid=grid)
    scale = _kj_per_kcal(temperature) * apbs_unit_factor(temperature)
    n = len(atoms[1])
    theirs = {k: np.array([out.forces[k][i] for i in range(n)]) for k in ("qf", "db", "ib", "tot")}
    largest = float(np.abs(theirs["tot"]).max())
    res: dict[str, Any] = {"largest": largest}
    for k, mine in (("qf", ours.qf), ("db", ours.db), ("ib", ours.ib), ("tot", ours.forces)):
        d = float(np.abs(mine * scale - theirs[k]).max())
        res[k] = {"max_abs": d, "max_rel_to_max": d / largest if largest > 0 else d}
    return res


@dataclass
class BindingReport:
    """Complex/receptor/ligand comparison on the complex's grid (MM-PBSA)."""

    complex: ComparisonReport
    receptor: ComparisonReport
    ligand: ComparisonReport

    @property
    def ddg(self) -> dict[str, float]:
        """ΔΔG_polar of binding (complex − receptor − ligand, kcal/mol) from APBS and pbsolve, and their differences."""
        a = self.complex.delta_g["apbs"] - self.receptor.delta_g["apbs"] - self.ligand.delta_g["apbs"]
        p = self.complex.delta_g["pbsolve"] - self.receptor.delta_g["pbsolve"] - self.ligand.delta_g["pbsolve"]
        uf = self.complex.unit_factor
        return {
            "apbs": a,
            "pbsolve": p,
            "diff": p - a,
            "rel": _rel(p, a),
            "diff_unit_corrected": p * uf - a,
        }

    def passed(self, energy_rtol: float = 1e-3, ddg_atol: float = 0.05) -> bool:
        """All three comparisons pass and the ΔΔG_polar values differ by at most
        ``ddg_atol`` kcal/mol in APBS units.

        The raw difference also contains the constant factor
        :func:`apbs_unit_factor` times ΔΔG, which alone exceeds 0.05 kcal/mol
        once ΔΔG passes about ±4000 kcal/mol (large complexes of like-charged
        partners), so it is removed before the check."""
        ok = all(r.passed(energy_rtol) for r in (self.complex, self.receptor, self.ligand))
        return ok and abs(self.ddg["diff_unit_corrected"]) <= ddg_atol

    def format(self) -> str:
        """The report as text, with the binding verdict."""
        parts = []
        for name in ("complex", "receptor", "ligand"):
            parts.append(f"[{name}]\n" + getattr(self, name).format())
        d = self.ddg
        parts.append(
            f"ΔΔG_polar(bind) (kcal/mol): APBS {d['apbs']:.4f}   pbsolve {d['pbsolve']:.4f}   "
            f"diff {d['diff']:+.4f} ({d['diff_unit_corrected']:+.5f} in APBS units)"
            f"\nbinding verdict: {'PASS' if self.passed() else 'FAIL'}"
        )
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data (JSON-serialisable)."""
        return {
            "complex": self.complex.to_dict(),
            "receptor": self.receptor.to_dict(),
            "ligand": self.ligand.to_dict(),
            "ddg": self.ddg,
        }


def _pqr_records(path: PathLike) -> list[str]:
    with open(path) as fh:
        return [ln for ln in fh if ln.split()[:1] in (["ATOM"], ["HETATM"])]


def _chain_of(record: str) -> Optional[str]:
    f = record.split()
    return f[4] if len(f) >= 11 else None


def compare_binding(
    pqr: PathLike,
    solver: Solver,
    ligand: npt.ArrayLike,
    *,
    workdir: PathLike = "apbs_compare",
    **kwargs: Any,
) -> BindingReport:
    """Compare complex, receptor and ligand (``ligand``: atom indices or a
    boolean mask) on the complex's grid, as in single-trajectory MM-PBSA."""
    pqr = pathlib.Path(pqr).resolve()
    work = pathlib.Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    coords, charges, radii = read_pqr(pqr)
    records = _pqr_records(pqr)
    if len(records) != len(charges):
        raise ApbsError("could not match PQR records to atoms")
    mask = np.zeros(len(charges), dtype=bool)
    lig = np.asarray(ligand)
    if lig.dtype == bool:
        mask[:] = lig
    else:
        mask[lig.astype(int)] = True
    if not mask.any() or mask.all():
        raise ValueError("the ligand selection must be a non-empty proper subset of the atoms")
    grid = kwargs.pop("grid", None) or solver.make_grid(coords, radii)
    reports = {}
    for name, sel in (("complex", np.ones_like(mask)), ("receptor", ~mask), ("ligand", mask)):
        sub = work / name
        sub.mkdir(exist_ok=True)
        path = sub / f"{name}.pqr"
        path.write_text("".join(r for r, s in zip(records, sel) if s) + "END\n")
        reports[name] = compare(
            path, solver, grid=grid, workdir=sub, atoms=(coords[sel], charges[sel], radii[sel]), **kwargs
        )
    rep = BindingReport(**reports)
    (work / "binding_report.json").write_text(json.dumps(rep.to_dict(), indent=2))
    return rep


# ============================================================== APBS input -> pbsolve


def read_parm(path: PathLike) -> dict[tuple[str, str], tuple[float, float, float]]:
    """An APBS flat parameter file (``read parm flat``).

    Parameters
    ----------
    path : str or os.PathLike
        The file: lines ``<residue> <atom> <charge> <radius> <epsilon>``,
        ``#`` comments.

    Returns
    -------
    dict
        ``(residue, atom)`` → ``(charge in e, radius in Å, epsilon in
        kJ/mol)``, as APBS reads them; with a parameter file, APBS replaces
        the radii of a PQR file by these.
    """
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split("#", 1)[0].split()
        if len(f) >= 5:
            out[(f[0], f[1])] = (float(f[2]), float(f[3]), float(f[4]))
    return out


def apolar_energy(
    coords: npt.ArrayLike,
    radii: npt.ArrayLike,
    epsilons: Optional[npt.ArrayLike] = None,
    *,
    gamma: float = _apolar.GAMMA,
    press: float = _apolar.PRESS,
    bconc: float = _apolar.BCONC,
    srad: float = 0.65,
    sdens: float = 100.0,
    dpos: float = 0.2,
    grid: float = 0.5,
    water_radius: float = _apolar.WATER_RADIUS,
    water_epsilon: float = _apolar.WATER_EPSILON,
    workdir: Optional[PathLike] = None,
    exe: Optional[str] = None,
) -> _apolar.ApolarEnergy:
    """APBS' ``apolar`` result for the input of :func:`pbsolve.apolar_energy`.

    Every atom becomes a residue of its own in a PQR file with a matching
    flat parameter file, so APBS sees the same radii and well depths. Each
    term runs as its own ``APOLAR`` calculation with only its coefficient
    set to 1 (area, volume) or ``bconc`` (dispersion), so APBS prints it with
    13 digits.

    Parameters
    ----------
    coords, radii, epsilons, gamma, press, bconc, srad, sdens, dpos, grid, water_radius, water_epsilon
        As :func:`pbsolve.apolar_energy` (kcal/mol units).
    workdir : str or os.PathLike, optional
        Where to run APBS (default: a temporary directory, removed after).
    exe : str, optional
        The APBS executable (default: ``$APBS`` or ``apbs`` on the PATH).

    Returns
    -------
    pbsolve.ApolarEnergy
        APBS' terms, converted to kcal/mol.

    Raises
    ------
    ApbsError
        If APBS is missing, fails or prints no energy.
    """
    import tempfile

    x = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(radii, dtype=np.float64).ravel()
    n = len(r)
    e = np.zeros(n) if epsilons is None else np.asarray(epsilons, dtype=np.float64).ravel()
    kj = _apolar.KJ_PER_KCAL
    use = [abs(c) > 1e-9 for c in (gamma, press, bconc)]
    if use[2] and epsilons is None:
        raise ValueError("the dispersion term (bconc != 0) needs the atoms' Lennard-Jones epsilons")
    xs, rs, es = x.tolist(), r.tolist(), (e * kj).tolist()  # Python floats: repr gives every digit
    pqr = "".join(f"ATOM {i + 1} A R{i + 1} {i + 1} {p[0]!r} {p[1]!r} {p[2]!r} 0.0 {rs[i]!r}\n" for i, p in enumerate(xs))
    parm = "".join(f"R{i + 1} A 0.0 {rs[i]!r} {es[i]!r}\n" for i in range(n))
    parm += f"WAT OW 0.0 {float(water_radius)!r} {float(water_epsilon) * kj!r}\n"
    total = re.compile(rf"Total non-polar energy = ({_NUM}) kJ/mol")

    def run(work: pathlib.Path, name: str, g: float, p: float, b: float, per_atom: str) -> tuple[float, npt.NDArray[np.float64]]:
        text = f"""read
    mol pqr mol.pqr
    parm flat parm.dat
end
apolar name {name}
    grid {float(grid)!r} {float(grid)!r} {float(grid)!r}
    mol 1
    srfm sacc
    swin 0.3
    srad {float(srad)!r}
    press {float(p)!r}
    gamma {float(g)!r}
    bconc {float(b)!r}
    sdens {float(sdens)!r}
    dpos {float(dpos)!r}
    temp 298.15
    calcenergy total
    calcforce no
end
quit
"""
        out = run_apbs(text, work, exe, name=name).text
        m = total.search(out)
        if m is None:
            raise ApbsError(f"APBS printed no non-polar energy; see {work / (name + '.out')}")
        atoms = np.zeros(n)
        for mm in re.finditer(rf"{per_atom} for atom (\d+): ({_NUM})", out):
            atoms[int(mm.group(1))] = float(mm.group(2))
        return float(m.group(1)), atoms

    nan = float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(workdir if workdir is not None else tmp)
        work.mkdir(parents=True, exist_ok=True)
        (work / "mol.pqr").write_text(pqr)
        (work / "parm.dat").write_text(parm)
        area, atom_area = run(work, "area", 1.0, 0.0, 0.0, "SASA") if use[0] else (nan, np.full(n, nan))
        volume = run(work, "volume", 0.0, 1.0, 0.0, "SASA")[0] if use[1] else nan
        disp, atom_disp = run(work, "dispersion", 0.0, 0.0, bconc, "WCA energy") if use[2] else (0.0, np.zeros(n))
    surface = gamma * area if use[0] else 0.0
    pv = press * volume if use[1] else 0.0
    return _apolar.ApolarEnergy(
        energy=surface + pv + disp / kj,
        surface=surface,
        pressure_volume=pv,
        dispersion=disp / kj,
        area=area,
        volume=volume,
        atom_area=atom_area,
        atom_dispersion=atom_disp / kj,
    )


@dataclass
class RunReport:
    """An APBS input file evaluated by pbsolve (and APBS)."""

    input: str
    notes: list[str]
    elecs: list[dict[str, Any]]  # name, pbsolve_kj, apbs_kj, rel, rel_unit_corrected, grid_issues
    prints: list[dict[str, Any]]  # expr, pbsolve_kj, apbs_kj, rel, rel_unit_corrected

    def passed(self, energy_rtol: float = 1e-3) -> bool:
        """Grids identical and every energy APBS reports matched within ``energy_rtol`` (relative)."""
        if any(e["grid_issues"] for e in self.elecs):
            return False
        vals = [p for p in self.prints if p.get("apbs_kj") is not None]
        vals += [e for e in self.elecs if e.get("apbs_kj") is not None]
        return bool(vals) and all(abs(v["rel"]) <= energy_rtol for v in vals)

    def format(self) -> str:
        """The report as text, as ``pbsolve apbs run`` prints it."""
        lines = [f"APBS input {self.input} evaluated by pbsolve"]
        lines += [f"  note: {n}" for n in self.notes]
        lines.append("  calculation              pbsolve (kJ/mol)   APBS (kJ/mol)   rel. diff  (APBS units)")

        def row(label: str, d: dict[str, Any]) -> str:
            a = d.get("apbs_kj")
            theirs = (
                f"{a:>15.4f}   {d['rel']:+.2e}  {d['rel_unit_corrected']:+.2e}" if a is not None else f"{'-':>15}"
            )
            return f"  {label:<24} {d['pbsolve_kj']:>16.4f} {theirs}"

        for e in self.elecs:
            lines.append(row(e["name"], e))
            lines += [f"    ! {s}" for s in e["grid_issues"]]
        for p in self.prints:
            lines.append(row("print " + p["expr"], p))
        if any(e.get("apbs_kj") is not None for e in self.elecs):
            lines.append(f"  verdict: {'PASS' if self.passed() else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data (JSON-serialisable)."""
        return dataclasses.asdict(self)


def run_input(
    path: PathLike,
    *,
    platform: str = "auto",
    with_apbs: bool = True,
    exe: Optional[str] = None,
    workdir: Optional[PathLike] = None,
    preconditioner: Optional[str] = None,
) -> RunReport:
    """Evaluate an APBS input file with pbsolve; with ``with_apbs`` also run
    APBS on the same file and compare energies and grids."""
    inp = parse_input(path)
    mols = [read_pqr(m) for m in inp.molecules]
    out: Optional[ApbsOutput] = None
    if with_apbs:
        text = pathlib.Path(path).read_text()
        src = pathlib.Path(path).resolve()
        work = pathlib.Path(workdir) if workdir else src.parent / f"{src.stem}_apbs"
        # Absolute molecule paths so the copy runs from any directory.
        lines = []
        for ln in text.splitlines():
            t = ln.split()
            if len(t) >= 3 and t[0].lower() == "mol" and t[1].lower() == "pqr":
                p = pathlib.Path(t[2])
                ln = f"    mol pqr {p if p.is_absolute() else (pathlib.Path(path).resolve().parent / p)}"
            lines.append(ln)
        out = run_apbs("\n".join(lines) + "\n", work, exe)
    groups = _split_calcs(inp.elecs, out) if out else [None] * len(inp.elecs)
    elecs: list[dict[str, Any]] = []
    uf = apbs_unit_factor(inp.elecs[0].temperature) if inp.elecs else 1.0
    energies: dict[str, float] = {}
    apbs_e: dict[str, Optional[float]] = {}
    for n, (e, grp) in enumerate(zip(inp.elecs, groups), start=1):
        coords, charges, radii = mols[e.mol - 1]
        s = e.solver(platform=platform, preconditioner=preconditioner)
        _, _, _, ekt = s.potential(coords, charges, radii, grid=e.grid)
        ours = ekt * kt_kj_per_mol(e.temperature)
        theirs = grp[-1].energy_kj if grp else None
        energies[e.name] = energies[str(n)] = ours
        apbs_e[e.name] = apbs_e[str(n)] = theirs
        elecs.append(
            {
                "name": e.name,
                "pbsolve_kj": ours,
                "apbs_kj": theirs,
                "rel": _rel(ours, theirs) if theirs is not None else None,
                "rel_unit_corrected": _rel(ours * uf, theirs) if theirs is not None else None,
                "grid_issues": _check_grids(e, grp) if grp else [],
            }
        )
    prints = []
    for terms in inp.prints:
        expr = " ".join(("- " if s < 0 else ("+ " if i else "")) + t for i, (s, t) in enumerate(terms))
        ours = sum(s * energies[t] for s, t in terms)
        known = all(apbs_e.get(t) is not None for _, t in terms)
        theirs = sum(s * apbs_e[t] for s, t in terms) if known and out else None  # type: ignore[misc]
        prints.append(
            {
                "expr": expr,
                "pbsolve_kj": ours,
                "apbs_kj": theirs,
                "rel": _rel(ours, theirs) if theirs is not None else None,
                "rel_unit_corrected": _rel(ours * uf, theirs) if theirs is not None else None,
            }
        )
    return RunReport(input=str(path), notes=inp.notes, elecs=elecs, prints=prints)


# ============================================================== CLI


def _selection(spec: str, n: int) -> npt.NDArray[np.int64]:
    """'1-20,25' (1-based, inclusive) -> 0-based indices."""
    idx: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        if lo < 1 or hi > n or lo > hi:
            raise ValueError(f"atom range '{part}' outside 1..{n}")
        idx += range(lo - 1, hi)
    return np.asarray(idx, dtype=np.int64)


def _add_solver_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--platform", default="auto")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--pdie", type=float, default=1.0)
    ap.add_argument("--sdie", type=float, default=78.54)
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--ionic-strength", type=float, default=0.0)
    ap.add_argument("--ion-radius", type=float, default=2.0)
    ap.add_argument("--srad", type=float, default=1.4)
    ap.add_argument("--sdens", type=float, default=10.0)
    ap.add_argument("--surface", default="mol", choices=["mol", "vdw", "spl2", "spl4"])
    ap.add_argument("--swin", type=float, default=0.3, help="spl2/spl4: half-width of the switching window (Å)")
    ap.add_argument(
        "--bcfl",
        default="mdh",
        choices=["mdh", "mdh-fast", "sdh", "zero"],
        help="boundary condition (mdh-fast: pbsolve's interpolated mdh; APBS runs mdh)",
    )
    ap.add_argument("--chgm", default="spl0", choices=["spl0", "spl2", "spl4"], help="charge assignment (APBS chgm)")
    ap.add_argument("--equation", default="lpbe", choices=["lpbe", "npbe"], help="linearized or nonlinear PBE")
    ap.add_argument("--cfac", type=float, default=1.7)
    ap.add_argument("--fadd", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=1e-8, help="pbsolve tolerance; also written as APBS etol")
    _add_preconditioner_arg(ap)


def _add_preconditioner_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--preconditioner",
        choices=["multigrid-f32", "multigrid", "jacobi"],
        help="CG preconditioner of pbsolve (default: the Solver default, multigrid-f32)",
    )


def _as_pqr(a: argparse.Namespace) -> str:
    """Convert a non-PQR input (with --coords/--radii) to PQR in the workdir."""
    if a.pqr.lower().endswith(".pqr") and a.coords is None and a.radii == "file" and not a.ff:
        return a.pqr
    from .io import load

    if a.ff:
        from .prep import prepare

        mol, rep = prepare(a.pqr, a.ff)
        print("# " + rep.summary().replace("\n", "\n# "), file=sys.stderr)
    else:
        mol = load(a.pqr, a.coords, include_dirs=a.include or None)
    if not a.keep_solvent:
        mol = mol.strip_solvent()
    if a.radii != "file":
        mol = mol.with_radii(a.radii)
    work = pathlib.Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out = work / (pathlib.Path(a.pqr).stem + ".pqr")
    mol.write_pqr(out)
    print(f"# converted {a.pqr} -> {out} ({mol.n_atoms} atoms)", file=sys.stderr)
    return str(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line ``pbsolve apbs``; returns the exit status."""
    from ._console import utf8_stdio

    utf8_stdio()
    ap = argparse.ArgumentParser(prog="pbsolve apbs", description="Reciprocal validation against APBS")
    sub = ap.add_subparsers(dest="action", required=True)

    c = sub.add_parser("compare", help="run pbsolve and APBS on a PQR file with identical settings")
    c.add_argument("pqr", help="PQR file, or any input pbsolve reads (converted to PQR in the workdir)")
    c.add_argument("--coords", help="coordinates for a topology input (inpcrd, gro, pdb, ...)")
    c.add_argument("--radii", choices=["file", "bondi", "mbondi", "mbondi2", "parse"], default="file")
    c.add_argument("--keep-solvent", action="store_true", help="do not strip water/monovalent ions")
    c.add_argument("-I", "--include", action="append", default=[], help="GROMACS include directory")
    c.add_argument("--ff", choices=["amber", "parse", "charmm"], help="prepare a PDB without charges with pbsolve.prep")
    _add_solver_args(c)
    lig = c.add_mutually_exclusive_group()
    lig.add_argument("--ligand-chain", help="also compare receptor/ligand/ΔΔG; ligand = this PQR chain")
    lig.add_argument("--ligand-atoms", help="ligand atoms as 1-based ranges, e.g. 1450-1670")
    c.add_argument("--workdir", default="apbs_compare")
    c.add_argument("--no-maps", action="store_true", help="skip map comparison and injection (faster)")
    c.add_argument("--forces", action="store_true", help="also compare the per-atom forces (needs --surface spl2/spl4)")
    c.add_argument("--atom-energies", action="store_true", help="also compare every atom's ΔG_polar (not with --chgm spl2)")
    c.add_argument("--apbs", help="APBS executable (default: $APBS or 'apbs' on PATH)")
    c.add_argument("--json", help="also write the report to this JSON file")
    c.add_argument("--energy-rtol", type=float, default=1e-3)

    r = sub.add_parser("run", help="evaluate an existing APBS input file with pbsolve (and APBS)")
    r.add_argument("input")
    r.add_argument("--platform", default="auto")
    r.add_argument("--no-apbs", action="store_true", help="only run pbsolve")
    r.add_argument("--apbs", help="APBS executable")
    r.add_argument("--workdir", help="where to run APBS (default: the input's directory)")
    _add_preconditioner_arg(r)
    r.add_argument("--json")
    r.add_argument("--energy-rtol", type=float, default=1e-3)

    a = ap.parse_args(argv)
    try:
        if a.action == "run":
            rep: Any = run_input(
                a.input,
                platform=a.platform,
                with_apbs=not a.no_apbs,
                exe=a.apbs,
                workdir=a.workdir,
                preconditioner=a.preconditioner,
            )
            ok = a.no_apbs or rep.passed(a.energy_rtol)
        else:
            solver = Solver(
                platform=a.platform,
                grid_spacing=a.spacing,
                pdie=a.pdie,
                sdie=a.sdie,
                temperature=a.temperature,
                ionic_strength=a.ionic_strength,
                ion_radius=a.ion_radius,
                srad=a.srad,
                sdens=a.sdens,
                surface=a.surface,
                swin=a.swin,
                bcfl=a.bcfl,
                chgm=a.chgm,
                equation=a.equation,
                cfac=a.cfac,
                fadd=a.fadd,
                tol=a.tol,
                dime_policy="apbs",
                **_preconditioner(a.preconditioner),
            )
            maps = not a.no_maps
            a.pqr = _as_pqr(a)
            if a.ligand_chain or a.ligand_atoms:
                n = len(read_pqr(a.pqr)[1])
                if a.ligand_chain:
                    chains = [_chain_of(rec) for rec in _pqr_records(a.pqr)]
                    if all(ch is None for ch in chains):
                        raise ValueError("the PQR has no chain IDs; use --ligand-atoms")
                    sel = np.asarray([ch == a.ligand_chain for ch in chains])
                else:
                    sel = _selection(a.ligand_atoms, n)
                rep = compare_binding(a.pqr, solver, sel, workdir=a.workdir, maps=maps, inject=maps, exe=a.apbs)
            else:
                rep = compare(
                    a.pqr, solver, workdir=a.workdir, maps=maps, inject=maps, exe=a.apbs, forces=a.forces,
                    atom_energies=a.atom_energies,
                )
            ok = rep.passed(a.energy_rtol)
    except (ApbsError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(rep.format())
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rep.to_dict(), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
