"""Load simulation output directly: structures, topologies and trajectories.

Supported (file format from the extension, override with ``format=``):

* structures: PDB (multi-model), PQR, mmCIF, GRO, mol2;
* topologies with charges: Amber ``prmtop``, CHARMM/NAMD ``psf``, GROMACS
  ``top``/``itp`` (``#include`` resolved via ``GMXLIB``/``include_dirs``);
* coordinates: Amber ``inpcrd``/``rst7``/``ncrst``;
* trajectories: DCD (CHARMM, NAMD, OpenMM), XTC, TRR, Amber NetCDF, ASCII mdcrd,
  multi-model PDB/GRO/mmCIF.

Typical use::

    import pbsolve
    mol = pbsolve.io.load("complex.prmtop", "complex.inpcrd")   # charges + radii from Amber
    mol = pbsolve.io.load("topol.top", "md.gro", radii="mbondi2")  # GROMACS: radii by scheme
    solute = mol.strip_solvent()
    lig = solute.select("resname LIG")
    for xyz in pbsolve.io.frames(mol, "md.xtc", step=10, atoms=~mol.solvent_mask()):
        ...

Selections (``Molecule.select``): ``all``, ``none``, ``protein``, ``water``,
``ions``, ``hydrogen``, ``hetero``, ``resname R..``, ``name N..``,
``element E..``, ``chain C..``, ``segid S..``, ``resid 1-10 15..``,
``index 0-99`` (0-based), ``serial 1-100`` (1-based), combined with
``and``/``or``/``not`` and parentheses. Names accept ``*``/``?`` wildcards.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from ._core import TrajectoryReader, read_structure
from .radii import SCHEMES, assign_radii

PathLike = Union[str, "os.PathLike[str]"]

__all__ = ["Molecule", "TrajectoryReader", "load", "frames", "write_pqr", "WATER_NAMES", "ION_NAMES"]

#: Residue names treated as water.
WATER_NAMES = frozenset(
    "HOH WAT SOL H2O DOD TIP3 TIP3P TIP4 TIP4P TIP4PEW TIP5 TIP5P T3P T4P T4E T5P SPC SPCE TP3 TP4 TP5 OPC OPC3 "
    "TIP3PFB SWM4".split()
)
#: Monovalent counter-ions (removed by default). Divalent/structural ions are kept unless asked.
MONOVALENT_IONS = frozenset(
    "NA NA+ Na+ SOD K K+ POT CL CL- Cl- CLA LI LI+ Li+ LIT RB RB+ Rb+ CS CS+ Cs+ CES F F- BR BR- Br- I I- IOD".split()
)
DIVALENT_IONS = frozenset("MG MG2 Mg2+ MG2+ CA CA2 Ca2+ CAL ZN ZN2 Zn2+ SR Sr2+ BA Ba2+ CD Cd2+ MN Mn2+".split())
ION_NAMES = MONOVALENT_IONS | DIVALENT_IONS

_AMINO = (
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "HID HIE HIP HSD HSE HSP CYX CYM ASH GLH LYN ARN TYM HYP MSE ACE NME NMA NHE".split()
)
PROTEIN_NAMES = frozenset(_AMINO + ["N" + r for r in _AMINO] + ["C" + r for r in _AMINO])


@dataclass
class Molecule:
    """Atoms with optional charges/radii and coordinates (all in Å, e)."""

    names: npt.NDArray[np.str_]
    resnames: npt.NDArray[np.str_]
    resids: npt.NDArray[np.int64]
    icodes: npt.NDArray[np.str_]
    chains: npt.NDArray[np.str_]
    segids: npt.NDArray[np.str_]
    elements: npt.NDArray[np.str_]
    atom_types: npt.NDArray[np.str_]
    hetero: npt.NDArray[np.bool_]
    charges: npt.NDArray[np.float64]  # NaN = unknown
    radii: npt.NDArray[np.float64]  # NaN = unknown
    masses: npt.NDArray[np.float64]
    bonds: npt.NDArray[np.int64]  # (B, 2)
    coords: Optional[npt.NDArray[np.float64]] = None  # (N, 3), first frame
    models: Optional[npt.NDArray[np.float64]] = None  # (F, N, 3) frames stored in the file
    cells: Optional[npt.NDArray[np.float64]] = None  # (F, 6)
    periodic: Optional[bool] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- basics
    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.names.shape[0])

    def __len__(self) -> int:
        return self.n_atoms

    @property
    def has_charges(self) -> bool:
        """Whether every atom has a charge."""
        return bool(self.n_atoms) and not np.isnan(self.charges).any()

    @property
    def has_radii(self) -> bool:
        """Whether every atom has a radius."""
        return bool(self.n_atoms) and not np.isnan(self.radii).any()

    def __repr__(self) -> str:
        what = [f"{self.n_atoms} atoms"]
        if self.has_charges:
            what.append(f"net charge {np.sum(self.charges):+.3f} e")
        what.append("radii" if self.has_radii else "no radii")
        if self.models is not None:
            what.append(f"{self.models.shape[0]} frame(s)")
        return f"Molecule({', '.join(what)}, source='{self.source}')"

    # ------------------------------------------------------------- subsets
    def subset(self, sel: Union[str, npt.ArrayLike]) -> "Molecule":
        """New Molecule with the selected atoms (selection string, mask or indices)."""
        mask = self._mask(sel)
        idx = np.nonzero(mask)[0]
        remap = -np.ones(self.n_atoms, dtype=np.int64)
        remap[idx] = np.arange(idx.size)
        b = self.bonds
        keep = mask[b[:, 0]] & mask[b[:, 1]] if b.size else np.zeros(0, dtype=bool)
        arrays = {
            k: getattr(self, k)[idx]
            for k in (
                "names", "resnames", "resids", "icodes", "chains", "segids", "elements", "atom_types",
                "hetero", "charges", "radii", "masses",
            )
        }
        return replace(
            self,
            **arrays,
            bonds=remap[b[keep]] if b.size else b,
            coords=None if self.coords is None else self.coords[idx],
            models=None if self.models is None else self.models[:, idx],
            notes=list(self.notes),
        )

    def _mask(self, sel: Union[str, npt.ArrayLike]) -> npt.NDArray[np.bool_]:
        if isinstance(sel, str):
            return self.select(sel)
        a = np.asarray(sel)
        if a.dtype == bool:
            if a.shape != (self.n_atoms,):
                raise ValueError(f"mask has shape {a.shape}, expected ({self.n_atoms},)")
            return a
        m = np.zeros(self.n_atoms, dtype=bool)
        m[a.astype(np.int64)] = True
        return m

    def solvent_mask(self, water: bool = True, ions: str = "monovalent") -> npt.NDArray[np.bool_]:
        """Atoms of water and/or ions. ``ions``: 'monovalent' (default), 'all' or 'none'."""
        if ions not in ("monovalent", "all", "none"):
            raise ValueError("ions must be 'monovalent', 'all' or 'none'")
        names = set()
        if water:
            names |= WATER_NAMES
        if ions == "monovalent":
            names |= MONOVALENT_IONS
        elif ions == "all":
            names |= ION_NAMES
        upper = {n.upper() for n in names}
        res = np.array([r.upper() in upper for r in self.resnames], dtype=bool)
        if ions != "none":
            # single-atom residues named like an ion (e.g. GROMACS "NA" in residue "NA")
            res |= np.array([r == n and r.upper() in {x.upper() for x in ION_NAMES} for r, n in zip(self.resnames, self.names)])
        return res

    def strip_solvent(self, water: bool = True, ions: str = "monovalent") -> "Molecule":
        """Remove water and (by default monovalent) ions, as done before implicit-solvent PB."""
        return self.subset(~self.solvent_mask(water, ions))

    # ------------------------------------------------------------- data
    def with_radii(self, scheme: str = "mbondi2", overwrite: bool = True) -> "Molecule":
        """Assign radii by element scheme (``bondi``, ``mbondi``, ``mbondi2``, ``parse``)."""
        r = assign_radii(self.elements, scheme, self.bonds if self.bonds.size else None, self.coords)
        if not overwrite and self.has_radii:
            return self
        missing = [i for i, e in enumerate(self.elements) if not e]
        m = replace(self, radii=r, notes=self.notes + [f"radii: {scheme}"])
        if missing:
            m.notes.append(f"{len(missing)} particles without element (virtual sites?) got radius 0")
        return m

    def require(self, charges: bool = True, radii: bool = True, coords: bool = True) -> None:
        """Raise ``ValueError`` with a helpful message if data needed for PB are missing."""
        if coords and self.coords is None:
            raise ValueError(f"{self.source}: no coordinates; pass a coordinate file to load()")
        if charges and not self.has_charges:
            raise ValueError(
                f"{self.source}: partial charges are missing; load a topology (prmtop, psf, top), a PQR or "
                "mol2 file, or assign a force field with pbsolve.prep"
            )
        if radii and not self.has_radii:
            raise ValueError(f"{self.source}: radii are missing; use load(..., radii='mbondi2') or with_radii()")

    def write_pqr(self, path: PathLike, coords: Optional[npt.ArrayLike] = None) -> None:
        """Write a PQR file (whitespace-separated, readable by APBS and pbsolve).
    Coordinates and charges are written with 6 decimals so that converting is
    lossless in practice: with a sharp dielectric boundary (``surface="mol"``,
    APBS ``srfm mol``) rounding coordinates to 1e-3–1e-4 Å can flip a few
    boundary faces next to a charge and shift ΔG_polar by ~0.1 %."""
        xyz = self.coords if coords is None else np.asarray(coords, dtype=np.float64)
        if xyz is None:
            raise ValueError("no coordinates to write")
        self.require(coords=False)
        write_pqr(path, self, xyz)

    # ------------------------------------------------------------- selection
    def select(self, expr: str) -> npt.NDArray[np.bool_]:
        """Boolean mask of the atoms matching a selection expression (see module docstring)."""
        return _Selection(self, expr).parse()


def write_pqr(path: PathLike, mol: Molecule, coords: npt.ArrayLike) -> None:
    """Write ``mol`` at positions ``coords`` ((N, 3), Å) as a whitespace-separated PQR file (charges in e, radii in Å).
    """
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.shape != (mol.n_atoms, 3):
        raise ValueError(f"coords have shape {xyz.shape}, expected ({mol.n_atoms}, 3)")
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("REMARK   1 written by pbsolve (charges e, radii A)\n")
        for i in range(mol.n_atoms):
            rec = "HETATM" if mol.hetero[i] else "ATOM"
            chain = mol.chains[i] or ""
            resid = f"{mol.resids[i]}{mol.icodes[i]}"
            name = mol.names[i] or "X"
            resn = mol.resnames[i] or "UNK"
            fields = [rec, str(i + 1), name, resn] + ([chain] if chain else []) + [resid]
            x, y, z = xyz[i]
            fh.write(
                " ".join(fields) + f" {x:.6f} {y:.6f} {z:.6f} {mol.charges[i]:.6f} {mol.radii[i]:.4f}\n"
            )
        fh.write("TER\nEND\n")


# ----------------------------------------------------------------- loading
def _molecule(d: dict, source: str) -> Molecule:
    n = len(d["names"])
    s = lambda k: np.array(d[k], dtype=object if n == 0 else str)  # noqa: E731
    coords = d["coords"]
    return Molecule(
        names=s("names"),
        resnames=s("resnames"),
        resids=np.asarray(d["resids"], dtype=np.int64),
        icodes=s("icodes"),
        chains=s("chains"),
        segids=s("segids"),
        elements=s("elements"),
        atom_types=s("atom_types"),
        hetero=np.asarray(d["hetero"], dtype=bool),
        charges=np.asarray(d["charges"], dtype=np.float64),
        radii=np.asarray(d["radii"], dtype=np.float64),
        masses=np.asarray(d["masses"], dtype=np.float64),
        bonds=np.asarray(d["bonds"], dtype=np.int64).reshape(-1, 2),
        coords=coords[0].copy() if coords.shape[0] else None,
        models=coords if coords.shape[0] else None,
        cells=d["cells"] if coords.shape[0] else None,
        periodic=d["periodic"],
        source=source,
        notes=list(d["notes"]),
    )


def load(
    path: PathLike,
    coords: Optional[PathLike] = None,
    *,
    format: Optional[str] = None,
    coords_format: Optional[str] = None,
    radii: Optional[str] = None,
    include_dirs: Optional[Sequence[PathLike]] = None,
) -> Molecule:
    """Read a structure or topology (``path``) and optionally coordinates (``coords``).

    ``coords`` may be any structure/restart file or a trajectory (its first
    frame is used). ``radii``: assign radii with a scheme (``bondi``,
    ``mbondi``, ``mbondi2``, ``parse``); by default radii come from the file
    (PQR, Amber prmtop) and are otherwise left unset.
    """
    d = read_structure(os.fspath(path), format, [os.fspath(p) for p in include_dirs] if include_dirs else None)
    mol = _molecule(d, os.fspath(path))
    if coords is not None:
        c = os.fspath(coords)
        reader = TrajectoryReader(c, coords_format, mol.n_atoms, mol.periodic)
        first = next(iter(reader), None)
        if first is None:
            raise ValueError(f"{c}: no coordinates")
        mol.coords = np.asarray(first[0], dtype=np.float64)
        mol.models = mol.coords[None].copy()
        cell = first[1]
        mol.cells = None if cell is None else np.asarray([cell], dtype=np.float64)
    if radii is not None:
        if radii not in SCHEMES:
            raise ValueError(f"unknown radius scheme '{radii}' (expected one of {', '.join(SCHEMES)})")
        mol = mol.with_radii(radii)
    return mol


def frames(
    mol: Union[Molecule, int],
    trajectory: PathLike,
    *,
    format: Optional[str] = None,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    atoms: Optional[Union[str, npt.ArrayLike]] = None,
) -> Iterator[npt.NDArray[np.float64]]:
    """Iterate over coordinates (Å) of frames ``start:stop:step`` of a trajectory,
    optionally restricted to ``atoms`` (selection, mask or indices of ``mol``)."""
    if step < 1 or start < 0:
        raise ValueError("need start >= 0 and step >= 1")
    n = mol if isinstance(mol, int) else mol.n_atoms
    periodic = None if isinstance(mol, int) else mol.periodic
    idx = None
    if atoms is not None:
        if isinstance(mol, int):
            idx = np.asarray(atoms)
        else:
            idx = np.nonzero(mol._mask(atoms))[0]
    reader = TrajectoryReader(os.fspath(trajectory), format, n, periodic)
    for k, (xyz, _cell, _t, _step) in enumerate(reader):
        if stop is not None and k >= stop:
            break
        if k < start or (k - start) % step:
            continue
        yield xyz if idx is None else xyz[idx]


# ----------------------------------------------------------------- selection language
_KEYWORDS = {"resname", "name", "element", "chain", "segid", "resid", "index", "serial", "type"}
_FLAGS = {"all", "none", "protein", "water", "ions", "hydrogen", "hetero", "solvent"}
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


class _Selection:
    def __init__(self, mol: Molecule, expr: str) -> None:
        self.mol = mol
        self.expr = expr
        self.toks = _TOKEN.findall(expr)
        self.pos = 0

    def error(self, msg: str) -> ValueError:
        return ValueError(f"selection '{self.expr}': {msg}")

    def peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def take(self) -> str:
        t = self.peek()
        if t is None:
            raise self.error("unexpected end")
        self.pos += 1
        return t

    def parse(self) -> npt.NDArray[np.bool_]:
        if not self.toks:
            raise self.error("empty selection")
        m = self.or_()
        if self.peek() is not None:
            raise self.error(f"unexpected '{self.peek()}'")
        return m

    def or_(self) -> npt.NDArray[np.bool_]:
        m = self.and_()
        while self.peek() == "or":
            self.take()
            m = m | self.and_()
        return m

    def and_(self) -> npt.NDArray[np.bool_]:
        m = self.not_()
        while self.peek() == "and":
            self.take()
            m = m & self.not_()
        return m

    def not_(self) -> npt.NDArray[np.bool_]:
        if self.peek() == "not":
            self.take()
            return ~self.not_()
        return self.atom()

    def values(self) -> list[str]:
        out = []
        while (t := self.peek()) is not None and t not in ("and", "or", "not", "(", ")"):
            if t in _KEYWORDS or t in _FLAGS:
                break
            out.append(self.take())
        if not out:
            raise self.error("keyword needs at least one value")
        return out

    def ranges(self, vals: list[str]) -> list[tuple[int, int]]:
        out = []
        for v in vals:
            m = re.fullmatch(r"(-?\d+)(?:(?:-|:|to)(-?\d+))?", v)
            if not m:
                raise self.error(f"'{v}' is not a number or range")
            lo = int(m.group(1))
            out.append((lo, int(m.group(2)) if m.group(2) else lo))
        return out

    def atom(self) -> npt.NDArray[np.bool_]:
        mol = self.mol
        n = mol.n_atoms
        t = self.take()
        if t == "(":
            m = self.or_()
            if self.take() != ")":
                raise self.error("missing ')'")
            return m
        if t == "all":
            return np.ones(n, dtype=bool)
        if t == "none":
            return np.zeros(n, dtype=bool)
        if t == "protein":
            return np.array([r.upper() in PROTEIN_NAMES for r in mol.resnames], dtype=bool)
        if t == "water":
            return mol.solvent_mask(water=True, ions="none")
        if t == "ions":
            return mol.solvent_mask(water=False, ions="all")
        if t == "solvent":
            return mol.solvent_mask()
        if t == "hydrogen":
            return mol.elements == "H"
        if t == "hetero":
            return mol.hetero.copy()
        if t in ("resname", "name", "element", "chain", "segid", "type"):
            field_ = {
                "resname": mol.resnames, "name": mol.names, "element": mol.elements,
                "chain": mol.chains, "segid": mol.segids, "type": mol.atom_types,
            }[t]
            vals = self.values()
            m = np.zeros(n, dtype=bool)
            for v in vals:
                if any(c in v for c in "*?["):
                    m |= np.array([fnmatch.fnmatchcase(x, v) for x in field_], dtype=bool)
                else:
                    m |= field_ == v
            return m
        if t in ("resid", "index", "serial"):
            arr = {"resid": mol.resids, "index": np.arange(n), "serial": np.arange(1, n + 1)}[t]
            m = np.zeros(n, dtype=bool)
            for lo, hi in self.ranges(self.values()):
                m |= (arr >= lo) & (arr <= hi)
            return m
        raise self.error(f"unknown keyword '{t}'")
