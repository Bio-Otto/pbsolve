# SPDX-License-Identifier: MIT AND BSD-3-Clause
# The structure-preparation algorithms (termini and patches, heavy-atom
# repair, hydrogen placement with the quaternion fit, debumping, hydrogen-bond
# optimisation, protonation states, force-field lookup and PQR formatting) follow pdb2pqr 3.8.0
# (Copyright (c) 2002-2024 J. E. Nielsen, N. A. Baker, Battelle Memorial
# Institute, P. Czodrowski & G. Klebe; BSD-3-Clause), and its data files are
# shipped in pbsolve/data/pdb2pqr. See THIRD_PARTY_NOTICES.md.
"""PDB → PQR without pdb2pqr: hydrogens, protonation states, force field.

Reproduces ``pdb2pqr --drop-water --keep-chain --whitespace`` byte for byte
for proteins (AMBER, PARSE, CHARMM), with pdb2pqr's hydrogen-bond optimisation
(HIS/ASN/GLN flips, hydroxyl and carboxyl hydrogens; ``optimize=False`` =
``--noopt``), heavy-atom repair and debumping (checked on 1YCR, 1TII, 1HPV,
5DK3; ``scripts/validate_prep.py``)::

    mol, report = pbsolve.prep.prepare("1abc.pdb", forcefield="amber")
    mol.write_pqr("1abc.pqr")      # or report.pqr_text for pdb2pqr's exact layout

Protonation (no pKa calculation by default, as pdb2pqr without PROPKA):
ASP/GLU charged, LYS/ARG positive, TYR/CYS neutral, termini charged, HIS
neutral; residues named ASH, GLH, HID, HIE, HIP,
LYN, CYM, CYX ... in the input keep that state; CYS pairs with SG–SG < 2.5 Å
become CYX; the optimisation picks HID/HIE from hydrogen bonds (as
pdb2pqr). ``ph=`` applies model-compound pKa values (simple rules);
``titration="propka"`` uses PROPKA 3 if it is installed (an optional,
separately licensed LGPL package; it is not bundled).

Not covered (fail loudly or reported): waters are dropped (pdb2pqr protonates
them with an optimiser); nucleic acids; ligands other than through
``ligand=`` (a mol2 file with charges).
"""

from __future__ import annotations

import argparse
import copy
import functools
import math
import os
import re
import string
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]

DATA = os.path.join(os.path.dirname(__file__), "data", "pdb2pqr")
FORCEFIELDS = ("amber", "parse", "charmm")
AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}  # fmt: skip
WATERS = {"HOH", "WAT", "SOL", "H2O", "DOD", "TIP3", "TIP"}
NUCLEIC = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "RA", "RC", "RG", "RU", "ADE", "CYT", "GUA", "THY", "URA"}

#: Model-compound pKa values used by ``ph=`` (side chains; termini are kept charged as in pdb2pqr).
MODEL_PKA = {"ASP": 3.9, "GLU": 4.3, "HIS": 6.0, "CYS": 8.3, "TYR": 10.1, "LYS": 10.5, "ARG": 12.5}

__all__ = ["PrepReport", "prepare", "main", "FORCEFIELDS", "MODEL_PKA"]


# ===================================================================== definitions
class _DAtom:
    def __init__(self, name, x, y, z, bonds):
        self.name, self.x, self.y, self.z, self.bonds = name, x, y, z, list(bonds)

    @property
    def coords(self):
        return [self.x, self.y, self.z]


class _DRes:
    def __init__(self, name):
        self.name = name
        self.map = {}
        self.altnames = {}
        self.dihedrals = []


class _Patch:
    def __init__(self):
        self.name = ""
        self.applyto = ""
        self.newname = ""
        self.map = {}
        self.remove = []
        self.altnames = {}
        self.dihedrals = []


def _parse_atoms(elem, holder):
    for a in elem.iter("atom"):
        nm = a.find("name").text.strip()

        def f(tag, a=a):
            e = a.find(tag)
            return float(e.text) if e is not None else 0.0

        holder.map[nm] = _DAtom(nm, f("x"), f("y"), f("z"), [b.text.strip() for b in a.findall("bond")])
        for alt in a.findall("altname"):
            holder.altnames[alt.text.strip()] = nm


@functools.lru_cache(maxsize=1)
def _definitions():
    """(defmap, patches) built like pdb2pqr's Definition (shared template, deep-copied per run)."""
    defmap: dict = {}
    for r in ET.parse(os.path.join(DATA, "AA.xml")).getroot().findall("residue"):
        d = _DRes(r.find("name").text.strip())
        _parse_atoms(r, d)
        d.dihedrals = [x.text.strip() for x in r.findall("dihedral")]
        defmap[d.name] = d
    patches: dict = {}
    plist = []
    for p in ET.parse(os.path.join(DATA, "PATCHES.xml")).getroot().findall("patch"):
        P = _Patch()
        P.name = p.find("name").text.strip()
        P.applyto = p.find("applyto").text.strip()
        nn = p.find("newname")
        P.newname = nn.text.strip() if nn is not None else ""
        _parse_atoms(p, P)
        P.remove = [x.text.strip() for x in p.iter("remove")]
        P.dihedrals = [x.text.strip() for x in p.iter("dihedral")]
        plist.append(P)

    def add_patch(P, refname, newname):
        if refname not in defmap:
            patches[newname] = P
            return
        pr = copy.deepcopy(defmap[refname])
        for an, at in P.map.items():
            pr.map[an] = at
            for b in at.bonds:
                if b in pr.map and an not in pr.map[b].bonds:
                    pr.map[b].bonds.append(an)
        pr.altnames.update(P.altnames)
        for rm in P.remove:
            if rm not in pr.map:
                continue
            rb = pr.map[rm].bonds
            del pr.map[rm]
            for b in rb:
                if rm in pr.map[b].bonds:
                    pr.map[b].bonds.remove(rm)
        pr.dihedrals += P.dihedrals
        defmap[newname] = pr
        patches[newname] = P

    for P in plist:
        if P.newname:
            for nm in list(defmap.keys()):
                if re.compile(P.applyto).match(nm):
                    add_patch(P, nm, P.newname.replace("*", nm))
        add_patch(P, P.applyto, P.name)
    return defmap, patches


@functools.lru_cache(maxsize=4)
def _ff_table(ff: str) -> dict:
    out = {}
    with open(os.path.join(DATA, f"{ff.upper()}.tsv"), encoding="ascii") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            w = line.rstrip("\n").split("\t")
            out[(w[0], w[1])] = (float(w[2]), float(w[3]))
    return out


# ===================================================================== quatfit (exact op order)
def _center(n, c):
    cen = [0.0, 0.0, 0.0]
    for i in range(n):
        cen[0] += c[i][0]
        cen[1] += c[i][1]
        cen[2] += c[i][2]
    for i in range(3):
        cen[i] = cen[i] / n
    return cen, [[c[i][0] - cen[0], c[i][1] - cen[1], c[i][2] - cen[2]] for i in range(n)]


def _jacobi(a, nrot):
    v = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    d = [a[j][j] for j in range(4)]
    for _ in range(nrot):
        dn = 0.0
        on = 0.0
        for j in range(4):
            dn += abs(d[j])
            for i in range(j):
                on += abs(a[i][j])
        if dn != 0 and on / dn <= 1e-12:
            break
        for j in range(1, 4):
            for i in range(j):
                b = a[i][j]
                if abs(b) > 0.0:
                    dma = d[j] - d[i]
                    if abs(dma) + abs(b) <= abs(dma):
                        t = b / dma
                    else:
                        q = 0.5 * dma / b
                        t = 1.0 / (abs(q) + math.sqrt(1 + q * q))
                        if q < 0:
                            t = t * -1
                    c = 1.0 / math.sqrt(t * t + 1)
                    s = t * c
                    a[i][j] = 0.0
                    for k in range(i):
                        tmp = c * a[k][i] - s * a[k][j]
                        a[k][j] = s * a[k][i] + c * a[k][j]
                        a[k][i] = tmp
                    for k in range(i + 1, j):
                        tmp = c * a[i][k] - s * a[k][j]
                        a[k][j] = s * a[i][k] + c * a[k][j]
                        a[i][k] = tmp
                    for k in range(j + 1, 4):
                        tmp = c * a[i][k] - s * a[j][k]
                        a[j][k] = s * a[i][k] + c * a[j][k]
                        a[i][k] = tmp
                    for k in range(4):
                        tmp = c * v[k][i] - s * v[k][j]
                        v[k][j] = s * v[k][i] + c * v[k][j]
                        v[k][i] = tmp
                    dt = c * c * d[i] + s * s * d[j] - 2.0 * c * s * b
                    d[j] = s * s * d[i] + c * c * d[j] + 2.0 * c * s * b
                    d[i] = dt
    for j in range(3):
        k = j
        dt = d[k]
        for i in range(j + 1, 4):
            if d[i] < dt:
                k = i
                dt = d[k]
        if k > j:
            d[k] = d[j]
            d[j] = dt
            for i in range(4):
                dt = v[i][k]
                v[i][k] = v[i][j]
                v[i][j] = dt
    return d, v


def _q2mat(q):
    u = [[0.0] * 3 for _ in range(3)]
    u[0][0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    u[0][1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    u[0][2] = 2.0 * (q[1] * q[3] + q[0] * q[2])
    u[1][0] = 2.0 * (q[2] * q[1] + q[0] * q[3])
    u[1][1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    u[1][2] = 2.0 * (q[2] * q[3] - q[0] * q[1])
    u[2][0] = 2.0 * (q[3] * q[1] - q[0] * q[2])
    u[2][1] = 2.0 * (q[3] * q[2] + q[0] * q[1])
    u[2][2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
    return u


def _rotmol(c, u):
    return [
        [
            u[0][0] * p[0] + u[1][0] * p[1] + u[2][0] * p[2],
            u[0][1] * p[0] + u[1][1] * p[1] + u[2][1] * p[2],
            u[0][2] * p[0] + u[1][2] * p[1] + u[2][2] * p[2],
        ]
        for p in c
    ]


def _find_coordinates(n, real, tmpl, tmpl_atom):
    rc, real_c = _center(n, real)
    tc, tmpl_c = _center(n, tmpl)
    s = [[0.0] * 3 for _ in range(3)]
    for i in range(n):
        for a in range(3):
            for b in range(3):
                s[a][b] += tmpl_c[i][a] * real_c[i][b]
    xxyx, xxyy, xxyz = s[0]
    xyyx, xyyy, xyyz = s[1]
    xzyx, xzyy, xzyz = s[2]
    c = [[0.0] * 4 for _ in range(4)]
    c[0][0] = xxyx + xyyy + xzyz
    c[0][1] = xzyy - xyyz
    c[0][2] = xxyz - xzyx
    c[0][3] = xyyx - xxyy
    c[1][1] = xxyx - xyyy - xzyz
    c[1][2] = xxyy + xyyx
    c[1][3] = xzyx + xxyz
    c[2][2] = xyyy - xzyz - xxyx
    c[2][3] = xyyz + xzyy
    c[3][3] = xzyz - xxyx - xyyy
    _, v = _jacobi(c, 30)
    u = _q2mat([v[i][3] for i in range(4)])
    p = [[tmpl_atom[0] - tc[0], tmpl_atom[1] - tc[1], tmpl_atom[2] - tc[2]]]
    r = _rotmol(p, u)[0]
    return [r[0] + rc[0], r[1] + rc[1], r[2] + rc[2]]


def _norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dist(a, b):
    return _norm3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])


def _qchichange(axis, pts, angle):
    r = math.pi * angle / 180.0
    n = _norm3(axis)
    lv = [axis[0] / n, axis[1] / n, axis[2] / n]
    c = math.cos(r)
    s = math.sin(r)
    R = [[0.0] * 3 for _ in range(3)]
    R[0][0] = c + lv[0] * lv[0] * (1.0 - c)
    R[1][1] = c + lv[1] * lv[1] * (1.0 - c)
    R[2][2] = c + lv[2] * lv[2] * (1.0 - c)
    R[1][0] = lv[0] * lv[1] * (1.0 - c) - lv[2] * s
    R[2][0] = lv[0] * lv[2] * (1.0 - c) + lv[1] * s
    R[0][1] = lv[1] * lv[0] * (1.0 - c) + lv[2] * s
    R[2][1] = lv[1] * lv[2] * (1.0 - c) - lv[0] * s
    R[0][2] = lv[2] * lv[0] * (1.0 - c) - lv[1] * s
    R[1][2] = lv[2] * lv[1] * (1.0 - c) + lv[0] * s
    return _rotmol(pts, R)


def _dihedral(p1, p2, p3, p4):
    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    d43 = sub(p4, p3)
    d32 = sub(p3, p2)
    d12 = sub(p1, p2)
    a = cross(d12, d32)
    na = _norm3(a)
    a = [x / na for x in a]
    b = cross(d43, d32)
    nb = _norm3(b)
    b = [x / nb for x in b]
    sc = dot(a, b)
    if abs(sc + 1.0) < 1e-7:
        val = 180.0
    elif abs(sc - 1.0) < 1e-7:
        val = 0.0
    else:
        val = 57.2958 * math.acos(sc)  # pdb2pqr's constant, not 180/pi
    if dot(cross(a, b), d32) < 0:
        val *= -1.0
    return val


# ===================================================================== structure
class _Atom:
    def __init__(self, rec, res):
        self.__dict__.update(rec)
        self.res = res
        self.bonds = []
        self.ref = None
        self.added = 0
        self.cell = None
        self.hdonor = 0
        self.hacceptor = 0
        self.refd = 0

    @property
    def coords(self):
        return [self.x, self.y, self.z]


class _Res:
    def __init__(self, recs, ref, amino):
        last = recs[-1]
        self.name = last["resname"]
        self.chain = last["chain"]
        self.resseq = last["resseq"]
        self.icode = last["icode"]
        self.ffname = self.name
        self.amino = amino
        self.ref = ref
        self.atoms = []
        self.map = {}
        self.nterm = self.cterm = False
        self.patches = []
        self.pep_c = self.pep_n = None
        self.ss = False
        self.ss_partner = None
        self.extra = None  # ligand parameters (charge, radius) per atom name
        self.fixed = 0
        self.was_flipped = False
        for r in recs:
            if amino and r["name"] in ref.altnames:
                r["name"] = ref.altnames[r["name"]]
            if r["name"] in self.map:
                continue  # first occurrence (altloc) wins
            a = _Atom(r, self)
            if amino:
                a.type = "ATOM"
            self.add_atom(a)

    def add_atom(self, a):
        self.atoms.append(a)
        self.map[a.name] = a
        if not self.amino:
            return
        a.ref = self.ref.map.get(a.name)
        if a.ref is None:
            return
        for b in a.ref.bonds:
            if b in self.map:
                o = self.map[b]
                if o not in a.bonds:
                    a.bonds.append(o)
                if a not in o.bonds:
                    o.bonds.append(a)

    def create_atom(self, name, xyz):
        a0 = self.atoms[0]
        rec = dict(type="ATOM", name=name, resname=a0.resname, chain=a0.chain, resseq=a0.resseq,
                   icode=a0.icode, x=xyz[0], y=xyz[1], z=xyz[2], occ=1.0, tf=0.0,
                   seg=getattr(a0, "seg", ""), element=name[0], chg=getattr(a0, "chg", ""))  # fmt: skip
        a = _Atom(rec, self)
        a.added = 1
        self.add_atom(a)

    def remove_atom(self, name):
        a = self.map.pop(name)
        self.atoms.remove(a)
        for b in a.bonds:
            if a in b.bonds:
                b.bonds.remove(a)

    def rename_atom(self, old, new):
        a = self.map[old]
        a.name = new
        self.map[new] = a
        del self.map[old]

    @property
    def label(self):
        return f"{self.name} {self.chain or '-'}{self.resseq}{self.icode}"


def _rotate_tetrahedral(a1, a2, ang):
    init = [a2.x - a1.x, a2.y - a1.y, a2.z - a1.z]
    mv = [b for b in a2.bonds if b is not a1]
    new = _qchichange(init, [[b.x - a1.x, b.y - a1.y, b.z - a1.z] for b in mv], ang)
    for b, c in zip(mv, new):
        b.x = c[0] + a1.x
        b.y = c[1] + a1.y
        b.z = c[2] + a1.z


def _nearest_bonds(ref, name):
    bonds = []
    lev2 = []
    at = ref.map[name]
    for b in at.bonds:
        if b not in bonds:
            bonds.append(b)
    for b in at.bonds:
        for b2 in ref.map[b].bonds:
            if b2 not in bonds and b2 != name:
                bonds.append(b2)
                lev2.append(b2)
    for b in lev2:
        for b3 in ref.map[b].bonds:
            if b3 not in bonds:
                bonds.append(b3)
    return bonds


@dataclass
class PrepReport:
    """What :func:`prepare` did."""

    forcefield: str
    ph: Optional[float]
    titration: str
    n_atoms: int = 0
    n_hydrogens_added: int = 0
    repaired: list[str] = field(default_factory=list)
    states: dict[str, str] = field(default_factory=dict)  # residue label -> force-field residue name
    dropped: list[str] = field(default_factory=list)  # atoms without parameters
    optimized: bool = True
    flipped: list[str] = field(default_factory=list)  # HIS/ASN/GLN side chains flipped by the optimisation
    warnings: list[str] = field(default_factory=list)
    net_charge: float = 0.0
    pqr_text: str = ""  # pdb2pqr --whitespace layout

    def summary(self) -> str:
        lines = [
            f"force field {self.forcefield.upper()}, protonation: "
            + (f"PROPKA at pH {self.ph}" if self.titration == "propka" else f"pH {self.ph}" if self.ph is not None else "standard states"),
            f"{self.n_atoms} atoms ({self.n_hydrogens_added} hydrogens added, {len(self.repaired)} heavy atoms rebuilt), "
            f"net charge {self.net_charge:+.4f} e",
        ]
        special = {k: v for k, v in self.states.items() if v[-3:] in ("HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "TYM", "AR0")}
        if special:
            lines.append("states: " + ", ".join(f"{k}->{v}" for k, v in special.items()))
        lines.append(
            "hydrogen-bond optimisation: "
            + ((f"{len(self.flipped)} flips (" + ", ".join(self.flipped) + ")") if self.flipped else "no flips")
            if self.optimized
            else "hydrogen-bond optimisation: off (--noopt)"
        )
        if self.dropped:
            lines.append(f"{len(self.dropped)} atoms without force-field parameters were omitted: " + ", ".join(self.dropped[:8]) + (" ..." if len(self.dropped) > 8 else ""))
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


class _Run:
    def __init__(self, recs_all, forcefield):
        self.defmap, self.patches = copy.deepcopy(_definitions())
        self.ff = forcefield
        self.warn: list[str] = []
        self._build(recs_all)

    # ---------------------------------------------------------------- build
    def _make_res(self, recs):
        nm = recs[-1]["resname"]
        if nm in self.defmap and self.defmap[nm].name in AA:
            return _Res(recs, self.defmap[nm], True)
        return _Res(recs, None, False)

    def _build(self, recs_all):
        nch = 1 + sum(1 for r in recs_all if r["kind"] == "TER")
        chains: dict = {}
        cur: list = []
        prev = None
        count = 0
        nmodels = 0

        def flush():
            nonlocal cur
            if cur:
                chains.setdefault(prev["chain"], []).append(self._make_res(cur))
            cur = []

        for r in recs_all:
            if r["kind"] == "atom":
                if r["chain"] == "" and nch > 1 and r["resname"] not in ("WAT", "HOH"):
                    r["chain"] = (string.ascii_uppercase + string.ascii_lowercase + string.digits)[count]
                if prev is None:
                    prev = r
                chains.setdefault(r["chain"], [])
                if (r["resseq"], r["icode"], r["chain"]) != (prev["resseq"], prev["icode"], prev["chain"]):
                    flush()
                cur.append(r)
                prev = r
            elif r["kind"] == "END":
                flush()
            elif r["kind"] == "MODEL":
                nmodels += 1
                if not cur:
                    continue
                if nmodels > 1:
                    flush()
                    break
            elif r["kind"] == "TER":
                count += 1
        if cur and nmodels <= 1:
            flush()
        chains = {k: v for k, v in chains.items() if v}
        # pdb2pqr: chains sorted by ID, the blank chain last ("ZZ"); later splits insert in place
        keys = sorted(("ZZ" if k == "" else k) for k in chains)
        self.chain_list = [["" if k == "ZZ" else k, chains["" if k == "ZZ" else k]] for k in keys]

    def _ordered(self):
        return self.chain_list

    @property
    def residues(self):
        return [r for _, c in self.chain_list for r in c]

    def _free_chain_id(self):
        used = {cid for cid, _ in self.chain_list}
        for ch in string.ascii_uppercase + string.ascii_lowercase:
            if ch not in used:
                return ch
        raise ValueError("no free chain identifier left")

    @staticmethod
    def _relabel(residues, cid):
        for rr in residues:
            rr.chain = cid
            for a in rr.atoms:
                a.chain = cid

    # ---------------------------------------------------------------- patches
    def apply_patch(self, res, pname):
        P = self.patches[pname]
        nr = res.ref if pname == "PEPTIDE" else copy.deepcopy(res.ref)
        for an, at in P.map.items():
            nr.map[an] = at
            for b in at.bonds:
                if b in nr.map and an not in nr.map[b].bonds:
                    nr.map[b].bonds.append(an)
        for rm in P.remove:
            if rm in res.map:
                res.remove_atom(rm)
            if rm not in nr.map:
                continue
            rb = nr.map[rm].bonds
            del nr.map[rm]
            for b in rb:
                nr.map[b].bonds.remove(rm)
        nr.dihedrals = nr.dihedrals + P.dihedrals if pname != "PEPTIDE" else nr.dihedrals
        res.ref = nr
        res.patches.append(pname)
        for a in list(res.atoms):
            if a.name in P.altnames:
                old = a.name
                a.name = P.altnames[old]
                res.map[a.name] = a
                del res.map[old]
        for an, a in res.map.items():
            if an in nr.map:
                a.ref = nr.map[an]

    def place_by_fit(self, res, name):
        coords = []
        refc = []
        for b in _nearest_bonds(res.ref, name):
            a = res.pep_n if b == "N+1" else res.pep_c if b == "C-1" else res.map.get(b)
            if a is None:
                continue
            coords.append(a.coords)
            refc.append(res.ref.map[b].coords)
            if len(coords) == 3:
                break
        if len(coords) < 3:
            return None
        return _find_coordinates(3, coords, refc, res.ref.map[name].coords)

    # ---------------------------------------------------------------- termini
    def _assign_termini(self, c):
        r0, rl = c[0], c[-1]
        if "N" in r0.map and "C" in rl.map and _dist(r0.map["N"].coords, rl.map["C"].coords) < 1.35:
            return
        if r0.amino:
            r0.nterm = True
            heavy = [a for a in r0.map["N"].bonds if a.name[0] != "H"] if "N" in r0.map else []
            self.apply_patch(r0, "NEUTRAL-NTERM" if len(heavy) > 1 else "NTERM")
        if rl.amino:
            rl.cterm = True
            self.apply_patch(rl, "CTERM")
        else:
            for r in reversed(c):
                if r.amino:
                    r.cterm = True
                    self.apply_patch(r, "CTERM")
                    break
                if r.name in ("NH2", "NME"):
                    break

    def set_termini(self):
        for _, c in self.chain_list:
            self._assign_termini(c)
        # Hidden chains (pdb2pqr set_termini): an amino acid with OXT that is not
        # the C-terminus ends a segment; the segment becomes a new chain inserted
        # before the current one, and termini of both are re-assigned.
        ch_num = 0
        last = None
        while ch_num < len(self.chain_list):
            entry = self.chain_list[ch_num]
            last = entry
            reslist = []
            for r in list(entry[1]):
                reslist.append(r)
                if r.amino and "OXT" in r.map and not r.cterm:
                    cid = self._free_chain_id()
                    for rr in reslist:
                        entry[1].remove(rr)
                    self._relabel(reslist, cid)
                    self.chain_list.insert(ch_num, [cid, reslist])
                    if entry[1]:
                        self._assign_termini(entry[1])
                    self._assign_termini(reslist)
                    self.warn.append(f"OXT in {r.label}: residues up to it form a new chain {cid}")
                    reslist = []
                    ch_num += 1
            ch_num += 1
        # A blank chain ID is replaced by the first free letter unless the last
        # chain processed holds only water (pdb2pqr checks the last chain).
        blank = next((e for e in self.chain_list if e[0] == ""), None)
        if blank is not None and last is not None and any(r.name not in WATERS for r in last[1]):
            cid = self._free_chain_id()
            blank[0] = cid
            self._relabel(blank[1], cid)

    def update_bonds(self):
        for r in self.residues:
            if r.amino and not (r.nterm or r.cterm):
                self.apply_patch(r, "PEPTIDE")
        for r in self.residues:
            if not r.amino:
                continue
            for a in r.atoms:
                if a.ref is None:
                    continue
                for b in a.ref.bonds:
                    if b in r.map and r.map[b] not in a.bonds:
                        a.bonds.append(r.map[b])
        for _, c in self._ordered():
            for r1, r2 in zip(c, c[1:]):
                if not (r1.amino and r2.amino):
                    continue
                a1 = r1.map.get("C")
                a2 = r2.map.get("N")
                if a1 is not None:
                    r2.pep_c = a1
                if a2 is not None:
                    r1.pep_n = a2
                if a1 is None or a2 is None:
                    continue
                if _dist(a1.coords, a2.coords) > 1.7:
                    r2.pep_c = None
                    r1.pep_n = None
                    self.warn.append(f"gap in backbone between {r1.label} and {r2.label}")

    # ---------------------------------------------------------------- repair
    def repair_heavy(self, report):
        nheavy = 0
        nmiss = 0
        for r in self.residues:
            if not r.amino:
                continue
            r.missing = []
            for an in r.ref.map:
                if an.startswith("H") or an in ("N+1", "C-1"):
                    continue
                nheavy += 1
                if an not in r.map:
                    r.missing.append(an)
                    nmiss += 1
        if not nmiss:
            return
        if nmiss / nheavy > 0.1:
            raise ValueError(
                f"{nmiss} of {nheavy} heavy atoms are missing (> 10 %); the structure is too incomplete to repair"
            )
        for r in self.residues:
            if not r.amino:
                continue
            for a in list(r.atoms):
                if a.name not in r.ref.map:
                    self.warn.append(f"extra atom {a.name} in {r.label} deleted")
                    r.remove_atom(a.name)
            miss = r.missing
            seen: dict = {}
            nm0 = len(miss)
            while miss:
                an = miss.pop(0)
                xyz = self.place_by_fit(r, an)
                if xyz is None:
                    seen[an] = seen.get(an, 0) + 1
                    miss.append(an)
                    if seen[an] > nm0:
                        raise ValueError(f"too few atoms present to rebuild {an} in {r.label}")
                else:
                    r.create_atom(an, xyz)
                    report.repaired.append(f"{r.label}:{an}")

    def update_ss(self):
        sg = [r.map["SG"] for r in self.residues if r.amino and r.ref.name == "CYS" and "SG" in r.map]
        partners = {id(a): [] for a in sg}
        for a in sg:
            for p in sg:
                if a is p or partners[id(a)]:
                    continue
                if _dist(a.coords, p.coords) < 2.5:
                    partners[id(a)].append(p)
                    partners[id(p)].append(a)
        for a in sg:
            if len(partners[id(a)]) == 1:
                a.res.ss = True
                a.res.ss_partner = partners[id(a)][0]
                self.apply_patch(a.res, "CYX")

    # ---------------------------------------------------------------- debump
    BACKBONE = ["N", "CA", "C", "O", "O2", "HA", "HN", "H", "tN"]

    class _Cells:
        def __init__(self, size):
            self.size = size
            self.m = {}

        def key(self, a):
            k = []
            for v in (a.x, a.y, a.z):
                k.append((int(v) - 1) // self.size * self.size if v < 0 else int(v) // self.size * self.size)
            return tuple(k)

        def add(self, a):
            k = self.key(a)
            self.m.setdefault(k, []).append(a)
            a.cell = k

        def remove(self, a):
            if getattr(a, "cell", None) is None:
                return
            self.m[a.cell].remove(a)
            a.cell = None

        def near(self, a):
            out = []
            c = a.cell
            if c is None:
                return out
            for i in (-self.size, 0, self.size):
                for j in (-self.size, 0, self.size):
                    for k in (-self.size, 0, self.size):
                        for b in self.m.get((c[0] + i, c[1] + j, c[2] + k), []):
                            if b is not a:
                                out.append(b)
            return out

    def set_donors_acceptors(self):
        for r in self.residues:
            if not r.amino:
                continue
            for a in r.atoms:
                a.hdonor = False
                a.hacceptor = False
                if a.name.startswith("N"):
                    bonded = 0
                    for b in a.bonds:
                        if b.name[0] == "H":
                            a.hdonor = True
                            bonded = 1
                            break
                    if not bonded and r.ref.name == "HIS":
                        a.hacceptor = True
                elif a.name.startswith("O") or (a.name.startswith("S") and r.ref.name == "CYS"):
                    a.hacceptor = True
                    for b in a.bonds:
                        if b.name[0] == "H":
                            a.hdonor = True
                            break

    @staticmethod
    def _shortest_path(graph, start, end, path=()):
        path = list(path) + [start]
        if start is end:
            return path
        if start not in graph:
            return None
        best = None
        for node in graph[start]:
            if node not in path:
                np_ = _Run._shortest_path(graph, node, end, path)
                if np_ and (not best or len(np_) < len(best)):
                    best = np_
        return best

    def set_refdist(self):
        for r in self.residues:
            if not r.amino:
                continue
            if "CA" not in r.map:
                raise ValueError(f"{r.label} has no CA atom")
            ca = r.map["CA"]
            g = {a: a.bonds for a in r.atoms}
            for a in r.atoms:
                if a.name in self.BACKBONE:
                    a.refd = -1
                elif r.cterm and a.name == "HO":
                    a.refd = 3
                elif r.nterm and a.name in ("H3", "H2"):
                    a.refd = 2
                else:
                    p = self._shortest_path(g, a, ca)
                    if p is None:
                        raise ValueError(f"atom {a.name} of {r.label} is not connected to CA")
                    a.refd = len(p) - 1

    def calc_dihedrals(self):
        for r in self.residues:
            if not r.amino:
                continue
            r.dih = []
            for d in r.ref.dihedrals:
                n = d.split()
                cs = [r.map[x].coords for x in n if x in r.map]
                r.dih.append(_dihedral(*cs) if len(cs) == 4 else None)

    @staticmethod
    def _moveable(r, pivot):
        rd = r.map[pivot].refd
        return [a.name for a in r.atoms if a.refd > rd]

    @staticmethod
    def _nearby(cells, a):
        out = {}
        r = a.res
        sz = 0.5 if a.name[0] == "H" else 1.0
        for c in cells.near(a):
            if c.res is r and (c in a.bonds or a in c.bonds):
                continue
            if not c.res.amino:
                continue
            if r.ref is not None and r.ref.name == "CYS" and r.ss_partner is c:
                continue
            if a.name[0] == "H" and a.bonds and a.bonds[0].hdonor and c.hacceptor:
                continue
            if c.name[0] == "H" and c.bonds and c.bonds[0].hdonor and a.hacceptor:
                continue
            d = _dist(a.coords, c.coords)
            cut = sz + (0.5 if c.name[0] == "H" else 1.0)
            if d < cut:
                out[c] = cut - d
        return out

    def _conflicts(self, cells, r):
        return [a.name for a in r.atoms if a.added and a.name != "H" and self._nearby(cells, a)]

    def _score(self, cells, r, n):
        sc = 0
        for nm in self._moveable(r, r.ref.dihedrals[n].split()[2]):
            for v in self._nearby(cells, r.map[nm]).values():
                sc += v
        return sc

    def _set_dih(self, cells, r, n, angle):
        diff = angle - r.dih[n]
        names = r.ref.dihedrals[n].split()
        cl = [r.map[x].coords for x in names]
        init = [cl[2][i] - cl[1][i] for i in range(3)]
        mv = self._moveable(r, names[2])
        new = _qchichange(init, [[r.map[m].x - cl[1][0], r.map[m].y - cl[1][1], r.map[m].z - cl[1][2]] for m in mv], diff)
        for m, c in zip(mv, new):
            a = r.map[m]
            cells.remove(a)
            a.x = c[0] + cl[1][0]
            a.y = c[1] + cl[1][1]
            a.z = c[2] + cl[1][2]
            cells.add(a)
        r.dih[n] = _dihedral(*[r.map[x].coords for x in names])

    def _pick(self, r, cn, old):
        bestnum = -1
        best = 0
        il = list(range(len(r.dih)))
        if old is not None and old >= 0 and il:
            del il[old]
            test = il[old:] + il[:old]
        else:
            test = il
        for i in test:
            if i == old or r.dih[i] is None:
                continue
            sc = 0
            mv = self._moveable(r, r.ref.dihedrals[i].split()[2])
            if cn == mv:
                return i
            for nm in cn:
                if nm in mv:
                    sc += 1
                    if sc > best:
                        best = sc
                        bestnum = i
        return bestnum

    def _debump_res(self, cells, r, cn):
        n = -1
        cur = cn
        for _ in range(10):
            n = self._pick(r, cur, n)
            if n == -1:
                return False
            bs = self._score(cells, r, n)
            ba = orig = r.dih[n]
            for i in range(1, 72):
                na = orig + 5.0 * i
                self._set_dih(cells, r, n, na)
                sc = self._score(cells, r, n)
                if sc == 0:
                    if not self._conflicts(cells, r):
                        return True
                    ba = na
                    break
                elif sc < bs and abs(bs - sc) > 1e-7:
                    bs = sc
                    ba = na
            self._set_dih(cells, r, n, ba)
            cur = self._conflicts(cells, r)
        return False

    def debump(self):
        cells = self._Cells(2)
        for a in (a for r in self.residues for a in r.atoms):
            a.cell = None
            cells.add(a)
        self.calc_dihedrals()
        self.set_donors_acceptors()
        self.set_refdist()
        for r in self.residues:
            if not r.amino:
                continue
            cn = self._conflicts(cells, r)
            if cn and not self._debump_res(cells, r, cn):
                self.warn.append(f"unable to debump {r.label}")

    # ---------------------------------------------------------------- hydrogens
    def _rebuild_tetrahedral(self, r, an):
        aref = r.ref.map.get(an)
        if aref is None:
            return False
        bn = aref.bonds[0]
        if bn not in r.map:
            return False
        hc = 0
        nxt = None
        for b in r.ref.map[bn].bonds:
            if b.startswith("H"):
                hc += 1
            elif b not in ("C-1", "N+1"):
                nxt = b
        if hc != 3 or nxt is None:
            return False
        ba = r.map[bn]
        na = r.map[nxt]
        nb = len(ba.bonds)
        if nb == 1:
            xyz = _find_coordinates(2, [ba.coords, na.coords], [r.ref.map[bn].coords, r.ref.map[nxt].coords], aref.coords)
            r.create_atom(an, xyz)
            if r.ref.name == "LEU":
                ang = _dihedral(r.map["CB"].coords, na.coords, ba.coords, xyz)
                _rotate_tetrahedral(na, ba, 60 - ang)
            elif r.ref.name == "ILE":
                if ba.name == "CD1" or ba.name != "CG2":
                    ang = _dihedral(r.map["CB"].coords, na.coords, ba.coords, xyz)
                else:
                    ang = _dihedral(r.map["CG1"].coords, na.coords, ba.coords, xyz)
                _rotate_tetrahedral(na, ba, 60 - ang)
            return True
        if nb == 2:
            h = None
            for b in ba.ref.bonds:
                if b in r.map and b.startswith("H"):
                    h = r.map[b]
                    break
            _rotate_tetrahedral(na, ba, 120)
            xyz = h.coords
            _rotate_tetrahedral(na, ba, -120)
            r.create_atom(an, xyz)
            return True
        if nb == 3:
            hs = [r.map[b] for b in ba.ref.bonds if b in r.map and b.startswith("H")]
            if len(hs) != 2:
                return False
            _rotate_tetrahedral(na, ba, 120)
            c1 = hs[0].coords
            _rotate_tetrahedral(na, ba, 120)
            c2 = hs[0].coords
            _rotate_tetrahedral(na, ba, 120)
            r.create_atom(an, c1 if _dist(hs[1].coords, c1) > 0.1 else c2)
            return True
        return False

    def add_hydrogens(self, report):
        for r in self.residues:
            if not r.amino:
                continue
            for an in list(r.ref.map):
                if not an.startswith("H") or an in r.map:
                    continue
                if r.ref.name == "CYS" and r.ss and an == "HG":
                    continue
                if self._rebuild_tetrahedral(r, an):
                    continue
                xyz = self.place_by_fit(r, an)
                if xyz is not None:
                    r.create_atom(an, xyz)
                else:
                    self.warn.append(f"could not rebuild {an} in {r.label}")

    def cleanup(self):
        """pdb2pqr HydrogenRoutines.cleanup: protonated ASP/GLU keep one carboxyl H."""
        for r in self.residues:
            if not r.amino:
                continue
            if r.name == "GLH" or "GLH" in r.patches:
                if "HE1" in r.map and "HE2" in r.map:
                    r.remove_atom("HE1")
            elif r.name == "ASH" or "ASH" in r.patches:
                if "HD1" in r.map and "HD2" in r.map:
                    r.remove_atom("HD1")

    def update_internal_bonds(self):
        for r in self.residues:
            if not r.amino:
                continue
            for a in r.atoms:
                if a.ref is None:
                    continue
                for b in a.ref.bonds:
                    if b in r.map and r.map[b] not in a.bonds:
                        a.bonds.append(r.map[b])

    def set_donors_acceptors_res(self, r):
        if r.ref is None:
            return
        for a in r.atoms:
            a.hdonor = False
            a.hacceptor = False
            if a.name.startswith("N"):
                bonded = 0
                for b in a.bonds:
                    if b.name[0] == "H":
                        a.hdonor = True
                        bonded = 1
                        break
                if not bonded and r.ref.name == "HIS":
                    a.hacceptor = True
            elif a.name.startswith("O") or (a.name.startswith("S") and r.ref.name == "CYS"):
                a.hacceptor = True
                for b in a.bonds:
                    if b.name[0] == "H":
                        a.hdonor = True
                        break

    # ---------------------------------------------------------------- states
    def set_states(self):
        for r in self.residues:
            if not r.amino:
                continue
            nm = r.ffname
            base = r.ref.name
            if base == "HIS":
                # pdb2pqr HIS.set_state: donor/acceptor flags left by the optimisation decide
                if "HIP" not in r.patches and r.name not in ("HIP", "HSP"):
                    nd1, ne2 = r.map.get("ND1"), r.map.get("NE2")
                    if nd1 is not None and nd1.hdonor and not nd1.hacceptor:
                        if "HE2" in r.map:
                            r.remove_atom("HE2")
                    elif (ne2 is not None and ne2.hdonor and not ne2.hacceptor) or (
                        nd1 is not None and nd1.hacceptor and not nd1.hdonor
                    ):
                        if "HD1" in r.map:
                            r.remove_atom("HD1")
                    elif "HE2" in r.map:
                        r.remove_atom("HE2")
                if "HD1" not in r.map and "HE2" not in r.map:
                    raise ValueError(f"{r.label} has neither HD1 nor HE2")
                nm = "HIP" if ("HD1" in r.map and "HE2" in r.map) else "HID" if "HD1" in r.map else "HIE"
            elif base == "CYS":
                if "CYX" in r.patches or r.name == "CYX" or r.ss:
                    nm = "CYX"
                elif "CYM" in r.patches or r.name == "CYM":
                    nm = "CYM"
                elif "HG" not in r.map:
                    nm = "CYX"
            else:
                for patch, name in (("AR0", "AR0"), ("ASH", "ASH"), ("GLH", "GLH"), ("LYN", "LYN"), ("TYM", "TYM")):
                    if patch in r.patches:
                        nm = name
            if r.nterm:
                nm = ("NEUTRAL-N" + nm) if ("NEUTRAL-NTERM" in r.patches and base != "PRO") else "N" + nm
            elif r.cterm:
                nm = ("NEUTRAL-C" + nm) if "NEUTRAL-CTERM" in r.patches else "C" + nm
            r.ffname = nm

    # ---------------------------------------------------------------- titration
    def remove_hydrogens(self):
        for r in self.residues:
            if not r.amino:
                continue
            for a in r.atoms[:]:
                if a.name[0] == "H":
                    r.remove_atom(a.name)

    def propka_text(self):
        """The model as pdb2pqr hands it to PROPKA (io.print_biomolecule_atoms, pdbfile=True)."""
        out = []
        cur = None
        serial = 0
        for r in self.residues:
            for a in r.atoms:
                if cur is None:
                    cur = a.chain
                elif a.chain != cur:
                    cur = a.chain
                    out.append("TER\n")
                serial += 1
                s = a.type.ljust(6)[:6] + f"{serial:d}".rjust(5)[:5] + " "
                n = a.name
                s += n.ljust(4)[:4] if (len(n) == 4 or len(n.strip("FLIP")) == 4) else " " + n.ljust(3)[:3]
                s += a.resname.ljust(4)[:4] if len(a.resname) == 4 else " " + a.resname.ljust(3)[:3]
                s += " " + a.chain.ljust(1)[:1] + f"{a.resseq:d}".rjust(4)[:4]
                s += f"{a.icode}   " if a.icode != "" else "    "
                s += f"{a.x:8.3f}".ljust(8)[:8] + f"{a.y:8.3f}".ljust(8)[:8] + f"{a.z:8.3f}".ljust(8)[:8]
                occ, tf = getattr(a, "occ", 0.0), getattr(a, "tf", 0.0)
                s += f"{occ:>6.2f}{tf:>6.2f}      "
                s += f"{getattr(a, 'seg', ''):4.4s}{getattr(a, 'element', ''):>2.2s}{getattr(a, 'chg', ''):2.2s}"
                out.append(s + "\n")
        out.append("TER\nEND")
        return "".join(out)

    def titrate(self, pkadic: dict, ph: float):
        """pdb2pqr Biomolecule.apply_pka_values for the side chains (keys from :func:`_pka_key`)."""
        ff = self.ff
        for r in self.residues:
            if not r.amino:
                continue
            resname = r.name
            key = _pka_key(resname, r.resseq, r.chain, r.icode)
            if key not in pkadic:
                continue
            value = pkadic[key]
            if resname == "ARG" and ph >= value:
                if ff == "parse":
                    self.apply_patch(r, "AR0")
            elif resname == "ASP" and ph < value:
                if not ((r.cterm or r.nterm) and ff == "amber"):
                    self.apply_patch(r, "ASH")
            elif resname == "CYS" and ph >= value:
                if ff != "charmm":
                    self.apply_patch(r, "CYM")
            elif resname == "GLU" and ph < value:
                if not ((r.cterm or r.nterm) and ff == "amber"):
                    self.apply_patch(r, "GLH")
            elif resname == "HIS" and ph < value:
                self.apply_patch(r, "HIP")
            elif resname == "LYS" and ph >= value:
                if ff != "charmm" and not (ff == "amber" and r.cterm):
                    self.apply_patch(r, "LYN")
            elif resname == "TYR" and ph >= value:
                if ff == "parse":
                    self.apply_patch(r, "TYM")

    # ---------------------------------------------------------------- output
    def output(self, report, extra_residues=()):
        table = _ff_table(self.ff)
        out = []
        atoms = []
        serial = 0
        q = 0.0
        for r in list(self.residues) + list(extra_residues):
            rn = r.ffname if r.amino else r.name
            for a in r.atoms:
                if r.extra is not None:
                    p = r.extra[a.name]
                else:
                    p = table.get((rn, a.name))
                if p is None:
                    report.dropped.append(f"{r.label}:{a.name}")
                    continue
                serial += 1
                q += p[0]
                s = a.type.ljust(6)[:6] + str(serial).rjust(5)[:5] + " "
                n = a.name
                if len(n) == 4:
                    s += n.ljust(4)[:4]
                elif len(n) == 3 and len(a.resname) == 4:
                    s += n.ljust(3)[:3] + " "
                else:
                    s += " " + n.ljust(3)[:3]
                s += a.resname.ljust(4)[:4] if len(a.resname) == 4 else " " + a.resname.ljust(3)[:3]
                s += " " + (r.chain.ljust(1)[:1]) + str(a.resseq).rjust(4)[:4]
                s += (a.icode + "   ") if a.icode != "" else "    "
                s += f"{a.x:8.3f}".ljust(8)[:8] + f"{a.y:8.3f}".ljust(8)[:8] + f"{a.z:8.3f}".ljust(8)[:8]
                s += f"{p[0]:.4f}".rjust(8)[:8] + f"{p[1]:.4f}".rjust(7)[:7]
                s = s[0:6] + " " + s[6:16] + " " + s[16:38] + " " + s[38:46] + " " + s[46:]
                out.append(s + "\n")
                atoms.append((a, r, p))
            if r.amino:
                report.states[r.label] = r.ffname
        report.pqr_text = "".join(out)
        report.n_atoms = len(atoms)
        report.n_hydrogens_added = sum(1 for a, r, _ in atoms if a.added and a.name.startswith("H") and r.amino)
        report.net_charge = q
        return atoms


# ===================================================================== hydrogen-bond optimisation
# pdb2pqr hydrogens/{__init__,optimize,structures}.py: Flip (HIS, ASN, GLN),
# Alcoholic (SER, THR, TYR, CYS), Carboxylic (ASH, GLH) and Generic (no-op).
# Waters are dropped by pbsolve, so the Water class is not needed.
ANGLE_CUTOFF = 20.0
DIST_CUTOFF = 3.3


class _HAtom:
    def __init__(self, name, bond):
        self.name = name
        self.bond = bond


class _HClass:
    def __init__(self, name, opttype, optangle, map_):
        self.name = name
        self.opttype = opttype
        self.optangle = optangle
        self.map = map_


@functools.lru_cache(maxsize=1)
def _hydrogen_defs():
    out = {}
    for c in ET.parse(os.path.join(DATA, "HYDROGENS.xml")).getroot().findall("class"):
        m = {}
        for a in c.findall("atom"):
            b = a.find("bond")
            m[a.find("name").text.strip()] = _HAtom(a.find("name").text.strip(), b.text.strip() if b is not None else "")
        oa = c.find("optangle")
        name = c.find("name").text.strip()
        out[name] = _HClass(name, c.find("opttype").text.strip(), oa.text.strip() if oa is not None else "", m)
    return out


def _is_h(a):
    return a.name[0] == "H"


def _hbond_angle(a1, a2, a3):
    """Angle a1-a2-a3 in degrees (pdb2pqr Optimize.get_hbond_angle)."""
    c = a2.coords
    v1 = [a3.x - c[0], a3.y - c[1], a3.z - c[2]]
    v2 = [a1.x - c[0], a1.y - c[1], a1.z - c[2]]
    n1 = _norm3(v1)
    n2 = _norm3(v2)
    v1 = [x / n1 for x in v1]
    v2 = [x / n2 for x in v2]
    dotted = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    if dotted > 1.0:
        dotted = 1.0
    elif dotted < -1.0:
        dotted = -1.0
    angle = abs(math.acos(dotted)) * 180.0 / math.pi
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


class _Opt:
    """pdb2pqr Optimize base class."""

    residue: _Res
    routines: "_HRoutines"

    def __str__(self):
        return f"{self.residue.label} ({self.optinstance.opttype})"

    def is_hbond(self, donor, acc):
        for dh in donor.bonds:
            if not _is_h(dh):
                continue
            if _dist(dh.coords, acc.coords) > DIST_CUTOFF:
                continue
            flag = True
            for ah in acc.bonds:
                if not _is_h(ah):
                    continue
                flag = False
                if _dist(dh.coords, ah.coords) < 1.5:
                    continue
                if _hbond_angle(dh, ah, acc) < 110.0:
                    flag = True
            if not flag:
                continue
            if _hbond_angle(acc, donor, dh) <= ANGLE_CUTOFF:
                return True
        return False

    @staticmethod
    def get_pair_energy(donor, acceptor):
        energy = 0.0
        if not (donor.hdonor and acceptor.hacceptor):
            return energy
        acceptorhs = [b for b in acceptor.bonds if _is_h(b)]
        for dh in (b for b in donor.bonds if _is_h(b)):
            dist = _dist(dh.coords, acceptor.coords)
            if dist > DIST_CUTOFF and dist < 5.0:
                energy += -1.0 / (dist * dist)
                continue
            for ah in acceptorhs:
                hdist = _dist(dh.coords, ah.coords)
                if hdist < 1.5:
                    energy += 10.0
                    continue
                angle1 = _hbond_angle(acceptor, donor, dh)
                if angle1 <= ANGLE_CUTOFF:
                    angle2 = _hbond_angle(dh, ah, acceptor)
                    if angle2 < 110.0:
                        angle2 = 1.0
                    else:
                        angle2 = (110.0 - angle2) / 110.0
                    angleterm = (ANGLE_CUTOFF - angle1) / ANGLE_CUTOFF
                    energy += -10.0 / pow(dist, 3) * angleterm * angle2
            if not acceptorhs:
                angle1 = _hbond_angle(acceptor, donor, dh)
                if angle1 <= ANGLE_CUTOFF:
                    angleterm = (ANGLE_CUTOFF - angle1) / ANGLE_CUTOFF
                    energy += -10.0 / pow(dist, 2) * angleterm
        return energy

    @staticmethod
    def _link(a, b):
        if b not in a.bonds:
            a.bonds.append(b)
        if a not in b.bonds:
            b.bonds.append(a)

    @staticmethod
    def make_atom_with_one_bond_h(atom, addname):
        r = atom.res
        nxt = atom.bonds[0]
        xyz = _find_coordinates(2, [atom.coords, nxt.coords], [r.ref.map[atom.name].coords, r.ref.map[nxt.name].coords],
                                r.ref.map[addname].coords)  # fmt: skip
        r.create_atom(addname, xyz)

    def make_atom_with_one_bond_lp(self, atom, addname):
        r = atom.res
        the_refname = ""
        for refname in atom.ref.bonds:
            if refname.startswith("H"):
                the_refname = refname
                break
        nxt = atom.bonds[0]
        xyz = _find_coordinates(2, [atom.coords, nxt.coords], [r.ref.map[atom.name].coords, r.ref.map[nxt.name].coords],
                                r.ref.map[the_refname].coords)  # fmt: skip
        r.create_atom(addname, xyz)
        self._link(atom, r.map[addname])

    def try_single_alcoholic_h(self, donor, acc, newatom):
        besten = 999.99
        bestcoords = []
        r = donor.res
        pivot = donor.bonds[0]
        for _ in range(72):
            _rotate_tetrahedral(pivot, donor, 5.0)
            if self.is_hbond(donor, acc):
                energy = self.get_pair_energy(donor, acc)
                if energy < besten:
                    bestcoords = newatom.coords
                    besten = energy
        if bestcoords != []:
            newatom.x, newatom.y, newatom.z = bestcoords
            self.routines.cells.add(newatom)
            return True
        r.remove_atom(newatom.name)
        return False

    def try_single_alcoholic_lp(self, acc, donor, newatom):
        r = acc.res
        pivot = acc.bonds[0]
        bestangle = 180.0
        bestcoords = []
        the_dh = None
        if not self.is_hbond(donor, acc):
            r.remove_atom(newatom.name)
            return False
        for dh in donor.bonds:
            if _is_h(dh) and _hbond_angle(acc, donor, dh) < ANGLE_CUTOFF:
                the_dh = dh
                break
        for _ in range(72):
            _rotate_tetrahedral(pivot, acc, 5.0)
            angle = abs(_hbond_angle(the_dh, acc, newatom))
            if angle < bestangle:
                bestangle = angle
                bestcoords = newatom.coords
        if bestangle > ANGLE_CUTOFF * 2.0:
            r.remove_atom(newatom.name)
            return False
        newatom.x, newatom.y, newatom.z = bestcoords
        self.routines.cells.add(newatom)
        return True

    @staticmethod
    def get_positions_with_two_bonds(atom):
        fixed = atom.bonds[0]
        rotate = atom.bonds[1]
        _rotate_tetrahedral(fixed, atom, 120)
        loc1 = rotate.coords
        _rotate_tetrahedral(fixed, atom, 120)
        loc2 = rotate.coords
        _rotate_tetrahedral(fixed, atom, 120)
        return loc1, loc2

    def try_positions_with_two_bonds_h(self, donor, acc, newname, loc1, loc2):
        besten = 999.99
        bestcoords = []
        r = donor.res
        r.create_atom(newname, loc1)
        if self.is_hbond(donor, acc):
            besten = self.get_pair_energy(donor, acc)
            bestcoords = loc1
        newatom = r.map[newname]
        newatom.x, newatom.y, newatom.z = loc2
        if self.is_hbond(donor, acc):
            energy = self.get_pair_energy(donor, acc)
            if energy < besten:
                bestcoords = loc2
        if bestcoords != []:
            newatom.x, newatom.y, newatom.z = bestcoords
            self.routines.cells.add(newatom)
            return 1
        r.remove_atom(newname)
        return 0

    def try_positions_with_two_bonds_lp(self, acc, donor, newname, loc1, loc2):
        bestangle = 180.0
        bestcoords = []
        r = acc.res
        the_dh = None
        if not self.is_hbond(donor, acc):
            return False
        for dh in donor.bonds:
            if _is_h(dh) and _hbond_angle(acc, donor, dh) < ANGLE_CUTOFF:
                the_dh = dh
                break
        r.create_atom(newname, loc1)
        newatom = r.map[newname]
        angle = abs(_hbond_angle(the_dh, acc, newatom))
        if angle < bestangle:
            bestangle = angle
            bestcoords = loc1
        newatom.x, newatom.y, newatom.z = loc2
        angle = _hbond_angle(the_dh, acc, newatom)
        if angle < bestangle:
            bestcoords = loc2
        if bestangle > ANGLE_CUTOFF * 2.0:
            r.remove_atom(newname)
            return False
        newatom.x, newatom.y, newatom.z = bestcoords
        self.routines.cells.add(newatom)
        self._link(acc, newatom)
        return True

    @staticmethod
    def get_position_with_three_bonds(atom):
        pivot = atom.bonds[0]
        rot1 = atom.bonds[1]
        rot2 = atom.bonds[2]
        _rotate_tetrahedral(pivot, atom, 120)
        c1 = rot1.coords
        _rotate_tetrahedral(pivot, atom, 120)
        c2 = rot1.coords
        _rotate_tetrahedral(pivot, atom, 120)
        if _dist(rot2.coords, c1) > 0.1:
            return c1
        return c2

    def try_positions_three_bonds_h(self, donor, acc, newname, loc):
        r = donor.res
        r.create_atom(newname, loc)
        if self.is_hbond(donor, acc):
            self.routines.cells.add(r.map[newname])
            return True
        r.remove_atom(newname)
        return False

    def try_positions_three_bonds_lp(self, acc, donor, newname, loc):
        r = acc.res
        if not self.is_hbond(donor, acc):
            return False
        the_dh = None
        for dh in donor.bonds:
            if _is_h(dh) and _hbond_angle(acc, donor, dh) < ANGLE_CUTOFF:
                the_dh = dh
                break
        r.create_atom(newname, loc)
        newatom = r.map[newname]
        if abs(_hbond_angle(the_dh, acc, newatom)) > ANGLE_CUTOFF * 2.0:
            r.remove_atom(newname)
            return False
        self.routines.cells.add(newatom)
        self._link(acc, newatom)
        return True


class _Flip(_Opt):
    def __init__(self, residue, optinstance, routines):
        self.optinstance = optinstance
        self.residue = residue
        self.routines = routines
        self.atomlist = []
        self.hbonds = []
        run = routines.run
        dihedral = optinstance.optangle
        pivot = dihedral.split()[2]
        moveablenames = run._moveable(residue, pivot)
        if residue.cterm:
            moveablenames = [n for n in moveablenames if n != "HO"]
        map_ = {name: residue.map[name].coords for name in moveablenames}
        anglenum = residue.ref.dihedrals.index(dihedral)
        newangle = 180.0 + residue.dih[anglenum]
        run._set_dih(routines.cells, residue, anglenum, newangle)
        for name, value in map_.items():
            newname = f"{name}FLIP"
            residue.create_atom(newname, value)
            newatom = residue.map[newname]
            routines.cells.add(newatom)
            newatom.ref = residue.ref.map[name]
            for bond in newatom.ref.bonds:
                for nb in (f"{bond}FLIP", bond):
                    if nb in residue.map:
                        self._link(newatom, residue.map[nb])
        run.set_donors_acceptors_res(residue)
        for name in moveablenames:
            for atom in (residue.map[name], residue.map[f"{name}FLIP"]):
                if not _is_h(atom) and (atom.hdonor or atom.hacceptor):
                    self.atomlist.append(atom)
        if residue.ref.name == "HIS" and residue.name == "HIS" and len(residue.patches) == 1:
            for atom in self.atomlist:
                if atom.name.startswith("N"):
                    atom.hacceptor = 1

    def try_both(self, donor, acc, accobj):
        if donor.res.fixed:
            return bool(accobj.try_acceptor(acc, donor))
        if acc.res.fixed:
            return bool(self.try_donor(donor, acc))
        if self.is_hbond(donor, acc):
            if accobj.try_acceptor(acc, donor):
                self.fix_flip(donor)
                donor.hacceptor = 0
                return True
            return False
        return False

    def try_donor(self, donor, acc):
        if not acc.hacceptor:
            return False
        if self.is_hbond(donor, acc):
            self.residue.fixed = donor.name
            self.fix_flip(donor)
            donor.hacceptor = 0
            return True
        return False

    def try_acceptor(self, acc, donor):
        if not donor.hdonor:
            return False
        if self.is_hbond(donor, acc):
            acc.res.fixed = acc.name
            self.fix_flip(acc)
            acc.hdonor = 0
            return True
        return False

    def fix_flip(self, bondatom):
        r = bondatom.res
        flag = 1 if bondatom.name.endswith("FLIP") else 0
        r.was_flipped = flag == 0
        for atom in list(r.atoms):
            name = atom.name
            if name.endswith("FLIP") and flag:
                if name[:-4] in r.map:
                    self.routines.cells.remove(r.map[name[:-4]])
                    r.remove_atom(name[:-4])
            elif name.endswith("FLIP"):
                self.routines.cells.remove(atom)
                r.remove_atom(name)
        r.fixed = 1

    def finalize(self):
        r = self.residue
        if r.fixed:
            return
        for atom in list(r.atoms):
            if atom.name.endswith("FLIP"):
                self.routines.cells.remove(atom)
                r.remove_atom(atom.name[:-4])
                r.rename_atom(atom.name, atom.name[:-4])
        r.fixed = 1

    def complete(self):
        r = self.residue
        self.finalize()
        for atom in list(r.atoms):
            if atom.name.endswith("FLIP"):
                r.rename_atom(atom.name, atom.name[:-4])


class _Generic(_Opt):
    def __init__(self, residue, optinstance, routines):
        self.optinstance = optinstance
        self.residue = residue
        self.routines = routines
        self.hbonds = []
        self.atomlist = []

    def finalize(self):
        pass

    def complete(self):
        if not self.residue.fixed:
            self.finalize()


class _Alcoholic(_Opt):
    def __init__(self, residue, optinstance, routines):
        self.optinstance = optinstance
        self.residue = residue
        self.routines = routines
        self.atomlist = []
        self.hbonds = []
        name = next(iter(optinstance.map.keys()))
        self.hname = name
        bondname = residue.ref.map[name].bonds[0]
        self.atomlist.append(residue.map[bondname])
        if name in residue.map:
            routines.cells.remove(residue.map[name])
            residue.remove_atom(name)

    def try_both(self, donor, acc, accobj):
        r = donor.res
        if donor.res.fixed:
            return bool(accobj.try_acceptor(acc, donor))
        if acc.res.fixed:
            return bool(self.try_donor(donor, acc))
        if self.try_donor(donor, acc):
            if accobj.try_acceptor(acc, donor):
                return True
            r.remove_atom(self.hname)
            return False
        return False

    def try_donor(self, donor, acc):
        r = self.residue
        if not acc.hacceptor:
            return False
        newname = self.hname
        if newname in r.map:
            return False
        nb = len(donor.bonds)
        if nb == 1:
            self.make_atom_with_one_bond_h(donor, newname)
            return self.try_single_alcoholic_h(donor, acc, donor.res.map[newname])
        if nb == 2:
            loc1, loc2 = self.get_positions_with_two_bonds(donor)
            return self.try_positions_with_two_bonds_h(donor, acc, newname, loc1, loc2)
        if nb == 3:
            loc = self.get_position_with_three_bonds(donor)
            return self.try_positions_three_bonds_h(donor, acc, newname, loc)
        return False

    def try_acceptor(self, acc, donor):
        r = acc.res
        if not donor.hdonor:
            return False
        if "LP2" in r.map:
            return False
        newname = "LP2" if "LP1" in r.map else "LP1"
        nb = len(acc.bonds)
        if nb == 1:
            self.make_atom_with_one_bond_lp(acc, newname)
            return self.try_single_alcoholic_lp(acc, donor, acc.res.map[newname])
        if nb == 2:
            loc1, loc2 = self.get_positions_with_two_bonds(acc)
            return self.try_positions_with_two_bonds_lp(acc, donor, newname, loc1, loc2)
        if nb == 3:
            loc = self.get_position_with_three_bonds(acc)
            return self.try_positions_three_bonds_lp(acc, donor, newname, loc)
        return False

    def _energy(self, atom, closeatoms):
        e = 0.0
        for c in closeatoms:
            e += self.get_pair_energy(atom, c) + self.get_pair_energy(c, atom)
        return e

    def finalize(self):
        r = self.residue
        atom = self.atomlist[0]
        addname = self.hname
        cells = self.routines.cells
        if r.fixed or addname in r.map:
            return
        nb = len(atom.bonds)
        if nb == 1:
            pivot = atom.bonds[0]
            bestcoords = []
            bestenergy = 999.99
            self.make_atom_with_one_bond_h(atom, addname)
            newatom = r.map[addname]
            cells.add(newatom)
            for _ in range(18):
                _rotate_tetrahedral(pivot, atom, 20.0)
                energy = self._energy(atom, cells.near(atom))
                if energy < bestenergy:
                    bestenergy = energy
                    bestcoords = newatom.coords
            if bestcoords != []:
                newatom.x, newatom.y, newatom.z = bestcoords
        elif nb == 2:
            loc1, loc2 = self.get_positions_with_two_bonds(atom)
            r.create_atom(addname, loc1)
            newatom = r.map[addname]
            cells.add(newatom)
            closeatoms = cells.near(atom)
            energy1 = self._energy(atom, closeatoms)
            cells.remove(newatom)
            newatom.x, newatom.y, newatom.z = loc2
            cells.add(newatom)
            energy2 = self._energy(atom, closeatoms)
            if energy2 > energy1:
                cells.remove(newatom)
                newatom.x, newatom.y, newatom.z = loc1
                cells.add(newatom)
        elif nb == 3:
            loc = self.get_position_with_three_bonds(atom)
            r.create_atom(addname, loc)
            cells.add(r.map[addname])

    def complete(self):
        r = self.residue
        self.finalize()
        r.fixed = 1
        for atom in list(r.atoms):
            if atom.name.startswith("LP"):
                r.remove_atom(atom.name)


class _Carboxylic(_Opt):
    def __init__(self, residue, optinstance, routines):
        self.optinstance = optinstance
        self.residue = residue
        self.routines = routines
        self.atomlist = []
        self.hbonds = []
        self.hlist = []
        run = routines.run
        hname1 = hname2 = ""
        for name in optinstance.map:
            if name.endswith("2"):
                hname2 = name
            else:
                hname1 = name
        b1 = residue.map[optinstance.map[hname1].bond]
        b2 = residue.map[optinstance.map[hname2].bond]
        longflag = 0
        piv = next(p for p in b1.bonds if not _is_h(p))
        d1 = _dist(piv.coords, b1.coords)
        d2 = _dist(piv.coords, b2.coords)
        order = [hname1, hname2]
        if d2 > d1 and abs(d1 - d2) > 0.05:
            longflag = 1
            order = [hname2, hname1]
        elif d1 > d2 and abs(d1 - d2) > 0.05:
            longflag = 1
            order = [hname1, hname2]
        for hname in order:
            bondatom = residue.map[optinstance.map[hname].bond]
            the_dihedral = next(d for d in residue.ref.dihedrals if d.endswith(hname))
            anglenum = residue.ref.dihedrals.index(the_dihedral)
            if residue.dih[anglenum] is None:
                self.atomlist.append(bondatom)
                continue
            run._set_dih(routines.cells, residue, anglenum, 180.0 + residue.dih[anglenum])
            hatom = residue.map[hname]
            newcoords = hatom.coords
            run._set_dih(routines.cells, residue, anglenum, 180.0 + residue.dih[anglenum])
            residue.rename_atom(hname, f"{hname}1")
            newname = f"{hname}2"
            residue.create_atom(newname, newcoords)
            newatom = residue.map[newname]
            routines.cells.add(newatom)
            newatom.refd = hatom.refd
            self._link(newatom, bondatom)
            self.atomlist.append(bondatom)
            self.hlist.append(residue.map[f"{hname}1"])
            self.hlist.append(residue.map[f"{hname}2"])
            if longflag:
                break
        run.set_donors_acceptors_res(residue)

    def try_both(self, donor, acc, accobj):
        if donor.res.fixed:
            return bool(accobj.try_acceptor(acc, donor))
        if acc.res.fixed:
            return bool(self.try_donor(donor, acc))
        if self.is_hbond(donor, acc):
            if accobj.try_acceptor(acc, donor):
                self.fix(donor, acc)
                return True
            return False
        return False

    def is_carboxylic_hbond(self, donor, acc):
        for dh in donor.bonds:
            if not _is_h(dh):
                continue
            if _dist(dh.coords, acc.coords) > DIST_CUTOFF:
                continue
            if _hbond_angle(acc, donor, dh) <= ANGLE_CUTOFF:
                return True
        return False

    def try_acceptor(self, acc, donor):
        r = acc.res
        cells = self.routines.cells
        if not donor.hdonor:
            return False
        if not self.is_carboxylic_hbond(donor, acc):
            return False
        donorhatom = None
        hyds = [h for h in self.hlist if _is_h(h)]
        if len(hyds) < 2:
            return True
        dist = _dist(hyds[0].coords, donor.coords)
        dist2 = _dist(hyds[1].coords, donor.coords)
        if dist < dist2:
            self.hlist.remove(hyds[0])
            cells.remove(hyds[0])
            r.remove_atom(hyds[0].name)
            donorhatom = r.map.get(hyds[1].name)
        elif hyds[1] in self.hlist:
            self.hlist.remove(hyds[1])
            cells.remove(hyds[1])
            r.remove_atom(hyds[1].name)
            if hyds[0].name in r.map:
                donorhatom = r.map[hyds[0].name]
            elif len(self.hlist) != 0 and self.hlist[0].name in r.map:
                donorhatom = r.map[self.hlist[0].name]
        if len(self.hlist) == 1:
            if donorhatom is not None:
                self.rename(donorhatom)
            r.fixed = 1
        return True

    def try_donor(self, donor, acc):
        if not acc.hacceptor:
            return False
        if self.is_hbond(donor, acc):
            self.fix(donor, acc)
            return True
        return False

    def fix(self, donor, acc):
        r = donor.res
        the_dh = None
        for dh in donor.bonds:
            if _is_h(dh) and _hbond_angle(acc, donor, dh) <= ANGLE_CUTOFF:
                the_dh = dh
                break
        for atom in self.hlist[:]:
            if atom is not the_dh:
                self.routines.cells.remove(atom)
                self.hlist.remove(atom)
                r.remove_atom(atom.name)
        self.rename(the_dh)
        r.fixed = 1

    def finalize(self):
        r = self.residue
        if r.fixed:
            return
        bestatom = None
        bestenergy = 999.99
        for h in self.hlist:
            energy = 0
            bonded = h.bonds[0]
            for c in self.routines.cells.near(bonded):
                energy += self.get_pair_energy(bonded, c) + self.get_pair_energy(c, bonded)
            if energy < bestenergy:
                bestenergy = energy
                bestatom = h
        for h in list(self.hlist):
            if bestatom is not h:
                self.hlist.remove(h)
                r.remove_atom(h.name)
        if bestatom is not None and len(bestatom.name) == 4:
            self.rename(bestatom)
        r.fixed = 1

    def _swap_bond_atoms(self, r):
        n0 = self.atomlist[0].name
        n1 = self.atomlist[1].name
        r.rename_atom(n0, "FLIP")
        r.rename_atom(n1, n0)
        r.rename_atom("FLIP", n1)

    def rename(self, hydatom):
        r = self.residue
        if hydatom.name not in r.map:
            return
        hname = None
        if len(hydatom.name) == 4:
            hname = hydatom.name[:-1]
            r.rename_atom(hydatom.name, hname)
        if len(self.atomlist) == 2:
            if hydatom.name.endswith("1"):
                r.rename_atom(hydatom.name, f"{hydatom.name[:-1]}2")
                self._swap_bond_atoms(r)
            elif hydatom.name.endswith("OXT"):
                r.rename_atom(hydatom.name, "HO")
                self._swap_bond_atoms(r)
        elif len(self.atomlist) == 1:
            if hname is None:
                raise ValueError(f"unexpected carboxylic hydrogen {hydatom.name} in {r.label}")
            for hn in (hname[:-1] + "1", hname[:-1] + "2"):
                bondatom = r.map[self.optinstance.map[hn].bond]
                if bondatom.name != self.atomlist[0].name:
                    self.atomlist.append(bondatom)
            if hydatom.name.endswith("1"):
                if hydatom.name[:-1] + "2" in r.map:
                    r.remove_atom(f"{hydatom.name[:-1]}2")
                r.rename_atom(hydatom.name, f"{hydatom.name[:-1]}2")
                self._swap_bond_atoms(r)
            elif hydatom.name.endswith("OXT"):
                r.rename_atom(hydatom.name, "HO")
                self._swap_bond_atoms(r)

    def complete(self):
        if len(self.hlist) == 2 and self.residue.fixed == 1:
            self.residue.fixed = 0
        if not self.residue.fixed:
            self.finalize()


_OPT_CLASSES = {"Flip": _Flip, "Alcoholic": _Alcoholic, "Carboxylic": _Carboxylic, "Generic": _Generic}


class _PotentialBond:
    def __init__(self, atom1, atom2, dist):
        self.atom1 = atom1
        self.atom2 = atom2
        self.dist = dist


def _sorted_by_dist(bonds_with_dist):
    """pdb2pqr: sort_dict_by_value (descending, stable) then reverse()."""
    items = sorted(bonds_with_dist.items(), key=lambda x: x[1], reverse=True)
    out = [k for k, _ in items]
    out.reverse()
    return out


def _analyze_connectivity(map_, key):
    clist = []
    keys = [key]
    while keys:
        key = keys[0]
        if key not in clist:
            clist.append(key)
            if key in map_:
                for value in map_[key]:
                    if value not in clist:
                        keys.append(value)
        keys.pop(keys.index(key))
    return clist


class _HRoutines:
    """pdb2pqr HydrogenRoutines (proteins; waters are not present)."""

    def __init__(self, run: "_Run"):
        self.run = run
        self.defs = _hydrogen_defs()
        self.cells = None
        self.optlist: list = []
        self.atomlist: list = []
        self.resmap: dict = {}

    def is_optimizeable(self, r):
        if not r.amino:
            return None
        opt = None
        if r.name in self.defs:
            opt = self.defs[r.name]
        elif r.ref.name in self.defs:
            opt = self.defs[r.ref.name]
        else:
            for patch in r.patches:
                if patch in self.defs:
                    opt = self.defs[patch]
                    break
        if opt is not None and opt.opttype == "Alcoholic":
            if next(iter(opt.map.keys())) not in r.ref.map:
                opt = None
        if opt is not None and opt.opttype == "Water":
            opt = None
        return opt

    def _setup(self):
        run = self.run
        self.cells = _Run._Cells(5)
        for a in (a for r in run.residues for a in r.atoms):
            a.cell = None
            self.cells.add(a)
        run.calc_dihedrals()
        run.set_donors_acceptors()
        run.update_internal_bonds()
        run.set_refdist()

    def initialize(self, optimize: bool):
        self._setup()
        for r in self.run.residues:
            if r.amino:
                r.fixed = 0
        if not optimize:
            return
        for r in self.run.residues:
            opt = self.is_optimizeable(r)
            if opt is None or r.fixed == 1:
                continue
            obj = _OPT_CLASSES[opt.opttype](r, opt, self)
            self.atomlist += obj.atomlist
            self.optlist.append(obj)
            self.resmap[r] = obj

    def optimize(self):
        optlist = self.optlist
        if not optlist:
            return
        connectivity = {}
        for obj in optlist:
            connectivity[obj] = []
            for atom in obj.atomlist:
                for close in self.cells.near(atom):
                    if atom.res is close.res:
                        continue
                    if not (close.hacceptor or close.hdonor):
                        continue
                    if atom.hdonor and not atom.hacceptor and not close.hacceptor:
                        continue
                    if atom.hacceptor and not atom.hdonor and not close.hdonor:
                        continue
                    dist = _dist(atom.coords, close.coords)
                    if dist < 4.3:
                        obj.hbonds.append(_PotentialBond(atom, close, dist))
                        if close in self.atomlist:
                            closeobj = self.resmap[close.res]
                            if closeobj not in connectivity[obj]:
                                connectivity[obj].append(closeobj)
        for obj in optlist:
            if len(obj.hbonds) == 0:
                if obj.residue.fixed:
                    continue
                obj.finalize()
        networks = []
        seen = []
        for obj1 in optlist:
            if obj1.residue.fixed or obj1 in seen:
                continue
            network = _analyze_connectivity(connectivity, obj1)
            for obj2 in network:
                if obj2 not in seen:
                    seen.append(obj2)
            networks.append(network)
        for network in networks:
            # 1. bonds to atoms that are not optimised
            hbondmap = {}
            for obj in network:
                for hb in obj.hbonds:
                    if hb.atom2 not in self.atomlist:
                        hbondmap[hb] = hb.dist
            for hb in _sorted_by_dist(hbondmap):
                atom, atom2 = hb.atom1, hb.atom2
                obj = self.resmap[atom.res]
                if atom.res.fixed:
                    continue
                if atom.hdonor:
                    obj.try_donor(atom, atom2)
                if atom.hacceptor:
                    obj.try_acceptor(atom, atom2)
            # 2. bonds between optimisable groups
            hbondmap = {}
            seenlist = []
            for obj in network:
                for hb in obj.hbonds:
                    if hb.atom2 in self.atomlist:
                        if (hb.atom2, hb.atom1) not in seenlist:
                            hbondmap[hb] = hb.dist
                            seenlist.append((hb.atom1, hb.atom2))
            for hb in _sorted_by_dist(hbondmap):
                atom, atom2 = hb.atom1, hb.atom2
                obj1 = self.resmap[atom.res]
                obj2 = self.resmap[atom2.res]
                if atom.name not in atom.res.map:
                    continue
                if atom2.name not in atom2.res.map:
                    continue
                res = 0
                if atom.hdonor and atom2.hacceptor:
                    res = obj1.try_both(atom, atom2, obj2)
                if atom.hacceptor and atom2.hdonor and res == 0:
                    obj2.try_both(atom2, atom, obj1)
            # (3. water-water pairs: none)
            for obj in network:
                obj.complete()


# ===================================================================== input
def _read_pdb_records(path: PathLike, drop_water: bool, report: PrepReport):
    recs = []
    nwat = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "":
                break  # as pdb2pqr: a blank line ends the input
            rec = line[0:6].strip()
            if rec in ("ATOM", "HETATM"):
                try:
                    r = dict(kind="atom", type=rec, name=line[12:16].strip(), altloc=line[16].strip(),
                             resname=line[17:20].strip(), chain=line[21].strip(), resseq=int(line[22:26]),
                             icode=line[26].strip(), x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]))  # fmt: skip
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{path}: cannot parse record: {raw.rstrip()}") from exc
                try:  # pdb2pqr: all five fall back together
                    r.update(occ=float(line[54:60].strip()), tf=float(line[60:66].strip()), seg=line[72:76].strip(),
                             element=line[76:78].strip(), chg=line[78:80].strip())  # fmt: skip
                except (ValueError, IndexError):
                    r.update(occ=0.0, tf=0.0, seg="", element="", chg="")
                if drop_water and r["resname"] in ("HOH", "WAT"):
                    nwat += 1
                    continue
                recs.append(r)
            elif rec in ("TER", "END", "MODEL"):
                recs.append(dict(kind=rec))
    if nwat:
        report.warnings.append(f"dropped {nwat} water atoms")
    return recs


def _records_from_structure(path: PathLike, report: PrepReport):
    """Non-PDB inputs (mmCIF, GRO, ...) through pbsolve.io; chains split by chain ID."""
    from .io import load

    mol = load(path)
    if mol.coords is None:
        raise ValueError(f"{path}: no coordinates")
    recs = []
    prev_chain = None
    for i in range(mol.n_atoms):
        if mol.resnames[i] in ("HOH", "WAT"):
            continue
        if prev_chain is not None and mol.chains[i] != prev_chain:
            recs.append(dict(kind="TER"))
        prev_chain = mol.chains[i]
        x, y, z = (float(v) for v in mol.coords[i])
        recs.append(dict(kind="atom", type="HETATM" if mol.hetero[i] else "ATOM", name=str(mol.names[i]),
                         altloc="", resname=str(mol.resnames[i])[:3], chain=str(mol.chains[i])[:1],
                         resseq=int(mol.resids[i]), icode=str(mol.icodes[i])[:1], x=x, y=y, z=z))  # fmt: skip
    report.warnings.append("non-PDB input: byte-identity with pdb2pqr is only checked for PDB files")
    return recs


def _pka_key(group, res_num, chain_id, ins_code=""):
    """pdb2pqr biomolecule.pka_key."""
    ins_code = "" if ins_code in (None, ".", "?") else ins_code.strip()
    return f"{group} {res_num}{ins_code} {chain_id or ''}".strip()


def _propka_pkas(pdb_text: str) -> dict:
    """Run PROPKA 3 on the model text exactly as pdb2pqr does; side-chain pKa by pdb2pqr key."""
    try:
        import propka.run  # optional, LGPL-2.1, not bundled
    except ImportError as exc:
        raise ImportError("titration='propka' needs PROPKA 3: pip install propka") from exc
    import contextlib
    import io as _io
    import logging

    logging.getLogger("propka").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(_io.StringIO()):
        mol = propka.run.single("input.pdb", optargs=(), stream=_io.StringIO(pdb_text), write_pka=False)
    out = {}
    for g in mol.conformations["AVR"].groups:
        at = g.atom
        if g.label.startswith(at.res_name):
            out[_pka_key(at.res_name, at.res_num, at.chain_id, at.icode)] = g.pka_value
    return out


# ===================================================================== public API
def prepare(
    path: PathLike,
    forcefield: str = "amber",
    *,
    ph: Optional[float] = None,
    titration: str = "none",
    ligand: Optional[PathLike] = None,
    ligand_radii: str = "mbondi2",
    optimize: bool = True,
):
    """Add hydrogens, choose protonation states and assign charges/radii.

    Returns ``(Molecule, PrepReport)``. ``forcefield``: 'amber', 'parse' or
    'charmm'. ``ph``: apply model pKa values (:data:`MODEL_PKA`) at this pH;
    ``titration='propka'``: use PROPKA 3 pKa values at ``ph`` (default 7).
    ``ligand``: a mol2 file with partial charges to append (radii from
    ``ligand_radii``). ``optimize``: pdb2pqr's hydrogen-bond optimisation
    (its default; ``False`` = ``--noopt``). Waters are dropped.
    """
    ff = forcefield.lower()
    if ff not in FORCEFIELDS:
        raise ValueError(f"unknown force field '{forcefield}' (expected {', '.join(FORCEFIELDS)})")
    if titration not in ("none", "propka"):
        raise ValueError("titration must be 'none' or 'propka'")
    if titration == "propka" and ph is None:
        ph = 7.0
    report = PrepReport(forcefield=ff, ph=ph, titration=titration, optimized=optimize)
    p = os.fspath(path)
    if p.lower().endswith((".pdb", ".ent")):
        recs = _read_pdb_records(p, drop_water=True, report=report)
    else:
        recs = _records_from_structure(p, report)
    names = {r["resname"] for r in recs if r["kind"] == "atom"}
    if names & NUCLEIC:
        raise NotImplementedError(f"nucleic acids ({', '.join(sorted(names & NUCLEIC))}) are not supported yet")
    run = _Run(recs, ff)
    run.set_termini()
    run.update_bonds()
    run.repair_heavy(report)
    run.update_ss()
    run.debump()
    if titration == "propka":
        run.remove_hydrogens()  # as pdb2pqr before PROPKA
        run.titrate(_propka_pkas(run.propka_text()), ph)
    elif ph is not None:
        pka = {_pka_key(r.name, r.resseq, r.chain, r.icode): MODEL_PKA[r.name]
               for r in run.residues if r.amino and r.name in MODEL_PKA}  # fmt: skip
        run.titrate(pka, ph)
    run.add_hydrogens(report)
    run.debump()
    hr = _HRoutines(run)
    hr.initialize(optimize)
    hr.optimize()
    report.flipped = [r.label for r in run.residues if r.amino and r.was_flipped]
    run.cleanup()
    run.set_states()
    extra = []
    if ligand is not None:
        extra.append(_ligand_residue(ligand, ligand_radii))
    atoms = run.output(report, extra)
    report.warnings[:0] = run.warn
    if abs(report.net_charge - round(report.net_charge)) > 1e-3:
        raise ValueError(
            f"non-integer total charge {report.net_charge:+.4f} e (missing parameters for: "
            + ", ".join(report.dropped[:10])
            + "); supply ligands with ligand=, or check the structure"
        )
    return _to_molecule(atoms, report), report


def _ligand_residue(path: PathLike, radii_scheme: str):
    from .io import load

    lig = load(path)
    if not lig.has_charges:
        raise ValueError(f"{path}: the ligand file has no partial charges")
    if lig.coords is None:
        raise ValueError(f"{path}: the ligand file has no coordinates")
    radii = lig.with_radii(radii_scheme).radii if not lig.has_radii else lig.radii
    resname = str(lig.resnames[0])[:3] if lig.n_atoms else "LIG"
    recs = [
        dict(kind="atom", type="HETATM", name=str(lig.names[i]), altloc="", resname=resname, chain="L",
             resseq=int(lig.resids[i]), icode="", x=float(lig.coords[i, 0]), y=float(lig.coords[i, 1]),
             z=float(lig.coords[i, 2]))  # fmt: skip
        for i in range(lig.n_atoms)
    ]
    if len({r["name"] for r in recs}) != len(recs):
        raise ValueError(f"{path}: ligand atom names must be unique")
    res = _Res(recs, None, False)
    res.extra = {str(lig.names[i]): (float(lig.charges[i]), float(radii[i])) for i in range(lig.n_atoms)}
    return res


def _to_molecule(atoms, report):
    from .io import Molecule

    n = len(atoms)
    s = lambda vals: np.array(vals, dtype=str) if n else np.array([], dtype=object)  # noqa: E731
    elements = []
    for a, r, _ in atoms:
        e = a.name[0] if r.amino else re.sub(r"[^A-Za-z]", "", a.name)[:1]
        elements.append(e.upper() if e else "")
    return Molecule(
        names=s([a.name for a, _, _ in atoms]),
        resnames=s([a.resname for a, _, _ in atoms]),
        resids=np.array([a.resseq for a, _, _ in atoms], dtype=np.int64),
        icodes=s([a.icode for a, _, _ in atoms]),
        chains=s([r.chain for _, r, _ in atoms]),
        segids=s([""] * n),
        elements=s(elements),
        atom_types=s([""] * n),
        hetero=np.array([a.type == "HETATM" for a, _, _ in atoms], dtype=bool),
        charges=np.array([p[0] for _, _, p in atoms], dtype=np.float64),
        radii=np.array([p[1] for _, _, p in atoms], dtype=np.float64),
        masses=np.full(n, np.nan),
        bonds=np.zeros((0, 2), dtype=np.int64),
        coords=np.array([a.coords for a, _, _ in atoms], dtype=np.float64).reshape(n, 3),
        source=f"pbsolve.prep ({report.forcefield})",
        notes=list(report.warnings),
    )


# ===================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    from ._console import utf8_stdio

    utf8_stdio()
    ap = argparse.ArgumentParser(
        prog="pbsolve prep",
        description="PDB/mmCIF -> PQR: add hydrogens, assign protonation states, optimise the hydrogen-bond "
        "network, assign charges and radii (pdb2pqr compatible).",
    )
    ap.add_argument("input", help="PDB file (mmCIF/GRO accepted, see notes)")
    ap.add_argument("output", help="output PQR file")
    ap.add_argument("--ff", default="amber", choices=FORCEFIELDS, help="force field for charges/radii (default amber)")
    ap.add_argument("--ph", type=float, help="apply model-compound pKa values at this pH")
    ap.add_argument("--titration", choices=["none", "propka"], default="none", help="propka: PROPKA 3 pKa values (optional package)")
    ap.add_argument("--ligand", help="mol2 file with partial charges to append")
    ap.add_argument("--ligand-radii", default="mbondi2", choices=["bondi", "mbondi", "mbondi2", "parse"])
    ap.add_argument("--noopt", action="store_true", help="skip the hydrogen-bond optimisation (pdb2pqr --noopt)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        _, rep = prepare(
            a.input, a.ff, ph=a.ph, titration=a.titration, ligand=a.ligand, ligand_radii=a.ligand_radii,
            optimize=not a.noopt,
        )
    except (ValueError, NotImplementedError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with open(a.output, "w", encoding="ascii") as fh:
        fh.write(rep.pqr_text)
    if not a.quiet:
        print(rep.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
