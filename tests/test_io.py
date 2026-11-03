"""Readers for simulation output (pbsolve.io / pbsolve-io).

Binary formats are written by small writers below (or, for compressed XTC,
taken from tests/data/synthetic.xtc, made by `gmx trjconv` from
synthetic.gro), so no MD package is needed. scripts/validate_io.py checks
the readers against MDAnalysis/ParmEd on files from tleap, OpenMM, cpptraj
and GROMACS.
"""

import pathlib
import struct

import numpy as np
import pytest

import pbsolve
from pbsolve import io

DATA = pathlib.Path(__file__).parent / "data"


# ----------------------------------------------------------------- writers
def write_dcd(path, frames, cells=None, big_endian=False):
    e = ">" if big_endian else "<"
    n = frames.shape[1]
    with open(path, "wb") as f:

        def rec(b):
            f.write(struct.pack(e + "i", len(b)))
            f.write(b)
            f.write(struct.pack(e + "i", len(b)))

        icntrl = [len(frames), 0, 1, 0, 0, 0, 0, 0, 0]
        tail = [1 if cells is not None else 0, 0, 0, 0, 0, 0, 0, 0, 0, 24]
        rec(b"CORD" + struct.pack(e + "9i", *icntrl) + struct.pack(e + "f", 0.01) + struct.pack(e + "10i", *tail))
        rec(struct.pack(e + "i", 1) + b"pbsolve test".ljust(80))
        rec(struct.pack(e + "i", n))
        for k, x in enumerate(frames):
            if cells is not None:
                a, b, c, al, be, ga = cells[k]
                rec(struct.pack(e + "6d", a, ga, b, be, al, c))
            for d in range(3):
                rec(x[:, d].astype(e + "f4").tobytes())


def _nc_name(s):
    b = s.encode()
    return struct.pack(">I", len(b)) + b + b"\0" * ((4 - len(b) % 4) % 4)


def write_amber_netcdf(path, frames, cells, times):
    """Amber NetCDF trajectory, NetCDF-3 64-bit offset (written from the format spec)."""
    nf, n, _ = frames.shape
    dims = [("frame", 0), ("spatial", 3), ("atom", n), ("cell_spatial", 3), ("cell_angular", 3)]
    text_att = lambda name, v: _nc_name(name) + struct.pack(">II", 2, len(v)) + v.encode() + b"\0" * ((4 - len(v) % 4) % 4)  # noqa: E731
    gatts = [text_att("Conventions", "AMBER"), text_att("ConventionVersion", "1.0")]
    # (name, dimids, nc_type, vsize)
    vars_ = [("time", [0], 5, 4), ("coordinates", [0, 2, 1], 5, 12 * n), ("cell_lengths", [0, 3], 6, 24), ("cell_angles", [0, 4], 6, 24)]

    def header(begins):
        h = b"CDF\x02" + struct.pack(">I", nf)
        h += struct.pack(">II", 0x0A, len(dims)) + b"".join(_nc_name(d) + struct.pack(">I", l) for d, l in dims)
        h += struct.pack(">II", 0x0C, len(gatts)) + b"".join(gatts)
        h += struct.pack(">II", 0x0B, len(vars_))
        for (name, dimids, t, vsize), b in zip(vars_, begins):
            h += _nc_name(name) + struct.pack(">I", len(dimids)) + b"".join(struct.pack(">I", d) for d in dimids)
            h += struct.pack(">II", 0, 0) + struct.pack(">II", t, vsize) + struct.pack(">Q", b)
        return h

    hsize = len(header([0] * len(vars_)))
    begins, off = [], hsize
    for v in vars_:
        begins.append(off)
        off += v[3]
    with open(path, "wb") as f:
        f.write(header(begins))
        for k in range(nf):
            f.write(struct.pack(">f", times[k]))
            f.write(frames[k].astype(">f4").tobytes())
            f.write(np.asarray(cells[k][:3], dtype=">f8").tobytes())
            f.write(np.asarray(cells[k][3:], dtype=">f8").tobytes())


def write_mdcrd(path, frames, box=None):
    with open(path, "w") as f:
        f.write("pbsolve test mdcrd\n")
        for x in frames:
            v = x.reshape(-1)
            for i in range(0, v.size, 10):
                f.write("".join(f"{a:8.3f}" for a in v[i : i + 10]) + "\n")
            if box is not None:
                f.write("".join(f"{a:8.3f}" for a in box) + "\n")


PRMTOP = """%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/26  00:00:00
%FLAG TITLE
%FORMAT(20a4)
test
%FLAG POINTERS
%FORMAT(10I8)
       4       2       2       1       0       0       0       0       0       0
       5       2       0       0       0       1       1       0       1       0
       0       0       0       0       0       0       0       1       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
N   H1  CL1 Na+
%FLAG CHARGE
%FORMAT(5E16.8)
 -7.28892000E+00  7.28892000E+00 -1.82223000E+01  1.82223000E+01
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       7       1      17      11
%FLAG MASS
%FORMAT(5E16.8)
  1.40100000E+01  1.00800000E+00  3.54500000E+01  2.29900000E+01
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
LIG Na+
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1       4
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
N   H   Cl  Na+
%FLAG RADIUS_SET
%FORMAT(1a80)
modified Bondi radii (mbondi2)
%FLAG RADII
%FORMAT(5E16.8)
  1.55000000E+00  1.30000000E+00  1.70000000E+00  1.86800000E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       6       2
"""

INPCRD = """test
    4
  -1.0000000-100.0000000   2.5000000   0.0000000   0.0000000   1.0000000
   1.5000000   0.0000000   0.0000000   9.0000000   9.0000000   9.0000000
  30.0000000  30.0000000  30.0000000  90.0000000  90.0000000  90.0000000
"""

PSF = """PSF EXT

       1 !NTITLE
 REMARKS test

       4 !NATOM
         1 LIG      1        LIG      N        NH1     -0.400000       14.0070           0
         2 LIG      1        LIG      HN       H        0.400000        1.0080           0
         3 LIG      1        LIG      CL1      CLGR1   -1.000000       35.4500           0
         4 ION      2        SOD      SOD      SOD      1.000000       22.9898           0

       2 !NBOND: bonds
       1       2       1       3
"""

TOP = """; test topology
#include "ff.itp"
#define EXTRA

[ moleculetype ]
LIG 3

[ atoms ]
  1  NX   1  LIG  N    1  -0.40  14.007
  2  HX   1  LIG  HN   1   0.40   1.008
  3  S    1  LIG  S1   1  -1.00
#ifdef EXTRA
#else
  4  HX   1  LIG  BAD  1   9.99
#endif

[ bonds ]
  1  2  1

#include "ions.itp"

[ system ]
test

[ molecules ]
LIG 2
NA  1
"""

FF_ITP = """[ defaults ]
1 2 yes 0.5 0.8333
[ atomtypes ]
NX   7  14.01  0.0  A  0.3  0.7
HX   1   1.008 0.0  A  0.1  0.1
S   16  32.06  0.0  A  0.35 1.0
NA  11  22.99  0.0  A  0.33 0.01
"""

IONS_ITP = """[ moleculetype ]
NA 1
[ atoms ]
1 NA 1 NA NA 1 1.0 22.99
"""


@pytest.fixture
def amber_files(tmp_path):
    (tmp_path / "t.prmtop").write_text(PRMTOP)
    (tmp_path / "t.inpcrd").write_text(INPCRD)
    return tmp_path / "t.prmtop", tmp_path / "t.inpcrd"


# ----------------------------------------------------------------- topologies
def test_prmtop_and_inpcrd(amber_files):
    mol = io.load(*amber_files)
    assert list(mol.names) == ["N", "H1", "CL1", "Na+"]
    np.testing.assert_allclose(mol.charges, [-0.4, 0.4, -1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(mol.radii, [1.55, 1.3, 1.7, 1.868])
    assert list(mol.elements) == ["N", "H", "Cl", "Na"]
    assert list(mol.resids) == [1, 1, 1, 2]
    assert sorted(map(tuple, mol.bonds)) == [(0, 1), (0, 2)]
    assert mol.periodic is True
    np.testing.assert_allclose(mol.coords[0], [-1.0, -100.0, 2.5])  # touching fixed-width fields
    np.testing.assert_allclose(mol.cells[0], [30, 30, 30, 90, 90, 90])
    solute = mol.strip_solvent()
    assert solute.n_atoms == 3 and solute.bonds.shape == (2, 2)


def test_psf_elements_and_bonds(tmp_path):
    p = tmp_path / "t.psf"
    p.write_text(PSF)
    mol = io.load(p)
    assert list(mol.elements) == ["N", "H", "Cl", "Na"]
    assert list(mol.segids) == ["LIG", "LIG", "LIG", "ION"]
    np.testing.assert_allclose(mol.charges, [-0.4, 0.4, -1.0, 1.0])
    assert np.isnan(mol.radii).all()
    r = mol.with_radii("mbondi2").radii
    np.testing.assert_allclose(r, [1.55, 1.3, 1.7, 1.5])  # H on N -> 1.3


def test_gromacs_top_preprocessor(tmp_path):
    (tmp_path / "t.top").write_text(TOP)
    (tmp_path / "ff.itp").write_text(FF_ITP)
    (tmp_path / "ions.itp").write_text(IONS_ITP)
    mol = io.load(tmp_path / "t.top")
    assert mol.n_atoms == 7  # 2 x LIG (3 atoms, #else branch skipped) + NA
    assert list(mol.elements[:3]) == ["N", "H", "S"]  # atom type named "S"
    np.testing.assert_allclose(mol.masses[2], 32.06)  # mass from [ atomtypes ]
    assert list(mol.segids) == ["LIG"] * 6 + ["NA"]
    assert sorted(map(tuple, mol.bonds)) == [(0, 1), (3, 4)]
    with pytest.raises(ValueError, match="not found"):
        (tmp_path / "ff.itp").unlink()
        io.load(tmp_path / "t.top")


# ----------------------------------------------------------------- trajectories
def test_xtc_trr_match_gro():
    gro = io.load(DATA / "synthetic.gro")
    assert gro.models.shape == (3, 40, 3)
    for name in ("synthetic.xtc", "synthetic.trr"):
        xyz = np.array(list(io.frames(gro, DATA / name)))
        np.testing.assert_allclose(xyz, gro.models, atol=1e-4)
    r = pbsolve.TrajectoryReader(str(DATA / "synthetic.xtc"))
    _, cell, t, step = next(r)
    np.testing.assert_allclose(cell, [40, 45, 50, 90, 90, 90], atol=1e-4)
    assert (t, step) == (0.0, 0)


@pytest.mark.parametrize("big_endian", [False, True])
def test_dcd(tmp_path, big_endian):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 7, 3)).astype(np.float32) * 10
    cells = [[30, 31, 32, 90, 90, 90]] * 4
    write_dcd(tmp_path / "t.dcd", x, cells, big_endian)
    r = pbsolve.TrajectoryReader(str(tmp_path / "t.dcd"))
    assert (r.n_atoms, r.n_frames) == (7, 4)
    got = list(r)
    np.testing.assert_array_equal(np.array([g[0] for g in got]), x.astype(np.float64))
    np.testing.assert_allclose(got[2][1], cells[2])


def test_amber_netcdf(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 5, 3)).astype(np.float32)
    cells = [[20 + k, 21, 22, 90, 91, 92] for k in range(3)]
    write_amber_netcdf(tmp_path / "t.nc", x, cells, [1.0, 2.0, 3.0])
    r = pbsolve.TrajectoryReader(str(tmp_path / "t.nc"))
    assert (r.n_atoms, r.n_frames) == (5, 3)
    got = list(r)
    np.testing.assert_array_equal(np.array([g[0] for g in got]), x.astype(np.float64))
    np.testing.assert_allclose(got[1][1], cells[1])
    assert got[2][2] == 3.0


@pytest.mark.parametrize("with_box", [True, False])
def test_mdcrd(tmp_path, with_box):
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(size=(3, 7, 3)) * 30, 3)
    write_mdcrd(tmp_path / "t.mdcrd", x, [30.0, 31.0, 32.0] if with_box else None)
    got = np.array(list(io.frames(7, tmp_path / "t.mdcrd")))
    np.testing.assert_allclose(got, x, atol=1e-9)
    with pytest.raises(ValueError, match="number of atoms"):
        pbsolve.TrajectoryReader(str(tmp_path / "t.mdcrd"))


def test_frames_stride_and_subset(tmp_path):
    x = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
    write_dcd(tmp_path / "t.dcd", x)
    got = list(io.frames(4, tmp_path / "t.dcd", start=1, stop=5, step=2, atoms=[0, 3]))
    assert len(got) == 2
    np.testing.assert_array_equal(got[1], x[3][[0, 3]])
    with pytest.raises(ValueError, match="atoms"):
        list(io.frames(5, tmp_path / "t.dcd"))


# ----------------------------------------------------------------- molecules
def test_selection_language(amber_files):
    mol = io.load(*amber_files)
    sel = mol.select
    assert list(sel("resname LIG")) == [True, True, True, False]
    assert list(sel("name H* or element Cl")) == [False, True, True, False]
    assert list(sel("not (resid 1 and hydrogen)")) == [True, False, True, True]
    assert list(sel("index 0-1 3")) == [True, True, False, True]
    assert list(sel("serial 4")) == [False, False, False, True]
    assert list(sel("ions")) == [False, False, False, True]
    for bad in ("resname", "resid x", "(resname LIG", "frobnicate 1"):
        with pytest.raises(ValueError):
            sel(bad)


def test_radii_from_coordinates_when_bonds_are_missing():
    el = ["N", "H", "C", "H", "O", "H"]
    xyz = np.array([[0, 0, 0], [1.0, 0, 0], [5, 0, 0], [6.05, 0, 0], [10, 0, 0], [10.95, 0, 0]], dtype=float)
    np.testing.assert_allclose(pbsolve.radii.assign_radii(el, "mbondi2", coords=xyz), [1.55, 1.3, 1.7, 1.2, 1.5, 1.2])
    np.testing.assert_allclose(pbsolve.radii.assign_radii(el, "mbondi", coords=xyz), [1.55, 1.3, 1.7, 1.3, 1.5, 0.8])


def test_pqr_round_trip_and_cli(tmp_path, amber_files):
    out = tmp_path / "t.pqr"
    from pbsolve.__main__ import main

    assert main(["convert", str(amber_files[0]), "--coords", str(amber_files[1]), str(out)]) == 0
    back = io.load(out)
    assert back.n_atoms == 3  # Na+ stripped by default
    np.testing.assert_allclose(back.charges, [-0.4, 0.4, -1.0])
    np.testing.assert_allclose(back.coords[2], [1.5, 0.0, 0.0])
    assert main(["solvation", str(out), "--spacing", "0.6", "--json"]) == 0
    assert main(["solvation", str(amber_files[0])]) == 2  # no coordinates -> clear error
