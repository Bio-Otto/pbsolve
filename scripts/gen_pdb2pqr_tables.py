#!/usr/bin/env python3
"""Regenerate python/pbsolve/data/pdb2pqr/{AMBER,PARSE,CHARMM}.tsv from an installed pdb2pqr.

    pip install pdb2pqr==3.8.0 && python scripts/gen_pdb2pqr_tables.py

The tables are pdb2pqr's force-field files (*.DAT + *.names) after pdb2pqr's
own name resolution, written as `ffname atom charge radius` (repr floats, so
'-0.0' entries keep their sign). AA.xml, PATCHES.xml and HYDROGENS.xml are copied verbatim.
"""
import os
import shutil

from pdb2pqr import forcefield, io

OUT = os.path.join(os.path.dirname(__file__), "..", "python", "pbsolve", "data", "pdb2pqr")
defn = io.get_definitions()
for ff in ("amber", "parse", "charmm"):
    f = forcefield.Forcefield(ff, defn, None)
    with open(os.path.join(OUT, f"{ff.upper()}.tsv"), "w") as out:
        out.write("# ffname\tatom\tcharge\tradius\n")
        for rname, res in f.map.items():
            for aname, atom in res.atoms.items():
                out.write(f"{rname}\t{aname}\t{atom.charge!r}\t{atom.radius!r}\n")
dat = os.path.join(os.path.dirname(io.__file__), "dat")
for name in ("AA.xml", "PATCHES.xml", "HYDROGENS.xml"):
    shutil.copy(os.path.join(dat, name), OUT)
print("tables written to", os.path.abspath(OUT))
