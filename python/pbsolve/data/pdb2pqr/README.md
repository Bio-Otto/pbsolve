# pdb2pqr data used by `pbsolve.prep`

* `AA.xml`, `PATCHES.xml`, `HYDROGENS.xml` — residue topologies, patches and
  optimisable hydrogen groups, copied verbatim from
  pdb2pqr 3.8.0 (`pdb2pqr/dat/`).
* `AMBER.tsv`, `PARSE.tsv`, `CHARMM.tsv` — the force-field charge/radius tables
  of pdb2pqr 3.8.0 (`*.DAT` + `*.names`) after pdb2pqr's own name resolution:
  `ffname atom charge radius`. Regenerate with `scripts/gen_pdb2pqr_tables.py`.

pdb2pqr is distributed under the BSD-3-Clause license; see `THIRD_PARTY_NOTICES.md`.
