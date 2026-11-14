# Third-party notices

pbsolve is distributed under the MIT License (see `LICENSE`). It contains
portions derived from the third-party software listed below, which remain
under their own licenses. Both licenses are permissive; redistributions of
pbsolve (source or binary) must keep this file.

## APBS 3.4.1 (BSD-3-Clause)

To reproduce APBS results exactly, parts of the following pbsolve files are
derived from the APBS 3.4.1 source code (https://github.com/Electrostatics/apbs):

| pbsolve file | derived from (APBS) |
|---|---|
| `crates/pbsolve-core/src/surface.rs` | `src/generic/vacc.c` (`VaccSurf_refSphere`, `Vacc_atomSurf`, `ivdwAccExclus`), `src/generic/vpbe.c` (cell-list radius), `src/mg/vpmg.c` (`markSphere`, `fillcoCoefMolDielNoSmooth`, `fillcoCoefMolIon`) |
| `crates/pbsolve-core/src/boundary.rs` | `src/mg/vpmg.c` (`bcflnew`, `bcfl1sp`, `bcfl_sdh`, `multipolebc`) |
| `crates/pbsolve-core/src/autogrid.rs` | `src/generic/mgparm.c` (`MGparm_check`), `src/generic/nosh.c` (`NOsh_setupCalcMGAUTO`) |
| `crates/pbsolve-core/src/charge.rs` | `src/mg/vpmg.c` (`bspline2`, `bspline4`, `fillcoChargeSpline2`, `fillcoPermanentMultipole`, `Vpmg_qfEnergy`) |

APBS is not bundled with pbsolve; `pbsolve.apbs` only runs an APBS executable
that the user installed, for validation. pbsolve is an independent project and
is not affiliated with or endorsed by the APBS developers, Battelle Memorial
Institute, Pacific Northwest National Laboratory, Washington University in
St. Louis or any other APBS contributor.

The APBS license follows verbatim:

```text
APBS -- Adaptive Poisson-Boltzmann Solver

  Nathan A. Baker (nathan.baker@pnnl.gov)
  Pacific Northwest National Laboratory

  Additional contributing authors listed in the code documentation.

Copyright (c) 2010-2022 Battelle Memorial Institute.
Developed at the Pacific Northwest National Laboratory, operated by Battelle Memorial Institute, Pacific Northwest Division for the U.S. Department of Energy.

Portions Copyright (c) 2002-2010, Washington University in St. Louis.
Portions Copyright (c) 2002-2010, Nathan A. Baker.
Portions Copyright (c) 1999-2002, The Regents of the University of California.
Portions Copyright (c) 1995, Michael Holst.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the developer nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## xdrfile 1.1.4 (BSD-2-Clause)

The XTC decompression in `crates/pbsolve-io/src/xdr.rs` (`decompress_coords`,
`BitReader`, `decode_ints`, `sizeofint`, `sizeofints`, `MAGICINTS`) is a Rust
port of xdrfile (as distributed with MDAnalysis), by Erik Lindahl and David
van der Spoel. The xdrfile license follows verbatim:

```text
Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## pdb2pqr 3.8.0 (BSD-3-Clause)

`python/pbsolve/prep.py` follows the algorithms of pdb2pqr 3.8.0
(https://github.com/Electrostatics/pdb2pqr): termini and patches, heavy-atom
repair, hydrogen placement with the quaternion fit, debumping, hydrogen-bond
optimisation, protonation states, force-field lookup and PQR formatting.
`python/pbsolve/data/pdb2pqr/` contains pdb2pqr's `AA.xml`, `PATCHES.xml` and
`HYDROGENS.xml` and its AMBER, PARSE and CHARMM
charge/radius tables (`*.tsv`, generated from pdb2pqr's `*.DAT`/`*.names`).
PROPKA (LGPL-2.1) is not bundled; `titration="propka"` imports it only if the
user installed it. pbsolve is not affiliated with or endorsed by the pdb2pqr
developers or their institutions. The pdb2pqr license follows verbatim:

```text
Copyright (c) 2002-2024, Jens Erik Nielsen; Nathan A. Baker; Battelle Memorial Institute, Developed at the Pacific Northwest National Laboratory, operated by Battelle Memorial Institute, Pacific Northwest Division for the U.S. Department Energy.; Paul Czodrowski & Gerhard Klebe, University of Marburg.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the names of University College Dublin, Battelle Memorial Institute, Pacific Northwest National Laboratory, US Department of Energy, or University of Marburg nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
