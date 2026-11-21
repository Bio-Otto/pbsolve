# Installation

pbsolve is a Python package with a compiled core (Rust), installed from
source until the conda-forge and PyPI packages exist.

```{include} ../README.md
:start-after: <!-- docs-install-start -->
:end-before: <!-- docs-install-end -->
```

## Building from source

Building needs Python 3.10 or newer, a Rust toolchain (1.85 or newer) and
[maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/Bio-Otto/pbsolve
cd pbsolve
pip install "maturin>=1.9,<2"
maturin develop --release                      # CPU only
maturin develop --release --features cuda-12   # CPU + CUDA 12
maturin develop --release --features cuda-13   # CPU + CUDA 13
```

`cuda-12` and `cuda-13` are mutually exclusive: pick the major version of the
CUDA driver and NVRTC you will run with. NVRTC comes from conda
(`cuda-nvrtc`) or pip (`nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-nvrtc` for 13);
`pbsolve info` shows which library was found.

## Checking the installation

`pbsolve test` solves a Born ion on every available platform and compares the
platforms; `pbsolve info` reports whether CUDA was compiled in, whether it is
usable and, if not, why.

## Optional tools

* [APBS](https://github.com/Electrostatics/apbs) 3.4.1 (`conda install -c
  conda-forge apbs=3.4.1`) for `pbsolve apbs compare` and the APBS checks of
  the test suite;
* [OpenMM](https://openmm.org) for `pbsolve.openmm_bridge`;
* pdb2pqr and PROPKA, only to compare `pbsolve.prep` against them.
