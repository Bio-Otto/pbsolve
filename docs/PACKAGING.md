# Packaging: one conda command, GPU when available

Goal (same model as OpenMM): `conda install -c conda-forge pbsolve` gives the
CUDA build on machines with an NVIDIA driver and the CPU build elsewhere.

## How the variants work

`recipe/meta.yaml` builds one CPU variant and one variant per CUDA major
(`cuda_compiler_version` from conda-forge's global pinning):

* CUDA variants get **build number + 200**, so the solver prefers them.
* CUDA variants **run-depend on `__cuda`** (a virtual package that exists only
  when an NVIDIA driver is installed), so they are not installable without a
  driver and the CPU variant is chosen instead.
* CUDA variants run-depend on `cuda-nvrtc` and on a matching `cuda-version`.
  The driver itself always comes from the system.
* `recipe/build.sh` turns `cuda_compiler_version=12.x` into the Cargo feature
  `cuda-12` via `MATURIN_PEP517_ARGS`. No nvcc is invoked: kernels are
  compiled at runtime by NVRTC.

Users can pin a CUDA major explicitly: `conda install -c conda-forge pbsolve cuda-version=12`.

Your MM-GBSA/MM-PBSA package then simply lists `pbsolve` (and `openmm`) as
run dependencies and inherits the automatic GPU selection.

* `compiler('cuda')` must stay in the recipe: conda-forge turns on the CUDA
  variants only when it finds it (`CF_CUDA_ENABLED`). nvcc is not used, but
  its activation needs the C++ compiler, so the CUDA variants list
  `compiler('cxx')` as well.
* conda-forge builds CPython 3.11–3.14 (its own minimum, CFEP-25) and CUDA
  12.9 and 13.4 (global pinning, 2026-11).

## Publishing on conda-forge

1. Take a release vX.Y.0 (CI runs for such tags only and creates the GitHub
   release when every job passes; patch tags vX.Y.1–vX.Y.9 run no CI). In a
   copy of `recipe/`, replace `source: path:` with the tag tarball
   `https://github.com/Bio-Otto/pbsolve/archive/refs/tags/vX.Y.Z.tar.gz` and
   its `sha256`.
2. Open a PR to `conda-forge/staged-recipes` that adds the copy as
   `recipes/pbsolve/` (`meta.yaml`, `build.sh`, `bld.bat`,
   `conda-forge.yml`).
3. After merge, the feedstock CI builds every CPU/CUDA × Python × platform
   variant. `conda-forge.yml` makes the version bot follow GitHub releases
   only (`bot.version_updates.sources: [githubreleases]`), so it opens update
   PRs for vX.Y.0 releases and not for patch tags.

## Local test builds

conda-forge's CI is the reference build environment. Without Docker, a local
build with conda-forge's pinning reproduces it (this built the three
variants of 0.1.2 for Python 3.12; see `docs/VALIDATION.md`):

```bash
conda create -p ./cf -c conda-forge conda-build conda-forge-pinning conda-smithy
./cf/bin/conda-smithy recipe-lint --conda-forge recipes/pbsolve   # the staged-recipes linter
printf 'python:\n  - 3.12.* *_cpython\nis_python_min:\n  - false\n' > py312.yaml
CF_CUDA_ENABLED=True ./cf/bin/conda-build -c conda-forge --override-channels \
    -m ./cf/conda_build_config.yaml -m py312.yaml --croot ./bld recipes/pbsolve
```

Build from a tarball copy of the recipe (as above): `source: path: ..` copies
the whole working tree, Cargo's `target/` included. Install the result with
`-c file://$PWD/bld -c conda-forge`; `CONDA_OVERRIDE_CUDA=` simulates a
machine without a driver.

## pip / wheels

`pip install .` (or `maturin build --release [--features cuda-12]`) works too.
A wheel built with `cuda-12` needs the NVIDIA driver plus NVRTC at runtime
(`pip install nvidia-cuda-nvrtc-cu12`; for `cuda-13`, `pip install
"nvidia-cuda-nvrtc>=13,<14"`); without them it runs on the CPU. The package
finds the pip NVRTC in `site-packages/nvidia/` by itself. The NVRTC may be
newer than the driver as long as the major version matches: kernels are
compiled to CUBIN, not PTX.

### The PyPI wheels

pip cannot choose between builds of one package by the driver, so PyPI gets
**one** wheel per Python version (CPython 3.10–3.14) and platform, and the
sdist:

| Platform | Wheel | CUDA |
|---|---|---|
| Linux x86_64, aarch64 | manylinux2014 (glibc ≥ 2.17) | `cuda-12` |
| Windows x86_64 | `win_amd64` | `cuda-12` |
| macOS arm64 (≥ 11), x86_64 (≥ 10.12) | `macosx` | none (no NVIDIA driver on macOS) |

```bash
pip install pbsolve              # CPU; CUDA too if an NVRTC is already installed
pip install "pbsolve[cuda12]"    # + nvidia-cuda-nvrtc-cu12, for the GPU (Linux, Windows)
```

* Without a driver or a GPU the wheel runs on the CPU (`pbsolve info` gives
  the reason). On macOS the extra installs nothing (pip has no macOS NVRTC).
* A `cuda-12` build runs with CUDA 12 and 13 drivers. Drivers run older
  builds; checked with the 13.2 driver.
* NVRTC 12.9 compiles for sm_50 to sm_121; NVRTC 13 starts at sm_75, so a
  `cuda-13` wheel would drop Maxwell, Pascal and Volta GPUs.
* A `cuda-13` build stays available from source
  (`MATURIN_PEP517_ARGS="--features cuda-13"`) and on conda-forge.
* Other platforms install from the sdist, which needs Rust ≥ 1.85.

CI (`.github/workflows/ci.yml`) builds each platform's wheels on its own
runner (`wheels`; Linux in the manylinux2014 container, Intel macOS on
`macos-15-intel`, GitHub's last Intel image, retired in 2027) and checks that
there is exactly one wheel per CPython 3.10–3.14. `wheel-test` installs each
platform's wheel with `[cuda12]` on a runner without a GPU (Python 3.10 and
3.14) and runs `pbsolve test`. A vX.Y.0 tag attaches the files to its GitHub
release (`release`) and uploads them to PyPI (`pypi`).

The `pypi` job uses trusted publishing (no stored token) and runs only after
the maintainer has done three things:

1. On pypi.org, under *Your projects → Publishing → Add a new pending
   publisher*, register owner `Bio-Otto`, repository `pbsolve`, workflow
   `ci.yml`, environment `pypi`.
2. In the repository's *Settings → Environments*, create the environment
   `pypi`.
3. In *Settings → Secrets and variables → Actions → Variables*, set
   `PUBLISH_PYPI` = `true`.

Local check of a manylinux wheel without a container (zig as the linker):

```bash
pip install "maturin>=1.9,<2" "ziglang==0.13.0" twine
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER   # set by conda's Rust
maturin build --release --zig --compatibility manylinux2014 --features cuda-12 -o dist
maturin sdist -o dist && twine check --strict dist/*
```

## Troubleshooting

`pbsolve info` (or `pbsolve.cuda_info()`) prints whether CUDA was compiled in,
whether it is usable, and why not (missing driver, missing NVRTC, no device).
It also gives the driver's CUDA version (`driver`), the NVRTC version
(`nvrtc`) and the NVRTC file in use (`nvrtc_library`).

These environment variables control the GPU runtime:

| Variable | Effect |
|---|---|
| `PBSOLVE_NVRTC=/path/to/libnvrtc.so.13` | use exactly this NVRTC (also disables the preload) |
| `PBSOLVE_NO_PRELOAD=1` | do not preload the environment's NVRTC |
| `PBSOLVE_CUDA_CODE=ptx` or `cubin` | force the kernel code type; the default `auto` uses CUBIN whenever NVRTC knows the GPU |
