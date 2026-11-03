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

## Publishing on conda-forge

1. Tag a release on GitHub; replace `source: path:` in the recipe with the
   release tarball `url` and `sha256`.
2. Open a PR to `conda-forge/staged-recipes` with `recipe/`.
3. After merge, the feedstock CI builds every CPU/CUDA × Python × platform variant.

## Local test builds

conda-forge's CI is the reference build environment. For a local CPU check:

```bash
conda install -c conda-forge conda-build conda-forge-pinning
conda build recipe -m "$CONDA_PREFIX/conda_build_config.yaml" \
    --variants "{cuda_compiler_version: ['None']}"
```

## pip / wheels

`pip install .` (or `maturin build --release [--features cuda-12]`) works too.
A wheel built with `cuda-12` needs the NVIDIA driver plus NVRTC at runtime
(e.g. `pip install nvidia-cuda-nvrtc-cu12`); without them it runs on the CPU.

## Troubleshooting

`pbsolve info` (or `pbsolve.cuda_info()`) prints whether CUDA was compiled in,
whether it is usable, and why not (missing driver, missing NVRTC, no device).
