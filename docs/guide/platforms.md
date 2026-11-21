# CPU and GPU

pbsolve runs the same algorithm on the CPU (all cores) and on NVIDIA GPUs
through CUDA. `platform="auto"`, the default, uses CUDA when it is usable and
the CPU otherwise; `"CPU"` and `"CUDA"` force one. `device` picks the GPU.

```python
import pbsolve

print(pbsolve.available_platforms())  # e.g. ['CUDA', 'CPU']
info = pbsolve.cuda_info()
print(info["compiled"], info["available"], info["reason"])

solver = pbsolve.Solver(platform="auto")
print(solver.platform, solver.device_name)
```

`pbsolve info` prints the same from the shell, and `pbsolve test` solves a
Born ion on every available platform and checks the results against the
analytic energy and against each other.

## What CUDA needs

* An NVIDIA driver. The CUDA toolkit is **not** needed: pbsolve compiles its
  kernels for your GPU at runtime with NVRTC.
* The NVRTC library of the CUDA major version pbsolve was built for
  (`cuda_info()["cuda_major"]`): the conda package `cuda-nvrtc`, or the pip
  package `nvidia-cuda-nvrtc-cu12` (CUDA 12) or `nvidia-cuda-nvrtc`
  (CUDA 13). pbsolve finds it in the active environment. With the Linux and
  Windows wheels (built for CUDA 12), `pip install "pbsolve[cuda12]"` installs it.

If anything is missing, a CUDA-enabled pbsolve still works on the CPU, and
`cuda_info()["reason"]` says why CUDA is not available. `platform="CUDA"`
then raises `RuntimeError` instead of falling back.

| Environment variable | Effect |
|---|---|
| `PBSOLVE_NVRTC` | path or name of the NVRTC library to load |
| `PBSOLVE_NO_PRELOAD=1` | do not preload the environment's NVRTC at import |
| `PBSOLVE_CUDA_CODE=ptx` or `cubin` | force the kernel format (diagnostics) |
| `RAYON_NUM_THREADS` | number of CPU threads (default: all cores); set it before the first calculation |

## Agreement between platforms

On CUDA the dielectric and ion maps, the operator, the right-hand side and
the multigrid hierarchy are built on the GPU with the CPU's operation order,
so these intermediate results are identical to the CPU's bit for bit. The
energies agree to rounding, 1e-13 relative or better (1YCR: 1.3e-13); the
test suite requires 1e-8 (`tests/test_platforms.py`).

The number of CPU threads does not change any result: sums run in a fixed
order of fixed-size chunks.

```python
cpu = pbsolve.Solver(platform="CPU", ionic_strength=0.15)
coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
reference = cpu.polar_solvation(coords, charges, radii).energy
if "CUDA" in pbsolve.available_platforms():
    gpu = pbsolve.Solver(platform="CUDA", ionic_strength=0.15)
    energy = gpu.polar_solvation(coords, charges, radii).energy
    print(f"CPU/CUDA relative difference: {abs(energy / reference - 1):.1e}")
```

## Work split on CUDA

Everything runs in double precision except the preconditioner's V-cycle with
the default `"multigrid-f32"`. The surface points of the molecular surface
and the analytic `mdh` boundary stay on the CPU: on consumer GPUs, whose
double-precision throughput is 1/32 to 1/64 of single precision, a GPU
version of the `mdh` sum would be slower. pbsolve solves the reference state
first; meanwhile the CPU builds the solvated state's surface and boundary.
The [tutorial](../tutorial) (section 7) shows the stages and
their timings, and [Architecture](../ARCHITECTURE) the details.

The first CUDA calculation in a process also sets up the GPU context, so
time a later one.

## GPU memory

A calculation keeps the GPU memory it frees between focusing levels and
returns it to the driver when it ends. Batches ([many calculations](many.md))
run up to 3 calculations at once on CUDA by default, but only as many as fit
in the GPU memory free at the time:
{py:meth}`~pbsolve.Solver.batch_workers` tells how many that is for a grid.

```python
grid = solver.make_grid(coords, radii)
print(solver.batch_workers(grid))
```

If a single calculation does not fit, use a coarser `grid_spacing`, a smaller
`fadd`, or the CPU.
