# Contributing

## Development setup

```bash
conda create -n pbsolve-dev -c conda-forge python=3.12 rust maturin numpy pytest openmm
conda activate pbsolve-dev
maturin develop --release                      # CPU-only build
maturin develop --release --features cuda-12   # adds the CUDA backend (no CUDA toolkit needed to build)
pytest -q && python -m pbsolve test
cargo test -p pbsolve-core
```

## Rules that keep the project healthy

1. **Physics lives in `pbsolve-core` only.** Backends implement the small
   `Backend` trait; the Python layer only converts arguments.
2. **Every change to numerics comes with numbers.** Run the Born-ion study
   (`cargo run --release -p pbsolve-core --example born`) and, if you touch
   surfaces, boundaries or energies, the APBS comparison
   (`scripts/compare_apbs.py`). Paste before/after results in the PR.
3. **Platforms must agree.** A new or modified backend must pass
   `tests/test_platforms.py` against the CPU reference.
4. **Determinism.** Reductions must have a fixed order; no atomics in sums.
5. `cargo fmt`, `cargo clippy -- -D warnings` and `pytest` must pass (CI enforces this).
