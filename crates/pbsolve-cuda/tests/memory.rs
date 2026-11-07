//! Device memory of polar-solvation calculations on the GPU, measured with
//! the memory pool's high watermarks. The pool is shared by every backend of
//! a process, so this file holds a single test (a test binary of its own,
//! never run next to other GPU tests). Passes trivially without a GPU.
//!
//!     cargo test -p pbsolve-cuda --features cuda-13 --test memory

#![cfg(any(feature = "cuda-12", feature = "cuda-13"))]

use pbsolve_core::batch::{device_workers, polar_solvation_batch, worker_device_bytes, Job};
use pbsolve_core::pb::polar_solvation;
use pbsolve_core::solver::solve_device_bytes;
use pbsolve_core::{Atom, BoundaryCondition, GridSpec, PbParams, Preconditioner};
use pbsolve_cuda::CudaBackend;

/// Reduction scratch of one backend precision (2 × 1024 f64).
const SCRATCH: u64 = 16 << 10;

fn atoms() -> Vec<Atom> {
    let mut s: u64 = 23;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..150)
        .map(|_| Atom {
            position: [rnd(), rnd(), rnd()].map(|v| 14.0 * (v - 0.5)),
            charge: rnd() - 0.5,
            radius: 1.2 + 0.7 * rnd(),
        })
        .collect()
}

fn spec(dims: [usize; 3]) -> GridSpec {
    let fine = dims.map(|n| 0.5 * (n - 1) as f64);
    GridSpec {
        center: [0.0; 3],
        dims,
        coarse_length: fine.map(|l| 1.6 * l),
        fine_length: fine,
    }
}

fn params(pc: Preconditioner) -> PbParams {
    PbParams {
        pdie: 2.0,
        preconditioner: pc,
        // The boundary is computed on the host; sdh keeps the test fast.
        bcfl: BoundaryCondition::parse("sdh").unwrap(),
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0)
}

/// The buffers of a calculation are exactly those `solve_device_bytes`
/// counts (the pool's peak exceeds it by the reduction scratch only), and
/// concurrent workers stay within what `device_workers` sets aside for them.
#[test]
fn device_memory_of_calculations_matches_the_estimate() {
    if !pbsolve_cuda::is_available() {
        eprintln!(
            "skipped, CUDA unavailable: {}",
            pbsolve_cuda::unavailable_reason().unwrap_or_default()
        );
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let (free, total) = gpu.memory_info().unwrap();
    assert!(free > 0 && free <= total, "free {free} of {total}");
    let atoms = atoms();
    let all = [
        Preconditioner::MultigridF32,
        Preconditioner::Multigrid,
        Preconditioner::Jacobi,
    ];
    // The last grid (7.2 M nodes, ~1 GB) is large enough for the pool to
    // reserve more than POOL_SLACK_BYTES beyond the buffers.
    for (dims, pcs) in [
        ([57, 49, 41], &all[..]),
        ([33, 45, 37], &all[..]),
        ([81, 65, 49], &all[..]),
        ([201, 177, 201], &all[..1]),
    ] {
        for &pc in pcs {
            gpu.reset_pool_peak().unwrap();
            polar_solvation(&gpu, &atoms, &params(pc), &spec(dims)).unwrap();
            let peak = gpu.pool_peak().unwrap();
            // The backend's own scratch is allocated before the reset, the
            // f32 twin's during the solve.
            let scratch = if pc == Preconditioner::MultigridF32 {
                2 * SCRATCH
            } else {
                SCRATCH
            };
            assert_eq!(peak.used, solve_device_bytes(dims, pc) + scratch, "{dims:?} {pc:?}");
            assert!(peak.reserved >= peak.used);
            assert!(
                peak.reserved <= worker_device_bytes([dims], pc),
                "{dims:?} {pc:?}: reserved {} > {}",
                peak.reserved,
                worker_device_bytes([dims], pc)
            );
        }
    }
    // Three workers at once, each on its own backend and stream.
    let dims = [97, 89, 81];
    let pc = Preconditioner::MultigridF32;
    let backends: Vec<CudaBackend> = (0..3).map(|_| CudaBackend::new(0).unwrap()).collect();
    let refs: Vec<&CudaBackend> = backends.iter().collect();
    let spec = spec(dims);
    let jobs: Vec<Job> = (0..6)
        .map(|_| Job {
            atoms: &atoms,
            spec: &spec,
        })
        .collect();
    let per_worker = worker_device_bytes([dims], pc);
    let (free, total) = gpu.memory_info().unwrap();
    if device_workers(3, per_worker, free, total) < 3 {
        eprintln!("concurrent part skipped: {} MiB free", free >> 20);
        return;
    }
    gpu.reset_pool_peak().unwrap();
    for r in polar_solvation_batch(&refs, &params(pc), &jobs) {
        r.unwrap();
    }
    let peak = gpu.pool_peak().unwrap();
    assert!(peak.used <= 3 * (solve_device_bytes(dims, pc) + 2 * SCRATCH) + SCRATCH);
    assert!(
        peak.reserved <= 3 * per_worker,
        "reserved {} MiB > {} MiB",
        peak.reserved >> 20,
        (3 * per_worker) >> 20
    );
}
