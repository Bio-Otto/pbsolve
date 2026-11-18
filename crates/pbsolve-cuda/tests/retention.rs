//! Memory the device's pool keeps for the calculations in progress
//! (`Backend::retain_memory`). The pool and its release threshold are shared
//! by every backend of a process, so this file holds a single test (a test
//! binary of its own, never run next to other GPU tests). Passes trivially
//! without a GPU.
//!
//!     cargo test -p pbsolve-cuda --features cuda-13 --test retention

#![cfg(any(feature = "cuda-12", feature = "cuda-13"))]

use cudarc::driver::sys;
use pbsolve_core::backend::MemoryHold;
use pbsolve_core::batch::{device_workers, polar_solvation_batch, worker_device_bytes, Job};
use pbsolve_core::pb::{polar_solvation, solve_focused, Medium};
use pbsolve_core::{Atom, Backend, BoundaryCondition, GridSpec, PbParams, Preconditioner};
use pbsolve_cuda::CudaBackend;

const MIB: u64 = 1 << 20;

fn atoms() -> Vec<Atom> {
    let mut s: u64 = 41;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..120)
        .map(|_| Atom {
            position: [rnd(), rnd(), rnd()].map(|v| 12.0 * (v - 0.5)),
            charge: rnd() - 0.5,
            radius: 1.2 + 0.7 * rnd(),
        })
        .collect()
}

/// Two focusing levels (coarse 1.6 × fine) at a spacing of 0.5 Å.
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
        bcfl: BoundaryCondition::parse("sdh").unwrap(),
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0)
}

/// Device 0's pool. `gpu` makes its context current first.
fn pool(gpu: &CudaBackend) -> sys::CUmemoryPool {
    gpu.pool_release_threshold().unwrap();
    let mut dev: sys::CUdevice = 0;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: valid out-pointers; device 0 exists (its context is current).
    unsafe {
        sys::cuDeviceGet(&mut dev, 0).result().unwrap();
        sys::cuDeviceGetMemPool(&mut pool, dev).result().unwrap();
    }
    pool
}

/// Sets the release threshold of device 0's pool (another library of the
/// process might have set one).
fn set_threshold(gpu: &CudaBackend, mut bytes: u64) {
    // SAFETY: the release threshold is a cuuint64_t.
    unsafe {
        sys::cuMemPoolSetAttribute(
            pool(gpu),
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut bytes as *mut u64).cast(),
        )
        .result()
        .unwrap();
    }
}

/// Reserved memory of device 0's pool without synchronising (a
/// synchronisation releases memory beyond the threshold by itself).
fn reserved_now(gpu: &CudaBackend) -> u64 {
    let mut v: u64 = 0;
    // SAFETY: the reserved memory is a cuuint64_t.
    unsafe {
        sys::cuMemPoolGetAttribute(
            pool(gpu),
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut v as *mut u64).cast(),
        )
        .result()
        .unwrap();
    }
    v
}

/// Reserved pool memory after allocating `bytes` and freeing them again.
fn reserved_after_freeing(gpu: &CudaBackend, bytes: u64) -> u64 {
    let v = gpu.zeros((bytes / 8) as usize).unwrap();
    drop(v);
    gpu.pool_usage().unwrap().reserved
}

/// Without a calculation the pool returns freed memory to the driver at the
/// next synchronisation (release threshold 0). A calculation raises the
/// threshold by what it retains, so the buffers of one focusing level are
/// reused by the next; at its end, also on an error, the threshold is
/// restored and the memory returned. Concurrent holds add up.
#[test]
fn calculations_keep_freed_memory_until_they_end() {
    if !pbsolve_cuda::is_available() {
        eprintln!(
            "skipped, CUDA unavailable: {}",
            pbsolve_cuda::unavailable_reason().unwrap_or_default()
        );
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 0, "the pool's default threshold");
    let idle = gpu.pool_usage().unwrap();
    let block = 64 * MIB;

    // The default: freed memory goes back at the next synchronisation.
    assert_eq!(reserved_after_freeing(&gpu, block), idle.reserved);

    // Holds raise the threshold by their bytes and keep freed memory.
    let outer = MemoryHold::new(&gpu, 256 * MIB).unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 256 * MIB);
    assert!(reserved_after_freeing(&gpu, block) >= idle.reserved + block);
    let twin = gpu.single().unwrap();
    let inner = MemoryHold::new(&twin, 100 * MIB).unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 356 * MIB, "holds add up");
    inner.release().unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 256 * MIB);
    assert!(
        gpu.pool_usage().unwrap().reserved >= idle.reserved + block,
        "kept while a hold remains"
    );
    drop(twin);
    drop(outer);
    assert_eq!(gpu.pool_release_threshold().unwrap(), 0);
    assert_eq!(reserved_now(&gpu), idle.reserved, "returned when the last hold ends");
    assert_eq!(gpu.pool_usage().unwrap(), idle);

    // A calculation retains what device_workers sets aside for it, and returns
    // everything when it ends.
    let atoms = atoms();
    let dims = [65, 57, 49];
    for pc in [Preconditioner::MultigridF32, Preconditioner::Jacobi] {
        gpu.reset_pool_peak().unwrap();
        let p = polar_solvation(&gpu, &atoms, &params(pc), &spec(dims)).unwrap();
        assert_eq!(reserved_now(&gpu), idle.reserved, "{pc:?}: returned at once");
        assert!(p.energy_kt.is_finite() && p.stats_solvated.len() == 2);
        assert!(gpu.pool_peak().unwrap().reserved <= worker_device_bytes([dims], pc));
        assert_eq!(gpu.pool_release_threshold().unwrap(), 0, "{pc:?}");
        assert_eq!(gpu.pool_usage().unwrap(), idle, "{pc:?}");
    }
    solve_focused(
        &gpu,
        &spec(dims),
        &atoms,
        &params(Preconditioner::Multigrid),
        Medium::Solvated,
    )
    .unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 0);
    assert_eq!(gpu.pool_usage().unwrap(), idle);

    // An atom inside the coarse grid but outside the fine one: the coarse
    // level is solved, the fine level fails, and the hold still ends.
    let mut outside = atoms.clone();
    outside.push(Atom {
        position: [19.0, 0.0, 0.0],
        charge: 1.0,
        radius: 1.5,
    });
    let pc = Preconditioner::Multigrid;
    let err = polar_solvation(&gpu, &outside, &params(pc), &spec(dims)).unwrap_err();
    assert!(err.to_string().contains("outside the interior"), "{err}");
    assert_eq!(gpu.pool_release_threshold().unwrap(), 0);
    assert_eq!(gpu.pool_usage().unwrap(), idle);

    // A threshold set by someone else is kept and restored.
    set_threshold(&gpu, 48 * MIB);
    polar_solvation(&gpu, &atoms, &params(pc), &spec(dims)).unwrap();
    assert_eq!(gpu.pool_release_threshold().unwrap(), 48 * MIB);
    assert!(gpu.pool_usage().unwrap().reserved <= idle.reserved + 48 * MIB);
    set_threshold(&gpu, 0);
    assert_eq!(gpu.pool_usage().unwrap(), idle);

    // Concurrent workers on their own backends and streams.
    let backends: Vec<CudaBackend> = (0..3).map(|_| CudaBackend::new(0).unwrap()).collect();
    let refs: Vec<&CudaBackend> = backends.iter().collect();
    let idle = gpu.pool_usage().unwrap();
    let pc = Preconditioner::MultigridF32;
    let spec = spec(dims);
    let jobs: Vec<Job> = (0..6)
        .map(|_| Job {
            atoms: &atoms,
            spec: &spec,
            atom_pdie: None,
        })
        .collect();
    let per_worker = worker_device_bytes([dims], pc);
    let (free, total) = gpu.memory_info().unwrap();
    if device_workers(3, per_worker, free, total) < 3 {
        eprintln!("concurrent part skipped: {} MiB free", free >> 20);
        return;
    }
    let sequential = polar_solvation(&gpu, &atoms, &params(pc), &spec).unwrap();
    gpu.reset_pool_peak().unwrap();
    for r in polar_solvation_batch(&refs, &params(pc), &jobs) {
        assert_eq!(r.unwrap().energy_kt.to_bits(), sequential.energy_kt.to_bits());
    }
    assert!(gpu.pool_peak().unwrap().reserved <= 3 * per_worker);
    assert_eq!(gpu.pool_release_threshold().unwrap(), 0);
    assert_eq!(gpu.pool_usage().unwrap(), idle);
}
