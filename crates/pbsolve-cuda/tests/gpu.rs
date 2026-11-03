//! GPU tests of the CUDA backend: NVRTC loading, code generation and
//! agreement with the CPU backend. Each test passes trivially (with a note on
//! stderr) when no usable CUDA device is present, so the suite also runs on
//! GPU-less machines.
//!
//!     cargo test -p pbsolve-cuda --features cuda-13 --test gpu

#![cfg(any(feature = "cuda-12", feature = "cuda-13"))]

use pbsolve_core::charge::spread_charges;
use pbsolve_core::operator::{assemble_rhs, StencilOperator};
use pbsolve_core::solver::{pcg, SolveStats, SolverOptions};
use pbsolve_core::units::bjerrum_length_vacuum;
use pbsolve_core::{Atom, Backend, CoefficientMaps, CpuBackend, Grid, PbParams, Preconditioner};
use pbsolve_cuda::{CodeType, CudaBackend};
use std::sync::Arc;

fn gpu_present(test: &str) -> bool {
    if pbsolve_cuda::is_available() {
        return true;
    }
    eprintln!(
        "{test}: skipped, CUDA unavailable: {}",
        pbsolve_cuda::unavailable_reason().unwrap_or_default()
    );
    false
}

fn overridden() -> bool {
    std::env::var_os("PBSOLVE_NVRTC").is_some_and(|v| !v.is_empty())
}

/// A small LPBE system: 60 random atoms on a 49³ grid, salt, ε 2/78.54.
fn system() -> (Arc<StencilOperator>, Vec<f64>) {
    let mut s: u64 = 11;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let atoms: Vec<Atom> = (0..60)
        .map(|_| Atom {
            position: [rnd(), rnd(), rnd()].map(|v| 16.0 * (v - 0.5)),
            charge: rnd() - 0.5,
            radius: 1.2 + 0.7 * rnd(),
        })
        .collect();
    let params = PbParams {
        pdie: 2.0,
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0);
    let grid = Grid::centered([0.0; 3], [30.0; 3], [49; 3]).unwrap();
    let op = Arc::new(StencilOperator::new(
        &grid,
        CoefficientMaps::solvated(&grid, &atoms, &params),
    ));
    let q = spread_charges(&grid, &atoms).unwrap();
    let b = assemble_rhs(
        &op,
        &q,
        &vec![0.0; grid.len()],
        bjerrum_length_vacuum(params.temperature),
    );
    (op, b)
}

fn solve<B: Backend>(be: &B, op: &Arc<StencilOperator>, b: &[f64], pc: Preconditioner) -> (Vec<f64>, SolveStats) {
    let opts = SolverOptions {
        preconditioner: pc,
        ..SolverOptions::new(1e-10, 10_000)
    };
    pcg(be, op.clone(), b, opts).unwrap()
}

fn relative_difference(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let n: f64 = a.iter().map(|x| x * x).sum();
    (d / n).sqrt()
}

/// The NVRTC of the build's CUDA major is preferred; the unversioned
/// `libnvrtc.so` (any major) is only a fallback when none is installed.
#[test]
fn nvrtc_of_the_build_major_is_preferred() {
    if !gpu_present("nvrtc_of_the_build_major_is_preferred") {
        return;
    }
    let info = pbsolve_cuda::nvrtc_info().unwrap();
    eprintln!(
        "NVRTC {:?} from {} ({:?}), archs {:?}",
        info.version, info.name, info.path, info.archs
    );
    assert!(!info.archs.is_empty(), "NVRTC reports no architectures");
    if overridden() || !cfg!(target_os = "linux") {
        return;
    }
    assert!(info.path.as_ref().is_some_and(|p| p.contains("libnvrtc")));
    let major = pbsolve_cuda::cuda_major();
    let versioned = format!("libnvrtc.so.{major}");
    // SAFETY: only checks whether the loader can resolve the name.
    let versioned_exists = unsafe { libloading::Library::new(&versioned) }.is_ok();
    if versioned_exists {
        assert_eq!(info.name, versioned);
        assert_eq!(info.version.0 as u32, major, "{versioned} is not CUDA {major}");
    } else {
        assert_eq!(info.name, "libnvrtc.so");
        eprintln!("no {versioned} installed: fell back to NVRTC {:?}", info.version);
    }
}

#[test]
fn kernels_are_built_as_a_cubin_for_the_device() {
    if !gpu_present("kernels_are_built_as_a_cubin_for_the_device") {
        return;
    }
    let be = CudaBackend::with_code(0, CodeType::Auto).unwrap();
    let info = pbsolve_cuda::nvrtc_info().unwrap();
    // "Device (cc 7.5, device 0)" → 75
    let name = be.device_name();
    let cc = name
        .split("cc ")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .unwrap()
        .replace('.', "");
    let cc: i32 = cc.parse().unwrap();
    if info.archs.contains(&cc) {
        assert_eq!(be.code_target(), format!("CUBIN sm_{cc}"));
    } else {
        assert!(be.code_target().starts_with("PTX compute_"), "{}", be.code_target());
    }
}

#[test]
fn cuda_matches_cpu_for_both_preconditioners() {
    if !gpu_present("cuda_matches_cpu_for_both_preconditioners") {
        return;
    }
    let (op, b) = system();
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    for pc in [Preconditioner::Multigrid, Preconditioner::Jacobi] {
        let (uc, sc) = solve(&cpu, &op, &b, pc);
        let (ug, sg) = solve(&gpu, &op, &b, pc);
        let rel = relative_difference(&uc, &ug);
        eprintln!(
            "{pc:?}: CPU {} / CUDA {} iterations, |Δu|/|u| = {rel:.2e} ({})",
            sc.iterations,
            sg.iterations,
            gpu.code_target()
        );
        assert!(rel < 1e-10, "{pc:?}: {rel}");
        assert!(sc.iterations.abs_diff(sg.iterations) <= 1);
        assert_eq!(sc.levels, sg.levels);
    }
}

#[test]
fn cubin_and_ptx_kernels_give_identical_results() {
    if !gpu_present("cubin_and_ptx_kernels_give_identical_results") {
        return;
    }
    let cubin = CudaBackend::with_code(0, CodeType::Cubin).unwrap();
    let ptx = match CudaBackend::with_code(0, CodeType::Ptx) {
        Ok(be) => be,
        Err(e) => {
            // PTX from an NVRTC newer than the driver cannot be loaded; that
            // is exactly why CUBIN is the default.
            let nv = pbsolve_cuda::nvrtc_info().unwrap().version;
            let drv = pbsolve_cuda::driver_version().unwrap();
            assert!(nv > drv, "PTX failed although NVRTC {nv:?} <= driver {drv:?}: {e}");
            eprintln!("PTX path skipped: NVRTC {nv:?} is newer than the driver {drv:?} ({e})");
            return;
        }
    };
    assert!(cubin.code_target().starts_with("CUBIN"));
    assert!(ptx.code_target().starts_with("PTX"));
    let (op, b) = system();
    for pc in [Preconditioner::Multigrid, Preconditioner::Jacobi] {
        let (u1, s1) = solve(&cubin, &op, &b, pc);
        let (u2, s2) = solve(&ptx, &op, &b, pc);
        assert_eq!(s1.iterations, s2.iterations);
        assert!(
            u1.iter().zip(&u2).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{pc:?} differs"
        );
    }
}

#[test]
fn repeated_backends_reuse_the_compiled_image() {
    if !gpu_present("repeated_backends_reuse_the_compiled_image") {
        return;
    }
    let _first = CudaBackend::new(0).unwrap();
    let t = std::time::Instant::now();
    for _ in 0..3 {
        CudaBackend::new(0).unwrap();
    }
    let per_backend = t.elapsed().as_secs_f64() / 3.0;
    eprintln!("backend creation with a cached image: {:.1} ms", 1e3 * per_backend);
    // Compiling the kernels takes hundreds of milliseconds; a cache hit only
    // creates a context and loads the module.
    assert!(per_backend < 0.5, "{per_backend} s");
}
