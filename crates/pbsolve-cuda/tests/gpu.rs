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
    for pc in [
        Preconditioner::Multigrid,
        Preconditioner::MultigridF32,
        Preconditioner::Jacobi,
    ] {
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
    for pc in [
        Preconditioner::Multigrid,
        Preconditioner::MultigridF32,
        Preconditioner::Jacobi,
    ] {
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

#[test]
fn fused_dot2_equals_two_dots_in_both_precisions() {
    if !gpu_present("fused_dot2_equals_two_dots_in_both_precisions") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let a: Vec<f64> = (0..300_001).map(|i| (i as f64 * 0.37).sin()).collect();
    let b: Vec<f64> = (0..300_001).map(|i| (i as f64 * 0.11).cos()).collect();
    let c: Vec<f64> = (0..300_001).map(|i| 1.0 / (1.0 + i as f64)).collect();
    fn check<B: Backend>(be: &B, a: &[f64], b: &[f64], c: &[f64]) {
        let (a, b, c) = (be.upload(a).unwrap(), be.upload(b).unwrap(), be.upload(c).unwrap());
        let (ab, ac) = be.dot2(&a, &b, &c).unwrap();
        assert_eq!(ab.to_bits(), be.dot(&a, &b).unwrap().to_bits());
        assert_eq!(ac.to_bits(), be.dot(&a, &c).unwrap().to_bits());
    }
    check(&gpu, &a, &b, &c);
    check(&gpu.single().unwrap(), &a, &b, &c);
}

#[test]
fn single_precision_kernels_match_the_cpu_bit_for_bit() {
    if !gpu_present("single_precision_kernels_match_the_cpu_bit_for_bit") {
        return;
    }
    // f32 kernels round identically on both platforms (same expression order,
    // no FMA contraction); only reductions may differ.
    let (op, b) = system();
    let gpu = CudaBackend::new(0).unwrap().single().unwrap();
    let cpu = pbsolve_core::CpuSingle::new();
    let (og, oc) = (gpu.prepare(op.clone()).unwrap(), cpu.prepare(op.clone()).unwrap());
    let (xg, xc) = (gpu.upload(&b).unwrap(), cpu.upload(&b).unwrap());
    let (mut yg, mut yc) = (gpu.zeros(b.len()).unwrap(), cpu.zeros(b.len()).unwrap());
    gpu.apply(&og, &xg, &mut yg).unwrap();
    cpu.apply(&oc, &xc, &mut yc).unwrap();
    assert_eq!(gpu.download(&yg).unwrap(), cpu.download(&yc).unwrap());
    let (mut sg, mut sc) = (gpu.zeros(b.len()).unwrap(), cpu.zeros(b.len()).unwrap());
    gpu.scale(0.3, &xg, &yg, &mut sg).unwrap();
    cpu.scale(0.3, &xc, &yc, &mut sc).unwrap();
    assert_eq!(gpu.download(&sg).unwrap(), cpu.download(&sc).unwrap());
    let t = pbsolve_core::multigrid::Transfer::new(op.dims).unwrap();
    let nc: usize = t.coarse.iter().product();
    let (mut cg, mut cc) = (gpu.zeros(nc).unwrap(), cpu.zeros(nc).unwrap());
    gpu.restrict(&t, &yg, &mut cg).unwrap();
    cpu.restrict(&t, &yc, &mut cc).unwrap();
    assert_eq!(gpu.download(&cg).unwrap(), cpu.download(&cc).unwrap());
    gpu.prolong_add(&t, &cg, &mut yg).unwrap();
    cpu.prolong_add(&t, &cc, &mut yc).unwrap();
    gpu.axpby(0.7, &xg, -1.3, &mut yg).unwrap();
    cpu.axpby(0.7, &xc, -1.3, &mut yc).unwrap();
    assert_eq!(gpu.download(&yg).unwrap(), cpu.download(&yc).unwrap());
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// An LPBE operator with a few atoms on a grid of the given node counts.
fn operator(dims: [usize; 3], spacing: f64, salt: f64) -> Arc<StencilOperator> {
    let len = dims.map(|n| (n - 1) as f64 * spacing);
    let grid = Grid::centered([0.3, -0.2, 0.1], len, dims).unwrap();
    let atoms: Vec<Atom> = [
        ([0.0, 0.0, 0.0], 1.0, 1.9),
        ([1.6, 0.4, -0.3], -0.6, 1.5),
        ([-1.2, 1.1, 0.8], 0.3, 1.7),
        ([0.4, -1.5, 1.2], -0.7, 1.2),
        ([-0.5, -0.4, -1.6], 0.0, 0.0),
    ]
    .into_iter()
    .map(|(position, charge, radius)| Atom {
        position,
        charge,
        radius,
    })
    .collect();
    let params = PbParams {
        pdie: 2.0,
        ..PbParams::default()
    }
    .with_salt(salt, 2.0);
    Arc::new(StencilOperator::new(
        &grid,
        CoefficientMaps::solvated(&grid, &atoms, &params),
    ))
}

/// The multigrid hierarchy built on the GPU (Galerkin products,
/// symmetrisation, inverse diagonals, Gershgorin bounds) equals the host
/// hierarchy bit for bit, and so do its f32 copies.
#[test]
fn device_multigrid_setup_equals_the_host_bit_for_bit() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, Levels};
    if !gpu_present("device_multigrid_setup_equals_the_host_bit_for_bit") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let single = gpu.single().unwrap();
    let cases = [
        (system().0, "random atoms, 49³"),
        (operator([66, 41, 6], 0.35, 0.15), "semi-coarsening"),
        (operator([50, 37, 29], 0.3, 0.0), "even and odd sizes, no salt"),
        (operator([97, 97, 97], 0.25, 1.0), "97³, 1 M"),
    ];
    for (op, what) in cases {
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let dev = gpu.prepare(op.clone()).unwrap();
        let l = Levels::build(&gpu, &op, &dev, Chebyshev::default()).unwrap().unwrap();
        assert_eq!(l.levels(), h.levels(), "{what}");
        assert_eq!(l.transfers, h.transfers, "{what}");
        assert_eq!(bits(&l.lmax), bits(&h.lmax), "{what}: Gershgorin bounds");
        assert_eq!(l.coarsest, h.coarsest, "{what}: coarsest factor");
        for (lev, s) in h.coarse.iter().enumerate() {
            let c = gpu.coarse_to_host(&l.coarse[lev]).unwrap();
            assert_eq!(bits(&c.coef), bits(&s.coef), "{what}: level {} operator", lev + 1);
            assert_eq!(
                bits(&gpu.download(&l.inv_diag[lev]).unwrap()),
                bits(&s.inv_diag),
                "{what}: level {} 1/diag",
                lev + 1
            );
        }
        // f32 copies: rounded on the device as the twin's upload rounds them.
        let up = Levels::upload(&single, h).unwrap();
        let down = l.demote(&gpu, &single).unwrap();
        for (lev, (a, b)) in up.coarse.iter().zip(&down.coarse).enumerate() {
            let (a, b) = (single.coarse_to_host(a).unwrap(), single.coarse_to_host(b).unwrap());
            assert_eq!(bits(&a.coef), bits(&b.coef), "{what}: f32 level {}", lev + 1);
        }
        for (a, b) in up.inv_diag.iter().zip(&down.inv_diag) {
            assert_eq!(bits(&single.download(a).unwrap()), bits(&single.download(b).unwrap()));
        }
        let (o32, d32) = (single.prepare(op.clone()).unwrap(), gpu.demote_operator(&dev).unwrap());
        let x = single.upload(&system_like(op.len())).unwrap();
        let (mut y1, mut y2) = (single.zeros(x.len()).unwrap(), single.zeros(x.len()).unwrap());
        single.apply(&o32, &x, &mut y1).unwrap();
        single.apply(&d32, &x, &mut y2).unwrap();
        assert_eq!(
            single.download(&y1).unwrap(),
            single.download(&y2).unwrap(),
            "{what}: f32 operator"
        );
    }
}

/// Deterministic test vector.
fn system_like(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0 - 0.5).collect()
}

/// A V-cycle built from device levels is the V-cycle of the host hierarchy
/// (both precisions), so solver results do not change.
#[test]
fn vcycle_of_device_levels_equals_the_host_vcycle() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, Levels, MultigridPreconditioner};
    if !gpu_present("vcycle_of_device_levels_equals_the_host_vcycle") {
        return;
    }
    let (op, b) = system();
    let gpu = CudaBackend::new(0).unwrap();
    let dev = gpu.prepare(op.clone()).unwrap();
    let r = gpu.upload(&b).unwrap();
    let mut host =
        MultigridPreconditioner::new(&gpu, &op, Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap()).unwrap();
    let levels = Levels::build(&gpu, &op, &dev, Chebyshev::default()).unwrap().unwrap();
    let mut device = MultigridPreconditioner::from_levels(&gpu, &op, levels).unwrap();
    let (mut z1, mut z2) = (gpu.zeros(b.len()).unwrap(), gpu.zeros(b.len()).unwrap());
    host.apply(&gpu, &dev, &r, &mut z1).unwrap();
    device.apply(&gpu, &dev, &r, &mut z2).unwrap();
    assert_eq!(bits(&gpu.download(&z1).unwrap()), bits(&gpu.download(&z2).unwrap()));

    let single = gpu.single().unwrap();
    let op32 = single.prepare(op.clone()).unwrap();
    let r32 = single.upload(&b).unwrap();
    let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
    let mut host = MultigridPreconditioner::new(&single, &op, h).unwrap();
    let levels = Levels::build(&gpu, &op, &dev, Chebyshev::default()).unwrap().unwrap();
    let mut device = MultigridPreconditioner::from_levels(&single, &op, levels.demote(&gpu, &single).unwrap()).unwrap();
    let (mut z1, mut z2) = (single.zeros(b.len()).unwrap(), single.zeros(b.len()).unwrap());
    host.apply(&single, &op32, &r32, &mut z1).unwrap();
    device
        .apply(&single, &gpu.demote_operator(&dev).unwrap(), &r32, &mut z2)
        .unwrap();
    assert_eq!(
        bits(&single.download(&z1).unwrap()),
        bits(&single.download(&z2).unwrap())
    );
}

#[test]
fn device_setup_reports_a_negative_diagonal_like_the_host() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, LevelOp, Levels, Transfer};
    if !gpu_present("device_setup_reports_a_negative_diagonal_like_the_host") {
        return;
    }
    let mut op = (*operator([33, 33, 33], 0.4, 0.15)).clone();
    for c in [16 + 33 * (16 + 33 * 16), 17 + 33 * (16 + 33 * 16)] {
        op.kvol[c] = -1e6;
    }
    let op = Arc::new(op);
    let host = Hierarchy::build(&op, Chebyshev::default()).unwrap_err().to_string();
    let gpu = CudaBackend::new(0).unwrap();
    let dev = gpu.prepare(op.clone()).unwrap();
    let device = Levels::build(&gpu, &op, &dev, Chebyshev::default())
        .err()
        .unwrap()
        .to_string();
    assert!(host.contains("non-positive Galerkin diagonal"), "{host}");
    assert_eq!(host, device);
    // The f32 twin does not build hierarchies.
    let single = gpu.single().unwrap();
    let d32 = single.prepare(op.clone()).unwrap();
    let t = Transfer::new(op.dims).unwrap();
    assert!(single.galerkin(&t, LevelOp::Fine(&d32)).is_err());
    assert!(single.gershgorin_lmax(LevelOp::Fine(&d32)).is_err());
}
