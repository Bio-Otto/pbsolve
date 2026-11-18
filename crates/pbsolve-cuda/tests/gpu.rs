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
use pbsolve_core::{Atom, Backend, ChargeMethod, CoefficientMaps, CpuBackend, Grid, PbParams, Preconditioner};
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
    let t = pbsolve_core::multigrid::Transfer::new(op.dims, op.periodic).unwrap();
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
        (operator([9, 11, 8], 0.35, 0.15), "too small to coarsen"),
    ];
    for (op, what) in cases {
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let dev = gpu.prepare(op.clone()).unwrap();
        let l = Levels::build(&gpu, op.dims, op.periodic, &dev, Chebyshev::default())
            .unwrap()
            .unwrap();
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
    let levels = Levels::build(&gpu, op.dims, op.periodic, &dev, Chebyshev::default())
        .unwrap()
        .unwrap();
    let mut device = MultigridPreconditioner::from_levels(&gpu, gpu.upload(&op.inv_diag).unwrap(), levels).unwrap();
    let (mut z1, mut z2) = (gpu.zeros(b.len()).unwrap(), gpu.zeros(b.len()).unwrap());
    host.apply(&gpu, &dev, &r, &mut z1).unwrap();
    device.apply(&gpu, &dev, &r, &mut z2).unwrap();
    assert_eq!(bits(&gpu.download(&z1).unwrap()), bits(&gpu.download(&z2).unwrap()));

    let single = gpu.single().unwrap();
    let op32 = single.prepare(op.clone()).unwrap();
    let r32 = single.upload(&b).unwrap();
    let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
    let mut host = MultigridPreconditioner::new(&single, &op, h).unwrap();
    let levels = Levels::build(&gpu, op.dims, op.periodic, &dev, Chebyshev::default())
        .unwrap()
        .unwrap();
    let inv32 = single.upload(&op.inv_diag).unwrap();
    let mut device =
        MultigridPreconditioner::from_levels(&single, inv32, levels.demote(&gpu, &single).unwrap()).unwrap();
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
    let device = Levels::build(&gpu, op.dims, op.periodic, &dev, Chebyshev::default())
        .err()
        .unwrap()
        .to_string();
    assert!(host.contains("non-positive Galerkin diagonal"), "{host}");
    assert_eq!(host, device);
    // The f32 twin does not build hierarchies.
    let single = gpu.single().unwrap();
    let d32 = single.prepare(op.clone()).unwrap();
    let t = Transfer::new(op.dims, op.periodic).unwrap();
    assert!(single.galerkin(&t, LevelOp::Fine(&d32)).is_err());
    assert!(single.gershgorin_lmax(LevelOp::Fine(&d32)).is_err());
}

/// Pseudo-random numbers in [0, 1).
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sphere marking on the GPU equals `mark_map` bit for bit: spheres inside,
/// partly and fully outside the grid, of zero radius, and nodes lying exactly
/// on a sphere (which count as inside); several passes and shifted grids.
#[test]
fn device_sphere_marking_equals_the_host_bit_for_bit() {
    use pbsolve_core::surface::{mark_map, Sphere};
    if !gpu_present("device_sphere_marking_equals_the_host_bit_for_bit") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let grid = Grid::centered([0.4, -0.3, 0.2], [14.0, 11.0, 9.5], [57, 45, 39]).unwrap();
    let mut rnd = rng(17);
    let random = |rnd: &mut dyn FnMut() -> f64, n: usize, rmax: f64| -> Vec<Sphere> {
        (0..n)
            .map(|_| Sphere {
                center: [rnd(), rnd(), rnd()].map(|v| 20.0 * (v - 0.5)),
                radius: rmax * rnd(),
            })
            .collect()
    };
    let mut solute = random(&mut rnd, 300, 4.0);
    // Exactly on nodes, radius a multiple of the spacing: boundary nodes are inside.
    let h = grid.spacing;
    for (i, j, k, m) in [(10, 12, 8, 4.0), (30, 20, 19, 3.0), (0, 0, 0, 2.0), (56, 44, 38, 5.0)] {
        let p = grid.position(i, j, k);
        solute.push(Sphere {
            center: p,
            radius: m * h[0],
        });
    }
    solute.push(Sphere {
        center: [0.0; 3],
        radius: 0.0,
    });
    solute.push(Sphere {
        center: [100.0, 0.0, 0.0],
        radius: 3.0,
    });
    let probes = random(&mut rnd, 2000, 1.4);
    let shifts = [
        [0.0; 3],
        [0.5 * h[0], 0.0, 0.0],
        [0.0, 0.5 * h[1], 0.0],
        [0.0, 0.0, 0.5 * h[2]],
    ];
    for passes in [
        vec![(&solute[..], 2.0), (&probes[..], 78.54)],
        vec![(&probes[..], 0.0)],
        vec![],
    ] {
        let maps = gpu.mark_maps(&grid, &shifts, 78.54, &passes).unwrap();
        assert_eq!(maps.len(), shifts.len());
        for (shift, map) in shifts.iter().zip(&maps) {
            let host = mark_map(&grid, *shift, 78.54, &passes);
            let device = gpu.download(map).unwrap();
            assert_eq!(bits(&device), bits(&host), "{} passes, shift {shift:?}", passes.len());
        }
    }
}

/// Atoms of a small protein-like cluster for the assembly tests.
fn cluster(n: usize, seed: u64) -> Vec<Atom> {
    let mut rnd = rng(seed);
    (0..n)
        .map(|i| Atom {
            position: [rnd(), rnd(), rnd()].map(|v| 14.0 * (v - 0.5)),
            charge: rnd() - 0.5,
            // Some hydrogens without radius (they still bury SAS points).
            radius: if i % 7 == 0 { 0.0 } else { 1.1 + 0.8 * rnd() },
        })
        .collect()
}

/// The maps of every surface and medium, built on the GPU, equal the host maps.
#[test]
fn device_coefficient_maps_equal_the_host_maps() {
    use pbsolve_core::surface::maps_on;
    use pbsolve_core::{MolecularSurface, SurfaceMethod};
    if !gpu_present("device_coefficient_maps_equal_the_host_maps") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let atoms = cluster(120, 3);
    let grid = Grid::centered([0.2, 0.1, -0.3], [26.0, 24.0, 22.0], [65, 57, 49]).unwrap();
    let base = PbParams {
        pdie: 2.0,
        ..PbParams::default()
    };
    let cases = [
        ("mol, 0.15 M", base.clone().with_salt(0.15, 2.0)),
        ("mol, no salt", base.clone()),
        (
            "vdw",
            PbParams {
                surface: SurfaceMethod::Vdw,
                ..base.clone().with_salt(0.15, 2.0)
            },
        ),
        (
            "harmonic",
            PbParams {
                surface: SurfaceMethod::HarmonicMol,
                ..base.clone().with_salt(0.5, 1.5)
            },
        ),
        (
            "mol, fractional interface",
            PbParams {
                interface: pbsolve_core::InterfaceMethod::Fractional,
                ..base.clone().with_salt(0.15, 2.0)
            },
        ),
    ];
    for (what, params) in cases {
        let surf = MolecularSurface::for_params(&atoms, &params);
        for (state, s) in [("solvated", Some(&surf)), ("reference", None)] {
            let host = maps_on(&CpuBackend::new(), &grid, &atoms, &params, s).unwrap();
            let device = maps_on(&gpu, &grid, &atoms, &params, s).unwrap();
            for (m, (a, b)) in host.iter().zip(&device).enumerate() {
                assert_eq!(bits(a), bits(&gpu.download(b).unwrap()), "{what}, {state}: map {m}");
            }
        }
    }
}

/// Operator, inverse diagonal, charges and right-hand side assembled on the
/// GPU equal the host's, and a solve from GPU-assembled maps equals the solve
/// of the host-assembled system (both on the GPU) bit for bit.
#[test]
fn device_assembly_equals_the_host_assembly_bit_for_bit() {
    use pbsolve_core::charge::spread_charges_sparse;
    use pbsolve_core::solve_with_maps;
    use pbsolve_core::surface::maps_on;
    use pbsolve_core::MolecularSurface;
    use std::f64::consts::PI;
    if !gpu_present("device_assembly_equals_the_host_assembly_bit_for_bit") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let atoms = cluster(150, 9);
    // Anisotropic spacing: three different face coefficients.
    let grid = Grid::centered([0.0, 0.3, -0.1], [27.0, 25.0, 23.0], [57, 49, 41]).unwrap();
    let params = PbParams {
        pdie: 4.0,
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0);
    let surf = MolecularSurface::for_params(&atoms, &params);
    let maps = CoefficientMaps::solvated_with(&grid, &atoms, &params, &surf);
    let host = StencilOperator::new(&grid, maps.clone());

    let (dev, inv) = gpu
        .assemble_operator(&grid, maps_on(&gpu, &grid, &atoms, &params, Some(&surf)).unwrap())
        .unwrap();
    let back = gpu.operator_to_host(&dev).unwrap();
    assert_eq!(bits(&host.coef), bits(&back.coef));
    for (a, b, what) in [
        (&host.eps_x, &back.eps_x, "eps_x"),
        (&host.eps_y, &back.eps_y, "eps_y"),
        (&host.eps_z, &back.eps_z, "eps_z"),
        (&host.kvol, &back.kvol, "kvol"),
        (&host.inv_diag, &back.inv_diag, "1/diag (host recomputation)"),
    ] {
        assert_eq!(bits(a), bits(b), "{what}");
    }
    assert_eq!(bits(&host.inv_diag), bits(&gpu.download(&inv).unwrap()), "1/diag");

    // Charges (sparse upload) and boundary values (scattered).
    let q = spread_charges(&grid, &atoms).unwrap();
    let (nodes, values) = spread_charges_sparse(&grid, &atoms).unwrap();
    let qd = gpu.scatter(grid.len(), &nodes, &values).unwrap();
    assert_eq!(bits(&q), bits(&gpu.download(&qd).unwrap()));
    let mut rnd = rng(23);
    let mut g = vec![0.0; grid.len()];
    let boundary = grid.boundary_indices();
    for &c in &boundary {
        g[c] = rnd() - 0.5;
    }
    let gb: Vec<f64> = boundary.iter().map(|&c| g[c]).collect();
    let gd = gpu.scatter(grid.len(), &boundary, &gb).unwrap();
    assert_eq!(bits(&g), bits(&gpu.download(&gd).unwrap()));

    let lb0 = bjerrum_length_vacuum(params.temperature);
    let b_host = assemble_rhs(&host, &q, &g, lb0);
    let mut bd = gpu.zeros(grid.len()).unwrap();
    gpu.apply(&dev, &gd, &mut bd).unwrap();
    gpu.source_rhs(4.0 * PI * lb0, &qd, &inv, &mut bd).unwrap();
    assert_eq!(bits(&b_host), bits(&gpu.download(&bd).unwrap()), "right-hand side");

    // Whole solves: host-assembled system uploaded vs maps assembled on the GPU.
    for pc in [
        Preconditioner::MultigridF32,
        Preconditioner::Multigrid,
        Preconditioner::Jacobi,
    ] {
        let opts = SolverOptions {
            preconditioner: pc,
            ..SolverOptions::new(1e-9, 10_000)
        };
        let (x, s1) = pcg(&gpu, Arc::new(host.clone()), &b_host, opts).unwrap();
        let u_host: Vec<f64> = x.iter().zip(&g).map(|(x, g)| x + g).collect();
        let (u_dev, s2) = solve_with_maps(&gpu, &grid, maps.clone(), &q, &g, params.temperature, opts).unwrap();
        assert_eq!(s1.iterations, s2.iterations, "{pc:?}");
        assert_eq!(s1.relative_residual.to_bits(), s2.relative_residual.to_bits(), "{pc:?}");
        assert_eq!(bits(&u_host), bits(&u_dev), "{pc:?}: potential");
    }
}

/// Assembly checks its arguments and runs on f64 backends only.
#[test]
fn device_assembly_rejects_bad_arguments() {
    use pbsolve_core::surface::Sphere;
    if !gpu_present("device_assembly_rejects_bad_arguments") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    for e in [
        gpu.scatter(10, &[3, 10], &[1.0, 2.0]).err().unwrap(),
        cpu.scatter(10, &[3, 10], &[1.0, 2.0]).err().unwrap(),
    ] {
        assert!(e.to_string().contains("index 10 outside a vector of 10"), "{e}");
    }
    for e in [
        gpu.scatter(10, &[3], &[1.0, 2.0]).err().unwrap(),
        cpu.scatter(10, &[3], &[1.0, 2.0]).err().unwrap(),
    ] {
        assert!(e.to_string().contains("1 indices but 2 values"), "{e}");
    }
    let grid = Grid::centered([0.0; 3], [8.0; 3], [9; 3]).unwrap();
    let ball = [Sphere {
        center: [0.0; 3],
        radius: 2.0,
    }];
    let single = gpu.single().unwrap();
    let e = single
        .mark_maps(&grid, &[[0.0; 3]], 1.0, &[(&ball[..], 2.0)])
        .err()
        .unwrap();
    assert!(e.to_string().contains("f64"), "{e}");
    assert!(single.scatter(10, &[1], &[1.0]).is_err());
    assert!(pbsolve_core::CpuSingle::new().scatter(10, &[1], &[1.0]).is_err());
    let maps = [0, 1, 2, 3].map(|_| single.zeros(grid.len()).unwrap());
    assert!(single.assemble_operator(&grid, maps).is_err());
    // Maps of the wrong length are refused, not read out of bounds.
    let short = [0, 1, 2, 3].map(|_| gpu.zeros(grid.len() - 1).unwrap());
    assert!(gpu.assemble_operator(&grid, short).is_err());
}

/// Spheres built so that `mark_map`'s exact arithmetic decides: nodes lying
/// exactly on a sphere (dyadic grid: every distance is exact) and nodes whose
/// squared distance rounds to either side of r² depending on the summation
/// order ((dz² + dy²) + dx² is the host's). The GPU must agree on all of them.
#[test]
fn device_sphere_marking_follows_the_host_arithmetic_at_the_boundary() {
    use pbsolve_core::surface::{mark_map, Sphere};
    if !gpu_present("device_sphere_marking_follows_the_host_arithmetic_at_the_boundary") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    // h = 0.5 and origin -10: node coordinates and their differences are exact.
    let grid = Grid::from_spacing([0.0; 3], [0.5; 3], [41, 41, 41]).unwrap();
    let mut on_boundary = Vec::new();
    for (i, j, k, r) in [(10, 12, 8, 2.0), (30, 20, 19, 1.5), (20, 20, 20, 3.0), (5, 33, 27, 2.5)] {
        on_boundary.push(Sphere {
            center: grid.position(i, j, k),
            radius: r,
        });
    }
    // Order-sensitive nodes: search sphere centres (off the dyadic lattice)
    // and radii for which the two summation orders disagree at a target node.
    let min = grid.origin;
    let h = grid.spacing;
    let mut rnd = rng(31);
    let mut sensitive = Vec::new();
    while sensitive.len() < 200 {
        let (i, j, k) = (
            10 + (rnd() * 20.0) as usize,
            10 + (rnd() * 20.0) as usize,
            10 + (rnd() * 20.0) as usize,
        );
        let p = grid.position(i, j, k);
        let center = [0, 1, 2].map(|a| p[a] + 3.0 * (rnd() - 0.5));
        let pos = [0, 1, 2].map(|a| center[a] - min[a]);
        let dx = pos[0] - h[0] * i as f64;
        let dy = pos[1] - h[1] * j as f64;
        let dz = pos[2] - h[2] * k as f64;
        let (dx2, dy2, dz2) = (dx * dx, dy * dy, dz * dz);
        let host_order = dz2 + dy2 + dx2;
        let other = dx2 + dy2 + dz2;
        if host_order == other {
            continue;
        }
        // A radius whose square equals the smaller sum: inside for one order only.
        let target = host_order.min(other);
        let mut r = target.sqrt();
        for _ in 0..8 {
            if r * r == target {
                break;
            }
            r = if r * r < target { r.next_up() } else { r.next_down() };
        }
        if r * r == target && dx2 + dy2 <= r * r {
            sensitive.push(Sphere { center, radius: r });
        }
    }
    let passes = [(&on_boundary[..], 1.0), (&sensitive[..], 2.0)];
    let host = mark_map(&grid, [0.0; 3], 0.0, &passes);
    let device = gpu.mark_maps(&grid, &[[0.0; 3]], 0.0, &passes).unwrap();
    // The test is only meaningful if the boundary cases were marked by the host.
    let marked = |v: f64| host.iter().filter(|&&x| x == v).count();
    assert!(marked(1.0) > 0 && marked(2.0) > 0);
    assert_eq!(bits(&gpu.download(&device[0]).unwrap()), bits(&host));
}

/// The operator of arbitrary maps (random ε and κ̄², so every rounding of
/// the diagonal shows) assembled on the GPU equals `StencilOperator::new`.
#[test]
fn device_operator_of_random_maps_equals_the_host() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, LevelMatrix, LevelOp, Levels};
    if !gpu_present("device_operator_of_random_maps_equals_the_host") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    // Random coefficients everywhere, the seams of periodic axes included.
    for periodic in [
        [false; 3],
        [true, false, false],
        [false, true, false],
        [true, true, false],
    ] {
        let grid = Grid::centered([0.1, 0.0, -0.2], [13.0, 17.0, 11.0], [33, 41, 29])
            .unwrap()
            .with_periodic(periodic);
        let mut rnd = rng(41);
        let mut map = |lo: f64, hi: f64| -> Vec<f64> { (0..grid.len()).map(|_| lo + (hi - lo) * rnd()).collect() };
        let mut maps = CoefficientMaps {
            eps_x: map(1.0, 80.0),
            eps_y: map(1.0, 80.0),
            eps_z: map(1.0, 80.0),
            kappa_sq: map(0.0, 0.3),
        };
        // Strong ions away from the x = 0 and y = 0 planes, so the rows of the
        // seams decide the Gershgorin bound (max of (2S + κ̄²V)/(S + κ̄²V)).
        for (c, k) in maps.kappa_sq.iter_mut().enumerate() {
            let (i, j, _) = grid.ijk(c);
            if i != 0 && j != 0 {
                *k += 30.0;
            }
        }
        let host = StencilOperator::new(&grid, maps.clone());
        let up = [&maps.eps_x, &maps.eps_y, &maps.eps_z, &maps.kappa_sq].map(|m| gpu.upload(m).unwrap());
        let (dev, inv) = gpu.assemble_operator(&grid, up).unwrap();
        let back = gpu.operator_to_host(&dev).unwrap();
        assert_eq!(bits(&host.kvol), bits(&back.kvol), "{periodic:?}");
        assert_eq!(bits(&host.inv_diag), bits(&gpu.download(&inv).unwrap()), "{periodic:?}");
        let x = gpu.upload(&system_like(grid.len())).unwrap();
        let mut y = gpu.zeros(grid.len()).unwrap();
        gpu.apply(&dev, &x, &mut y).unwrap();
        let mut yh = vec![0.0; grid.len()];
        host.apply(&system_like(grid.len()), &mut yh);
        assert_eq!(bits(&yh), bits(&gpu.download(&y).unwrap()), "{periodic:?}");
        // Gershgorin bound and Galerkin hierarchy of the random operator.
        assert_eq!(
            gpu.gershgorin_lmax(LevelOp::Fine(&dev)).unwrap().to_bits(),
            LevelMatrix::Seven(&host).gershgorin_lmax().to_bits(),
            "{periodic:?}"
        );
        let h = Hierarchy::build(&host, Chebyshev::default()).unwrap().unwrap();
        let l = Levels::build(&gpu, grid.dims, periodic, &dev, Chebyshev::default())
            .unwrap()
            .unwrap();
        assert_eq!(bits(&l.lmax), bits(&h.lmax), "{periodic:?}");
        for (lev, c) in h.coarse.iter().enumerate() {
            let d = gpu.coarse_to_host(&l.coarse[lev]).unwrap();
            assert_eq!(bits(&d.coef), bits(&c.coef), "{periodic:?} level {}", lev + 1);
        }
    }
}

/// `gather` reads the requested values of a device vector exactly, in the
/// requested order (repetitions, any order, none), like the CPU backend, and
/// refuses indices outside the vector.
#[test]
fn device_gather_reads_the_requested_values() {
    if !gpu_present("device_gather_reads_the_requested_values") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    let mut rnd = rng(31);
    let n = 100_003;
    let mut host: Vec<f64> = (0..n).map(|_| 1e3 * (rnd() - 0.5)).collect();
    host[7] = -0.0;
    host[8] = f64::MIN_POSITIVE / 4.0; // subnormal
    let v = gpu.upload(&host).unwrap();
    let mut idx: Vec<usize> = (0..40_000).map(|_| (rnd() * n as f64) as usize).collect();
    idx.extend([0, n - 1, 7, 8, 7, n - 1]);
    let want: Vec<u64> = idx.iter().map(|&c| host[c].to_bits()).collect();
    assert_eq!(bits(&gpu.gather(&v, &idx).unwrap()), want);
    assert_eq!(bits(&cpu.gather(&host, &idx).unwrap()), want);
    assert!(gpu.gather(&v, &[]).unwrap().is_empty());
    for e in [
        gpu.gather(&v, &[3, n]).err().unwrap(),
        cpu.gather(&host, &[3, n]).err().unwrap(),
    ] {
        assert!(
            e.to_string().contains(&format!("index {n} outside a vector of {n}")),
            "{e}"
        );
    }
    let single = gpu.single().unwrap();
    let v32 = single.zeros(10).unwrap();
    assert!(single.gather(&v32, &[1]).unwrap_err().to_string().contains("f64"));
}

/// On the GPU, energies taken from u at the charged nodes, with the focusing
/// boundaries read at the stencil nodes only, equal the energies of the full
/// potentials bit for bit (one to three focusing levels, a focusing level
/// that reads coarse boundary nodes, both multigrid precisions).
#[test]
fn device_energies_from_the_charged_nodes_equal_the_full_potential_energies() {
    use pbsolve_core::pb::{electrostatic_energy_kt, polar_solvation, solve_focused, Medium};
    use pbsolve_core::GridSpec;
    if !gpu_present("device_energies_from_the_charged_nodes_equal_the_full_potential_energies") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    // polar_solvation builds the solvated state's inputs in the background.
    assert!(gpu.runs_on_device());
    let atoms = cluster(150, 23);
    let fine = [24.0, 22.0, 26.0];
    let spec = |coarse: [f64; 3]| GridSpec {
        center: [0.1, -0.2, 0.3],
        dims: [49, 45, 53],
        coarse_length: coarse,
        fine_length: fine,
    };
    let params = PbParams {
        pdie: 2.0,
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0);
    let cases = [
        (spec(fine), params.clone()),
        (spec(fine.map(|l| 2.0 * l)), params.clone()),
        (spec([2.0 * fine[0], fine[1], 1.5 * fine[2]]), params.clone()),
        (
            spec(fine.map(|l| 9.0 * l)),
            PbParams {
                preconditioner: Preconditioner::Multigrid,
                ..params.clone()
            },
        ),
        (
            spec(fine.map(|l| 2.0 * l)),
            PbParams {
                chgm: ChargeMethod::Spline2,
                ..params.clone()
            },
        ),
        (
            spec(fine.map(|l| 2.0 * l)),
            PbParams {
                chgm: ChargeMethod::Spline4,
                ..params.clone()
            },
        ),
    ];
    let mut levels = Vec::new();
    for (spec, params) in cases {
        let r = polar_solvation(&gpu, &atoms, &params, &spec).unwrap();
        if params.chgm != ChargeMethod::Trilinear {
            // The host spreads the charges for both backends; only the solves differ.
            let cpu = polar_solvation(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
            let rel = (r.energy_kt - cpu.energy_kt).abs() / cpu.energy_kt.abs();
            assert!(rel < 1e-12, "{:?}: CPU vs CUDA {rel:e}", params.chgm);
        }
        for (medium, kt) in [(Medium::Solvated, r.solvated_kt), (Medium::Reference, r.reference_kt)] {
            let field = solve_focused(&gpu, &spec, &atoms, &params, medium).unwrap();
            let full = electrostatic_energy_kt(&field, &atoms, params.chgm).unwrap();
            assert_eq!(
                kt.to_bits(),
                full.to_bits(),
                "{:?} {medium:?}: {kt} vs {full}",
                spec.coarse_length
            );
        }
        levels.push(r.stats_solvated.len());
    }
    assert_eq!(levels, [1, 2, 2, 3, 2, 2]);
}

/// Each fused smoother step on the GPU (one kernel) equals the same step run
/// operation by operation on the GPU and on the CPU, in both precisions, on
/// the fine 7-point and a coarse 27-point operator.
#[test]
fn fused_smoother_steps_on_the_gpu_equal_their_sequences() {
    use pbsolve_core::backend::{chebyshev_first_by_steps, chebyshev_next_by_steps, residual_by_steps};
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, LevelOp};
    if !gpu_present("fused_smoother_steps_on_the_gpu_equal_their_sequences") {
        return;
    }
    let (op, _) = system();
    let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
    let coarse = h.coarse[0].clone();
    let gpu = CudaBackend::new(0).unwrap();

    /// Every step on `be`, fused and by steps: the bits of every output, in
    /// (fused, by steps) pairs.
    fn run<B: Backend>(
        be: &B,
        op: &Arc<StencilOperator>,
        coarse: &Arc<pbsolve_core::multigrid::Stencil27>,
    ) -> Vec<Vec<u64>> {
        let fine = be.prepare(op.clone()).unwrap();
        let co = be.prepare_coarse(coarse.clone()).unwrap();
        let vec = |len: usize, seed: usize| be.upload(&system_like(len + 31 * seed)[31 * seed..]).unwrap();
        let down = |v: &B::Vector| bits(&be.download(v).unwrap());
        let mut out = Vec::new();
        for (level, len) in [
            (LevelOp::<B>::Fine(&fine), op.len()),
            (LevelOp::Coarse(&co), coarse.len()),
        ] {
            let (x, b) = (vec(len, 1), vec(len, 2));
            for from_b in [true, false] {
                let src = from_b.then_some(&b);
                let (mut t, mut r) = (vec(len, 3), vec(len, 4));
                be.residual(level, &x, src, &mut t, &mut r).unwrap();
                out.push(down(&r));
                let (mut t, mut r) = (vec(len, 3), vec(len, 4));
                residual_by_steps(be, level, &x, src, &mut t, &mut r).unwrap();
                out.push(down(&r));
            }
            let (w, r) = (vec(len, 5), vec(len, 6));
            for overwrite in [true, false] {
                let (mut d1, mut x1) = (vec(len, 7), vec(len, 8));
                be.chebyshev_first(0.37, &w, &r, &mut d1, &mut x1, overwrite).unwrap();
                let (mut d2, mut x2) = (vec(len, 7), vec(len, 8));
                chebyshev_first_by_steps(be, 0.37, &w, &r, &mut d2, &mut x2, overwrite).unwrap();
                out.extend([down(&d1), down(&d2), down(&x1), down(&x2)]);
            }
            let (mut t1, mut d1, mut x1) = (vec(len, 9), vec(len, 10), vec(len, 11));
            be.chebyshev_next(0.81, &w, &r, -0.29, &mut t1, &mut d1, &mut x1)
                .unwrap();
            let (mut t2, mut d2, mut x2) = (vec(len, 9), vec(len, 10), vec(len, 11));
            chebyshev_next_by_steps(be, 0.81, &w, &r, -0.29, &mut t2, &mut d2, &mut x2).unwrap();
            out.extend([down(&d1), down(&d2), down(&x1), down(&x2)]);
        }
        out
    }

    let cpu = CpuBackend::new();
    for (name, on_gpu, on_cpu) in [
        ("f64", run(&gpu, &op, &coarse), run(&cpu, &op, &coarse)),
        (
            "f32",
            run(&gpu.single().unwrap(), &op, &coarse),
            run(&cpu.single().unwrap(), &op, &coarse),
        ),
    ] {
        // Fused and step-by-step results come in pairs.
        for (k, pair) in on_gpu.chunks(2).enumerate() {
            assert!(pair[0] == pair[1], "{name}: GPU fused != GPU steps (output {k})");
        }
        assert!(on_gpu == on_cpu, "{name}: GPU != CPU");
    }
}

/// A whole V-cycle (fused smoother) gives the same bits on the GPU as on the
/// CPU, in both precisions.
#[test]
fn gpu_vcycle_equals_the_cpu_vcycle_bit_for_bit() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, MultigridPreconditioner};
    if !gpu_present("gpu_vcycle_equals_the_cpu_vcycle_bit_for_bit") {
        return;
    }
    let (op, b) = system();
    fn vcycle<B: Backend>(be: &B, op: &Arc<StencilOperator>, b: &[f64]) -> Vec<u64> {
        let h = Hierarchy::build(op, Chebyshev::default()).unwrap().unwrap();
        let mut m = MultigridPreconditioner::new(be, op, h).unwrap();
        let dev = be.prepare(op.clone()).unwrap();
        let mut z = be.zeros(b.len()).unwrap();
        m.apply(be, &dev, &be.upload(b).unwrap(), &mut z).unwrap();
        bits(&be.download(&z).unwrap())
    }
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    assert!(vcycle(&gpu, &op, &b) == vcycle(&cpu, &op, &b), "f64");
    let (g32, c32) = (gpu.single().unwrap(), cpu.single().unwrap());
    assert!(vcycle(&g32, &op, &b) == vcycle(&c32, &op, &b), "f32");
}

/// Each CG step equals its sequence of operations on the GPU bit for bit
/// (both precisions; with and without z and the f32 residual): the vectors
/// and the sums, the latter reduced exactly as `dot` reduces them. CUDA fuses
/// `cg_update` and `promote_dot`; `apply_dot` runs its default.
/// The vectors also equal the CPU's (the sums differ only in the order the
/// two platforms add them). Grids from less than one block to several
/// elements per thread of the 1024-block reduction grid.
#[test]
fn fused_cg_steps_on_the_gpu_equal_their_sequences() {
    use pbsolve_core::backend::{apply_dot_by_steps, cg_update_by_steps, promote_dot_by_steps};
    if !gpu_present("fused_cg_steps_on_the_gpu_equal_their_sequences") {
        return;
    }

    /// Outputs of every step on `be`, fused and by steps: vectors and sums,
    /// each in (fused, by steps) pairs.
    fn run<B: Backend>(be: &B, op: &Arc<StencilOperator>) -> (Vec<Vec<u64>>, Vec<u64>) {
        let single = be.single().unwrap();
        let n = op.len();
        let host = |seed: usize| system_like(n + 31 * seed)[31 * seed..].to_vec();
        let vec = |seed: usize| be.upload(&host(seed)).unwrap();
        let vec32 = |seed: usize| single.upload(&host(seed)).unwrap();
        let down = |v: &B::Vector| bits(&be.download(v).unwrap());
        let down32 = |v: &<B::Single as Backend>::Vector| bits(&single.download(v).unwrap());
        let (mut vectors, mut sums) = (Vec::new(), Vec::new());

        let dop = be.prepare(op.clone()).unwrap();
        let p = vec(1);
        let (mut q1, mut q2) = (vec(2), vec(2));
        let a = be.apply_dot(&dop, &p, &mut q1).unwrap();
        let b = apply_dot_by_steps(be, &dop, &p, &mut q2).unwrap();
        vectors.extend([down(&q1), down(&q2)]);
        sums.extend([a.to_bits(), b.to_bits()]);

        let (p, q, z) = (vec(3), vec(4), vec(5));
        for alpha in [0.731, -1.37e-3] {
            for with_z in [false, true] {
                for with_rounded in [false, true] {
                    let z = with_z.then_some(&z);
                    let (mut x1, mut r1, mut s1) = (vec(6), vec(7), vec32(8));
                    let a = be
                        .cg_update(alpha, &p, &q, &mut x1, &mut r1, z, with_rounded.then_some(&mut s1))
                        .unwrap();
                    let (mut x2, mut r2, mut s2) = (vec(6), vec(7), vec32(8));
                    let b = cg_update_by_steps(be, alpha, &p, &q, &mut x2, &mut r2, z, with_rounded.then_some(&mut s2))
                        .unwrap();
                    vectors.extend([down(&x1), down(&x2), down(&r1), down(&r2), down32(&s1), down32(&s2)]);
                    sums.extend([a.0.to_bits(), b.0.to_bits(), a.1.to_bits(), b.1.to_bits()]);
                }
            }
        }

        let (z32, r) = (vec32(9), vec(10));
        let (mut z1, mut z2) = (vec(11), vec(11));
        let a = be.promote_dot(&z32, &r, &mut z1).unwrap();
        let b = promote_dot_by_steps(be, &z32, &r, &mut z2).unwrap();
        vectors.extend([down(&z1), down(&z2)]);
        sums.extend([a.to_bits(), b.to_bits()]);

        // Every product −0.0.
        let (zero32, minus_zero) = (single.zeros(n).unwrap(), be.upload(&vec![-0.0; n]).unwrap());
        let a = be.promote_dot(&zero32, &minus_zero, &mut z1).unwrap();
        let b = promote_dot_by_steps(be, &zero32, &minus_zero, &mut z2).unwrap();
        sums.extend([a.to_bits(), b.to_bits()]);
        (vectors, sums)
    }

    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    // 150 nodes (one block), 18 200, 117 649 (one element per thread),
    // 343 000 and 4.2 M (several per thread).
    for dims in [[6, 5, 5], [130, 7, 20], [49, 49, 49], [70, 70, 70], [161, 161, 161]] {
        let op = operator(dims, 0.4, 0.15);
        for (name, (gv, gs), (cv, _)) in [
            ("f64", run(&gpu, &op), run(&cpu, &op)),
            (
                "f32",
                run(&gpu.single().unwrap(), &op),
                run(&cpu.single().unwrap(), &op),
            ),
        ] {
            for (k, pair) in gv.chunks(2).enumerate() {
                assert!(
                    pair[0] == pair[1],
                    "{dims:?} {name}: GPU fused != GPU steps (vector {k})"
                );
            }
            for (k, pair) in gs.chunks(2).enumerate() {
                assert_eq!(pair[0], pair[1], "{dims:?} {name}: GPU fused != GPU steps (sum {k})");
            }
            assert!(gv == cv, "{dims:?} {name}: GPU vectors != CPU vectors");
        }
    }
}

/// A whole CG solve on the GPU with the fused steps equals the solve through
/// `Stepwise` (every smoother and CG step as its sequence of kernels, also in
/// the f32 twin) bit for bit: every preconditioner, converged and stopped at
/// `max_iter`, on grids of less and more nodes than the reduction grid has
/// threads.
#[test]
fn gpu_cg_solves_equal_the_stepwise_solves_bit_for_bit() {
    use pbsolve_core::backend::Stepwise;
    use pbsolve_core::PbError;
    if !gpu_present("gpu_cg_solves_equal_the_stepwise_solves_bit_for_bit") {
        return;
    }
    fn outcome<B: Backend>(
        be: &B,
        op: &Arc<StencilOperator>,
        b: &[f64],
        opts: SolverOptions,
    ) -> (Vec<u64>, usize, u64) {
        match pcg(be, op.clone(), b, opts) {
            Ok((x, s)) => (bits(&x), s.iterations, s.relative_residual.to_bits()),
            Err(PbError::NotConverged { iterations, residual }) => (Vec::new(), iterations, residual.to_bits()),
            Err(e) => panic!("{e}"),
        }
    }
    let gpu = CudaBackend::new(0).unwrap();
    let stepwise = Stepwise(CudaBackend::new(0).unwrap());
    let large = operator([81, 81, 81], 0.4, 0.15);
    let large_b: Vec<f64> = system_like(large.len())
        .iter()
        .zip(&large.inv_diag)
        .map(|(v, d)| if *d == 0.0 { 0.0 } else { *v })
        .collect();
    for (op, b) in [system(), (large, large_b)] {
        for pc in [
            Preconditioner::MultigridF32,
            Preconditioner::Multigrid,
            Preconditioner::Jacobi,
        ] {
            for (tol, max_iter) in [(1e-10, 10_000), (1e-30, 3)] {
                let opts = SolverOptions {
                    preconditioner: pc,
                    ..SolverOptions::new(tol, max_iter)
                };
                let fused = outcome(&gpu, &op, &b, opts);
                assert!(
                    fused == outcome(&stepwise, &op, &b, opts),
                    "{:?} {pc:?} tol {tol:e}: fused != Stepwise",
                    op.dims
                );
            }
        }
    }
}

/// The reaction-field decomposition (`coulomb = "analytic"`) gives the CPU's
/// energies on the GPU: one to three focusing levels, both interfaces, with
/// and without ions. The source is built on the host from maps read back
/// from the device.
#[test]
fn analytic_coulomb_matches_the_cpu() {
    use pbsolve_core::pb::polar_solvation;
    use pbsolve_core::{CoulombMethod, GridSpec, InterfaceMethod};
    if !gpu_present("analytic_coulomb_matches_the_cpu") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let atoms = cluster(150, 23);
    let fine = [24.0, 22.0, 26.0];
    let spec = |scale: f64| GridSpec {
        center: [0.1, -0.2, 0.3],
        dims: [49, 45, 53],
        coarse_length: fine.map(|l| scale * l),
        fine_length: fine,
    };
    for (scale, interface, salt) in [
        (1.0, InterfaceMethod::Sharp, 0.15),
        (2.0, InterfaceMethod::Fractional, 0.15),
        (9.0, InterfaceMethod::Sharp, 0.0),
    ] {
        let params = PbParams {
            pdie: 2.0,
            interface,
            coulomb: CoulombMethod::Analytic,
            ..PbParams::default()
        }
        .with_salt(salt, 2.0);
        let spec = spec(scale);
        let r = polar_solvation(&gpu, &atoms, &params, &spec).unwrap();
        let cpu = polar_solvation(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
        let rel = (r.energy_kt - cpu.energy_kt).abs() / cpu.energy_kt.abs();
        assert!(
            rel < 1e-10,
            "{scale} {interface:?}: CPU {} vs CUDA {} ({rel:e})",
            cpu.energy_kt,
            r.energy_kt
        );
        assert_eq!(r.reference_kt.to_bits(), cpu.reference_kt.to_bits());
        assert!(r.stats_reference.is_empty());
        assert_eq!(r.stats_solvated.len(), cpu.stats_solvated.len());
    }
}

/// The nonlinear equation (`equation = "npbe"`, and `smpbe`) gives the CPU's
/// energies and potentials on the GPU: one to three focusing levels, a 1:1
/// and a 2:1 salt, finite-size ions. The Boltzmann terms run in CUDA kernels.
#[test]
fn nonlinear_pb_matches_the_cpu() {
    use pbsolve_core::pb::{polar_solvation, solve_focused, Medium};
    use pbsolve_core::{Equation, GridSpec, IonSpecies};
    if !gpu_present("nonlinear_pb_matches_the_cpu") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let mut atoms = cluster(150, 23);
    for a in atoms.iter_mut().step_by(5) {
        a.charge *= 4.0;
    }
    let fine = [24.0, 22.0, 26.0];
    let spec = |scale: f64| GridSpec {
        center: [0.1, -0.2, 0.3],
        dims: [49, 45, 53],
        coarse_length: fine.map(|l| scale * l),
        fine_length: fine,
    };
    let two_one = vec![
        IonSpecies {
            charge: 2.0,
            concentration: 0.05,
            radius: 2.0,
        },
        IonSpecies {
            charge: -1.0,
            concentration: 0.1,
            radius: 2.0,
        },
    ];
    // The last case: size-modified ions (`smpbe`, 6 Å lattice cells).
    for (scale, ions, size) in [
        (1.0, None, 0.0),
        (2.0, Some(two_one), 0.0),
        (9.0, None, 0.0),
        (2.0, None, 6.0),
    ] {
        let mut params = PbParams {
            pdie: 2.0,
            equation: if size > 0.0 {
                Equation::SizeModified
            } else {
                Equation::Nonlinear
            },
            ion_size: size,
            tol: 1e-10,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        if let Some(ions) = ions {
            params.ions = ions;
        }
        let spec = spec(scale);
        let r = polar_solvation(&gpu, &atoms, &params, &spec).unwrap();
        let cpu = polar_solvation(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
        let rel = (r.solvated_kt - cpu.solvated_kt).abs() / cpu.solvated_kt.abs();
        assert!(
            rel < 1e-10,
            "{scale}: CPU {} vs CUDA {} ({rel:e})",
            cpu.solvated_kt,
            r.solvated_kt
        );
        assert_eq!(r.stats_solvated.len(), cpu.stats_solvated.len());
        let field = solve_focused(&gpu, &spec, &atoms, &params, Medium::Solvated).unwrap();
        let e = field.energy_kt.expect("the nonlinear functional");
        assert!((e - r.solvated_kt).abs() < 1e-10 * e.abs(), "{e} vs {}", r.solvated_kt);
    }
}

/// `op` with periodic axes (its maps are uniform solvent at the seams).
fn with_periodic(op: &StencilOperator, periodic: [bool; 3]) -> Arc<StencilOperator> {
    let op = op.clone();
    Arc::new(StencilOperator::from_parts(
        op.dims, periodic, op.coef, op.eps_x, op.eps_y, op.eps_z, op.kvol,
    ))
}

/// Periodic x and y axes: the multigrid setup on the device equals the host
/// hierarchy, and the operator, the fused residuals, the transfers, the
/// coarse operators and a whole V-cycle give the CPU's bits on the GPU in
/// both precisions; the device assembly equals the host operator.
#[test]
fn periodic_kernels_equal_the_cpu_bit_for_bit() {
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, LevelOp, Levels, MultigridPreconditioner};
    use pbsolve_core::surface::maps_on;
    if !gpu_present("periodic_kernels_equal_the_cpu_bit_for_bit") {
        return;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    let lateral = [true, true, false];
    // [5, 5, 129] coarsens to periodic axes of two unknowns (3 nodes), where
    // both neighbours of a node are the same node.
    let cases = [
        (operator([49, 49, 33], 0.35, 0.15), lateral),
        (operator([41, 33, 17], 0.35, 0.15), [true, false, false]),
        (operator([33, 41, 17], 0.35, 1.0), [false, true, false]),
        (operator([5, 5, 129], 0.5, 0.15), lateral),
    ];
    /// The bits of A x, the fused residual, restriction, the coarse operator
    /// and its residual, prolongation and one V-cycle on `be`.
    fn steps<B: Backend>(be: &B, op: &Arc<StencilOperator>, h: &Hierarchy, b: &[f64]) -> Vec<Vec<u64>> {
        let fine = be.prepare(op.clone()).unwrap();
        let down = |v: &B::Vector| bits(&be.download(v).unwrap());
        let n = b.len();
        let x = be.upload(b).unwrap();
        let mut out = Vec::new();
        let mut y = be.zeros(n).unwrap();
        be.apply(&fine, &x, &mut y).unwrap();
        out.push(down(&y));
        let (mut t, mut r) = (be.zeros(n).unwrap(), be.upload(&system_like(n + 5)[5..]).unwrap());
        be.residual(LevelOp::Fine(&fine), &x, Some(&y), &mut t, &mut r).unwrap();
        out.push(down(&r));
        be.residual(LevelOp::Fine(&fine), &y, None, &mut t, &mut r).unwrap();
        out.push(down(&r));
        if let (Some(t0), Some(c0)) = (h.transfers.first(), h.coarse.first()) {
            let nc = c0.len();
            let mut rc = be.zeros(nc).unwrap();
            be.restrict(t0, &r, &mut rc).unwrap();
            out.push(down(&rc));
            let co = be.prepare_coarse(c0.clone()).unwrap();
            let mut yc = be.zeros(nc).unwrap();
            be.apply_coarse(&co, &rc, &mut yc).unwrap();
            out.push(down(&yc));
            let (mut tc, mut zc) = (be.zeros(nc).unwrap(), be.upload(&system_like(nc + 3)[3..]).unwrap());
            be.residual(LevelOp::Coarse(&co), &rc, Some(&yc), &mut tc, &mut zc)
                .unwrap();
            out.push(down(&zc));
            be.prolong_add(t0, &zc, &mut y).unwrap();
            out.push(down(&y));
        }
        let mut m = MultigridPreconditioner::new(be, op, h.clone()).unwrap();
        let mut z = be.zeros(n).unwrap();
        m.apply(be, &fine, &x, &mut z).unwrap();
        out.push(down(&z));
        out
    }
    for (base, periodic) in cases {
        let op = with_periodic(&base, periodic);
        let what = format!("{:?} {periodic:?}", op.dims);
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let dev = gpu.prepare(op.clone()).unwrap();
        let l = Levels::build(&gpu, op.dims, periodic, &dev, Chebyshev::default())
            .unwrap()
            .unwrap();
        assert_eq!(l.transfers, h.transfers, "{what}");
        assert_eq!(bits(&l.lmax), bits(&h.lmax), "{what}: Gershgorin bounds");
        assert_eq!(l.coarsest, h.coarsest, "{what}: coarsest factor");
        assert!(!h.coarse.is_empty(), "{what}: coarsened");
        for (lev, s) in h.coarse.iter().enumerate() {
            let c = gpu.coarse_to_host(&l.coarse[lev]).unwrap();
            assert_eq!(c.periodic, periodic, "{what}");
            assert_eq!(bits(&c.coef), bits(&s.coef), "{what}: level {} operator", lev + 1);
            assert_eq!(
                bits(&gpu.download(&l.inv_diag[lev]).unwrap()),
                bits(&s.inv_diag),
                "{what}: level {} 1/diag",
                lev + 1
            );
        }
        let b = system_like(op.len());
        assert!(steps(&gpu, &op, &h, &b) == steps(&cpu, &op, &h, &b), "{what}: f64");
        let (g32, c32) = (gpu.single().unwrap(), cpu.single().unwrap());
        assert!(steps(&g32, &op, &h, &b) == steps(&c32, &op, &h, &b), "{what}: f32");
    }
    // Assembly of a periodic grid's operator on the device.
    let atoms = cluster(80, 31);
    let grid = Grid::centered([0.0, 0.3, -0.1], [30.0, 26.0, 23.0], [61, 53, 41])
        .unwrap()
        .with_periodic(lateral);
    let params = PbParams {
        pdie: 4.0,
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0);
    let surf = pbsolve_core::MolecularSurface::for_params(&atoms, &params);
    let host = StencilOperator::new(&grid, CoefficientMaps::solvated_with(&grid, &atoms, &params, &surf));
    let (dev, inv) = gpu
        .assemble_operator(&grid, maps_on(&gpu, &grid, &atoms, &params, Some(&surf)).unwrap())
        .unwrap();
    let back = gpu.operator_to_host(&dev).unwrap();
    assert_eq!(back.periodic, lateral);
    assert_eq!(bits(&host.kvol), bits(&back.kvol), "kvol");
    assert_eq!(bits(&host.inv_diag), bits(&gpu.download(&inv).unwrap()), "1/diag");
}
