//! Device throughput of the multigrid V-cycle in f64 and f32: the same
//! operator and right-hand side, `REPS` cycles queued back to back with a
//! single synchronisation, so host latency does not hide the kernel time.
//!
//!     cargo run --release -p pbsolve-cuda --features cuda-13 --example precision

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
mod bench {
    use pbsolve_core::charge::spread_charges;
    use pbsolve_core::multigrid::{Chebyshev, Hierarchy, MultigridPreconditioner};
    use pbsolve_core::operator::{assemble_rhs, StencilOperator};
    use pbsolve_core::solver::{pcg, SolverOptions};
    use pbsolve_core::units::bjerrum_length_vacuum;
    use pbsolve_core::{Atom, Backend, CoefficientMaps, Grid, PbParams, Preconditioner};
    use pbsolve_cuda::CudaBackend;
    use std::sync::Arc;
    use std::time::Instant;

    const REPS: usize = 40;

    fn globule(n: usize) -> Vec<Atom> {
        let mut s: u64 = 2024;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut atoms = Vec::with_capacity(n);
        while atoms.len() < n {
            let p = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5].map(|v| 30.0 * v);
            if p.iter().map(|v| v * v).sum::<f64>() > 14.0 * 14.0 {
                continue;
            }
            atoms.push(Atom {
                position: p,
                charge: rnd() - 0.5,
                radius: 1.2 + 0.7 * rnd(),
            });
        }
        atoms
    }

    fn time_cycles<B: Backend>(be: &B, op: &Arc<StencilOperator>, b: &[f64]) -> (f64, f64) {
        let h = Hierarchy::build(op, Chebyshev::default()).unwrap().unwrap();
        let mut mg = MultigridPreconditioner::new(be, op, h).unwrap();
        let dop = be.prepare(op.clone()).unwrap();
        let r = be.upload(b).unwrap();
        let mut z = be.zeros(b.len()).unwrap();
        let mut y = be.zeros(b.len()).unwrap();
        mg.apply(be, &dop, &r, &mut z).unwrap();
        be.dot(&z, &z).unwrap(); // synchronise
                                 // Fine-level operator applications only (pure bandwidth).
        let t = Instant::now();
        for _ in 0..REPS {
            be.apply(&dop, &z, &mut y).unwrap();
        }
        be.dot(&y, &y).unwrap();
        let apply = t.elapsed().as_secs_f64() / REPS as f64;
        // Whole V-cycles (each contains one host round trip for the coarsest solve).
        let t = Instant::now();
        for _ in 0..REPS {
            mg.apply(be, &dop, &r, &mut z).unwrap();
        }
        be.dot(&z, &z).unwrap();
        (apply, t.elapsed().as_secs_f64() / REPS as f64)
    }

    pub fn main() {
        let gpu = CudaBackend::new(0).expect("CUDA device");
        let single = gpu.single().unwrap();
        println!("{}  ({REPS} repetitions each, ms per call)", gpu.device_name());
        println!(
            "{:>12} | {:>9} {:>9} {:>6} | {:>9} {:>9} {:>6} | {:>13} {:>13} {:>6}",
            "grid",
            "apply f64",
            "apply f32",
            "ratio",
            "cycle f64",
            "cycle f32",
            "ratio",
            "CG f64-MG",
            "CG f32-MG",
            "ratio"
        );
        let atoms = globule(800);
        let params = PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        for n in [97usize, 129, 161, 193] {
            let grid = Grid::centered([0.0; 3], [44.0; 3], [n; 3]).unwrap();
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
            let (a64, c64) = time_cycles(&gpu, &op, &b);
            let (a32, c32) = time_cycles(&single, &op, &b);
            // Complete solves (setup + CG to 1e-8), the solver's own timings.
            let solve = |pc| {
                let opts = SolverOptions {
                    preconditioner: pc,
                    ..SolverOptions::new(1e-8, 1000)
                };
                pcg(&gpu, op.clone(), &b, opts).unwrap().1
            };
            let (s64, s32) = (solve(Preconditioner::Multigrid), solve(Preconditioner::MultigridF32));
            println!(
                "{:>12} | {:>9.3} {:>9.3} {:>6.2} | {:>9.3} {:>9.3} {:>6.2} | {:>6.0} ({:>2} it) {:>6.0} ({:>2} it) {:>6.2}",
                format!("{n}³"),
                1e3 * a64,
                1e3 * a32,
                a64 / a32,
                1e3 * c64,
                1e3 * c32,
                c64 / c32,
                1e3 * s64.solve_seconds,
                s64.iterations,
                1e3 * s32.solve_seconds,
                s32.iterations,
                s64.solve_seconds / s32.solve_seconds
            );
        }
    }
}

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
fn main() {
    bench::main();
}

#[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
fn main() {
    eprintln!("build with --features cuda-12 or --features cuda-13");
}
