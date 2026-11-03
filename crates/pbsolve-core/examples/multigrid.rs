//! Linear-solver study: Jacobi-PCG versus multigrid-PCG on one LPBE system
//! (synthetic 800-atom globule, sharp molecular surface, 0.15 M salt) for
//! growing grids. Reports iterations, multigrid setup and solve times and the
//! difference between the two solutions.
//!
//!     cargo run --release -p pbsolve-core --example multigrid

use pbsolve_core::charge::spread_charges;
use pbsolve_core::multigrid::{Chebyshev, Hierarchy};
use pbsolve_core::operator::{assemble_rhs, StencilOperator};
use pbsolve_core::solver::{pcg, SolverOptions};
use pbsolve_core::units::bjerrum_length_vacuum;
use pbsolve_core::{Atom, CoefficientMaps, CpuBackend, Grid, PbParams, Preconditioner};
use std::sync::Arc;
use std::time::Instant;

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
        let radius = if atoms.len() % 5 == 0 { 0.0 } else { 1.2 + 0.7 * rnd() };
        atoms.push(Atom {
            position: p,
            charge: rnd() - 0.5,
            radius,
        });
    }
    atoms
}

fn main() {
    let atoms = globule(800);
    let params = PbParams {
        pdie: 2.0,
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0);
    let lb0 = bjerrum_length_vacuum(params.temperature);
    println!("800-atom globule, ε 2/78.54, 0.15 M, srfm mol; tol 1e-8; zero Dirichlet boundary");
    println!(
        "{:>15} {:>6} {:>7} | {:>7} {:>9} | {:>6} {:>8} {:>8} | {:>9}",
        "dims", "h (Å)", "levels", "Jacobi", "time s", "MG", "setup s", "total s", "|Δu|/|u|"
    );
    for n in [65usize, 97, 129, 161] {
        let len = 44.0;
        let grid = Grid::centered([0.0; 3], [len; 3], [n; 3]).unwrap();
        let maps = CoefficientMaps::solvated(&grid, &atoms, &params);
        let op = Arc::new(StencilOperator::new(&grid, maps));
        let q = spread_charges(&grid, &atoms).unwrap();
        let b = assemble_rhs(&op, &q, &vec![0.0; grid.len()], lb0);
        let be = CpuBackend::new();

        let t = Instant::now();
        let (uj, sj) = pcg(
            &be,
            op.clone(),
            &b,
            SolverOptions {
                preconditioner: Preconditioner::Jacobi,
                ..SolverOptions::new(1e-8, 50_000)
            },
        )
        .unwrap();
        let tj = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let setup = t.elapsed().as_secs_f64();
        let levels = h.levels();
        let t = Instant::now();
        let (um, sm) = pcg(&be, op.clone(), &b, SolverOptions::new(1e-8, 50_000)).unwrap();
        let tm = t.elapsed().as_secs_f64();

        let diff = uj.iter().zip(&um).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        let norm = uj.iter().map(|a| a * a).sum::<f64>().sqrt();
        println!(
            "{:>15} {:>6.3} {:>7} | {:>7} {:>9.2} | {:>6} {:>8.2} {:>8.2} | {:>9.1e}",
            format!("{n}x{n}x{n}"),
            grid.spacing[0],
            levels,
            sj.iterations,
            tj,
            sm.iterations,
            setup,
            tm,
            diff / norm
        );
    }
}
