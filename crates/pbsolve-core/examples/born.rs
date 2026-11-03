//! Born-ion convergence study on the CPU backend.
//!
//!     cargo run --release -p pbsolve-core --example born

use pbsolve_core::units::{bjerrum_length_vacuum, kt_in_kcal_per_mol};
use pbsolve_core::{polar_solvation, Atom, CpuBackend, GridSpec, PbParams, SurfaceMethod};
use std::time::Instant;

fn main() {
    let a = 2.0;
    let atoms = [Atom {
        position: [0.1, -0.05, 0.02],
        charge: 1.0,
        radius: a,
    }];
    let base = PbParams::default();
    let kt = kt_in_kcal_per_mol(base.temperature);
    let exact = -(bjerrum_length_vacuum(base.temperature) / (2.0 * a)) * (1.0 / base.pdie - 1.0 / base.sdie) * kt;
    println!("Born ion a = {a} Å, analytic ΔG = {exact:.3} kcal/mol");
    println!(
        "{:>10} {:>8} {:>12} {:>9} {:>10} {:>8}",
        "surface", "h (Å)", "ΔG kcal/mol", "error %", "CG iters", "time s"
    );
    for surface in [SurfaceMethod::Mol, SurfaceMethod::HarmonicMol] {
        for h in [0.5, 0.35, 0.25, 0.2] {
            let params = PbParams {
                surface,
                ..base.clone()
            };
            let spec = GridSpec::auto(&atoms, h, 1.7, 12.0).unwrap();
            let t = Instant::now();
            let r = polar_solvation(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
            let iters: usize = r
                .stats_solvated
                .iter()
                .chain(&r.stats_reference)
                .map(|s| s.iterations)
                .sum();
            println!(
                "{:>10} {:>8.3} {:>12.3} {:>9.2} {:>10} {:>8.2}",
                surface.as_str(),
                spec.fine_spacing()[0],
                r.energy_kcal,
                100.0 * (r.energy_kcal - exact) / exact.abs(),
                iters,
                t.elapsed().as_secs_f64()
            );
        }
    }
}
