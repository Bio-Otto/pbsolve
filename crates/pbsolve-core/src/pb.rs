//! High-level drivers: focused LPBE solves and the polar solvation energy
//! ΔG_polar = ½ Σ q_i [u_solvated(r_i) − u_reference(r_i)], with both states
//! solved on identical grids so grid self-energies cancel exactly.

use crate::autogrid::GridSpec;
use crate::backend::Backend;
use crate::boundary::{analytic_boundary, focus_boundary, BoundaryMedium};
use crate::charge::spread_charges;
use crate::error::Result;
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::operator::{assemble_rhs, StencilOperator};
use crate::params::PbParams;
use crate::solver::{pcg, SolveStats, SolverOptions};
use crate::surface::{CoefficientMaps, MolecularSurface};
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Which dielectric/ionic environment to solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medium {
    /// Solute in solvent with mobile ions.
    Solvated,
    /// Homogeneous medium with ε = pdie and no ions (APBS: second `elec`
    /// block with `sdie = pdie`, no `ion`).
    Reference,
}

/// Wall time (seconds) spent in each stage of a calculation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Dielectric and ion-accessibility maps (molecular surface).
    pub maps: f64,
    /// Boundary values (analytic on the coarsest level, then focusing).
    pub boundary: f64,
    /// Charge spreading, operator and right-hand side assembly.
    pub assembly: f64,
    /// Preconditioner setup (multigrid hierarchy) and uploads.
    pub setup: f64,
    /// CG iterations.
    pub solve: f64,
    /// Energy evaluation.
    pub energy: f64,
}

impl Timings {
    pub fn total(&self) -> f64 {
        self.maps + self.boundary + self.assembly + self.setup + self.solve + self.energy
    }
}

impl std::ops::AddAssign for Timings {
    fn add_assign(&mut self, o: Self) {
        self.maps += o.maps;
        self.boundary += o.boundary;
        self.assembly += o.assembly;
        self.setup += o.setup;
        self.solve += o.solve;
        self.energy += o.energy;
    }
}

/// Potential (kT/e) on the finest grid.
#[derive(Clone, Debug)]
pub struct FieldSolution {
    pub grid: Grid,
    pub potential: Vec<f64>,
    /// One entry per grid level (coarse first).
    pub stats: Vec<SolveStats>,
    /// Summed over all levels.
    pub timings: Timings,
}

#[derive(Clone, Debug)]
pub struct PolarSolvation {
    /// ΔG_polar in kcal/mol.
    pub energy_kcal: f64,
    /// ΔG_polar in kT.
    pub energy_kt: f64,
    /// ½ Σ q u in the solvated state (kT).
    pub solvated_kt: f64,
    /// ½ Σ q u in the reference state (kT).
    pub reference_kt: f64,
    pub stats_solvated: Vec<SolveStats>,
    pub stats_reference: Vec<SolveStats>,
    pub grid: GridSpec,
    /// Both states and the energy evaluation.
    pub timings: Timings,
}

/// Dielectric and ion maps of one grid level for the given medium.
pub fn coefficient_maps(grid: &Grid, atoms: &[Atom], params: &PbParams, medium: Medium) -> CoefficientMaps {
    let surf = (medium == Medium::Solvated).then(|| MolecularSurface::for_params(atoms, params));
    maps_with(grid, atoms, params, surf.as_ref())
}

/// Maps of one level; `surf` is the solvated state's surface (None: reference).
fn maps_with(grid: &Grid, atoms: &[Atom], params: &PbParams, surf: Option<&MolecularSurface>) -> CoefficientMaps {
    match surf {
        Some(s) => CoefficientMaps::solvated_with(grid, atoms, params, s),
        None => CoefficientMaps::homogeneous(grid, params.pdie),
    }
}

fn boundary_medium(params: &PbParams, medium: Medium) -> BoundaryMedium {
    let lb0 = bjerrum_length_vacuum(params.temperature);
    match medium {
        Medium::Reference => BoundaryMedium {
            eps: params.pdie,
            eps_solute: params.pdie,
            kappa: 0.0,
            lb0,
        },
        Medium::Solvated => {
            let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
            BoundaryMedium {
                eps: params.sdie,
                eps_solute: params.pdie,
                kappa: (kbar2 / params.sdie).sqrt(),
                lb0,
            }
        }
    }
}

/// Solves the LPBE on `grid` for given coefficient maps, node charges (e)
/// and Dirichlet data `g_ext` (boundary nodes; interior ignored).
/// Returns u = x + g_ext in kT/e.
///
/// This is the discretisation + solver alone, independent of how the maps
/// were built — used for "map injection" tests against APBS.
pub fn solve_with_maps<B: Backend>(
    be: &B,
    grid: &Grid,
    maps: CoefficientMaps,
    node_charge: &[f64],
    g_ext: &[f64],
    temperature: f64,
    opts: SolverOptions,
) -> Result<(Vec<f64>, SolveStats)> {
    let n = grid.len();
    for (name, len) in [
        ("eps_x", maps.eps_x.len()),
        ("eps_y", maps.eps_y.len()),
        ("eps_z", maps.eps_z.len()),
        ("kappa_sq", maps.kappa_sq.len()),
        ("node_charge", node_charge.len()),
        ("g_ext", g_ext.len()),
    ] {
        if len != n {
            return crate::error::invalid(format!("{name} has {len} values, grid has {n} nodes"));
        }
    }
    let bad_eps = [&maps.eps_x, &maps.eps_y, &maps.eps_z]
        .iter()
        .any(|m| m.par_iter().any(|e| e.is_nan() || *e <= 0.0));
    if bad_eps || maps.kappa_sq.par_iter().any(|k| k.is_nan() || *k < 0.0) {
        return crate::error::invalid("dielectric maps must be positive and kappa_sq non-negative");
    }
    let mut g = vec![0.0; n];
    for c in grid.boundary_indices() {
        g[c] = g_ext[c];
    }
    let lb0 = bjerrum_length_vacuum(temperature);
    let op = Arc::new(StencilOperator::new(grid, maps));
    let b = assemble_rhs(&op, node_charge, &g, lb0);
    let (mut u, stats) = pcg(be, op, &b, opts)?;
    u.par_iter_mut().zip(g.par_iter()).for_each(|(ui, gi)| *ui += gi);
    Ok((u, stats))
}

/// One LPBE solve on `grid` with Dirichlet data `g_ext`. Returns u = x + g_ext.
fn solve_level<B: Backend>(
    be: &B,
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
    g_ext: Vec<f64>,
    timings: &mut Timings,
) -> Result<(Vec<f64>, SolveStats)> {
    let t = Instant::now();
    let maps = maps_with(grid, atoms, params, surf);
    timings.maps += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let q = spread_charges(grid, atoms)?;
    let opts = SolverOptions {
        tol: params.tol,
        max_iter: params.max_iter,
        preconditioner: params.preconditioner,
    };
    let (u, stats) = solve_with_maps(be, grid, maps, &q, &g_ext, params.temperature, opts)?;
    let total = t.elapsed().as_secs_f64();
    timings.setup += stats.setup_seconds;
    timings.solve += stats.solve_seconds;
    timings.assembly += (total - stats.setup_seconds - stats.solve_seconds).max(0.0);
    Ok((u, stats))
}

/// Focusing solve over all levels of `spec` (APBS mg-auto sequence, see
/// [`GridSpec::levels`]): the outermost level uses the analytic boundary
/// condition `bcfl`, every finer level takes its boundary from the previous one.
pub fn solve_focused<B: Backend>(
    be: &B,
    spec: &GridSpec,
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
) -> Result<FieldSolution> {
    params.validate()?;
    let levels = spec.levels()?;
    let bm = boundary_medium(params, medium);
    let mut stats = Vec::with_capacity(levels.len());
    let mut timings = Timings::default();
    // The molecular surface depends only on the atoms: built once for all levels.
    let t = Instant::now();
    let surf = (medium == Medium::Solvated).then(|| MolecularSurface::for_params(atoms, params));
    timings.maps += t.elapsed().as_secs_f64();
    let mut prev: Option<(Grid, Vec<f64>)> = None;
    for grid in levels {
        let t = Instant::now();
        let g = match &prev {
            None => analytic_boundary(&grid, atoms, params.bcfl, &bm),
            Some((pg, pu)) => focus_boundary(&grid, pg, pu)?,
        };
        timings.boundary += t.elapsed().as_secs_f64();
        let (u, s) = solve_level(be, &grid, atoms, params, surf.as_ref(), g, &mut timings)?;
        stats.push(s);
        prev = Some((grid, u));
    }
    let (grid, potential) = prev.expect("at least one level");
    Ok(FieldSolution {
        grid,
        potential,
        stats,
        timings,
    })
}

/// ½ Σ q_i u(r_i) in kT (trilinear interpolation, identical to APBS
/// `calcenergy total` with `chgm spl0`).
pub fn electrostatic_energy_kt(sol: &FieldSolution, atoms: &[Atom]) -> Result<f64> {
    let q = spread_charges(&sol.grid, atoms)?;
    Ok(0.5 * q.iter().zip(&sol.potential).map(|(a, b)| a * b).sum::<f64>())
}

/// Polar (electrostatic) solvation free energy on the given grid.
pub fn polar_solvation<B: Backend>(
    be: &B,
    atoms: &[Atom],
    params: &PbParams,
    spec: &GridSpec,
) -> Result<PolarSolvation> {
    let solv = solve_focused(be, spec, atoms, params, Medium::Solvated)?;
    let refr = solve_focused(be, spec, atoms, params, Medium::Reference)?;
    let t = Instant::now();
    let solvated_kt = electrostatic_energy_kt(&solv, atoms)?;
    let reference_kt = electrostatic_energy_kt(&refr, atoms)?;
    let energy_kt = solvated_kt - reference_kt;
    let mut timings = solv.timings;
    timings += refr.timings;
    timings.energy = t.elapsed().as_secs_f64();
    Ok(PolarSolvation {
        energy_kcal: energy_kt * kt_in_kcal_per_mol(params.temperature),
        energy_kt,
        solvated_kt,
        reference_kt,
        stats_solvated: solv.stats,
        stats_reference: refr.stats,
        grid: spec.clone(),
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::params::{Preconditioner, SurfaceMethod};

    /// Born ion: ΔG = −(l_B0 z² / 2a)(1/ε_in − 1/ε_out) kT.
    fn born(params: &PbParams, a: f64) -> (f64, f64) {
        let atoms = [Atom {
            position: [0.1, -0.05, 0.02],
            charge: 1.0,
            radius: a,
        }];
        let spec = GridSpec::auto(&atoms, 0.2, 1.7, 12.0).unwrap();
        let res = polar_solvation(&CpuBackend::new(), &atoms, params, &spec).unwrap();
        let lb0 = bjerrum_length_vacuum(params.temperature);
        let exact = -(lb0 / (2.0 * a)) * (1.0 / params.pdie - 1.0 / params.sdie);
        (res.energy_kt, exact)
    }

    #[test]
    fn born_ion_matches_analytic_solution() {
        for surface in [SurfaceMethod::Mol, SurfaceMethod::HarmonicMol] {
            let params = PbParams {
                surface,
                ..PbParams::default()
            };
            let (num, exact) = born(&params, 2.0);
            let err = (num - exact).abs() / exact.abs();
            assert!(
                err < 0.03,
                "{surface:?}: numeric {num:.3} kT vs analytic {exact:.3} kT ({:.2}%)",
                100.0 * err
            );
        }
    }

    fn small_protein() -> Vec<Atom> {
        // 40 atoms on a deterministic pseudo-random walk: crevices, mixed charges.
        let mut atoms = Vec::new();
        let mut p = [0.0f64; 3];
        let mut s: u64 = 12345;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for i in 0..40 {
            for v in p.iter_mut() {
                *v += 2.4 * next();
            }
            let radius = if i % 7 == 3 { 0.0 } else { 1.2 + 0.8 * (next() + 0.5) };
            atoms.push(Atom {
                position: p,
                charge: 0.8 * next(),
                radius,
            });
        }
        atoms
    }

    fn solve(atoms: &[Atom], params: &PbParams, spacing: f64) -> PolarSolvation {
        let spec = GridSpec::auto(atoms, spacing, 1.7, 12.0).unwrap();
        polar_solvation(&CpuBackend::new(), atoms, params, &spec).unwrap()
    }

    #[test]
    fn multigrid_and_jacobi_give_the_same_energies() {
        let base = PbParams {
            pdie: 2.0,
            tol: 1e-10,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        for (atoms, h) in [(small_protein(), 0.45), (small_protein(), 0.3)] {
            let mg = solve(
                &atoms,
                &PbParams {
                    preconditioner: Preconditioner::Multigrid,
                    ..base.clone()
                },
                h,
            );
            let jac = solve(
                &atoms,
                &PbParams {
                    preconditioner: Preconditioner::Jacobi,
                    ..base.clone()
                },
                h,
            );
            for (a, b) in [
                (mg.solvated_kt, jac.solvated_kt),
                (mg.reference_kt, jac.reference_kt),
                (mg.energy_kt, jac.energy_kt),
            ] {
                assert!((a - b).abs() < 1e-8 * b.abs(), "h={h}: multigrid {a} vs Jacobi {b}");
            }
            for s in mg.stats_solvated.iter().chain(&mg.stats_reference) {
                assert_eq!(s.preconditioner, Preconditioner::Multigrid);
                assert!(s.levels >= 2, "{s:?}");
            }
            for s in jac.stats_solvated.iter().chain(&jac.stats_reference) {
                assert_eq!((s.preconditioner, s.levels), (Preconditioner::Jacobi, 1));
            }
            // The f32 V-cycle changes neither the problem nor the accuracy.
            let mixed = solve(
                &atoms,
                &PbParams {
                    preconditioner: Preconditioner::MultigridF32,
                    ..base.clone()
                },
                h,
            );
            for (a, b) in [
                (mixed.solvated_kt, mg.solvated_kt),
                (mixed.reference_kt, mg.reference_kt),
                (mixed.energy_kt, mg.energy_kt),
            ] {
                assert!(
                    (a - b).abs() < 1e-10 * b.abs(),
                    "h={h}: multigrid-f32 {a} vs multigrid {b}"
                );
            }
            for (s, t) in mixed.stats_solvated.iter().zip(&mg.stats_solvated) {
                assert_eq!((s.preconditioner, s.levels), (Preconditioner::MultigridF32, t.levels));
                assert!(s.iterations <= t.iterations + 1, "{s:?} vs {t:?}");
            }
        }
    }

    #[test]
    fn multigrid_iterations_do_not_grow_with_refinement() {
        let atoms = small_protein();
        let mut mg_iters = Vec::new();
        let mut jac_iters = Vec::new();
        for h in [0.6, 0.4, 0.3, 0.2] {
            for (pre, out) in [
                (Preconditioner::Multigrid, &mut mg_iters),
                (Preconditioner::Jacobi, &mut jac_iters),
            ] {
                let params = PbParams {
                    preconditioner: pre,
                    ..PbParams::default()
                }
                .with_salt(0.1, 2.0);
                let r = solve(&atoms, &params, h);
                out.push(r.stats_solvated.iter().map(|s| s.iterations).max().unwrap());
            }
        }
        eprintln!("max CG iterations per level, h = 0.6/0.4/0.3/0.2 A: multigrid {mg_iters:?}, Jacobi {jac_iters:?}");
        let (lo, hi) = (*mg_iters.iter().min().unwrap(), *mg_iters.iter().max().unwrap());
        assert!(hi <= 20, "multigrid iterations {mg_iters:?}");
        assert!(hi <= lo + 4, "multigrid iterations grow with refinement: {mg_iters:?}");
        assert!(jac_iters[3] > 5 * hi, "Jacobi {jac_iters:?} vs multigrid {mg_iters:?}");
    }

    #[test]
    fn multigrid_results_do_not_depend_on_the_thread_count() {
        let atoms = small_protein();
        for preconditioner in [Preconditioner::MultigridF32, Preconditioner::Multigrid] {
            let params = PbParams {
                preconditioner,
                ..PbParams::default()
            }
            .with_salt(0.15, 2.0);
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| solve(&atoms, &params, 0.4))
            };
            let (a, b) = (run(1), run(4));
            assert_eq!(a.energy_kt.to_bits(), b.energy_kt.to_bits(), "{preconditioner:?}");
            // Everything but the wall times must match exactly.
            let numerics = |r: &PolarSolvation| {
                r.stats_solvated
                    .iter()
                    .chain(&r.stats_reference)
                    .map(|s| (s.iterations, s.relative_residual.to_bits(), s.preconditioner, s.levels))
                    .collect::<Vec<_>>()
            };
            assert_eq!(numerics(&a), numerics(&b));
            assert!(numerics(&a).iter().all(|n| n.2 == preconditioner));
        }
    }

    #[test]
    fn default_parameters_use_the_single_precision_multigrid() {
        assert_eq!(PbParams::default().preconditioner, Preconditioner::MultigridF32);
        let atoms = small_protein();
        let params = PbParams::default().with_salt(0.15, 2.0);
        let default = solve(&atoms, &params, 0.4);
        let f64_cycle = solve(
            &atoms,
            &PbParams {
                preconditioner: Preconditioner::Multigrid,
                ..params.clone()
            },
            0.4,
        );
        for s in default.stats_solvated.iter().chain(&default.stats_reference) {
            assert_eq!(s.preconditioner, Preconditioner::MultigridF32);
            assert!(s.levels >= 2 && s.relative_residual < params.tol, "{s:?}");
        }
        let rel = (default.energy_kt - f64_cycle.energy_kt).abs() / f64_cycle.energy_kt.abs();
        assert!(
            rel < 1e-9,
            "default {} vs f64 multigrid {}",
            default.energy_kt,
            f64_cycle.energy_kt
        );
    }

    #[test]
    fn born_ion_with_jacobi_matches_multigrid() {
        let mg = born(&PbParams::default(), 2.0).0;
        let jac = born(
            &PbParams {
                preconditioner: Preconditioner::Jacobi,
                ..PbParams::default()
            },
            2.0,
        )
        .0;
        assert!((mg - jac).abs() < 1e-6 * jac.abs(), "{mg} vs {jac}");
    }
}
