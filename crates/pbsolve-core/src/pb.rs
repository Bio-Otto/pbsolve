//! High-level drivers: focused LPBE solves and the polar solvation energy
//! ΔG_polar = ½ Σ q_i [u_solvated(r_i) − u_reference(r_i)], with both states
//! solved on identical grids so grid self-energies cancel exactly.

use crate::autogrid::GridSpec;
use crate::backend::Backend;
use crate::boundary::{analytic_boundary_values, focus_boundary_values, focus_nodes, BoundaryMedium};
use crate::charge::{spread_charges, spread_charges_sparse};
use crate::error::Result;
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::params::PbParams;
use crate::solver::{pcg_on, SolveStats, SolverOptions};
use crate::surface::{maps_on, CoefficientMaps, MolecularSurface};
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol};
use rayon::prelude::*;
use std::f64::consts::PI;
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
    let maps = [
        be.upload_owned(maps.eps_x)?,
        be.upload_owned(maps.eps_y)?,
        be.upload_owned(maps.eps_z)?,
        be.upload_owned(maps.kappa_sq)?,
    ];
    let q = be.upload(node_charge)?;
    let boundary = grid.boundary_indices();
    let g: Vec<f64> = boundary.iter().map(|&c| g_ext[c]).collect();
    let lb0 = bjerrum_length_vacuum(temperature);
    let (x, mut stats) = solve_assembled(be, grid, &boundary, &g, maps, q, lb0, opts)?;
    let t = Instant::now();
    let sol = LevelSolution::<B> {
        grid: grid.clone(),
        x,
        g,
    };
    let (_, u) = sol.potential(be)?;
    stats.solve_seconds += t.elapsed().as_secs_f64();
    Ok((u, stats))
}

/// Assembles the system of maps and node charges already on `be` (as
/// [`crate::operator::StencilOperator::new`] and
/// [`crate::operator::assemble_rhs`] do on the host) and solves it for x;
/// u = x + g with g = `g` on the `boundary` nodes and 0 inside.
#[allow(clippy::too_many_arguments)]
fn solve_assembled<B: Backend>(
    be: &B,
    grid: &Grid,
    boundary: &[usize],
    g: &[f64],
    maps: [B::Vector; 4],
    q: B::Vector,
    lb0: f64,
    opts: SolverOptions,
) -> Result<(B::Vector, SolveStats)> {
    let n = grid.len();
    let (op, inv_diag) = be.assemble_operator(grid, maps)?;
    // b = 4π l_B0 q − A g on interior nodes, 0 on the boundary.
    let mut b = be.zeros(n)?;
    {
        let g = be.scatter(n, boundary, g)?;
        be.apply(&op, &g, &mut b)?;
    }
    be.source_rhs(4.0 * PI * lb0, &q, &inv_diag, &mut b)?;
    drop(q);
    be.synchronize()?;
    pcg_on(be, grid.dims, &op, inv_diag, b, opts)
}

/// The solution of one level, still on the backend: u = x + g, with g the
/// Dirichlet values on the boundary nodes and 0.0 on interior ones.
struct LevelSolution<B: Backend> {
    grid: Grid,
    x: B::Vector,
    /// g on `grid.boundary_indices()`, in that order.
    g: Vec<f64>,
}

impl<B: Backend> LevelSolution<B> {
    /// g at node `c` (0.0 inside).
    fn g_at(&self, c: usize) -> f64 {
        let (i, j, k) = self.grid.ijk(c);
        self.grid.boundary_position(i, j, k).map_or(0.0, |m| self.g[m])
    }

    /// u at the given nodes, computed as [`Self::potential`] computes it,
    /// reading only those values of x.
    fn values_at(&self, be: &B, nodes: &[usize]) -> Result<Vec<f64>> {
        let x = be.gather(&self.x, nodes)?;
        Ok(nodes.iter().zip(x).map(|(&c, x)| x + self.g_at(c)).collect())
    }

    /// The grid and the full potential u = x + g.
    fn potential(self, be: &B) -> Result<(Grid, Vec<f64>)> {
        let mut u = be.download(&self.x)?;
        drop(self.x);
        add_boundary_values(&self.grid, &mut u, &self.g);
        Ok((self.grid, u))
    }
}

/// u ← u + g, with g = `g` (in `grid.boundary_indices()` order) on boundary
/// nodes and 0.0 on interior ones.
fn add_boundary_values(grid: &Grid, u: &mut [f64], g: &[f64]) {
    let [nx, ny, _] = grid.dims;
    u.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                plane[i + nx * j] += grid.boundary_position(i, j, k).map_or(0.0, |m| g[m]);
            }
        }
    });
}

/// Node charges of a level ([`spread_charges_sparse`]): nodes in increasing
/// order and their charges.
type NodeCharges = (Vec<usize>, Vec<f64>);

/// One LPBE solve on `grid` with Dirichlet values `g` (on
/// `grid.boundary_indices()`). Also returns the node charges.
#[allow(clippy::too_many_arguments)]
fn solve_level<B: Backend>(
    be: &B,
    grid: Grid,
    g: Vec<f64>,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
    timings: &mut Timings,
) -> Result<(LevelSolution<B>, SolveStats, NodeCharges)> {
    let t = Instant::now();
    // Maps built from validated parameters are valid (ε > 0, κ̄² ≥ 0).
    let maps = maps_on(be, &grid, atoms, params, surf)?;
    be.synchronize()?;
    timings.maps += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (nodes, charges) = spread_charges_sparse(&grid, atoms)?;
    let q = be.scatter(grid.len(), &nodes, &charges)?;
    let opts = SolverOptions {
        tol: params.tol,
        max_iter: params.max_iter,
        preconditioner: params.preconditioner,
    };
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let (x, mut stats) = solve_assembled(be, &grid, &grid.boundary_indices(), &g, maps, q, lb0, opts)?;
    // The solver's buffers are freed at the next synchronisation (the CUDA
    // memory pool returns them to the driver then); that belongs to the solve.
    let freeing = Instant::now();
    be.synchronize()?;
    stats.solve_seconds += freeing.elapsed().as_secs_f64();
    let total = t.elapsed().as_secs_f64();
    timings.setup += stats.setup_seconds;
    timings.solve += stats.solve_seconds;
    timings.assembly += (total - stats.setup_seconds - stats.solve_seconds).max(0.0);
    Ok((LevelSolution { grid, x, g }, stats, (nodes, charges)))
}

/// Boundary values of `fine` focused from the solution of the coarser level,
/// as [`crate::boundary::focus_boundary`] computes them, reading only the coarse nodes the
/// interpolation uses ([`focus_nodes`]).
fn focus_from<B: Backend>(be: &B, fine: &Grid, coarse: &LevelSolution<B>) -> Result<Vec<f64>> {
    let nodes = focus_nodes(fine, &coarse.grid);
    let u = coarse.values_at(be, &nodes.indices())?;
    focus_boundary_values(fine, &coarse.grid, |c| u[nodes.position(c).expect("a focusing node")])
}

/// Runs the focusing sequence of `spec` (APBS mg-auto, see
/// [`GridSpec::levels`]): the outermost level uses the analytic boundary
/// condition `bcfl`, every finer level takes its boundary from the previous
/// one, read from the backend at the nodes it needs. `finish` turns the
/// finest level's solution and node charges into the result.
fn focused<B: Backend, T>(
    be: &B,
    spec: &GridSpec,
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
    finish: impl FnOnce(LevelSolution<B>, NodeCharges, &mut Timings) -> Result<T>,
) -> Result<(T, Vec<SolveStats>, Timings)> {
    params.validate()?;
    let levels = spec.levels()?;
    let bm = boundary_medium(params, medium);
    let mut stats = Vec::with_capacity(levels.len());
    let mut timings = Timings::default();
    // The molecular surface depends only on the atoms: built once for all levels.
    let t = Instant::now();
    let surf = (medium == Medium::Solvated).then(|| MolecularSurface::for_params(atoms, params));
    timings.maps += t.elapsed().as_secs_f64();
    let mut last: Option<(LevelSolution<B>, NodeCharges)> = None;
    for grid in levels {
        let t = Instant::now();
        let coarse = last.take();
        let g = match &coarse {
            None => analytic_boundary_values(&grid, atoms, params.bcfl, &bm),
            Some((coarse, _)) => focus_from(be, &grid, coarse)?,
        };
        timings.boundary += t.elapsed().as_secs_f64();
        // The coarser level is released before this one allocates (freeing
        // memory belongs to no stage).
        drop(coarse);
        let (sol, s, charges) = solve_level(be, grid, g, atoms, params, surf.as_ref(), &mut timings)?;
        stats.push(s);
        last = Some((sol, charges));
    }
    let (sol, charges) = last.expect("at least one level");
    let out = finish(sol, charges, &mut timings)?;
    Ok((out, stats, timings))
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
    let ((grid, potential), stats, timings) = focused(be, spec, atoms, params, medium, |sol, _, timings| {
        let t = Instant::now();
        let field = sol.potential(be)?;
        timings.solve += t.elapsed().as_secs_f64();
        Ok(field)
    })?;
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

/// [`electrostatic_energy_kt`] of the finest level from u at the charged
/// nodes only: the same products in the same order without the zero terms,
/// hence the same value (a total of exactly zero may differ in its sign).
fn energy_at_charges<B: Backend>(be: &B, sol: &LevelSolution<B>, (nodes, charges): NodeCharges) -> Result<f64> {
    let u = sol.values_at(be, &nodes)?;
    Ok(0.5 * charges.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>())
}

/// Polar (electrostatic) solvation free energy on the given grid.
pub fn polar_solvation<B: Backend>(
    be: &B,
    atoms: &[Atom],
    params: &PbParams,
    spec: &GridSpec,
) -> Result<PolarSolvation> {
    let energy = |sol: LevelSolution<B>, charges, timings: &mut Timings| {
        let t = Instant::now();
        let e = energy_at_charges(be, &sol, charges)?;
        timings.energy += t.elapsed().as_secs_f64();
        Ok(e)
    };
    let (solvated_kt, stats_solvated, mut timings) = focused(be, spec, atoms, params, Medium::Solvated, energy)?;
    let (reference_kt, stats_reference, t) = focused(be, spec, atoms, params, Medium::Reference, energy)?;
    timings += t;
    let energy_kt = solvated_kt - reference_kt;
    Ok(PolarSolvation {
        energy_kcal: energy_kt * kt_in_kcal_per_mol(params.temperature),
        energy_kt,
        solvated_kt,
        reference_kt,
        stats_solvated,
        stats_reference,
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

    /// The pipeline before potentials stayed on the backend, from the host
    /// references: every level's potential in full, boundaries from the full
    /// potential of the coarser level, maps and charges built on the host.
    fn host_focused(spec: &GridSpec, atoms: &[Atom], params: &PbParams, medium: Medium) -> (Grid, Vec<f64>) {
        use crate::boundary::{analytic_boundary, focus_boundary};
        let be = CpuBackend::new();
        let bm = boundary_medium(params, medium);
        let opts = SolverOptions {
            tol: params.tol,
            max_iter: params.max_iter,
            preconditioner: params.preconditioner,
        };
        let mut prev: Option<(Grid, Vec<f64>)> = None;
        for grid in spec.levels().unwrap() {
            let g = match &prev {
                None => analytic_boundary(&grid, atoms, params.bcfl, &bm),
                Some((pg, pu)) => focus_boundary(&grid, pg, pu).unwrap(),
            };
            let maps = coefficient_maps(&grid, atoms, params, medium);
            let q = spread_charges(&grid, atoms).unwrap();
            let (u, _) = solve_with_maps(&be, &grid, maps, &q, &g, params.temperature, opts).unwrap();
            prev = Some((grid, u));
        }
        prev.unwrap()
    }

    /// Energies taken from u at the charged nodes, with the focusing
    /// boundaries read at the stencil nodes only, equal the energies of the
    /// full potentials bit for bit; so do the potentials of `solve_focused`
    /// and of the host references. One to three focusing levels, every
    /// boundary condition and preconditioner.
    #[test]
    fn energies_from_the_charged_nodes_equal_the_full_potential_energies() {
        use crate::params::BoundaryCondition;
        let atoms = small_protein();
        let auto = GridSpec::auto(&atoms, 0.5, 1.7, 12.0).unwrap();
        let fine = auto.fine_length;
        let two = GridSpec {
            coarse_length: fine.map(|l| 2.0 * l),
            ..auto.clone()
        };
        let three = GridSpec {
            coarse_length: fine.map(|l| 9.0 * l),
            ..auto.clone()
        };
        let single = GridSpec {
            coarse_length: fine,
            ..auto.clone()
        };
        // Equal lengths along y: the fine y faces lie on the coarse ones, so
        // focusing reads coarse boundary nodes (u = g there).
        let flush = GridSpec {
            coarse_length: [2.0 * fine[0], fine[1], 1.5 * fine[2]],
            ..auto.clone()
        };
        let salt = PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        let cases = [
            (&two, salt.clone()),
            (&three, salt.clone()),
            (&single, salt.clone()),
            (&flush, salt.clone()),
            (
                &two,
                PbParams {
                    bcfl: BoundaryCondition::Sdh,
                    preconditioner: Preconditioner::Multigrid,
                    ..salt.clone()
                },
            ),
            (
                &two,
                PbParams {
                    bcfl: BoundaryCondition::MdhFast,
                    ..salt.clone()
                },
            ),
            (
                &three,
                PbParams {
                    bcfl: BoundaryCondition::Zero,
                    preconditioner: Preconditioner::Jacobi,
                    ..PbParams::default()
                },
            ),
        ];
        let mut levels = Vec::new();
        for (spec, params) in cases {
            let be = CpuBackend::new();
            let r = polar_solvation(&be, &atoms, &params, spec).unwrap();
            let label = format!(
                "{:?} {:?}, {} levels",
                params.bcfl,
                params.preconditioner,
                r.stats_solvated.len()
            );
            for (medium, kt) in [(Medium::Solvated, r.solvated_kt), (Medium::Reference, r.reference_kt)] {
                let field = solve_focused(&be, spec, &atoms, &params, medium).unwrap();
                let full = electrostatic_energy_kt(&field, &atoms).unwrap();
                assert_eq!(kt.to_bits(), full.to_bits(), "{label} {medium:?}: {kt} vs {full}");
                let (grid, u) = host_focused(spec, &atoms, &params, medium);
                assert_eq!(field.grid, grid);
                assert!(
                    field.potential.iter().zip(&u).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label} {medium:?}: potential"
                );
            }
            assert_eq!(r.energy_kt.to_bits(), (r.solvated_kt - r.reference_kt).to_bits());
            levels.push(r.stats_solvated.len());
        }
        assert_eq!(levels, [2, 3, 1, 2, 2, 2, 3]);
        let coarse = &flush.levels().unwrap()[0];
        let read = focus_nodes(&flush.levels().unwrap()[1], coarse).indices();
        assert!(read.iter().any(|&c| {
            let (i, j, k) = coarse.ijk(c);
            coarse.is_boundary(i, j, k)
        }));
        // Without charges both energies are zero.
        let neutral: Vec<Atom> = atoms.iter().map(|a| Atom { charge: 0.0, ..*a }).collect();
        let r = polar_solvation(&CpuBackend::new(), &neutral, &salt, &auto).unwrap();
        assert_eq!((r.solvated_kt, r.reference_kt, r.energy_kt), (0.0, 0.0, 0.0));
    }

    /// `add_boundary_values` adds g on the boundary and 0.0 inside, which
    /// turns an interior −0.0 into +0.0 as the full-length form did.
    #[test]
    fn boundary_values_are_added_on_the_boundary_nodes_only() {
        let grid = Grid::centered([0.0; 3], [6.0, 5.0, 7.0], [7, 6, 9]).unwrap();
        let bidx = grid.boundary_indices();
        let g: Vec<f64> = (0..bidx.len())
            .map(|m| if m % 5 == 0 { -0.0 } else { m as f64 - 40.0 })
            .collect();
        let mut g_ext = vec![0.0; grid.len()];
        for (&c, &v) in bidx.iter().zip(&g) {
            g_ext[c] = v;
        }
        let x: Vec<f64> = (0..grid.len())
            .map(|c| if c % 3 == 0 { -0.0 } else { 0.25 * c as f64 })
            .collect();
        let mut u = x.clone();
        add_boundary_values(&grid, &mut u, &g);
        for c in 0..grid.len() {
            let (i, j, k) = grid.ijk(c);
            let want = x[c] + if grid.is_boundary(i, j, k) { g_ext[c] } else { 0.0 };
            assert_eq!(u[c].to_bits(), want.to_bits(), "node {c}");
        }
        let interior_zero =
            (0..grid.len()).find(|&c| x[c].to_bits() == (-0.0f64).to_bits() && bidx.binary_search(&c).is_err());
        assert_eq!(u[interior_zero.unwrap()].to_bits(), 0.0f64.to_bits());
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
