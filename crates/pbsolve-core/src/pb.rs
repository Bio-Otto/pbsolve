//! High-level drivers: focused LPBE solves and the polar solvation energy
//! ΔG_polar = ½ Σ q_i [u_solvated(r_i) − u_reference(r_i)], with both states
//! solved on identical grids so grid self-energies cancel exactly.

use crate::autogrid::GridSpec;
use crate::backend::{Backend, MemoryHold};
use crate::batch::calculation_device_bytes;
use crate::boundary::{analytic_boundary_values, focus_boundary_values, focus_nodes, BoundaryMedium};
use crate::charge::{spread_charges_sparse_with, spread_charges_with};
use crate::coulomb::{coulomb_energy_kt, Split};
use crate::error::{PbError, Result};
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::nonlinear::{self, IonTerms, NonlinearLevel};
use crate::params::{ChargeMethod, CoulombMethod, PbParams};
use crate::solver::{pcg_on, pcg_setup, SolveStats, SolverOptions};
use crate::surface::{maps_on, CoefficientMaps, MolecularSurface};
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::mpsc;
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
    /// Boundary values (analytic on the coarsest level, then focusing). On a
    /// device backend (CUDA) the solvated state's analytic boundary (and its
    /// molecular surface, under `maps`) is built in the background while the
    /// device works; only the time spent waiting for it counts.
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
    /// The free-energy functional (kT) of a nonlinear state with ions
    /// ([`crate::nonlinear`]); None for linear states, whose energy is
    /// ½ Σ q u ([`electrostatic_energy_kt`]).
    pub energy_kt: Option<f64>,
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
    check_maps(grid, &maps, node_charge, g_ext)?;
    let maps = [
        be.upload_owned(maps.eps_x)?,
        be.upload_owned(maps.eps_y)?,
        be.upload_owned(maps.eps_z)?,
        be.upload_owned(maps.kappa_sq)?,
    ];
    let boundary = grid.boundary_indices();
    let g: Vec<f64> = boundary.iter().map(|&c| g_ext[c]).collect();
    let lb0 = bjerrum_length_vacuum(temperature);
    let charges = || be.upload(node_charge);
    let solved = solve_assembled(be, grid, &boundary, Pending::Ready(g), maps, charges, lb0, opts)?;
    let mut stats = solved.stats;
    let t = Instant::now();
    let sol = LevelSolution::<B> {
        grid: grid.clone(),
        x: solved.x,
        g: solved.g,
    };
    let (_, u) = sol.potential(be)?;
    stats.solve_seconds += t.elapsed().as_secs_f64();
    Ok((u, stats))
}

/// [`solve_with_maps`] for the nonlinear equation with the ions of `params`
/// (`kappa_sq` is κ̄² times the ion accessibility, as for the linear one).
/// Without ions it is the linear solve.
pub fn solve_with_maps_nonlinear<B: Backend>(
    be: &B,
    grid: &Grid,
    maps: CoefficientMaps,
    node_charge: &[f64],
    g_ext: &[f64],
    params: &PbParams,
    opts: SolverOptions,
) -> Result<(Vec<f64>, SolveStats)> {
    let Some(ions) = IonTerms::new(params) else {
        return solve_with_maps(be, grid, maps, node_charge, g_ext, params.temperature, opts);
    };
    check_maps(grid, &maps, node_charge, g_ext)?;
    let g: Vec<f64> = grid.boundary_indices().iter().map(|&c| g_ext[c]).collect();
    let (nodes, charges): (Vec<usize>, Vec<f64>) = node_charge
        .iter()
        .enumerate()
        .filter(|(_, q)| **q != 0.0)
        .map(|(c, q)| (c, *q))
        .unzip();
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let (sol, stats, _) = nonlinear::solve_level(be, grid.clone(), g, maps, (&nodes, &charges), &ions, lb0, opts)?;
    let (_, u) = sol.potential(be)?;
    Ok((u, stats))
}

fn check_maps(grid: &Grid, maps: &CoefficientMaps, node_charge: &[f64], g_ext: &[f64]) -> Result<()> {
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
    Ok(())
}

/// A host input that is ready, or still being built in the background
/// ([`with_background_inputs`]).
pub(crate) enum Pending<T> {
    Ready(T),
    Later(mpsc::Receiver<T>),
}

impl<T> Pending<T> {
    /// The value, once it is ready.
    fn wait(self) -> Result<T> {
        match self {
            Self::Ready(value) => Ok(value),
            Self::Later(rx) => rx
                .recv()
                .map_err(|_| PbError::Backend("the host work in the background stopped".into())),
        }
    }
}

/// The result of [`solve_assembled`].
pub(crate) struct Solved<B: Backend> {
    pub(crate) x: B::Vector,
    /// The Dirichlet values on the boundary nodes.
    pub(crate) g: Vec<f64>,
    pub(crate) stats: SolveStats,
    /// Seconds spent waiting for `g` from the background.
    pub(crate) waited: f64,
}

/// Assembles the system of maps already on `be` (as
/// [`crate::operator::StencilOperator::new`] and
/// [`crate::operator::assemble_rhs`] do on the host) and solves it for x;
/// u = x + g with g on the `boundary` nodes and 0 inside. `charges` puts the
/// node charges on the backend. Boundary values still being built in the
/// background are waited for only after the preconditioner is set up, as
/// the setup does not need them; the results are the same either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_assembled<B: Backend>(
    be: &B,
    grid: &Grid,
    boundary: &[usize],
    g: Pending<Vec<f64>>,
    maps: [B::Vector; 4],
    charges: impl FnOnce() -> Result<B::Vector>,
    lb0: f64,
    opts: SolverOptions,
) -> Result<Solved<B>> {
    let n = grid.len();
    let (op, inv_diag) = be.assemble_operator(grid, maps)?;
    // b = 4π l_B0 q − A g on interior nodes, 0 on the boundary.
    let rhs = |g: &[f64], inv_diag: &B::Vector| -> Result<B::Vector> {
        let mut b = be.zeros(n)?;
        {
            let g = be.scatter(n, boundary, g)?;
            be.apply(&op, &g, &mut b)?;
        }
        let q = charges()?;
        be.source_rhs(4.0 * PI * lb0, &q, inv_diag, &mut b)?;
        Ok(b)
    };
    match g {
        Pending::Ready(g) => {
            let b = rhs(&g, &inv_diag)?;
            be.synchronize()?;
            let (x, stats) = pcg_on(be, grid.dims, &op, inv_diag, b, opts)?;
            Ok(Solved {
                x,
                g,
                stats,
                waited: 0.0,
            })
        }
        later => {
            // The setup takes the inverse diagonal, which the right-hand side
            // needs too. The copy takes the place b held during the setup, so
            // the memory peak (in the iterations) does not change.
            let mut d = be.zeros(n)?;
            be.copy(&inv_diag, &mut d)?;
            let prepared = pcg_setup(be, grid.dims, &op, inv_diag, opts)?;
            let t = Instant::now();
            let g = later.wait()?;
            let waited = t.elapsed().as_secs_f64();
            let b = rhs(&g, &d)?;
            drop(d);
            be.synchronize()?;
            let (x, stats) = prepared.solve(be, &op, b)?;
            Ok(Solved { x, g, stats, waited })
        }
    }
}

/// The solution of one level, still on the backend: u = x + g, with g the
/// Dirichlet values on the boundary nodes and 0.0 on interior ones.
pub(crate) struct LevelSolution<B: Backend> {
    pub(crate) grid: Grid,
    pub(crate) x: B::Vector,
    /// g on `grid.boundary_indices()`, in that order.
    pub(crate) g: Vec<f64>,
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

/// Node charges of a level ([`spread_charges_sparse_with`]): nodes in
/// increasing order and their charges.
type NodeCharges = (Vec<usize>, Vec<f64>);

/// What a level keeps besides its solution for the energy: the region χ of
/// the reaction-field decomposition, or the host solution of a nonlinear
/// level.
pub(crate) enum LevelExtra {
    None,
    Split(Split),
    Nonlinear(NonlinearLevel),
}

/// One PB solve on `grid` with Dirichlet values `g` (on
/// `grid.boundary_indices()`). Also returns the node charges the energy is
/// taken with ([`ChargeMethod::energy_method`]). With `decompose` it solves
/// for the reaction field v = u − χG instead of u ([`crate::coulomb`]) and
/// also returns the region χ. The solvated state of `equation = npbe` with
/// ions is solved by Newton's method ([`crate::nonlinear`]).
#[allow(clippy::too_many_arguments)]
fn solve_level<B: Backend>(
    be: &B,
    grid: Grid,
    g: Pending<Vec<f64>>,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
    decompose: bool,
    timings: &mut Timings,
) -> Result<(LevelSolution<B>, SolveStats, NodeCharges, LevelExtra)> {
    let ions = IonTerms::new(params);
    if let (Some(surf), Some(ions)) = (surf, ions) {
        return solve_nonlinear_level(be, grid, g, atoms, params, surf, &ions, timings);
    }
    let t = Instant::now();
    // Maps built from validated parameters are valid (ε > 0, κ̄² ≥ 0).
    let maps = maps_on(be, &grid, atoms, params, surf)?;
    be.synchronize()?;
    timings.maps += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let (split, (nodes, charges)) = if decompose {
        let split = Split::new(&grid, atoms);
        let source = split.source(be, &grid, &maps, atoms, params.pdie, lb0)?;
        (Some(split), source)
    } else {
        (None, spread_charges_sparse_with(&grid, atoms, params.chgm)?)
    };
    let opts = solver_options(params);
    let scatter = || be.scatter(grid.len(), &nodes, &charges);
    let solved = solve_assembled(be, &grid, &grid.boundary_indices(), g, maps, scatter, lb0, opts)?;
    let (x, g, mut stats) = (solved.x, solved.g, solved.stats);
    // The solver's buffers are freed at the next synchronisation (where no
    // calculation holds memory, the CUDA pool also returns them to the driver
    // then); that belongs to the solve.
    let freeing = Instant::now();
    be.synchronize()?;
    stats.solve_seconds += freeing.elapsed().as_secs_f64();
    let total = t.elapsed().as_secs_f64();
    timings.boundary += solved.waited;
    timings.setup += stats.setup_seconds;
    timings.solve += stats.solve_seconds;
    timings.assembly += (total - solved.waited - stats.setup_seconds - stats.solve_seconds).max(0.0);
    let energy_charges = match params.chgm.energy_method() {
        m if m == params.chgm && split.is_none() => (nodes, charges),
        m => spread_charges_sparse_with(&grid, atoms, m)?,
    };
    let extra = split.map_or(LevelExtra::None, LevelExtra::Split);
    Ok((LevelSolution { grid, x, g }, stats, energy_charges, extra))
}

/// The free-energy functional of a nonlinear finest level (kT), minus the
/// coarser levels' outer terms `external` (dielectric, ions); None for a
/// linear level.
fn nonlinear_energy(
    extra: &LevelExtra,
    params: &PbParams,
    lb0: f64,
    charges: (&[usize], &[f64]),
    external: [f64; 2],
) -> Option<f64> {
    match (extra, IonTerms::new(params)) {
        (LevelExtra::Nonlinear(level), Some(ions)) => {
            Some(level.energy(&ions, lb0, charges) - external[0] - external[1])
        }
        _ => None,
    }
}

fn solver_options(params: &PbParams) -> SolverOptions {
    SolverOptions {
        tol: params.tol,
        max_iter: params.max_iter,
        preconditioner: params.preconditioner,
    }
}

/// [`solve_level`] for the solvated state of the nonlinear equation: the
/// maps are built as for the linear equation and read back to the host,
/// where the Newton iteration evaluates the Boltzmann terms; the linear
/// steps run on `be`.
#[allow(clippy::too_many_arguments)]
fn solve_nonlinear_level<B: Backend>(
    be: &B,
    grid: Grid,
    g: Pending<Vec<f64>>,
    atoms: &[Atom],
    params: &PbParams,
    surf: &MolecularSurface,
    ions: &IonTerms,
    timings: &mut Timings,
) -> Result<(LevelSolution<B>, SolveStats, NodeCharges, LevelExtra)> {
    let t = Instant::now();
    let maps = {
        // Released before the Newton iteration allocates.
        let [eps_x, eps_y, eps_z, kappa_sq] = maps_on(be, &grid, atoms, params, Some(surf))?;
        CoefficientMaps {
            eps_x: be.download(&eps_x)?,
            eps_y: be.download(&eps_y)?,
            eps_z: be.download(&eps_z)?,
            kappa_sq: be.download(&kappa_sq)?,
        }
    };
    be.synchronize()?;
    timings.maps += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let g = g.wait()?;
    timings.boundary += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let (nodes, charges) = spread_charges_sparse_with(&grid, atoms, params.chgm)?;
    let (sol, mut stats, level) =
        nonlinear::solve_level(be, grid, g, maps, (&nodes, &charges), ions, lb0, solver_options(params))?;
    let freeing = Instant::now();
    be.synchronize()?;
    stats.solve_seconds += freeing.elapsed().as_secs_f64();
    let total = t.elapsed().as_secs_f64();
    timings.setup += stats.setup_seconds;
    timings.solve += stats.solve_seconds;
    timings.assembly += (total - stats.setup_seconds - stats.solve_seconds).max(0.0);
    let energy_charges = match params.chgm.energy_method() {
        m if m == params.chgm => (nodes, charges),
        m => spread_charges_sparse_with(&sol.grid, atoms, m)?,
    };
    Ok((sol, stats, energy_charges, LevelExtra::Nonlinear(level)))
}

/// Boundary values of `fine` focused from the solution of the coarser level,
/// as [`crate::boundary::focus_boundary`] computes them, reading only the coarse nodes the
/// interpolation uses ([`focus_nodes`]).
fn focus_from<B: Backend>(be: &B, fine: &Grid, coarse: &LevelSolution<B>) -> Result<Vec<f64>> {
    let nodes = focus_nodes(fine, &coarse.grid);
    let u = coarse.values_at(be, &nodes.indices())?;
    focus_boundary_values(fine, &coarse.grid, |c| u[nodes.position(c).expect("a focusing node")])
}

/// The host inputs of one state that do not depend on the backend: its
/// molecular surface (solvated state only) and the analytic boundary values
/// (condition `bcfl`) of its outermost level.
struct StateInputs {
    surf: Pending<Option<MolecularSurface>>,
    outer_boundary: Pending<Vec<f64>>,
    /// Time spent building them in the foreground.
    timings: Timings,
}

impl StateInputs {
    /// Builds the inputs of `medium` for the focusing `levels` now (surface
    /// first, boundary last), timing them as `maps` and `boundary`.
    fn build(levels: &[Grid], atoms: &[Atom], params: &PbParams, medium: Medium) -> Self {
        let mut timings = Timings::default();
        let t = Instant::now();
        let surf = surface_of(atoms, params, medium);
        timings.maps += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let outer_boundary = outer_boundary_of(levels, atoms, params, medium);
        timings.boundary += t.elapsed().as_secs_f64();
        Self {
            surf: Pending::Ready(surf),
            outer_boundary: Pending::Ready(outer_boundary),
            timings,
        }
    }
}

/// The molecular surface of the solvated state (None for the reference
/// state). It depends only on the atoms: built once for all levels.
fn surface_of(atoms: &[Atom], params: &PbParams, medium: Medium) -> Option<MolecularSurface> {
    (medium == Medium::Solvated).then(|| MolecularSurface::for_params(atoms, params))
}

/// The analytic boundary values of the outermost of the focusing `levels`.
fn outer_boundary_of(levels: &[Grid], atoms: &[Atom], params: &PbParams, medium: Medium) -> Vec<f64> {
    analytic_boundary_values(&levels[0], atoms, params.bcfl, &boundary_medium(params, medium))
}

/// Runs the focusing sequence `levels` (APBS mg-auto, see
/// [`GridSpec::levels`]) of the state whose `inputs` are given: the outermost
/// level takes their analytic boundary, every finer level takes its boundary
/// from the previous one, read from the backend at the nodes it needs.
/// With `decompose` the finest level solves for the reaction field
/// ([`solve_level`]). `finish` turns the finest level's solution, node
/// charges and region χ into the result. Time spent waiting for inputs from
/// the background counts as map time (the surface) and boundary time (the
/// boundary).
#[allow(clippy::type_complexity)]
fn focused<B: Backend, T>(
    be: &B,
    levels: &[Grid],
    atoms: &[Atom],
    params: &PbParams,
    inputs: StateInputs,
    decompose: bool,
    finish: impl FnOnce(LevelSolution<B>, NodeCharges, LevelExtra, [f64; 2], &mut Timings) -> Result<T>,
) -> Result<(T, Vec<SolveStats>, Timings)> {
    let StateInputs {
        surf,
        outer_boundary,
        mut timings,
    } = inputs;
    let t = Instant::now();
    let surf = surf.wait()?;
    timings.maps += t.elapsed().as_secs_f64();
    let mut outer_boundary = Some(outer_boundary);
    let mut stats = Vec::with_capacity(levels.len());
    let mut last: Option<(LevelSolution<B>, NodeCharges, LevelExtra)> = None;
    // The dielectric and ion terms of the nonlinear functional outside the
    // finer boxes (APBS `Vpmg_extEnergy`).
    let mut external = [0.0; 2];
    let ions = IonTerms::new(params);
    let lb0 = bjerrum_length_vacuum(params.temperature);
    for (index, grid) in levels.iter().enumerate() {
        let coarse = last.take();
        if let (Some((_, _, LevelExtra::Nonlinear(level))), Some(ions)) = (&coarse, &ions) {
            let t = Instant::now();
            let (diel, ion) = level.external_terms(grid, ions, lb0);
            external[0] += diel;
            external[1] += ion;
            timings.energy += t.elapsed().as_secs_f64();
        }
        let t = Instant::now();
        let g = match &coarse {
            None => outer_boundary.take().expect("the outermost level comes first"),
            // Coarser levels solve for u itself (only the finest decomposes).
            Some((coarse, _, _)) => Pending::Ready(focus_from(be, grid, coarse)?),
        };
        timings.boundary += t.elapsed().as_secs_f64();
        // The coarser level is released before this one allocates (freeing
        // memory belongs to no stage).
        drop(coarse);
        let finest = index + 1 == levels.len();
        let solved = solve_level(
            be,
            grid.clone(),
            g,
            atoms,
            params,
            surf.as_ref(),
            decompose && finest,
            &mut timings,
        )?;
        let (sol, s, charges, split) = solved;
        stats.push(s);
        last = Some((sol, charges, split));
    }
    let (sol, charges, extra) = last.expect("at least one level");
    let out = finish(sol, charges, extra, external, &mut timings)?;
    Ok((out, stats, timings))
}

/// Keeps the memory freed between the levels of one calculation on `spec`
/// for its next level, up to what [`crate::batch::device_workers`] sets aside
/// for the calculation.
fn hold_memory<'a, B: Backend>(be: &'a B, spec: &GridSpec, params: &PbParams) -> Result<MemoryHold<'a, B>> {
    MemoryHold::new(be, calculation_device_bytes([spec.dims], params))
}

/// A pool for host work in the background of a device calculation, with the
/// threads of the current pool but one, which stays free to drive the
/// device. A pool of its own, because a rayon thread that waits for its own
/// tasks runs queued ones meanwhile, and tasks sent to a busy pool queue
/// behind its work: in one pool the device would wait for background tasks.
/// None if the threads cannot be started (the work then runs in turn).
fn background_pool() -> Option<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads().saturating_sub(1).max(1))
        .thread_name(|i| format!("pbsolve-background-{i}"))
        .build()
        .ok()
}

/// The pool for host work in the background of `be`'s calculations, if it
/// runs on a device ([`Backend::runs_on_device`]).
fn background_pool_for<B: Backend>(be: &B) -> Option<rayon::ThreadPool> {
    be.runs_on_device().then(background_pool).flatten()
}

/// Builds the host inputs of the state `medium` in `pool` (its surface, then
/// the boundary of its outermost level) while `run` uses them on this
/// thread. The device thus assembles the maps, the operator and the
/// preconditioner of the outermost level while its boundary is still being
/// built. The background stops before the boundary if `run` no longer waits
/// (it returned, for example with an error).
fn with_background_inputs<R>(
    pool: &rayon::ThreadPool,
    levels: &[Grid],
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
    run: impl FnOnce(StateInputs) -> R,
) -> R {
    let (surf_tx, surf_rx) = mpsc::channel();
    let (boundary_tx, boundary_rx) = mpsc::channel();
    let inputs = StateInputs {
        surf: Pending::Later(surf_rx),
        outer_boundary: Pending::Later(boundary_rx),
        timings: Timings::default(),
    };
    std::thread::scope(|s| {
        s.spawn(|| {
            pool.install(|| {
                if surf_tx.send(surface_of(atoms, params, medium)).is_ok() {
                    // An error here means `run` returned: nobody needs the values.
                    let _ = boundary_tx.send(outer_boundary_of(levels, atoms, params, medium));
                }
            })
        });
        run(inputs)
    })
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
    let hold = hold_memory(be, spec, params)?;
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let field =
        |sol: LevelSolution<B>, (nodes, charges): NodeCharges, extra, external: [f64; 2], timings: &mut Timings| {
            let t = Instant::now();
            let energy = nonlinear_energy(&extra, params, lb0, (&nodes, &charges), external);
            timings.energy += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let field = sol.potential(be)?;
            timings.solve += t.elapsed().as_secs_f64();
            Ok((field, energy))
        };
    let (((grid, potential), energy_kt), stats, timings) = match background_pool_for(be) {
        Some(pool) => with_background_inputs(&pool, &levels, atoms, params, medium, |inputs| {
            focused(be, &levels, atoms, params, inputs, false, field)
        })?,
        None => focused(
            be,
            &levels,
            atoms,
            params,
            StateInputs::build(&levels, atoms, params, medium),
            false,
            field,
        )?,
    };
    hold.release()?;
    Ok(FieldSolution {
        grid,
        potential,
        stats,
        timings,
        energy_kt,
    })
}

/// ½ Σ q_i u(r_i) in kT, with the weights of `chgm`'s
/// [`ChargeMethod::energy_method`] (as APBS `calcenergy total`: trilinear
/// interpolation for `spl0` and `spl4`, the `spl2` weights for `spl2`).
pub fn electrostatic_energy_kt(sol: &FieldSolution, atoms: &[Atom], chgm: ChargeMethod) -> Result<f64> {
    let q = spread_charges_with(&sol.grid, atoms, chgm.energy_method())?;
    Ok(0.5 * q.iter().zip(&sol.potential).map(|(a, b)| a * b).sum::<f64>())
}

/// [`electrostatic_energy_kt`] of the finest level from u at the charged
/// nodes only: the same products in the same order without the zero terms,
/// hence the same value (a total of exactly zero may differ in its sign).
fn energy_at_charges<B: Backend>(be: &B, sol: &LevelSolution<B>, (nodes, charges): NodeCharges) -> Result<f64> {
    let u = sol.values_at(be, &nodes)?;
    Ok(0.5 * charges.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>())
}

/// Polar (electrostatic) solvation free energy on the given grid. With
/// [`CoulombMethod::Grid`] the reference state is solved first; on a device
/// backend the host builds the solvated state's surface and outermost
/// boundary in the background. With [`CoulombMethod::Analytic`] only the
/// solvated state is solved, its finest level for the reaction field
/// ([`crate::coulomb`]).
pub fn polar_solvation<B: Backend>(
    be: &B,
    atoms: &[Atom],
    params: &PbParams,
    spec: &GridSpec,
) -> Result<PolarSolvation> {
    let lb0 = bjerrum_length_vacuum(params.temperature);
    let energy =
        |sol: LevelSolution<B>, (nodes, charges): NodeCharges, extra, external: [f64; 2], timings: &mut Timings| {
            let t = Instant::now();
            let e = match nonlinear_energy(&extra, params, lb0, (&nodes, &charges), external) {
                Some(e) => e,
                None => energy_at_charges(be, &sol, (nodes, charges))?,
            };
            timings.energy += t.elapsed().as_secs_f64();
            Ok(e)
        };
    params.validate()?;
    let levels = spec.levels()?;
    let hold = hold_memory(be, spec, params)?;
    let result = |solvated_kt: f64, reference_kt: f64, stats_solvated, stats_reference, timings| {
        let energy_kt = solvated_kt - reference_kt;
        PolarSolvation {
            energy_kcal: energy_kt * kt_in_kcal_per_mol(params.temperature),
            energy_kt,
            solvated_kt,
            reference_kt,
            stats_solvated,
            stats_reference,
            grid: spec.clone(),
            timings,
        }
    };
    if params.coulomb == CoulombMethod::Analytic {
        let reaction = |sol: LevelSolution<B>, (nodes, charges): NodeCharges, extra, _, timings: &mut Timings| {
            let t = Instant::now();
            let v = sol.values_at(be, &nodes)?;
            let LevelExtra::Split(split) = extra else {
                unreachable!("the finest level is decomposed")
            };
            let e = split.reaction_energy(&sol.grid, atoms, params.pdie, lb0, (&nodes, &charges), &v);
            timings.energy += t.elapsed().as_secs_f64();
            Ok(e)
        };
        let run = |inputs| focused(be, &levels, atoms, params, inputs, true, reaction);
        let (energy_kt, stats, mut timings) = match background_pool_for(be) {
            Some(pool) => with_background_inputs(&pool, &levels, atoms, params, Medium::Solvated, run)?,
            None => run(StateInputs::build(&levels, atoms, params, Medium::Solvated))?,
        };
        hold.release()?;
        // The single states without self-energies: the Coulomb energy in pdie,
        // and that plus the reaction-field energy.
        let t = Instant::now();
        let coulomb_kt = coulomb_energy_kt(atoms, params.pdie, lb0);
        timings.energy += t.elapsed().as_secs_f64();
        return Ok(result(coulomb_kt + energy_kt, coulomb_kt, stats, Vec::new(), timings));
    }
    // The states are independent, so their order does not change any result.
    // The reference state, whose inputs are cheap (no surface, no `exp` in
    // its boundary), is solved first. On a device backend the host builds
    // the solvated state's surface and outermost boundary (with `mdh` the
    // largest host stage) in the background meanwhile. The reference state's
    // boundary is built before, with all threads, while the device has
    // nothing to do: built in the background too, it would compete with the
    // solvated inputs for the same threads.
    let reference = StateInputs::build(&levels, atoms, params, Medium::Reference);
    let ((reference_kt, stats_reference, ref_timings), (solvated_kt, stats_solvated, mut timings)) =
        match background_pool_for(be) {
            Some(pool) => with_background_inputs(
                &pool,
                &levels,
                atoms,
                params,
                Medium::Solvated,
                |solvated| -> Result<_> {
                    let reference = focused(be, &levels, atoms, params, reference, false, energy)?;
                    Ok((reference, focused(be, &levels, atoms, params, solvated, false, energy)?))
                },
            )?,
            None => {
                let reference = focused(be, &levels, atoms, params, reference, false, energy)?;
                let solvated = StateInputs::build(&levels, atoms, params, Medium::Solvated);
                (reference, focused(be, &levels, atoms, params, solvated, false, energy)?)
            }
        };
    timings += ref_timings;
    hold.release()?;
    Ok(result(
        solvated_kt,
        reference_kt,
        stats_solvated,
        stats_reference,
        timings,
    ))
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
            let q = spread_charges_with(&grid, atoms, params.chgm).unwrap();
            let (u, _) = solve_with_maps(&be, &grid, maps, &q, &g, params.temperature, opts).unwrap();
            prev = Some((grid, u));
        }
        prev.unwrap()
    }

    /// Energies taken from u at the charged nodes, with the focusing
    /// boundaries read at the stencil nodes only, equal the energies of the
    /// full potentials bit for bit; so do the potentials of `solve_focused`
    /// and of the host references. One to three focusing levels, every
    /// boundary condition, preconditioner and charge method.
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
            (
                &two,
                PbParams {
                    chgm: ChargeMethod::Spline2,
                    ..salt.clone()
                },
            ),
            (
                &three,
                PbParams {
                    chgm: ChargeMethod::Spline4,
                    ..salt.clone()
                },
            ),
        ];
        let mut levels = Vec::new();
        for (spec, params) in cases {
            let be = CpuBackend::new();
            let r = polar_solvation(&be, &atoms, &params, spec).unwrap();
            let label = format!(
                "{:?} {:?} {:?}, {} levels",
                params.bcfl,
                params.preconditioner,
                params.chgm,
                r.stats_solvated.len()
            );
            for (medium, kt) in [(Medium::Solvated, r.solvated_kt), (Medium::Reference, r.reference_kt)] {
                let field = solve_focused(&be, spec, &atoms, &params, medium).unwrap();
                let full = electrostatic_energy_kt(&field, &atoms, params.chgm).unwrap();
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
        assert_eq!(levels, [2, 3, 1, 2, 2, 2, 3, 2, 3]);
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

    /// Boundary values and where to send them.
    type Gate = Option<(mpsc::Sender<Vec<f64>>, Vec<f64>)>;

    /// Delegates to the CPU backend; on its first Galerkin product (the
    /// multigrid setup) it sends the boundary values through `gate`.
    struct GatedSetup {
        cpu: CpuBackend,
        gate: std::sync::Mutex<Gate>,
    }

    macro_rules! delegate {
        ($(fn $name:ident(&self $(, $a:ident: $t:ty)*) -> $r:ty;)*) => {$(
            fn $name(&self $(, $a: $t)*) -> $r {
                self.cpu.$name($($a),*)
            }
        )*};
    }

    impl Backend for GatedSetup {
        type Vector = <CpuBackend as Backend>::Vector;
        type Operator = <CpuBackend as Backend>::Operator;
        type Coarse = <CpuBackend as Backend>::Coarse;
        type Single = crate::cpu::CpuSingle;

        delegate! {
            fn platform(&self) -> &'static str;
            fn device_name(&self) -> String;
            fn single(&self) -> Result<crate::cpu::CpuSingle>;
            fn demote(&self, x: &Self::Vector, y: &mut <crate::cpu::CpuSingle as Backend>::Vector) -> Result<()>;
            fn promote(&self, x: &<crate::cpu::CpuSingle as Backend>::Vector, y: &mut Self::Vector) -> Result<()>;
            fn prepare(&self, op: std::sync::Arc<crate::operator::StencilOperator>) -> Result<Self::Operator>;
            fn prepare_coarse(&self, op: std::sync::Arc<crate::multigrid::Stencil27>) -> Result<Self::Coarse>;
            fn upload(&self, host: &[f64]) -> Result<Self::Vector>;
            fn zeros(&self, n: usize) -> Result<Self::Vector>;
            fn download(&self, v: &Self::Vector) -> Result<Vec<f64>>;
            fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()>;
            fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()>;
            fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64>;
            fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
            fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
            fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn restrict(&self, t: &crate::multigrid::Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()>;
            fn prolong_add(&self, t: &crate::multigrid::Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()>;
            fn coarse_to_host(&self, op: &Self::Coarse) -> Result<crate::multigrid::Stencil27>;
            fn demote_operator(&self, op: &Self::Operator) -> Result<<crate::cpu::CpuSingle as Backend>::Operator>;
            fn demote_coarse(&self, op: &Self::Coarse) -> Result<<crate::cpu::CpuSingle as Backend>::Coarse>;
            fn scatter(&self, n: usize, idx: &[usize], values: &[f64]) -> Result<Self::Vector>;
            fn assemble_operator(&self, grid: &Grid, maps: [Self::Vector; 4]) -> Result<(Self::Operator, Self::Vector)>;
            fn source_rhs(&self, src: f64, q: &Self::Vector, d: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
            fn operator_to_host(&self, op: &Self::Operator) -> Result<crate::operator::StencilOperator>;
        }

        fn mark_maps(
            &self,
            grid: &Grid,
            shifts: &[[f64; 3]],
            fill: f64,
            passes: &[(&[crate::surface::Sphere], f64)],
        ) -> Result<Vec<Self::Vector>> {
            self.cpu.mark_maps(grid, shifts, fill, passes)
        }

        fn galerkin(
            &self,
            t: &crate::multigrid::Transfer,
            fine: crate::multigrid::LevelOp<'_, Self>,
        ) -> Result<(Self::Coarse, Self::Vector)> {
            if let Some((tx, g)) = self.gate.lock().unwrap().take() {
                tx.send(g).unwrap();
            }
            self.cpu.galerkin(t, lower(fine))
        }

        fn gershgorin_lmax(&self, level: crate::multigrid::LevelOp<'_, Self>) -> Result<f64> {
            self.cpu.gershgorin_lmax(lower(level))
        }
    }

    fn lower(op: crate::multigrid::LevelOp<'_, GatedSetup>) -> crate::multigrid::LevelOp<'_, CpuBackend> {
        match op {
            crate::multigrid::LevelOp::Fine(o) => crate::multigrid::LevelOp::Fine(o),
            crate::multigrid::LevelOp::Coarse(c) => crate::multigrid::LevelOp::Coarse(c),
        }
    }

    /// The problem of [`pending_boundary_values_are_waited_for_after_the_setup`].
    struct GateCase {
        atoms: Vec<Atom>,
        params: PbParams,
        grid: Grid,
        g: Vec<f64>,
        charges: NodeCharges,
    }

    /// Solves the outermost level of `case` with boundary values that are
    /// ready or (`pending`) arrive at the first Galerkin product.
    fn solve_gated(case: &GateCase, pc: Preconditioner, pending: bool) -> (Vec<f64>, usize) {
        let (tx, rx) = mpsc::channel();
        let be = GatedSetup {
            cpu: CpuBackend::new(),
            gate: std::sync::Mutex::new(pending.then(|| (tx, case.g.clone()))),
        };
        let g = if pending {
            Pending::Later(rx)
        } else {
            Pending::Ready(case.g.clone())
        };
        let maps = CoefficientMaps::solvated(&case.grid, &case.atoms, &case.params);
        let maps = [maps.eps_x, maps.eps_y, maps.eps_z, maps.kappa_sq].map(|m| be.upload(&m).unwrap());
        let (nodes, charges) = &case.charges;
        let scatter = || be.scatter(case.grid.len(), nodes, charges);
        let opts = SolverOptions {
            preconditioner: pc,
            ..SolverOptions::new(case.params.tol, case.params.max_iter)
        };
        let lb0 = bjerrum_length_vacuum(case.params.temperature);
        let boundary = case.grid.boundary_indices();
        let solved = solve_assembled(&be, &case.grid, &boundary, g, maps, scatter, lb0, opts).unwrap();
        (be.download(&solved.x).unwrap(), solved.stats.iterations)
    }

    /// Boundary values still being built in the background are waited for
    /// only after the multigrid setup: here they arrive during the setup (a
    /// wait before it would never end), and the solution equals the one
    /// from values that were ready, bit for bit.
    #[test]
    fn pending_boundary_values_are_waited_for_after_the_setup() {
        let atoms = small_protein();
        let params = PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        let grid = GridSpec::auto(&atoms, 0.5, 1.7, 12.0)
            .unwrap()
            .levels()
            .unwrap()
            .remove(0);
        let g = analytic_boundary_values(&grid, &atoms, params.bcfl, &boundary_medium(&params, Medium::Solvated));
        let charges = crate::charge::spread_charges_sparse(&grid, &atoms).unwrap();
        let case = std::sync::Arc::new(GateCase {
            atoms,
            params,
            grid,
            g,
            charges,
        });
        for pc in [Preconditioner::MultigridF32, Preconditioner::Multigrid] {
            let ready = solve_gated(&case, pc, false);
            // In a thread of its own: a wait before the setup would hang.
            let (done_tx, done_rx) = mpsc::channel();
            let shared = case.clone();
            std::thread::spawn(move || done_tx.send(solve_gated(&shared, pc, true)).unwrap());
            let pending = done_rx
                .recv_timeout(std::time::Duration::from_secs(120))
                .expect("the boundary values were waited for before the setup");
            assert!(ready.1 > 0);
            assert_eq!(pending.1, ready.1, "{pc:?}");
            assert!(
                pending.0.iter().zip(&ready.0).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{pc:?}"
            );
        }
    }
}
