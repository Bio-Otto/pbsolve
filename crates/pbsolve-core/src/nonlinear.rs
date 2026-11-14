//! Nonlinear Poisson–Boltzmann equation ([`Equation::Nonlinear`], APBS
//! `npbe`).
//!
//! On the interior nodes of a level, with the volume scaling of
//! [`crate::operator`] (APBS `Vc_vecpmg`):
//!
//! ```text
//! F(u) = A_ε u + V N(u) − 4π l_B0 q = 0
//! N(u)  = κ̄²_c Σ_j −(c_j q_j / 2I) exp(−q_j u)
//! N'(u) = κ̄²_c Σ_j  (c_j q_j² / 2I) exp(−q_j u)
//! ```
//!
//! A_ε is the operator without its ion term, κ̄²_c the ion map, c_j and q_j
//! the concentrations and valences, I the ionic strength; for a 1:1 salt
//! N = κ̄² sinh u. The exponents are clamped to ±85 as in APBS. Newton's
//! method solves J δ = −F with J = A_ε + V N'(u), which is the linear
//! operator with κ̄² replaced by N'(u), so the linear solver (multigrid CG,
//! CPU or CUDA) is reused unchanged; steps are halved until ‖F‖ decreases
//! (Armijo). The iteration starts from the linearized solution, solved to
//! 1e-3. It is inexact [Holst & Saied 1995]: each step solves to
//! η = min(0.1, ‖F‖/‖b‖), and not tighter than the final target needs.
//!
//! The energy is APBS' free-energy functional (`Vpmg_energy` for `npbe`
//! with ions) [Sharp & Honig 1990]:
//!
//! ```text
//! G = Σ q u − ½ Σ_faces ε (Δu)² c_a / 4π l_B0 − V Σ_nodes κ̄²_c Σ_j (c_j / 2I)(e^{−q_j u} − 1) / 4π l_B0
//! ```
//!
//! The dielectric and ion terms of every coarser focusing level are added
//! for its nodes outside the next finer box (APBS `Vpmg_extEnergy`).
//! [`Equation`]: crate::params::Equation
//! [`Equation::Nonlinear`]: crate::params::Equation::Nonlinear

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::grid::Grid;
use crate::operator::StencilOperator;
use crate::params::PbParams;
use crate::pb::{solve_assembled, LevelSolution, Pending};
use crate::solver::{pcg_on, SolveStats, SolverOptions};
use crate::surface::CoefficientMaps;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

/// APBS' limit on the exponent of the Boltzmann factors (`SINH_MAX`, `EXPMAX`).
const EXP_LIMIT: f64 = 85.0;
/// Newton iterations allowed per level.
const MAX_NEWTON: usize = 100;
/// Tolerance of the linearized starting solution.
const START_TOL: f64 = 1e-3;
/// Largest relative tolerance of a Newton step's linear solve.
const MAX_FORCING: f64 = 0.1;
/// APBS `VSMALL` and `VPMGSMALL` (partition weights of `Vpmg_extEnergy`).
const VSMALL: f64 = 1.0e-9;
const VPMGSMALL: f64 = 1.0e-12;

fn boltzmann(x: f64) -> f64 {
    x.clamp(-EXP_LIMIT, EXP_LIMIT).exp()
}

/// Whether node `c` of a grid of `dims` is interior (not on the boundary).
#[inline]
pub(crate) fn is_interior([nx, ny, nz]: [usize; 3], c: usize) -> bool {
    let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
    i > 0 && j > 0 && k > 0 && i + 1 < nx && j + 1 < ny && k + 1 < nz
}

/// N'(u) at one node: κ̄² Σ_j a_j q_j² e^{−q_j u} for `ions` = [(q_j, a_j =
/// c_j / 2I)] ([`crate::backend::Backend::boltzmann_slope`]; the CUDA
/// kernel `pb_boltzmann_slope` evaluates the same expressions).
#[inline]
pub(crate) fn slope_at(ions: &[[f64; 2]], u: f64, kappa_sq: f64) -> f64 {
    if kappa_sq > 0.0 {
        let mut s = 0.0;
        for &[q, a] in ions {
            s += a * q * q * boltzmann(-q * u);
        }
        kappa_sq * s
    } else {
        0.0
    }
}

/// The residual at one node from y = (J u) there, J assembled with κ̄² =
/// `slope`: 0 on the boundary, else y + V (N(u) − slope·u) − f
/// ([`crate::backend::Backend::boltzmann_residual`]; CUDA
/// `pb_boltzmann_residual`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn residual_at(
    ions: &[[f64; 2]],
    vol: f64,
    u: f64,
    kappa_sq: f64,
    slope: f64,
    f: f64,
    interior: bool,
    y: f64,
) -> f64 {
    if !interior {
        return 0.0;
    }
    let mut n = 0.0;
    if kappa_sq > 0.0 {
        for &[q, a] in ions {
            n -= a * q * boltzmann(-q * u);
        }
        n *= kappa_sq;
    }
    y + vol * (n - slope * u) - f
}

/// The mobile ions as the nonlinear term needs them: (q_j, c_j / 2I).
#[derive(Clone, Debug)]
pub struct IonTerms {
    species: Vec<[f64; 2]>,
}

impl IonTerms {
    /// The species as (q_j, c_j / 2I).
    pub fn species(&self) -> &[[f64; 2]] {
        &self.species
    }

    /// None without ions (zero ionic strength): the equation is then linear.
    pub fn new(params: &PbParams) -> Option<Self> {
        let two_i = 2.0 * params.ionic_strength();
        (two_i > 0.0).then(|| Self {
            species: params
                .ions
                .iter()
                .filter(|i| i.concentration > 0.0)
                .map(|i| [i.charge, i.concentration / two_i])
                .collect(),
        })
    }

    /// Σ_j (c_j / 2I)(e^{−q_j u} − 1), the ion energy density / κ̄².
    #[inline]
    fn density(&self, u: f64) -> f64 {
        self.species.iter().map(|&[q, a]| a * (boltzmann(-q * u) - 1.0)).sum()
    }
}

/// The solution of one nonlinear level on the host, with its maps: what the
/// energy functional needs.
pub(crate) struct NonlinearLevel {
    grid: Grid,
    maps: CoefficientMaps,
    u: Vec<f64>,
}

impl NonlinearLevel {
    /// The dielectric and ion terms of the functional (kT) with node weights
    /// `weight` (faces: the mean of their two nodes), over APBS' loop
    /// `i < nx−1, j < ny−1, k < nz−1` for all three faces of a node.
    fn energy_terms(&self, ions: &IonTerms, lb0: f64, weight: impl Fn(usize) -> f64 + Sync) -> (f64, f64) {
        let [nx, ny, nz] = self.grid.dims;
        let coef = StencilOperator::face_coefficients(&self.grid);
        let vol = self.grid.cell_volume();
        let (u, m) = (&self.u, &self.maps);
        let eps = [&m.eps_x, &m.eps_y, &m.eps_z];
        let strides = [1, nx, nx * ny];
        let diel: f64 = (0..nz - 1)
            .into_par_iter()
            .map(|k| {
                let mut e = 0.0;
                for j in 0..ny - 1 {
                    for i in 0..nx - 1 {
                        let c = i + nx * (j + ny * k);
                        for a in 0..3 {
                            let nb = c + strides[a];
                            let w = 0.5 * (weight(c) + weight(nb));
                            let d = u[c] - u[nb];
                            e += coef[a] * w * eps[a][c] * d * d;
                        }
                    }
                }
                e
            })
            .sum();
        let ion: f64 = (0..u.len())
            .into_par_iter()
            .map(|c| {
                let k2 = m.kappa_sq[c] * weight(c);
                if k2 > 0.0 {
                    k2 * ions.density(u[c])
                } else {
                    0.0
                }
            })
            .sum();
        let scale = 1.0 / (4.0 * PI * lb0);
        (0.5 * diel * scale, vol * ion * scale)
    }

    /// The functional on this (the finest) level: Σ q u at the energy's node
    /// charges minus its dielectric and ion terms (kT).
    pub fn energy(&self, ions: &IonTerms, lb0: f64, (nodes, charges): (&[usize], &[f64])) -> f64 {
        let qf: f64 = nodes.iter().zip(charges).map(|(&c, q)| q * self.u[c]).sum();
        let (diel, ion) = self.energy_terms(ions, lb0, |_| 1.0);
        qf - diel - ion
    }

    /// The dielectric and ion terms of this level outside the box of the
    /// finer level `fine` (APBS `Vpmg_extEnergy` with `Vpmg_setPart`): nodes
    /// count with the complement of their share of the box.
    pub fn external_terms(&self, fine: &Grid, ions: &IonTerms, lb0: f64) -> (f64, f64) {
        let g = &self.grid;
        let lo = fine.origin;
        let hi: [f64; 3] = [0, 1, 2].map(|a| fine.origin[a] + (fine.dims[a] - 1) as f64 * fine.spacing[a]);
        let axes: Vec<Vec<f64>> = (0..3)
            .map(|a| {
                (0..g.dims[a])
                    .map(|i| box_share(g.origin[a] + i as f64 * g.spacing[a], g.spacing[a], lo[a], hi[a]))
                    .collect()
            })
            .collect();
        let weight = |c: usize| {
            let (i, j, k) = g.ijk(c);
            let mut p = axes[0][i] * axes[1][j] * axes[2][k];
            if p.abs() < VPMGSMALL {
                p = 0.0;
            }
            if p > VSMALL {
                p = 1.0;
            }
            1.0 - p
        };
        self.energy_terms(ions, lb0, weight)
    }
}

/// The share of a node at coordinate `x` (spacing `h`) in the interval
/// [lo, hi] of a finer box (APBS `Vpmg_setPart` without neighbouring
/// processors): 1 inside, the overlap of [x − h/2, x + h/2] near the ends.
fn box_share(x: f64, h: f64, lo: f64, hi: f64) -> f64 {
    if (x < hi - h / 2.0 && x > lo + h / 2.0) || (x - lo).abs() < VPMGSMALL || (x - hi).abs() < VPMGSMALL {
        1.0
    } else if x > hi + h / 2.0 || x < lo - h / 2.0 {
        0.0
    } else {
        let x0 = (x - h / 2.0).max(lo);
        let x1 = (x + h / 2.0).min(hi);
        ((x1 - x0).abs() / h).min(1.0)
    }
}

/// The Jacobian of a Newton step on a backend: the operator, its inverse
/// diagonal (moved into the linear solver) and N'(u).
type Jacobian<B> = (
    <B as Backend>::Operator,
    Option<<B as Backend>::Vector>,
    <B as Backend>::Vector,
);

/// Solves one level of the nonlinear equation with Dirichlet values `g` (on
/// `grid.boundary_indices()`), host maps `maps` and node charges (e).
/// Returns the solution on `be` (as the linear levels do), the solver
/// statistics (CG iterations summed over the Newton steps, the final
/// nonlinear residual ‖F‖/‖b‖) and the host solution for the energy.
///
/// The iteration runs on `be`: the Boltzmann terms
/// ([`Backend::boltzmann_slope`], [`Backend::boltzmann_residual`]), the
/// Jacobian and its multigrid CG. The residual of a trial u is J u completed
/// by the ion terms, with J the Jacobian of the current step, so the line
/// search needs no further operator.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_level<B: Backend>(
    be: &B,
    grid: Grid,
    g: Vec<f64>,
    maps: CoefficientMaps,
    (nodes, charges): (&[usize], &[f64]),
    ions: &IonTerms,
    lb0: f64,
    opts: SolverOptions,
) -> Result<(LevelSolution<B>, SolveStats, NonlinearLevel)> {
    let n = grid.len();
    let boundary = grid.boundary_indices();
    let vol = grid.cell_volume();
    let src = 4.0 * PI * lb0;
    let species = ions.species();
    let upload_eps = || -> Result<[B::Vector; 3]> {
        Ok([
            be.upload(&maps.eps_x)?,
            be.upload(&maps.eps_y)?,
            be.upload(&maps.eps_z)?,
        ])
    };
    let mut stats = SolveStats {
        iterations: 0,
        relative_residual: 0.0,
        preconditioner: opts.preconditioner,
        levels: 1,
        setup_seconds: 0.0,
        solve_seconds: 0.0,
    };
    let mut add = |s: SolveStats| {
        stats.iterations += s.iterations;
        stats.preconditioner = s.preconditioner;
        stats.levels = s.levels;
        stats.setup_seconds += s.setup_seconds;
        stats.solve_seconds += s.solve_seconds;
    };

    // The linearized solution as the starting point.
    let [ex, ey, ez] = upload_eps()?;
    let maps_lin = [ex, ey, ez, be.upload(&maps.kappa_sq)?];
    let scatter = || be.scatter(n, nodes, charges);
    let start = SolverOptions {
        tol: opts.tol.max(START_TOL),
        ..opts
    };
    let solved = solve_assembled(
        be,
        &grid,
        &boundary,
        Pending::Ready(g.clone()),
        maps_lin,
        scatter,
        lb0,
        start,
    )?;
    let mut u = be.download(&solved.x)?;
    drop(solved.x);
    add(solved.stats);
    let t = Instant::now();
    for (&c, v) in boundary.iter().zip(&g) {
        u[c] = *v;
    }
    // Where the ions are, the linearized potential overshoots once |u| > 1:
    // start from the planar Gouy–Chapman relation u = 2 asinh(u_lin / 2)
    // there (Grahame), shifted to leave |u| ≤ 1 unchanged.
    let mut interior = vec![true; n];
    for &c in &boundary {
        interior[c] = false;
    }
    u.par_iter_mut().enumerate().for_each(|(c, u)| {
        if interior[c] && maps.kappa_sq[c] > 0.0 && u.abs() > 1.0 {
            *u = u.signum() * (1.0 + 2.0 * ((u.abs() - 1.0) / 2.0).asinh());
        }
    });
    drop(interior);
    let mut u = be.upload_owned(u)?;
    let kappa = be.upload(&maps.kappa_sq)?;
    let scaled: Vec<f64> = charges.iter().map(|q| src * q).collect();
    let f = be.scatter(n, nodes, &scaled)?;

    // The Jacobian at u: the operator with κ̄² = N'(u), its inverse
    // diagonal and N'(u).
    let jacobian = |u: &B::Vector| -> Result<Jacobian<B>> {
        let mut slope = be.zeros(n)?;
        be.boltzmann_slope(species, u, &kappa, &mut slope)?;
        let mut ion_map = be.zeros(n)?;
        be.copy(&slope, &mut ion_map)?;
        let [ex, ey, ez] = upload_eps()?;
        let (op, inv_diag) = be.assemble_operator(&grid, [ex, ey, ez, ion_map])?;
        Ok((op, Some(inv_diag), slope))
    };
    // r = F(u) = J u + V (N(u) − N'_J u) − f on the interior; returns ‖F‖.
    let residual = |op: &B::Operator, slope: &B::Vector, u: &B::Vector, r: &mut B::Vector| -> Result<f64> {
        be.apply(op, u, r)?;
        be.boltzmann_residual(species, vol, grid.dims, u, &kappa, slope, &f, r)?;
        Ok(be.dot(r, r)?.sqrt())
    };
    let (mut op, mut inv_diag, mut slope) = jacobian(&u)?;
    // ‖b‖ of the linear system, f − A g: J g has no ion term inside (g is 0 there).
    let b_norm = {
        let gv = be.scatter(n, &boundary, &g)?;
        let mut b = be.zeros(n)?;
        be.apply(&op, &gv, &mut b)?;
        be.source_rhs(1.0, &f, inv_diag.as_ref().expect("a fresh Jacobian"), &mut b)?;
        be.dot(&b, &b)?.sqrt().max(f64::MIN_POSITIVE)
    };
    let mut r = Some(be.zeros(n)?);
    let mut norm = residual(&op, &slope, &u, r.as_mut().expect("a residual"))?;
    let mut host_seconds = t.elapsed().as_secs_f64();
    let mut steps = 0;
    while norm > opts.tol * b_norm {
        if steps == MAX_NEWTON {
            return Err(PbError::NotConverged {
                iterations: steps,
                residual: norm / b_norm,
            });
        }
        // J δ = F (the Newton step is −δ), inexact: ‖J δ − F‖ ≤ η ‖F‖. The
        // residual and the inverse diagonal move into the solver.
        let relative = norm / b_norm;
        let step = SolverOptions {
            tol: relative
                .min(MAX_FORCING)
                .max(0.5 * opts.tol / relative)
                .min(MAX_FORCING),
            ..opts
        };
        be.synchronize()?;
        let d = inv_diag.take().expect("a fresh Jacobian");
        let (delta, s) = pcg_on(be, grid.dims, &op, d, r.take().expect("a residual"), step)?;
        add(s);
        // The solver's buffers return to the pool at a synchronisation.
        be.synchronize()?;
        let mut buf = be.zeros(n)?;
        // Backtracking until ‖F‖ decreases (Armijo, c = 1e-4), u updated in place.
        let t = Instant::now();
        let (mut lambda, mut taken) = (1.0, 0.0);
        loop {
            be.axpy(-(lambda - taken), &delta, &mut u)?;
            taken = lambda;
            let trial = residual(&op, &slope, &u, &mut buf)?;
            if trial <= (1.0 - 1e-4 * lambda) * norm || lambda < 1.0 / 1024.0 {
                norm = trial;
                break;
            }
            lambda *= 0.5;
        }
        drop(delta);
        r = Some(buf);
        steps += 1;
        if norm > opts.tol * b_norm {
            drop((op, slope));
            be.synchronize()?;
            (op, inv_diag, slope) = jacobian(&u)?;
        }
        be.synchronize()?;
        host_seconds += t.elapsed().as_secs_f64();
    }
    drop((op, inv_diag, slope, r));
    stats.relative_residual = norm / b_norm;
    stats.solve_seconds += host_seconds;

    // The host copy for the energy; on `be` u − g (zero on the boundary).
    let t = Instant::now();
    let u_host = be.download(&u)?;
    let gv = be.scatter(n, &boundary, &g)?;
    be.axpy(-1.0, &gv, &mut u)?;
    drop(gv);
    stats.solve_seconds += t.elapsed().as_secs_f64();
    let sol = LevelSolution {
        grid: grid.clone(),
        x: u,
        g,
    };
    Ok((sol, stats, NonlinearLevel { grid, maps, u: u_host }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autogrid::GridSpec;
    use crate::charge::spread_charges_sparse_with;
    use crate::cpu::CpuBackend;
    use crate::molecule::Atom;
    use crate::params::{ChargeMethod, Equation, IonSpecies, Preconditioner};
    use crate::pb::polar_solvation;
    use crate::units::bjerrum_length_vacuum;

    #[test]
    fn a_1_1_salt_gives_sinh_and_cosh() {
        let params = PbParams::default().with_salt(0.15, 2.0);
        let ions = IonTerms::new(&params).unwrap();
        // N(u) with κ̄² = 1: the residual of y = 0, slope 0, f = 0, V = 1.
        let term = |ions: &IonTerms, u: f64| {
            let sp = ions.species();
            (residual_at(sp, 1.0, u, 1.0, 0.0, 0.0, true, 0.0), slope_at(sp, u, 1.0))
        };
        for u in [-3.0, -0.2, 0.0, 0.7, 5.0] {
            let (n, dn) = term(&ions, u);
            assert!((n - f64::sinh(u)).abs() < 1e-12 * f64::cosh(u), "{u}: {n}");
            assert!((dn - f64::cosh(u)).abs() < 1e-12 * f64::cosh(u), "{u}: {dn}");
            assert!((ions.density(u) - (f64::cosh(u) - 1.0)).abs() < 1e-12 * f64::cosh(u));
        }
        // Clamped as in APBS.
        assert_eq!(term(&ions, 200.0).1, f64::cosh(85.0));
        // The residual's mask: 0 on the boundary.
        assert_eq!(residual_at(ions.species(), 1.0, 2.0, 1.0, 3.0, 4.0, false, 5.0), 0.0);
        // Any electroneutral salt: N(0) = 0 and N'(0) = 1 (the linearized κ̄²).
        let asym = PbParams {
            ions: vec![
                IonSpecies {
                    charge: 2.0,
                    concentration: 0.1,
                    radius: 2.0,
                },
                IonSpecies {
                    charge: -1.0,
                    concentration: 0.2,
                    radius: 2.0,
                },
            ],
            ..PbParams::default()
        };
        let (n, dn) = term(&IonTerms::new(&asym).unwrap(), 0.0);
        assert!(n.abs() < 1e-15 && (dn - 1.0).abs() < 1e-15, "{n} {dn}");
        assert!(IonTerms::new(&PbParams::default()).is_none());
    }

    #[test]
    fn box_share_follows_apbs_set_part() {
        let (lo, hi, h) = (-2.0, 3.0, 1.0);
        assert_eq!(box_share(0.0, h, lo, hi), 1.0);
        assert_eq!(box_share(lo, h, lo, hi), 1.0);
        assert_eq!(box_share(hi, h, lo, hi), 1.0);
        assert!((box_share(3.25, h, lo, hi) - 0.25).abs() < 1e-15);
        assert!((box_share(-2.25, h, lo, hi) - 0.25).abs() < 1e-15);
        assert!((box_share(2.75, h, lo, hi) - 0.75).abs() < 1e-15);
        assert_eq!(box_share(3.5, h, lo, hi), 0.0);
        assert_eq!(box_share(5.0, h, lo, hi), 0.0);
    }

    /// The outer terms count the coarse nodes more than half a coarse
    /// spacing outside the finer box (every node with a share of the box is
    /// left to the finer level), faces with the mean of their nodes.
    #[test]
    fn external_terms_count_the_nodes_beyond_the_finer_box() {
        let coarse = Grid::centered([0.0; 3], [20.0, 18.0, 16.0], [21, 19, 17]).unwrap();
        let fine = Grid::centered([0.3, -0.2, 0.1], [8.2, 7.1, 6.3], [33, 29, 27]).unwrap();
        let n = coarse.len();
        let pos = |c: usize| {
            let (i, j, k) = coarse.ijk(c);
            coarse.position(i, j, k)
        };
        let field = |c: usize, a: f64| {
            let x = pos(c);
            (a * x[0]).sin() + 0.3 * (0.5 * x[1] - x[2]).cos()
        };
        let maps = CoefficientMaps {
            eps_x: (0..n).map(|c| 2.0 + field(c, 0.7).abs()).collect(),
            eps_y: (0..n).map(|c| 3.0 + field(c, 0.4).abs()).collect(),
            eps_z: (0..n).map(|c| 4.0 + field(c, 0.2).abs()).collect(),
            kappa_sq: (0..n).map(|c| if c % 3 == 0 { 0.0 } else { 1.27 }).collect(),
        };
        let u: Vec<f64> = (0..n).map(|c| 2.0 * field(c, 0.3)).collect();
        let level = NonlinearLevel {
            grid: coarse.clone(),
            maps: maps.clone(),
            u: u.clone(),
        };
        let ions = IonTerms::new(&PbParams::default().with_salt(0.15, 2.0)).unwrap();
        let lb0 = 7.0;
        let (diel, ion) = level.external_terms(&fine, &ions, lb0);

        let hi: [f64; 3] = [0, 1, 2].map(|a| fine.origin[a] + (fine.dims[a] - 1) as f64 * fine.spacing[a]);
        let outside = |c: usize| {
            let x = pos(c);
            let far = (0..3).any(|a| {
                let h = coarse.spacing[a] / 2.0;
                x[a] > hi[a] + h || x[a] < fine.origin[a] - h
            });
            if far {
                1.0
            } else {
                0.0
            }
        };
        let [nx, ny, nz] = coarse.dims;
        let coef = StencilOperator::face_coefficients(&coarse);
        let eps = [&maps.eps_x, &maps.eps_y, &maps.eps_z];
        let mut d = 0.0;
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let c = i + nx * (j + ny * k);
                    for (a, step) in [1, nx, nx * ny].into_iter().enumerate() {
                        let w = 0.5 * (outside(c) + outside(c + step));
                        d += coef[a] * w * eps[a][c] * (u[c] - u[c + step]).powi(2);
                    }
                }
            }
        }
        let scale = 1.0 / (4.0 * PI * lb0);
        let vol = coarse.cell_volume();
        let i: f64 = (0..n)
            .map(|c| outside(c) * maps.kappa_sq[c] * (f64::cosh(u[c]) - 1.0))
            .sum();
        let (d, i) = (0.5 * d * scale, vol * i * scale);
        assert!(d > 0.0 && i > 0.0);
        assert!(
            (diel - d).abs() < 1e-12 * d && (ion - i).abs() < 1e-12 * i,
            "{diel} vs {d}, {ion} vs {i}"
        );
    }

    fn atoms(scale: f64) -> Vec<Atom> {
        [
            ([0.13, -0.21, 0.07], -3.0, 2.2),
            ([2.31, 0.52, -0.33], 1.5, 1.8),
            ([-1.12, 1.41, 0.88], -2.0, 1.6),
        ]
        .map(|(position, charge, radius)| Atom {
            position,
            charge: scale * charge,
            radius,
        })
        .to_vec()
    }

    fn params(equation: Equation) -> PbParams {
        PbParams {
            pdie: 2.0,
            equation,
            tol: 1e-11,
            preconditioner: Preconditioner::Multigrid,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0)
    }

    /// The functional is stationary at the discrete solution: its gradient
    /// is −F/(4π l_B0), so a perturbation δ changes it by O(δ²).
    #[test]
    fn the_energy_functional_is_stationary_at_the_solution() {
        let atoms = atoms(1.0);
        let p = params(Equation::Nonlinear);
        let grid = Grid::centered([0.3, 0.2, -0.1], [16.0, 15.0, 14.0], [33, 31, 29]).unwrap();
        let surf = crate::surface::MolecularSurface::for_params(&atoms, &p);
        let maps = CoefficientMaps::solvated_with(&grid, &atoms, &p, &surf);
        let (nodes, charges) = spread_charges_sparse_with(&grid, &atoms, ChargeMethod::Trilinear).unwrap();
        let lb0 = bjerrum_length_vacuum(p.temperature);
        let ions = IonTerms::new(&p).unwrap();
        let g = vec![0.0; grid.boundary_indices().len()];
        let opts = SolverOptions {
            tol: 1e-12,
            max_iter: 1000,
            preconditioner: Preconditioner::Multigrid,
        };
        let (_, stats, mut level) =
            solve_level(&CpuBackend::new(), grid, g, maps, (&nodes, &charges), &ions, lb0, opts).unwrap();
        assert!(stats.relative_residual < 1e-12, "{stats:?}");
        let e0 = level.energy(&ions, lb0, (&nodes, &charges));
        // A smooth bump around the molecule, in the solvent and in the solute.
        let bump: Vec<f64> = (0..level.u.len())
            .map(|c| {
                let (i, j, k) = level.grid.ijk(c);
                if level.grid.is_boundary(i, j, k) {
                    return 0.0;
                }
                let x = level.grid.position(i, j, k);
                (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / 20.0).exp()
            })
            .collect();
        let u0 = level.u.clone();
        let mut change = |d: f64| {
            for ((u, a), b) in level.u.iter_mut().zip(&u0).zip(&bump) {
                *u = a + d * b;
            }
            level.energy(&ions, lb0, (&nodes, &charges)) - e0
        };
        let (big, small) = (change(1e-2), change(5e-3));
        // Second order: halving δ quarters the change, and it is a maximum.
        assert!(big < 0.0 && small < 0.0, "{big} {small}");
        assert!((big / small - 4.0).abs() < 0.05, "ratio {}", big / small);
    }

    fn solve(scale: f64, equation: Equation) -> crate::pb::PolarSolvation {
        let atoms = atoms(scale);
        let spec = GridSpec::auto(&atoms, 0.4, 1.7, 12.0).unwrap();
        polar_solvation(&CpuBackend::new(), &atoms, &params(equation), &spec).unwrap()
    }

    /// Weak potentials: the nonlinear potential is the linearized one up to
    /// O(q²). The energies differ by the functional's truncation at the
    /// boxes (field energy beyond them, ~1e-3 here), which, unlike a
    /// nonlinear effect, does not depend on the charge scale.
    #[test]
    fn small_charges_give_the_linearized_potential() {
        use crate::pb::{solve_focused, Medium};
        let gap = |scale: f64| {
            let atoms = atoms(scale);
            let spec = GridSpec::auto(&atoms, 0.4, 1.7, 12.0).unwrap();
            let be = CpuBackend::new();
            let field = |e| {
                solve_focused(&be, &spec, &atoms, &params(e), Medium::Solvated)
                    .unwrap()
                    .potential
            };
            let (lin, non) = (field(Equation::Linear), field(Equation::Nonlinear));
            let max = lin.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let diff = lin.iter().zip(&non).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                diff < 1e-7 * max * (scale / 1e-3).powi(2),
                "scale {scale}: {diff} of {max}"
            );
            let (lin, non) = (solve(scale, Equation::Linear), solve(scale, Equation::Nonlinear));
            assert_eq!(non.reference_kt.to_bits(), lin.reference_kt.to_bits());
            (non.energy_kt - lin.energy_kt) / lin.energy_kt.abs()
        };
        let (small, larger) = (gap(1e-3), gap(1e-2));
        assert!(
            small.abs() < 2e-3 && (small - larger).abs() < 1e-6,
            "{small:e} {larger:e}"
        );
    }

    /// Strong potentials: beyond |u| ~ 1 the Boltzmann ions screen more
    /// strongly than the linearized equation (sinh u > u), so ΔG_polar is
    /// more negative; Newton converges from the linearized start.
    #[test]
    fn large_charges_converge_and_screen_less() {
        let (lin, non) = (solve(4.0, Equation::Linear), solve(4.0, Equation::Nonlinear));
        for s in &non.stats_solvated {
            assert!(s.relative_residual < 1e-11, "{s:?}");
        }
        assert!(
            non.energy_kt < lin.energy_kt && non.energy_kt > 1.02 * lin.energy_kt,
            "{} vs {}",
            non.energy_kt,
            lin.energy_kt
        );
    }
}
