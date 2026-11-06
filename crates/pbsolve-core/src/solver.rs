//! Preconditioned conjugate gradient, generic over [`Backend`].
//!
//! Preconditioners: Jacobi (diagonal scaling) or a geometric multigrid
//! V-cycle ([`crate::multigrid`]), which keeps the iteration count roughly
//! independent of the grid spacing.

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::multigrid::{Chebyshev, Levels, MultigridPreconditioner};
use crate::operator::StencilOperator;
use crate::params::Preconditioner;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct SolverOptions {
    /// Stop when ‖r‖ / ‖b‖ < tol.
    pub tol: f64,
    pub max_iter: usize,
    /// Requested preconditioner (multigrid falls back to Jacobi when the grid
    /// cannot be coarsened far enough; see [`SolveStats::preconditioner`]).
    pub preconditioner: Preconditioner,
}

impl SolverOptions {
    pub fn new(tol: f64, max_iter: usize) -> Self {
        Self {
            tol,
            max_iter,
            preconditioner: Preconditioner::DEFAULT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveStats {
    pub iterations: usize,
    pub relative_residual: f64,
    /// Preconditioner actually used.
    pub preconditioner: Preconditioner,
    /// Number of multigrid levels including the fine grid (1 for Jacobi).
    pub levels: usize,
    /// Wall time of the preconditioner setup and operator upload (s).
    pub setup_seconds: f64,
    /// Wall time of the CG iterations including the final download (s).
    pub solve_seconds: f64,
}

/// The f32 V-cycle of [`Preconditioner::MultigridF32`]: z = promote(M⁻¹ demote(r)).
struct Mixed<B: Backend> {
    single: B::Single,
    op: <B::Single as Backend>::Operator,
    mg: MultigridPreconditioner<B::Single>,
    r: <B::Single as Backend>::Vector,
    z: <B::Single as Backend>::Vector,
}

enum Precond<B: Backend> {
    Jacobi(B::Vector),
    Multigrid(Box<MultigridPreconditioner<B>>),
    Mixed(Box<Mixed<B>>),
}

impl<B: Backend> Precond<B> {
    /// `dev` is `op` prepared on `be`; the hierarchy is built on the backend.
    fn build(be: &B, op: &Arc<StencilOperator>, dev: &B::Operator, kind: Preconditioner) -> Result<Self> {
        match kind {
            Preconditioner::Jacobi => {}
            Preconditioner::Multigrid => {
                if let Some(l) = Levels::build(be, op, dev, Chebyshev::default())? {
                    return Ok(Self::Multigrid(Box::new(MultigridPreconditioner::from_levels(
                        be, op, l,
                    )?)));
                }
            }
            Preconditioner::MultigridF32 => {
                if let Some(l) = Levels::build(be, op, dev, Chebyshev::default())? {
                    let single = be.single()?;
                    let l = l.demote(be, &single)?;
                    let mg = MultigridPreconditioner::from_levels(&single, op, l)?;
                    let n = op.len();
                    return Ok(Self::Mixed(Box::new(Mixed {
                        op: be.demote_operator(dev)?,
                        r: single.zeros(n)?,
                        z: single.zeros(n)?,
                        mg,
                        single,
                    })));
                }
            }
        }
        Ok(Self::Jacobi(be.upload(&op.inv_diag)?))
    }

    fn kind(&self) -> (Preconditioner, usize) {
        match self {
            Self::Jacobi(_) => (Preconditioner::Jacobi, 1),
            Self::Multigrid(m) => (Preconditioner::Multigrid, m.levels()),
            Self::Mixed(m) => (Preconditioner::MultigridF32, m.mg.levels()),
        }
    }

    fn apply(&mut self, be: &B, op: &B::Operator, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
        match self {
            Self::Jacobi(inv_diag) => be.scale(1.0, inv_diag, r, z),
            Self::Multigrid(m) => m.apply(be, op, r, z),
            Self::Mixed(m) => {
                be.demote(r, &mut m.r)?;
                m.mg.apply(&m.single, &m.op, &m.r, &mut m.z)?;
                be.promote(&m.z, z)
            }
        }
    }

    /// A preconditioner that is not exactly symmetric (rounded to f32) needs
    /// the flexible CG update.
    fn is_flexible(&self) -> bool {
        matches!(self, Self::Mixed(_))
    }
}

/// Solves A x = b (x on interior nodes, zero on the boundary).
pub fn pcg<B: Backend>(
    be: &B,
    op: Arc<StencilOperator>,
    b: &[f64],
    opts: SolverOptions,
) -> Result<(Vec<f64>, SolveStats)> {
    let start = Instant::now();
    let n = b.len();
    let bv = be.upload(b)?;
    let bnorm = be.dot(&bv, &bv)?.sqrt();
    if bnorm == 0.0 {
        return Ok((
            vec![0.0; n],
            SolveStats {
                iterations: 0,
                relative_residual: 0.0,
                preconditioner: opts.preconditioner,
                levels: 1,
                setup_seconds: start.elapsed().as_secs_f64(),
                solve_seconds: 0.0,
            },
        ));
    }
    let dop = be.prepare(op.clone())?;
    let mut pre = Precond::build(be, &op, &dop, opts.preconditioner)?;
    // Release the host operator now (device backends hold their own copy),
    // not after the solve.
    drop(op);
    let flexible = pre.is_flexible();
    let (preconditioner, levels) = pre.kind();
    let setup_seconds = start.elapsed().as_secs_f64();
    let iterating = Instant::now();

    let mut x = be.zeros(n)?;
    let mut r = bv;
    let mut z = be.zeros(n)?;
    let mut q = be.zeros(n)?;
    let mut p = be.zeros(n)?;
    pre.apply(be, &dop, &r, &mut z)?;
    be.copy(&z, &mut p)?;
    let mut rz = be.dot(&r, &z)?;

    let mut rel = 1.0;
    for it in 1..=opts.max_iter {
        be.apply(&dop, &p, &mut q)?;
        let pq = be.dot(&p, &q)?;
        if pq.is_nan() || pq <= 0.0 {
            return Err(PbError::Backend(format!(
                "operator is not positive definite (pᵀAp = {pq:.3e} at iteration {it})"
            )));
        }
        let alpha = rz / pq;
        be.axpy(alpha, &p, &mut x)?;
        be.axpy(-alpha, &q, &mut r)?;
        // Flexible CG also needs r_{k+1}ᵀz_k (z still holds z_k): fused with ‖r‖².
        let (rr, rz_prev) = if flexible {
            be.dot2(&r, &r, &z)?
        } else {
            (be.dot(&r, &r)?, 0.0)
        };
        rel = rr.sqrt() / bnorm;
        if rel < opts.tol {
            let x = be.download(&x)?;
            return Ok((
                x,
                SolveStats {
                    iterations: it,
                    relative_residual: rel,
                    preconditioner,
                    levels,
                    setup_seconds,
                    solve_seconds: iterating.elapsed().as_secs_f64(),
                },
            ));
        }
        // Flexible CG (Golub & Ye 1999; Notay 2000): Polak–Ribière
        // β = r_{k+1}ᵀ(z_{k+1} − z_k) / r_kᵀz_k keeps the search directions
        // conjugate when the preconditioner is slightly non-symmetric. For an
        // exactly symmetric one r_{k+1}ᵀz_k = 0 and this is plain PCG.
        pre.apply(be, &dop, &r, &mut z)?;
        let rz_new = be.dot(&r, &z)?;
        if rz_new.is_nan() || rz_new <= 0.0 {
            return Err(PbError::Backend(format!(
                "preconditioner is not positive definite (rᵀz = {rz_new:.3e} at iteration {it})"
            )));
        }
        let beta = if flexible { (rz_new - rz_prev) / rz } else { rz_new / rz };
        rz = rz_new;
        be.xpay(&z, beta, &mut p)?;
    }
    Err(PbError::NotConverged {
        iterations: opts.max_iter,
        residual: rel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charge::spread_charges;
    use crate::cpu::CpuBackend;
    use crate::grid::Grid;
    use crate::molecule::Atom;
    use crate::operator::assemble_rhs;
    use crate::params::PbParams;
    use crate::surface::CoefficientMaps;
    use crate::units::bjerrum_length_vacuum;

    fn system(n: usize) -> (Arc<StencilOperator>, Vec<f64>) {
        let mut s: u64 = 77;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let atoms: Vec<Atom> = (0..40)
            .map(|_| Atom {
                position: [rnd(), rnd(), rnd()].map(|v| 14.0 * (v - 0.5)),
                charge: rnd() - 0.5,
                radius: 1.2 + 0.7 * rnd(),
            })
            .collect();
        let params = PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        let grid = Grid::centered([0.0; 3], [26.0; 3], [n; 3]).unwrap();
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

    fn solve(op: &Arc<StencilOperator>, b: &[f64], pc: Preconditioner, tol: f64) -> (Vec<f64>, SolveStats) {
        let opts = SolverOptions {
            preconditioner: pc,
            ..SolverOptions::new(tol, 500)
        };
        pcg(&CpuBackend::new(), op.clone(), b, opts).unwrap()
    }

    fn true_residual(op: &StencilOperator, x: &[f64], b: &[f64]) -> f64 {
        let mut ax = vec![0.0; x.len()];
        op.apply(x, &mut ax);
        let r: f64 = ax.iter().zip(b).map(|(a, c)| (c - a) * (c - a)).sum();
        (r / b.iter().map(|v| v * v).sum::<f64>()).sqrt()
    }

    #[test]
    fn single_precision_multigrid_reaches_double_precision_accuracy() {
        let (op, b) = system(41);
        for tol in [1e-8, 1e-11, 1e-13] {
            let (x64, s64) = solve(&op, &b, Preconditioner::Multigrid, tol);
            let (x32, s32) = solve(&op, &b, Preconditioner::MultigridF32, tol);
            assert_eq!(s32.preconditioner, Preconditioner::MultigridF32);
            assert_eq!(s32.levels, s64.levels);
            // The f32 cycle does not slow CG down, even far below f32 precision.
            assert!(
                s32.iterations <= s64.iterations + 1,
                "{tol}: {} vs {}",
                s32.iterations,
                s64.iterations
            );
            let res = true_residual(&op, &x32, &b);
            assert!(res < 2.0 * tol, "{tol}: true residual {res:e}");
            let d: f64 = x32.iter().zip(&x64).map(|(a, c)| (a - c) * (a - c)).sum();
            let n: f64 = x64.iter().map(|v| v * v).sum();
            assert!(
                (d / n).sqrt() < 1e3 * tol,
                "{tol}: solutions differ by {:e}",
                (d / n).sqrt()
            );
        }
    }

    #[test]
    fn single_precision_results_do_not_depend_on_the_thread_count() {
        let (op, b) = system(33);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| solve(&op, &b, Preconditioner::MultigridF32, 1e-9))
        };
        let ((x1, s1), (x4, s4)) = (run(1), run(4));
        assert_eq!(s1.iterations, s4.iterations);
        assert!(x1.iter().zip(&x4).all(|(a, c)| a.to_bits() == c.to_bits()));
    }

    #[test]
    fn dot2_equals_two_dots() {
        let be = CpuBackend::new();
        let a: Vec<f64> = (0..70_000).map(|i| (i as f64 * 0.37).sin()).collect();
        let b: Vec<f64> = (0..70_000).map(|i| (i as f64 * 0.11).cos()).collect();
        let c: Vec<f64> = (0..70_000).map(|i| 1.0 / (1.0 + i as f64)).collect();
        let (ab, ac) = be.dot2(&a, &b, &c).unwrap();
        assert_eq!(ab.to_bits(), be.dot(&a, &b).unwrap().to_bits());
        assert_eq!(ac.to_bits(), be.dot(&a, &c).unwrap().to_bits());
    }

    #[test]
    fn preconditioner_names_round_trip() {
        for p in [
            Preconditioner::Jacobi,
            Preconditioner::Multigrid,
            Preconditioner::MultigridF32,
        ] {
            assert_eq!(Preconditioner::parse(p.as_str()).unwrap(), p);
        }
        assert_eq!(Preconditioner::parse("MG-F32").unwrap(), Preconditioner::MultigridF32);
        assert!(Preconditioner::parse("multigrid-f16").is_err());
        // One default everywhere: PbParams, SolverOptions (and the Python Solver).
        assert_eq!(Preconditioner::DEFAULT, Preconditioner::MultigridF32);
        assert_eq!(SolverOptions::new(1e-8, 10).preconditioner, Preconditioner::DEFAULT);
        assert_eq!(crate::PbParams::default().preconditioner, Preconditioner::DEFAULT);
    }
}
