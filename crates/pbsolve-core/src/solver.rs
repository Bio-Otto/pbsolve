//! Preconditioned conjugate gradient, generic over [`Backend`].
//!
//! Preconditioners: Jacobi (diagonal scaling) or a geometric multigrid
//! V-cycle ([`crate::multigrid`]), which keeps the iteration count roughly
//! independent of the grid spacing.

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::multigrid::{Chebyshev, Hierarchy, MultigridPreconditioner};
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
            preconditioner: Preconditioner::Multigrid,
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

enum Precond<B: Backend> {
    Jacobi(B::Vector),
    Multigrid(Box<MultigridPreconditioner<B>>),
}

impl<B: Backend> Precond<B> {
    fn build(be: &B, op: &StencilOperator, kind: Preconditioner) -> Result<Self> {
        if kind == Preconditioner::Multigrid {
            if let Some(h) = Hierarchy::build(op, Chebyshev::default())? {
                return Ok(Self::Multigrid(Box::new(MultigridPreconditioner::new(be, op, h)?)));
            }
        }
        Ok(Self::Jacobi(be.upload(&op.inv_diag)?))
    }

    fn kind(&self) -> (Preconditioner, usize) {
        match self {
            Self::Jacobi(_) => (Preconditioner::Jacobi, 1),
            Self::Multigrid(m) => (Preconditioner::Multigrid, m.levels()),
        }
    }

    fn apply(&mut self, be: &B, op: &B::Operator, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
        match self {
            Self::Jacobi(inv_diag) => be.scale(1.0, inv_diag, r, z),
            Self::Multigrid(m) => m.apply(be, op, r, z),
        }
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
    let mut pre = Precond::build(be, &op, opts.preconditioner)?;
    let (preconditioner, levels) = pre.kind();
    let dop = be.prepare(op)?;
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
        rel = be.dot(&r, &r)?.sqrt() / bnorm;
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
        pre.apply(be, &dop, &r, &mut z)?;
        let rz_new = be.dot(&r, &z)?;
        if rz_new.is_nan() || rz_new <= 0.0 {
            return Err(PbError::Backend(format!(
                "preconditioner is not positive definite (rᵀz = {rz_new:.3e} at iteration {it})"
            )));
        }
        let beta = rz_new / rz;
        rz = rz_new;
        be.xpay(&z, beta, &mut p)?;
    }
    Err(PbError::NotConverged {
        iterations: opts.max_iter,
        residual: rel,
    })
}
