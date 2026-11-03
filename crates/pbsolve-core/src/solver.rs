//! Jacobi-preconditioned conjugate gradient, generic over [`Backend`].
//!
//! Roadmap: a geometric multigrid preconditioner (V-cycle on the 8k+1 grids
//! `GridSpec::auto` produces) is the next big speed-up;
//! it plugs in by replacing `precondition`, without touching the physics.

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::operator::StencilOperator;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct SolverOptions {
    /// Stop when ‖r‖ / ‖b‖ < tol.
    pub tol: f64,
    pub max_iter: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveStats {
    pub iterations: usize,
    pub relative_residual: f64,
}

/// Solves A x = b (x on interior nodes, zero on the boundary).
pub fn pcg<B: Backend>(
    be: &B,
    op: Arc<StencilOperator>,
    b: &[f64],
    opts: SolverOptions,
) -> Result<(Vec<f64>, SolveStats)> {
    let n = b.len();
    let dop = be.prepare(op)?;
    let bv = be.upload(b)?;
    let bnorm = be.dot(&bv, &bv)?.sqrt();
    if bnorm == 0.0 {
        return Ok((
            vec![0.0; n],
            SolveStats {
                iterations: 0,
                relative_residual: 0.0,
            },
        ));
    }

    let mut x = be.zeros(n)?;
    let mut r = bv;
    let mut z = be.zeros(n)?;
    let mut q = be.zeros(n)?;
    be.precondition(&dop, &r, &mut z)?;
    let mut p = be.upload(&be.download(&z)?)?;
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
            return Ok((
                be.download(&x)?,
                SolveStats {
                    iterations: it,
                    relative_residual: rel,
                },
            ));
        }
        be.precondition(&dop, &r, &mut z)?;
        let rz_new = be.dot(&r, &z)?;
        let beta = rz_new / rz;
        rz = rz_new;
        be.xpay(&z, beta, &mut p)?;
    }
    Err(PbError::NotConverged {
        iterations: opts.max_iter,
        residual: rel,
    })
}
