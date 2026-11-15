//! Multithreaded CPU backend (rayon). Always available; it is also the
//! reference implementation every accelerator is validated against.
//!
//! [`CpuBackend`] works in f64. [`CpuSingle`] is its f32 twin used by the
//! mixed-precision multigrid preconditioner. Both are one generic
//! implementation; for f64 it performs exactly the operations of the
//! original f64-only code.

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::grid::Grid;
use crate::multigrid::{self, Coarse27, LevelMatrix, LevelOp, Stencil27, Transfer};
use crate::operator::{apply7_into, apply7_with, Stencil7, StencilOperator};
use crate::real::Real;
use crate::surface::{self, CoefficientMaps, Sphere};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

/// Fixed chunk size => reduction order independent of the thread count.
const CHUNK: usize = 1 << 14;

/// Σ a_i b_i over one chunk, in order: the partial sum of every dot product
/// (`dot` and the fused CG steps).
fn chunk_dot<T: Real>(a: &[T], b: &[T]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p.to_f64() * q.to_f64()).sum()
}

/// The chunks' partial sums added in chunk order.
fn total(partials: &[f64]) -> f64 {
    partials.iter().sum()
}

/// How operators are held for a given precision: f64 shares the host
/// operators, f32 keeps rounded copies.
pub trait CpuReal: Real {
    type Op: Send + Sync;
    type Coarse: Send + Sync;
    fn prepare(op: Arc<StencilOperator>) -> Self::Op;
    fn prepare_coarse(op: Arc<Stencil27>) -> Self::Coarse;
    fn apply(op: &Self::Op, x: &[Self], y: &mut [Self]);
    fn apply_coarse(op: &Self::Coarse, x: &[Self], y: &mut [Self]);
    /// [`crate::operator::apply7_with`] / [`multigrid::apply27_with`] of the
    /// operators (fused residuals).
    fn apply_with<F: Fn(Self, usize, Self) -> Self + Sync>(op: &Self::Op, x: &[Self], out: &mut [Self], f: F);
    fn apply_coarse_with<F: Fn(Self, usize, Self) -> Self + Sync>(
        op: &Self::Coarse,
        x: &[Self],
        out: &mut [Self],
        f: F,
    );
    /// [`crate::operator::apply7_into`] of the operator (fused A p and p·q).
    fn apply_into(op: &Self::Op, x: &[Self], first: usize, out: &mut [Self]);
    /// The f64 matrices behind the operators (multigrid setup); `None` for
    /// rounded copies.
    fn fine_matrix(op: &Self::Op) -> Option<&StencilOperator>;
    fn coarse_matrix(op: &Self::Coarse) -> Option<&Stencil27>;
    /// f32 copies for the twin.
    fn demote_op(op: &Self::Op) -> Arc<Stencil7<f32>>;
    fn demote_coarse(op: &Self::Coarse) -> Arc<Coarse27<f32>>;
    /// Host f64 data as a vector of this precision without a copy; for f32
    /// (which needs a rounded copy) the data is handed back.
    fn from_host(v: Vec<f64>) -> std::result::Result<Vec<Self>, Vec<f64>>;
    /// The f64 data of a vector of this precision; `None` for f32.
    fn into_host(v: Vec<Self>) -> Option<Vec<f64>>;
}

impl CpuReal for f64 {
    type Op = Arc<StencilOperator>;
    type Coarse = Arc<Stencil27>;
    fn prepare(op: Arc<StencilOperator>) -> Self::Op {
        op
    }
    fn prepare_coarse(op: Arc<Stencil27>) -> Self::Coarse {
        op
    }
    fn apply(op: &Self::Op, x: &[f64], y: &mut [f64]) {
        op.apply(x, y)
    }
    fn apply_coarse(op: &Self::Coarse, x: &[f64], y: &mut [f64]) {
        op.apply(x, y)
    }
    fn apply_with<F: Fn(f64, usize, f64) -> f64 + Sync>(op: &Self::Op, x: &[f64], out: &mut [f64], f: F) {
        apply7_with(op.dims, op.coef, &op.eps_x, &op.eps_y, &op.eps_z, &op.kvol, x, out, f)
    }
    fn apply_coarse_with<F: Fn(f64, usize, f64) -> f64 + Sync>(op: &Self::Coarse, x: &[f64], out: &mut [f64], f: F) {
        multigrid::apply27_with(op.dims, &op.coef, x, out, f)
    }
    fn apply_into(op: &Self::Op, x: &[f64], first: usize, out: &mut [f64]) {
        apply7_into(
            op.dims, op.coef, &op.eps_x, &op.eps_y, &op.eps_z, &op.kvol, x, first, out,
        )
    }
    fn fine_matrix(op: &Self::Op) -> Option<&StencilOperator> {
        Some(op)
    }
    fn coarse_matrix(op: &Self::Coarse) -> Option<&Stencil27> {
        Some(op)
    }
    fn demote_op(op: &Self::Op) -> Arc<Stencil7<f32>> {
        Arc::new(Stencil7::from_operator(op))
    }
    fn demote_coarse(op: &Self::Coarse) -> Arc<Coarse27<f32>> {
        Arc::new(Coarse27::from_stencil(op))
    }
    fn from_host(v: Vec<f64>) -> std::result::Result<Vec<f64>, Vec<f64>> {
        Ok(v)
    }
    fn into_host(v: Vec<f64>) -> Option<Vec<f64>> {
        Some(v)
    }
}

impl CpuReal for f32 {
    type Op = Arc<Stencil7<f32>>;
    type Coarse = Arc<Coarse27<f32>>;
    fn prepare(op: Arc<StencilOperator>) -> Self::Op {
        Arc::new(Stencil7::from_operator(&op))
    }
    fn prepare_coarse(op: Arc<Stencil27>) -> Self::Coarse {
        Arc::new(Coarse27::from_stencil(&op))
    }
    fn apply(op: &Self::Op, x: &[f32], y: &mut [f32]) {
        op.apply(x, y)
    }
    fn apply_coarse(op: &Self::Coarse, x: &[f32], y: &mut [f32]) {
        op.apply(x, y)
    }
    fn apply_with<F: Fn(f32, usize, f32) -> f32 + Sync>(op: &Self::Op, x: &[f32], out: &mut [f32], f: F) {
        apply7_with(op.dims, op.coef, &op.eps_x, &op.eps_y, &op.eps_z, &op.kvol, x, out, f)
    }
    fn apply_coarse_with<F: Fn(f32, usize, f32) -> f32 + Sync>(op: &Self::Coarse, x: &[f32], out: &mut [f32], f: F) {
        multigrid::apply27_with(op.dims, &op.coef, x, out, f)
    }
    fn apply_into(op: &Self::Op, x: &[f32], first: usize, out: &mut [f32]) {
        apply7_into(
            op.dims, op.coef, &op.eps_x, &op.eps_y, &op.eps_z, &op.kvol, x, first, out,
        )
    }
    fn fine_matrix(_: &Self::Op) -> Option<&StencilOperator> {
        None
    }
    fn coarse_matrix(_: &Self::Coarse) -> Option<&Stencil27> {
        None
    }
    fn demote_op(op: &Self::Op) -> Arc<Stencil7<f32>> {
        op.clone()
    }
    fn demote_coarse(op: &Self::Coarse) -> Arc<Coarse27<f32>> {
        op.clone()
    }
    fn from_host(v: Vec<f64>) -> std::result::Result<Vec<f32>, Vec<f64>> {
        Err(v)
    }
    fn into_host(_: Vec<f32>) -> Option<Vec<f64>> {
        None
    }
}

/// The f64 matrix of a level (multigrid setup runs in f64 only).
fn level_matrix<T: CpuReal>(level: LevelOp<'_, Cpu<T>>) -> Result<LevelMatrix<'_>> {
    let m = match level {
        LevelOp::Fine(op) => T::fine_matrix(op).map(LevelMatrix::Seven),
        LevelOp::Coarse(op) => T::coarse_matrix(op).map(LevelMatrix::TwentySeven),
    };
    m.ok_or_else(|| PbError::Backend("multigrid setup needs the f64 backend".into()))
}

fn assembly<V>(v: Option<V>) -> Result<V> {
    v.ok_or_else(|| PbError::Backend("assembly needs the f64 backend".into()))
}

/// CPU backend computing in `T`.
pub struct Cpu<T>(PhantomData<fn() -> T>);

/// The f64 CPU backend.
pub type CpuBackend = Cpu<f64>;
/// The f32 CPU backend (mixed-precision multigrid).
pub type CpuSingle = Cpu<f32>;

impl<T> Cpu<T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for Cpu<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for Cpu<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for Cpu<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cpu<{}>", std::any::type_name::<T>())
    }
}

impl<T: CpuReal> Backend for Cpu<T> {
    type Vector = Vec<T>;
    type Operator = T::Op;
    type Coarse = T::Coarse;
    type Single = CpuSingle;

    fn platform(&self) -> &'static str {
        "CPU"
    }

    fn device_name(&self) -> String {
        format!("CPU ({} threads)", rayon::current_num_threads())
    }

    fn single(&self) -> Result<Self::Single> {
        Ok(CpuSingle::new())
    }

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator> {
        Ok(T::prepare(op))
    }

    fn prepare_coarse(&self, op: Arc<Stencil27>) -> Result<Self::Coarse> {
        Ok(T::prepare_coarse(op))
    }

    fn upload(&self, host: &[f64]) -> Result<Self::Vector> {
        Ok(host.par_iter().map(|&v| T::from_f64(v)).collect())
    }

    fn upload_owned(&self, host: Vec<f64>) -> Result<Self::Vector> {
        match T::from_host(host) {
            Ok(v) => Ok(v),
            Err(host) => self.upload(&host),
        }
    }

    fn zeros(&self, n: usize) -> Result<Self::Vector> {
        Ok(vec![T::ZERO; n])
    }

    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>> {
        Ok(v.par_iter().map(|x| x.to_f64()).collect())
    }

    fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()> {
        host.par_iter_mut()
            .zip(v.par_iter())
            .with_min_len(CHUNK)
            .for_each(|(h, x)| *h = x.to_f64());
        Ok(())
    }

    fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()> {
        v.par_iter_mut()
            .zip(host.par_iter())
            .with_min_len(CHUNK)
            .for_each(|(x, &h)| *x = T::from_f64(h));
        Ok(())
    }

    fn demote(&self, x: &Self::Vector, y: &mut Vec<f32>) -> Result<()> {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(y, x)| *y = f32::from_f64(x.to_f64()));
        Ok(())
    }

    fn promote(&self, x: &Vec<f32>, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(y, x)| *y = T::from_f64(x.to_f64()));
        Ok(())
    }

    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        T::apply(op, x, y);
        Ok(())
    }

    fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        T::apply_coarse(op, x, y);
        Ok(())
    }

    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64> {
        let partial: Vec<f64> = a
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(x, y)| chunk_dot(x, y))
            .collect();
        Ok(total(&partial))
    }

    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let alpha = T::from_f64(alpha);
        y.par_iter_mut().zip(x.par_iter()).for_each(|(y, x)| *y += alpha * *x);
        Ok(())
    }

    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let beta = T::from_f64(beta);
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(y, x)| *y = *x + beta * *y);
        Ok(())
    }

    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let (alpha, beta) = (T::from_f64(alpha), T::from_f64(beta));
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(y, x)| *y = alpha * *x + beta * *y);
        Ok(())
    }

    fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let alpha = T::from_f64(alpha);
        y.par_iter_mut()
            .zip(x.par_iter().zip(d.par_iter()))
            .for_each(|(y, (x, d))| *y = (alpha * *d) * *x);
        Ok(())
    }

    fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut().zip(x.par_iter()).for_each(|(y, x)| *y = *x);
        Ok(())
    }

    fn restrict(&self, t: &Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()> {
        multigrid::restrict(t, fine, coarse);
        Ok(())
    }

    fn prolong_add(&self, t: &Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()> {
        multigrid::prolong_add(t, coarse, fine);
        Ok(())
    }

    // The smoother steps in one pass each, with the expressions of the
    // separate operations (`axpy`: y + α·x; `scale`: (α·d)·x; `axpby`:
    // α·x + β·y), so the results equal the step-by-step defaults bit for bit.

    fn residual(
        &self,
        op: LevelOp<'_, Self>,
        x: &Self::Vector,
        b: Option<&Self::Vector>,
        _t: &mut Self::Vector,
        r: &mut Self::Vector,
    ) -> Result<()> {
        let minus_one = T::from_f64(-1.0);
        match (op, b) {
            (LevelOp::Fine(o), Some(b)) => T::apply_with(o, x, r, |_, c, v| b[c] + minus_one * v),
            (LevelOp::Fine(o), None) => T::apply_with(o, x, r, |r, _, v| r + minus_one * v),
            (LevelOp::Coarse(o), Some(b)) => T::apply_coarse_with(o, x, r, |_, c, v| b[c] + minus_one * v),
            (LevelOp::Coarse(o), None) => T::apply_coarse_with(o, x, r, |r, _, v| r + minus_one * v),
        }
        Ok(())
    }

    fn chebyshev_first(
        &self,
        s: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
        overwrite: bool,
    ) -> Result<()> {
        let (s, one) = (T::from_f64(s), T::from_f64(1.0));
        d.par_iter_mut()
            .zip(x.par_iter_mut())
            .zip(w.par_iter().zip(r.par_iter()))
            .for_each(|((d, x), (w, r))| {
                *d = (s * *w) * *r;
                if overwrite {
                    *x = *d;
                } else {
                    *x += one * *d;
                }
            });
        Ok(())
    }

    fn chebyshev_next(
        &self,
        c1: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        c2: f64,
        _t: &mut Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
    ) -> Result<()> {
        let (c1, c2, one) = (T::from_f64(c1), T::from_f64(c2), T::from_f64(1.0));
        d.par_iter_mut()
            .zip(x.par_iter_mut())
            .zip(w.par_iter().zip(r.par_iter()))
            .for_each(|((d, x), (w, r))| {
                let t = (c1 * *w) * *r;
                *d = one * t + c2 * *d;
                *x += one * *d;
            });
        Ok(())
    }

    // The CG steps in one parallel pass each: every chunk of `dot` is
    // updated with the expressions of the separate operations (`apply`;
    // `axpy`: y + α·x; `demote`; `promote`) and then summed by `chunk_dot`
    // while it is still in cache, so the results equal the step-by-step
    // defaults bit for bit.

    fn apply_dot(&self, op: &Self::Operator, p: &Self::Vector, q: &mut Self::Vector) -> Result<f64> {
        let partial: Vec<f64> = q
            .par_chunks_mut(CHUNK)
            .zip(p.par_chunks(CHUNK))
            .enumerate()
            .map(|(k, (qc, pc))| {
                T::apply_into(op, p, k * CHUNK, qc);
                chunk_dot(pc, qc)
            })
            .collect();
        Ok(total(&partial))
    }

    fn cg_update(
        &self,
        alpha: f64,
        p: &Self::Vector,
        q: &Self::Vector,
        x: &mut Self::Vector,
        r: &mut Self::Vector,
        z: Option<&Self::Vector>,
        rounded: Option<&mut Vec<f32>>,
    ) -> Result<(f64, f64)> {
        let n = r.len();
        let lengths_match = [p.len(), q.len(), x.len()].iter().all(|&m| m == n)
            && z.is_none_or(|z| z.len() == n)
            && rounded.as_ref().is_none_or(|v| v.len() == n);
        if !lengths_match {
            return Err(PbError::Backend("cg_update: vectors of different lengths".into()));
        }
        let (a, minus_a) = (T::from_f64(alpha), T::from_f64(-alpha));
        let mut rounded: Vec<Option<&mut [f32]>> = match rounded {
            Some(v) => v.chunks_mut(CHUNK).map(Some).collect(),
            None => (0..n.div_ceil(CHUNK)).map(|_| None).collect(),
        };
        let (rr, rz): (Vec<f64>, Vec<f64>) = x
            .par_chunks_mut(CHUNK)
            .zip(r.par_chunks_mut(CHUNK))
            .zip(p.par_chunks(CHUNK).zip(q.par_chunks(CHUNK)))
            .zip(rounded.par_iter_mut())
            .enumerate()
            .map(|(k, (((xc, rc), (pc, qc)), out))| {
                for (x, p) in xc.iter_mut().zip(pc) {
                    *x += a * *p;
                }
                for (r, q) in rc.iter_mut().zip(qc) {
                    *r += minus_a * *q;
                }
                let rz = z.map_or(0.0, |z| chunk_dot(rc, &z[k * CHUNK..k * CHUNK + rc.len()]));
                if let Some(out) = out {
                    for (o, r) in out.iter_mut().zip(rc.iter()) {
                        *o = f32::from_f64(r.to_f64());
                    }
                }
                (chunk_dot(rc, rc), rz)
            })
            .unzip();
        Ok((total(&rr), if z.is_some() { total(&rz) } else { 0.0 }))
    }

    fn promote_dot(&self, z32: &Vec<f32>, r: &Self::Vector, z: &mut Self::Vector) -> Result<f64> {
        let partial: Vec<f64> = z
            .par_chunks_mut(CHUNK)
            .zip(z32.par_chunks(CHUNK).zip(r.par_chunks(CHUNK)))
            .map(|(zc, (sc, rc))| {
                for (z, s) in zc.iter_mut().zip(sc) {
                    *z = T::from_f64(s.to_f64());
                }
                chunk_dot(rc, zc)
            })
            .collect();
        Ok(total(&partial))
    }

    fn galerkin(&self, t: &Transfer, fine: LevelOp<'_, Self>) -> Result<(Self::Coarse, Self::Vector)> {
        let s = multigrid::galerkin(t, level_matrix(fine)?)?;
        let inv_diag = self.upload(&s.inv_diag)?;
        Ok((T::prepare_coarse(Arc::new(s)), inv_diag))
    }

    fn gershgorin_lmax(&self, level: LevelOp<'_, Self>) -> Result<f64> {
        Ok(level_matrix(level)?.gershgorin_lmax())
    }

    fn coarse_to_host(&self, op: &Self::Coarse) -> Result<Stencil27> {
        T::coarse_matrix(op)
            .cloned()
            .ok_or_else(|| PbError::Backend("multigrid setup needs the f64 backend".into()))
    }

    fn demote_operator(&self, op: &Self::Operator) -> Result<Arc<Stencil7<f32>>> {
        Ok(T::demote_op(op))
    }

    fn demote_coarse(&self, op: &Self::Coarse) -> Result<Arc<Coarse27<f32>>> {
        Ok(T::demote_coarse(op))
    }

    fn mark_maps(
        &self,
        grid: &Grid,
        shifts: &[[f64; 3]],
        fill: f64,
        passes: &[(&[Sphere], f64)],
    ) -> Result<Vec<Self::Vector>> {
        assembly((T::NAME == "f64").then_some(()))?;
        shifts
            .iter()
            .map(|&shift| assembly(T::from_host(surface::mark_map(grid, shift, fill, passes)).ok()))
            .collect()
    }

    fn scatter(&self, n: usize, idx: &[usize], values: &[f64]) -> Result<Self::Vector> {
        assembly((T::NAME == "f64").then_some(()))?;
        crate::backend::check_scatter(n, idx, values)?;
        let mut v = vec![0.0; n];
        for (&c, &x) in idx.iter().zip(values) {
            v[c] = x;
        }
        assembly(T::from_host(v).ok())
    }

    fn assemble_operator(&self, grid: &Grid, maps: [Self::Vector; 4]) -> Result<(Self::Operator, Self::Vector)> {
        let n = grid.len();
        if let Some(m) = maps.iter().find(|m| m.len() != n) {
            return Err(PbError::Backend(format!(
                "a map has {} values, grid has {n} nodes",
                m.len()
            )));
        }
        let [eps_x, eps_y, eps_z, kappa_sq] = maps.map(T::into_host);
        let maps = CoefficientMaps {
            eps_x: assembly(eps_x)?,
            eps_y: assembly(eps_y)?,
            eps_z: assembly(eps_z)?,
            kappa_sq: assembly(kappa_sq)?,
        };
        let op = StencilOperator::new(grid, maps);
        let inv_diag = assembly(T::from_host(op.inv_diag.clone()).ok())?;
        Ok((T::prepare(Arc::new(op)), inv_diag))
    }

    fn source_rhs(&self, src: f64, q: &Self::Vector, d: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        assembly((T::NAME == "f64").then_some(()))?;
        let src = T::from_f64(src);
        y.par_iter_mut()
            .zip(q.par_iter().zip(d.par_iter()))
            .for_each(|(y, (q, d))| *y = if *d == T::ZERO { T::ZERO } else { src * *q - *y });
        Ok(())
    }

    fn operator_to_host(&self, op: &Self::Operator) -> Result<StencilOperator> {
        assembly(T::fine_matrix(op).cloned())
    }

    fn boltzmann_slope(
        &self,
        ions: &[[f64; 3]],
        u: &Self::Vector,
        kappa_sq: &Self::Vector,
        slope: &mut Self::Vector,
    ) -> Result<()> {
        assembly((T::NAME == "f64").then_some(()))?;
        slope
            .par_iter_mut()
            .with_min_len(CHUNK)
            .zip(u.par_iter().zip(kappa_sq.par_iter()))
            .for_each(|(s, (u, k))| *s = T::from_f64(crate::nonlinear::slope_at(ions, u.to_f64(), k.to_f64())));
        Ok(())
    }

    fn boltzmann_residual(
        &self,
        ions: &[[f64; 3]],
        vol: f64,
        dims: [usize; 3],
        u: &Self::Vector,
        kappa_sq: &Self::Vector,
        slope: &Self::Vector,
        f: &Self::Vector,
        y: &mut Self::Vector,
    ) -> Result<()> {
        assembly((T::NAME == "f64").then_some(()))?;
        y.par_iter_mut().with_min_len(CHUNK).enumerate().for_each(|(c, y)| {
            let r = crate::nonlinear::residual_at(
                ions,
                vol,
                u[c].to_f64(),
                kappa_sq[c].to_f64(),
                slope[c].to_f64(),
                f[c].to_f64(),
                crate::nonlinear::is_interior(dims, c),
                y.to_f64(),
            );
            *y = T::from_f64(r);
        });
        Ok(())
    }

    fn gather(&self, v: &Self::Vector, idx: &[usize]) -> Result<Vec<f64>> {
        assembly((T::NAME == "f64").then_some(()))?;
        crate::backend::check_gather(v.len(), idx)?;
        Ok(idx.par_iter().with_min_len(CHUNK).map(|&c| v[c].to_f64()).collect())
    }
}
