//! Multithreaded CPU backend (rayon). Always available; it is also the
//! reference implementation every accelerator is validated against.
//!
//! [`CpuBackend`] works in f64. [`CpuSingle`] is its f32 twin used by the
//! mixed-precision multigrid preconditioner. Both are one generic
//! implementation; for f64 it performs exactly the operations of the
//! original f64-only code.

use crate::backend::Backend;
use crate::error::Result;
use crate::multigrid::{self, Coarse27, Stencil27, Transfer};
use crate::operator::{Stencil7, StencilOperator};
use crate::real::Real;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

/// Fixed chunk size => reduction order independent of the thread count.
const CHUNK: usize = 1 << 14;

/// How operators are held for a given precision: f64 shares the host
/// operators, f32 keeps rounded copies.
pub trait CpuReal: Real {
    type Op: Send + Sync;
    type Coarse: Send + Sync;
    fn prepare(op: Arc<StencilOperator>) -> Self::Op;
    fn prepare_coarse(op: Arc<Stencil27>) -> Self::Coarse;
    fn apply(op: &Self::Op, x: &[Self], y: &mut [Self]);
    fn apply_coarse(op: &Self::Coarse, x: &[Self], y: &mut [Self]);
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
            .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.to_f64() * q.to_f64()).sum::<f64>())
            .collect();
        Ok(partial.iter().sum())
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
}
