//! Multithreaded CPU backend (rayon). Always available; it is also the
//! reference implementation every accelerator is validated against.

use crate::backend::Backend;
use crate::error::Result;
use crate::multigrid::{self, Stencil27, Transfer};
use crate::operator::StencilOperator;
use rayon::prelude::*;
use std::sync::Arc;

/// Fixed chunk size => reduction order independent of the thread count.
const CHUNK: usize = 1 << 14;

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    type Vector = Vec<f64>;
    type Operator = Arc<StencilOperator>;
    type Coarse = Arc<Stencil27>;

    fn platform(&self) -> &'static str {
        "CPU"
    }

    fn device_name(&self) -> String {
        format!("CPU ({} threads)", rayon::current_num_threads())
    }

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator> {
        Ok(op)
    }

    fn prepare_coarse(&self, op: Arc<Stencil27>) -> Result<Self::Coarse> {
        Ok(op)
    }

    fn upload(&self, host: &[f64]) -> Result<Self::Vector> {
        Ok(host.to_vec())
    }

    fn zeros(&self, n: usize) -> Result<Self::Vector> {
        Ok(vec![0.0; n])
    }

    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>> {
        Ok(v.clone())
    }

    fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()> {
        host.copy_from_slice(v);
        Ok(())
    }

    fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()> {
        v.copy_from_slice(host);
        Ok(())
    }

    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        op.apply(x, y);
        Ok(())
    }

    fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        op.apply(x, y);
        Ok(())
    }

    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64> {
        let partial: Vec<f64> = a
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>())
            .collect();
        Ok(partial.iter().sum())
    }

    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut().zip(x.par_iter()).for_each(|(y, x)| *y += alpha * x);
        Ok(())
    }

    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut().zip(x.par_iter()).for_each(|(y, x)| *y = x + beta * *y);
        Ok(())
    }

    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(y, x)| *y = alpha * x + beta * *y);
        Ok(())
    }

    fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        y.par_iter_mut()
            .zip(x.par_iter().zip(d.par_iter()))
            .for_each(|(y, (x, d))| *y = (alpha * d) * x);
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
