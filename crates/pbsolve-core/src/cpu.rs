//! Multithreaded CPU backend (rayon). Always available; it is also the
//! reference implementation every accelerator is validated against.

use crate::backend::Backend;
use crate::error::Result;
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

    fn platform(&self) -> &'static str {
        "CPU"
    }

    fn device_name(&self) -> String {
        format!("CPU ({} threads)", rayon::current_num_threads())
    }

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator> {
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

    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        op.apply(x, y);
        Ok(())
    }

    fn precondition(&self, op: &Self::Operator, r: &Self::Vector, z: &mut Self::Vector) -> Result<()> {
        z.par_iter_mut()
            .zip(r.par_iter())
            .zip(op.inv_diag.par_iter())
            .for_each(|((z, r), d)| *z = r * d);
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
}
