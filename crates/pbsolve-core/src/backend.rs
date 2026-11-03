//! The minimal set of kernels an accelerator must provide.
//!
//! The Krylov solver in [`crate::solver`] is written only against this trait,
//! so CPU, CUDA (and any future HIP/Metal/SYCL backend) share one algorithm.
//! Implementations MUST make `dot` deterministic (fixed reduction order).

use crate::error::Result;
use crate::operator::StencilOperator;
use std::sync::Arc;

pub trait Backend {
    /// Device-resident vector of f64.
    type Vector;
    /// Device-resident copy of the stencil coefficients.
    type Operator;

    /// Human-readable platform name, e.g. "CPU" or "CUDA".
    fn platform(&self) -> &'static str;
    /// Device description for logging, e.g. "NVIDIA A100 (cc 8.0)".
    fn device_name(&self) -> String;

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator>;
    fn upload(&self, host: &[f64]) -> Result<Self::Vector>;
    fn zeros(&self, n: usize) -> Result<Self::Vector>;
    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>>;

    /// y = A x (boundary rows zero).
    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// z = D⁻¹ r (Jacobi).
    fn precondition(&self, op: &Self::Operator, r: &Self::Vector, z: &mut Self::Vector) -> Result<()>;
    /// Σ a_i b_i, deterministic.
    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64>;
    /// y ← y + α x
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// y ← x + β y
    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
}
