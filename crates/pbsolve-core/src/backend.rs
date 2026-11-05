//! The minimal set of kernels an accelerator must provide.
//!
//! The Krylov solver in [`crate::solver`] and the multigrid V-cycle in
//! [`crate::multigrid`] are written only against this trait, so CPU, CUDA
//! (and any future HIP/Metal/SYCL backend) share one algorithm.
//! Implementations MUST make `dot` deterministic (fixed reduction order) and
//! evaluate every element-wise expression exactly as documented here, so that
//! all backends round identically.

use crate::error::Result;
use crate::multigrid::{Stencil27, Transfer};
use crate::operator::StencilOperator;
use std::sync::Arc;

pub trait Backend {
    /// Device-resident vector of f64.
    type Vector;
    /// Device-resident copy of the fine-grid 7-point operator.
    type Operator;
    /// Device-resident copy of a coarse-grid 27-point (Galerkin) operator.
    type Coarse;
    /// Single-precision twin on the same device (and CUDA stream) whose
    /// vectors hold f32. It runs the multigrid V-cycle of
    /// [`crate::params::Preconditioner::MultigridF32`]; its own `Single` is
    /// itself. Scalars and host data stay f64 in the interface and are
    /// rounded to f32 by the twin.
    type Single: Backend;

    /// Human-readable platform name, e.g. "CPU" or "CUDA".
    fn platform(&self) -> &'static str;
    /// Device description for logging, e.g. "NVIDIA A100 (cc 8.0)".
    fn device_name(&self) -> String;

    /// A handle to the single-precision twin.
    fn single(&self) -> Result<Self::Single>;
    /// y_i ← round_f32(x_i)
    fn demote(&self, x: &Self::Vector, y: &mut <Self::Single as Backend>::Vector) -> Result<()>;
    /// y_i ← x_i (exact widening)
    fn promote(&self, x: &<Self::Single as Backend>::Vector, y: &mut Self::Vector) -> Result<()>;

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator>;
    fn prepare_coarse(&self, op: Arc<Stencil27>) -> Result<Self::Coarse>;
    fn upload(&self, host: &[f64]) -> Result<Self::Vector>;
    fn zeros(&self, n: usize) -> Result<Self::Vector>;
    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>>;
    /// Copies a device vector into an existing host buffer of the same length.
    fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()>;
    /// Copies a host buffer into an existing device vector of the same length.
    fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()>;

    /// y = A x with the 7-point operator (boundary rows zero).
    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// y = A x with a 27-point operator (boundary rows zero).
    fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// Σ a_i b_i, deterministic.
    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64>;
    /// (a·b, a·c) in one pass where the backend can fuse it (one device
    /// synchronisation instead of two). Each value must equal `dot` bit for bit.
    fn dot2(&self, a: &Self::Vector, b: &Self::Vector, c: &Self::Vector) -> Result<(f64, f64)> {
        Ok((self.dot(a, b)?, self.dot(a, c)?))
    }
    /// y_i ← y_i + α x_i
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// y_i ← x_i + β y_i
    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
    /// y_i ← α x_i + β y_i
    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
    /// y_i ← (α d_i) x_i  (diagonal scaling: Jacobi, smoothers)
    fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// y ← x
    fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// coarse ← Pᵀ fine ([`crate::multigrid::restrict`]).
    fn restrict(&self, t: &Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()>;
    /// fine ← fine + P coarse ([`crate::multigrid::prolong_add`]).
    fn prolong_add(&self, t: &Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()>;
}
