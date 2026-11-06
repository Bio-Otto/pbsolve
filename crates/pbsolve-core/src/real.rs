//! Floating-point types the solver kernels are written for.
//!
//! The PB problem, the outer Krylov iteration and every energy are always in
//! `f64`. The multigrid preconditioner may run in `f32`
//! ([`crate::params::Preconditioner::MultigridF32`]), so the kernels are
//! generic over [`Real`]. For `f64` every conversion is the identity and
//! the generic kernels perform exactly the operations of the original `f64`
//! code.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

pub trait Real:
    Copy
    + Send
    + Sync
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    /// "f64" or "f32" (kernel name suffix, diagnostics).
    const NAME: &'static str;
    /// Rounds to nearest (identity for f64).
    fn from_f64(v: f64) -> Self;
    /// Exact widening (identity for f64).
    fn to_f64(self) -> f64;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const NAME: &'static str = "f64";
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const NAME: &'static str = "f32";
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Element-wise conversion of a slice (parallel).
pub fn convert<S: Real, D: Real>(src: &[S]) -> Vec<D> {
    use rayon::prelude::*;
    src.par_iter().map(|v| D::from_f64(v.to_f64())).collect()
}

/// `n` zeros written in parallel. A fresh large allocation is mapped lazily,
/// page by page; a device-to-host copy into it would fault every page in on
/// one thread (~1.4 GB/s here). Writing the zeros first spreads that over
/// the thread pool.
pub fn zeros_touched<T: Real>(n: usize) -> Vec<T> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(|_| T::ZERO).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(f64::from_f64(0.1), 0.1);
        assert_eq!(f32::from_f64(0.1), 0.1f32);
        assert_eq!(0.1f32.to_f64(), 0.1f32 as f64);
        // Transfer weights and their products are exact in f32.
        for w in [0.125, 0.25, 0.5, 1.0] {
            assert_eq!(f32::from_f64(w).to_f64(), w);
        }
        let v: Vec<f32> = convert(&[1.0f64, -2.5, 1e-3]);
        assert_eq!(v, [1.0f32, -2.5, 1e-3f32]);
    }
}
