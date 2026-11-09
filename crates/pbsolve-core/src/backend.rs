//! The minimal set of kernels an accelerator must provide.
//!
//! The Krylov solver in [`crate::solver`] and the multigrid V-cycle in
//! [`crate::multigrid`] are written only against this trait, so CPU, CUDA
//! (and any future HIP/Metal/SYCL backend) share one algorithm.
//! Implementations MUST make `dot` deterministic (fixed reduction order) and
//! evaluate every element-wise expression exactly as documented here, so that
//! all backends round identically.

use crate::error::Result;
use crate::grid::Grid;
use crate::multigrid::{LevelOp, Stencil27, Transfer};
use crate::operator::StencilOperator;
use crate::surface::Sphere;
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
    /// [`upload`](Self::upload) of a buffer the caller no longer needs (the
    /// f64 CPU backend keeps it without a copy).
    fn upload_owned(&self, host: Vec<f64>) -> Result<Self::Vector> {
        self.upload(&host)
    }
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

    // Steps of the multigrid smoother ([`crate::multigrid`]). Each one is a
    // sequence of the operations above; the defaults run that sequence
    // ([`residual_by_steps`], [`chebyshev_first_by_steps`],
    // [`chebyshev_next_by_steps`]). A backend may fuse it into one pass if it
    // evaluates every element with the same expressions, so that the results
    // are identical bit for bit; the scratch vector `t` may then be left
    // untouched, and callers must not read it afterwards.

    /// Residual of a level: r_i ← s_i + (−1)·(A x)_i, where s = `b`, or s = `r`
    /// itself when `b` is `None`. As `apply` (or `apply_coarse`) into `t`,
    /// `copy(b, r)` and `axpy(−1, t, r)`.
    fn residual(
        &self,
        op: LevelOp<'_, Self>,
        x: &Self::Vector,
        b: Option<&Self::Vector>,
        t: &mut Self::Vector,
        r: &mut Self::Vector,
    ) -> Result<()>
    where
        Self: Sized,
    {
        residual_by_steps(self, op, x, b, t, r)
    }
    /// First Chebyshev step: d_i ← (s·w_i)·r_i, then x_i ← d_i (`overwrite`)
    /// or x_i ← x_i + 1·d_i. As `scale(s, w, r, d)` and `copy(d, x)` or
    /// `axpy(1, d, x)`.
    fn chebyshev_first(
        &self,
        s: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
        overwrite: bool,
    ) -> Result<()>
    where
        Self: Sized,
    {
        chebyshev_first_by_steps(self, s, w, r, d, x, overwrite)
    }
    /// Later Chebyshev step: t_i ← (c1·w_i)·r_i, d_i ← 1·t_i + c2·d_i,
    /// x_i ← x_i + 1·d_i. As `scale(c1, w, r, t)`, `axpby(1, t, c2, d)` and
    /// `axpy(1, d, x)`.
    #[allow(clippy::too_many_arguments)]
    fn chebyshev_next(
        &self,
        c1: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        c2: f64,
        t: &mut Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
    ) -> Result<()>
    where
        Self: Sized,
    {
        chebyshev_next_by_steps(self, c1, w, r, c2, t, d, x)
    }

    // Multigrid setup ([`crate::multigrid::Levels::build`]). Only f64
    // backends build hierarchies; their twins return an error.

    /// The Galerkin operator Pᵀ A P of a level and its inverse diagonal,
    /// bit-identical to [`crate::multigrid::galerkin`] (same operation order).
    fn galerkin(&self, t: &Transfer, fine: LevelOp<'_, Self>) -> Result<(Self::Coarse, Self::Vector)>
    where
        Self: Sized;
    /// Gershgorin bound λmax(D⁻¹A) of a level, equal to
    /// [`crate::multigrid::LevelMatrix::gershgorin_lmax`].
    fn gershgorin_lmax(&self, level: LevelOp<'_, Self>) -> Result<f64>
    where
        Self: Sized;
    /// A host copy of a coarse operator (for the coarsest-level factorisation).
    fn coarse_to_host(&self, op: &Self::Coarse) -> Result<Stencil27>;
    /// The fine operator rounded to f32 for the twin, as the twin's
    /// [`prepare`](Self::prepare) of the host operator would round it.
    fn demote_operator(&self, op: &Self::Operator) -> Result<<Self::Single as Backend>::Operator>;
    /// A coarse operator rounded to f32 for the twin, as the twin's
    /// [`prepare_coarse`](Self::prepare_coarse) would round it.
    fn demote_coarse(&self, op: &Self::Coarse) -> Result<<Self::Single as Backend>::Coarse>;

    // Assembly of the linear system ([`crate::pb`]). Only f64 backends
    // assemble; their twins return an error.

    /// One map per entry of `shifts` on the grid shifted by it (Å): `fill`
    /// everywhere, then for each pass `(spheres, value)` in order, `value` at
    /// every node within (≤) one of the spheres. Bit-identical to
    /// [`crate::surface::mark_map`].
    fn mark_maps(
        &self,
        grid: &Grid,
        shifts: &[[f64; 3]],
        fill: f64,
        passes: &[(&[Sphere], f64)],
    ) -> Result<Vec<Self::Vector>>;
    /// A vector of length `n`, zero except `v[idx[m]] = values[m]` (distinct indices).
    fn scatter(&self, n: usize, idx: &[usize], values: &[f64]) -> Result<Self::Vector>;
    /// The 7-point operator of `grid` from its maps `[eps_x, eps_y, eps_z,
    /// kappa_sq]` and its inverse diagonal (0 on boundary nodes), as
    /// [`StencilOperator::new`] computes them.
    fn assemble_operator(&self, grid: &Grid, maps: [Self::Vector; 4]) -> Result<(Self::Operator, Self::Vector)>;
    /// y_i ← 0 where d_i = 0, else src·q_i − y_i: with y = A g and d = 1/diag(A)
    /// this is the right-hand side of [`crate::operator::assemble_rhs`].
    fn source_rhs(&self, src: f64, q: &Self::Vector, d: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
    /// A host copy of a fine operator (for grids too small to coarsen, whose
    /// multigrid is a single factorised level).
    fn operator_to_host(&self, op: &Self::Operator) -> Result<StencilOperator>;
    /// The values `v[idx[m]]` in the order of `idx` (indices below the length
    /// of `v`, repetitions allowed). The pipeline reads the potential at the
    /// charges and at the nodes the next focusing level interpolates instead
    /// of downloading it; backends override the default, which downloads `v`.
    fn gather(&self, v: &Self::Vector, idx: &[usize]) -> Result<Vec<f64>> {
        let all = self.download(v)?;
        check_gather(all.len(), idx)?;
        Ok(idx.iter().map(|&c| all[c]).collect())
    }
    /// Waits until all queued work has finished (stage timings). Backends that
    /// compute synchronously keep the default no-op.
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    /// Whether the kernels run on a device and leave the host's threads
    /// mostly idle (CUDA). The pipeline then builds host-side inputs of a
    /// later stage on those threads in the background. Results do not
    /// depend on it. Default: false (kernels run on the host's threads).
    fn runs_on_device(&self) -> bool {
        false
    }
    /// Keeps up to `bytes` more of the memory that a calculation frees
    /// reserved for its later allocations, until the matching
    /// [`release_memory`](Self::release_memory). A backend whose allocator
    /// returns freed memory to the system (the CUDA memory pool) would
    /// otherwise reserve it again for every focusing level. Calls may
    /// overlap (concurrent workers); the bytes add up. Results do not change.
    /// Use it through [`MemoryHold`]. Default: no-op.
    fn retain_memory(&self, _bytes: u64) -> Result<()> {
        Ok(())
    }
    /// Ends a [`retain_memory`](Self::retain_memory) of `bytes`: freed memory
    /// that no calculation retains any more is returned to the system.
    /// Default: no-op.
    fn release_memory(&self, _bytes: u64) -> Result<()> {
        Ok(())
    }
}

/// Memory retained with [`Backend::retain_memory`] for one calculation.
/// [`release`](Self::release) ends it and reports errors; dropping it (on an
/// error path) ends it too and ignores them.
#[must_use = "the memory is released when the hold is dropped"]
pub struct MemoryHold<'a, B: Backend> {
    be: Option<&'a B>,
    bytes: u64,
}

impl<'a, B: Backend> MemoryHold<'a, B> {
    pub fn new(be: &'a B, bytes: u64) -> Result<Self> {
        be.retain_memory(bytes)?;
        Ok(Self { be: Some(be), bytes })
    }

    pub fn release(mut self) -> Result<()> {
        match self.be.take() {
            Some(be) => be.release_memory(self.bytes),
            None => Ok(()),
        }
    }
}

impl<B: Backend> Drop for MemoryHold<'_, B> {
    fn drop(&mut self) {
        if let Some(be) = self.be.take() {
            // An error here is reported by the backend's next operation.
            let _ = be.release_memory(self.bytes);
        }
    }
}

/// [`Backend::residual`] as its sequence of operations (the default, and the
/// reference fused implementations are tested against).
pub fn residual_by_steps<B: Backend>(
    be: &B,
    op: LevelOp<'_, B>,
    x: &B::Vector,
    b: Option<&B::Vector>,
    t: &mut B::Vector,
    r: &mut B::Vector,
) -> Result<()> {
    match op {
        LevelOp::Fine(o) => be.apply(o, x, t)?,
        LevelOp::Coarse(o) => be.apply_coarse(o, x, t)?,
    }
    if let Some(b) = b {
        be.copy(b, r)?;
    }
    be.axpy(-1.0, t, r)
}

/// [`Backend::chebyshev_first`] as its sequence of operations.
pub fn chebyshev_first_by_steps<B: Backend>(
    be: &B,
    s: f64,
    w: &B::Vector,
    r: &B::Vector,
    d: &mut B::Vector,
    x: &mut B::Vector,
    overwrite: bool,
) -> Result<()> {
    be.scale(s, w, r, d)?;
    if overwrite {
        be.copy(d, x)
    } else {
        be.axpy(1.0, d, x)
    }
}

/// [`Backend::chebyshev_next`] as its sequence of operations.
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_next_by_steps<B: Backend>(
    be: &B,
    c1: f64,
    w: &B::Vector,
    r: &B::Vector,
    c2: f64,
    t: &mut B::Vector,
    d: &mut B::Vector,
    x: &mut B::Vector,
) -> Result<()> {
    be.scale(c1, w, r, t)?;
    be.axpby(1.0, t, c2, d)?;
    be.axpy(1.0, d, x)
}

/// Checks the arguments of [`Backend::scatter`]: as many values as indices,
/// every index below `n`.
pub fn check_scatter(n: usize, idx: &[usize], values: &[f64]) -> Result<()> {
    if idx.len() != values.len() {
        return crate::error::invalid(format!("scatter: {} indices but {} values", idx.len(), values.len()));
    }
    if let Some(c) = idx.iter().find(|&&c| c >= n) {
        return crate::error::invalid(format!("scatter: index {c} outside a vector of {n} values"));
    }
    Ok(())
}

/// Checks the indices of [`Backend::gather`]: every index below `len`.
pub fn check_gather(len: usize, idx: &[usize]) -> Result<()> {
    if let Some(c) = idx.iter().find(|&&c| c >= len) {
        return crate::error::invalid(format!("gather: index {c} outside a vector of {len} values"));
    }
    Ok(())
}
