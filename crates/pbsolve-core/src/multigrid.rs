//! Geometric multigrid preconditioner for the LPBE operator.
//!
//! * **Coarsening** is vertex based and done per axis: an axis with `n >= 5`
//!   nodes becomes `n/2 + 1` nodes, coarse node `I` sitting on fine node `2I`
//!   (for even `n` the last coarse node sits on the fine boundary node `n-1`).
//!   Axes with fewer nodes are kept (semi-coarsening), so every grid size works.
//! * **Transfer:** trilinear prolongation `P` (interior to interior; boundary
//!   values are zero in the correction equation) and restriction `R = Pᵀ`.
//! * **Coarse operators:** Galerkin `A_c = Pᵀ A P` (27-point stencils), made
//!   exactly symmetric. Galerkin operators keep the variational property that
//!   makes multigrid robust for the strongly discontinuous dielectric
//!   coefficient [Alcouffe et al. 1981; Holst & Saied 1993].
//! * **Smoother:** a Chebyshev polynomial in `D⁻¹A` [Adams et al. 2003], used
//!   before and after the coarse correction. It needs only operator
//!   applications and diagonal scaling (no colouring, same kernels on every
//!   backend) and keeps the V-cycle symmetric.
//! * **Coarsest level:** exact solve with a banded Cholesky factorisation on
//!   the host.
//!
//! The V-cycle is therefore a fixed symmetric positive definite operator and
//! is used as the preconditioner of CG (MGCG, [Tatebe 1993]).
//!
//! The setup is defined here on the host ([`Hierarchy::build`], the
//! reference). [`Levels::build`] runs the same steps through the backend's
//! setup methods, so a GPU builds the Galerkin operators, inverse diagonals
//! and Gershgorin bounds in device memory, bit-identical to the host; only
//! the small coarsest level is factorised on the host. The cycle itself is
//! written once against [`Backend`].

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::operator::StencilOperator;
use crate::real::{convert, Real};
use rayon::prelude::*;
use std::sync::Arc;

/// Axes with at least this many nodes are coarsened.
pub const MIN_COARSEN_NODES: usize = 5;
/// Coarsening stops once a level has at most this many unknowns.
pub const COARSEST_MAX_UNKNOWNS: usize = 1000;
/// Largest banded factor (number of stored values) accepted for the coarsest level.
pub const MAX_BAND_VALUES: usize = 16_000_000;

/// Index of stencil offset (dx, dy, dz) ∈ {-1,0,1}³; 13 is the centre and
/// `26 - o` is the opposite offset.
#[inline]
pub const fn offset_index(dx: i32, dy: i32, dz: i32) -> usize {
    ((dx + 1) + 3 * ((dy + 1) + 3 * (dz + 1))) as usize
}

/// The offset (dx, dy, dz) of stencil entry `o`.
#[inline]
pub const fn offset_of(o: usize) -> [i32; 3] {
    let o = o as i32;
    [o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1]
}

#[inline]
fn is_interior(i: usize, j: usize, k: usize, d: [usize; 3]) -> bool {
    i > 0 && j > 0 && k > 0 && i + 1 < d[0] && j + 1 < d[1] && k + 1 < d[2]
}

/// Number of interior nodes (unknowns) of a grid.
pub fn interior_count(d: [usize; 3]) -> usize {
    d.iter().map(|&n| n.saturating_sub(2)).product()
}

/// Dimensions of the coarse levels (1, 2, …) of the hierarchy of a `fine`
/// grid: the level sequence of [`Hierarchy::build`] and [`Levels::build`]
/// depends on the dimensions only.
pub fn coarse_dims(fine: [usize; 3]) -> Vec<[usize; 3]> {
    let mut levels = Vec::new();
    let mut dims = fine;
    while interior_count(dims) > COARSEST_MAX_UNKNOWNS {
        let Some(t) = Transfer::new(dims) else { break };
        dims = t.coarse;
        levels.push(dims);
    }
    levels
}

/// Dimensions of the coarsest level of the hierarchy of a `fine` grid (the
/// fine grid itself if it is not coarsened).
pub fn coarsest_dims(fine: [usize; 3]) -> [usize; 3] {
    coarse_dims(fine).last().copied().unwrap_or(fine)
}

/// False where [`Hierarchy::build`] and [`Levels::build`] return `None`
/// (coarsest level too large for the banded factorisation).
pub fn hierarchy_fits(fine: [usize; 3]) -> bool {
    BandedCholesky::storage(coarsest_dims(fine)) <= MAX_BAND_VALUES
}

// ------------------------------------------------------------------ transfer

/// Grid transfer between one level and the next coarser one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transfer {
    pub fine: [usize; 3],
    pub coarse: [usize; 3],
    /// Axes that are coarsened (the others map node `i` to node `i`).
    pub coarsen: [bool; 3],
}

impl Transfer {
    /// The transfer to the next coarser grid, or `None` if no axis can be coarsened.
    pub fn new(fine: [usize; 3]) -> Option<Self> {
        let coarsen = fine.map(|n| n >= MIN_COARSEN_NODES);
        if !coarsen.iter().any(|&c| c) {
            return None;
        }
        let coarse = [0, 1, 2].map(|a| if coarsen[a] { fine[a] / 2 + 1 } else { fine[a] });
        Some(Self { fine, coarse, coarsen })
    }
}

/// Coarse nodes (index, weight) that fine interior node `f` interpolates from.
#[inline]
pub fn prolong_axis(f: usize, coarsen: bool) -> ([usize; 2], [f64; 2], usize) {
    if !coarsen {
        ([f, 0], [1.0, 0.0], 1)
    } else if f % 2 == 0 {
        ([f / 2, 0], [1.0, 0.0], 1)
    } else {
        ([(f - 1) / 2, f.div_ceil(2)], [0.5, 0.5], 2)
    }
}

/// Fine interior nodes (index, weight) that coarse interior node `c`
/// restricts from; `n` is the number of fine nodes on this axis.
#[inline]
pub fn restrict_axis(c: usize, coarsen: bool, n: usize) -> ([usize; 3], [f64; 3], usize) {
    if !coarsen {
        return ([c, 0, 0], [1.0, 0.0, 0.0], 1);
    }
    let mut idx = [0usize; 3];
    let mut w = [0.0f64; 3];
    let mut m = 0;
    for (f, wf) in [(2 * c - 1, 0.5), (2 * c, 1.0), (2 * c + 1, 0.5)] {
        if f > 0 && f + 1 < n {
            idx[m] = f;
            w[m] = wf;
            m += 1;
        }
    }
    (idx, w, m)
}

/// `coarse = Pᵀ fine` on interior coarse nodes, 0 on coarse boundary nodes.
/// Reads only interior fine nodes. CUDA: `pb_restrict` (same summation order).
pub fn restrict<T: Real>(t: &Transfer, fine: &[T], coarse: &mut [T]) {
    let [nx, ny, nz] = t.fine;
    let [mx, my, _] = t.coarse;
    coarse.par_chunks_mut(mx * my).enumerate().for_each(|(kc, plane)| {
        for jc in 0..my {
            for ic in 0..mx {
                let out = &mut plane[ic + mx * jc];
                if !is_interior(ic, jc, kc, t.coarse) {
                    *out = T::ZERO;
                    continue;
                }
                let (fx, wx, mx_) = restrict_axis(ic, t.coarsen[0], nx);
                let (fy, wy, my_) = restrict_axis(jc, t.coarsen[1], ny);
                let (fz, wz, mz_) = restrict_axis(kc, t.coarsen[2], nz);
                let (wx, wy, wz) = (wx.map(T::from_f64), wy.map(T::from_f64), wz.map(T::from_f64));
                let mut acc = T::ZERO;
                for c in 0..mz_ {
                    for b in 0..my_ {
                        for a in 0..mx_ {
                            let w = wx[a] * wy[b] * wz[c];
                            acc += w * fine[fx[a] + nx * (fy[b] + ny * fz[c])];
                        }
                    }
                }
                *out = acc;
            }
        }
    });
}

/// `fine += P coarse` on interior fine nodes (boundary nodes untouched).
/// Coarse boundary values must be zero. CUDA: `pb_prolong_add`.
pub fn prolong_add<T: Real>(t: &Transfer, coarse: &[T], fine: &mut [T]) {
    let [nx, ny, _] = t.fine;
    let [mx, my, _] = t.coarse;
    fine.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                if !is_interior(i, j, k, t.fine) {
                    continue;
                }
                let (cx, wx, mx_) = prolong_axis(i, t.coarsen[0]);
                let (cy, wy, my_) = prolong_axis(j, t.coarsen[1]);
                let (cz, wz, mz_) = prolong_axis(k, t.coarsen[2]);
                let (wx, wy, wz) = (wx.map(T::from_f64), wy.map(T::from_f64), wz.map(T::from_f64));
                let mut acc = T::ZERO;
                for c in 0..mz_ {
                    for b in 0..my_ {
                        for a in 0..mx_ {
                            let w = wx[a] * wy[b] * wz[c];
                            acc += w * coarse[cx[a] + mx * (cy[b] + my * cz[c])];
                        }
                    }
                }
                let out = &mut plane[i + nx * j];
                *out += acc;
            }
        }
    });
}

// ------------------------------------------------------------------ operators

/// A 27-point operator on a grid: `y_c = Σ_o coef[o·n + c] · x[c + offset(o)]`
/// on interior nodes, 0 on boundary nodes. Coefficients are stored per offset
/// (structure of arrays, coalesced on GPUs) and are exactly symmetric:
/// `coef[o][c] == coef[26-o][c + offset(o)]`.
#[derive(Clone, Debug)]
pub struct Stencil27 {
    pub dims: [usize; 3],
    pub coef: Vec<f64>,
    /// 1/diag on interior nodes, 0 on boundary nodes.
    pub inv_diag: Vec<f64>,
}

impl Stencil27 {
    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// y = A x (CUDA: `pb_apply27`, same summation order).
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        apply27(self.dims, &self.coef, x, y);
    }
}

/// A 27-point operator in another precision (multigrid in f32).
#[derive(Clone, Debug)]
pub struct Coarse27<T: Real> {
    pub dims: [usize; 3],
    pub coef: Vec<T>,
}

impl<T: Real> Coarse27<T> {
    /// Rounds every coefficient of `op` to `T`.
    pub fn from_stencil(op: &Stencil27) -> Self {
        Self {
            dims: op.dims,
            coef: convert(&op.coef),
        }
    }

    pub fn apply(&self, x: &[T], y: &mut [T]) {
        apply27(self.dims, &self.coef, x, y);
    }
}

/// y = A x for a 27-point operator in any precision (CUDA: `pb_apply27`).
pub fn apply27<T: Real>(dims: [usize; 3], coef: &[T], x: &[T], y: &mut [T]) {
    apply27_with(dims, coef, x, y, |_, _, v| v);
}

/// `out[c] ← combine(out[c], c, v_c)` for every node c, where v_c is the value
/// [`apply27`] writes ((A x)_c, 0 on boundary nodes); see
/// [`crate::operator::apply7_with`].
pub fn apply27_with<T: Real, F>(dims: [usize; 3], coef: &[T], x: &[T], out: &mut [T], combine: F)
where
    F: Fn(T, usize, T) -> T + Sync,
{
    let [nx, ny, _] = dims;
    let n = dims.iter().product::<usize>();
    let off = stencil_offsets(dims);
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                let o = &mut plane[i + nx * j];
                let c = i + nx * (j + ny * k);
                if !is_interior(i, j, k, dims) {
                    *o = combine(*o, c, T::ZERO);
                    continue;
                }
                let mut acc = T::ZERO;
                for (m, d) in off.iter().enumerate() {
                    acc += coef[m * n + c] * x[(c as isize + d) as usize];
                }
                *o = combine(*o, c, acc);
            }
        }
    });
}

/// Linear index offsets of the 27 stencil entries.
pub fn stencil_offsets(dims: [usize; 3]) -> [isize; 27] {
    let (sy, sz) = (dims[0] as isize, (dims[0] * dims[1]) as isize);
    std::array::from_fn(|o| {
        let [dx, dy, dz] = offset_of(o);
        dx as isize + sy * dy as isize + sz * dz as isize
    })
}

/// Read access to the matrix entries of a level (fine 7-point or coarse 27-point).
#[derive(Clone, Copy)]
pub enum LevelMatrix<'a> {
    Seven(&'a StencilOperator),
    TwentySeven(&'a Stencil27),
}

impl LevelMatrix<'_> {
    pub fn dims(&self) -> [usize; 3] {
        match self {
            Self::Seven(op) => op.dims,
            Self::TwentySeven(s) => s.dims,
        }
    }

    /// Calls `f(o, a)` for every stencil entry `a` (offset index `o`) of
    /// interior row `c`, zero entries possibly included.
    #[inline]
    pub fn for_each_entry(&self, c: usize, mut f: impl FnMut(usize, f64)) {
        match self {
            Self::Seven(op) => {
                let [nx, ny, _] = op.dims;
                let (sy, sz) = (nx, nx * ny);
                let [cx, cy, cz] = op.coef;
                let (exm, exp) = (op.eps_x[c - 1], op.eps_x[c]);
                let (eym, eyp) = (op.eps_y[c - sy], op.eps_y[c]);
                let (ezm, ezp) = (op.eps_z[c - sz], op.eps_z[c]);
                let diag = cx * (exm + exp) + cy * (eym + eyp) + cz * (ezm + ezp) + op.kvol[c];
                f(offset_index(0, 0, -1), -cz * ezm);
                f(offset_index(0, -1, 0), -cy * eym);
                f(offset_index(-1, 0, 0), -cx * exm);
                f(13, diag);
                f(offset_index(1, 0, 0), -cx * exp);
                f(offset_index(0, 1, 0), -cy * eyp);
                f(offset_index(0, 0, 1), -cz * ezp);
            }
            Self::TwentySeven(s) => {
                let n = s.len();
                for o in 0..27 {
                    f(o, s.coef[o * n + c]);
                }
            }
        }
    }

    /// Gershgorin bound on the largest eigenvalue of D⁻¹A (≥ the true value).
    pub fn gershgorin_lmax(&self) -> f64 {
        let d = self.dims();
        let n = d[0] * d[1] * d[2];
        (0..n)
            .into_par_iter()
            .filter(|&c| is_interior(c % d[0], (c / d[0]) % d[1], c / (d[0] * d[1]), d))
            .map(|c| {
                let (mut sum, mut diag) = (0.0, 0.0);
                self.for_each_entry(c, |o, a| {
                    sum += a.abs();
                    if o == 13 {
                        diag = a;
                    }
                });
                sum / diag
            })
            .reduce(|| 0.0, f64::max)
    }
}

/// Galerkin coarse operator `Pᵀ A P`, symmetrised exactly.
pub fn galerkin(t: &Transfer, fine: LevelMatrix<'_>) -> Result<Stencil27> {
    let fd = t.fine;
    let cd = t.coarse;
    let (fnx, fny) = (fd[0], fd[1]);
    let n = cd[0] * cd[1] * cd[2];
    // Row-wise accumulation, one coarse node per task (deterministic).
    let rows: Vec<[f64; 27]> = (0..n)
        .into_par_iter()
        .map(|ci| {
            let (ic, jc, kc) = (ci % cd[0], (ci / cd[0]) % cd[1], ci / (cd[0] * cd[1]));
            let mut acc = [0.0f64; 27];
            if !is_interior(ic, jc, kc, cd) {
                return acc;
            }
            let ccoord = [ic, jc, kc];
            let (fx, wx, mx) = restrict_axis(ic, t.coarsen[0], fd[0]);
            let (fy, wy, my) = restrict_axis(jc, t.coarsen[1], fd[1]);
            let (fz, wz, mz) = restrict_axis(kc, t.coarsen[2], fd[2]);
            for c in 0..mz {
                for b in 0..my {
                    for a in 0..mx {
                        let wf = wx[a] * wy[b] * wz[c];
                        let fcoord = [fx[a], fy[b], fz[c]];
                        let f = fcoord[0] + fnx * (fcoord[1] + fny * fcoord[2]);
                        fine.for_each_entry(f, |o, val| {
                            if val == 0.0 {
                                return;
                            }
                            let d = offset_of(o);
                            let g = [0, 1, 2].map(|ax| (fcoord[ax] as i64 + d[ax] as i64) as usize);
                            if !is_interior(g[0], g[1], g[2], fd) {
                                return;
                            }
                            let px = prolong_axis(g[0], t.coarsen[0]);
                            let py = prolong_axis(g[1], t.coarsen[1]);
                            let pz = prolong_axis(g[2], t.coarsen[2]);
                            for r in 0..pz.2 {
                                for q in 0..py.2 {
                                    for p in 0..px.2 {
                                        let jcrd = [px.0[p], py.0[q], pz.0[r]];
                                        if !is_interior(jcrd[0], jcrd[1], jcrd[2], cd) {
                                            continue;
                                        }
                                        let w = px.1[p] * py.1[q] * pz.1[r];
                                        let off = [0, 1, 2].map(|ax| (jcrd[ax] as i64 - ccoord[ax] as i64) as i32);
                                        acc[offset_index(off[0], off[1], off[2])] += wf * val * w;
                                    }
                                }
                            }
                        });
                    }
                }
            }
            acc
        })
        .collect();
    // Symmetrise: entries (c, o) and (c + offset(o), 26 - o) become their mean.
    let offs = stencil_offsets(cd);
    let mut coef = vec![0.0; 27 * n];
    coef.par_chunks_mut(n).enumerate().for_each(|(o, out)| {
        for (c, v) in out.iter_mut().enumerate() {
            if rows[c][13] == 0.0 {
                continue; // boundary row
            }
            let partner = (c as isize + offs[o]) as usize;
            *v = 0.5 * (rows[c][o] + rows[partner][26 - o]);
        }
    });
    Stencil27::from_coef(cd, coef)
}

impl Stencil27 {
    /// A Galerkin operator from its symmetrised coefficients: computes the
    /// inverse diagonal and rejects a NaN or negative diagonal (the first one
    /// in node order is reported).
    pub fn from_coef(dims: [usize; 3], coef: Vec<f64>) -> Result<Self> {
        let n = dims[0] * dims[1] * dims[2];
        let mut inv_diag = vec![0.0; n];
        for (c, d) in inv_diag.iter_mut().enumerate() {
            let diag = coef[13 * n + c];
            if diag != 0.0 {
                if diag.is_nan() || diag < 0.0 {
                    return Err(non_positive_diagonal(diag, dims));
                }
                *d = 1.0 / diag;
            }
        }
        Ok(Self { dims, coef, inv_diag })
    }
}

/// The error of a Galerkin operator with diagonal `diag` (NaN or negative).
fn non_positive_diagonal(diag: f64, dims: [usize; 3]) -> PbError {
    PbError::Backend(format!(
        "multigrid: non-positive Galerkin diagonal {diag:.3e} on grid {dims:?}"
    ))
}

// ------------------------------------------------------------------ coarsest level

/// Banded Cholesky factor of a level matrix restricted to interior nodes
/// (natural x-fastest ordering), used as the exact coarsest-level solver.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedCholesky {
    dims: [usize; 3],
    n: usize,
    bw: usize,
    /// Row-major lower band: `l[i * (bw + 1) + (i - j)]` = L[i][j].
    l: Vec<f64>,
}

impl BandedCholesky {
    /// Grid dimensions of the factorised level.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Storage the factor of a grid would need (number of values).
    pub fn storage(dims: [usize; 3]) -> usize {
        let (ix, iy) = (dims[0].saturating_sub(2), dims[1].saturating_sub(2));
        interior_count(dims) * (2 + ix + ix * iy)
    }

    pub fn factor(m: LevelMatrix<'_>) -> Result<Self> {
        let dims = m.dims();
        let [ix, iy, iz] = dims.map(|n| n.saturating_sub(2));
        let n = ix * iy * iz;
        if n == 0 {
            return crate::error::invalid("multigrid: coarsest grid has no interior nodes");
        }
        let bw = (1 + ix + ix * iy).min(n - 1);
        let w = bw + 1;
        let mut l = vec![0.0; n * w];
        // Fill the lower band of A.
        for p in 0..n {
            let (i, j, k) = (p % ix + 1, (p / ix) % iy + 1, p / (ix * iy) + 1);
            let c = i + dims[0] * (j + dims[1] * k);
            m.for_each_entry(c, |o, a| {
                let [dx, dy, dz] = offset_of(o);
                let (qi, qj, qk) = (i as i64 + dx as i64, j as i64 + dy as i64, k as i64 + dz as i64);
                if qi < 1 || qj < 1 || qk < 1 || qi > ix as i64 || qj > iy as i64 || qk > iz as i64 {
                    return;
                }
                let q = (qi as usize - 1) + ix * ((qj as usize - 1) + iy * (qk as usize - 1));
                if q <= p {
                    l[p * w + (p - q)] += a;
                }
            });
        }
        // In-place band Cholesky (L Lᵀ).
        for j in 0..n {
            let lo = j.saturating_sub(bw);
            let mut d = l[j * w];
            for k in lo..j {
                let v = l[j * w + (j - k)];
                d -= v * v;
            }
            if d.is_nan() || d <= 0.0 {
                return Err(PbError::Backend(format!(
                    "multigrid: coarsest-level matrix is not positive definite (pivot {d:.3e})"
                )));
            }
            let djj = d.sqrt();
            l[j * w] = djj;
            for i in j + 1..(j + bw + 1).min(n) {
                let lo_i = i.saturating_sub(bw).max(lo);
                let mut s = l[i * w + (i - j)];
                for k in lo_i..j {
                    s -= l[i * w + (i - k)] * l[j * w + (j - k)];
                }
                l[i * w + (i - j)] = s / djj;
            }
        }
        Ok(Self { dims, n, bw, l })
    }

    /// Solves A x = b; `b` and `x` are full-grid vectors (boundary of x set to 0).
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        let dims = self.dims;
        let [ix, iy, _] = dims.map(|n| n - 2);
        let w = self.bw + 1;
        let full = |p: usize| (p % ix + 1) + dims[0] * ((p / ix) % iy + 1 + dims[1] * (p / (ix * iy) + 1));
        let mut y: Vec<f64> = (0..self.n).map(|p| b[full(p)]).collect();
        for i in 0..self.n {
            let lo = i.saturating_sub(self.bw);
            let s = (lo..i).fold(y[i], |s, k| s - self.l[i * w + (i - k)] * y[k]);
            y[i] = s / self.l[i * w];
        }
        for i in (0..self.n).rev() {
            let hi = (i + self.bw + 1).min(self.n);
            let s = (i + 1..hi).fold(y[i], |s, k| s - self.l[k * w + (k - i)] * y[k]);
            y[i] = s / self.l[i * w];
        }
        x.fill(0.0);
        for (p, v) in y.into_iter().enumerate() {
            x[full(p)] = v;
        }
    }
}

// ------------------------------------------------------------------ hierarchy (host)

/// Chebyshev smoother settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev {
    /// Polynomial degree (operator applications per smoothing step).
    pub degree: usize,
    /// The smoother targets eigenvalues of D⁻¹A in [λmax / ratio, λmax].
    pub ratio: f64,
}

/// Degree 2 on [λmax/10, λmax]: chosen from a scan of degree 1–4 and
/// ratio 3–15 on an 800-atom globule (65³–161³ grids, `examples/multigrid.rs`).
/// Degree 1 was ~10 % faster on the CPU but needs ~40 % more CG iterations
/// (each costs global reductions, which matter more on GPUs); higher degrees
/// save iterations but not time.
impl Default for Chebyshev {
    fn default() -> Self {
        Self { degree: 2, ratio: 10.0 }
    }
}

/// Host-side multigrid setup: transfers, Galerkin operators, smoother
/// bounds and the coarsest-level factorisation.
#[derive(Clone, Debug)]
pub struct Hierarchy {
    /// `transfers[l]` maps level `l` to level `l + 1` (level 0 = fine grid).
    pub transfers: Vec<Transfer>,
    /// Galerkin operators of levels 1.. (empty if the fine grid is the coarsest).
    pub coarse: Vec<Arc<Stencil27>>,
    /// Gershgorin bounds λmax(D⁻¹A) of every level (fine first).
    pub lmax: Vec<f64>,
    pub coarsest: BandedCholesky,
    pub smoother: Chebyshev,
}

impl Hierarchy {
    /// Builds the hierarchy, or returns `None` if the coarsest level would be
    /// too large for the direct solver (callers then fall back to Jacobi).
    pub fn build(fine: &StencilOperator, smoother: Chebyshev) -> Result<Option<Self>> {
        let mut transfers = Vec::new();
        let mut coarse: Vec<Arc<Stencil27>> = Vec::new();
        let mut lmax = vec![LevelMatrix::Seven(fine).gershgorin_lmax()];
        let mut dims = fine.dims;
        while interior_count(dims) > COARSEST_MAX_UNKNOWNS {
            let Some(t) = Transfer::new(dims) else { break };
            let next = {
                let m = match coarse.last() {
                    Some(s) => LevelMatrix::TwentySeven(s),
                    None => LevelMatrix::Seven(fine),
                };
                galerkin(&t, m)?
            };
            lmax.push(LevelMatrix::TwentySeven(&next).gershgorin_lmax());
            transfers.push(t);
            coarse.push(Arc::new(next));
            dims = t.coarse;
        }
        if BandedCholesky::storage(dims) > MAX_BAND_VALUES {
            return Ok(None);
        }
        let coarsest = match coarse.last() {
            Some(s) => BandedCholesky::factor(LevelMatrix::TwentySeven(s))?,
            None => BandedCholesky::factor(LevelMatrix::Seven(fine))?,
        };
        Ok(Some(Self {
            transfers,
            coarse,
            lmax,
            coarsest,
            smoother,
        }))
    }

    /// Number of levels including the fine grid.
    pub fn levels(&self) -> usize {
        self.coarse.len() + 1
    }
}

/// The coarse levels of a multigrid hierarchy held by a backend: everything
/// the V-cycle needs besides the fine operator.
pub struct Levels<B: Backend> {
    /// `transfers[l]` maps level `l` to level `l + 1` (level 0 = fine grid).
    pub transfers: Vec<Transfer>,
    /// Galerkin operators of levels 1..
    pub coarse: Vec<B::Coarse>,
    /// 1/diag of levels 1.. (0 on boundary nodes).
    pub inv_diag: Vec<B::Vector>,
    /// Gershgorin bounds λmax(D⁻¹A) of every level (fine first).
    pub lmax: Vec<f64>,
    pub coarsest: BandedCholesky,
    pub smoother: Chebyshev,
}

impl<B: Backend> Levels<B> {
    /// Builds the hierarchy of the fine operator `dev` (grid `fine_dims`)
    /// with the backend's own setup kernels (`be` must compute in f64). The
    /// steps and every operator are those of [`Hierarchy::build`] for the
    /// same operator, bit for bit; only the coarsest level is factorised on
    /// the host. Returns `None` where [`Hierarchy::build`] does.
    pub fn build(be: &B, fine_dims: [usize; 3], dev: &B::Operator, smoother: Chebyshev) -> Result<Option<Self>> {
        let mut transfers = Vec::new();
        let mut coarse: Vec<B::Coarse> = Vec::new();
        let mut inv_diag = Vec::new();
        let mut lmax = vec![be.gershgorin_lmax(LevelOp::Fine(dev))?];
        let mut dims = fine_dims;
        while interior_count(dims) > COARSEST_MAX_UNKNOWNS {
            let Some(t) = Transfer::new(dims) else { break };
            let (next, inv) = {
                let m = match coarse.last() {
                    Some(s) => LevelOp::Coarse(s),
                    None => LevelOp::Fine(dev),
                };
                be.galerkin(&t, m)?
            };
            lmax.push(be.gershgorin_lmax(LevelOp::Coarse(&next))?);
            transfers.push(t);
            coarse.push(next);
            inv_diag.push(inv);
            dims = t.coarse;
        }
        if BandedCholesky::storage(dims) > MAX_BAND_VALUES {
            return Ok(None);
        }
        let coarsest = match coarse.last() {
            Some(s) => BandedCholesky::factor(LevelMatrix::TwentySeven(&be.coarse_to_host(s)?))?,
            None => BandedCholesky::factor(LevelMatrix::Seven(&be.operator_to_host(dev)?))?,
        };
        Ok(Some(Self {
            transfers,
            coarse,
            inv_diag,
            lmax,
            coarsest,
            smoother,
        }))
    }

    /// Puts a host hierarchy on the backend.
    pub fn upload(be: &B, h: Hierarchy) -> Result<Self> {
        let mut coarse = Vec::with_capacity(h.coarse.len());
        let mut inv_diag = Vec::with_capacity(h.coarse.len());
        for s in h.coarse {
            inv_diag.push(be.upload(&s.inv_diag)?);
            coarse.push(be.prepare_coarse(s)?);
        }
        Ok(Self {
            transfers: h.transfers,
            coarse,
            inv_diag,
            lmax: h.lmax,
            coarsest: h.coarsest,
            smoother: h.smoother,
        })
    }

    /// The same levels on `single`, the twin of `be`: coefficients and inverse
    /// diagonals rounded to f32 on the device, as uploading the host
    /// hierarchy to the twin would round them.
    pub fn demote(self, be: &B, single: &B::Single) -> Result<Levels<B::Single>> {
        let mut coarse = Vec::with_capacity(self.coarse.len());
        let mut inv_diag = Vec::with_capacity(self.coarse.len());
        for (c, d) in self.coarse.iter().zip(&self.inv_diag) {
            coarse.push(be.demote_coarse(c)?);
            let mut d32 = single.zeros(self.transfers[inv_diag.len()].coarse.iter().product())?;
            be.demote(d, &mut d32)?;
            inv_diag.push(d32);
        }
        Ok(Levels {
            transfers: self.transfers,
            coarse,
            inv_diag,
            lmax: self.lmax,
            coarsest: self.coarsest,
            smoother: self.smoother,
        })
    }

    /// Number of levels including the fine grid.
    pub fn levels(&self) -> usize {
        self.coarse.len() + 1
    }

    /// Dimensions of the fine grid.
    pub fn fine_dims(&self) -> [usize; 3] {
        self.transfers.first().map_or(self.coarsest.dims(), |t| t.fine)
    }
}

// ------------------------------------------------------------------ V-cycle (any backend)

/// The operator of one level as prepared on a backend.
pub enum LevelOp<'a, B: Backend> {
    Fine(&'a B::Operator),
    Coarse(&'a B::Coarse),
}

// Copy without requiring `B: Copy` (a derive would): it only holds a reference.
impl<B: Backend> Clone for LevelOp<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for LevelOp<'_, B> {}

impl<B: Backend> LevelOp<'_, B> {
    /// y = A x (the step-by-step reference of the smoother in the tests).
    #[cfg(test)]
    fn apply(&self, be: &B, x: &B::Vector, y: &mut B::Vector) -> Result<()> {
        match self {
            Self::Fine(op) => be.apply(op, x, y),
            Self::Coarse(op) => be.apply_coarse(op, x, y),
        }
    }
}

struct Work<B: Backend> {
    inv_diag: B::Vector,
    lmax: f64,
    r: B::Vector,
    d: B::Vector,
    t: B::Vector,
}

impl<B: Backend> Work<B> {
    fn new(be: &B, inv_diag: B::Vector, n: usize, lmax: f64) -> Result<Self> {
        Ok(Self {
            inv_diag,
            lmax,
            r: be.zeros(n)?,
            d: be.zeros(n)?,
            t: be.zeros(n)?,
        })
    }
}

struct CoarseLevel<B: Backend> {
    op: B::Coarse,
    work: Work<B>,
    x: B::Vector,
    b: B::Vector,
}

/// Multigrid V-cycle preconditioner on a backend.
pub struct MultigridPreconditioner<B: Backend> {
    fine: Work<B>,
    levels: Vec<CoarseLevel<B>>,
    transfers: Vec<Transfer>,
    coarsest: BandedCholesky,
    smoother: Chebyshev,
    host_b: Vec<f64>,
    host_x: Vec<f64>,
}

impl<B: Backend> MultigridPreconditioner<B> {
    /// The V-cycle of a host hierarchy, uploaded to `be`.
    pub fn new(be: &B, fine: &StencilOperator, h: Hierarchy) -> Result<Self> {
        Self::from_levels(be, be.upload(&fine.inv_diag)?, Levels::upload(be, h)?)
    }

    /// The V-cycle of levels already on `be`; `inv_diag` is the inverse
    /// diagonal of the fine operator (0 on boundary nodes).
    pub fn from_levels(be: &B, inv_diag: B::Vector, l: Levels<B>) -> Result<Self> {
        let n: usize = l.fine_dims().iter().product();
        let mut levels = Vec::with_capacity(l.coarse.len());
        let mut nc = n;
        for (lev, (op, inv_diag)) in l.coarse.into_iter().zip(l.inv_diag).enumerate() {
            nc = l.transfers[lev].coarse.iter().product();
            levels.push(CoarseLevel {
                op,
                work: Work::new(be, inv_diag, nc, l.lmax[lev + 1])?,
                x: be.zeros(nc)?,
                b: be.zeros(nc)?,
            });
        }
        Ok(Self {
            fine: Work::new(be, inv_diag, n, l.lmax[0])?,
            levels,
            transfers: l.transfers,
            coarsest: l.coarsest,
            smoother: l.smoother,
            host_b: vec![0.0; nc],
            host_x: vec![0.0; nc],
        })
    }

    /// Number of levels including the fine grid.
    pub fn levels(&self) -> usize {
        self.levels.len() + 1
    }

    /// z = M⁻¹ r (one V-cycle with zero initial guess).
    pub fn apply(&mut self, be: &B, op: &B::Operator, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
        let mut coarse = Coarsest {
            chol: &self.coarsest,
            host_b: &mut self.host_b,
            host_x: &mut self.host_x,
        };
        vcycle(
            be,
            self.smoother,
            LevelOp::Fine(op),
            &mut self.fine,
            z,
            r,
            &mut self.levels,
            &self.transfers,
            &mut coarse,
        )
    }
}

struct Coarsest<'a> {
    chol: &'a BandedCholesky,
    host_b: &'a mut Vec<f64>,
    host_x: &'a mut Vec<f64>,
}

impl Coarsest<'_> {
    fn solve<B: Backend>(&mut self, be: &B, b: &B::Vector, x: &mut B::Vector) -> Result<()> {
        be.read(b, self.host_b)?;
        self.chol.solve(self.host_b, self.host_x);
        be.write(self.host_x, x)
    }
}

#[allow(clippy::too_many_arguments)]
fn vcycle<B: Backend>(
    be: &B,
    cheb: Chebyshev,
    op: LevelOp<'_, B>,
    work: &mut Work<B>,
    x: &mut B::Vector,
    b: &B::Vector,
    below: &mut [CoarseLevel<B>],
    transfers: &[Transfer],
    coarsest: &mut Coarsest<'_>,
) -> Result<()> {
    let Some((next, rest)) = below.split_first_mut() else {
        return coarsest.solve(be, b, x);
    };
    smooth(be, cheb, op, work, x, b, true)?;
    // r = b - A x, restricted to the next level's right-hand side.
    be.residual(op, x, Some(b), &mut work.t, &mut work.r)?;
    be.restrict(&transfers[0], &work.r, &mut next.b)?;
    let CoarseLevel {
        op: nop,
        work: nwork,
        x: nx,
        b: nb,
    } = next;
    vcycle(
        be,
        cheb,
        LevelOp::Coarse(nop),
        nwork,
        nx,
        nb,
        rest,
        &transfers[1..],
        coarsest,
    )?;
    be.prolong_add(&transfers[0], nx, x)?;
    smooth(be, cheb, op, work, x, b, false)
}

/// Chebyshev smoothing of A x = b with D⁻¹ scaling (Saad, Iterative Methods,
/// Alg. 12.1). With `zero_init` the incoming `x` is ignored.
///
/// Written with the fused smoother steps of [`Backend`]: per degree one
/// residual update and one vector pass. The operations and their order are
/// those of the step-by-step form (the test reference `smooth_by_steps`):
/// with `zero_init` the residual starts as `b` itself, which the step-by-step
/// form copied into `r` first.
fn smooth<B: Backend>(
    be: &B,
    cheb: Chebyshev,
    op: LevelOp<'_, B>,
    w: &mut Work<B>,
    x: &mut B::Vector,
    b: &B::Vector,
    zero_init: bool,
) -> Result<()> {
    let lmax = w.lmax;
    let lmin = lmax / cheb.ratio;
    let theta = 0.5 * (lmax + lmin);
    let delta = 0.5 * (lmax - lmin);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    if cheb.degree == 0 {
        return Ok(()); // no smoothing: x is left as it is, as before
    }
    if zero_init {
        be.chebyshev_first(1.0 / theta, &w.inv_diag, b, &mut w.d, x, true)?;
    } else {
        be.residual(op, x, Some(b), &mut w.t, &mut w.r)?;
        be.chebyshev_first(1.0 / theta, &w.inv_diag, &w.r, &mut w.d, x, false)?;
    }
    for k in 1..cheb.degree {
        // r ← r − A d; with zero_init the residual so far is b.
        let src = (zero_init && k == 1).then_some(b);
        be.residual(op, &w.d, src, &mut w.t, &mut w.r)?;
        let rho_new = 1.0 / (2.0 * sigma - rho);
        be.chebyshev_next(
            2.0 * rho_new / delta,
            &w.inv_diag,
            &w.r,
            rho_new * rho,
            &mut w.t,
            &mut w.d,
            x,
        )?;
        rho = rho_new;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::grid::Grid;
    use crate::molecule::Atom;
    use crate::params::PbParams;
    use crate::surface::CoefficientMaps;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    fn mask_boundary(v: &mut [f64], d: [usize; 3]) {
        for (c, x) in v.iter_mut().enumerate() {
            if !is_interior(c % d[0], (c / d[0]) % d[1], c / (d[0] * d[1]), d) {
                *x = 0.0;
            }
        }
    }

    fn interior_random(d: [usize; 3], seed: u64) -> Vec<f64> {
        let mut v = pseudo_random(d[0] * d[1] * d[2], seed);
        mask_boundary(&mut v, d);
        v
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn norm(a: &[f64]) -> f64 {
        dot(a, a).sqrt()
    }

    /// Sharp molecular dielectric (ε 2 / 78.54), salt with ion exclusion,
    /// several overlapping atoms of different radii: the hard case for multigrid.
    fn molecule_operator(dims: [usize; 3], spacing: f64) -> StencilOperator {
        let len = dims.map(|n| (n - 1) as f64 * spacing);
        let grid = Grid::centered([0.3, -0.2, 0.1], len, dims).unwrap();
        let atoms = [
            Atom {
                position: [0.0, 0.0, 0.0],
                charge: 1.0,
                radius: 1.9,
            },
            Atom {
                position: [1.6, 0.4, -0.3],
                charge: -0.6,
                radius: 1.5,
            },
            Atom {
                position: [-1.2, 1.1, 0.8],
                charge: 0.3,
                radius: 1.7,
            },
            Atom {
                position: [0.4, -1.5, 1.2],
                charge: -0.7,
                radius: 1.2,
            },
            Atom {
                position: [-0.5, -0.4, -1.6],
                charge: 0.0,
                radius: 0.0,
            },
        ];
        let params = PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0);
        StencilOperator::new(&grid, CoefficientMaps::solvated(&grid, &atoms, &params))
    }

    fn prolong(t: &Transfer, c: &[f64]) -> Vec<f64> {
        let mut f = vec![0.0; t.fine.iter().product()];
        prolong_add(t, c, &mut f);
        f
    }

    #[test]
    fn transfer_coarsens_each_axis_independently() {
        let t = Transfer::new([9, 8, 3]).unwrap();
        assert_eq!(t.coarse, [5, 5, 3]);
        assert_eq!(t.coarsen, [true, true, false]);
        let t = Transfer::new([121, 121, 105]).unwrap();
        assert_eq!(t.coarse, [61, 61, 53]);
        assert!(Transfer::new([4, 3, 4]).is_none());
        assert_eq!(offset_of(offset_index(-1, 0, 1)), [-1, 0, 1]);
        assert_eq!(offset_index(0, 0, 0), 13);
        for o in 0..27 {
            assert_eq!(offset_of(26 - o), offset_of(o).map(|v| -v));
        }
    }

    #[test]
    fn restriction_is_the_transpose_of_prolongation() {
        for dims in [[9, 9, 9], [10, 8, 12], [17, 6, 4], [5, 7, 33], [6, 6, 6]] {
            let t = Transfer::new(dims).unwrap();
            let f = interior_random(t.fine, 1);
            let c = interior_random(t.coarse, 2);
            let mut rf = vec![0.0; t.coarse.iter().product()];
            restrict(&t, &f, &mut rf);
            let pc = prolong(&t, &c);
            let (lhs, rhs) = (dot(&rf, &c), dot(&f, &pc));
            assert!(
                (lhs - rhs).abs() < 1e-13 * lhs.abs().max(1.0),
                "{dims:?}: {lhs} vs {rhs}"
            );
            // Prolongation leaves boundary nodes at zero, restriction writes zero there.
            let mut tmp = pc.clone();
            mask_boundary(&mut tmp, t.fine);
            assert_eq!(tmp, pc);
            let mut tmp = rf.clone();
            mask_boundary(&mut tmp, t.coarse);
            assert_eq!(tmp, rf);
        }
    }

    #[test]
    fn prolongation_interpolates_linear_functions() {
        let t = Transfer::new([17, 13, 9]).unwrap();
        let lin = |x: f64, y: f64, z: f64| 0.3 + 1.7 * x - 0.4 * y + 2.2 * z;
        let mut c = vec![0.0; t.coarse.iter().product()];
        let [mx, my, mz] = t.coarse;
        for k in 1..mz - 1 {
            for j in 1..my - 1 {
                for i in 1..mx - 1 {
                    c[i + mx * (j + my * k)] = lin((2 * i) as f64, (2 * j) as f64, (2 * k) as f64);
                }
            }
        }
        let f = prolong(&t, &c);
        let [nx, ny, nz] = t.fine;
        let mut checked = 0;
        // Fine nodes whose interpolation uses interior coarse nodes only.
        for k in 2..nz - 2 {
            for j in 2..ny - 2 {
                for i in 2..nx - 2 {
                    let v = f[i + nx * (j + ny * k)];
                    let e = lin(i as f64, j as f64, k as f64);
                    assert!((v - e).abs() < 1e-12 * e.abs().max(1.0), "({i},{j},{k}): {v} vs {e}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 500);
    }

    #[test]
    fn galerkin_operator_equals_the_triple_product() {
        for dims in [[13, 12, 11], [17, 17, 17], [9, 20, 6]] {
            let op = molecule_operator(dims, 0.6);
            let t1 = Transfer::new(dims).unwrap();
            let a1 = galerkin(&t1, LevelMatrix::Seven(&op)).unwrap();
            let t2 = Transfer::new(t1.coarse).unwrap();
            let a2 = galerkin(&t2, LevelMatrix::TwentySeven(&a1)).unwrap();
            // A_c x == Pᵀ (A (P x)) on both levels, for random interior x.
            for (t, coarse_op, fine_apply) in [
                (
                    t1,
                    &a1,
                    Box::new(|x: &[f64], y: &mut [f64]| op.apply(x, y)) as Box<dyn Fn(&[f64], &mut [f64])>,
                ),
                (t2, &a2, Box::new(|x: &[f64], y: &mut [f64]| a1.apply(x, y))),
            ] {
                let x = interior_random(t.coarse, 7);
                let mut ax = vec![0.0; x.len()];
                coarse_op.apply(&x, &mut ax);
                let px = prolong(&t, &x);
                let mut apx = vec![0.0; px.len()];
                fine_apply(&px, &mut apx);
                let mut rapx = vec![0.0; x.len()];
                restrict(&t, &apx, &mut rapx);
                let err: Vec<f64> = ax.iter().zip(&rapx).map(|(a, b)| a - b).collect();
                assert!(
                    norm(&err) < 1e-12 * norm(&ax),
                    "{dims:?}: |A_c x - RAPx| = {:e}",
                    norm(&err)
                );
            }
        }
    }

    #[test]
    fn galerkin_operator_is_exactly_symmetric_and_positive_definite() {
        let op = molecule_operator([17, 15, 13], 0.5);
        let t = Transfer::new(op.dims).unwrap();
        let a = galerkin(&t, LevelMatrix::Seven(&op)).unwrap();
        let n = a.len();
        let offs = stencil_offsets(a.dims);
        let d = a.dims;
        for c in 0..n {
            if !is_interior(c % d[0], (c / d[0]) % d[1], c / (d[0] * d[1]), d) {
                assert!((0..27).all(|o| a.coef[o * n + c] == 0.0));
                continue;
            }
            assert!(a.coef[13 * n + c] > 0.0);
            for (o, off) in offs.iter().enumerate() {
                let p = (c as isize + off) as usize;
                assert_eq!(a.coef[o * n + c].to_bits(), a.coef[(26 - o) * n + p].to_bits());
            }
        }
        for seed in 0..5 {
            let x = interior_random(d, 100 + seed);
            let mut ax = vec![0.0; n];
            a.apply(&x, &mut ax);
            assert!(dot(&x, &ax) > 0.0);
        }
    }

    #[test]
    fn galerkin_preserves_constants_of_the_laplacian() {
        // Pure diffusion (no ions): rows of P^T A P sum to zero away from the boundary.
        let dims = [17, 17, 17];
        let grid = Grid::centered([0.0; 3], [8.0; 3], dims).unwrap();
        let maps = CoefficientMaps::homogeneous(&grid, 3.0);
        let op = StencilOperator::new(&grid, maps);
        let t = Transfer::new(dims).unwrap();
        let a = galerkin(&t, LevelMatrix::Seven(&op)).unwrap();
        let n = a.len();
        let [mx, my, mz] = a.dims;
        for k in 2..mz - 2 {
            for j in 2..my - 2 {
                for i in 2..mx - 2 {
                    let c = i + mx * (j + my * k);
                    let s: f64 = (0..27).map(|o| a.coef[o * n + c]).sum();
                    assert!(s.abs() < 1e-12 * a.coef[13 * n + c], "row sum {s}");
                }
            }
        }
    }

    #[test]
    fn gershgorin_bound_of_the_fine_operator_is_at_most_two() {
        let op = molecule_operator([17, 15, 13], 0.5);
        let l = LevelMatrix::Seven(&op).gershgorin_lmax();
        // Row sum = diag + Σ|offdiag| = 2 diag − κ̄²V ≤ 2 diag (up to rounding).
        assert!(l > 1.5 && l <= 2.0 + 1e-14, "{l}");
    }

    #[test]
    fn banded_cholesky_solves_both_operator_types() {
        let op = molecule_operator([9, 8, 10], 0.7);
        let t = Transfer::new(op.dims).unwrap();
        let a = galerkin(&t, LevelMatrix::Seven(&op)).unwrap();
        for (m, apply) in [
            (
                LevelMatrix::Seven(&op),
                Box::new(|x: &[f64], y: &mut [f64]| op.apply(x, y)) as Box<dyn Fn(&[f64], &mut [f64])>,
            ),
            (
                LevelMatrix::TwentySeven(&a),
                Box::new(|x: &[f64], y: &mut [f64]| a.apply(x, y)),
            ),
        ] {
            let d = m.dims();
            let chol = BandedCholesky::factor(m).unwrap();
            let b = interior_random(d, 3);
            let mut x = vec![1.0; b.len()];
            chol.solve(&b, &mut x);
            let mut ax = vec![0.0; b.len()];
            apply(&x, &mut ax);
            let err: Vec<f64> = ax.iter().zip(&b).map(|(p, q)| p - q).collect();
            assert!(norm(&err) < 1e-11 * norm(&b), "{d:?}: {:e}", norm(&err));
        }
    }

    #[test]
    fn hierarchy_reaches_a_small_coarsest_level() {
        let op = molecule_operator([33, 30, 17], 0.5);
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        assert!(h.levels() >= 3);
        let last = h.coarse.last().unwrap().dims;
        assert!(interior_count(last) <= COARSEST_MAX_UNKNOWNS, "{last:?}");
        assert_eq!(h.lmax.len(), h.levels());
        for (l, t) in h.transfers.iter().enumerate() {
            assert_eq!(t.coarse, h.coarse[l].dims);
        }
        // A grid with <= COARSEST_MAX_UNKNOWNS unknowns is solved directly.
        let small = molecule_operator([9, 9, 9], 0.8);
        let h = Hierarchy::build(&small, Chebyshev::default()).unwrap().unwrap();
        assert_eq!(h.levels(), 1);
    }

    fn vcycle_once(m: &mut MultigridPreconditioner<CpuBackend>, op: &Arc<StencilOperator>, r: &[f64]) -> Vec<f64> {
        let be = CpuBackend::new();
        let mut z = vec![0.0; r.len()];
        m.apply(&be, op, &r.to_vec(), &mut z).unwrap();
        z
    }

    #[test]
    fn vcycle_is_a_symmetric_positive_definite_operator() {
        let be = CpuBackend::new();
        for dims in [[33, 30, 17], [26, 26, 26]] {
            let op = Arc::new(molecule_operator(dims, 0.45));
            let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
            assert!(h.levels() >= 2);
            let mut m = MultigridPreconditioner::new(&be, &op, h).unwrap();
            for seed in 0..3 {
                let x = interior_random(dims, 10 + seed);
                let y = interior_random(dims, 20 + seed);
                let mx = vcycle_once(&mut m, &op, &x);
                let my = vcycle_once(&mut m, &op, &y);
                let (a, b) = (dot(&y, &mx), dot(&x, &my));
                assert!((a - b).abs() < 1e-11 * a.abs().max(b.abs()), "{dims:?}: {a} vs {b}");
                assert!(dot(&x, &mx) > 0.0);
                // The output is zero on the boundary (CG search directions stay in the interior space).
                let mut masked = mx.clone();
                mask_boundary(&mut masked, dims);
                assert_eq!(masked, mx);
            }
        }
    }

    #[test]
    fn vcycle_reduces_the_error_of_a_stationary_iteration() {
        // x ← x + M(b − A x): the V-cycle alone must contract the error
        // strongly (typical multigrid factor ≪ 1) on the molecular problem.
        let be = CpuBackend::new();
        let dims = [41, 41, 41];
        let op = Arc::new(molecule_operator(dims, 0.35));
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let mut m = MultigridPreconditioner::new(&be, &op, h).unwrap();
        let b = interior_random(dims, 5);
        let mut x = vec![0.0; b.len()];
        let mut ax = vec![0.0; b.len()];
        let mut res = b.clone();
        let r0 = norm(&res);
        for _ in 0..6 {
            let z = vcycle_once(&mut m, &op, &res);
            x.iter_mut().zip(&z).for_each(|(a, b)| *a += b);
            op.apply(&x, &mut ax);
            res = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        }
        let factor = (norm(&res) / r0).powf(1.0 / 6.0);
        eprintln!("V-cycle residual reduction factor: {factor:.4}");
        assert!(factor < 0.35, "average residual reduction per V-cycle {factor:.3}");
    }

    // ---------------------------------------------------------- single precision

    fn rel(a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        (d / dot(b, b)).sqrt()
    }

    #[test]
    fn single_precision_kernels_match_f64_to_rounding() {
        use crate::cpu::CpuSingle;
        let dims = [25, 23, 21];
        let op = Arc::new(molecule_operator(dims, 0.5));
        let x = pseudo_random(op.len(), 3);
        let (be, sp) = (CpuBackend::new(), CpuSingle::new());
        let x32 = sp.upload(&x).unwrap();
        // 7-point operator.
        let mut y = vec![0.0; x.len()];
        op.apply(&x, &mut y);
        let op32 = sp.prepare(op.clone()).unwrap();
        let mut y32 = sp.zeros(x.len()).unwrap();
        sp.apply(&op32, &x32, &mut y32).unwrap();
        assert!(rel(&sp.download(&y32).unwrap(), &y) < 2e-6);
        // Galerkin operator and transfers.
        let t = Transfer::new(dims).unwrap();
        let g = Arc::new(galerkin(&t, LevelMatrix::Seven(&op)).unwrap());
        let mut c = vec![0.0; g.len()];
        be.restrict(&t, &x, &mut c).unwrap();
        let mut c32 = sp.zeros(g.len()).unwrap();
        sp.restrict(&t, &x32, &mut c32).unwrap();
        assert!(rel(&sp.download(&c32).unwrap(), &c) < 2e-7);
        let mut gc = vec![0.0; g.len()];
        g.apply(&c, &mut gc);
        let g32 = sp.prepare_coarse(g.clone()).unwrap();
        let mut gc32 = sp.zeros(g.len()).unwrap();
        sp.apply_coarse(&g32, &sp.upload(&c).unwrap(), &mut gc32).unwrap();
        assert!(rel(&sp.download(&gc32).unwrap(), &gc) < 2e-6);
        // Conversions: demote rounds, promote widens exactly.
        let mut d32 = sp.zeros(x.len()).unwrap();
        be.demote(&x, &mut d32).unwrap();
        assert_eq!(d32, x.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let mut back = vec![0.0; x.len()];
        be.promote(&d32, &mut back).unwrap();
        assert!(back.iter().zip(&d32).all(|(a, b)| *a == *b as f64));
    }

    #[test]
    fn single_precision_vcycle_approximates_the_f64_vcycle() {
        use crate::cpu::CpuSingle;
        let dims = [41, 41, 41];
        let op = Arc::new(molecule_operator(dims, 0.35));
        let r = interior_random(dims, 11);
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let mut m64 = MultigridPreconditioner::new(&CpuBackend::new(), &op, h).unwrap();
        let z64 = vcycle_once(&mut m64, &op, &r);
        let sp = CpuSingle::new();
        let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
        let mut m32 = MultigridPreconditioner::new(&sp, &op, h).unwrap();
        let op32 = sp.prepare(op.clone()).unwrap();
        let mut z32 = sp.zeros(r.len()).unwrap();
        m32.apply(&sp, &op32, &sp.upload(&r).unwrap(), &mut z32).unwrap();
        let err = rel(&sp.download(&z32).unwrap(), &z64);
        eprintln!("f32 V-cycle vs f64 V-cycle: {err:.2e}");
        assert!(err < 1e-5, "{err}");
        // Not exactly symmetric any more, but only at the rounding level.
        let s = interior_random(dims, 12);
        let mut w32 = sp.zeros(r.len()).unwrap();
        m32.apply(&sp, &op32, &sp.upload(&s).unwrap(), &mut w32).unwrap();
        let (a, b) = (
            dot(&s, &sp.download(&z32).unwrap()),
            dot(&r, &sp.download(&w32).unwrap()),
        );
        assert!(((a - b) / a).abs() < 1e-5, "sᵀMr = {a}, rᵀMs = {b}");
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // ------------------------------------------------ fused smoother steps

    /// The smoother as written before the fused steps, every operation on its
    /// own: the reference `smooth` must reproduce bit for bit.
    fn smooth_by_steps<B: Backend>(
        be: &B,
        cheb: Chebyshev,
        op: &LevelOp<'_, B>,
        w: &mut Work<B>,
        x: &mut B::Vector,
        b: &B::Vector,
        zero_init: bool,
    ) -> Result<()> {
        let lmax = w.lmax;
        let lmin = lmax / cheb.ratio;
        let theta = 0.5 * (lmax + lmin);
        let delta = 0.5 * (lmax - lmin);
        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;
        be.copy(b, &mut w.r)?;
        if !zero_init {
            op.apply(be, x, &mut w.t)?;
            be.axpy(-1.0, &w.t, &mut w.r)?;
        }
        be.scale(1.0 / theta, &w.inv_diag, &w.r, &mut w.d)?;
        for k in 0..cheb.degree {
            if zero_init && k == 0 {
                be.copy(&w.d, x)?;
            } else {
                be.axpy(1.0, &w.d, x)?;
            }
            if k + 1 == cheb.degree {
                break;
            }
            op.apply(be, &w.d, &mut w.t)?;
            be.axpy(-1.0, &w.t, &mut w.r)?;
            let rho_new = 1.0 / (2.0 * sigma - rho);
            be.scale(2.0 * rho_new / delta, &w.inv_diag, &w.r, &mut w.t)?;
            be.axpby(1.0, &w.t, rho_new * rho, &mut w.d)?;
            rho = rho_new;
        }
        Ok(())
    }

    /// `vcycle` as written before the fused steps.
    #[allow(clippy::too_many_arguments)]
    fn vcycle_by_steps<B: Backend>(
        be: &B,
        cheb: Chebyshev,
        op: LevelOp<'_, B>,
        work: &mut Work<B>,
        x: &mut B::Vector,
        b: &B::Vector,
        below: &mut [CoarseLevel<B>],
        transfers: &[Transfer],
        coarsest: &mut Coarsest<'_>,
    ) -> Result<()> {
        let Some((next, rest)) = below.split_first_mut() else {
            return coarsest.solve(be, b, x);
        };
        smooth_by_steps(be, cheb, &op, work, x, b, true)?;
        op.apply(be, x, &mut work.t)?;
        be.copy(b, &mut work.r)?;
        be.axpy(-1.0, &work.t, &mut work.r)?;
        be.restrict(&transfers[0], &work.r, &mut next.b)?;
        let CoarseLevel {
            op: nop,
            work: nwork,
            x: nx,
            b: nb,
        } = next;
        vcycle_by_steps(
            be,
            cheb,
            LevelOp::Coarse(nop),
            nwork,
            nx,
            nb,
            rest,
            &transfers[1..],
            coarsest,
        )?;
        be.prolong_add(&transfers[0], nx, x)?;
        smooth_by_steps(be, cheb, &op, work, x, b, false)
    }

    impl<B: Backend> MultigridPreconditioner<B> {
        /// [`Self::apply`] through [`vcycle_by_steps`].
        fn apply_by_steps(&mut self, be: &B, op: &B::Operator, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
            let mut coarse = Coarsest {
                chol: &self.coarsest,
                host_b: &mut self.host_b,
                host_x: &mut self.host_x,
            };
            vcycle_by_steps(
                be,
                self.smoother,
                LevelOp::Fine(op),
                &mut self.fine,
                z,
                r,
                &mut self.levels,
                &self.transfers,
                &mut coarse,
            )
        }
    }

    /// One V-cycle of the fused smoother and of the step-by-step one on `be`.
    fn both_vcycles<B: Backend>(be: &B, dims: [usize; 3], cheb: Chebyshev, seed: u64) -> (Vec<u64>, Vec<u64>) {
        let op = Arc::new(molecule_operator(dims, 0.35));
        let dev = be.prepare(op.clone()).unwrap();
        let r = be.upload(&interior_random(dims, seed)).unwrap();
        let run = |steps: bool| {
            let h = Hierarchy::build(&op, cheb).unwrap().unwrap();
            let mut m = MultigridPreconditioner::new(be, &op, h).unwrap();
            let mut z = be.zeros(dims.iter().product()).unwrap();
            // Twice: the result must not depend on what the work vectors hold.
            for _ in 0..2 {
                if steps {
                    m.apply_by_steps(be, &dev, &r, &mut z).unwrap();
                } else {
                    m.apply(be, &dev, &r, &mut z).unwrap();
                }
            }
            bits(&be.download(&z).unwrap())
        };
        (run(false), run(true))
    }

    /// The fused smoother steps change no bit of a V-cycle, in f64 and f32,
    /// for every smoother degree and with semi-coarsening.
    #[test]
    fn the_fused_smoother_equals_the_step_by_step_smoother() {
        use crate::cpu::CpuSingle;
        for dims in [[41, 41, 41], [66, 41, 6], [50, 37, 29]] {
            for degree in [1, 2, 3] {
                let cheb = Chebyshev { degree, ratio: 10.0 };
                let (fused, steps) = both_vcycles(&CpuBackend::new(), dims, cheb, 21);
                assert!(fused == steps, "f64 {dims:?} degree {degree}");
                let (fused, steps) = both_vcycles(&CpuSingle::new(), dims, cheb, 22);
                assert!(fused == steps, "f32 {dims:?} degree {degree}");
                assert!(fused.iter().any(|&b| b != 0));
            }
        }
    }

    /// Each fused step of the CPU backends equals its step-by-step default,
    /// on the fine 7-point and a coarse 27-point operator.
    #[test]
    fn fused_smoother_steps_equal_their_sequences() {
        use crate::backend::{chebyshev_first_by_steps, chebyshev_next_by_steps, residual_by_steps};
        use crate::cpu::CpuSingle;
        fn check<B: Backend>(be: &B) {
            let dims = [23, 19, 17];
            let op = Arc::new(molecule_operator(dims, 0.6));
            let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
            let fine = be.prepare(op.clone()).unwrap();
            let coarse = be.prepare_coarse(h.coarse[0].clone()).unwrap();
            let nc = h.coarse[0].len();
            let n = op.len();
            let vec = |len: usize, seed: u64| be.upload(&pseudo_random(len, seed)).unwrap();
            let down = |v: &B::Vector| bits(&be.download(v).unwrap());
            for (level, len) in [(LevelOp::<B>::Fine(&fine), n), (LevelOp::Coarse(&coarse), nc)] {
                let (x, b) = (vec(len, 1), vec(len, 2));
                for from_b in [true, false] {
                    let src = from_b.then_some(&b);
                    let (mut t1, mut r1) = (vec(len, 3), vec(len, 4));
                    let (mut t2, mut r2) = (vec(len, 3), vec(len, 4));
                    be.residual(level, &x, src, &mut t1, &mut r1).unwrap();
                    residual_by_steps(be, level, &x, src, &mut t2, &mut r2).unwrap();
                    assert!(down(&r1) == down(&r2), "residual, from b: {from_b}");
                }
                let (w, r) = (vec(len, 5), vec(len, 6));
                for overwrite in [true, false] {
                    let (mut d1, mut x1) = (vec(len, 7), vec(len, 8));
                    let (mut d2, mut x2) = (vec(len, 7), vec(len, 8));
                    be.chebyshev_first(0.37, &w, &r, &mut d1, &mut x1, overwrite).unwrap();
                    chebyshev_first_by_steps(be, 0.37, &w, &r, &mut d2, &mut x2, overwrite).unwrap();
                    assert!(
                        down(&d1) == down(&d2) && down(&x1) == down(&x2),
                        "first, overwrite: {overwrite}"
                    );
                }
                let (mut t1, mut d1, mut x1) = (vec(len, 9), vec(len, 10), vec(len, 11));
                let (mut t2, mut d2, mut x2) = (vec(len, 9), vec(len, 10), vec(len, 11));
                be.chebyshev_next(0.81, &w, &r, -0.29, &mut t1, &mut d1, &mut x1)
                    .unwrap();
                chebyshev_next_by_steps(be, 0.81, &w, &r, -0.29, &mut t2, &mut d2, &mut x2).unwrap();
                assert!(down(&d1) == down(&d2) && down(&x1) == down(&x2), "next");
            }
        }
        check(&CpuBackend::new());
        check(&CpuSingle::new());
    }

    /// Levels built through the backend interface equal the host hierarchy
    /// bit for bit, in f64 and after rounding to the f32 twin.
    #[test]
    fn levels_built_on_the_cpu_backend_equal_the_host_hierarchy() {
        use crate::cpu::CpuSingle;
        // [9, 11, 8] is too small to coarsen: its only level is factorised
        // from the backend's copy of the fine operator.
        for dims in [[41, 41, 41], [66, 41, 6], [50, 37, 29], [9, 11, 8]] {
            let op = Arc::new(molecule_operator(dims, 0.35));
            let be = CpuBackend::new();
            let dev = be.prepare(op.clone()).unwrap();
            let h = Hierarchy::build(&op, Chebyshev::default()).unwrap().unwrap();
            let l = Levels::build(&be, op.dims, &dev, Chebyshev::default())
                .unwrap()
                .unwrap();
            assert_eq!(l.levels(), h.levels(), "{dims:?}");
            assert_eq!(l.transfers, h.transfers);
            assert!(hierarchy_fits(dims));
            assert_eq!(coarsest_dims(dims), h.coarsest.dims());
            let coarse: Vec<[usize; 3]> = h.transfers.iter().map(|t| t.coarse).collect();
            assert_eq!(coarse_dims(dims), coarse);
            assert_eq!(bits(&l.lmax), bits(&h.lmax));
            assert_eq!(l.coarsest, h.coarsest);
            for (lev, s) in h.coarse.iter().enumerate() {
                assert_eq!(bits(&l.coarse[lev].coef), bits(&s.coef), "{dims:?} level {}", lev + 1);
                assert_eq!(bits(&l.inv_diag[lev]), bits(&s.inv_diag));
            }
            // Semi-coarsening: the last axis stops coarsening below 5 nodes.
            if dims == [66, 41, 6] {
                assert!(h.transfers.iter().any(|t| !t.coarsen[2]));
            }
            let sp = CpuSingle::new();
            let up = Levels::upload(&sp, h.clone()).unwrap();
            let down = l.demote(&be, &sp).unwrap();
            for (a, b) in up.coarse.iter().zip(&down.coarse) {
                assert_eq!(a.coef, b.coef);
            }
            assert_eq!(up.inv_diag, down.inv_diag);
            let (o32, d32) = (sp.prepare(op.clone()).unwrap(), be.demote_operator(&dev).unwrap());
            let x = sp.upload(&interior_random(dims, 3)).unwrap();
            let (mut y1, mut y2) = (sp.zeros(x.len()).unwrap(), sp.zeros(x.len()).unwrap());
            sp.apply(&o32, &x, &mut y1).unwrap();
            sp.apply(&d32, &x, &mut y2).unwrap();
            assert_eq!(y1, y2);
        }
    }

    #[test]
    fn a_negative_galerkin_diagonal_is_reported_the_same_way() {
        let dims = [33, 33, 33];
        let mut op = molecule_operator(dims, 0.4);
        // A strongly negative κ²V block makes the coarse diagonal negative.
        for c in [16 + 33 * (16 + 33 * 16), 17 + 33 * (16 + 33 * 16)] {
            op.kvol[c] = -1e6;
        }
        let op = Arc::new(op);
        let host = Hierarchy::build(&op, Chebyshev::default()).unwrap_err().to_string();
        let be = CpuBackend::new();
        let dev = be.prepare(op.clone()).unwrap();
        let levels = Levels::build(&be, op.dims, &dev, Chebyshev::default())
            .err()
            .unwrap()
            .to_string();
        assert!(host.contains("non-positive Galerkin diagonal"), "{host}");
        assert_eq!(host, levels);
    }

    #[test]
    fn single_precision_backends_do_not_build_hierarchies() {
        use crate::cpu::CpuSingle;
        let op = Arc::new(molecule_operator([17, 17, 17], 0.5));
        let sp = CpuSingle::new();
        let dev = sp.prepare(op.clone()).unwrap();
        let t = Transfer::new(op.dims).unwrap();
        let err = sp.galerkin(&t, LevelOp::Fine(&dev)).err().unwrap().to_string();
        assert!(err.contains("f64"), "{err}");
        assert!(sp.gershgorin_lmax(LevelOp::Fine(&dev)).is_err());
    }
}
