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
//! Host-side setup (transfer, Galerkin products, factorisation) lives here
//! and is shared by all backends; the cycle itself is written once against
//! [`Backend`].

use crate::backend::Backend;
use crate::error::{PbError, Result};
use crate::operator::StencilOperator;
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
pub fn restrict(t: &Transfer, fine: &[f64], coarse: &mut [f64]) {
    let [nx, ny, nz] = t.fine;
    let [mx, my, _] = t.coarse;
    coarse.par_chunks_mut(mx * my).enumerate().for_each(|(kc, plane)| {
        for jc in 0..my {
            for ic in 0..mx {
                let out = &mut plane[ic + mx * jc];
                if !is_interior(ic, jc, kc, t.coarse) {
                    *out = 0.0;
                    continue;
                }
                let (fx, wx, mx_) = restrict_axis(ic, t.coarsen[0], nx);
                let (fy, wy, my_) = restrict_axis(jc, t.coarsen[1], ny);
                let (fz, wz, mz_) = restrict_axis(kc, t.coarsen[2], nz);
                let mut acc = 0.0;
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
pub fn prolong_add(t: &Transfer, coarse: &[f64], fine: &mut [f64]) {
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
                let mut acc = 0.0;
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
        let [nx, ny, _] = self.dims;
        let n = self.len();
        let off = stencil_offsets(self.dims);
        y.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
            for j in 0..ny {
                for i in 0..nx {
                    let out = &mut plane[i + nx * j];
                    if !is_interior(i, j, k, self.dims) {
                        *out = 0.0;
                        continue;
                    }
                    let c = i + nx * (j + ny * k);
                    let mut acc = 0.0;
                    for (o, d) in off.iter().enumerate() {
                        acc += self.coef[o * n + c] * x[(c as isize + d) as usize];
                    }
                    *out = acc;
                }
            }
        });
    }
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
    let mut inv_diag = vec![0.0; n];
    for (c, d) in inv_diag.iter_mut().enumerate() {
        let diag = coef[13 * n + c];
        if diag != 0.0 {
            if diag.is_nan() || diag < 0.0 {
                return Err(PbError::Backend(format!(
                    "multigrid: non-positive Galerkin diagonal {diag:.3e} on grid {cd:?}"
                )));
            }
            *d = 1.0 / diag;
        }
    }
    Ok(Stencil27 {
        dims: cd,
        coef,
        inv_diag,
    })
}

// ------------------------------------------------------------------ coarsest level

/// Banded Cholesky factor of a level matrix restricted to interior nodes
/// (natural x-fastest ordering), used as the exact coarsest-level solver.
#[derive(Clone, Debug)]
pub struct BandedCholesky {
    dims: [usize; 3],
    n: usize,
    bw: usize,
    /// Row-major lower band: `l[i * (bw + 1) + (i - j)]` = L[i][j].
    l: Vec<f64>,
}

impl BandedCholesky {
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

// ------------------------------------------------------------------ V-cycle (any backend)

/// The operator of one level as prepared on a backend.
pub enum LevelOp<'a, B: Backend> {
    Fine(&'a B::Operator),
    Coarse(&'a B::Coarse),
}

impl<B: Backend> LevelOp<'_, B> {
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
    fn new(be: &B, inv_diag: &[f64], lmax: f64) -> Result<Self> {
        let n = inv_diag.len();
        Ok(Self {
            inv_diag: be.upload(inv_diag)?,
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
    pub fn new(be: &B, fine: &StencilOperator, h: Hierarchy) -> Result<Self> {
        let mut levels = Vec::with_capacity(h.coarse.len());
        for (l, s) in h.coarse.iter().enumerate() {
            levels.push(CoarseLevel {
                op: be.prepare_coarse(s.clone())?,
                work: Work::new(be, &s.inv_diag, h.lmax[l + 1])?,
                x: be.zeros(s.len())?,
                b: be.zeros(s.len())?,
            });
        }
        let nc = match h.coarse.last() {
            Some(s) => s.len(),
            None => fine.len(),
        };
        Ok(Self {
            fine: Work::new(be, &fine.inv_diag, h.lmax[0])?,
            levels,
            transfers: h.transfers,
            coarsest: h.coarsest,
            smoother: h.smoother,
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
    smooth(be, cheb, &op, work, x, b, true)?;
    // r = b - A x, restricted to the next level's right-hand side.
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
    smooth(be, cheb, &op, work, x, b, false)
}

/// Chebyshev smoothing of A x = b with D⁻¹ scaling (Saad, Iterative Methods,
/// Alg. 12.1). With `zero_init` the incoming `x` is ignored.
fn smooth<B: Backend>(
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
}
