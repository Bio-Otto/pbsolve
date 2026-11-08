// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vpmg.c), Copyright (c) 2010-2022 Battelle
// Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! Dirichlet boundary values. [`analytic_boundary`] and [`focus_boundary`]
//! return a vector of full grid length with values on boundary nodes and
//! zeros in the interior ("g_ext"); their `*_values` forms return the same
//! values on the boundary nodes only, in [`Grid::boundary_indices`] order.
//!
//! Both analytic conditions follow APBS 3.x exactly (`vpmg.c`):
//!
//! * `mdh` (`bcflnew`): Σ_i (l_B0 q_i / ε_w) e^{−κ(d−a_i)} / ((1+κ a_i) d),
//!   with `a_i` the atom radius (no ion radius added);
//! * `sdh` (`bcfl_sdh` + `multipolebc`): charge, dipole and traceless
//!   quadrupole of the solute about its coordinate-box centre, each with the
//!   exact exterior solution for a sphere (radius = max |r_i − c| + a_i,
//!   dielectric ε_p) in a Debye–Hückel medium (ε_w, κ).
//!
//! `mdh-fast` (not in APBS) evaluates the `mdh` sum exactly only at the
//! Chebyshev points of rectangular patches of the box faces and interpolates
//! it to the other boundary nodes (tensor-product barycentric Lagrange
//! interpolation, as in kernel-independent treecodes [Wang, Krasny &
//! Tlupova 2020]). The potential is analytic on a patch away from the atoms;
//! the degree follows from the patch size and its distance to the nearest
//! atom, and patches near atoms are split or summed directly.

use crate::error::{PbError, Result};
use crate::grid::{Grid, NodeSet};
use crate::molecule::{bounding_box, dist2, Atom};
use crate::params::BoundaryCondition;
use rayon::prelude::*;

/// Medium seen by the boundary: homogeneous dielectric with Debye screening.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryMedium {
    /// Dielectric constant of the medium at the boundary (APBS `sdie` of the calculation).
    pub eps: f64,
    /// Dielectric constant inside the solute sphere (APBS `pdie`; used by `sdh`).
    pub eps_solute: f64,
    /// Debye screening constant κ (Å⁻¹).
    pub kappa: f64,
    /// Vacuum Bjerrum length (Å).
    pub lb0: f64,
}

/// Debye–Hückel potential (kT/e) of charge z with exclusion radius a at
/// distance d (APBS `bcfl1sp`). Reference form; the `mdh` sum evaluates it
/// through [`DhSources`].
#[cfg(test)]
fn dh(z: f64, a: f64, d: f64, m: &BoundaryMedium) -> f64 {
    if m.kappa > 0.0 {
        m.lb0 * z / m.eps * (-(m.kappa) * (d - a)).exp() / ((1.0 + m.kappa * a) * d)
    } else {
        m.lb0 * z / (m.eps * d)
    }
}

/// The charged atoms of an `mdh` sum with their atom-only factors hoisted
/// out of the node loop. Each atom–node term performs exactly the operations
/// of `dh`, in the same order (`((l_B0 z / ε) · e^{−κ(d−a)}) / ((1+κa) d)`,
/// or `(l_B0 z) / (ε d)` without salt), so the sums are bit-identical to the
/// direct form, with one division less per term.
struct DhSources {
    position: Vec<[f64; 3]>,
    /// l_B0 z / ε (κ > 0) or l_B0 z (κ = 0).
    scale: Vec<f64>,
    radius: Vec<f64>,
    /// 1 + κ a.
    screen: Vec<f64>,
}

impl DhSources {
    fn new(atoms: &[Atom], m: &BoundaryMedium) -> Self {
        let charged: Vec<&Atom> = atoms.iter().filter(|a| a.charge != 0.0).collect();
        Self {
            position: charged.iter().map(|a| a.position).collect(),
            scale: charged
                .iter()
                .map(|a| {
                    if m.kappa > 0.0 {
                        m.lb0 * a.charge / m.eps
                    } else {
                        m.lb0 * a.charge
                    }
                })
                .collect(),
            radius: charged.iter().map(|a| a.radius).collect(),
            screen: charged.iter().map(|a| 1.0 + m.kappa * a.radius).collect(),
        }
    }

    /// Σ_i dh(q_i, a_i, |p − r_i|) at `NODES` points at once. Every point
    /// keeps its own sum in atom order, starting from the neutral element of
    /// `f64: Sum`, exactly like `iter().map(dh).sum()`. Interleaving the
    /// points only gives the CPU independent dependency chains (and reuses
    /// each atom's data), so the results are bit-identical.
    ///
    /// Where the CPU has AVX2 the same code runs with the points in the lanes
    /// of 256-bit registers: lane-wise IEEE operations (subtraction,
    /// multiplication, division, square root) round exactly like their scalar
    /// forms, Rust never contracts them into FMAs, and `exp` stays the scalar
    /// libm call per point, so both paths are bit-identical
    /// (`vectorised_mdh_terms_equal_the_scalar_ones_bit_for_bit`).
    #[inline]
    fn potentials(&self, p: &[[f64; 3]; NODES], m: &BoundaryMedium) -> [f64; NODES] {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above).
            return unsafe { self.potentials_avx2(p, m) };
        }
        self.potentials_scalar(p, m)
    }

    /// [`Self::potentials`] compiled for AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn potentials_avx2(&self, p: &[[f64; 3]; NODES], m: &BoundaryMedium) -> [f64; NODES] {
        self.potentials_scalar(p, m)
    }

    /// [`Self::potentials`] for the baseline instruction set.
    #[inline(always)]
    fn potentials_scalar(&self, p: &[[f64; 3]; NODES], m: &BoundaryMedium) -> [f64; NODES] {
        let zero: f64 = std::iter::empty::<f64>().sum();
        let mut acc = [zero; NODES];
        let atoms = self.position.iter().zip(&self.scale);
        if m.kappa > 0.0 {
            let k = m.kappa;
            for ((r, s), (a, w)) in atoms.zip(self.radius.iter().zip(&self.screen)) {
                for (acc, p) in acc.iter_mut().zip(p) {
                    let d = dist2(*p, *r).sqrt();
                    *acc += s * (-k * (d - a)).exp() / (w * d);
                }
            }
        } else {
            for (r, s) in atoms {
                for (acc, p) in acc.iter_mut().zip(p) {
                    let d = dist2(*p, *r).sqrt();
                    *acc += s / (m.eps * d);
                }
            }
        }
        acc
    }
}

/// Boundary nodes evaluated together by [`DhSources::potentials`].
const NODES: usize = 4;

/// Radial factors of the monopole, dipole and quadrupole potentials outside a
/// sphere of radius `rad` and dielectric ε_p in a medium (ε_w, κ) — APBS `multipolebc`.
fn multipole_factors(r: f64, m: &BoundaryMedium, rad: f64) -> [f64; 3] {
    let eps_r = m.eps / m.eps_solute;
    let r2 = r * r;
    let r3 = r2 * r;
    let r5 = r3 * r2;
    let mut t = [(1.0 / m.eps) / r, -(1.0 / m.eps) / r3, (1.0 / m.eps) * 3.0 / r5];
    if m.kappa <= 0.0 {
        t[1] *= (3.0 * eps_r) / (1.0 + 2.0 * eps_r);
        t[2] *= (5.0 * eps_r) / (2.0 + 3.0 * eps_r);
    } else {
        let ka = m.kappa * rad;
        let ka2 = ka * ka;
        let ka3 = ka2 * ka;
        let kr = m.kappa * r;
        let kr2 = kr * kr;
        let e = (ka - kr).exp();
        t[0] *= e / (1.0 + ka);
        t[1] *= 3.0 * eps_r * e * (1.0 + kr) / (1.0 + ka + eps_r * (2.0 + 2.0 * ka + ka2));
        t[2] *= 5.0 * eps_r * e * (3.0 + 3.0 * kr + kr2)
            / (6.0 + 6.0 * ka + 2.0 * ka2 + eps_r * (9.0 + 9.0 * ka + 4.0 * ka2 + ka3));
    }
    t
}

/// Charge, dipole and traceless quadrupole (APBS convention, already divided by 3).
struct Multipoles {
    center: [f64; 3],
    radius: f64,
    charge: f64,
    dipole: [f64; 3],
    quad: [[f64; 3]; 3],
}

impl Multipoles {
    fn of(atoms: &[Atom]) -> Self {
        let (lo, hi) = bounding_box(atoms, false);
        let center = [0, 1, 2].map(|a| 0.5 * (hi[a] + lo[a]));
        let radius = atoms
            .iter()
            .map(|a| dist2(a.position, center).sqrt() + a.radius)
            .fold(0.0, f64::max);
        let mut charge = 0.0;
        let mut dipole = [0.0; 3];
        let mut quad = [[0.0; 3]; 3];
        for at in atoms {
            let q = at.charge;
            let r = [0, 1, 2].map(|a| at.position[a] - center[a]);
            charge += q;
            for a in 0..3 {
                dipole[a] += r[a] * q;
            }
            let traced = [0, 1, 2].map(|a| [0, 1, 2].map(|b| r[a] * r[b] * q));
            let qave = (traced[0][0] + traced[1][1] + traced[2][2]) / 3.0;
            for a in 0..3 {
                for b in 0..3 {
                    let diag = if a == b { qave } else { 0.0 };
                    quad[a][b] += 1.5 * (traced[a][b] - diag);
                }
            }
        }
        for row in quad.iter_mut() {
            for v in row.iter_mut() {
                *v /= 3.0;
            }
        }
        Self {
            center,
            radius,
            charge,
            dipole,
            quad,
        }
    }

    fn potential(&self, p: [f64; 3], m: &BoundaryMedium) -> f64 {
        let r = [0, 1, 2].map(|a| p[a] - self.center[a]);
        let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let t = multipole_factors(d, m, self.radius);
        let pre = m.lb0;
        let q = &self.quad;
        let mut val = pre * self.charge * t[0];
        for (mu, x) in self.dipole.iter().zip(&r) {
            val -= pre * mu * x * t[1];
        }
        for (a, x) in r.iter().enumerate() {
            val += pre * q[a][a] * x * x * t[2];
        }
        val += pre * 2.0 * q[0][1] * r[0] * r[1] * t[2];
        val += pre * 2.0 * q[0][2] * r[0] * r[2] * t[2];
        val += pre * 2.0 * q[1][2] * r[1] * r[2] * t[2];
        val
    }
}

/// Exact `mdh` sums at the given points, in parallel.
fn mdh_at(src: &DhSources, m: &BoundaryMedium, points: &[[f64; 3]]) -> Vec<f64> {
    let mut values = vec![0.0; points.len()];
    values
        .par_chunks_mut(NODES)
        .zip(points.par_chunks(NODES))
        .for_each(|(out, pts)| out.copy_from_slice(&mdh_chunk(src, m, pts)[..out.len()]));
    values
}

/// Exact `mdh` sums at up to `NODES` points (a short chunk repeats its
/// first point in the unused slots).
#[inline]
fn mdh_chunk(src: &DhSources, m: &BoundaryMedium, pts: &[[f64; 3]]) -> [f64; NODES] {
    let p: [[f64; 3]; NODES] = std::array::from_fn(|l| pts[l.min(pts.len() - 1)]);
    src.potentials(&p, m)
}

pub fn analytic_boundary(grid: &Grid, atoms: &[Atom], bc: BoundaryCondition, m: &BoundaryMedium) -> Vec<f64> {
    let mut g = vec![0.0; grid.len()];
    for (c, v) in grid
        .boundary_indices()
        .into_iter()
        .zip(analytic_boundary_values(grid, atoms, bc, m))
    {
        g[c] = v;
    }
    g
}

/// [`analytic_boundary`] on the boundary nodes only, in
/// [`Grid::boundary_indices`] order.
pub fn analytic_boundary_values(grid: &Grid, atoms: &[Atom], bc: BoundaryCondition, m: &BoundaryMedium) -> Vec<f64> {
    let bidx = grid.boundary_indices();
    match bc {
        BoundaryCondition::Zero => vec![0.0; bidx.len()],
        BoundaryCondition::Mdh => {
            let points: Vec<[f64; 3]> = bidx
                .iter()
                .map(|&c| {
                    let (i, j, k) = grid.ijk(c);
                    grid.position(i, j, k)
                })
                .collect();
            mdh_at(&DhSources::new(atoms, m), m, &points)
        }
        BoundaryCondition::MdhFast => {
            let mut values = vec![0.0; bidx.len()];
            mdh_fast(grid, &DhSources::new(atoms, m), m, MDH_FAST_TOLERANCE, |c, v| {
                let (i, j, k) = grid.ijk(c);
                values[grid.boundary_position(i, j, k).expect("patches hold boundary nodes")] = v;
            });
            values
        }
        BoundaryCondition::Sdh => {
            let mp = Multipoles::of(atoms);
            bidx.par_iter()
                .map(|&c| {
                    let (i, j, k) = grid.ijk(c);
                    mp.potential(grid.position(i, j, k), m)
                })
                .collect()
        }
    }
}

// ------------------------------------------------------------------ mdh-fast

/// Accuracy of `mdh-fast`: on every interpolated patch the error is at most
/// this fraction of the largest Σ_i |dh_i| on the patch (the summed magnitudes
/// of the atom terms, which bound the potential before cancellation between
/// charges). Measured worst case: 1.8e-12, for κ up to 0.6 Å⁻¹ (tests).
pub const MDH_FAST_TOLERANCE: f64 = 1e-10;
/// Largest interpolation degree per patch axis; patches needing more are split.
const MAX_DEGREE: usize = 32;
/// Patches with at most this many nodes are summed directly.
const MIN_PATCH: usize = 64;
/// Weight of the screening term of the degree estimate. Calibrated on the
/// test geometries for κ = 0.125–0.6 Å⁻¹: 0.5 keeps the error ≤ 1.8e-12
/// (0.25 reached 2.9e-10 at κ = 0.6; the full bound, 1.0, was ~100× tighter
/// than needed and summed up to 2× more points directly).
const SCREENING: f64 = 0.5;
/// Safety factor of the a-priori degree estimate (see [`degree`]). Measured
/// on clusters centred and next to a face, with and without salt, the
/// degree ln(1/tol)/ln ρ already reached the tolerance (the smallest
/// sufficient degree was 0–2 lower); the factor 100 adds ~20 % to it.
const DEGREE_SAFETY: f64 = 1e2;

/// A rectangle of boundary nodes on one face: normal axis `axis` at node
/// index `fixed`, node indices `lo..=hi` along the in-plane axes `[ua, va]`.
#[derive(Clone, Copy, Debug)]
struct Patch {
    axis: usize,
    fixed: usize,
    ua: usize,
    va: usize,
    lo: [usize; 2],
    hi: [usize; 2],
}

/// How a patch is evaluated: directly, or by interpolation of the given degrees.
#[derive(Clone, Copy, Debug)]
struct Leaf {
    patch: Patch,
    degree: Option<[usize; 2]>,
}

impl Patch {
    /// The six faces, partitioning the boundary nodes: x faces whole, y faces
    /// without the x-boundary nodes, z faces without the x- and y-boundary nodes.
    fn faces(d: [usize; 3]) -> [Patch; 6] {
        let [nx, ny, nz] = d;
        let f = |axis, fixed, ua, va, lo, hi| Patch {
            axis,
            fixed,
            ua,
            va,
            lo,
            hi,
        };
        [
            f(0, 0, 1, 2, [0, 0], [ny - 1, nz - 1]),
            f(0, nx - 1, 1, 2, [0, 0], [ny - 1, nz - 1]),
            f(1, 0, 0, 2, [1, 0], [nx - 2, nz - 1]),
            f(1, ny - 1, 0, 2, [1, 0], [nx - 2, nz - 1]),
            f(2, 0, 0, 1, [1, 1], [nx - 2, ny - 2]),
            f(2, nz - 1, 0, 1, [1, 1], [nx - 2, ny - 2]),
        ]
    }

    fn nodes(&self) -> usize {
        (self.hi[0] - self.lo[0] + 1) * (self.hi[1] - self.lo[1] + 1)
    }

    /// Grid indices of in-plane node (u, v).
    fn ijk(&self, u: usize, v: usize) -> (usize, usize, usize) {
        let mut n = [0; 3];
        n[self.axis] = self.fixed;
        n[self.ua] = u;
        n[self.va] = v;
        (n[0], n[1], n[2])
    }

    /// Centre and half-length (Å) of the patch along in-plane axis `a` (0 or 1).
    fn extent(&self, grid: &Grid, a: usize) -> (f64, f64) {
        let ax = [self.ua, self.va][a];
        let x = |i: usize| grid.origin[ax] + i as f64 * grid.spacing[ax];
        let (x0, x1) = (x(self.lo[a]), x(self.hi[a]));
        (0.5 * (x0 + x1), 0.5 * (x1 - x0))
    }

    /// Smallest distance (Å) from a source to the patch rectangle.
    fn distance(&self, grid: &Grid, src: &DhSources) -> f64 {
        let plane = grid.origin[self.axis] + self.fixed as f64 * grid.spacing[self.axis];
        let (cu, lu) = self.extent(grid, 0);
        let (cv, lv) = self.extent(grid, 1);
        src.position
            .iter()
            .map(|r| {
                let dn = r[self.axis] - plane;
                let du = ((r[self.ua] - cu).abs() - lu).max(0.0);
                let dv = ((r[self.va] - cv).abs() - lv).max(0.0);
                dn * dn + du * du + dv * dv
            })
            .fold(f64::INFINITY, f64::min)
            .sqrt()
    }

    /// The two halves along the longer in-plane side, if it has ≥ 2 nodes.
    fn split(&self, grid: &Grid) -> Option<(Patch, Patch)> {
        let a = if self.extent(grid, 0).1 >= self.extent(grid, 1).1 {
            0
        } else {
            1
        };
        let a = if self.hi[a] > self.lo[a] { a } else { 1 - a };
        if self.hi[a] == self.lo[a] {
            return None;
        }
        let mid = (self.lo[a] + self.hi[a]) / 2;
        let (mut p, mut q) = (*self, *self);
        p.hi[a] = mid;
        q.lo[a] = mid + 1;
        Some((p, q))
    }
}

/// Chebyshev degree that interpolates the potential of sources at distance
/// ≥ `dist` along a segment of half-length `half` to accuracy `tol` relative
/// to the largest term sum on the segment, in a medium with screening `kappa`.
///
/// A source at distance d from the segment is a singularity outside the
/// Bernstein ellipse E_ρ with ρ = d/L + sqrt((d/L)² + 1), and the error of
/// degree n decays like M(ρ') ρ'^(−n) for any ρ' < ρ, M being the largest
/// value on E_ρ' [Trefethen, Approximation Theory and Approximation
/// Practice, Thm 8.2]. Screening makes |e^(−κr)| grow by up to e^(κb') off
/// the real axis (b' = L(ρ' − 1/ρ')/2, the semi-minor axis), so the degree is
/// minimised over ρ' = ρ^t of (ln(S/tol) + κb') / ln ρ'. `DEGREE_SAFETY` (S)
/// covers the remaining constants and the 2-D Lebesgue factor.
fn degree(half: f64, dist: f64, kappa: f64, tol: f64) -> Option<usize> {
    if half <= 0.0 {
        return Some(0);
    }
    let r = dist / half;
    let rho = r + (r * r + 1.0).sqrt();
    if rho.is_nan() || rho <= 1.0 + 1e-9 {
        return None;
    }
    let base = (DEGREE_SAFETY / tol).ln();
    let n = (1..=20)
        .map(|i| {
            let rp = rho.powf(i as f64 / 20.0);
            (base + SCREENING * kappa * half * 0.5 * (rp - 1.0 / rp)) / rp.ln()
        })
        .fold(f64::INFINITY, f64::min)
        .ceil();
    (n <= MAX_DEGREE as f64).then_some(n.max(1.0) as usize)
}

/// Splits a face into leaves (interpolated or direct).
fn plan(p: Patch, grid: &Grid, src: &DhSources, kappa: f64, tol: f64) -> Vec<Leaf> {
    let direct = vec![Leaf { patch: p, degree: None }];
    if p.nodes() <= MIN_PATCH || src.position.is_empty() {
        return direct;
    }
    let dist = p.distance(grid, src);
    let deg = (
        degree(p.extent(grid, 0).1, dist, kappa, tol),
        degree(p.extent(grid, 1).1, dist, kappa, tol),
    );
    if let (Some(nu), Some(nv)) = deg {
        // Interpolate only when it needs ≥ 2× fewer exact sums than the
        // nodes; smaller patches would only need relatively more.
        return if 2 * (nu + 1) * (nv + 1) <= p.nodes() {
            vec![Leaf {
                patch: p,
                degree: Some([nu, nv]),
            }]
        } else {
            direct
        };
    }
    match p.split(grid) {
        Some((a, b)) => {
            let (mut x, y) = rayon::join(|| plan(a, grid, src, kappa, tol), || plan(b, grid, src, kappa, tol));
            x.extend(y);
            x
        }
        None => direct,
    }
}

/// Chebyshev points of the second kind on [c − l, c + l] and their
/// barycentric weights.
fn chebyshev(c: f64, l: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![c], vec![1.0]);
    }
    let x = (0..=n)
        .map(|k| c + l * (std::f64::consts::PI * k as f64 / n as f64).cos())
        .collect();
    let w = (0..=n)
        .map(|k| {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            if k == 0 || k == n {
                0.5 * s
            } else {
                s
            }
        })
        .collect();
    (x, w)
}

/// Normalised barycentric basis values at `x` (sum 1).
fn basis(x: f64, nodes: &[f64], w: &[f64]) -> Vec<f64> {
    if let Some(k) = nodes.iter().position(|&xk| xk == x) {
        let mut e = vec![0.0; nodes.len()];
        e[k] = 1.0;
        return e;
    }
    let mut a: Vec<f64> = nodes.iter().zip(w).map(|(xk, wk)| wk / (x - xk)).collect();
    let s: f64 = a.iter().sum();
    a.iter_mut().for_each(|v| *v /= s);
    a
}

/// Evaluates one leaf, returning (grid index, value) pairs.
fn evaluate(leaf: &Leaf, grid: &Grid, src: &DhSources, m: &BoundaryMedium) -> Vec<(usize, f64)> {
    let p = leaf.patch;
    let us = p.lo[0]..=p.hi[0];
    let vs = p.lo[1]..=p.hi[1];
    let index = |u, v| {
        let (i, j, k) = p.ijk(u, v);
        grid.index(i, j, k)
    };
    let Some([nu, nv]) = leaf.degree else {
        let ids: Vec<(usize, usize, usize)> = vs.flat_map(|v| us.clone().map(move |u| p.ijk(u, v))).collect();
        let pts: Vec<[f64; 3]> = ids.iter().map(|&(i, j, k)| grid.position(i, j, k)).collect();
        let mut out = Vec::with_capacity(ids.len());
        for (c, chunk) in ids.chunks(NODES).zip(pts.chunks(NODES)) {
            let v = mdh_chunk(src, m, chunk);
            out.extend(c.iter().zip(v).map(|(&(i, j, k), v)| (grid.index(i, j, k), v)));
        }
        return out;
    };
    let plane = grid.origin[p.axis] + p.fixed as f64 * grid.spacing[p.axis];
    let (cu, lu) = p.extent(grid, 0);
    let (cv, lv) = p.extent(grid, 1);
    let (xu, wu) = chebyshev(cu, lu, nu);
    let (xv, wv) = chebyshev(cv, lv, nv);
    // Exact sums at the (nu+1)(nv+1) points, u fastest.
    let pts: Vec<[f64; 3]> = xv
        .iter()
        .flat_map(|&y| {
            xu.iter().map(move |&x| {
                let mut r = [0.0; 3];
                r[p.axis] = plane;
                r[p.ua] = x;
                r[p.va] = y;
                r
            })
        })
        .collect();
    let f: Vec<f64> = pts
        .chunks(NODES)
        .flat_map(|c| mdh_chunk(src, m, c)[..c.len()].to_vec())
        .collect();
    let coord = |a: usize, i: usize| grid.origin[a] + i as f64 * grid.spacing[a];
    let bu: Vec<Vec<f64>> = us.clone().map(|u| basis(coord(p.ua, u), &xu, &wu)).collect();
    let mut out = Vec::with_capacity(p.nodes());
    for v in vs {
        let bv = basis(coord(p.va, v), &xv, &wv);
        // g_k = Σ_l bv_l f[l][k], then value = Σ_k bu_k g_k.
        let g: Vec<f64> = (0..=nu)
            .map(|k| bv.iter().enumerate().map(|(l, b)| b * f[l * (nu + 1) + k]).sum())
            .collect();
        for (u, b) in us.clone().zip(&bu) {
            out.push((index(u, v), b.iter().zip(&g).map(|(x, y)| x * y).sum()));
        }
    }
    out
}

/// `mdh` on the boundary of `grid` by patch-wise interpolation (see the
/// module documentation); calls `write(node, value)` once per boundary node.
fn mdh_fast(grid: &Grid, src: &DhSources, m: &BoundaryMedium, tol: f64, mut write: impl FnMut(usize, f64)) {
    let leaves: Vec<Leaf> = Patch::faces(grid.dims)
        .into_par_iter()
        .flat_map_iter(|f| plan(f, grid, src, m.kappa, tol))
        .collect();
    let values: Vec<Vec<(usize, f64)>> = leaves.par_iter().map(|l| evaluate(l, grid, src, m)).collect();
    for (c, v) in values.into_iter().flatten() {
        write(c, v);
    }
}

/// Boundary of a fine grid taken from a coarse solution (focusing).
pub fn focus_boundary(fine: &Grid, coarse: &Grid, u_coarse: &[f64]) -> Result<Vec<f64>> {
    let mut g = vec![0.0; fine.len()];
    let values = focus_boundary_values(fine, coarse, |c| u_coarse[c])?;
    for (c, v) in fine.boundary_indices().into_iter().zip(values) {
        g[c] = v;
    }
    Ok(g)
}

/// [`focus_boundary`] on the boundary nodes of `fine` only, in
/// [`Grid::boundary_indices`] order, with the coarse solution read node by
/// node through `u_coarse`. Only the nodes of [`focus_nodes`] are read.
pub fn focus_boundary_values(fine: &Grid, coarse: &Grid, u_coarse: impl Fn(usize) -> f64 + Sync) -> Result<Vec<f64>> {
    fine.boundary_indices()
        .into_par_iter()
        .map(|c| {
            let (i, j, k) = fine.ijk(c);
            coarse
                .interpolate_with(&u_coarse, fine.position(i, j, k))
                .ok_or_else(|| PbError::InvalidInput("fine grid is not contained in the coarse grid".into()))
        })
        .collect()
}

/// The nodes of `coarse` that focusing the boundary of `fine` reads: the
/// trilinear stencils of the fine boundary nodes. (Boundary nodes outside
/// `coarse` add nothing; [`focus_boundary_values`] reports them.)
pub fn focus_nodes(fine: &Grid, coarse: &Grid) -> NodeSet {
    let stencils = fine.boundary_indices().into_par_iter().flat_map_iter(|c| {
        let (i, j, k) = fine.ijk(c);
        coarse
            .trilinear(fine.position(i, j, k))
            .into_iter()
            .flat_map(|(idx, _)| idx)
    });
    NodeSet::from_indices(coarse.len(), stencils)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medium(eps: f64, eps_solute: f64, kappa: f64) -> BoundaryMedium {
        BoundaryMedium {
            eps,
            eps_solute,
            kappa,
            lb0: 560.0,
        }
    }

    fn atom(p: [f64; 3], q: f64, r: f64) -> Atom {
        Atom {
            position: p,
            charge: q,
            radius: r,
        }
    }

    #[test]
    fn sdh_of_a_centred_ion_is_debye_huckel() {
        let m = medium(78.54, 2.0, 0.12);
        let mp = Multipoles::of(&[atom([1.0, 2.0, 3.0], -1.5, 2.2)]);
        let p = [9.0, -4.0, 10.0];
        let d = dist2(p, [1.0, 2.0, 3.0]).sqrt();
        let want = dh(-1.5, 2.2, d, &m);
        assert!((mp.potential(p, &m) - want).abs() < 1e-12 * want.abs());
    }

    /// The hoisted `mdh` sum equals the direct sum of `dh` terms bit for bit.
    #[test]
    fn mdh_equals_the_direct_sum_bit_for_bit() {
        let mut s: u64 = 7;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut atoms: Vec<Atom> = (0..300)
            .map(|_| atom([0, 1, 2].map(|_| 12.0 * (rnd() - 0.5)), rnd() - 0.5, 0.8 + 1.5 * rnd()))
            .collect();
        atoms[5].charge = 0.0; // skipped by both forms
        atoms[9].radius = 0.0;
        let grid = Grid::centered([0.3, -0.2, 0.1], [40.0, 36.0, 44.0], [17, 15, 19]).unwrap();
        for m in [
            medium(78.54, 2.0, 0.125),
            medium(2.0, 2.0, 0.0),
            medium(80.0, 1.0, 1e-300),
        ] {
            let g = analytic_boundary(&grid, &atoms, BoundaryCondition::Mdh, &m);
            for c in grid.boundary_indices() {
                let (i, j, k) = grid.ijk(c);
                let p = grid.position(i, j, k);
                let want: f64 = atoms
                    .iter()
                    .filter(|a| a.charge != 0.0)
                    .map(|a| dh(a.charge, a.radius, dist2(p, a.position).sqrt(), &m))
                    .sum();
                assert_eq!(g[c].to_bits(), want.to_bits(), "node {c}, kappa {}", m.kappa);
            }
            assert!(g.iter().all(|v| v.is_finite()));
        }
    }

    /// The AVX2 build of the `mdh` terms gives the scalar sums bit for bit,
    /// with and without salt, at points near and far from the atoms.
    #[test]
    fn vectorised_mdh_terms_equal_the_scalar_ones_bit_for_bit() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("skipped: no AVX2");
                return;
            }
            let mut atoms = cluster(500, [0.5, -1.0, 2.0], 30.0, 23);
            atoms[3].radius = 0.0;
            atoms[4].charge = 0.0;
            let points = cluster(4000, [0.0; 3], 400.0, 29);
            for m in [
                medium(78.54, 2.0, 0.125),
                medium(78.54, 2.0, 0.6),
                medium(2.0, 2.0, 0.0),
                medium(80.0, 1.0, 1e-300),
            ] {
                let src = DhSources::new(&atoms, &m);
                for chunk in points.chunks_exact(NODES) {
                    let p: [[f64; 3]; NODES] = std::array::from_fn(|l| chunk[l].position);
                    let scalar = src.potentials_scalar(&p, &m);
                    // SAFETY: AVX2 is available (checked above).
                    let vector = unsafe { src.potentials_avx2(&p, &m) };
                    assert_eq!(scalar.map(f64::to_bits), vector.map(f64::to_bits), "kappa {}", m.kappa);
                    assert_eq!(src.potentials(&p, &m).map(f64::to_bits), vector.map(f64::to_bits));
                }
            }
        }
    }

    fn cluster(n: usize, center: [f64; 3], size: f64, seed: u64) -> Vec<Atom> {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let p = [0, 1, 2].map(|a| center[a] + size * (rnd() - 0.5));
                atom(p, rnd() - 0.5, 1.0 + rnd())
            })
            .collect()
    }

    /// For every interpolated patch: max |fast − exact| / (largest Σ_i |dh_i|
    /// on the patch); and the fraction of boundary nodes summed exactly.
    fn fast_error(grid: &Grid, atoms: &[Atom], m: &BoundaryMedium) -> (f64, f64) {
        let src = DhSources::new(atoms, m);
        let leaves: Vec<Leaf> = Patch::faces(grid.dims)
            .into_iter()
            .flat_map(|f| plan(f, grid, &src, m.kappa, MDH_FAST_TOLERANCE))
            .collect();
        let (mut worst, mut work) = (0.0f64, 0usize);
        for leaf in &leaves {
            let Some([nu, nv]) = leaf.degree else {
                work += leaf.patch.nodes();
                continue;
            };
            work += (nu + 1) * (nv + 1);
            let exact = evaluate(&Leaf { degree: None, ..*leaf }, grid, &src, m);
            let scale = exact
                .iter()
                .map(|&(c, _)| {
                    let (i, j, k) = grid.ijk(c);
                    let q = grid.position(i, j, k);
                    atoms
                        .iter()
                        .filter(|a| a.charge != 0.0)
                        .map(|a| dh(a.charge, a.radius, dist2(q, a.position).sqrt(), m).abs())
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            for ((c, f), (d, e)) in evaluate(leaf, grid, &src, m).into_iter().zip(exact) {
                assert_eq!(c, d);
                worst = worst.max((f - e).abs() / scale);
            }
        }
        // The assembled boundary is exactly the union of the leaves.
        let g = analytic_boundary(grid, atoms, BoundaryCondition::MdhFast, m);
        for leaf in &leaves {
            for (c, v) in evaluate(leaf, grid, &src, m) {
                assert_eq!(g[c].to_bits(), v.to_bits());
            }
        }
        (worst, work as f64 / grid.boundary_indices().len() as f64)
    }

    /// Molecule-sized clusters in a small and a large box and a lone ion near
    /// a corner, from no salt to κ = 0.6 Å⁻¹ (~3 M): the error stays below the
    /// tolerance (measured worst case 1.8e-12), and interpolation replaces
    /// most exact sums on the large box at physiological salt.
    #[test]
    fn mdh_fast_is_within_its_tolerance_of_mdh() {
        let small = Grid::centered([0.0; 3], [80.0, 70.0, 75.0], [97, 81, 97]).unwrap();
        let large = Grid::centered([0.0; 3], [170.0, 160.0, 165.0], [257, 241, 257]).unwrap();
        let cases = [
            ("small box", &small, cluster(400, [0.0; 3], 30.0, 3)),
            ("near a face", &small, cluster(300, [27.0, 3.0, -2.0], 20.0, 5)),
            ("corner ion", &small, vec![atom([36.0, 31.0, -33.0], 1.0, 2.0)]),
            ("large box", &large, cluster(800, [0.0; 3], 90.0, 9)),
        ];
        for (name, grid, atoms) in &cases {
            for kappa in [0.0, 0.125, 0.33, 0.6] {
                let m = medium(78.54, 2.0, kappa);
                let (err, work) = fast_error(grid, atoms, &m);
                eprintln!(
                    "{name}, kappa {kappa}: max error {err:.2e} of the patch scale, exact sums {:.1} %",
                    100.0 * work
                );
                assert!(err <= MDH_FAST_TOLERANCE, "{name}, kappa {kappa}: {err:e}");
                if *name == "large box" && kappa == 0.125 {
                    assert!(work < 0.2, "{name}: interpolation saved too little ({work})");
                }
            }
        }
    }

    #[test]
    fn mdh_fast_does_not_depend_on_the_thread_count() {
        let grid = Grid::centered([0.0; 3], [60.0, 50.0, 55.0], [65, 57, 61]).unwrap();
        let atoms = cluster(200, [1.0, -2.0, 0.5], 20.0, 11);
        let m = medium(78.54, 2.0, 0.1);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| analytic_boundary(&grid, &atoms, BoundaryCondition::MdhFast, &m))
        };
        let (a, b) = (run(1), run(5));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn mdh_fast_faces_partition_the_boundary() {
        for dims in [[5, 5, 5], [9, 6, 7], [33, 17, 21]] {
            let grid = Grid::centered([0.0; 3], [10.0, 9.0, 8.0], dims).unwrap();
            let mut seen = vec![0u8; grid.len()];
            for f in Patch::faces(dims) {
                for v in f.lo[1]..=f.hi[1] {
                    for u in f.lo[0]..=f.hi[0] {
                        let (i, j, k) = f.ijk(u, v);
                        seen[grid.index(i, j, k)] += 1;
                    }
                }
            }
            for (c, &n) in seen.iter().enumerate() {
                let (i, j, k) = grid.ijk(c);
                assert_eq!(n, u8::from(grid.is_boundary(i, j, k)), "{dims:?} node {c}");
            }
        }
    }

    /// The boundary-only forms return the boundary nodes of the full forms,
    /// bit for bit, in `boundary_indices` order.
    #[test]
    fn boundary_values_are_the_boundary_nodes_of_the_full_vectors() {
        let grid = Grid::centered([0.4, -0.3, 0.2], [44.0, 40.0, 48.0], [41, 33, 45]).unwrap();
        let atoms = cluster(120, [1.0, -2.0, 0.5], 14.0, 17);
        let bidx = grid.boundary_indices();
        let m = medium(78.54, 2.0, 0.125);
        for bc in [
            BoundaryCondition::Zero,
            BoundaryCondition::Mdh,
            BoundaryCondition::MdhFast,
            BoundaryCondition::Sdh,
        ] {
            let full = analytic_boundary(&grid, &atoms, bc, &m);
            let values = analytic_boundary_values(&grid, &atoms, bc, &m);
            assert_eq!(values.len(), bidx.len());
            for (&c, v) in bidx.iter().zip(&values) {
                assert_eq!(full[c].to_bits(), v.to_bits(), "{bc:?} node {c}");
            }
            let interior = (0..grid.len()).filter(|c| bidx.binary_search(c).is_err());
            assert!(interior.into_iter().all(|c| full[c].to_bits() == 0), "{bc:?}");
            assert!(bc == BoundaryCondition::Zero || values.iter().any(|&v| v != 0.0));
        }
        // Focusing a finer, shifted grid from a coarse field.
        let fine = Grid::centered([1.1, -0.7, 0.9], [20.0, 17.0, 22.0], [33, 29, 37]).unwrap();
        let u: Vec<f64> = (0..grid.len())
            .map(|c| ((c * 7919) % 1013) as f64 / 7.0 - 70.0)
            .collect();
        let full = focus_boundary(&fine, &grid, &u).unwrap();
        let values = focus_boundary_values(&fine, &grid, |c| u[c]).unwrap();
        for (&c, v) in fine.boundary_indices().iter().zip(&values) {
            assert_eq!(full[c].to_bits(), v.to_bits(), "focus node {c}");
        }
    }

    /// `focus_nodes` is exactly the set of coarse nodes focusing reads.
    #[test]
    fn focus_nodes_are_the_nodes_focusing_reads() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let coarse = Grid::centered([0.0; 3], [40.0, 36.0, 44.0], [41, 37, 45]).unwrap();
        for (center, lengths, dims) in [
            ([1.1, -0.7, 0.9], [20.0, 17.0, 22.0], [33, 29, 37]),
            ([0.0; 3], [40.0, 36.0, 44.0], [41, 37, 45]),
            ([-3.2, 2.1, 0.4], [9.0, 11.0, 7.0], [9, 13, 7]),
        ] {
            let fine = Grid::centered(center, lengths, dims).unwrap();
            let read: Vec<AtomicBool> = (0..coarse.len()).map(|_| AtomicBool::new(false)).collect();
            focus_boundary_values(&fine, &coarse, |c| {
                read[c].store(true, Ordering::Relaxed);
                1.0
            })
            .unwrap();
            let want: Vec<usize> = (0..coarse.len()).filter(|&c| read[c].load(Ordering::Relaxed)).collect();
            assert_eq!(focus_nodes(&fine, &coarse).indices(), want, "{dims:?}");
        }
        // A fine grid sticking out of the coarse one.
        let outside = Grid::centered([15.0, 0.0, 0.0], [20.0, 17.0, 22.0], [33, 29, 37]).unwrap();
        let err = focus_boundary_values(&outside, &coarse, |_| 0.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not contained"), "{err}");
        let u = vec![0.0; coarse.len()];
        assert_eq!(focus_boundary(&outside, &coarse, &u).unwrap_err().to_string(), err);
    }

    #[test]
    fn sdh_dipole_reduces_to_vacuum_when_dielectrics_match() {
        // ±q at ±s on x: far-field ≈ l_B0 μ cosθ / r² with μ = 2qs (plus O(s³/r⁴)).
        let m = medium(1.0, 1.0, 0.0);
        let (q, s) = (0.5, 0.25);
        let mp = Multipoles::of(&[atom([s, 0.0, 0.0], q, 1.0), atom([-s, 0.0, 0.0], -q, 1.0)]);
        let p = [30.0, 40.0, 0.0];
        let r: f64 = 50.0;
        let want = m.lb0 * 2.0 * q * s * (30.0 / r) / (r * r);
        let got = mp.potential(p, &m);
        assert!((got - want).abs() < 1e-3 * want.abs(), "{got} vs {want}");
    }
}
