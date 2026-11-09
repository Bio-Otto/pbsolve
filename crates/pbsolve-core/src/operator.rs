//! Finite-volume discretisation of the LPBE, scaled by the cell volume so the
//! matrix is symmetric positive definite:
//!
//! ```text
//! (A u)_c = Σ_axes c_a [ε_{c-½}(u_c - u_{c-1}) + ε_{c+½}(u_c - u_{c+1})] + κ̄²_c V u_c
//! c_x = h_y h_z / h_x  (etc.),   V = h_x h_y h_z
//! ```
//!
//! Only interior nodes are unknowns; boundary rows are identically zero and
//! Dirichlet values enter through the right-hand side.

use crate::grid::Grid;
use crate::real::{convert, Real};
use crate::surface::CoefficientMaps;
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct StencilOperator {
    pub dims: [usize; 3],
    /// Face coefficients c_x, c_y, c_z.
    pub coef: [f64; 3],
    pub eps_x: Vec<f64>,
    pub eps_y: Vec<f64>,
    pub eps_z: Vec<f64>,
    /// κ̄² · V at every node.
    pub kvol: Vec<f64>,
    /// 1/diag(A) on interior nodes, 0 on boundary nodes (Jacobi preconditioner).
    pub inv_diag: Vec<f64>,
}

impl StencilOperator {
    pub fn new(grid: &Grid, maps: CoefficientMaps) -> Self {
        let coef = Self::face_coefficients(grid);
        let vol = grid.cell_volume();
        let kvol: Vec<f64> = maps.kappa_sq.par_iter().map(|k| k * vol).collect();
        Self::from_parts(grid.dims, coef, maps.eps_x, maps.eps_y, maps.eps_z, kvol)
    }

    /// The face coefficients c_x = h_y h_z / h_x, c_y, c_z of `grid`.
    pub fn face_coefficients(grid: &Grid) -> [f64; 3] {
        let [hx, hy, hz] = grid.spacing;
        [hy * hz / hx, hx * hz / hy, hx * hy / hz]
    }

    /// The operator of given face coefficients, ε maps and κ̄²V; computes the
    /// inverse diagonal as [`new`](Self::new) does.
    pub fn from_parts(
        dims: [usize; 3],
        coef: [f64; 3],
        eps_x: Vec<f64>,
        eps_y: Vec<f64>,
        eps_z: Vec<f64>,
        kvol: Vec<f64>,
    ) -> Self {
        let [nx, ny, nz] = dims;
        let (sx, sy, sz) = (1usize, nx, nx * ny);
        let mut inv_diag = vec![0.0; nx * ny * nz];
        inv_diag.par_iter_mut().enumerate().for_each(|(c, d)| {
            let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
            if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
                return;
            }
            let diag = coef[0] * (eps_x[c - sx] + eps_x[c])
                + coef[1] * (eps_y[c - sy] + eps_y[c])
                + coef[2] * (eps_z[c - sz] + eps_z[c])
                + kvol[c];
            *d = 1.0 / diag;
        });
        Self {
            dims,
            coef,
            eps_x,
            eps_y,
            eps_z,
            kvol,
            inv_diag,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// y = A x on interior nodes, y = 0 on boundary nodes. Boundary values of
    /// `x` ARE read (needed for right-hand-side assembly).
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        apply7(
            self.dims,
            self.coef,
            &self.eps_x,
            &self.eps_y,
            &self.eps_z,
            &self.kvol,
            x,
            y,
        );
    }
}

/// The 7-point operator in another precision (multigrid in f32).
#[derive(Clone, Debug)]
pub struct Stencil7<T: Real> {
    pub dims: [usize; 3],
    pub coef: [T; 3],
    pub eps_x: Vec<T>,
    pub eps_y: Vec<T>,
    pub eps_z: Vec<T>,
    pub kvol: Vec<T>,
}

impl<T: Real> Stencil7<T> {
    /// Rounds every coefficient of `op` to `T`.
    pub fn from_operator(op: &StencilOperator) -> Self {
        Self {
            dims: op.dims,
            coef: op.coef.map(T::from_f64),
            eps_x: convert(&op.eps_x),
            eps_y: convert(&op.eps_y),
            eps_z: convert(&op.eps_z),
            kvol: convert(&op.kvol),
        }
    }

    pub fn apply(&self, x: &[T], y: &mut [T]) {
        apply7(
            self.dims,
            self.coef,
            &self.eps_x,
            &self.eps_y,
            &self.eps_z,
            &self.kvol,
            x,
            y,
        );
    }
}

/// y = A x for the 7-point operator in any precision (CUDA: `pb_apply`, same
/// expression order).
#[allow(clippy::too_many_arguments)]
pub fn apply7<T: Real>(
    dims: [usize; 3],
    coef: [T; 3],
    eps_x: &[T],
    eps_y: &[T],
    eps_z: &[T],
    kvol: &[T],
    x: &[T],
    y: &mut [T],
) {
    apply7_with(dims, coef, eps_x, eps_y, eps_z, kvol, x, y, |_, _, v| v);
}

/// `out[c] ← combine(out[c], c, v_c)` for every node c, where v_c is the value
/// [`apply7`] writes: (A x)_c on interior nodes, 0 on boundary nodes. With
/// `combine = |_, _, v| v` this is `apply7`; the multigrid residuals
/// ([`crate::backend::Backend::residual`]) use it to fuse A x with the update.
#[allow(clippy::too_many_arguments)]
pub fn apply7_with<T: Real, F>(
    dims: [usize; 3],
    coef: [T; 3],
    eps_x: &[T],
    eps_y: &[T],
    eps_z: &[T],
    kvol: &[T],
    x: &[T],
    out: &mut [T],
    combine: F,
) where
    F: Fn(T, usize, T) -> T + Sync,
{
    let [nx, ny, nz] = dims;
    let (sx, sy, sz) = (1usize, nx, nx * ny);
    let [cx, cy, cz] = coef;
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        let base = k * nx * ny;
        if k == 0 || k == nz - 1 {
            for (m, o) in plane.iter_mut().enumerate() {
                *o = combine(*o, base + m, T::ZERO);
            }
            return;
        }
        for j in 0..ny {
            let first = base + j * nx;
            let row = &mut plane[j * nx..(j + 1) * nx];
            if j == 0 || j == ny - 1 {
                for (i, o) in row.iter_mut().enumerate() {
                    *o = combine(*o, first + i, T::ZERO);
                }
                continue;
            }
            row[0] = combine(row[0], first, T::ZERO);
            row[nx - 1] = combine(row[nx - 1], first + nx - 1, T::ZERO);
            for (i, o) in row.iter_mut().enumerate().take(nx - 1).skip(1) {
                let c = first + i;
                let xc = x[c];
                let v = cx * (eps_x[c - sx] * (xc - x[c - sx]) + eps_x[c] * (xc - x[c + sx]))
                    + cy * (eps_y[c - sy] * (xc - x[c - sy]) + eps_y[c] * (xc - x[c + sy]))
                    + cz * (eps_z[c - sz] * (xc - x[c - sz]) + eps_z[c] * (xc - x[c + sz]))
                    + kvol[c] * xc;
                *o = combine(*o, c, v);
            }
        }
    });
}

/// b = 4π l_B0 q − A g_ext on interior nodes (0 on the boundary), so that the
/// full solution is u = x + g_ext with A x = b.
pub fn assemble_rhs(op: &StencilOperator, node_charge: &[f64], g_ext: &[f64], lb0: f64) -> Vec<f64> {
    let mut ag = vec![0.0; op.len()];
    op.apply(g_ext, &mut ag);
    let src = 4.0 * PI * lb0;
    node_charge
        .par_iter()
        .zip(ag.par_iter())
        .zip(op.inv_diag.par_iter())
        .map(|((q, a), d)| if *d == 0.0 { 0.0 } else { src * q - a })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Atom;
    use crate::params::PbParams;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn operator_is_symmetric() {
        let grid = Grid::centered([0.0; 3], [12.0, 10.0, 11.0], [13, 11, 12]).unwrap();
        let atoms = [Atom {
            position: [0.5, 0.0, -0.3],
            charge: 1.0,
            radius: 2.0,
        }];
        let params = PbParams::default().with_salt(0.15, 2.0);
        let op = StencilOperator::new(&grid, CoefficientMaps::solvated(&grid, &atoms, &params));
        let mask = |v: &mut Vec<f64>| {
            for (c, x) in v.iter_mut().enumerate() {
                if op.inv_diag[c] == 0.0 {
                    *x = 0.0;
                }
            }
        };
        let mut x = pseudo_random(grid.len(), 1);
        let mut y = pseudo_random(grid.len(), 2);
        mask(&mut x);
        mask(&mut y);
        let (mut ax, mut ay) = (vec![0.0; grid.len()], vec![0.0; grid.len()]);
        op.apply(&x, &mut ax);
        op.apply(&y, &mut ay);
        let yax: f64 = y.iter().zip(&ax).map(|(a, b)| a * b).sum();
        let xay: f64 = x.iter().zip(&ay).map(|(a, b)| a * b).sum();
        assert!((yax - xay).abs() < 1e-9 * yax.abs().max(1.0), "{yax} vs {xay}");
    }
}
