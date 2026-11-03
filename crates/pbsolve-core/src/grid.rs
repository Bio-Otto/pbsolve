//! Regular Cartesian grid. Storage order is x-fastest:
//! `index(i, j, k) = i + nx * (j + ny * k)`. The same layout is used by every
//! backend so fields can be moved between them without reordering.

use crate::error::{invalid, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// Number of nodes along x, y, z.
    pub dims: [usize; 3],
    /// Node spacing along x, y, z in Å.
    pub spacing: [f64; 3],
    /// Position of node (0, 0, 0) in Å.
    pub origin: [f64; 3],
}

impl Grid {
    /// Grid of `dims` nodes spanning `lengths` (Å), centred at `center`.
    pub fn centered(center: [f64; 3], lengths: [f64; 3], dims: [usize; 3]) -> Result<Self> {
        for a in 0..3 {
            if dims[a] < 5 {
                return invalid(format!("grid needs at least 5 nodes per axis, got {:?}", dims));
            }
            if lengths[a].is_nan() || lengths[a] <= 0.0 {
                return invalid(format!("grid lengths must be positive, got {:?}", lengths));
            }
        }
        let spacing = [0, 1, 2].map(|a| lengths[a] / (dims[a] - 1) as f64);
        Self::from_spacing(center, spacing, dims)
    }

    /// Grid of `dims` nodes with the given spacing, centred at `center`.
    /// The origin is computed exactly as APBS does (`Vpmgp`:
    /// `xlen = (dime-1)·h`, `xmin = xcent − xlen/2`), so node coordinates are
    /// bit-identical to APBS for the same centre and spacing.
    pub fn from_spacing(center: [f64; 3], spacing: [f64; 3], dims: [usize; 3]) -> Result<Self> {
        let mut origin = [0.0; 3];
        for a in 0..3 {
            if dims[a] < 5 {
                return invalid(format!("grid needs at least 5 nodes per axis, got {:?}", dims));
            }
            if spacing[a].is_nan() || spacing[a] <= 0.0 || !center[a].is_finite() {
                return invalid(format!(
                    "grid spacing must be positive and the centre finite, got spacing {spacing:?}, centre {center:?}"
                ));
            }
            let length = (dims[a] - 1) as f64 * spacing[a];
            origin[a] = center[a] - length / 2.0;
        }
        let n = dims[0] as u128 * dims[1] as u128 * dims[2] as u128;
        if n > i32::MAX as u128 {
            return invalid("grid has more than 2^31 nodes");
        }
        Ok(Self { dims, spacing, origin })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    #[inline]
    pub fn ijk(&self, idx: usize) -> (usize, usize, usize) {
        let nx = self.dims[0];
        let ny = self.dims[1];
        (idx % nx, (idx / nx) % ny, idx / (nx * ny))
    }

    #[inline]
    pub fn position(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            self.origin[0] + i as f64 * self.spacing[0],
            self.origin[1] + j as f64 * self.spacing[1],
            self.origin[2] + k as f64 * self.spacing[2],
        ]
    }

    pub fn lengths(&self) -> [f64; 3] {
        [0, 1, 2].map(|a| self.spacing[a] * (self.dims[a] - 1) as f64)
    }

    pub fn upper(&self) -> [f64; 3] {
        let l = self.lengths();
        [0, 1, 2].map(|a| self.origin[a] + l[a])
    }

    pub fn center(&self) -> [f64; 3] {
        let l = self.lengths();
        [0, 1, 2].map(|a| self.origin[a] + 0.5 * l[a])
    }

    #[inline]
    pub fn cell_volume(&self) -> f64 {
        self.spacing[0] * self.spacing[1] * self.spacing[2]
    }

    #[inline]
    pub fn is_boundary(&self, i: usize, j: usize, k: usize) -> bool {
        i == 0 || j == 0 || k == 0 || i == self.dims[0] - 1 || j == self.dims[1] - 1 || k == self.dims[2] - 1
    }

    /// True if `p` lies inside the box (boundary included).
    pub fn contains(&self, p: [f64; 3]) -> bool {
        let up = self.upper();
        (0..3).all(|a| p[a] >= self.origin[a] - 1e-9 && p[a] <= up[a] + 1e-9)
    }

    /// True if the 8-node trilinear stencil of `p` touches no boundary node,
    /// i.e. a charge at `p` can be represented on the interior unknowns.
    pub fn contains_interior(&self, p: [f64; 3]) -> bool {
        let up = self.upper();
        (0..3).all(|a| p[a] >= self.origin[a] + self.spacing[a] && p[a] < up[a] - self.spacing[a])
    }

    /// Trilinear stencil (8 node indices and weights) of point `p`.
    /// Returns `None` if `p` lies outside the grid.
    pub fn trilinear(&self, p: [f64; 3]) -> Option<([usize; 8], [f64; 8])> {
        if !self.contains(p) {
            return None;
        }
        let mut base = [0usize; 3];
        let mut t = [0.0f64; 3];
        for a in 0..3 {
            let f = (p[a] - self.origin[a]) / self.spacing[a];
            let i0 = (f.floor().max(0.0) as usize).min(self.dims[a] - 2);
            base[a] = i0;
            t[a] = (f - i0 as f64).clamp(0.0, 1.0);
        }
        let mut idx = [0usize; 8];
        let mut w = [0.0f64; 8];
        let mut n = 0;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let wx = if di == 0 { 1.0 - t[0] } else { t[0] };
                    let wy = if dj == 0 { 1.0 - t[1] } else { t[1] };
                    let wz = if dk == 0 { 1.0 - t[2] } else { t[2] };
                    idx[n] = self.index(base[0] + di, base[1] + dj, base[2] + dk);
                    w[n] = wx * wy * wz;
                    n += 1;
                }
            }
        }
        Some((idx, w))
    }

    /// Trilinear interpolation of a node field at `p`.
    pub fn interpolate(&self, field: &[f64], p: [f64; 3]) -> Option<f64> {
        let (idx, w) = self.trilinear(p)?;
        Some((0..8).map(|n| field[idx[n]] * w[n]).sum())
    }

    /// Indices of all boundary nodes.
    pub fn boundary_indices(&self) -> Vec<usize> {
        let [nx, ny, nz] = self.dims;
        let mut out = Vec::with_capacity(2 * (nx * ny + ny * nz + nx * nz));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if self.is_boundary(i, j, k) {
                        out.push(self.index(i, j, k));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_is_exact_for_linear_fields() {
        let g = Grid::centered([1.0, -2.0, 0.5], [10.0, 12.0, 8.0], [11, 13, 9]).unwrap();
        let f: Vec<f64> = (0..g.len())
            .map(|idx| {
                let (i, j, k) = g.ijk(idx);
                let p = g.position(i, j, k);
                2.0 * p[0] - 3.0 * p[1] + 0.5 * p[2] + 1.0
            })
            .collect();
        let p = [0.37, -1.11, 2.9];
        let exact = 2.0 * p[0] - 3.0 * p[1] + 0.5 * p[2] + 1.0;
        assert!((g.interpolate(&f, p).unwrap() - exact).abs() < 1e-12);
    }
}
