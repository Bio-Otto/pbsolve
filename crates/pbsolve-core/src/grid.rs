//! Regular Cartesian grid. Storage order is x-fastest:
//! `index(i, j, k) = i + nx * (j + ny * k)`. The same layout is used by every
//! backend so fields can be moved between them without reordering.

use crate::error::{invalid, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// Number of nodes along x, y, z.
    pub dims: [usize; 3],
    /// Node spacing along x, y, z in Å.
    pub spacing: [f64; 3],
    /// Position of node (0, 0, 0) in Å.
    pub origin: [f64; 3],
    /// Periodic axes (x and y only). On a periodic axis of `n` nodes, node
    /// `n − 1` is the image of node 0: nodes `0 ..= n − 2` are unknowns, the
    /// last node is neither an unknown nor a Dirichlet node (see [`axis_unknown`]).
    pub periodic: [bool; 3],
}

/// Whether node `i` of an axis of `n` nodes is an unknown: `1 ..= n − 2`, and
/// node 0 too on a periodic axis (whose node `n − 1` is the image of node 0).
#[inline]
pub fn axis_unknown(i: usize, n: usize, periodic: bool) -> bool {
    i + 1 < n && (i > 0 || periodic)
}

/// The neighbour `i − 1` of unknown node `i` (node `n − 2` for node 0 of a periodic axis).
#[inline]
pub fn axis_prev(i: usize, n: usize, periodic: bool) -> usize {
    if periodic && i == 0 {
        n - 2
    } else {
        i - 1
    }
}

/// The neighbour `i + 1` of unknown node `i` (node 0 for node `n − 2` of a periodic axis).
#[inline]
pub fn axis_next(i: usize, n: usize, periodic: bool) -> usize {
    if periodic && i + 2 == n {
        0
    } else {
        i + 1
    }
}

/// Whether all three axes satisfy [`axis_unknown`].
#[inline]
pub fn is_unknown(i: usize, j: usize, k: usize, dims: [usize; 3], periodic: [bool; 3]) -> bool {
    axis_unknown(i, dims[0], periodic[0])
        && axis_unknown(j, dims[1], periodic[1])
        && axis_unknown(k, dims[2], periodic[2])
}

/// Number of unknowns of a grid: `n − 2` per axis, `n − 1` per periodic axis.
pub fn unknown_count(dims: [usize; 3], periodic: [bool; 3]) -> usize {
    (0..3)
        .map(|a| dims[a].saturating_sub(2 - usize::from(periodic[a])))
        .product()
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
        Ok(Self {
            dims,
            spacing,
            origin,
            periodic: [false; 3],
        })
    }

    /// This grid with the given periodic axes (x and y only).
    pub fn with_periodic(mut self, periodic: [bool; 3]) -> Self {
        assert!(!periodic[2], "the z axis cannot be periodic");
        self.periodic = periodic;
        self
    }

    /// True if an axis is periodic.
    #[inline]
    pub fn is_periodic(&self) -> bool {
        self.periodic.iter().any(|&p| p)
    }

    /// The node a node stands for: the image node 0 for node `n − 1` of a
    /// periodic axis, the node itself otherwise.
    #[inline]
    pub fn canonical(&self, c: usize) -> usize {
        if !self.is_periodic() {
            return c;
        }
        let (mut i, mut j, k) = self.ijk(c);
        if self.periodic[0] && i + 1 == self.dims[0] {
            i = 0;
        }
        if self.periodic[1] && j + 1 == self.dims[1] {
            j = 0;
        }
        self.index(i, j, k)
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

    /// True for a Dirichlet node: a node on a face of a non-periodic axis.
    #[inline]
    pub fn is_boundary(&self, i: usize, j: usize, k: usize) -> bool {
        let [px, py, _] = self.periodic;
        (!px && (i == 0 || i == self.dims[0] - 1))
            || (!py && (j == 0 || j == self.dims[1] - 1))
            || k == 0
            || k == self.dims[2] - 1
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
        self.interpolate_with(|c| field[c], p)
    }

    /// [`interpolate`](Self::interpolate) of a field read node by node
    /// through `field` (same arithmetic, same result).
    pub fn interpolate_with(&self, field: impl Fn(usize) -> f64, p: [f64; 3]) -> Option<f64> {
        let (idx, w) = self.trilinear(p)?;
        Some((0..8).map(|n| field(idx[n]) * w[n]).sum())
    }

    /// The position of node (i, j, k) in
    /// [`boundary_indices`](Self::boundary_indices), if it is a boundary node.
    #[inline]
    pub fn boundary_position(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        let [nx, ny, nz] = self.dims;
        if k == 0 || k == nz - 1 {
            // Bottom and top planes are boundary nodes throughout.
            let below = if k == 0 {
                0
            } else {
                nx * ny + (nz - 2) * self.inner_plane_boundary()
            };
            return Some(below + i + nx * j);
        }
        let below = nx * ny + (k - 1) * self.inner_plane_boundary();
        let (bx, by) = (!self.periodic[0], !self.periodic[1]);
        // Rows j = 0 and ny − 1 are full rows (y faces), the rows between
        // them contribute their end nodes (x faces).
        let ends = if bx { 2 } else { 0 };
        if by && j == 0 {
            Some(below + i)
        } else if by && j == ny - 1 {
            Some(below + nx + ends * (ny - 2) + i)
        } else if bx && (i == 0 || i == nx - 1) {
            let first = if by { nx } else { 0 };
            Some(below + first + 2 * (j - usize::from(by)) + usize::from(i != 0))
        } else {
            None
        }
    }

    /// Boundary nodes of a plane 0 < k < nz − 1: the two full rows of the y
    /// faces and both ends of the rows between them (x faces); periodic axes
    /// have no faces.
    #[inline]
    fn inner_plane_boundary(&self) -> usize {
        let [nx, ny, _] = self.dims;
        let (bx, by) = (!self.periodic[0], !self.periodic[1]);
        let rows = if by { ny - 2 } else { ny };
        usize::from(by) * 2 * nx + usize::from(bx) * 2 * rows
    }

    /// Indices of all boundary nodes, in increasing order.
    pub fn boundary_indices(&self) -> Vec<usize> {
        let [nx, ny, nz] = self.dims;
        let (bx, by) = (!self.periodic[0], !self.periodic[1]);
        let mut out = Vec::with_capacity(2 * nx * ny + (nz - 2) * self.inner_plane_boundary());
        for k in 0..nz {
            for j in 0..ny {
                let row = nx * (j + ny * k);
                if k == 0 || k == nz - 1 || (by && (j == 0 || j == ny - 1)) {
                    out.extend(row..row + nx);
                } else if bx {
                    out.push(row);
                    out.push(row + nx - 1);
                }
            }
        }
        out
    }
}

/// A set of node indices below `n`, one bit per node: built from a parallel
/// iterator, it lists its members in increasing order and gives the position
/// of a member in that list in constant time.
#[derive(Clone, Debug)]
pub struct NodeSet {
    words: Vec<u64>,
    /// Members in the words before each word.
    before: Vec<usize>,
    len: usize,
}

impl NodeSet {
    /// The set of the indices `idx` yields (repetitions allowed).
    ///
    /// # Panics
    /// If an index is `n` or larger.
    pub fn from_indices(n: usize, idx: impl ParallelIterator<Item = usize>) -> Self {
        let bits: Vec<AtomicU64> = (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        idx.for_each(|c| {
            assert!(c < n, "node {c} outside a grid of {n} nodes");
            bits[c / 64].fetch_or(1 << (c % 64), Ordering::Relaxed);
        });
        let words: Vec<u64> = bits.into_iter().map(AtomicU64::into_inner).collect();
        let mut before = Vec::with_capacity(words.len());
        let mut len = 0;
        for w in &words {
            before.push(len);
            len += w.count_ones() as usize;
        }
        Self { words, before, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The members in increasing order.
    pub fn indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len);
        for (k, &w) in self.words.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                out.push(64 * k + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        out
    }

    /// The position of `c` in [`indices`](Self::indices), if it is a member.
    #[inline]
    pub fn position(&self, c: usize) -> Option<usize> {
        let w = *self.words.get(c / 64)?;
        let bit = 1u64 << (c % 64);
        (w & bit != 0).then(|| self.before[c / 64] + (w & (bit - 1)).count_ones() as usize)
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

    #[test]
    fn interpolation_through_a_closure_equals_the_slice_form() {
        let g = Grid::centered([0.3, 0.1, -0.2], [9.0, 7.0, 8.0], [10, 9, 11]).unwrap();
        let f: Vec<f64> = (0..g.len()).map(|c| ((c * 7919) % 1000) as f64 / 37.0 - 9.0).collect();
        for p in [[0.37, -1.11, 2.9], [-4.2, 3.3, -3.8], [4.8, 3.6, 3.8], [0.0; 3]] {
            let a = g.interpolate(&f, p).unwrap();
            let b = g.interpolate_with(|c| f[c], p).unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "{p:?}");
        }
        assert_eq!(g.interpolate_with(|c| f[c], [9.0, 0.0, 0.0]), None);
    }

    #[test]
    fn node_set_lists_and_ranks_its_members() {
        for n in [0, 1, 63, 64, 65, 1000, 4097] {
            let picks: Vec<usize> = (0..3 * n)
                .map(|m| (m * 2654435761) % n.max(1))
                .filter(|_| n > 0)
                .collect();
            let set = NodeSet::from_indices(n, picks.par_iter().copied());
            let mut want = picks.clone();
            want.sort_unstable();
            want.dedup();
            assert_eq!(set.indices(), want, "n = {n}");
            assert_eq!((set.len(), set.is_empty()), (want.len(), want.is_empty()));
            for c in 0..n + 70 {
                assert_eq!(set.position(c), want.binary_search(&c).ok(), "n = {n}, c = {c}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside a grid of 10 nodes")]
    fn node_set_rejects_indices_outside_the_grid() {
        NodeSet::from_indices(10, [3usize, 10].into_par_iter());
    }

    #[test]
    fn boundary_indices_are_the_boundary_nodes_in_order() {
        let periodic = [
            [false; 3],
            [true, false, false],
            [false, true, false],
            [true, true, false],
        ];
        for (dims, periodic) in [[5, 5, 5], [7, 5, 6], [9, 17, 5], [33, 6, 11]]
            .into_iter()
            .flat_map(|d| periodic.map(|p| (d, p)))
        {
            let g = Grid::from_spacing([0.0; 3], [0.5; 3], dims)
                .unwrap()
                .with_periodic(periodic);
            let brute: Vec<usize> = (0..g.len())
                .filter(|&c| {
                    let (i, j, k) = g.ijk(c);
                    g.is_boundary(i, j, k)
                })
                .collect();
            assert_eq!(g.boundary_indices(), brute, "{dims:?} {periodic:?}");
            for c in 0..g.len() {
                let (i, j, k) = g.ijk(c);
                assert_eq!(
                    g.boundary_position(i, j, k),
                    brute.binary_search(&c).ok(),
                    "{dims:?} {periodic:?} node {c}"
                );
            }
        }
    }
}
