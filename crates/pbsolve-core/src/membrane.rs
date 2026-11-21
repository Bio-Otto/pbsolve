// SPDX-License-Identifier: MIT AND BSD-3-Clause
// The membrane rule is derived from APBS 3.4.1 examples/helix/draw_membrane2.c (M. Grabe), Copyright (c) 2010-2022
// Battelle Memorial Institute and other APBS contributors, BSD-3-Clause. See THIRD_PARTY_NOTICES.md.

//! Implicit membrane: a slab normal to z without mobile ions, of one
//! dielectric (APBS' `draw_membrane2`, APBSmem [Callenberg et al. 2010];
//! Amber `membraneopt=1`) or with a depth profile (Greene et al. 2019).
//!
//! The rule is that of `draw_membrane2` (APBS 3.4.1 `examples/helix`): a
//! face of a dielectric map whose midpoint lies in the slab (bottom ≤ z ≤
//! top, both included), farther than the exclusion radius from the axis and
//! whose value exceeds `pdie + 0.05` (not solute) takes the membrane's
//! dielectric; a node in the slab outside the exclusion radius loses its ion
//! accessibility. The exclusion radius grows linearly from `exclusion[0]` at
//! the bottom to `exclusion[1]` at the top (a cone for channel pores), around
//! the vertical axis through `draw_membrane2`'s centre, the grid's centre
//! shifted by half a spacing in x and y (draw_membrane2 takes the x spacing for
//! y too; the same on its cubic grids). With `fill` only the parts connected
//! to the sides of the box in their z plane become membrane, so that pores and
//! cavities enclosed by the solute keep their water and ions.

use crate::grid::Grid;
use std::collections::VecDeque;
use std::sync::Arc;

/// The dielectric profile of Greene et al. (2019), Table 1 "method 3", as
/// (depth below the membrane surface in Å, ε) for a 40 Å membrane: ε = 80 at
/// the headgroup plane (20 Å from the centre), 1 at the centre.
pub const GREENE_2019: [[f64; 2]; 5] = [[0.0, 80.0], [5.0, 20.0], [10.0, 4.0], [15.0, 2.0], [20.0, 1.0]];

/// Weight of the solvent-accessible surface at height `z` for the non-polar
/// term of a membrane from `bottom` to `bottom + thickness` (GBSW, Im, Feig &
/// Brooks 2003): 0 in the slab farther than `width` from its surfaces, 1 in
/// the water farther than `width` from them, and between
/// H = ½ + ¾ s − ¼ s³ with s = (|z − centre| − thickness/2)/width, which is
/// continuous with its slope. `width` 0 is a step (0 on the surfaces).
pub fn surface_switch(z: f64, bottom: f64, thickness: f64, width: f64) -> f64 {
    let u = (z - (bottom + 0.5 * thickness)).abs() - 0.5 * thickness;
    if u <= -width {
        0.0
    } else if u >= width {
        1.0
    } else {
        let s = u / width;
        0.5 + 0.75 * s - 0.25 * s * s * s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Membrane {
    /// z of the lower membrane surface, Å (APBS `zmem`).
    pub bottom: f64,
    /// Thickness, Å (APBS `Lmem`).
    pub thickness: f64,
    /// Dielectric of a uniform membrane (APBS `mdie`).
    pub dielectric: f64,
    /// Depth profile: (depth below the nearer surface in Å, ε) with depths
    /// increasing from 0, interpolated by PCHIP and constant beyond the ends;
    /// `None` for the uniform `dielectric`.
    pub profile: Option<Arc<[[f64; 2]]>>,
    /// Exclusion radius (Å) at the bottom and at the top (APBS `R_bottom`, `R_top`).
    pub exclusion: [f64; 2],
    /// Keep regions not connected to the sides of the box in their z plane as solvent.
    pub fill: bool,
}

impl Membrane {
    pub fn top(&self) -> f64 {
        self.bottom + self.thickness
    }

    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            self.bottom,
            self.thickness,
            self.dielectric,
            self.exclusion[0],
            self.exclusion[1],
        ];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err("membrane parameters must be finite".into());
        }
        if self.thickness <= 0.0 || self.dielectric <= 0.0 {
            return Err("membrane thickness and dielectric must be positive".into());
        }
        if self.exclusion.iter().any(|r| *r < 0.0) {
            return Err("membrane exclusion radii must be >= 0".into());
        }
        if let Some(p) = &self.profile {
            if p.len() < 2 || p[0][0] != 0.0 {
                return Err("a membrane profile needs at least two points, the first at depth 0".into());
            }
            if p.windows(2).any(|w| w[1][0].is_nan() || w[1][0] <= w[0][0]) {
                return Err("membrane profile depths must increase".into());
            }
            if p.iter().any(|q| !q[1].is_finite() || q[1] <= 0.0) {
                return Err("membrane profile dielectrics must be positive".into());
            }
        }
        Ok(())
    }

    /// Dielectric at height z inside the slab.
    pub fn dielectric_at(&self, z: f64) -> f64 {
        self.value(self.profile.as_deref().map(Pchip::new).as_ref(), z)
    }

    fn value(&self, profile: Option<&Pchip>, z: f64) -> f64 {
        match profile {
            None => self.dielectric,
            Some(p) => p.eval((z - self.bottom).min(self.top() - z)),
        }
    }

    /// Exclusion radius at height z (`draw_membrane2`'s linear interpolation).
    fn exclusion_at(&self, z: f64) -> f64 {
        let (z0, z1) = (self.bottom, self.top());
        let [r0, r1] = self.exclusion;
        (r1 * (z - z0) - r0 * (z - z1)) / (z1 - z0)
    }

    /// Whether a point is in the slab outside the exclusion radius.
    fn covers(&self, p: [f64; 3], axis: [f64; 2]) -> bool {
        let r = ((p[0] - axis[0]).powi(2) + (p[1] - axis[1]).powi(2)).sqrt();
        p[2] <= self.top() && p[2] >= self.bottom && r > self.exclusion_at(p[2])
    }

    /// Adds the membrane to the solvated state's maps `[eps_x, eps_y, eps_z,
    /// kappa_sq]` of one level.
    pub fn apply(&self, grid: &Grid, pdie: f64, maps: &mut [Vec<f64>; 4]) {
        let h = grid.spacing;
        let [nx, ny, nz] = grid.dims;
        // draw_membrane2: x0 = origin(dielx) + nx·hx/2 − hx/2 = origin + nx·hx/2, y0 = origin + ny·hx/2
        // (hx where hy is meant; the same on cubic grids).
        let axis = [
            grid.origin[0] + nx as f64 * h[0] / 2.0,
            grid.origin[1] + ny as f64 * h[1] / 2.0,
        ];
        let shifts = [
            [0.5 * h[0], 0.0, 0.0],
            [0.0, 0.5 * h[1], 0.0],
            [0.0, 0.0, 0.5 * h[2]],
            [0.0; 3],
        ];
        let pchip = self.profile.as_deref().map(Pchip::new);
        let value = |z: f64| self.value(pchip.as_ref(), z);
        for (a, map) in maps.iter_mut().enumerate() {
            let shift = shifts[a];
            let mut mark = vec![false; map.len()];
            for k in 0..nz {
                let z = grid.origin[2] + k as f64 * h[2] + shift[2];
                if z > self.top() || z < self.bottom {
                    continue;
                }
                for j in 0..ny {
                    for i in 0..nx {
                        let c = i + nx * (j + ny * k);
                        let p = [
                            grid.origin[0] + i as f64 * h[0] + shift[0],
                            grid.origin[1] + j as f64 * h[1] + shift[1],
                            z,
                        ];
                        // Faces: not solute. Nodes: draw_membrane2 zeroes every node; those without
                        // ions stay 0 anyway, and `fill` must not connect through them.
                        let open = if a < 3 { map[c] > pdie + 0.05 } else { map[c] > 0.0 };
                        mark[c] = open && self.covers(p, axis);
                    }
                }
            }
            if self.fill {
                keep_connected_to_the_sides(&mut mark, grid.dims);
            }
            for k in 0..nz {
                let z = grid.origin[2] + k as f64 * h[2] + shift[2];
                for c in nx * ny * k..nx * ny * (k + 1) {
                    if mark[c] {
                        map[c] = if a < 3 { value(z) } else { 0.0 };
                    }
                }
            }
        }
    }
}

impl Membrane {
    /// The membrane potential across the slab without solute (kT/e) at the z
    /// nodes of `grid`, with the bath below the membrane held at `v` and the
    /// one above at 0: the finite-difference solution of the one-dimensional
    /// problem a column of the grid far from any solute forms (its z faces in
    /// the slab take the membrane's dielectric, its nodes in the slab no ions),
    /// with `v` at the grid's bottom face and 0 at its top face. It is the
    /// boundary of the membrane-potential field (Roux 1997; the planar
    /// solution of APBSmem, here on the grid and for any profile).
    pub fn potential_profile(&self, grid: &Grid, sdie: f64, kappa_bar_sq: f64, v: f64) -> Vec<f64> {
        let nz = grid.dims[2];
        let h = grid.spacing[2];
        let z = |k: f64| grid.origin[2] + k * h;
        let pchip = self.profile.as_deref().map(Pchip::new);
        let in_slab = |zz: f64| zz >= self.bottom && zz <= self.top();
        // Face k + ½ and node k of the column.
        let eps: Vec<f64> = (0..nz)
            .map(|k| {
                let zf = z(k as f64 + 0.5);
                if in_slab(zf) {
                    self.value(pchip.as_ref(), zf)
                } else {
                    sdie
                }
            })
            .collect();
        let kap: Vec<f64> = (0..nz)
            .map(|k| if in_slab(z(k as f64)) { 0.0 } else { kappa_bar_sq })
            .collect();
        // Rows k = 1 .. nz − 2 (per unit area, divided by h):
        // (ε_{k−½}(φ_k − φ_{k−1}) + ε_{k+½}(φ_k − φ_{k+1}))/h² + κ̄²_k φ_k = κ̄²_k v Θ_k.
        let mut phi = vec![0.0; nz];
        phi[0] = v;
        let n = nz - 2;
        let (mut c_prime, mut d_prime) = (vec![0.0; n], vec![0.0; n]);
        for r in 0..n {
            let k = r + 1;
            let (lo, hi) = (eps[k - 1] / (h * h), eps[k] / (h * h));
            let diag = lo + hi + kap[k];
            let mut rhs = if z(k as f64) < self.bottom { kap[k] * v } else { 0.0 };
            if k == 1 {
                rhs += lo * phi[0];
            }
            let sub = if k == 1 { 0.0 } else { -lo };
            let sup = if k + 2 == nz { 0.0 } else { -hi };
            let denom = diag - sub * if r > 0 { c_prime[r - 1] } else { 0.0 };
            c_prime[r] = sup / denom;
            d_prime[r] = (rhs - sub * if r > 0 { d_prime[r - 1] } else { 0.0 }) / denom;
        }
        for r in (0..n).rev() {
            phi[r + 1] = d_prime[r] - c_prime[r] * if r + 1 < n { phi[r + 2] } else { 0.0 };
        }
        phi
    }

    /// The source of the membrane-potential field as node charges (e): the
    /// ions of the bath below the membrane (nodes below the bottom of the slab
    /// with ions; pores in the slab belong to the bath above, as in APBSmem)
    /// held at `v` (kT/e), κ̄²·V·v/(4π l_B0) per node with the node's κ̄² from
    /// `kappa_sq` and cell volume V (APBS `draw_membrane2`'s charge map).
    pub fn potential_source(&self, grid: &Grid, kappa_sq: &[f64], v: f64, lb0: f64) -> Vec<f64> {
        let plane = grid.dims[0] * grid.dims[1];
        let scale = grid.cell_volume() * v / (4.0 * std::f64::consts::PI * lb0);
        kappa_sq
            .iter()
            .enumerate()
            .map(|(c, &k)| {
                let z = grid.origin[2] + (c / plane) as f64 * grid.spacing[2];
                if z < self.bottom {
                    k * scale
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Clears the marks of every z plane that no 4-connected path of marks links
/// to the plane's edge.
fn keep_connected_to_the_sides(mark: &mut [bool], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    for k in 0..nz {
        let m = &mut mark[plane * k..plane * (k + 1)];
        let mut seen = vec![false; plane];
        let mut queue = VecDeque::new();
        for j in 0..ny {
            for i in 0..nx {
                if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) && m[i + nx * j] {
                    seen[i + nx * j] = true;
                    queue.push_back((i, j));
                }
            }
        }
        while let Some((i, j)) = queue.pop_front() {
            let next = [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)];
            for (a, b) in next {
                if a < nx && b < ny && m[a + nx * b] && !seen[a + nx * b] {
                    seen[a + nx * b] = true;
                    queue.push_back((a, b));
                }
            }
        }
        for (v, s) in m.iter_mut().zip(seen) {
            *v &= s;
        }
    }
}

/// Monotone piecewise cubic Hermite interpolation (Fritsch & Carlson 1980;
/// the slopes of SciPy's `PchipInterpolator`), constant beyond the ends.
pub struct Pchip<'a> {
    points: &'a [[f64; 2]],
    slopes: Vec<f64>,
}

impl<'a> Pchip<'a> {
    pub fn new(points: &'a [[f64; 2]]) -> Self {
        let n = points.len();
        let h: Vec<f64> = points.windows(2).map(|w| w[1][0] - w[0][0]).collect();
        let delta: Vec<f64> = points
            .windows(2)
            .map(|w| (w[1][1] - w[0][1]) / (w[1][0] - w[0][0]))
            .collect();
        let mut d = vec![0.0; n];
        if n == 2 {
            d = vec![delta[0]; 2];
        } else {
            for k in 1..n - 1 {
                if delta[k - 1] * delta[k] > 0.0 {
                    // Weighted harmonic mean of the neighbouring secants.
                    let (w1, w2) = (2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]);
                    d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
                }
            }
            d[0] = end_slope(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = end_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }
        Self { points, slopes: d }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let p = self.points;
        let n = p.len();
        if x <= p[0][0] {
            return p[0][1];
        }
        if x >= p[n - 1][0] {
            return p[n - 1][1];
        }
        let k = p.partition_point(|q| q[0] <= x) - 1;
        let (x0, x1, y0, y1) = (p[k][0], p[k + 1][0], p[k][1], p[k + 1][1]);
        let h = x1 - x0;
        let t = (x - x0) / h;
        let (t2, t3) = (t * t, t * t * t);
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * y0 + h10 * h * self.slopes[k] + h01 * y1 + h11 * h * self.slopes[k + 1]
    }
}

/// SciPy's one-sided three-point end slope, kept shape-preserving.
fn end_slope(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    // numpy.sign: 0 for 0 (f64::signum gives ±1).
    let sign = |x: f64| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0) {
        0.0
    } else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs() {
        3.0 * m0
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab() -> Membrane {
        Membrane {
            bottom: -7.3,
            thickness: 12.4,
            dielectric: 2.0,
            profile: None,
            exclusion: [0.0, 0.0],
            fill: false,
        }
    }

    #[test]
    fn pchip_passes_through_the_points_and_is_monotone() {
        let p = Pchip::new(&GREENE_2019);
        for q in GREENE_2019 {
            assert_eq!(p.eval(q[0]), q[1]);
        }
        let mut last = f64::INFINITY;
        for i in 0..=2000 {
            let v = p.eval(-1.0 + i as f64 * 0.011);
            assert!(v <= last + 1e-12 && (1.0..=80.0).contains(&v), "{v}");
            last = v;
        }
        assert_eq!(p.eval(30.0), 1.0);
        assert_eq!(p.eval(-2.0), 80.0);
    }

    #[test]
    fn the_slab_gets_the_membrane_and_loses_its_ions() {
        let grid = Grid::centered([0.0; 3], [12.0, 12.0, 20.0], [13, 13, 21]).unwrap();
        let n = grid.len();
        let (pdie, sdie) = (1.0, 78.54);
        let mut maps = [vec![sdie; n], vec![sdie; n], vec![sdie; n], vec![1.5; n]];
        maps[0][grid.index(6, 6, 10)] = pdie; // a solute face in the slab stays
                                              // Both surfaces on node planes (z = −7 and 5): included, as in draw_membrane2.
        let m = Membrane {
            bottom: -7.0,
            thickness: 12.0,
            ..slab()
        };
        m.apply(&grid, pdie, &mut maps);
        let [nx, ny, _] = grid.dims;
        for (a, map) in maps.iter().enumerate() {
            for (c, v) in map.iter().enumerate() {
                let k = c / (nx * ny);
                let z = grid.origin[2] + k as f64 + if a == 2 { 0.5 } else { 0.0 };
                let inside = (m.bottom..=m.top()).contains(&z);
                let expected = match (a, inside) {
                    (0, _) if c == grid.index(6, 6, 10) => pdie,
                    (3, true) => 0.0,
                    (3, false) => 1.5,
                    (_, true) => 2.0,
                    (_, false) => sdie,
                };
                assert_eq!(*v, expected, "map {a} node {c}");
            }
        }
    }

    #[test]
    fn exclusion_cone_and_fill_keep_the_pore() {
        let grid = Grid::centered([0.0; 3], [20.0, 20.0, 20.0], [21, 21, 21]).unwrap();
        let n = grid.len();
        let [nx, ny, _] = grid.dims;
        // A solute ring of radius 4–6 Å around the axis; water inside it.
        let axis = [grid.origin[0] + 10.5, grid.origin[1] + 10.5];
        let ring = |c: usize, shift: [f64; 2]| {
            let (i, j) = (c % nx, (c / nx) % ny);
            let r = ((grid.origin[0] + i as f64 + shift[0] - axis[0]).powi(2)
                + (grid.origin[1] + j as f64 + shift[1] - axis[1]).powi(2))
            .sqrt();
            (4.0..=6.0).contains(&r)
        };
        let shifts = [[0.5, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]];
        let fresh = || -> [Vec<f64>; 4] {
            let mut maps = [vec![80.0; n], vec![80.0; n], vec![80.0; n], vec![1.0; n]];
            for (a, map) in maps.iter_mut().enumerate() {
                for (c, v) in map.iter_mut().enumerate() {
                    if ring(c, shifts[a]) {
                        *v = if a < 3 { 2.0 } else { 0.0 };
                    }
                }
            }
            maps
        };
        let inside = |c: usize, a: usize| {
            let (i, j) = (c % nx, (c / nx) % ny);
            let s = shifts[a];
            ((grid.origin[0] + i as f64 + s[0] - axis[0]).powi(2)
                + (grid.origin[1] + j as f64 + s[1] - axis[1]).powi(2))
            .sqrt()
                < 4.0
        };
        let mid = grid.index(0, 0, 10); // the z = 0 plane
        let plane = mid..mid + nx * ny;
        // Without fill the pore becomes membrane; a cone of radius 3 keeps its middle.
        let mut plain = fresh();
        let m = Membrane {
            bottom: -5.2,
            thickness: 10.4,
            ..slab()
        };
        m.apply(&grid, 2.0, &mut plain);
        assert!(plane.clone().filter(|&c| inside(c, 3)).all(|c| plain[3][c] == 0.0));
        let mut cone = fresh();
        Membrane {
            exclusion: [3.0, 3.0],
            ..m.clone()
        }
        .apply(&grid, 2.0, &mut cone);
        let r_lt_3 = |c: usize| {
            let (i, j) = (c % nx, (c / nx) % ny);
            ((grid.origin[0] + i as f64 - axis[0]).powi(2) + (grid.origin[1] + j as f64 - axis[1]).powi(2)).sqrt()
                <= 3.0
        };
        assert!(plane.clone().filter(|&c| r_lt_3(c)).all(|c| cone[3][c] == 1.0));
        // With fill the whole pore keeps its water and ions, the outside becomes membrane.
        let mut filled = fresh();
        Membrane {
            fill: true,
            ..m.clone()
        }
        .apply(&grid, 2.0, &mut filled);
        for a in 0..4 {
            for c in plane.clone() {
                let water = if a < 3 { 80.0 } else { 1.0 };
                let membrane = if a < 3 { 2.0 } else { 0.0 };
                if inside(c, a) {
                    assert_eq!(filled[a][c], water, "pore, map {a}");
                } else if !ring(c, shifts[a]) {
                    assert_eq!(filled[a][c], membrane, "outside, map {a}");
                }
            }
        }
    }

    #[test]
    fn fill_reaches_pockets_open_in_any_direction() {
        // A wall shaped like ∩ whose pocket opens towards one side only; the pocket is reached
        // only by moving in that direction from the box edge.
        let n = 11;
        for dir in 0..4 {
            let mut mark = vec![true; n * n];
            for t in 2..9 {
                // the wall: three sides of the square 2..8, open on side `dir`
                let wall = [(t, 8), (t, 2), (8, t), (2, t)];
                for (side, &(i, j)) in wall.iter().enumerate() {
                    if side != dir {
                        mark[i + n * j] = false;
                    }
                }
            }
            // Close the outside at the open side's far edge so the pocket is entered only from there.
            keep_connected_to_the_sides(&mut mark, [n, n, 1]);
            assert!(mark[5 + n * 5], "pocket open towards side {dir}");
        }
        // A closed square keeps its inside out.
        let mut mark = vec![true; n * n];
        for t in 2..9 {
            for (i, j) in [(t, 8), (t, 2), (8, t), (2, t)] {
                mark[i + n * j] = false;
            }
        }
        keep_connected_to_the_sides(&mut mark, [n, n, 1]);
        assert!(!mark[5 + n * 5] && mark[0]);
    }

    #[test]
    fn the_axis_is_the_centre_shifted_by_half_a_spacing_on_any_grid() {
        // hx = 1, hy = 2: the axis is at (0.5, 1.0); the cone of radius 1.5 keeps node (5, 5) at (0, 0).
        let grid = Grid::centered([0.0; 3], [10.0, 20.0, 10.0], [11, 11, 11]).unwrap();
        let n = grid.len();
        let mut maps = [vec![80.0; n], vec![80.0; n], vec![80.0; n], vec![1.0; n]];
        Membrane {
            bottom: -3.0,
            thickness: 6.0,
            exclusion: [1.5, 1.5],
            ..slab()
        }
        .apply(&grid, 1.0, &mut maps);
        assert_eq!(maps[3][grid.index(5, 5, 5)], 1.0);
        assert_eq!(maps[3][grid.index(5, 4, 5)], 0.0); // (0, −2): 3.04 from the axis
        assert_eq!(maps[3][grid.index(3, 5, 5)], 0.0);
    }

    #[test]
    fn the_profile_follows_the_depth_below_the_nearer_surface() {
        let m = Membrane {
            bottom: -25.0,
            thickness: 50.0,
            profile: Some(GREENE_2019.to_vec().into()),
            ..slab()
        };
        assert_eq!(m.dielectric_at(-25.0), 80.0);
        assert_eq!(m.dielectric_at(25.0), 80.0);
        assert_eq!(m.dielectric_at(-15.0), 4.0);
        assert_eq!(m.dielectric_at(20.0), 20.0);
        assert_eq!(m.dielectric_at(10.0), 2.0);
        assert_eq!(m.dielectric_at(0.0), 1.0); // the core beyond 20 Å depth
        assert_eq!(m.dielectric_at(3.0), m.dielectric_at(-3.0));
    }

    /// The column solution is the finite-difference solution of the
    /// one-dimensional problem (checked row by row), takes v and 0 at the
    /// ends, and converges to the exact profile of a uniform slab between
    /// salt water (linear in the slab, exponential in the baths); its source
    /// is the lower bath's ions.
    #[test]
    fn the_potential_profile_solves_the_column_and_converges_to_the_slab() {
        let m = Membrane {
            bottom: -15.0,
            thickness: 30.0,
            dielectric: 2.0,
            profile: None,
            exclusion: [0.0, 0.0],
            fill: false,
        };
        let (sdie, kbar2, v): (f64, f64, f64) = (78.54, 1.27, 3.9);
        let kappa = (kbar2 / sdie).sqrt();
        let a = 2.0 * v / (2.0 * 2.0 + sdie * kappa * 30.0);
        let exact = |z: f64| {
            if z < -15.0 {
                v - a * (kappa * (z + 15.0)).exp()
            } else if z > 15.0 {
                a * (-kappa * (z - 15.0)).exp()
            } else {
                (v - a) + (2.0 * a - v) * (z + 15.0) / 30.0
            }
        };
        let mut errors = Vec::new();
        let slope = (2.0 * a - v).abs() / 30.0;
        for (n, centre) in [(161, 0.1), (641, 0.1), (641, 0.0), (1281, 0.03)] {
            let grid = Grid::centered([0.0, 0.0, centre], [10.0, 10.0, 160.0], [5, 5, n]).unwrap();
            let phi = m.potential_profile(&grid, sdie, kbar2, v);
            assert_eq!((phi[0], phi[n - 1]), (v, 0.0));
            let h = grid.spacing[2];
            let z = |k: usize| grid.origin[2] + k as f64 * h;
            let in_slab = |zz: f64| (-15.0..=15.0).contains(&zz);
            let eps = |k: usize| if in_slab(z(k) + 0.5 * h) { 2.0 } else { sdie };
            for k in 1..n - 1 {
                let kap = if in_slab(z(k)) { 0.0 } else { kbar2 };
                let lhs =
                    (eps(k - 1) * (phi[k] - phi[k - 1]) + eps(k) * (phi[k] - phi[k + 1])) / (h * h) + kap * phi[k];
                let rhs = if z(k) < -15.0 { kap * v } else { 0.0 };
                assert!((lhs - rhs).abs() < 1e-9, "row {k}: {lhs} vs {rhs}");
            }
            let error = (0..n).map(|k| (phi[k] - exact(z(k))).abs()).fold(0.0, f64::max);
            // The slab's faces and nodes place each surface within a cell: at
            // most h·|slope| (measured: 0.1 Å shifts give 0.011–0.012 kT/e).
            assert!(
                error < h * slope + 1e-3,
                "n {n}, centre {centre}: {error} > {}",
                h * slope
            );
            errors.push(error);
        }
        assert!(errors[3] < 0.5 * errors[1], "{errors:?}");
        // A slab from the second node on: its first row couples φ_0 through the
        // water face below it, ε_{1/2} = sdie, not the slab face above.
        let grid = Grid::centered([0.0, 0.0, 4.0], [10.0, 10.0, 40.0], [5, 5, 41]).unwrap();
        let phi = m.potential_profile(&grid, sdie, kbar2, v);
        let row = sdie * (phi[1] - v) + 2.0 * (phi[1] - phi[2]);
        assert!(row.abs() < 1e-9, "{row}");
        // The source: κ̄² V v / (4π l_B) below the slab, 0 elsewhere.
        let grid = Grid::centered([0.0; 3], [8.0, 8.0, 40.0], [5, 5, 21]).unwrap();
        let kap = vec![2.0; grid.len()];
        let q = m.potential_source(&grid, &kap, v, 560.0);
        let per = 2.0 * grid.cell_volume() * v / (4.0 * std::f64::consts::PI * 560.0);
        for (c, &x) in q.iter().enumerate() {
            let below = grid.origin[2] + grid.ijk(c).2 as f64 * grid.spacing[2] < -15.0;
            assert_eq!(x, if below { per } else { 0.0 });
        }
    }
}
