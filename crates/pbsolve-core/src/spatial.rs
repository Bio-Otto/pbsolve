//! Uniform cell list (CSR layout) for fixed-radius neighbour queries.

pub struct CellList {
    origin: [f64; 3],
    cell: f64,
    dims: [usize; 3],
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl CellList {
    /// Bins `points` into cubic cells of edge `cell` (Å).
    pub fn new(points: &[[f64; 3]], cell: f64) -> Self {
        let cell = cell.max(1e-3);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        if points.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let dims = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / cell).floor() as usize + 1);
        let ncell = dims[0] * dims[1] * dims[2];
        let cell_of = |p: &[f64; 3]| -> usize {
            let c = [0, 1, 2].map(|a| (((p[a] - lo[a]) / cell) as usize).min(dims[a] - 1));
            c[0] + dims[0] * (c[1] + dims[1] * c[2])
        };
        let mut counts = vec![0u32; ncell + 1];
        for p in points {
            counts[cell_of(p) + 1] += 1;
        }
        for c in 0..ncell {
            counts[c + 1] += counts[c];
        }
        let mut fill = counts.clone();
        let mut items = vec![0u32; points.len()];
        for (n, p) in points.iter().enumerate() {
            let c = cell_of(p);
            items[fill[c] as usize] = n as u32;
            fill[c] += 1;
        }
        Self {
            origin: lo,
            cell,
            dims,
            starts: counts,
            items,
        }
    }

    /// Calls `f(index)` for every point in cells overlapping the sphere
    /// (p, radius). Stops early and returns `true` as soon as `f` returns `true`.
    /// Candidates are a superset of the true neighbours; `f` must test distance.
    #[inline]
    pub fn any_near<F: FnMut(usize) -> bool>(&self, p: [f64; 3], radius: f64, mut f: F) -> bool {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for a in 0..3 {
            let l = ((p[a] - radius - self.origin[a]) / self.cell).floor();
            let h = ((p[a] + radius - self.origin[a]) / self.cell).floor();
            if h < 0.0 || l > (self.dims[a] - 1) as f64 {
                return false;
            }
            lo[a] = l.max(0.0) as usize;
            hi[a] = (h as usize).min(self.dims[a] - 1);
        }
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    let c = i + self.dims[0] * (j + self.dims[1] * k);
                    let (s, e) = (self.starts[c] as usize, self.starts[c + 1] as usize);
                    for &n in &self.items[s..e] {
                        if f(n as usize) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}
