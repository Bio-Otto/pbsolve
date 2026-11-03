// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vpmg.c), Copyright (c) 2010-2022 Battelle
// Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! Dirichlet boundary values. The returned vector has full grid length with
//! values on boundary nodes and zeros in the interior ("g_ext").
//!
//! Both analytic conditions follow APBS 3.x exactly (`vpmg.c`):
//!
//! * `mdh` (`bcflnew`): Σ_i (l_B0 q_i / ε_w) e^{−κ(d−a_i)} / ((1+κ a_i) d),
//!   with `a_i` the atom radius (no ion radius added);
//! * `sdh` (`bcfl_sdh` + `multipolebc`): charge, dipole and traceless
//!   quadrupole of the solute about its coordinate-box centre, each with the
//!   exact exterior solution for a sphere (radius = max |r_i − c| + a_i,
//!   dielectric ε_p) in a Debye–Hückel medium (ε_w, κ).

use crate::error::{invalid, Result};
use crate::grid::Grid;
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

/// Debye–Hückel potential (kT/e) of charge z with exclusion radius a at distance d (APBS `bcfl1sp`).
#[inline]
fn dh(z: f64, a: f64, d: f64, m: &BoundaryMedium) -> f64 {
    if m.kappa > 0.0 {
        m.lb0 * z / m.eps * (-(m.kappa) * (d - a)).exp() / ((1.0 + m.kappa * a) * d)
    } else {
        m.lb0 * z / (m.eps * d)
    }
}

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

pub fn analytic_boundary(grid: &Grid, atoms: &[Atom], bc: BoundaryCondition, m: &BoundaryMedium) -> Vec<f64> {
    let mut g = vec![0.0; grid.len()];
    let bidx = grid.boundary_indices();
    let values: Vec<f64> = match bc {
        BoundaryCondition::Zero => return g,
        BoundaryCondition::Mdh => bidx
            .par_iter()
            .map(|&c| {
                let (i, j, k) = grid.ijk(c);
                let p = grid.position(i, j, k);
                atoms
                    .iter()
                    .filter(|a| a.charge != 0.0)
                    .map(|a| dh(a.charge, a.radius, dist2(p, a.position).sqrt(), m))
                    .sum()
            })
            .collect(),
        BoundaryCondition::Sdh => {
            let mp = Multipoles::of(atoms);
            bidx.par_iter()
                .map(|&c| {
                    let (i, j, k) = grid.ijk(c);
                    mp.potential(grid.position(i, j, k), m)
                })
                .collect()
        }
    };
    for (c, v) in bidx.into_iter().zip(values) {
        g[c] = v;
    }
    g
}

/// Boundary of a fine grid taken from a coarse solution (focusing).
pub fn focus_boundary(fine: &Grid, coarse: &Grid, u_coarse: &[f64]) -> Result<Vec<f64>> {
    let mut g = vec![0.0; fine.len()];
    for c in fine.boundary_indices() {
        let (i, j, k) = fine.ijk(c);
        let p = fine.position(i, j, k);
        match coarse.interpolate(u_coarse, p) {
            Some(v) => g[c] = v,
            None => return invalid("fine grid is not contained in the coarse grid"),
        }
    }
    Ok(g)
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
