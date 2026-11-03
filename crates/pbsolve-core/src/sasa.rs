//! Solvent-accessible surface area (Shrake–Rupley) for the non-polar term.
//! (The dielectric surface uses APBS' own sphere lattice, see `surface.rs`.)

use crate::molecule::{dist2, Atom};
use crate::spatial::CellList;
use rayon::prelude::*;
use std::f64::consts::PI;

/// `n` quasi-uniform points on the unit sphere (golden-spiral lattice).
pub fn unit_sphere_points(n: usize) -> Vec<[f64; 3]> {
    let n = n.max(1);
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * i as f64;
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect()
}

/// Per-atom SASA in Å² (Shrake–Rupley, `n_points` per atom).
/// Atoms with zero radius get zero area and do not occlude others.
pub fn sasa(atoms: &[Atom], probe: f64, n_points: usize) -> Vec<f64> {
    let unit = unit_sphere_points(n_points);
    let pos: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
    let max_r = atoms.iter().map(|a| a.radius).fold(0.0, f64::max) + probe;
    let cells = CellList::new(&pos, max_r.max(1.0));
    atoms
        .par_iter()
        .enumerate()
        .map(|(i, ai)| {
            if ai.radius <= 0.0 {
                return 0.0;
            }
            let ri = ai.radius + probe;
            let mut exposed = 0usize;
            for u in &unit {
                let p = [
                    ai.position[0] + ri * u[0],
                    ai.position[1] + ri * u[1],
                    ai.position[2] + ri * u[2],
                ];
                let buried = cells.any_near(p, max_r, |j| {
                    if j == i || atoms[j].radius <= 0.0 {
                        return false;
                    }
                    let rj = atoms[j].radius + probe;
                    dist2(p, atoms[j].position) < rj * rj
                });
                if !buried {
                    exposed += 1;
                }
            }
            4.0 * PI * ri * ri * exposed as f64 / unit.len() as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_sphere_area() {
        let atoms = [Atom {
            position: [0.0; 3],
            charge: 0.0,
            radius: 1.6,
        }];
        let a = sasa(&atoms, 1.4, 960)[0];
        let exact = 4.0 * PI * 3.0 * 3.0;
        assert!((a - exact).abs() / exact < 1e-12);
    }

    #[test]
    fn two_overlapping_spheres_lose_area() {
        let atoms = [
            Atom {
                position: [0.0; 3],
                charge: 0.0,
                radius: 1.5,
            },
            Atom {
                position: [2.0, 0.0, 0.0],
                charge: 0.0,
                radius: 1.5,
            },
        ];
        let a = sasa(&atoms, 1.4, 2000);
        // Each SAS sphere (R = 2.9) loses a cap of height h = R - d/2 = 1.9.
        let r: f64 = 2.9;
        let exact = 4.0 * PI * r * r - 2.0 * PI * r * 1.9;
        assert!((a[0] - exact).abs() / exact < 0.01, "{} vs {}", a[0], exact);
    }
}
