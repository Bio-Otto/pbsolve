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
    sasa_weighted(atoms, probe, n_points, |_| 1.0)
}

/// [`sasa`] with every exposed point's share of the area weighted by
/// `weight(z)` of the point's z (a membrane's [`surface_switch`]).
/// A weight of 1 everywhere gives [`sasa`] bit for bit.
///
/// [`surface_switch`]: crate::membrane::surface_switch
pub fn sasa_weighted(atoms: &[Atom], probe: f64, n_points: usize, weight: impl Fn(f64) -> f64 + Sync) -> Vec<f64> {
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
            let mut exposed = 0.0;
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
                    exposed += weight(p[2]);
                }
            }
            4.0 * PI * ri * ri * exposed / unit.len() as f64
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

    #[test]
    fn the_membrane_switch_is_smooth_and_symmetric() {
        use crate::membrane::surface_switch;
        let h = |z: f64| surface_switch(z, -15.0, 30.0, 2.5);
        assert_eq!(
            (h(0.0), h(-12.5), h(12.5), h(17.5), h(-40.0)),
            (0.0, 0.0, 0.0, 1.0, 1.0)
        );
        assert_eq!((h(15.0), h(-15.0)), (0.5, 0.5));
        for u in [0.3, 1.0, 2.0, 2.4] {
            assert!((h(15.0 + u) + h(15.0 - u) - 1.0).abs() < 1e-15);
            assert!((h(-15.0 - u) - h(15.0 + u)).abs() < 1e-15);
        }
        // Slope 3/(4w) on the surface, 0 at both ends of the switch.
        let d = |z: f64| (h(z + 1e-6) - h(z - 1e-6)) / 2e-6;
        assert!((d(15.0) - 0.3).abs() < 1e-8 && d(17.5 - 1e-5).abs() < 1e-4 && d(12.5 + 1e-5).abs() < 1e-4);
        // Width 0: a step, 0 on the surfaces.
        let step = |z: f64| surface_switch(z, -15.0, 30.0, 0.0);
        assert_eq!(
            (step(14.999), step(15.0), step(15.001), step(-15.001)),
            (0.0, 0.0, 1.0, 1.0)
        );
    }

    #[test]
    fn the_weighted_area_integrates_the_switch_over_the_sphere() {
        use crate::membrane::surface_switch;
        let atom = |position: [f64; 3], radius: f64| Atom {
            position,
            charge: 0.0,
            radius,
        };
        // Weight 1 is the plain area bit for bit.
        let atoms: Vec<Atom> = (0..12)
            .map(|i| {
                atom(
                    [(i as f64 * 1.3).sin() * 4.0, i as f64 * 0.9, (i as f64).cos() * 3.0],
                    1.2 + 0.1 * i as f64,
                )
            })
            .collect();
        assert_eq!(sasa_weighted(&atoms, 1.4, 500, |_| 1.0), sasa(&atoms, 1.4, 500));
        // One sphere across the top surface: dA = 2πR dz, and the lattice's
        // points are evenly spaced in z, so the area is the midpoint rule of
        // 2πR ∫ H dz (exact with a step up to one point's share).
        let r = 3.0;
        for (z0, width) in [(14.0, 2.5), (16.2, 2.5), (14.0, 0.3), (15.0, 0.0)] {
            let a = sasa_weighted(&[atom([0.3, 0.1, z0], r - 1.4)], 1.4, 960, |z| {
                surface_switch(z, -15.0, 30.0, width)
            })[0];
            let n = 200_000;
            let dz = 2.0 * r / n as f64;
            let exact: f64 = (0..n)
                .map(|k| 2.0 * PI * r * dz * surface_switch(z0 - r + (k as f64 + 0.5) * dz, -15.0, 30.0, width))
                .sum();
            let tol = if width < 1.0 {
                4.0 * PI * r * r / 960.0
            } else {
                1e-5 * exact
            };
            assert!((a - exact).abs() <= tol, "z0 {z0} w {width}: {a} vs {exact}");
        }
        // Deep in the slab nothing is exposed; far above all of it.
        let mid = sasa_weighted(&[atom([0.0; 3], 1.6)], 1.4, 960, |z| {
            surface_switch(z, -15.0, 30.0, 2.5)
        });
        let far = sasa_weighted(&[atom([0.0, 0.0, 30.0], 1.6)], 1.4, 960, |z| {
            surface_switch(z, -15.0, 30.0, 2.5)
        });
        assert_eq!(mid[0], 0.0);
        assert_eq!(far[0], 4.0 * PI * 9.0);
    }
}
