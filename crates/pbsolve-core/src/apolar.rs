// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vacc.c: Vacc_SASA, Vacc_atomSurf, Vacc_totalSAV,
// Vacc_wcaEnergyAtom; vclist.c: Vclist_setupGrid; routines.c: initAPOL), Copyright (c) 2010-2022
// Battelle Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! The non-polar solvation terms of APBS `apolar` [Wagoner & Baker 2006]:
//!
//! G_np = γ A + p V + ρ̄ Σ_i ∫ u_i^att(y) θ(y) dy.
//!
//! * A, the solvent-accessible surface area: each atom's sphere of radius
//!   r + srad, sampled by APBS' reference sphere, counts the fraction of its
//!   points outside every other atom's sphere (`Vacc_atomSurf`).
//! * V, the solvent-accessible volume (the union of those spheres): the
//!   trapezoidal rule with spacing `grid` over APBS' cell-list box
//!   (`Vacc_totalSAV`).
//! * The dispersion integral: per atom, the attractive part of its
//!   Lennard-Jones interaction with water oxygen in the Weeks–Chandler–
//!   Andersen split, ε_iw [(σ/r)¹² − 2(σ/r)⁶] beyond σ = R_i + R_w and −ε_iw
//!   inside (ε_iw = √(ε_i ε_w)), times the solvent density ρ̄, integrated
//!   over the solvent-accessible points within 14 Å by the trapezoidal rule
//!   on a lattice anchored at integer coordinates (`Vacc_wcaEnergyAtom`).
//!
//! The loops follow APBS: coordinates accumulated in steps of `grid` as APBS
//! accumulates them, the same edge weights and the same order of the sums,
//! so the terms equal APBS'. The cell list of an `apolar` calculation has
//! the radius srad + 2·dpos, which sets the reference sphere's size and the
//! volume's box.

use crate::error::{invalid, Result};
use crate::molecule::Atom;
use crate::surface::{apbs_reference_sphere, reference_sphere_points, sas_point, Accessibility};
use rayon::prelude::*;
use std::f64::consts::PI;

/// APBS `VSMALL`.
const VSMALL: f64 = 1.0e-9;
/// APBS `VCLIST_INFLATE`: the cell-list box reaches 1.42 (r_max + radius)
/// beyond the atom centres.
const VCLIST_INFLATE: f64 = 1.42;
/// Half-width (Å) of the dispersion integral's box and its cutoff radius.
const WCA_REACH: i32 = 14;

fn check(atoms: &[Atom], srad: f64, dpos: f64) -> Result<()> {
    if atoms.is_empty() {
        return invalid("no atoms");
    }
    if !(srad >= 0.0 && dpos >= 0.0) {
        return invalid("srad and dpos must be non-negative");
    }
    Ok(())
}

fn check_grid(grid: [f64; 3]) -> Result<()> {
    if grid.iter().any(|&h| h.is_nan() || h <= 0.0 || h.is_infinite()) {
        return invalid("grid spacings must be positive");
    }
    Ok(())
}

/// Solvent-accessible surface area of every atom (Å²; APBS
/// `Vacc_atomSASA`). Atoms of radius 0 have none but still bury points of
/// the others within `srad`.
pub fn apbs_sasa(atoms: &[Atom], srad: f64, sdens: f64, dpos: f64) -> Result<Vec<f64>> {
    check(atoms, srad, dpos)?;
    if sdens.is_nan() || sdens <= 0.0 {
        return invalid("sdens must be positive");
    }
    let reference = apbs_reference_sphere(reference_sphere_points(atoms, srad + 2.0 * dpos, sdens));
    let acc = Accessibility::new(atoms, srad);
    let n_ref = reference.len() as f64;
    Ok(atoms
        .par_iter()
        .enumerate()
        .map(|(i, a)| {
            if a.radius < VSMALL {
                return 0.0;
            }
            let rad = a.radius + srad;
            let n = reference
                .iter()
                .filter(|u| acc.accessible(sas_point(a, srad, u), Some(i)))
                .count();
            4.0 * PI * rad * rad * (n as f64) / n_ref
        })
        .collect())
}

/// The coordinates APBS visits from `lo` to `hi` in steps of `h` (each one
/// the previous plus `h`), with their trapezoidal weights.
fn axis(lo: f64, hi: f64, h: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut x = lo;
    while x <= hi {
        let w = if (x - lo).abs() < VSMALL || (x - hi).abs() < VSMALL {
            0.5
        } else {
            1.0
        };
        out.push((x, w));
        x += h;
    }
    out
}

/// Solvent-accessible volume (Å³; APBS `Vacc_totalSAV`).
pub fn apbs_sav(atoms: &[Atom], srad: f64, dpos: f64, grid: [f64; 3]) -> Result<f64> {
    check(atoms, srad, dpos)?;
    check_grid(grid)?;
    // APBS' cell-list box (`Vclist_setupGrid`, CLIST_AUTO_DOMAIN).
    let r_max = atoms.iter().map(|a| a.radius).fold(-1.0, f64::max);
    let pad = VCLIST_INFLATE * (r_max + (srad + 2.0 * dpos));
    let mut lo = [f64::MAX; 3];
    let mut hi = [-f64::MAX; 3];
    for a in atoms {
        for k in 0..3 {
            lo[k] = lo[k].min(a.position[k]);
            hi[k] = hi[k].max(a.position[k]);
        }
    }
    let [xs, ys, zs] = [0, 1, 2].map(|k| axis(lo[k] - pad, hi[k] + pad, grid[k]));
    let acc = Accessibility::new(atoms, srad);
    // The points' accessibility in parallel, the sum in APBS' order.
    let mut sav = 0.0;
    for planes in xs.chunks(16) {
        let buried: Vec<Vec<bool>> = planes
            .par_iter()
            .map(|&(x, _)| {
                ys.iter()
                    .flat_map(|&(y, _)| zs.iter().map(move |&(z, _)| [x, y, z]))
                    .map(|p| !acc.accessible(p, None))
                    .collect()
            })
            .collect();
        for (&(_, wx), plane) in planes.iter().zip(&buried) {
            let mut n = 0;
            for &(_, wy) in &ys {
                for &(_, wz) in &zs {
                    let w = wx * wy * wz;
                    sav += w * (if plane[n] { 1.0 } else { 0.0 });
                    n += 1;
                }
            }
        }
    }
    Ok(sav * (grid[0] * grid[1] * grid[2]))
}

/// The water oxygen's Lennard-Jones parameters for [`wca_energies`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Water {
    /// Radius R_w (Å): r_min of an atom–water pair is R_i + R_w.
    pub radius: f64,
    /// Well depth ε_w, in the unit of the atoms' ε.
    pub epsilon: f64,
}

/// The dispersion energy of every atom (APBS `Vacc_wcaEnergyAtom`), in the
/// unit of `epsilons` and `water.epsilon`; `bconc` is the solvent number
/// density ρ̄ (Å⁻³). The atoms' `radius` is their Lennard-Jones radius R_i.
pub fn wca_energies(
    atoms: &[Atom],
    epsilons: &[f64],
    water: Water,
    bconc: f64,
    srad: f64,
    grid: [f64; 3],
) -> Result<Vec<f64>> {
    check(atoms, srad, 0.0)?;
    check_grid(grid)?;
    if epsilons.len() != atoms.len() {
        return invalid(format!("{} epsilons for {} atoms", epsilons.len(), atoms.len()));
    }
    let negative = |x: f64| x.is_nan() || x < 0.0;
    if epsilons.iter().chain([&water.epsilon]).any(|&e| negative(e)) || negative(water.radius) {
        return invalid("Lennard-Jones well depths and radii must be non-negative");
    }
    if bconc.abs() < VSMALL {
        return Ok(vec![0.0; atoms.len()]);
    }
    let acc = Accessibility::new(atoms, srad);
    let reach = f64::from(WCA_REACH);
    Ok(atoms
        .par_iter()
        .zip(epsilons)
        .map(|(a, &eps)| {
            let pos = a.position;
            let sigma = a.radius + water.radius;
            let epsilon = (eps * water.epsilon).sqrt();
            let sigma6 = sigma.powf(6.0);
            let sigma12 = sigma.powf(12.0);
            // APBS converts pos ∓ 14 to int, truncating towards zero.
            let [xs, ys, zs] = [0, 1, 2].map(|k| {
                let lo = (pos[k] - reach) as i32;
                let hi = (pos[k] + reach) as i32;
                axis(f64::from(lo), f64::from(hi), grid[k])
            });
            let mut energy = 0.0;
            for &(x, wx) in &xs {
                for &(y, wy) in &ys {
                    for &(z, wz) in &zs {
                        let w = wx * wy * wz;
                        let x2 = (x - pos[0]) * (x - pos[0]);
                        let y2 = (y - pos[1]) * (y - pos[1]);
                        let z2 = (z - pos[2]) * (z - pos[2]);
                        let r = (x2 + y2 + z2).sqrt();
                        // APBS tests the accessibility first; beyond 14 Å the term is 0 either way.
                        let eni = if r <= reach && acc.accessible([x, y, z], None) {
                            let chi = 1.0;
                            if r >= sigma {
                                chi * bconc * epsilon * (-2.0 * sigma6 / r.powf(6.0) + sigma12 / r.powf(12.0))
                            } else {
                                -epsilon * chi * bconc
                            }
                        } else {
                            0.0
                        };
                        energy += eni * w;
                    }
                }
            }
            energy * (grid[0] * grid[1] * grid[2])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(position: [f64; 3], radius: f64) -> Atom {
        Atom {
            position,
            charge: 0.0,
            radius,
        }
    }

    #[test]
    fn one_sphere_has_its_area_and_volume() {
        let a = [atom([0.13, -0.21, 0.07], 1.7)];
        let r: f64 = 1.7 + 1.4;
        let area = apbs_sasa(&a, 1.4, 10.0, 0.2).unwrap()[0];
        assert!((area - 4.0 * PI * r * r).abs() < 1e-12, "{area}");
        // Trapezoidal rule on a 0.1 Å grid: within 0.3 % of 4/3 π r³.
        let v = apbs_sav(&a, 1.4, 0.2, [0.1; 3]).unwrap();
        let exact = 4.0 / 3.0 * PI * r.powi(3);
        assert!((v / exact - 1.0).abs() < 3e-3, "{v} vs {exact}");
    }

    #[test]
    fn two_spheres_share_their_lens() {
        // Two spheres of radius 2 at distance 3: union 2·(4/3 π 8) − lens, and
        // each loses the cap of height 0.5 from its area.
        let a = [atom([0.0, 0.0, 0.0], 2.0), atom([3.0, 0.0, 0.0], 2.0)];
        let lens = PI * (4.0 * 2.0 + 3.0) * (2.0 * 2.0 - 3.0f64).powi(2) / 12.0;
        let union = 2.0 * 4.0 / 3.0 * PI * 8.0 - lens;
        let v = apbs_sav(&a, 0.0, 0.0, [0.05; 3]).unwrap();
        assert!((v / union - 1.0).abs() < 2e-3, "{v} vs {union}");
        let area = apbs_sasa(&a, 0.0, 200.0, 0.0).unwrap();
        let exact = 4.0 * PI * 4.0 - 2.0 * PI * 2.0 * 0.5;
        for s in area {
            assert!((s / exact - 1.0).abs() < 5e-3, "{s} vs {exact}");
        }
    }

    #[test]
    fn dispersion_of_one_atom_is_the_radial_integral() {
        // One atom, no other: θ = 1 beyond r + srad, so the energy is
        // ρ̄ ∫ u 4π r² dr from r + srad to 14 Å (σ ≥ r + srad here).
        let (r, srad, rw, eps, epsw, rho) = (1.9, 0.65, 1.7683, 0.4577, 0.6364, 0.033428);
        let a = [atom([0.31, -0.17, 0.42], r)];
        let e = wca_energies(
            &a,
            &[eps],
            Water {
                radius: rw,
                epsilon: epsw,
            },
            rho,
            srad,
            [0.1; 3],
        )
        .unwrap()[0];
        let (s, ew) = (r + rw, (eps * epsw).sqrt());
        // ∫ 4π t² ε[(s/t)¹² − 2(s/t)⁶] dt from s to 14, and −ε 4π t²/3 from r + srad to s.
        let shell = |t: f64| 4.0 * PI * ew * (-(s.powi(12)) / (9.0 * t.powi(9)) + 2.0 * s.powi(6) / (3.0 * t.powi(3)));
        let inner = -ew * 4.0 / 3.0 * PI * (s.powi(3) - (r + srad).powi(3));
        let exact = rho * (shell(14.0) - shell(s) + inner);
        assert!((e / exact - 1.0).abs() < 3e-3, "{e} vs {exact}");
        // No solvent, no dispersion.
        assert_eq!(
            wca_energies(
                &a,
                &[eps],
                Water {
                    radius: rw,
                    epsilon: epsw
                },
                0.0,
                srad,
                [0.5; 3]
            )
            .unwrap(),
            [0.0]
        );
    }
}
