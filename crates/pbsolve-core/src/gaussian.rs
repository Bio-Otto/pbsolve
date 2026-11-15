// SPDX-License-Identifier: MIT

//! The Gaussian smooth dielectric (`surface = gauss`, not an APBS option)
//! [Grant, Pickup & Nicholls 2001; Li et al. 2013]: no surface, but a
//! molecular density built from atomic Gaussians.
//!
//! An atom of radius R > 0 at distance d has the density g = exp(−d²/σ²R²)
//! (σ = `gauss_sigma`); the molecule's density is ρ = 1 − Π (1 − g) and
//!
//! ε = ρ·pdie + (1 − ρ)·sdie = pdie + (sdie − pdie)·H,   H = Π S,  S = 1 − g,
//!
//! the form of the spline surfaces ([`crate::spline`]) with a switch S that
//! is 0 at an atom's centre and tends to 1. ε is taken at the face
//! midpoints, as there.
//!
//! The mobile ions pay the Born energy of leaving the bulk solvent for the
//! local dielectric [Jia et al. 2017], z² e²/(8π ε0 r_ion)·(1/ε − 1/sdie) ≥ 0,
//! so their accessibility at a node is
//!
//! K = exp(−z² l_B0/(2 r_ion) · (1/ε − 1/sdie)),
//!
//! and κ̄² = κ̄²_bulk·K (one valence and radius for all species, checked by
//! [`PbParams::validate`]). The penalty is bounded (z² l_B0/(2 r_ion)·(1/pdie −
//! 1/sdie), 138 kT for pdie = 1 and r_ion = 2 Å) while the potential next to a
//! point charge is not, so the Boltzmann density of `npbe` diverges at the
//! charges: `gauss` takes `lpbe` and `smpbe` (whose density is bounded).
//!
//! **Reference state.** ΔG_polar needs a reference in which every charge
//! sees the same ε around it as in the solvated state; otherwise the grid
//! self-energies of the charges do not cancel. A homogeneous `pdie` does not
//! do this here: ε rises from `pdie` within a few tenths of an Å of every
//! nucleus, and ΔG_polar of 1YCR goes from −39 000 to −61 000 kcal/mol
//! between h = 0.5 and 0.25 Å (docs/VALIDATION.md). As Li et al. (2014), the
//! reference state keeps the Gaussian ε wherever it is at most
//! `gauss_contour` (20) and puts `pdie` beyond, without ions: the molecule
//! keeps its internal dielectric, only the solvent around it is removed.
//! (Li et al. put vacuum beyond the contour; with pdie = 1 this is the same.)
//! A charge of radius 0 outside the contour would see different ε in the two
//! states; [`crate::pb`] rejects it.
//!
//! Atoms of radius 0 carry no density. Each Gaussian is cut off where
//! g ≤ 1e-12 (d > 5.26 σR), which leaves ε within 1e-10 of the full sum.
//! g is evaluated as the product of its three axis factors, precomputed
//! per atom (S = 1 − g then carries an absolute error of ~1e-16).

use crate::grid::Grid;
use crate::molecule::Atom;
use crate::params::PbParams;
use crate::spline::SmoothMaps;
use crate::units::bjerrum_length_vacuum;
use rayon::prelude::*;

/// A Gaussian counts where it exceeds this.
const CUTOFF: f64 = 1.0e-12;
/// −ln(CUTOFF): the cutoff distance is σR times its square root.
const CUTOFF_EXPONENT: f64 = 27.631_021_115_928_547;

/// The Born penalty z² l_B0/(2 r_ion) of the ions (kT per unit of 1/ε); 0
/// without ions.
pub(crate) fn ion_penalty(params: &PbParams) -> f64 {
    params.ions.iter().find(|i| i.concentration > 0.0).map_or(0.0, |i| {
        i.charge * i.charge * bjerrum_length_vacuum(params.temperature) / (2.0 * i.radius)
    })
}

/// One atom as the maps see it: the node range whose nodes or faces it
/// reaches and, per axis and node of that range, its Gaussian factor at the
/// node and at the node's face.
struct Placed {
    lo: [usize; 3],
    hi: [usize; 3],
    factors: [Vec<[f64; 2]>; 3],
}

/// The dielectric (faces) and ion-accessibility (nodes) maps of `gauss` on
/// `grid`. Atoms outside the grid still shape it within their cutoff.
pub fn gaussian_maps(grid: &Grid, atoms: &[Atom], params: &PbParams) -> SmoothMaps {
    let [nx, ny, _] = grid.dims;
    let h = grid.spacing;
    let origin = grid.origin;
    let placed: Vec<Placed> = atoms
        .iter()
        .filter(|a| a.radius > 0.0)
        .filter_map(|a| {
            let s = params.gauss_sigma * a.radius;
            let (cut, inv) = (s * CUTOFF_EXPONENT.sqrt(), 1.0 / (s * s));
            let (mut lo, mut hi) = ([0; 3], [0; 3]);
            let mut factors: [Vec<[f64; 2]>; 3] = Default::default();
            for ax in 0..3 {
                // Node i sits at i·h, its face at (i + ½)·h.
                let pos = a.position[ax] - origin[ax];
                let first = ((pos - cut) / h[ax] - 0.5).ceil();
                let last = ((pos + cut) / h[ax]).floor();
                let top = (grid.dims[ax] - 1) as f64;
                if last < 0.0 || first > top {
                    return None;
                }
                lo[ax] = first.max(0.0) as usize;
                hi[ax] = last.min(top) as usize;
                factors[ax] = (lo[ax]..=hi[ax])
                    .map(|i| [i as f64, i as f64 + 0.5].map(|t| (-sq(pos - t * h[ax]) * inv).exp()))
                    .collect();
            }
            Some(Placed { lo, hi, factors })
        })
        .collect();
    let n = grid.len();
    let plane = nx * ny;
    let (mut node, mut ex, mut ey, mut ez) = (vec![1.0; n], vec![1.0; n], vec![1.0; n], vec![1.0; n]);
    node.par_chunks_mut(plane)
        .zip(ex.par_chunks_mut(plane))
        .zip(ey.par_chunks_mut(plane))
        .zip(ez.par_chunks_mut(plane))
        .enumerate()
        .for_each(|(k, (((hn, hx), hy), hz))| {
            // Per point the atoms come in order: the maps do not depend on the threads.
            for p in placed.iter().filter(|p| p.lo[2] <= k && k <= p.hi[2]) {
                let [gz, gzf] = p.factors[2][k - p.lo[2]];
                for j in p.lo[1]..=p.hi[1] {
                    let [gy, gyf] = p.factors[1][j - p.lo[1]];
                    let (gyz, gfz, gyf_z) = (gy * gz, gy * gzf, gyf * gz);
                    if gyz.max(gfz).max(gyf_z) <= CUTOFF {
                        continue;
                    }
                    for (i, &[gx, gxf]) in (p.lo[0]..=p.hi[0]).zip(&p.factors[0]) {
                        let c = i + nx * j;
                        for (m, g) in [
                            (&mut hn[c], gx * gyz),
                            (&mut hx[c], gxf * gyz),
                            (&mut hy[c], gx * gyf_z),
                            (&mut hz[c], gx * gfz),
                        ] {
                            if g > CUTOFF {
                                *m *= 1.0 - g;
                            }
                        }
                    }
                }
            }
        });
    let (pdie, sdie) = (params.pdie, params.sdie);
    let fill = |v: &mut Vec<f64>| v.par_iter_mut().for_each(|e| *e = pdie + (sdie - pdie) * *e);
    fill(&mut ex);
    fill(&mut ey);
    fill(&mut ez);
    let penalty = ion_penalty(params);
    node.par_iter_mut().for_each(|k| {
        let eps = pdie + (sdie - pdie) * *k;
        *k = (-penalty * (1.0 / eps - 1.0 / sdie)).exp();
    });
    SmoothMaps {
        eps: [ex, ey, ez],
        accessibility: node,
    }
}

/// The reference state's face dielectric [Li et al. 2014]: the Gaussian ε
/// where it is at most `gauss_contour`, `pdie` beyond.
pub fn contour_faces(grid: &Grid, atoms: &[Atom], params: &PbParams) -> [Vec<f64>; 3] {
    let (contour, pdie) = (params.gauss_contour, params.pdie);
    gaussian_maps(grid, atoms, params).eps.map(|mut v| {
        v.par_iter_mut().for_each(|e| {
            if *e > contour {
                *e = pdie;
            }
        });
        v
    })
}

/// The Gaussian ε at one point (all atoms, with the maps' cutoff).
pub fn gaussian_eps_at(point: [f64; 3], atoms: &[Atom], params: &PbParams) -> f64 {
    let solvent: f64 = atoms
        .iter()
        .filter(|a| a.radius > 0.0)
        .map(|a| {
            let s = params.gauss_sigma * a.radius;
            let g = (0..3)
                .map(|k| (-sq(point[k] - a.position[k]) / (s * s)).exp())
                .product::<f64>();
            if g > CUTOFF {
                1.0 - g
            } else {
                1.0
            }
        })
        .product();
    params.pdie + (params.sdie - params.pdie) * solvent
}

#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autogrid::GridSpec;
    use crate::cpu::CpuBackend;
    use crate::params::{BoundaryCondition, CoulombMethod, IonSpecies, SurfaceMethod};
    use crate::pb::polar_solvation;

    fn atoms() -> Vec<Atom> {
        [
            ([0.13, -0.21, 0.07], 0.6, 1.8),
            ([1.71, 0.52, -0.33], -0.4, 1.5),
            ([0.62, 0.11, 0.48], 0.35, 0.0),
            // Outside the grid, within reach of it.
            ([6.4, -0.3, 0.2], -0.25, 1.7),
        ]
        .map(|(position, charge, radius)| Atom {
            position,
            charge,
            radius,
        })
        .to_vec()
    }

    fn params() -> PbParams {
        PbParams {
            pdie: 2.0,
            surface: SurfaceMethod::Gaussian,
            gauss_sigma: 0.8,
            ..PbParams::default()
        }
        .with_salt(0.15, 1.6)
    }

    /// ε and K at a point from the formulas, every atom, no cutoff.
    fn brute(point: [f64; 3], atoms: &[Atom], p: &PbParams) -> (f64, f64) {
        let solvent: f64 = atoms
            .iter()
            .filter(|a| a.radius > 0.0)
            .map(|a| {
                let d2: f64 = (0..3).map(|k| (point[k] - a.position[k]).powi(2)).sum();
                -(-d2 / (p.gauss_sigma * a.radius).powi(2)).exp_m1()
            })
            .product();
        let eps = p.pdie + (p.sdie - p.pdie) * solvent;
        let born = bjerrum_length_vacuum(p.temperature) / (2.0 * 1.6);
        (eps, (-born * (1.0 / eps - 1.0 / p.sdie)).exp())
    }

    #[test]
    fn maps_are_the_gaussian_dielectric_and_the_born_accessibility() {
        let p = params();
        let atoms = atoms();
        let grid = Grid::centered([0.3, -0.1, 0.05], [9.0, 8.4, 7.2], [31, 29, 25]).unwrap();
        let maps = gaussian_maps(&grid, &atoms, &p);
        let h = grid.spacing;
        let (mut deps, mut dk, mut partial) = (0.0f64, 0.0f64, 0);
        for k in 0..grid.dims[2] {
            for j in 0..grid.dims[1] {
                for i in 0..grid.dims[0] {
                    let c = grid.index(i, j, k);
                    let x = grid.position(i, j, k);
                    let (_, kn) = brute(x, &atoms, &p);
                    dk = dk.max((maps.accessibility[c] - kn).abs());
                    for ax in 0..3 {
                        let mut f = x;
                        f[ax] += 0.5 * h[ax];
                        let (e, _) = brute(f, &atoms, &p);
                        deps = deps.max((maps.eps[ax][c] - e).abs());
                        partial += usize::from(e > p.pdie + 1.0 && e < p.sdie - 1.0);
                    }
                }
            }
        }
        // The cutoff (1e-12 per atom, times sdie − pdie) and rounding.
        assert!(deps < 1e-9 && dk < 1e-9, "max |Δε| {deps:.2e}, |ΔK| {dk:.2e}");
        assert!(partial > 1000, "{partial} graded faces");
        // The reference keeps ε up to the contour and puts pdie beyond.
        let contour = contour_faces(&grid, &atoms, &p);
        for (kept, eps) in contour.iter().zip(&maps.eps) {
            for (r, e) in kept.iter().zip(eps) {
                assert_eq!(*r, if *e > p.gauss_contour { p.pdie } else { *e });
            }
        }
        // The point evaluation agrees with the node map.
        let c = grid.index(17, 13, 11);
        let (e, _) = brute(grid.position(17, 13, 11), &atoms, &p);
        assert!((gaussian_eps_at(grid.position(17, 13, 11), &atoms, &p) - e).abs() < 1e-9);
        assert!(maps.accessibility[c] > 0.0);
    }

    #[test]
    fn parameters_are_checked() {
        let ok = params();
        assert!(ok.validate().is_ok());
        let bad = [
            PbParams {
                gauss_sigma: 0.0,
                ..ok.clone()
            },
            PbParams {
                gauss_contour: 2.0,
                ..ok.clone()
            },
            PbParams {
                coulomb: CoulombMethod::Analytic,
                ..ok.clone()
            },
            PbParams {
                ions: vec![
                    IonSpecies {
                        charge: 2.0,
                        concentration: 0.1,
                        radius: 2.0,
                    },
                    IonSpecies {
                        charge: -1.0,
                        concentration: 0.2,
                        radius: 2.0,
                    },
                ],
                ..ok.clone()
            },
            ok.clone().with_salt(0.1, 0.0),
            PbParams {
                equation: crate::params::Equation::Nonlinear,
                ..ok.clone()
            },
        ];
        for p in bad {
            assert!(p.validate().is_err(), "{p:?}");
        }
        // Other surfaces ignore the Gaussian settings.
        let mol = PbParams {
            surface: SurfaceMethod::Mol,
            gauss_contour: 0.5,
            ..PbParams::default()
        };
        assert!(mol.validate().is_ok());
    }

    #[test]
    fn a_contour_above_sdie_leaves_nothing_to_solvate() {
        // Without ions and with zero boundary values the two states are the same system.
        let p = PbParams {
            surface: SurfaceMethod::Gaussian,
            gauss_contour: 1.0e9,
            bcfl: BoundaryCondition::Zero,
            ..PbParams::default()
        };
        let spec = GridSpec::single([0.4, 0.1, 0.0], [14.0, 13.0, 12.0], [29, 27, 25]);
        let r = polar_solvation(&CpuBackend::new(), &atoms()[..2], &p, &spec).unwrap();
        assert_eq!(r.energy_kcal, 0.0);
        let normal = PbParams {
            gauss_contour: 20.0,
            ..p
        };
        assert!(
            polar_solvation(&CpuBackend::new(), &atoms()[..2], &normal, &spec)
                .unwrap()
                .energy_kcal
                < -10.0
        );
    }

    #[test]
    fn exposed_charges_of_radius_zero_are_rejected() {
        let p = PbParams {
            surface: SurfaceMethod::Gaussian,
            ..PbParams::default()
        };
        let spec = GridSpec::single([0.4, 0.1, 0.0], [16.0, 15.0, 14.0], [33, 31, 29]);
        // Atom 2 sits inside atom 0's density: accepted.
        assert!(polar_solvation(&CpuBackend::new(), &atoms()[..3], &p, &spec).is_ok());
        let mut out = atoms()[..3].to_vec();
        out[2].position = [0.62, 3.4, 0.48];
        let err = polar_solvation(&CpuBackend::new(), &out, &p, &spec)
            .unwrap_err()
            .to_string();
        assert!(err.contains("atom 2") && err.contains("gauss_contour"), "{err}");
    }
}
