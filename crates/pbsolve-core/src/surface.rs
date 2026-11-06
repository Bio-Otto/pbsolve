// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vacc.c, vpbe.c, vpmg.c), Copyright (c) 2010-2022 Battelle
// Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! Coefficient maps: dielectric on the three staggered (face-centred) meshes
//! and the ion-accessibility-weighted κ̄² on grid nodes.
//!
//! `eps_x[index(i,j,k)]` is ε at (i+½, j, k); analogously for y and z. The
//! last face along each axis lies outside the grid and is never read.
//!
//! # APBS `srfm mol` parity
//!
//! The molecular surface reproduces APBS 3.x (`Vpmg fillcoCoefMolDielNoSmooth`,
//! `Vacc`) operation for operation, so dielectric maps are identical:
//!
//! 1. every node starts as solvent (`sdie`);
//! 2. nodes within `r_i + srad` of an atom with `r_i > 1e-9` (boundary
//!    included) become solute (`pdie`) — the solvent-accessible volume;
//! 3. one *reference sphere* of `⌈4π (r_max + R_clist)² · sdens⌉` requested
//!    points (APBS' θ/φ lattice, `VaccSurf_refSphere`) is scaled onto every
//!    SAS sphere; a point is *accessible* unless it lies strictly inside
//!    `r_j + srad` of another atom j (zero-radius atoms included);
//! 4. nodes within `srad` of an accessible point (boundary included) become
//!    solvent again — the probe carves out the re-entrant region.
//!
//! `R_clist = max(srad, largest ion radius) + 0.5` is APBS' cell-list radius,
//! which sets the reference-sphere density. Node tests use APBS' `markSphere`
//! arithmetic (`(z² + y²) + x² ≤ r²` in grid-relative coordinates).
//! The ion-accessibility map marks `r_i + r_ion` spheres the same way.

use crate::backend::Backend;
use crate::cpu::CpuBackend;
use crate::error::{PbError, Result};
use crate::grid::Grid;
use crate::molecule::{dist2, Atom};
use crate::params::{PbParams, SurfaceMethod};
use crate::spatial::CellList;
use crate::units::kappa_bar_squared;
use rayon::prelude::*;
use std::f64::consts::PI;

/// APBS `VSMALL`: radii at or below this do not shape the surface.
pub const APBS_VSMALL: f64 = 1.0e-9;

/// APBS `MAX_SPLINE_WINDOW`, added to the cell-list radius in `Vpbe_ctor2`.
const APBS_SPLINE_WINDOW: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct CoefficientMaps {
    pub eps_x: Vec<f64>,
    pub eps_y: Vec<f64>,
    pub eps_z: Vec<f64>,
    /// κ̄² (Å⁻²) at nodes; zero inside the ion-exclusion region.
    pub kappa_sq: Vec<f64>,
}

impl CoefficientMaps {
    /// Uniform dielectric, no mobile ions (reference state).
    pub fn homogeneous(grid: &Grid, eps: f64) -> Self {
        let n = grid.len();
        let filled = || {
            let mut v = vec![0.0; n];
            v.par_iter_mut().for_each(|e| *e = eps);
            v
        };
        Self {
            eps_x: filled(),
            eps_y: filled(),
            eps_z: filled(),
            kappa_sq: vec![0.0; n],
        }
    }

    /// Dielectric and ion maps for the solvated state.
    pub fn solvated(grid: &Grid, atoms: &[Atom], params: &PbParams) -> Self {
        Self::solvated_with(grid, atoms, params, &MolecularSurface::for_params(atoms, params))
    }

    /// [`Self::solvated`] with a prebuilt surface (it depends only on the
    /// atoms and `params`, so focusing levels share it).
    pub fn solvated_with(grid: &Grid, atoms: &[Atom], params: &PbParams, surf: &MolecularSurface) -> Self {
        let [eps_x, eps_y, eps_z, kappa_sq] =
            maps_on(&CpuBackend::new(), grid, atoms, params, Some(surf)).expect("the f64 CPU backend builds maps");
        Self {
            eps_x,
            eps_y,
            eps_z,
            kappa_sq,
        }
    }
}

/// The maps `[eps_x, eps_y, eps_z, kappa_sq]` of one grid level built on a
/// backend: the solvated state's for a surface `surf` (prebuilt, shared by
/// focusing levels), else the reference state's (ε = pdie, no ions). The
/// values are those of [`CoefficientMaps::solvated_with`] and
/// [`CoefficientMaps::homogeneous`], bit for bit, on every backend.
pub fn maps_on<B: Backend>(
    be: &B,
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
) -> Result<[B::Vector; 4]> {
    let Some(surf) = surf else {
        let [x, y, z] = exactly(be.mark_maps(grid, &[[0.0; 3]; 3], params.pdie, &[])?)?;
        return Ok([x, y, z, be.zeros(grid.len())?]);
    };
    let (pdie, sdie) = (params.pdie, params.sdie);
    let h = grid.spacing;
    let [eps_x, eps_y, eps_z] = match params.surface {
        SurfaceMethod::Mol | SurfaceMethod::Vdw => exactly(be.mark_maps(
            grid,
            &[[0.5 * h[0], 0.0, 0.0], [0.0, 0.5 * h[1], 0.0], [0.0, 0.0, 0.5 * h[2]]],
            sdie,
            &[(&surf.solute, pdie), (&surf.probes, sdie)],
        )?)?,
        SurfaceMethod::HarmonicMol => {
            // Not an APBS surface; built on the host.
            let node = surf.dielectric_map(grid, [0.0; 3], pdie, sdie);
            let (x, y, z) = harmonic_faces(grid, &node);
            [be.upload_owned(x)?, be.upload_owned(y)?, be.upload_owned(z)?]
        }
    };

    let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
    let kappa_sq = if kbar2 > 0.0 {
        let ion_r = params.ion_exclusion_radius();
        let spheres: Vec<Sphere> = atoms
            .iter()
            .filter(|a| a.radius > APBS_VSMALL)
            .map(|a| Sphere {
                center: a.position,
                radius: ion_r + a.radius,
            })
            .collect();
        let [k] = exactly(be.mark_maps(grid, &[[0.0; 3]], kbar2, &[(&spheres, 0.0)])?)?;
        k
    } else {
        be.zeros(grid.len())?
    };
    Ok([eps_x, eps_y, eps_z, kappa_sq])
}

/// The maps of [`Backend::mark_maps`], one per shift.
fn exactly<V, const N: usize>(maps: Vec<V>) -> Result<[V; N]> {
    maps.try_into()
        .map_err(|m: Vec<V>| PbError::Backend(format!("mark_maps returned {} maps for {N} shifts", m.len())))
}

/// APBS' cell-list radius (`Vpbe_ctor2`): max(srad, largest ion radius) + 0.5 Å.
pub fn apbs_clist_radius(params: &PbParams) -> f64 {
    params.srad.max(params.ion_exclusion_radius()) + APBS_SPLINE_WINDOW
}

fn harmonic_faces(grid: &Grid, node: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let [nx, ny, nz] = grid.dims;
    let strides = [1, nx, nx * ny];
    let mut maps = [vec![0.0; grid.len()], vec![0.0; grid.len()], vec![0.0; grid.len()]];
    for (axis, map) in maps.iter_mut().enumerate() {
        map.par_iter_mut().enumerate().for_each(|(c, e)| {
            let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
            let last = [i == nx - 1, j == ny - 1, k == nz - 1][axis];
            *e = if last {
                node[c]
            } else {
                let (a, b) = (node[c], node[c + strides[axis]]);
                2.0 * a * b / (a + b)
            };
        });
    }
    let [x, y, z] = maps;
    (x, y, z)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: [f64; 3],
    pub radius: f64,
}

/// APBS `VRINT`: round half up.
#[inline]
fn vrint(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// APBS `VaccSurf_refSphere(npts)`: θ/φ lattice on the unit sphere. The
/// returned count generally differs from `npts`.
pub fn apbs_reference_sphere(npts: usize) -> Vec<[f64; 3]> {
    let frac = npts as f64 / 4.0;
    let ntheta = vrint((PI * frac).sqrt());
    let dtheta = PI / ntheta as f64;
    let nphimax = 2 * ntheta;
    let mut pts = Vec::new();
    for itheta in 0..ntheta {
        let theta = dtheta * itheta as f64;
        let (sintheta, costheta) = (theta.sin(), theta.cos());
        let nphi = vrint(sintheta * nphimax as f64);
        if nphi != 0 {
            let dphi = 2.0 * PI / nphi as f64;
            for iphi in 0..nphi {
                let phi = dphi * iphi as f64;
                pts.push([phi.cos() * sintheta, phi.sin() * sintheta, costheta]);
            }
        }
    }
    pts
}

/// Molecular (solvent-excluded) surface in APBS `srfm mol` form: solute
/// spheres (SAS volume) minus probe spheres around accessible SAS points.
/// With `srad = 0` it is the union of van der Waals spheres.
#[derive(Clone, Debug)]
pub struct MolecularSurface {
    /// Atom spheres of radius `r_i + srad` (only `r_i > 1e-9`).
    pub solute: Vec<Sphere>,
    /// Probe spheres (radius `srad`) centred on accessible SAS points.
    pub probes: Vec<Sphere>,
    /// Number of points in the reference sphere (0 when `srad = 0`).
    pub reference_points: usize,
}

impl MolecularSurface {
    /// The surface `params.surface` asks for (`vdw`: no probe).
    pub fn for_params(atoms: &[Atom], params: &PbParams) -> Self {
        let srad = match params.surface {
            SurfaceMethod::Vdw => 0.0,
            SurfaceMethod::Mol | SurfaceMethod::HarmonicMol => params.srad,
        };
        Self::new(atoms, srad, params.sdens, apbs_clist_radius(params))
    }

    /// `clist_radius` is APBS' cell-list radius (see [`apbs_clist_radius`]);
    /// it only affects the density of SAS points.
    pub fn new(atoms: &[Atom], srad: f64, sdens: f64, clist_radius: f64) -> Self {
        let solute = atoms
            .iter()
            .filter(|a| a.radius > APBS_VSMALL)
            .map(|a| Sphere {
                center: a.position,
                radius: a.radius + srad,
            })
            .collect();
        if srad <= APBS_VSMALL {
            return Self {
                solute,
                probes: Vec::new(),
                reference_points: 0,
            };
        }
        let max_r = atoms.iter().map(|a| a.radius).fold(0.0, f64::max);
        let maxrad = max_r + clist_radius;
        let nsphere = (4.0 * PI * maxrad * maxrad * sdens).ceil() as usize;
        let reference = apbs_reference_sphere(nsphere);

        let pos: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let reach = max_r + srad;
        let cells = CellList::new(&pos, reach.max(1.0));
        let reference_ref = &reference;
        let cells_ref = &cells;
        let probes: Vec<Sphere> = atoms
            .par_iter()
            .enumerate()
            .filter(|(_, a)| a.radius >= APBS_VSMALL)
            .flat_map_iter(|(i, a)| {
                let rad = a.radius + srad;
                reference_ref.iter().filter_map(move |u| {
                    let p = [
                        rad * u[0] + a.position[0],
                        rad * u[1] + a.position[1],
                        rad * u[2] + a.position[2],
                    ];
                    let buried = cells_ref.any_near(p, reach, |j| {
                        if j == i {
                            return false;
                        }
                        let rj = atoms[j].radius + srad;
                        dist2(p, atoms[j].position) < rj * rj
                    });
                    (!buried).then_some(Sphere {
                        center: p,
                        radius: srad,
                    })
                })
            })
            .collect();
        Self {
            solute,
            probes,
            reference_points: reference.len(),
        }
    }

    /// ε on the grid shifted by `shift` (Å): `pdie` in the solute, `sdie` elsewhere.
    pub fn dielectric_map(&self, grid: &Grid, shift: [f64; 3], pdie: f64, sdie: f64) -> Vec<f64> {
        mark_map(grid, shift, sdie, &[(&self.solute, pdie), (&self.probes, sdie)])
    }
}

/// APBS `markSphere` index range along one axis: nodes that may lie within
/// `rtot` of `pos` (grid-relative), clamped to `[0, n-1]`.
#[inline]
fn mark_range(pos: f64, rtot: f64, h: f64, n: usize) -> Option<(usize, usize)> {
    let range = rtot + 0.5 * h;
    let lo = ((pos - range) / h).ceil().max(0.0);
    let hi = ((pos + range) / h).floor().min((n - 1) as f64);
    (hi >= lo).then_some((lo as usize, hi as usize))
}

/// Fills a node map with `fill`, then applies each pass `(spheres, value)` in
/// order, setting `value` at every node within (≤) a sphere — APBS
/// `markSphere` on a grid whose origin is shifted by `shift`. Parallel over
/// z-planes; passes are applied per plane in order, which is equivalent to
/// APBS' sequential loops because marking within a pass is idempotent.
pub fn mark_map(grid: &Grid, shift: [f64; 3], fill: f64, passes: &[(&[Sphere], f64)]) -> Vec<f64> {
    let [nx, ny, nz] = grid.dims;
    let [hx, hy, hz] = grid.spacing;
    let min = [0, 1, 2].map(|a| grid.origin[a] + shift[a]);

    // Per pass: CSR buckets of sphere indices by z-plane.
    let buckets: Vec<(Vec<u32>, Vec<u32>)> = passes
        .iter()
        .map(|(spheres, _)| {
            let ranges: Vec<Option<(usize, usize)>> = spheres
                .iter()
                .map(|s| mark_range(s.center[2] - min[2], s.radius, hz, nz))
                .collect();
            let mut starts = vec![0u32; nz + 1];
            for (lo, hi) in ranges.iter().flatten() {
                for k in *lo..=*hi {
                    starts[k + 1] += 1;
                }
            }
            for k in 0..nz {
                starts[k + 1] += starts[k];
            }
            let mut fillp = starts.clone();
            let mut items = vec![0u32; starts[nz] as usize];
            for (n, r) in ranges.iter().enumerate() {
                if let Some((lo, hi)) = r {
                    for k in *lo..=*hi {
                        items[fillp[k] as usize] = n as u32;
                        fillp[k] += 1;
                    }
                }
            }
            (starts, items)
        })
        .collect();

    // Zeroed allocation, filled per plane: the pages are touched in parallel.
    let mut out = vec![0.0; grid.len()];
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        plane.fill(fill);
        let fk = k as f64;
        for ((spheres, value), (starts, items)) in passes.iter().zip(&buckets) {
            for &n in &items[starts[k] as usize..starts[k + 1] as usize] {
                let s = spheres[n as usize];
                let posx = s.center[0] - min[0];
                let posy = s.center[1] - min[1];
                let posz = s.center[2] - min[2];
                let rtot2 = s.radius * s.radius;
                let (Some((imin, imax)), Some((jmin, jmax))) =
                    (mark_range(posx, s.radius, hx, nx), mark_range(posy, s.radius, hy, ny))
                else {
                    continue;
                };
                let dz = posz - hz * fk;
                let dz2 = dz * dz;
                for i in imin..=imax {
                    let dx = posx - hx * i as f64;
                    let dx2 = dx * dx;
                    for j in jmin..=jmax {
                        let dy = posy - hy * j as f64;
                        let dy2 = dy * dy;
                        if dx2 + dy2 > rtot2 {
                            continue;
                        }
                        if dz2 + dy2 + dx2 <= rtot2 {
                            plane[i + nx * j] = *value;
                        }
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(p: [f64; 3], r: f64) -> Atom {
        Atom {
            position: p,
            charge: 0.0,
            radius: r,
        }
    }

    #[test]
    fn reference_sphere_matches_apbs_count() {
        // APBS 3.4.1, ion r = 2 Å, srad 1.4, ion radius 2.0, sdens 10:
        // maxrad = 2 + (2.0 + 0.5) = 4.5 -> "Using 2584-point reference sphere".
        let n = (4.0 * PI * 4.5 * 4.5 * 10.0f64).ceil() as usize;
        assert_eq!(apbs_reference_sphere(n).len(), 2584);
        for p in apbs_reference_sphere(100) {
            assert!((p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1.0).abs() < 1e-12);
        }
    }

    /// Brute-force classification with the same arithmetic as `mark_map`.
    fn brute(grid: &Grid, shift: [f64; 3], surf: &MolecularSurface, i: usize, j: usize, k: usize) -> bool {
        let min = [0, 1, 2].map(|a| grid.origin[a] + shift[a]);
        let inside = |s: &Sphere| {
            let dx = (s.center[0] - min[0]) - grid.spacing[0] * i as f64;
            let dy = (s.center[1] - min[1]) - grid.spacing[1] * j as f64;
            let dz = (s.center[2] - min[2]) - grid.spacing[2] * k as f64;
            dz * dz + dy * dy + dx * dx <= s.radius * s.radius
        };
        surf.solute.iter().any(inside) && !surf.probes.iter().any(inside)
    }

    #[test]
    fn marking_equals_brute_force_classification() {
        let atoms = [
            atom([-1.6, 0.1, 0.0], 1.7),
            atom([1.5, 0.0, 0.2], 1.6),
            atom([0.0, 2.1, -0.3], 1.5),
            atom([0.2, 0.9, 0.4], 0.0),
        ];
        let surf = MolecularSurface::new(&atoms, 1.4, 4.0, 1.9);
        assert!(!surf.probes.is_empty());
        let grid = Grid::centered([0.05, 0.4, 0.1], [12.0, 11.0, 10.0], [25, 23, 21]).unwrap();
        let shift = [0.5 * grid.spacing[0], 0.0, 0.0];
        let map = surf.dielectric_map(&grid, shift, 2.0, 80.0);
        for k in 0..21 {
            for j in 0..23 {
                for i in 0..25 {
                    let solute = brute(&grid, shift, &surf, i, j, k);
                    assert_eq!(map[grid.index(i, j, k)] == 2.0, solute, "node {i} {j} {k}");
                }
            }
        }
    }

    #[test]
    fn crevice_between_two_spheres_is_filled_by_ses() {
        // Two touching spheres: the neck region is outside both vdW spheres
        // but inaccessible to a 1.4 Å probe -> solute for SES, solvent for vdW.
        let atoms = [atom([-1.5, 0.0, 0.0], 1.5), atom([1.5, 0.0, 0.0], 1.5)];
        let grid = Grid::centered([0.0, 0.3, 0.0], [8.0, 8.0, 8.0], [81, 81, 81]).unwrap();
        let node = grid.index(40, 40, 40); // (0, 0.3, 0)
        let ses = MolecularSurface::new(&atoms, 1.4, 10.0, 1.9).dielectric_map(&grid, [0.0; 3], 1.0, 80.0);
        let vdw = MolecularSurface::new(&atoms, 0.0, 10.0, 1.9).dielectric_map(&grid, [0.0; 3], 1.0, 80.0);
        assert_eq!(ses[node], 1.0);
        assert_eq!(vdw[node], 80.0);
        // Far away (y = 3.9) is solvent for both.
        assert_eq!(ses[grid.index(40, 76, 40)], 80.0);
    }

    #[test]
    fn zero_radius_atoms_bury_sas_points_but_add_no_volume() {
        let a = atom([0.0; 3], 1.5);
        let h = atom([1.5 + 1.4, 0.0, 0.0], 0.0); // sits on a's SAS
        let with_h = MolecularSurface::new(&[a, h], 1.4, 10.0, 1.9);
        let without = MolecularSurface::new(&[a], 1.4, 10.0, 1.9);
        assert_eq!(with_h.solute.len(), 1);
        assert!(with_h.probes.len() < without.probes.len());
    }
}
