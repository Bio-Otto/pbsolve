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
use crate::gaussian::{contour_faces, gaussian_maps};
use crate::grid::Grid;
use crate::molecule::{dist2, Atom};
use crate::params::{InterfaceMethod, PbParams, SurfaceMethod};
use crate::spatial::CellList;
use crate::spline::{spline_maps, SmoothMaps};
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

    /// The reference state's maps: ε = `pdie` (with `gauss`, the Gaussian ε
    /// inside its contour) and no ions.
    pub fn reference(grid: &Grid, atoms: &[Atom], params: &PbParams) -> Self {
        let [eps_x, eps_y, eps_z, kappa_sq] =
            maps_on(&CpuBackend::new(), grid, atoms, params, None).expect("the f64 CPU backend builds maps");
        Self {
            eps_x,
            eps_y,
            eps_z,
            kappa_sq,
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
/// focusing levels), else the reference state's (ε = pdie, no ions; with
/// `gauss` the Gaussian ε inside its contour). The values do not depend on
/// the backend, bit for bit.
pub fn maps_on<B: Backend>(
    be: &B,
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
) -> Result<[B::Vector; 4]> {
    let Some(surf) = surf else {
        if let Some(eps) = &params.atom_pdie {
            // shortcut: the reference state rebuilds the surface on every level; share it with the
            // solvated state's if the surface ever shows up in the timings.
            let surf = MolecularSurface::for_params(atoms, params);
            let [x, y, z] = atom_dielectric_faces(grid, atoms, &surf, eps, params.pdie)?;
            return Ok([
                be.upload_owned(x)?,
                be.upload_owned(y)?,
                be.upload_owned(z)?,
                be.zeros(grid.len())?,
            ]);
        }
        if params.surface == SurfaceMethod::Gaussian {
            let [x, y, z] = contour_faces(grid, atoms, params);
            return Ok([
                be.upload_owned(x)?,
                be.upload_owned(y)?,
                be.upload_owned(z)?,
                be.zeros(grid.len())?,
            ]);
        }
        let [x, y, z] = exactly(be.mark_maps(grid, &[[0.0; 3]; 3], params.pdie, &[])?)?;
        return Ok([x, y, z, be.zeros(grid.len())?]);
    };
    let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
    if params.surface.is_smooth() {
        // Built on the host (spline surfaces: APBS `fillcoCoefSpline*`).
        let SmoothMaps {
            eps: [x, y, z],
            mut accessibility,
        } = if params.surface == SurfaceMethod::Gaussian {
            gaussian_maps(grid, atoms, params)
        } else {
            spline_maps(grid, atoms, params)
        };
        accessibility.par_iter_mut().for_each(|k| *k *= kbar2);
        return Ok([
            be.upload_owned(x)?,
            be.upload_owned(y)?,
            be.upload_owned(z)?,
            be.upload_owned(accessibility)?,
        ]);
    }
    let (pdie, sdie) = (params.pdie, params.sdie);
    let h = grid.spacing;
    let [eps_x, eps_y, eps_z] = match params.surface {
        _ if params.atom_pdie.is_some() => {
            // Not an APBS option; built on the host.
            let eps = params.atom_pdie.as_deref().unwrap_or_default();
            let [x, y, z] = atom_dielectric_faces(grid, atoms, surf, eps, sdie)?;
            [be.upload_owned(x)?, be.upload_owned(y)?, be.upload_owned(z)?]
        }
        SurfaceMethod::Mol | SurfaceMethod::Vdw if params.interface == InterfaceMethod::Fractional => {
            // Not an APBS option; built on the host.
            let [x, y, z] = fractional_faces(grid, surf, pdie, sdie);
            [be.upload_owned(x)?, be.upload_owned(y)?, be.upload_owned(z)?]
        }
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
        SurfaceMethod::Spline2 | SurfaceMethod::Spline4 | SurfaceMethod::Gaussian => unreachable!("built above"),
    };

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

/// Face dielectrics with one interior dielectric per atom (`atom_pdie`, the
/// multiple dielectrics of DelPhi; Rocchia, Alexov & Honig 2001). A face inside
/// the solute of `surf` (the sharp `mol`/`vdw` marking, as [`mark_map`] does
/// it) takes `eps[n]` of the atom n that owns its midpoint x: the smallest
/// power distance |x − c_n|² − r_n² (the Laguerre cells of the spheres, whose
/// walls pass through the circles where two spheres meet), ties to the lower
/// index; radius-0 atoms take part with r = 0. Every other face gets `outside`.
/// With all `eps` equal to `pdie` the maps equal the sharp marking bit for bit.
pub fn atom_dielectric_faces(
    grid: &Grid,
    atoms: &[Atom],
    surf: &MolecularSurface,
    eps: &[f64],
    outside: f64,
) -> Result<[Vec<f64>; 3]> {
    if eps.len() != atoms.len() {
        return Err(PbError::InvalidInput(format!(
            "{} atom dielectrics for {} atoms",
            eps.len(),
            atoms.len()
        )));
    }
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
    let reach = surf.solute.iter().map(|s| s.radius).fold(0.0, f64::max);
    let cells = CellList::new(&centers, reach.max(1.0));
    // An interior face lies within r_a + srad of some atom a, whose power distance is at most
    // (r_a + srad)² − r_a²; an atom j with a smaller one is then within r_max + srad. 2h of margin
    // covers the grid-relative arithmetic of the marking.
    let radius = reach + 2.0 * grid.spacing.iter().copied().fold(0.0, f64::max);
    let h = grid.spacing;
    let [nx, ny, _] = grid.dims;
    let shifts = [[0.5 * h[0], 0.0, 0.0], [0.0, 0.5 * h[1], 0.0], [0.0, 0.0, 0.5 * h[2]]];
    Ok(shifts.map(|shift| {
        let mut map = mark_map(grid, shift, 0.0, &[(&surf.solute, 1.0), (&surf.probes, 0.0)]);
        map.par_iter_mut().enumerate().for_each(|(c, e)| {
            if *e == 0.0 {
                *e = outside;
                return;
            }
            let p = grid.position(c % nx, (c / nx) % ny, c / (nx * ny));
            let x = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
            let mut best = (f64::INFINITY, usize::MAX);
            cells.any_near(x, radius, |n| {
                let d = dist2(x, atoms[n].position) - atoms[n].radius * atoms[n].radius;
                if d < best.0 || (d == best.0 && n < best.1) {
                    best = (d, n);
                }
                false
            });
            *e = eps[best.1];
        });
        map
    }))
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

/// Face dielectrics by the fractional harmonic average (`interface =
/// "fractional"`): the grid edge between a face's two nodes is split exactly
/// into its lengths inside and outside the solute of `surf` (solute spheres
/// minus probe spheres), and ε = h / (ℓ_in/pdie + ℓ_out/sdie), the effective
/// dielectric of that edge in one dimension. A face whose edge does not meet
/// the surface gets `pdie` or `sdie`, as with the sharp marking.
pub fn fractional_faces(grid: &Grid, surf: &MolecularSurface, pdie: f64, sdie: f64) -> [Vec<f64>; 3] {
    let cells = |s: &[Sphere]| {
        let centers: Vec<[f64; 3]> = s.iter().map(|s| s.center).collect();
        let rmax = s.iter().map(|s| s.radius).fold(0.0, f64::max);
        (CellList::new(&centers, rmax.max(1.0)), rmax)
    };
    let (solute, rsol) = cells(&surf.solute);
    let (probes, rprobe) = cells(&surf.probes);
    // Node classes for the edges that start there (any axis, length <= reach):
    // `near` = 0, the edge cannot meet a solute sphere (sdie); `deep` = 1, it
    // lies inside one and meets no probe (pdie). Only the rest is split.
    let reach = grid.spacing.iter().copied().fold(0.0, f64::max) * (1.0 + 1e-6) + 1e-9;
    let grown = |s: &[Sphere], d: f64| -> Vec<Sphere> {
        s.iter()
            .filter(|s| s.radius + d > 0.0)
            .map(|s| Sphere {
                center: s.center,
                radius: s.radius + d,
            })
            .collect()
    };
    let near = mark_map(grid, [0.0; 3], 0.0, &[(&grown(&surf.solute, reach), 1.0)]);
    let deep = mark_map(
        grid,
        [0.0; 3],
        0.0,
        &[(&grown(&surf.solute, -reach), 1.0), (&grown(&surf.probes, reach), 0.0)],
    );
    let [nx, ny, _] = grid.dims;
    [0, 1, 2].map(|axis| {
        let h = grid.spacing[axis];
        let mut map = vec![sdie; grid.len()];
        map.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
            let (mut inner, mut carved) = (Vec::new(), Vec::new());
            for j in 0..ny {
                for i in 0..nx {
                    let node = i + nx * (j + ny * k);
                    if near[node] == 0.0 {
                        continue;
                    }
                    if deep[node] == 1.0 {
                        plane[i + nx * j] = pdie;
                        continue;
                    }
                    let p = [0, 1, 2].map(|a| grid.origin[a] + [i, j, k][a] as f64 * grid.spacing[a]);
                    let mut mid = p;
                    mid[axis] += 0.5 * h;
                    // Lengths along the edge, t in [0, h]: inside some solute sphere ...
                    inner.clear();
                    let covered = solute.any_near(mid, rsol + 0.5 * h, |n| match chord(&surf.solute[n], p, axis, h) {
                        Some((t0, t1)) if t0 <= 0.0 && t1 >= h => true,
                        Some(iv) => {
                            inner.push(iv);
                            false
                        }
                        None => false,
                    });
                    if covered {
                        inner.clear();
                        inner.push((0.0, h));
                    }
                    let length = merge(&mut inner);
                    if length <= 0.0 {
                        continue;
                    }
                    // ... minus the parts inside a probe sphere (one probe often removes all).
                    let (first, last) = (inner[0].0, inner[inner.len() - 1].1);
                    carved.clear();
                    let cleared =
                        probes.any_near(mid, rprobe + 0.5 * h, |n| match chord(&surf.probes[n], p, axis, h) {
                            Some((t0, t1)) if t0 <= first && t1 >= last => true,
                            Some(iv) => {
                                carved.push(iv);
                                false
                            }
                            None => false,
                        });
                    if cleared {
                        continue;
                    }
                    merge(&mut carved);
                    let f = (length - overlap(&inner, &carved)) / h;
                    plane[i + nx * j] = if f >= 1.0 {
                        pdie
                    } else if f <= 0.0 {
                        sdie
                    } else {
                        1.0 / (f / pdie + (1.0 - f) / sdie)
                    };
                }
            }
        });
        map
    })
}

/// The part `(t0, t1)`, 0 ≤ t0 < t1 ≤ h, of the segment `p + t e_axis`
/// (0 ≤ t ≤ h) inside the sphere, if any.
#[inline]
fn chord(s: &Sphere, p: [f64; 3], axis: usize, h: f64) -> Option<(f64, f64)> {
    let d2: f64 = (0..3)
        .filter(|&a| a != axis)
        .map(|a| (p[a] - s.center[a]).powi(2))
        .sum();
    let r2 = s.radius * s.radius;
    if d2 >= r2 {
        return None;
    }
    let half = (r2 - d2).sqrt();
    let c = s.center[axis] - p[axis];
    let (t0, t1) = ((c - half).max(0.0), (c + half).min(h));
    (t1 > t0).then_some((t0, t1))
}

/// Sorts and merges intervals in place into disjoint ones; returns their total length.
fn merge(iv: &mut Vec<(f64, f64)>) -> f64 {
    iv.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut n = 0;
    for m in 0..iv.len() {
        if n > 0 && iv[m].0 <= iv[n - 1].1 {
            iv[n - 1].1 = iv[n - 1].1.max(iv[m].1);
        } else {
            iv[n] = iv[m];
            n += 1;
        }
    }
    iv.truncate(n);
    iv.iter().map(|(a, b)| b - a).sum()
}

/// Total length of the intersection of two sorted, disjoint interval lists.
fn overlap(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let (mut i, mut j, mut total) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        let (lo, hi) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
        if hi > lo {
            total += hi - lo;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
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
            // The smooth maps do not use the surface beyond marking the state as solvated.
            SurfaceMethod::Vdw | SurfaceMethod::Spline2 | SurfaceMethod::Spline4 | SurfaceMethod::Gaussian => 0.0,
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
        let reference = apbs_reference_sphere(reference_sphere_points(atoms, clist_radius, sdens));
        let reference_ref = &reference;
        let acc = Accessibility::new(atoms, srad);
        let acc_ref = &acc;
        let probes: Vec<Sphere> = atoms
            .par_iter()
            .enumerate()
            .filter(|(_, a)| a.radius >= APBS_VSMALL)
            .flat_map_iter(|(i, a)| {
                reference_ref.iter().filter_map(move |u| {
                    let p = sas_point(a, srad, u);
                    acc_ref.accessible(p, Some(i)).then_some(Sphere {
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

/// APBS' size of the reference sphere (`Vacc_storeParms`): `sdens` points per
/// Å² of the largest atom radius plus the cell-list radius.
pub(crate) fn reference_sphere_points(atoms: &[Atom], clist_radius: f64, sdens: f64) -> usize {
    let maxrad = atoms.iter().map(|a| a.radius).fold(0.0, f64::max) + clist_radius;
    (4.0 * PI * maxrad * maxrad * sdens).ceil() as usize
}

/// The reference-sphere point `u` on atom `a`'s sphere of radius r + srad
/// (`Vacc_atomSurf`).
#[inline]
pub(crate) fn sas_point(a: &Atom, srad: f64, u: &[f64; 3]) -> [f64; 3] {
    let rad = a.radius + srad;
    [
        rad * u[0] + a.position[0],
        rad * u[1] + a.position[1],
        rad * u[2] + a.position[2],
    ]
}

/// APBS' probe accessibility (`ivdwAccExclus`): a point is accessible unless
/// it lies strictly inside the r + srad sphere of an atom (radius-0 atoms
/// included), optionally ignoring one atom.
pub(crate) struct Accessibility<'a> {
    atoms: &'a [Atom],
    srad: f64,
    reach: f64,
    cells: CellList,
}

impl<'a> Accessibility<'a> {
    pub(crate) fn new(atoms: &'a [Atom], srad: f64) -> Self {
        let pos: Vec<[f64; 3]> = atoms.iter().map(|a| a.position).collect();
        let reach = atoms.iter().map(|a| a.radius).fold(0.0, f64::max) + srad;
        Self {
            atoms,
            srad,
            reach,
            cells: CellList::new(&pos, reach.max(1.0)),
        }
    }

    #[inline]
    pub(crate) fn accessible(&self, p: [f64; 3], except: Option<usize>) -> bool {
        !self.cells.any_near(p, self.reach, |j| {
            if Some(j) == except {
                return false;
            }
            let rj = self.atoms[j].radius + self.srad;
            dist2(p, self.atoms[j].position) < rj * rj
        })
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

    fn cluster() -> (Vec<Atom>, Grid) {
        let atoms = vec![
            atom([-1.6, 0.1, 0.0], 1.7),
            atom([1.5, 0.0, 0.2], 1.6),
            atom([0.0, 2.1, -0.3], 1.5),
            atom([0.2, 0.9, 0.4], 0.0),
            atom([0.4, -1.3, 1.1], 1.2),
            atom([-0.7, 0.6, -1.4], 1.9),
        ];
        let grid = Grid::centered([0.05, 0.4, 0.1], [13.0, 12.0, 11.0], [27, 25, 23]).unwrap();
        (atoms, grid)
    }

    #[test]
    fn uniform_atom_dielectrics_give_the_sharp_maps_bit_for_bit() {
        let (atoms, grid) = cluster();
        for surface in [SurfaceMethod::Mol, SurfaceMethod::Vdw] {
            let plain = PbParams {
                pdie: 2.0,
                surface,
                ..PbParams::default()
            };
            let uniform = PbParams {
                atom_pdie: Some(vec![2.0; atoms.len()].into()),
                ..plain.clone()
            };
            let a = CoefficientMaps::solvated(&grid, &atoms, &plain);
            let b = CoefficientMaps::solvated(&grid, &atoms, &uniform);
            let to_bits = |m: &CoefficientMaps| -> Vec<u64> {
                [&m.eps_x, &m.eps_y, &m.eps_z]
                    .iter()
                    .flat_map(|v| v.iter().map(|x| x.to_bits()))
                    .collect()
            };
            assert_eq!(to_bits(&a), to_bits(&b), "{surface:?} solvated");
            let r = CoefficientMaps::reference(&grid, &atoms, &uniform);
            assert!(
                to_bits(&r).iter().all(|&x| x == 2.0f64.to_bits()),
                "{surface:?} reference"
            );
        }
    }

    #[test]
    fn each_interior_face_takes_the_dielectric_of_its_power_cell() {
        let (atoms, grid) = cluster();
        let eps: Vec<f64> = (0..atoms.len()).map(|n| 2.0 + 3.0 * n as f64).collect();
        let base = PbParams {
            pdie: 1.5,
            ..PbParams::default()
        };
        let params = PbParams {
            atom_pdie: Some(eps.clone().into()),
            ..base.clone()
        };
        let sharp = CoefficientMaps::solvated(&grid, &atoms, &base);
        let solvated = CoefficientMaps::solvated(&grid, &atoms, &params);
        let reference = CoefficientMaps::reference(&grid, &atoms, &params);
        let h = grid.spacing;
        let mut interior = 0;
        for (axis, (s, (v, r))) in [&sharp.eps_x, &sharp.eps_y, &sharp.eps_z]
            .into_iter()
            .zip([&solvated.eps_x, &solvated.eps_y, &solvated.eps_z].into_iter().zip([
                &reference.eps_x,
                &reference.eps_y,
                &reference.eps_z,
            ]))
            .enumerate()
        {
            for c in 0..grid.len() {
                if s[c] == base.sdie {
                    assert_eq!((v[c], r[c]), (base.sdie, base.pdie), "exterior face {c}");
                    continue;
                }
                interior += 1;
                let [nx, ny, _] = grid.dims;
                let mut x = grid.position(c % nx, (c / nx) % ny, c / (nx * ny));
                x[axis] += 0.5 * h[axis];
                // Brute force over every atom.
                let power = |a: &Atom| dist2(x, a.position) - a.radius * a.radius;
                let owner = (0..atoms.len())
                    .min_by(|&i, &j| power(&atoms[i]).total_cmp(&power(&atoms[j])))
                    .unwrap();
                assert_eq!((v[c], r[c]), (eps[owner], eps[owner]), "face {c} of axis {axis}");
            }
        }
        assert!(interior > 1000);
        // Every atom owns faces (also the radius-0 atom 3, at the edge of atom 1).
        let used: std::collections::BTreeSet<u64> = solvated.eps_x.iter().map(|e| e.to_bits()).collect();
        assert!(eps.iter().all(|e| used.contains(&e.to_bits())));
    }

    #[test]
    fn atom_dielectrics_are_validated() {
        let (atoms, grid) = cluster();
        let ok = PbParams {
            atom_pdie: Some(vec![4.0; atoms.len()].into()),
            ..PbParams::default()
        };
        assert!(ok.validate().is_ok());
        for n in [2, atoms.len() + 1] {
            let wrong = PbParams {
                atom_pdie: Some(vec![4.0; n].into()),
                ..ok.clone()
            };
            assert!(
                maps_on(&CpuBackend::new(), &grid, &atoms, &wrong, None).is_err(),
                "{n} values"
            );
        }
        for bad in [
            PbParams {
                atom_pdie: Some(vec![0.0; atoms.len()].into()),
                ..ok.clone()
            },
            PbParams {
                atom_pdie: Some(vec![f64::NAN; atoms.len()].into()),
                ..ok.clone()
            },
            PbParams {
                surface: SurfaceMethod::Spline2,
                ..ok.clone()
            },
            PbParams {
                interface: InterfaceMethod::Fractional,
                ..ok.clone()
            },
            PbParams {
                coulomb: crate::params::CoulombMethod::Analytic,
                ..ok.clone()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
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

    /// Fraction of an edge inside the solute, independently of `fractional_faces`:
    /// split the edge at every sphere crossing and classify the midpoint of each piece.
    fn sampled_fraction(surf: &MolecularSurface, p: [f64; 3], axis: usize, h: f64) -> f64 {
        let mut ts = vec![0.0, h];
        for s in surf.solute.iter().chain(&surf.probes) {
            let d2: f64 = (0..3)
                .filter(|&a| a != axis)
                .map(|a| (p[a] - s.center[a]).powi(2))
                .sum();
            if d2 < s.radius * s.radius {
                let half = (s.radius * s.radius - d2).sqrt();
                ts.extend([s.center[axis] - p[axis] - half, s.center[axis] - p[axis] + half]);
            }
        }
        ts.retain(|t| (0.0..=h).contains(t));
        ts.sort_by(f64::total_cmp);
        let inside = |q: [f64; 3], spheres: &[Sphere]| spheres.iter().any(|s| dist2(q, s.center) < s.radius * s.radius);
        let mut length = 0.0;
        for w in ts.windows(2) {
            let mut q = p;
            q[axis] += 0.5 * (w[0] + w[1]);
            if inside(q, &surf.solute) && !inside(q, &surf.probes) {
                length += w[1] - w[0];
            }
        }
        length / h
    }

    #[test]
    fn fractional_faces_split_every_edge_exactly() {
        let atoms = [
            atom([-1.6, 0.1, 0.0], 1.7),
            atom([1.5, 0.0, 0.2], 1.6),
            atom([0.0, 2.1, -0.3], 1.5),
            atom([0.2, 0.9, 0.4], 0.0),
        ];
        let (pdie, sdie) = (2.0, 80.0);
        for srad in [1.4, 0.0] {
            let surf = MolecularSurface::new(&atoms, srad, 4.0, 1.9);
            let grid = Grid::centered([0.05, 0.4, 0.1], [12.0, 10.5, 9.0], [25, 23, 21]).unwrap();
            let maps = fractional_faces(&grid, &surf, pdie, sdie);
            let mut partial = 0;
            for (axis, map) in maps.iter().enumerate() {
                let h = grid.spacing[axis];
                let sharp = {
                    let mut shift = [0.0; 3];
                    shift[axis] = 0.5 * h;
                    surf.dielectric_map(&grid, shift, pdie, sdie)
                };
                for k in 0..21 {
                    for j in 0..23 {
                        for i in 0..25 {
                            let n = grid.index(i, j, k);
                            let p = [0, 1, 2].map(|a| grid.origin[a] + [i, j, k][a] as f64 * grid.spacing[a]);
                            let f = sampled_fraction(&surf, p, axis, h);
                            let expected = 1.0 / (f / pdie + (1.0 - f) / sdie);
                            assert!(
                                (map[n] - expected).abs() <= 1e-12 * expected,
                                "{srad} {axis} {i} {j} {k}"
                            );
                            if f == 0.0 || f == 1.0 {
                                // The edge does not meet the surface: as the sharp marking.
                                assert_eq!(map[n], sharp[n], "{srad} {axis} {i} {j} {k}");
                            } else {
                                partial += 1;
                            }
                        }
                    }
                }
            }
            assert!(partial > 300, "{srad}: {partial} cut faces");
        }
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
