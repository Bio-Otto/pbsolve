// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vpmg.c: qfForceSpline1, qfForceSpline2, qfForceSpline4,
// Vpmg_dbForce, Vpmg_ibForce), Copyright (c) 2010-2022 Battelle Memorial Institute and other
// APBS contributors, BSD-3-Clause. See THIRD_PARTY_NOTICES.md for the full license text.

//! Forces on the atoms (APBS `calcforce`) from one state's finest level.
//!
//! The PB free energy is the functional of the discrete potential u
//! ([`crate::nonlinear`], and for the linearized equation the same with the
//! ion density ½u²):
//!
//! G(u; R) = Σ_n q_n(R) u_n − ½ Σ_faces c_f ε_f(R) (Δu_f)² / 4π l_B
//!           − V Σ_n κ̄²_n(R) ρ(u_n) / 4π l_B,
//!
//! stationary in u at the solution. The force on an atom, −dG/dR, is
//! therefore the explicit derivative at fixed u [Im, Beglov & Roux 1998;
//! Gilson et al. 1993], three terms:
//!
//! * **qf** (charge–field): −q ∇_R Σ_n W_n(R) u_n, the derivative of the
//!   charge-assignment weights W (`chgm`);
//! * **db** (dielectric boundary): ½ Σ_faces c_f ∂ε_f/∂R (Δu_f)² / 4π l_B;
//! * **ib** (ionic boundary): V Σ_n ∂κ̄²_n/∂R ρ(u_n) / 4π l_B.
//!
//! db and ib need ε and κ̄² differentiable in R: the spline surfaces
//! ([`crate::spline`]), where ε_f = pdie + (sdie − pdie) H_f and ∂H_f/∂R_a =
//! H_f ∇ ln S_a. The loops, their node ranges and the order of the sums
//! follow APBS 3.4.1. Where APBS silently drops the force of an atom too
//! close to the edge of the grid, pbsolve fails.

use crate::charge::{bspline2, bspline4, dbspline2, dbspline4};
use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::nonlinear::IonTerms;
use crate::params::{ChargeMethod, PbParams};
use crate::spline::{spline_maps, LogGradient};
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared};
use rayon::prelude::*;
use std::f64::consts::PI;

/// APBS `VPMGSMALL`.
const VPMGSMALL: f64 = 1.0e-12;

/// The forces of one state on every atom by term, kT/Å.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForceTerms {
    /// Charge–field force.
    pub qf: Vec<[f64; 3]>,
    /// Dielectric-boundary force (zero in a uniform dielectric).
    pub db: Vec<[f64; 3]>,
    /// Ionic-boundary force (zero without ions).
    pub ib: Vec<[f64; 3]>,
}

/// The forces of a state whose finest level `grid` has the potential `u`
/// (kT/e, every node): qf, and with `solvated` also db and ib of the spline
/// surface `params.surface`.
pub fn state_forces(grid: &Grid, atoms: &[Atom], params: &PbParams, u: &[f64], solvated: bool) -> Result<ForceTerms> {
    let qf = charge_field_forces(grid, atoms, params.chgm, u);
    let zero = vec![[0.0; 3]; atoms.len()];
    if !solvated {
        return Ok(ForceTerms {
            qf,
            db: zero.clone(),
            ib: zero,
        });
    }
    if !params.surface.is_spline() {
        return invalid(format!(
            "forces need surface 'spl2' or 'spl4' (smooth in the atom positions), not '{}'",
            params.surface.as_str()
        ));
    }
    let maps = spline_maps(grid, atoms, params);
    let izmagic = 1.0 / (4.0 * PI * bjerrum_length_vacuum(params.temperature));
    let db = atoms
        .par_iter()
        .enumerate()
        .map(|(n, a)| dielectric_boundary_force(grid, n, a, &maps.eps, u, params, izmagic))
        .collect::<Result<Vec<_>>>()?;
    let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
    // ρ(u)·2: u² for the linearized equation, as APBS; the Boltzmann terms otherwise.
    let ions = IonTerms::new(params);
    let twice_density = |v: f64| match &ions {
        Some(ions) => 2.0 * ions.density(v),
        None => v * v,
    };
    let ib = atoms
        .par_iter()
        .enumerate()
        .map(|(n, a)| {
            ionic_boundary_force(
                grid,
                n,
                a,
                &maps.accessibility,
                u,
                params,
                kbar2,
                &twice_density,
                izmagic,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ForceTerms { qf, db, ib })
}

/// −q ∇_R Σ W_n(R) u_n for every atom with the weights of `chgm` (APBS
/// `qfForceSpline1`, `…2`, `…4`). Uncharged atoms get no force. The charged
/// atoms' stencils lie inside the grid ([`crate::charge`] checked them).
pub fn charge_field_forces(grid: &Grid, atoms: &[Atom], chgm: ChargeMethod, u: &[f64]) -> Vec<[f64; 3]> {
    atoms
        .par_iter()
        .map(|a| {
            if a.charge == 0.0 {
                return [0.0; 3];
            }
            match chgm {
                ChargeMethod::Trilinear => qf_trilinear(grid, a, u),
                ChargeMethod::Spline2 => qf_spline2(grid, a, u),
                ChargeMethod::Spline4 => qf_spline4(grid, a, u),
            }
        })
        .collect()
}

/// APBS `qfForceSpline1`: the derivative of the trilinear weights, zero
/// along an axis on whose grid line the atom sits.
fn qf_trilinear(grid: &Grid, atom: &Atom, u: &[f64]) -> [f64; 3] {
    let (lo, up, h) = (grid.origin, grid.upper(), grid.spacing);
    let p = atom.position;
    if (0..3).any(|a| p[a] <= lo[a] || p[a] >= up[a]) {
        return [0.0; 3];
    }
    let f = [0, 1, 2].map(|a| (p[a] - lo[a]) / h[a]);
    let hi = f.map(|x| x.ceil() as usize);
    let lw = f.map(|x| x.floor() as usize);
    let (dx, dy, dz) = (f[0] - lw[0] as f64, f[1] - lw[1] as f64, f[2] - lw[2] as f64);
    let at = |i: usize, j: usize, k: usize| u[grid.index(i, j, k)];
    let (ih, il, jh, jl, kh, kl) = (hi[0], lw[0], hi[1], lw[1], hi[2], lw[2]);
    let q = atom.charge;
    let inside = |d: f64| d > VPMGSMALL && (1.0 - d).abs() > VPMGSMALL;
    let fx = if inside(dx) {
        -q * (dy * dz * at(ih, jh, kh)
            + dy * (1.0 - dz) * at(ih, jh, kl)
            + (1.0 - dy) * dz * at(ih, jl, kh)
            + (1.0 - dy) * (1.0 - dz) * at(ih, jl, kl)
            - dy * dz * at(il, jh, kh)
            - dy * (1.0 - dz) * at(il, jh, kl)
            - (1.0 - dy) * dz * at(il, jl, kh)
            - (1.0 - dy) * (1.0 - dz) * at(il, jl, kl))
            / h[0]
    } else {
        0.0
    };
    let fy = if inside(dy) {
        -q * (dx * dz * at(ih, jh, kh) + dx * (1.0 - dz) * at(ih, jh, kl)
            - dx * dz * at(ih, jl, kh)
            - dx * (1.0 - dz) * at(ih, jl, kl)
            + (1.0 - dx) * dz * at(il, jh, kh)
            + (1.0 - dx) * (1.0 - dz) * at(il, jh, kl)
            - (1.0 - dx) * dz * at(il, jl, kh)
            - (1.0 - dx) * (1.0 - dz) * at(il, jl, kl))
            / h[1]
    } else {
        0.0
    };
    let fz = if inside(dz) {
        -q * (dy * dx * at(ih, jh, kh) - dy * dx * at(ih, jh, kl) + (1.0 - dy) * dx * at(ih, jl, kh)
            - (1.0 - dy) * dx * at(ih, jl, kl)
            + dy * (1.0 - dx) * at(il, jh, kh)
            - dy * (1.0 - dx) * at(il, jh, kl)
            + (1.0 - dy) * (1.0 - dx) * at(il, jl, kh)
            - (1.0 - dy) * (1.0 - dx) * at(il, jl, kl))
            / h[2]
    } else {
        0.0
    };
    [fx, fy, fz]
}

/// The node range of a B-spline stencil along one axis: APBS' floor − `half`
/// … ceil + `half`, clamped to the grid.
fn stencil(f: f64, half: i64, n: usize) -> (usize, usize) {
    let lo = (f.floor() as i64 - half).max(0) as usize;
    let hi = ((f.ceil() as i64 + half) as usize).min(n - 1);
    (lo, hi)
}

/// APBS `qfForceSpline2`: the derivative of the order-3 B-spline weights.
fn qf_spline2(grid: &Grid, atom: &Atom, u: &[f64]) -> [f64; 3] {
    let (lo, up, h) = (grid.origin, grid.upper(), grid.spacing);
    let p = atom.position;
    if (0..3).any(|a| p[a] <= lo[a] + h[a] || p[a] >= up[a] - h[a]) {
        return [0.0; 3];
    }
    let f = [0, 1, 2].map(|a| (p[a] - lo[a]) / h[a]);
    let r = [0, 1, 2].map(|a| stencil(f[a], 1, grid.dims[a]));
    let chi = |i: usize, f: f64| 1.5 + (i as f64 - f);
    let q = atom.charge;
    let mut force = [0.0; 3];
    for ii in r[0].0..=r[0].1 {
        let (mx, dmx) = (bspline2(chi(ii, f[0])), dbspline2(chi(ii, f[0])));
        for jj in r[1].0..=r[1].1 {
            let (my, dmy) = (bspline2(chi(jj, f[1])), dbspline2(chi(jj, f[1])));
            for kk in r[2].0..=r[2].1 {
                let (mz, dmz) = (bspline2(chi(kk, f[2])), dbspline2(chi(kk, f[2])));
                let v = u[grid.index(ii, jj, kk)];
                force[0] += (q * dmx * my * mz * v) / h[0];
                force[1] += (q * mx * dmy * mz * v) / h[1];
                force[2] += (q * mx * my * dmz * v) / h[2];
            }
        }
    }
    force
}

/// APBS `qfForceSpline4`: the derivative of the order-5 B-spline weights.
fn qf_spline4(grid: &Grid, atom: &Atom, u: &[f64]) -> [f64; 3] {
    let (lo, up, h) = (grid.origin, grid.upper(), grid.spacing);
    let p = atom.position;
    let mut e = [0.0; 3];
    if (0..3).all(|a| p[a] > lo[a] + 2.0 * h[a] && p[a] < up[a] - 2.0 * h[a]) {
        let f = [0, 1, 2].map(|a| (p[a] - lo[a]) / h[a]);
        let r = [0, 1, 2].map(|a| stencil(f[a], 2, grid.dims[a]));
        let chi = |i: usize, f: f64| 2.5 + (i as f64 - f);
        for ii in r[0].0..=r[0].1 {
            let (mx, dmx) = (bspline4(chi(ii, f[0])), dbspline4(chi(ii, f[0])));
            for jj in r[1].0..=r[1].1 {
                let (my, dmy) = (bspline4(chi(jj, f[1])), dbspline4(chi(jj, f[1])));
                for kk in r[2].0..=r[2].1 {
                    let (mz, dmz) = (bspline4(chi(kk, f[2])), dbspline4(chi(kk, f[2])));
                    let v = u[grid.index(ii, jj, kk)];
                    e[0] += v * dmx * my * mz / h[0];
                    e[1] += v * mx * dmy * mz / h[1];
                    e[2] += v * mx * my * dmz / h[2];
                }
            }
        }
    }
    e.map(|x| x * atom.charge)
}

/// The nodes `imin..=imax` along one axis whose faces atom `n` can reach
/// (APBS `Vpmg_dbForce`), which must leave one node to each edge.
fn reach(pos: f64, rtot: f64, h: f64, nodes: usize, n: usize, atom: &Atom) -> Result<(usize, usize)> {
    let lo = ((pos - rtot) / h).floor();
    let hi = ((pos + rtot) / h).ceil();
    if pos <= rtot || lo < 1.0 || hi > (nodes - 2) as f64 {
        return invalid(format!(
            "atom {n} at {:?}: its boundary force needs the finest grid to extend {rtot:.2} Å and one node \
             beyond it; enlarge the fine grid (fadd)",
            atom.position
        ));
    }
    Ok((lo as usize, hi as usize))
}

/// ½ Σ_faces c_f ∂ε_f/∂R (Δu)² / 4π l_B for atom `n` (APBS `Vpmg_dbForce`).
#[allow(clippy::too_many_arguments)]
fn dielectric_boundary_force(
    grid: &Grid,
    n: usize,
    atom: &Atom,
    eps: &[Vec<f64>; 3],
    u: &[f64],
    params: &PbParams,
    izmagic: f64,
) -> Result<[f64; 3]> {
    let (epsp, epsw) = (params.pdie, params.sdie);
    if (epsp - epsw).abs() < VPMGSMALL || atom.radius <= 0.0 {
        // A uniform dielectric, or an atom that does not shape the surface.
        return Ok([0.0; 3]);
    }
    let deps = epsw - epsp;
    let depsi = 1.0 / deps;
    let (lo, h) = (grid.origin, grid.spacing);
    let rtot = atom.radius + params.swin + params.srad;
    let pos = [0, 1, 2].map(|a| atom.position[a] - lo[a]);
    let mut r = [(0, 0); 3];
    for a in 0..3 {
        r[a] = reach(pos[a], rtot, h[a], grid.dims[a], n, atom)?;
    }
    let grad = LogGradient::new(params.surface, atom, params.swin, 0.0);
    // dH at a face: ∇ ln S_a times the face's H (from its ε).
    let d_h = |map: &[f64], c: usize, point: [f64; 3]| -> [f64; 3] {
        let hf = (map[c] - epsp) * depsi;
        grad.at(point).map(|g| g * hf)
    };
    let [nx, ny, _] = grid.dims;
    let (sx, sy, sz) = (1, nx, nx * ny);
    let mut force = [0.0; 3];
    for i in r[0].0..=r[0].1 {
        let (x, xp, xm) = (
            i as f64 * h[0] + lo[0],
            (i as f64 + 0.5) * h[0] + lo[0],
            (i as f64 - 0.5) * h[0] + lo[0],
        );
        for j in r[1].0..=r[1].1 {
            let (y, yp, ym) = (
                j as f64 * h[1] + lo[1],
                (j as f64 + 0.5) * h[1] + lo[1],
                (j as f64 - 0.5) * h[1] + lo[1],
            );
            for k in r[2].0..=r[2].1 {
                let (z, zp, zm) = (
                    k as f64 * h[2] + lo[2],
                    (k as f64 + 0.5) * h[2] + lo[2],
                    (k as f64 - 0.5) * h[2] + lo[2],
                );
                let c = grid.index(i, j, k);
                let hx = d_h(&eps[0], c, [xp, y, z]);
                let hy = d_h(&eps[1], c, [x, yp, z]);
                let hz = d_h(&eps[2], c, [x, y, zp]);
                let hxm = d_h(&eps[0], c - sx, [xm, y, z]);
                let hym = d_h(&eps[1], c - sy, [x, ym, z]);
                let hzm = d_h(&eps[2], c - sz, [x, y, zm]);
                let uc = u[c];
                let (dxp, dxm) = (u[c + sx] - uc, u[c - sx] - uc);
                let (dyp, dym) = (u[c + sy] - uc, u[c - sy] - uc);
                let (dzp, dzm) = (u[c + sz] - uc, u[c - sz] - uc);
                for l in 0..3 {
                    let tgrad = (hx[l] * dxp + hxm[l] * dxm) / (h[0] * h[0])
                        + (hy[l] * dyp + hym[l] * dym) / (h[1] * h[1])
                        + (hz[l] * dzp + hzm[l] * dzm) / (h[2] * h[2]);
                    force[l] += uc * tgrad;
                }
            }
        }
    }
    Ok(force.map(|f| -f * h[0] * h[1] * h[2] * deps * 0.5 * izmagic))
}

/// V Σ_n ∂κ̄²_n/∂R ρ(u_n) / 4π l_B for atom `n` (APBS `Vpmg_ibForce`), with
/// `twice_density` = 2ρ.
#[allow(clippy::too_many_arguments)]
fn ionic_boundary_force(
    grid: &Grid,
    n: usize,
    atom: &Atom,
    access: &[f64],
    u: &[f64],
    params: &PbParams,
    kbar2: f64,
    twice_density: &(impl Fn(f64) -> f64 + Sync),
    izmagic: f64,
) -> Result<[f64; 3]> {
    if kbar2 < VPMGSMALL || atom.radius <= 0.0 {
        return Ok([0.0; 3]);
    }
    let (lo, up, h) = (grid.origin, grid.upper(), grid.spacing);
    if (0..3).any(|a| atom.position[a] <= lo[a] || atom.position[a] >= up[a]) {
        return invalid(format!(
            "atom {n} at {:?} lies outside the finest grid; its ionic boundary force needs it inside",
            atom.position
        ));
    }
    let irad = params.ion_exclusion_radius();
    let pos = [0, 1, 2].map(|a| atom.position[a] - lo[a]);
    let rtot = irad + atom.radius + params.swin;
    let rtot2 = rtot * rtot;
    // The atom lies inside the box, so p + d > 0.
    let range = |p: f64, d: f64, h: f64, n: usize| {
        let lo = ((p - d) / h).ceil().max(0.0) as usize;
        let hi = ((p + d) / h).floor().min((n - 1) as f64) as usize;
        lo..=hi
    };
    let sq = |x: f64| x * x;
    let grad = LogGradient::new(params.surface, atom, params.swin, irad);
    let mut force = [0.0; 3];
    for i in range(pos[0], rtot + 0.5 * h[0], h[0], grid.dims[0]) {
        let dx2 = sq(pos[0] - h[0] * i as f64);
        let dy = if rtot2 > dx2 {
            (rtot2 - dx2).sqrt() + 0.5 * h[1]
        } else {
            0.5 * h[1]
        };
        for j in range(pos[1], dy, h[1], grid.dims[1]) {
            let dy2 = sq(pos[1] - h[1] * j as f64);
            let dz = if rtot2 > dx2 + dy2 {
                (rtot2 - dx2 - dy2).sqrt() + 0.5 * h[2]
            } else {
                0.5 * h[2]
            };
            for k in range(pos[2], dz, h[2], grid.dims[2]) {
                let dz2 = sq(k as f64 * h[2] - pos[2]);
                if dz2 + dy2 + dx2 <= rtot2 {
                    let point = [
                        i as f64 * h[0] + lo[0],
                        j as f64 * h[1] + lo[1],
                        k as f64 * h[2] + lo[2],
                    ];
                    let tgrad = grad.at(point);
                    let c = grid.index(i, j, k);
                    let fmag = twice_density(u[c]) * access[c];
                    for l in 0..3 {
                        force[l] += kbar2 * fmag * tgrad[l];
                    }
                }
            }
        }
    }
    Ok(force.map(|f| f * 0.5 * h[0] * h[1] * h[2] * izmagic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autogrid::GridSpec;
    use crate::cpu::CpuBackend;
    use crate::params::{BoundaryCondition, Equation, SurfaceMethod};
    use crate::pb::{polar_forces, polar_solvation};

    fn molecule(charge_scale: f64) -> Vec<Atom> {
        // Off the grid lines, so that the trilinear weights are smooth nearby (on a
        // line APBS sets that component of the trilinear force to zero).
        [
            ([0.137, -0.211, 0.093], 0.7, 1.9),
            ([1.613, 0.427, -0.288], -0.9, 1.6),
            ([-0.79, 1.31, 0.673], 0.45, 1.4),
            ([0.41, -1.12, -1.36], -0.25, 0.0), // radius 0: charge only
        ]
        .map(|(position, charge, radius)| Atom {
            position,
            charge: charge * charge_scale,
            radius,
        })
        .to_vec()
    }

    /// One level with zero boundary values: the force is the exact gradient
    /// of the discrete energy, so it equals central differences of it.
    fn check_finite_differences(params: &PbParams, charge_scale: f64, rtol: f64) {
        let be = CpuBackend::new();
        let atoms = molecule(charge_scale);
        // Unequal spacings per axis (0.4, 0.38, 0.42 Å).
        let spec = GridSpec::single([0.31, 0.12, -0.17], [16.0, 15.2, 16.8], [41; 3]);
        let f = polar_forces(&be, &atoms, params, &spec).unwrap();
        let energy = |atoms: &[Atom]| polar_solvation(&be, atoms, params, &spec).unwrap().energy_kcal;
        assert!((energy(&atoms) - f.solvation.energy_kcal).abs() < 1e-12 * f.solvation.energy_kcal.abs());
        let delta = 1e-4;
        let mut worst: f64 = 0.0;
        let scale = f.qf.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        for n in 0..atoms.len() {
            for l in 0..3 {
                let shifted = |d: f64| {
                    let mut a = atoms.clone();
                    a[n].position[l] += d;
                    energy(&a)
                };
                // Central differences, Richardson-extrapolated: O(δ⁴) where the
                // energy is smooth (spl2's switch is C¹: a face within δ of a
                // window edge leaves O(δ)); measured 2e-9 to 4e-9 of the largest qf.
                let central = |d: f64| -(shifted(d) - shifted(-d)) / (2.0 * d);
                let fd = (4.0 * central(0.5 * delta) - central(delta)) / 3.0;
                let total = f.qf[n][l] + f.db[n][l] + f.ib[n][l];
                worst = worst.max((total - fd).abs() / scale);
            }
        }
        eprintln!(
            "{:?}/{:?}/{:?}: worst {worst:e}",
            params.surface, params.equation, params.chgm
        );
        assert!(
            worst < rtol,
            "{:?}/{:?}/{:?}: worst {worst:e}",
            params.surface,
            params.equation,
            params.chgm
        );
        // Every term contributes (ib is small for these weak potentials at the ions).
        let size = |v: &[[f64; 3]]| v.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(
            size(&f.db) > 1e-3 * scale && size(&f.ib) > 1e-6 * scale,
            "db {} ib {}",
            size(&f.db),
            size(&f.ib)
        );
    }

    fn params(surface: SurfaceMethod, chgm: ChargeMethod, equation: Equation) -> PbParams {
        PbParams {
            surface,
            chgm,
            equation,
            pdie: 2.0,
            bcfl: BoundaryCondition::Zero,
            tol: 1e-13,
            ..PbParams::default()
        }
        .with_salt(0.15, 1.2)
    }

    #[test]
    fn charge_field_forces_in_a_linear_potential() {
        // Every charge assignment reproduces a linear potential exactly, so
        // the force is −q ∇u; on a grid line APBS zeroes the trilinear component.
        let grid = Grid::from_spacing([0.0; 3], [0.4, 0.38, 0.42], [33; 3]).unwrap();
        let slope = [0.7, -1.3, 0.4];
        let u: Vec<f64> = (0..grid.len())
            .map(|c| {
                let (i, j, k) = grid.ijk(c);
                let p = grid.position(i, j, k);
                slope[0] * p[0] + slope[1] * p[1] + slope[2] * p[2]
            })
            .collect();
        let off = Atom {
            position: [0.137, -0.211, 0.093],
            charge: -1.5,
            radius: 1.0,
        };
        for chgm in [ChargeMethod::Trilinear, ChargeMethod::Spline2, ChargeMethod::Spline4] {
            let f = charge_field_forces(&grid, &[off], chgm, &u)[0];
            for l in 0..3 {
                assert!((f[l] + off.charge * slope[l]).abs() < 1e-12, "{chgm:?} {l}: {f:?}");
            }
        }
        let on_node = Atom {
            position: grid.position(16, 17, 15),
            ..off
        };
        assert_eq!(
            charge_field_forces(&grid, &[on_node], ChargeMethod::Trilinear, &u)[0],
            [0.0; 3]
        );
        let uncharged = Atom { charge: 0.0, ..off };
        assert_eq!(
            charge_field_forces(&grid, &[uncharged], ChargeMethod::Spline2, &u)[0],
            [0.0; 3]
        );
    }

    #[test]
    fn forces_are_the_gradient_of_the_discrete_energy() {
        for chgm in [ChargeMethod::Trilinear, ChargeMethod::Spline2] {
            check_finite_differences(&params(SurfaceMethod::Spline2, chgm, Equation::Linear), 1.0, 1e-8);
        }
        // spl4 surfaces carry APBS' rounding noise in the maps (up to 2e-7).
        check_finite_differences(
            &params(SurfaceMethod::Spline4, ChargeMethod::Spline2, Equation::Linear),
            1.0,
            5e-6,
        );
    }

    #[test]
    fn nonlinear_forces_are_the_gradient_of_the_functional() {
        // Charges × 6: potentials of several kT/e at the ions, where the
        // Boltzmann terms differ from the linearized ones.
        let mut p = params(SurfaceMethod::Spline2, ChargeMethod::Spline2, Equation::Nonlinear);
        check_finite_differences(&p, 6.0, 1e-8);
        p.equation = Equation::SizeModified;
        p.ion_size = 6.0;
        check_finite_differences(&p, 6.0, 1e-8);
    }
}
