// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vpmg.c: fillcoCoefSpline, fillcoCoefSpline4; vacc.c:
// Vacc_splineAccGradAtomNorm, Vacc_splineAccGradAtomNorm4), Copyright (c) 2010-2022 Battelle
// Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! APBS' spline surfaces (`srfm spl2`, `spl4`): smooth dielectric and
//! ion-accessibility maps, differentiable in the atom positions, which APBS
//! requires for forces.
//!
//! Every atom of radius r has a switching function S(d) of the distance d
//! to its centre that rises from 0 at d = r − w to 1 at d = r + w (w =
//! `swin`). The solvent fraction at a point is H = Π_atoms S, and the face
//! dielectric is ε = pdie + (sdie − pdie)·H. The ion accessibility K is the
//! same product with r + r_ion, and κ̄² = κ̄²_bulk·K.
//!
//! * `spl2`: S is the cubic ¾ s²/w² − ¼ s³/w³ of s = d − r + w (zero slope
//!   at both ends);
//! * `spl4`: S is the septic polynomial in d with zero first three
//!   derivatives at both ends, clamped to [0, 1].
//!
//! The maps follow APBS operation for operation (atoms in order, the
//! products, the guards that stop multiplying once a value is ≤ 1e-12, and
//! `spl4`'s different ways of squaring the distance per map), so they are
//! identical to APBS' `dielx/y/z` and `kappa`. Atoms outside the box of a
//! level are skipped, as APBS skips them.

use crate::grid::Grid;
use crate::molecule::Atom;
use crate::params::{PbParams, SurfaceMethod};
use rayon::prelude::*;

/// APBS `VPMGSMALL`.
const VPMGSMALL: f64 = 1.0e-12;
/// APBS `VSMALL`.
const VSMALL: f64 = 1.0e-9;

/// One atom's switching function S(d).
#[derive(Clone, Copy, Debug)]
enum Switch {
    /// `spl2`: radius r, window w and APBS' 1/w², 1/w³.
    Cubic { r: f64, w: f64, w2i: f64, w3i: f64 },
    /// `spl4`: the polynomial's coefficients c0 … c7.
    Septic([f64; 8]),
}

impl Switch {
    fn new(surface: SurfaceMethod, r: f64, w: f64) -> Self {
        match surface {
            SurfaceMethod::Spline4 => Self::Septic(septic(r, w)),
            _ => Self::Cubic {
                r,
                w,
                w2i: 1.0 / (w * w),
                w3i: 1.0 / (w * w * w),
            },
        }
    }

    /// S at distance `dist` inside the window; `sq` is the square of `dist`
    /// that `spl4` uses (APBS takes dist² for some maps, sqrt(dist²)² for
    /// others).
    #[inline]
    fn value(&self, dist: f64, sq: f64) -> f64 {
        match *self {
            Self::Cubic { r, w, w2i, w3i } => {
                let sm = dist - r + w;
                let sm2 = sm * sm;
                0.75 * sm2 * w2i - 0.25 * sm * sm2 * w3i
            }
            Self::Septic(c) => {
                let sm = dist;
                let sm2 = sq;
                let sm3 = sm2 * sm;
                let sm4 = sm3 * sm;
                let sm5 = sm4 * sm;
                let sm6 = sm5 * sm;
                let sm7 = sm6 * sm;
                let value =
                    c[0] + c[1] * sm + c[2] * sm2 + c[3] * sm3 + c[4] * sm4 + c[5] * sm5 + c[6] * sm6 + c[7] * sm7;
                value.clamp(0.0, 1.0)
            }
        }
    }
}

/// The coefficients of `spl4`'s septic switch over [r − w, r + w]
/// (APBS `fillcoCoefSpline4`, `Vacc_splineAccGradAtomNorm4`).
fn septic(r: f64, w: f64) -> [f64; 8] {
    let b = r - w;
    let e = r + w;
    let e2 = e * e;
    let e3 = e2 * e;
    let e4 = e3 * e;
    let e5 = e4 * e;
    let e6 = e5 * e;
    let e7 = e6 * e;
    let b2 = b * b;
    let b3 = b2 * b;
    let b4 = b3 * b;
    let b5 = b4 * b;
    let b6 = b5 * b;
    let b7 = b6 * b;
    let denom =
        e7 - 7.0 * b * e6 + 21.0 * b2 * e5 - 35.0 * e4 * b3 + 35.0 * e3 * b4 - 21.0 * b5 * e2 + 7.0 * e * b6 - b7;
    [
        b4 * (35.0 * e3 - 21.0 * b * e2 + 7.0 * e * b2 - b3) / denom,
        -140.0 * b3 * e3 / denom,
        210.0 * e2 * b2 * (e + b) / denom,
        -140.0 * e * b * (e2 + 3.0 * b * e + b2) / denom,
        35.0 * (e3 + 9.0 * b * e2 + 9.0 * e * b2 + b3) / denom,
        -84.0 * (e2 + 3.0 * b * e + b2) / denom,
        70.0 * (e + b) / denom,
        -20.0 / denom,
    ]
}

/// The maps of a smooth dielectric (spline surface or `gauss`) on one grid.
#[derive(Clone, Debug)]
pub struct SmoothMaps {
    /// ε on the x, y and z faces (as [`crate::surface::CoefficientMaps`]).
    pub eps: [Vec<f64>; 3],
    /// The ion accessibility K ∈ [0, 1] on the nodes (APBS' `kappa` map
    /// before its ionic-strength mask).
    pub accessibility: Vec<f64>,
}

/// One atom as the maps see it: grid-relative position, squared window
/// limits of both maps, switches and the node range it can reach.
struct Placed {
    pos: [f64; 3],
    /// Dielectric: S = 1 beyond `stot2`, 0 within `sctot2` (squared).
    stot2: f64,
    sctot2: f64,
    diel: Switch,
    /// Ion accessibility, the same with radius r + r_ion.
    itot2: f64,
    ictot2: f64,
    ion: Switch,
    lo: [usize; 3],
    hi: [usize; 3],
}

/// The dielectric and ion-accessibility maps of `params.surface` (`spl2` or
/// `spl4`) on `grid` (APBS `fillcoCoefSpline`, `fillcoCoefSpline4`).
pub fn spline_maps(grid: &Grid, atoms: &[Atom], params: &PbParams) -> SmoothMaps {
    let [nx, ny, _] = grid.dims;
    let h = grid.spacing;
    let (lo, up) = (grid.origin, grid.upper());
    let w = params.swin;
    let irad = params.ion_exclusion_radius();
    let placed: Vec<Placed> = atoms
        .iter()
        .filter(|a| {
            // Atoms off the box are ignored, as APBS ignores them.
            let off = (0..3).any(|ax| a.position[ax] <= lo[ax] || a.position[ax] >= up[ax]);
            !off && a.radius > VPMGSMALL
        })
        .map(|a| {
            let arad = a.radius;
            let pos = [0, 1, 2].map(|ax| a.position[ax] - lo[ax]);
            let itot = irad + arad + w;
            let ictot = (irad + arad - w).max(0.0);
            let stot = arad + w;
            let sctot = (arad - w).max(0.0);
            let rtot = itot.max(stot);
            let mut lo_n = [0; 3];
            let mut hi_n = [0; 3];
            for ax in 0..3 {
                let d = rtot + 0.5 * h[ax];
                lo_n[ax] = ((pos[ax] - d) / h[ax]).floor().max(0.0) as usize;
                hi_n[ax] = ((pos[ax] + d) / h[ax]).ceil().min((grid.dims[ax] - 1) as f64) as usize;
            }
            Placed {
                pos,
                stot2: stot * stot,
                sctot2: sctot * sctot,
                diel: Switch::new(params.surface, arad, w),
                itot2: itot * itot,
                ictot2: ictot * ictot,
                ion: Switch::new(params.surface, irad + arad, w),
                lo: lo_n,
                hi: hi_n,
            }
        })
        .collect();
    let n = grid.len();
    let plane = nx * ny;
    let (mut kappa, mut ex, mut ey, mut ez) = (vec![1.0; n], vec![1.0; n], vec![1.0; n], vec![1.0; n]);
    let septic = params.surface == SurfaceMethod::Spline4;
    kappa
        .par_chunks_mut(plane)
        .zip(ex.par_chunks_mut(plane))
        .zip(ey.par_chunks_mut(plane))
        .zip(ez.par_chunks_mut(plane))
        .enumerate()
        .for_each(|(k, (((kap, epx), epy), epz))| {
            // Per node the atoms come in order, as in APBS' sequential loops.
            for p in placed.iter().filter(|p| p.lo[2] <= k && k <= p.hi[2]) {
                let dz2 = sq(p.pos[2] - k as f64 * h[2]);
                let zf2 = sq(p.pos[2] - (k as f64 + 0.5) * h[2]);
                for i in p.lo[0]..=p.hi[0] {
                    let dx2 = sq(p.pos[0] - h[0] * i as f64);
                    let xf2 = sq(p.pos[0] - (i as f64 + 0.5) * h[0]);
                    for j in p.lo[1]..=p.hi[1] {
                        let dy2 = sq(p.pos[1] - h[1] * j as f64);
                        let c = i + nx * j;
                        if kap[c] > VPMGSMALL {
                            let dist2 = dz2 + dy2 + dx2;
                            if dist2 <= p.ictot2 {
                                kap[c] = 0.0;
                            } else if dist2 < p.itot2 {
                                kap[c] *= p.ion.value(dist2.sqrt(), dist2);
                            }
                        }
                        // spl4 squares sqrt(dist²) on the x and y faces, takes dist² on z.
                        let faces = [
                            (&mut epx[c], dz2 + dy2 + xf2, true),
                            (&mut epy[c], dz2 + dx2 + sq(p.pos[1] - (j as f64 + 0.5) * h[1]), true),
                            (&mut epz[c], dy2 + dx2 + zf2, false),
                        ];
                        for (e, dist2, resquare) in faces {
                            if *e > VPMGSMALL {
                                if dist2 <= p.sctot2 {
                                    *e = 0.0;
                                } else if dist2 < p.stot2 {
                                    let dist = dist2.sqrt();
                                    let s2 = if septic && resquare { dist * dist } else { dist2 };
                                    *e *= p.diel.value(dist, s2);
                                }
                            }
                        }
                    }
                }
            }
        });
    let (pdie, sdie) = (params.pdie, params.sdie);
    let fill = |v: &mut Vec<f64>| v.par_iter_mut().for_each(|e| *e = (sdie - pdie) * *e + pdie);
    fill(&mut ex);
    fill(&mut ey);
    fill(&mut ez);
    SmoothMaps {
        eps: [ex, ey, ez],
        accessibility: kappa,
    }
}

#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

/// The gradient of ln S_a with respect to the position of atom a, at a point
/// (APBS `Vacc_splineAccGradAtomNorm`, `…Norm4`): what the forces need,
/// since ∂H/∂R_a = H·∇ ln S_a.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogGradient {
    /// None for atoms of radius 0, which do not shape the surface.
    centre: Option<[f64; 3]>,
    /// The radius plus the inflation (the ion radius for the ion map).
    arad: f64,
    w: f64,
    switch: Switch,
}

impl LogGradient {
    pub(crate) fn new(surface: SurfaceMethod, atom: &Atom, win: f64, infrad: f64) -> Self {
        let arad = atom.radius + infrad;
        Self {
            centre: (atom.radius > 0.0).then_some(atom.position),
            arad,
            w: win,
            switch: Switch::new(surface, arad, win),
        }
    }

    /// ∇_R ln S at `point` (Å⁻¹); zero outside the window.
    #[inline]
    pub(crate) fn at(&self, point: [f64; 3]) -> [f64; 3] {
        let Some(c) = self.centre else {
            return [0.0; 3];
        };
        let (arad, win) = (self.arad, self.w);
        let dist = (sq(c[0] - point[0]) + sq(c[1] - point[1]) + sq(c[2] - point[2])).sqrt();
        if dist < arad - win
            || dist > arad + win
            || (dist - (arad - win)).abs() < VSMALL
            || (dist - (arad + win)).abs() < VSMALL
        {
            return [0.0; 3];
        }
        let (chi, slope) = match self.switch {
            Switch::Cubic { w2i, w3i, .. } => {
                let sm = dist - arad + win;
                let sm2 = sm * sm;
                (
                    0.75 * sm2 * w2i - 0.25 * sm * sm2 * w3i,
                    1.5 * sm * w2i - 0.75 * sm2 * w3i,
                )
            }
            Switch::Septic(k) => {
                let sm = dist;
                let sm2 = sm * sm;
                let sm3 = sm2 * sm;
                let sm4 = sm3 * sm;
                let sm5 = sm4 * sm;
                let sm6 = sm5 * sm;
                let sm7 = sm6 * sm;
                let chi =
                    k[0] + k[1] * sm + k[2] * sm2 + k[3] * sm3 + k[4] * sm4 + k[5] * sm5 + k[6] * sm6 + k[7] * sm7;
                let slope = k[1]
                    + 2.0 * k[2] * sm
                    + 3.0 * k[3] * sm2
                    + 4.0 * k[4] * sm3
                    + 5.0 * k[5] * sm4
                    + 6.0 * k[6] * sm5
                    + 7.0 * k[7] * sm6;
                if chi <= 0.0 {
                    return [0.0; 3];
                }
                (chi.min(1.0), slope)
            }
        };
        [0, 1, 2].map(|a| -(slope / chi) * ((point[a] - c[a]) / dist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S as APBS evaluates it (0 and 1 outside the window).
    fn switch_at(surface: SurfaceMethod, r: f64, w: f64, d: f64) -> f64 {
        if d <= r - w {
            0.0
        } else if d >= r + w {
            1.0
        } else {
            Switch::new(surface, r, w).value(d, d * d)
        }
    }

    /// S in closed form: the cubic of s = d − r + w, or the septic smoothstep
    /// 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷ of t = (d − r + w)/2w.
    fn exact_switch(surface: SurfaceMethod, r: f64, w: f64, d: f64) -> f64 {
        let s = (d - r + w).clamp(0.0, 2.0 * w);
        match surface {
            SurfaceMethod::Spline4 => {
                let t = s / (2.0 * w);
                t.powi(4) * (35.0 - 84.0 * t + 70.0 * t * t - 20.0 * t.powi(3))
            }
            _ => 0.75 * s * s / (w * w) - 0.25 * s.powi(3) / w.powi(3),
        }
    }

    #[test]
    fn switches_are_the_smoothsteps_apbs_intends() {
        // APBS evaluates spl4 in monomials of d, with coefficients up to 1e7
        // for r = 3.7 Å (an atom plus an ion radius): it loses up to 7 digits.
        for (surface, r, noise) in [
            (SurfaceMethod::Spline2, 1.7, 1e-14),
            (SurfaceMethod::Spline2, 3.7, 1e-14),
            (SurfaceMethod::Spline4, 1.2, 2e-10),
            (SurfaceMethod::Spline4, 1.7, 1e-9),
            (SurfaceMethod::Spline4, 3.7, 4e-7),
        ] {
            let w = 0.3;
            let worst = (0..=1000)
                .map(|i| {
                    let d = r - w + 0.0006 * i as f64;
                    (switch_at(surface, r, w, d) - exact_switch(surface, r, w, d)).abs()
                })
                .fold(0.0, f64::max);
            assert!(worst < noise, "{surface:?} r = {r}: {worst:e}");
            // The closed forms: 0 → ½ → 1 with zero slope at the ends.
            for (d, v) in [(r - w, 0.0), (r, 0.5), (r + w, 1.0)] {
                assert!((exact_switch(surface, r, w, d) - v).abs() < 1e-14);
            }
            let slope = |d: f64| (exact_switch(surface, r, w, d + 1e-7) - exact_switch(surface, r, w, d - 1e-7)) / 2e-7;
            assert!(slope(r - w).abs() < 1e-6 && slope(r + w).abs() < 1e-6, "{surface:?}");
        }
    }

    #[test]
    fn log_gradients_are_the_derivatives_of_ln_s() {
        // spl4's analytic slope carries APBS' rounding noise.
        for (surface, tol) in [(SurfaceMethod::Spline2, 1e-7), (SurfaceMethod::Spline4, 1e-5)] {
            for infrad in [0.0, 2.0] {
                let atom = Atom {
                    position: [0.3, -0.2, 0.1],
                    charge: 0.0,
                    radius: 1.5,
                };
                let r = atom.radius + infrad;
                let g = LogGradient::new(surface, &atom, 0.3, infrad);
                let point = [0.3 + 0.62 * r, -0.2 + 0.55 * r, 0.1 + 0.53 * r]; // inside the window
                let ln_s = |c: [f64; 3]| {
                    let d = ((point[0] - c[0]).powi(2) + (point[1] - c[1]).powi(2) + (point[2] - c[2]).powi(2)).sqrt();
                    exact_switch(surface, r, 0.3, d).ln()
                };
                let analytic = g.at(point);
                for a in 0..3 {
                    let (mut p, mut m) = (atom.position, atom.position);
                    p[a] += 1e-6;
                    m[a] -= 1e-6;
                    let fd = (ln_s(p) - ln_s(m)) / 2e-6;
                    assert!(
                        (analytic[a] - fd).abs() < tol * fd.abs().max(1.0),
                        "{surface:?} {a}: {analytic:?} {fd}"
                    );
                }
                // Zero outside the window and for radius-0 atoms.
                assert_eq!(g.at([0.3 + r + 0.31, -0.2, 0.1]), [0.0; 3]);
                let bare = Atom { radius: 0.0, ..atom };
                assert_eq!(LogGradient::new(surface, &bare, 0.3, infrad).at(point), [0.0; 3]);
            }
        }
    }

    #[test]
    fn maps_are_products_of_the_atoms_switches() {
        let grid = Grid::from_spacing([0.0; 3], [0.4; 3], [33; 3]).unwrap();
        let atoms = [
            Atom {
                position: [0.13, -0.07, 0.05],
                charge: 1.0,
                radius: 2.0,
            },
            Atom {
                position: [1.9, 0.4, -0.3],
                charge: -1.0,
                radius: 1.6,
            },
            // Radius 0 and off the box: ignored.
            Atom {
                position: [0.5, 0.5, 0.5],
                charge: 0.4,
                radius: 0.0,
            },
            Atom {
                position: [9.0, 0.0, 0.0],
                charge: 0.0,
                radius: 3.0,
            },
        ];
        // spl4: APBS' rounding noise, up to 2e-7 for the ion map (r + 2 Å).
        for (surface, tol) in [(SurfaceMethod::Spline2, 1e-12), (SurfaceMethod::Spline4, 1e-6)] {
            let params = PbParams {
                surface,
                pdie: 2.0,
                sdie: 80.0,
                swin: 0.3,
                ..PbParams::default()
            }
            .with_salt(0.1, 2.0);
            let m = spline_maps(&grid, &atoms, &params);
            let h = |p: [f64; 3], infrad: f64| -> f64 {
                atoms[..2]
                    .iter()
                    .map(|a| {
                        let d = ((p[0] - a.position[0]).powi(2)
                            + (p[1] - a.position[1]).powi(2)
                            + (p[2] - a.position[2]).powi(2))
                        .sqrt();
                        switch_at(surface, a.radius + infrad, 0.3, d)
                    })
                    .product()
            };
            let mut partial = 0;
            for c in 0..grid.len() {
                let (i, j, k) = grid.ijk(c);
                let node = grid.position(i, j, k);
                let kk = h(node, 2.0);
                assert!((m.accessibility[c] - kk).abs() < tol, "{surface:?} kappa at {c}");
                let face = [node[0] + 0.2, node[1], node[2]];
                let e = 2.0 + 78.0 * h(face, 0.0);
                assert!((m.eps[0][c] - e).abs() < 100.0 * tol, "{surface:?} eps_x at {c}");
                partial += usize::from(e > 2.0 && e < 80.0);
            }
            assert!(partial > 100, "{surface:?}: the window must cut many faces ({partial})");
        }
    }
}
