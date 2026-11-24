//! The exact gradient of ΔG_polar with respect to the atomic charges and the
//! dielectric constants, by the adjoint method (ROADMAP 4.20).
//!
//! One state's energy is G = ½ wᵀu_L: the energy weights w (the charges
//! spread with `chgm.energy_method()`) times the potential of the finest of
//! the focusing levels ℓ = 0 … L. On every level the interior nodes solve
//! A_ℓ u_ℓ = 4π l_B0 S_ℓ q (S: the spreading weights of `chgm`), with the
//! Dirichlet values g_ℓ on the boundary nodes: the analytic condition on
//! level 0, the coarser level's potential interpolated trilinearly (g_ℓ =
//! P_ℓ u_ℓ₋₁) on the others. Each u_ℓ is linear in q and depends on the
//! dielectric constants through A_ℓ and g_0.
//!
//! Let r_L = ∂G/∂u_L = ½ w. On level ℓ the adjoint λ_ℓ (zero on the
//! boundary) solves A_ℓ λ_ℓ = r_ℓ on the interior nodes, the same operator
//! as the forward solve (A is symmetric). The cotangent of the boundary
//! values is c_ℓ = r_ℓ − A_IBᵀ λ_ℓ, i.e. at a boundary node b
//!
//! ```text
//! c_ℓ,b = r_ℓ,b + Σ_{interior neighbours i} c_a ε_f λ_ℓ,i      (c_a = h_y h_z / h_x, …)
//! ```
//!
//! and it reaches the next coarser level as r_ℓ₋₁ = P_ℓᵀ c_ℓ. Then, for any
//! parameter p,
//!
//! ```text
//! dG/dp = ½ (∂w/∂p)ᵀ u_L + Σ_ℓ λ_ℓᵀ (4π l_B0 S_ℓ ∂q/∂p − (∂A_ℓ/∂p) u_ℓ) + c_0ᵀ ∂g_0/∂p,
//! λᵀ (∂A/∂p) u = Σ_faces c_a (∂ε_f/∂p) (λ_i − λ_j)(u_i − u_j),
//! ```
//!
//! one extra solve per level and state. For the charges this gives
//! dG/dq_i = ½ W_iᵀ u_L + 4π l_B0 Σ_ℓ S_ℓ,iᵀ λ_ℓ + Σ_b c_0,b K_b,i (K: the
//! boundary condition's response to a unit charge); on a single level with
//! `bcfl = zero` and W = S it is the reaction potential W_iᵀ u_L. For the
//! dielectric constants ∂ε_f/∂pdie and ∂ε_f/∂sdie come from the maps built
//! once more with pdie = 1, sdie = 2: ε = pdie + (sdie − pdie) H with the
//! sharp and the spline surfaces (H = ε₁₂ − 1), ε = 1/((1 − θ)/pdie + θ/sdie)
//! with the fractional interface (θ = 2 − 2/ε₁₂). κ̄² does not depend on
//! them; the `mdh` boundary does (1/ε, and κ = √(κ̄²/ε) in the solvated state).

use crate::autogrid::GridSpec;
use crate::backend::Backend;
use crate::boundary::{focus_boundary_values, multipole_factors, Multipoles};
use crate::charge::{atom_energies, spread_charges_sparse_with, spread_charges_with};
use crate::error::{invalid, PbError, Result};
use crate::forces::{charge_field_forces, dielectric_boundary_sums, ionic_boundary_sums};
use crate::grid::Grid;
use crate::molecule::{dist2, Atom};
use crate::nonlinear::IonTerms;
use crate::operator::StencilOperator;
use crate::params::{BoundaryCondition, CoulombMethod, InterfaceMethod, PbParams, SurfaceMethod};
use crate::pb::{
    boundary_medium, check_gaussian_charges, maps_with, outer_boundary_of, solve_with_maps, solver_options, Medium,
    PolarSolvation, Timings,
};
use crate::solver::SolveStats;
use crate::surface::MolecularSurface;
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

/// ΔG_polar with its exact gradient ([`polar_gradient`]).
#[derive(Clone, Debug)]
pub struct PolarGradient {
    /// The energy calculation, equal to [`crate::pb::polar_solvation`]'s.
    pub solvation: PolarSolvation,
    /// dΔG_polar/dq_i of every atom (uncharged ones included), kcal/mol/e.
    pub charges: Vec<f64>,
    /// dΔG_polar/d pdie and dΔG_polar/d sdie in kcal/mol, if asked for.
    pub dielectrics: Option<[f64; 2]>,
    /// dΔG_polar/dr of every atom in kcal/mol/Å (minus the force), if asked for.
    pub positions: Option<Vec<[f64; 3]>>,
    /// dΔG_polar/d radius of every atom in kcal/mol/Å, if asked for.
    pub radii: Option<Vec<f64>>,
}

/// What [`polar_gradient`] differentiates besides the charges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wrt {
    /// pdie and sdie.
    pub dielectric_constants: bool,
    /// Every atom's position (spline surfaces).
    pub positions: bool,
    /// Every atom's radius (spline surfaces).
    pub radii: bool,
}

/// [`crate::pb::polar_solvation`] and its gradient with respect to every
/// atom's charge and, as `wrt` asks, to pdie and sdie, the atoms' positions
/// and their radii: exact for the discrete energy, through every focusing
/// level and the outer boundary condition (`mdh-fast` is differentiated as
/// `mdh`, which it reproduces to 1e-10). Needs the linearized equation and
/// `coulomb = grid`, no membrane and no lateral period. The dielectric
/// derivatives need surface `mol`, `vdw`, `spl2` or `spl4` without per-atom
/// dielectrics, positions and radii a spline surface (`spl2`, `spl4`), and
/// both `bcfl` `zero`, `mdh` or `mdh-fast`. Costs about twice the energy: one
/// adjoint solve per level and state.
pub fn polar_gradient<B: Backend>(
    be: &B,
    atoms: &[Atom],
    params: &PbParams,
    spec: &GridSpec,
    wrt: Wrt,
) -> Result<PolarGradient> {
    check(params, wrt)?;
    params.validate()?;
    check_gaussian_charges(atoms, params)?;
    let levels = spec.levels()?;
    let mut timings = Timings::default();
    // One state at a time: only its levels' potentials are kept.
    let reference = State::solve(be, &levels, atoms, params, Medium::Reference, None, &mut timings)?;
    let dref = reference.gradient(be, atoms, params, Medium::Reference, None, wrt, &mut timings)?;
    let t = Instant::now();
    let surf = MolecularSurface::for_params(atoms, params);
    timings.maps += t.elapsed().as_secs_f64();
    let solvated = State::solve(be, &levels, atoms, params, Medium::Solvated, Some(&surf), &mut timings)?;
    let dsolv = solvated.gradient(be, atoms, params, Medium::Solvated, Some(&surf), wrt, &mut timings)?;
    let kt = kt_in_kcal_per_mol(params.temperature);
    let energy_kt = solvated.energy_kt - reference.energy_kt;
    let solvation = PolarSolvation {
        energy_kcal: energy_kt * kt,
        energy_kt,
        solvated_kt: solvated.energy_kt,
        reference_kt: reference.energy_kt,
        stats_solvated: solvated.stats,
        stats_reference: reference.stats,
        grid: spec.clone(),
        timings,
        atom_energies_kcal: Some(
            solvated
                .atom_energies
                .iter()
                .zip(&reference.atom_energies)
                .map(|(s, r)| (s - r) * kt)
                .collect(),
        ),
        membrane_potential_kcal: 0.0,
    };
    Ok(PolarGradient {
        solvation,
        charges: dsolv
            .charges
            .iter()
            .zip(&dref.charges)
            .map(|(s, r)| (s - r) * kt)
            .collect(),
        dielectrics: wrt
            .dielectric_constants
            .then(|| [0, 1].map(|p| (dsolv.dielectrics[p] - dref.dielectrics[p]) * kt)),
        positions: wrt.positions.then(|| {
            dsolv
                .positions
                .iter()
                .zip(&dref.positions)
                .map(|(s, r)| [0, 1, 2].map(|l| (s[l] - r[l]) * kt))
                .collect()
        }),
        radii: wrt
            .radii
            .then(|| dsolv.radii.iter().zip(&dref.radii).map(|(s, r)| (s - r) * kt).collect()),
    })
}

/// The options [`polar_gradient`] supports.
fn check(params: &PbParams, wrt: Wrt) -> Result<()> {
    if params.coulomb == CoulombMethod::Analytic {
        return invalid("gradients need coulomb 'grid'");
    }
    if IonTerms::new(params).is_some() {
        return invalid("gradients need the linearized equation ('lpbe'; 'npbe' and 'smpbe' only without ions)");
    }
    if params.membrane.is_some() {
        return invalid("gradients with a membrane are not available");
    }
    if params.period.is_some() {
        return invalid("gradients with periodic lateral boundaries are not available");
    }
    if wrt.positions || wrt.radii {
        if !params.surface.is_spline() {
            return invalid(format!(
                "derivatives with respect to positions and radii need surface 'spl2' or 'spl4' (smooth in the \
                 atoms), not '{}'",
                params.surface.as_str()
            ));
        }
        if params.bcfl == BoundaryCondition::Sdh {
            return invalid("derivatives with respect to positions and radii need bcfl 'zero', 'mdh' or 'mdh-fast'");
        }
    }
    if wrt.dielectric_constants {
        if params.atom_pdie.is_some() {
            return invalid("dielectric derivatives with per-atom dielectrics are not available");
        }
        if !matches!(
            params.surface,
            SurfaceMethod::Mol | SurfaceMethod::Vdw | SurfaceMethod::Spline2 | SurfaceMethod::Spline4
        ) {
            return invalid(format!(
                "dielectric derivatives need surface 'mol', 'vdw', 'spl2' or 'spl4', not '{}'",
                params.surface.as_str()
            ));
        }
        if params.bcfl == BoundaryCondition::Sdh {
            return invalid("dielectric derivatives need bcfl 'zero', 'mdh' or 'mdh-fast', not 'sdh'");
        }
    }
    Ok(())
}

/// One focusing level of a state: its grid and potential u (kT/e, every node).
struct Level {
    grid: Grid,
    u: Vec<f64>,
}

/// A state solved on every focusing level, with its energy.
struct State {
    levels: Vec<Level>,
    stats: Vec<SolveStats>,
    energy_kt: f64,
    atom_energies: Vec<f64>,
}

/// One state's gradient in kT per unit.
struct StateGradient {
    charges: Vec<f64>,
    /// d/d pdie, d/d sdie.
    dielectrics: [f64; 2],
    positions: Vec<[f64; 3]>,
    radii: Vec<f64>,
}

/// g on the boundary nodes (in `boundary_indices` order) as a full-length vector.
fn embed(grid: &Grid, g: &[f64]) -> Vec<f64> {
    let mut full = vec![0.0; grid.len()];
    for (c, v) in grid.boundary_indices().into_iter().zip(g) {
        full[c] = *v;
    }
    full
}

impl State {
    /// The focusing sequence of `medium` with the arithmetic of
    /// [`crate::pb::polar_solvation`] (same maps, charges, boundary values and
    /// solver, energy summed over the charged nodes in the same order): the
    /// same energy bit for bit.
    fn solve<B: Backend>(
        be: &B,
        levels: &[Grid],
        atoms: &[Atom],
        params: &PbParams,
        medium: Medium,
        surf: Option<&MolecularSurface>,
        timings: &mut Timings,
    ) -> Result<Self> {
        let opts = solver_options(params);
        let mut solved: Vec<Level> = Vec::with_capacity(levels.len());
        let mut stats = Vec::with_capacity(levels.len());
        for grid in levels {
            let t = Instant::now();
            let g = match solved.last() {
                None => outer_boundary_of(levels, atoms, params, medium),
                Some(coarse) => focus_boundary_values(grid, &coarse.grid, |c| coarse.u[c])?,
            };
            let g = embed(grid, &g);
            timings.boundary += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let maps = maps_with(grid, atoms, params, surf);
            timings.maps += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let q = spread_charges_with(grid, atoms, params.chgm)?;
            timings.assembly += t.elapsed().as_secs_f64();
            let (u, s) = solve_with_maps(be, grid, maps, &q, &g, params.temperature, opts)?;
            timings.setup += s.setup_seconds;
            timings.solve += s.solve_seconds;
            stats.push(s);
            solved.push(Level { grid: grid.clone(), u });
        }
        let t = Instant::now();
        let finest = solved.last().expect("at least one level");
        let method = params.chgm.energy_method();
        let (nodes, charges) = spread_charges_sparse_with(&finest.grid, atoms, method)?;
        let u: Vec<f64> = nodes.iter().map(|&c| finest.u[c]).collect();
        let energy_kt = 0.5 * charges.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>();
        let atom_energies = atom_energies(&finest.grid, atoms, method, |c| finest.u[c])?;
        timings.energy += t.elapsed().as_secs_f64();
        Ok(Self {
            levels: solved,
            stats,
            energy_kt,
            atom_energies,
        })
    }

    /// The adjoint pass, finest level first (module documentation).
    #[allow(clippy::too_many_arguments)]
    fn gradient<B: Backend>(
        &self,
        be: &B,
        atoms: &[Atom],
        params: &PbParams,
        medium: Medium,
        surf: Option<&MolecularSurface>,
        wrt: Wrt,
        timings: &mut Timings,
    ) -> Result<StateGradient> {
        let opts = solver_options(params);
        let source = 4.0 * PI * bjerrum_length_vacuum(params.temperature);
        // W_iᵀ f for every atom: the weights of a unit charge at its position.
        let unit: Vec<Atom> = atoms.iter().map(|a| Atom { charge: 1.0, ..*a }).collect();
        let weighted = |grid: &Grid, method, f: &[f64]| -> Result<Vec<f64>> {
            Ok(atom_energies(grid, &unit, method, |c| f[c])?
                .into_iter()
                .map(|x| 2.0 * x)
                .collect())
        };
        let t = Instant::now();
        let finest = self.levels.last().expect("at least one level");
        let method = params.chgm.energy_method();
        let mut out = StateGradient {
            charges: weighted(&finest.grid, method, &finest.u)?
                .into_iter()
                .map(|x| 0.5 * x)
                .collect(),
            dielectrics: [0.0; 2],
            positions: vec![[0.0; 3]; atoms.len()],
            radii: vec![0.0; atoms.len()],
        };
        if wrt.positions {
            // ½ (∂w/∂r)ᵀ u_L: the energy weights move with the atoms
            // (`charge_field_forces` is −q ∇(Wᵀu)).
            for (p, f) in out
                .positions
                .iter_mut()
                .zip(charge_field_forces(&finest.grid, atoms, method, &finest.u))
            {
                (0..3).for_each(|l| p[l] -= 0.5 * f[l]);
            }
        }
        let mut r: Vec<f64> = spread_charges_with(&finest.grid, atoms, method)?
            .into_iter()
            .map(|w| 0.5 * w)
            .collect();
        let geometry = (wrt.positions || wrt.radii) && surf.is_some();
        timings.energy += t.elapsed().as_secs_f64();
        for (index, level) in self.levels.iter().enumerate().rev() {
            let grid = &level.grid;
            let t = Instant::now();
            let maps = maps_with(grid, atoms, params, surf);
            let eps = [maps.eps_x.clone(), maps.eps_y.clone(), maps.eps_z.clone()];
            let kappa_sq = geometry.then(|| maps.kappa_sq.clone());
            timings.maps += t.elapsed().as_secs_f64();
            let rhs: Vec<f64> = r.iter().map(|v| v / source).collect();
            let (mut lambda, s) =
                solve_with_maps(be, grid, maps, &rhs, &vec![0.0; grid.len()], params.temperature, opts)?;
            timings.setup += s.setup_seconds;
            timings.solve += s.solve_seconds;
            let t = Instant::now();
            let boundary = grid.boundary_indices();
            for &b in &boundary {
                lambda[b] = 0.0;
            }
            for (d, w) in out.charges.iter_mut().zip(weighted(grid, params.chgm, &lambda)?) {
                *d += source * w;
            }
            if wrt.dielectric_constants {
                let face = face_derivatives(grid, atoms, params, surf, &eps);
                let [p, s] = face_sums(grid, &face, &lambda, &level.u);
                out.dielectrics[0] -= p;
                out.dielectrics[1] -= s;
            }
            if wrt.positions {
                // 4π l_B0 λᵀ ∂(S q)/∂r: the spread charges move with the atoms.
                for (p, f) in out
                    .positions
                    .iter_mut()
                    .zip(charge_field_forces(grid, atoms, params.chgm, &lambda))
                {
                    (0..3).for_each(|l| p[l] -= source * f[l]);
                }
            }
            if let Some(kappa_sq) = &kappa_sq {
                let terms = map_terms(grid, atoms, params, &eps, kappa_sq, &lambda, &level.u)?;
                for ((p, r), d) in out.positions.iter_mut().zip(out.radii.iter_mut()).zip(&terms) {
                    (0..3).for_each(|l| p[l] += d[l]);
                    *r += d[3];
                }
            }
            let c = boundary_cotangent(grid, &boundary, &eps, &lambda, &r);
            match index {
                0 => outer_boundary_terms(grid, &boundary, atoms, params, medium, &c, &mut out),
                _ => r = transposed_focus(grid, &boundary, &self.levels[index - 1].grid, &c)?,
            }
            timings.energy += t.elapsed().as_secs_f64();
        }
        if !wrt.positions {
            out.positions.clear();
        }
        if !wrt.radii {
            out.radii.clear();
        }
        Ok(out)
    }
}

/// −λᵀ(∂A/∂R)u on one level of the solvated state of a spline surface, for
/// every atom: its position (0–2) and radius (3). The dielectric and ionic
/// boundary sums of [`crate::forces`] with the adjoint λ as the first field;
/// ∂ε_f = (sdie − pdie) ∂H_f and ∂κ̄²_n = κ̄² ∂K_n.
#[allow(clippy::too_many_arguments)]
fn map_terms(
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    eps: &[Vec<f64>; 3],
    kappa_sq: &[f64],
    lambda: &[f64],
    u: &[f64],
) -> Result<Vec<[f64; 4]>> {
    let vol = grid.cell_volume();
    let deps = params.sdie - params.pdie;
    let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
    let access: Vec<f64> = if kbar2 > 0.0 {
        kappa_sq.par_iter().map(|k| k / kbar2).collect()
    } else {
        Vec::new()
    };
    let weight = |c: usize| lambda[c] * u[c];
    atoms
        .par_iter()
        .enumerate()
        .map(|(n, a)| {
            let d = dielectric_boundary_sums(grid, n, a, eps, lambda, u, params)?;
            let i = if kbar2 > 0.0 {
                ionic_boundary_sums(grid, n, a, &access, &weight, params, kbar2)?
            } else {
                [0.0; 4]
            };
            Ok([0, 1, 2, 3].map(|l| vol * (deps * d[l] - i[l])))
        })
        .collect()
}

/// ∂ε_f/∂pdie and ∂ε_f/∂sdie of every face, per axis.
enum FaceDerivatives {
    /// The reference state: ε = pdie everywhere.
    Uniform,
    /// ε = pdie + (sdie − pdie) H: H per face.
    Linear([Vec<f64>; 3]),
    /// ε = 1/((1 − θ)/pdie + θ/sdie): (∂ε/∂pdie, ∂ε/∂sdie) per face.
    Harmonic([Vec<[f64; 2]>; 3]),
}

impl FaceDerivatives {
    #[inline]
    fn at(&self, axis: usize, c: usize) -> [f64; 2] {
        match self {
            Self::Uniform => [1.0, 0.0],
            Self::Linear(h) => [1.0 - h[axis][c], h[axis][c]],
            Self::Harmonic(d) => d[axis][c],
        }
    }
}

/// The face derivatives of the state whose surface is `surf` (None: the
/// reference state), from its maps `eps` and the same maps built with
/// pdie = 1, sdie = 2.
fn face_derivatives(
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    surf: Option<&MolecularSurface>,
    eps: &[Vec<f64>; 3],
) -> FaceDerivatives {
    let Some(surf) = surf else {
        return FaceDerivatives::Uniform;
    };
    let unit = PbParams {
        pdie: 1.0,
        sdie: 2.0,
        ..params.clone()
    };
    let m = maps_with(grid, atoms, &unit, Some(surf));
    let unit_eps = [m.eps_x, m.eps_y, m.eps_z];
    if params.interface == InterfaceMethod::Fractional {
        let (p, s) = (params.pdie, params.sdie);
        FaceDerivatives::Harmonic([0, 1, 2].map(|a| {
            unit_eps[a]
                .par_iter()
                .zip(&eps[a])
                .map(|(e12, e)| {
                    let theta = 2.0 - 2.0 / e12;
                    [e * e * (1.0 - theta) / (p * p), e * e * theta / (s * s)]
                })
                .collect()
        }))
    } else {
        FaceDerivatives::Linear(unit_eps.map(|v| v.into_par_iter().map(|e12| e12 - 1.0).collect()))
    }
}

/// Σ_faces c_a (∂ε_f/∂p)(λ_i − λ_j)(u_i − u_j) for p = pdie, sdie, over the
/// faces between two nodes of the grid; summed per z plane, then over the
/// planes in order (deterministic).
fn face_sums(grid: &Grid, face: &FaceDerivatives, lambda: &[f64], u: &[f64]) -> [f64; 2] {
    let coef = StencilOperator::face_coefficients(grid);
    let [nx, ny, nz] = grid.dims;
    let stride = [1, nx, nx * ny];
    let planes: Vec<[f64; 2]> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let mut acc = [0.0; 2];
            for j in 0..ny {
                for i in 0..nx {
                    let c = i + nx * (j + ny * k);
                    let pos = [i, j, k];
                    for a in 0..3 {
                        if pos[a] + 1 >= grid.dims[a] {
                            continue;
                        }
                        let n = c + stride[a];
                        let product = coef[a] * (lambda[c] - lambda[n]) * (u[c] - u[n]);
                        let d = face.at(a, c);
                        acc[0] += d[0] * product;
                        acc[1] += d[1] * product;
                    }
                }
            }
            acc
        })
        .collect();
    planes.iter().fold([0.0; 2], |acc, p| [acc[0] + p[0], acc[1] + p[1]])
}

/// c_b = r_b + Σ_{neighbours i} c_a ε_f λ_i at every boundary node b (λ is
/// zero on the boundary), in `boundary` order.
fn boundary_cotangent(grid: &Grid, boundary: &[usize], eps: &[Vec<f64>; 3], lambda: &[f64], r: &[f64]) -> Vec<f64> {
    let coef = StencilOperator::face_coefficients(grid);
    let [nx, ny, _] = grid.dims;
    let stride = [1, nx, nx * ny];
    boundary
        .par_iter()
        .map(|&b| {
            let (i, j, k) = grid.ijk(b);
            let pos = [i, j, k];
            let mut s = r[b];
            for a in 0..3 {
                if pos[a] + 1 < grid.dims[a] {
                    s += coef[a] * eps[a][b] * lambda[b + stride[a]];
                }
                if pos[a] > 0 {
                    s += coef[a] * eps[a][b - stride[a]] * lambda[b - stride[a]];
                }
            }
            s
        })
        .collect()
}

/// Pᵀ c: the fine level's boundary cotangent spread back onto the coarse
/// nodes its focused boundary values were interpolated from.
fn transposed_focus(fine: &Grid, boundary: &[usize], coarse: &Grid, c: &[f64]) -> Result<Vec<f64>> {
    let mut r = vec![0.0; coarse.len()];
    for (&b, cb) in boundary.iter().zip(c) {
        let (i, j, k) = fine.ijk(b);
        let (idx, w) = coarse
            .trilinear(fine.position(i, j, k))
            .ok_or_else(|| PbError::InvalidInput("fine grid is not contained in the coarse grid".into()))?;
        for n in 0..8 {
            r[idx[n]] += cb * w[n];
        }
    }
    Ok(r)
}

/// c_0ᵀ ∂g_0/∂p of the analytic boundary condition on the outermost level:
/// for every charge, ε (pdie in the reference state, sdie in the solvated
/// one) and, with `mdh`, every atom's position and radius.
fn outer_boundary_terms(
    grid: &Grid,
    boundary: &[usize],
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
    c: &[f64],
    out: &mut StateGradient,
) {
    let m = boundary_medium(params, medium);
    let points: Vec<[f64; 3]> = boundary
        .iter()
        .map(|&b| {
            let (i, j, k) = grid.ijk(b);
            grid.position(i, j, k)
        })
        .collect();
    match params.bcfl {
        BoundaryCondition::Zero => {}
        BoundaryCondition::Mdh | BoundaryCondition::MdhFast => {
            // shortcut: a direct sum over atoms × boundary nodes (as `mdh` itself); a
            // transposed `mdh-fast` would make it cheaper for very large proteins.
            let k = m.kappa;
            let sums: Vec<[f64; 6]> = atoms
                .par_iter()
                .map(|a| {
                    let mut s = [0.0; 6];
                    for (p, cb) in points.iter().zip(c) {
                        let d = dist2(*p, a.position).sqrt();
                        // The response g to a unit charge and its derivatives in ε (κ =
                        // √(κ̄²/ε)), in the distance and in the atom's radius.
                        let (g, dg, dd, da) = if k > 0.0 {
                            let g = m.lb0 / m.eps * (-k * (d - a.radius)).exp() / ((1.0 + k * a.radius) * d);
                            let slope = (d - a.radius) + a.radius / (1.0 + k * a.radius);
                            (
                                g,
                                g / m.eps * (0.5 * k * slope - 1.0),
                                -g * (k + 1.0 / d),
                                g * k * k * a.radius / (1.0 + k * a.radius),
                            )
                        } else {
                            let g = m.lb0 / (m.eps * d);
                            (g, -g / m.eps, -g / d, 0.0)
                        };
                        s[0] += cb * g;
                        s[1] += cb * dg;
                        for l in 0..3 {
                            s[2 + l] += cb * dd * (a.position[l] - p[l]) / d;
                        }
                        s[5] += cb * da;
                    }
                    s
                })
                .collect();
            for (d, s) in out.charges.iter_mut().zip(&sums) {
                *d += s[0];
            }
            let de: f64 = atoms.iter().zip(&sums).map(|(a, s)| a.charge * s[1]).sum();
            out.dielectrics[if medium == Medium::Reference { 0 } else { 1 }] += de;
            for (n, (a, s)) in atoms.iter().zip(&sums).enumerate() {
                if let Some(p) = out.positions.get_mut(n) {
                    (0..3).for_each(|l| p[l] += a.charge * s[2 + l]);
                }
                if let Some(r) = out.radii.get_mut(n) {
                    *r += a.charge * s[5];
                }
            }
        }
        BoundaryCondition::Sdh => {
            // ∂g/∂q_i of the multipole condition: its charge, dipole and
            // quadrupole are linear in q; centre and radius do not depend on it.
            let mp = Multipoles::of(atoms);
            let mut t0 = 0.0;
            let mut t1 = [0.0; 3];
            let mut t2 = [[0.0; 3]; 3];
            for (p, cb) in points.iter().zip(c) {
                let x = [0, 1, 2].map(|a| p[a] - mp.center[a]);
                let t = multipole_factors((x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt(), &m, mp.radius);
                t0 += cb * t[0];
                for a in 0..3 {
                    t1[a] += cb * x[a] * t[1];
                    for b in 0..3 {
                        t2[a][b] += cb * x[a] * x[b] * t[2];
                    }
                }
            }
            for (d, at) in out.charges.iter_mut().zip(atoms) {
                let r = [0, 1, 2].map(|a| at.position[a] - mp.center[a]);
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                let mut g = t0;
                for a in 0..3 {
                    g -= r[a] * t1[a];
                    for b in 0..3 {
                        let trace = if a == b { r2 / 3.0 } else { 0.0 };
                        g += 0.5 * (r[a] * r[b] - trace) * t2[a][b];
                    }
                }
                *d += m.lb0 * g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::params::{ChargeMethod, Equation};
    use crate::pb::{polar_solvation, solve_focused};

    fn atoms() -> Vec<Atom> {
        let atom = |position, charge, radius| Atom {
            position,
            charge,
            radius,
        };
        vec![
            atom([0.3, -0.2, 0.1], 0.8, 1.9),
            atom([2.1, 0.9, -0.4], -0.55, 1.6),
            atom([-1.4, 1.7, 0.8], 0.0, 1.5),
            atom([0.6, -2.2, 1.6], -0.45, 1.7),
            atom([-0.9, -0.6, -1.9], 0.3, 1.2),
        ]
    }

    /// Three focusing levels.
    fn focused() -> GridSpec {
        GridSpec {
            center: [0.1, 0.0, 0.05],
            dims: [33, 33, 33],
            coarse_length: [44.0, 40.0, 46.0],
            fine_length: [15.0, 14.0, 16.0],
        }
    }

    fn single() -> GridSpec {
        GridSpec::single([0.1, 0.0, 0.05], [17.0, 16.0, 18.0], [41, 41, 41])
    }

    fn params() -> PbParams {
        PbParams {
            pdie: 2.0,
            tol: 1e-12,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0)
    }

    fn energy(atoms: &[Atom], params: &PbParams, spec: &GridSpec) -> f64 {
        polar_solvation(&CpuBackend::new(), atoms, params, spec)
            .unwrap()
            .energy_kcal
    }

    fn gradient(atoms: &[Atom], params: &PbParams, spec: &GridSpec, dielectrics: bool) -> PolarGradient {
        let wrt = Wrt {
            dielectric_constants: dielectrics,
            ..Wrt::default()
        };
        polar_gradient(&CpuBackend::new(), atoms, params, spec, wrt).unwrap()
    }

    fn geometry(atoms: &[Atom], params: &PbParams, spec: &GridSpec) -> PolarGradient {
        let wrt = Wrt {
            positions: true,
            radii: true,
            ..Wrt::default()
        };
        polar_gradient(&CpuBackend::new(), atoms, params, spec, wrt).unwrap()
    }

    /// The configurations the finite-difference tests run: surfaces, interfaces,
    /// boundary conditions and charge methods.
    fn configurations() -> Vec<(&'static str, PbParams, GridSpec)> {
        let p = params();
        vec![
            ("mol mdh focused", p.clone(), focused()),
            (
                "mol fractional mdh-fast focused",
                PbParams {
                    interface: InterfaceMethod::Fractional,
                    bcfl: BoundaryCondition::MdhFast,
                    ..p.clone()
                },
                focused(),
            ),
            (
                "spl4 charges spl2 surface zero single",
                PbParams {
                    surface: SurfaceMethod::Spline2,
                    chgm: ChargeMethod::Spline4,
                    bcfl: BoundaryCondition::Zero,
                    ..p.clone()
                },
                single(),
            ),
            (
                "vdw no ions mdh focused",
                PbParams {
                    surface: SurfaceMethod::Vdw,
                    chgm: ChargeMethod::Spline2,
                    ions: Vec::new(),
                    ..p.clone()
                },
                focused(),
            ),
        ]
    }

    #[test]
    fn the_energy_is_that_of_polar_solvation_bit_for_bit() {
        let atoms = atoms();
        for (name, params, spec) in configurations() {
            let want = polar_solvation(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
            let got = gradient(&atoms, &params, &spec, false).solvation;
            assert_eq!(got.energy_kt.to_bits(), want.energy_kt.to_bits(), "{name}");
            assert_eq!(got.solvated_kt.to_bits(), want.solvated_kt.to_bits(), "{name}");
            assert_eq!(got.atom_energies_kcal, want.atom_energies_kcal, "{name}");
        }
    }

    /// ΔG_polar is quadratic in the charges: central differences are exact
    /// up to rounding and the solver tolerance.
    #[test]
    fn charge_derivatives_equal_central_differences() {
        let atoms = atoms();
        let mut configurations = configurations();
        configurations.push((
            "mol sdh focused",
            PbParams {
                bcfl: BoundaryCondition::Sdh,
                ..params()
            },
            focused(),
        ));
        configurations.push((
            "gauss mdh focused",
            PbParams {
                surface: SurfaceMethod::Gaussian,
                ..params()
            },
            focused(),
        ));
        for (name, params, spec) in configurations {
            let g = gradient(&atoms, &params, &spec, false);
            let scale = g.charges.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for i in 0..atoms.len() {
                let h = 1e-3;
                let shifted = |d: f64| {
                    let mut a = atoms.clone();
                    a[i].charge += d;
                    energy(&a, &params, &spec)
                };
                let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
                assert!(
                    (g.charges[i] - fd).abs() < 1e-7 * scale,
                    "{name}, atom {i}: adjoint {} vs central difference {fd}",
                    g.charges[i]
                );
            }
        }
    }

    /// G is homogeneous of degree 2 in the charges: Σ q_i ∂ΔG/∂q_i = 2 ΔG.
    #[test]
    fn charge_derivatives_satisfy_euler() {
        let atoms = atoms();
        for (name, params, spec) in configurations() {
            let g = gradient(&atoms, &params, &spec, false);
            let euler: f64 = atoms.iter().zip(&g.charges).map(|(a, d)| a.charge * d).sum();
            let e = g.solvation.energy_kcal;
            assert!(
                (euler - 2.0 * e).abs() < 1e-9 * e.abs(),
                "{name}: {euler} vs {}",
                2.0 * e
            );
        }
    }

    /// On one level with zero boundary values and W = S the derivative is the
    /// reaction potential at the atom.
    #[test]
    fn on_one_level_the_charge_derivative_is_the_reaction_potential() {
        let atoms = atoms();
        let params = PbParams {
            bcfl: BoundaryCondition::Zero,
            ..params()
        };
        let spec = single();
        let g = gradient(&atoms, &params, &spec, false);
        let be = CpuBackend::new();
        let s = solve_focused(&be, &spec, &atoms, &params, Medium::Solvated).unwrap();
        let r = solve_focused(&be, &spec, &atoms, &params, Medium::Reference).unwrap();
        let kt = kt_in_kcal_per_mol(params.temperature);
        for (i, a) in atoms.iter().enumerate() {
            let phi = |u: &[f64]| s.grid.interpolate(u, a.position).unwrap();
            let reaction = (phi(&s.potential) - phi(&r.potential)) * kt;
            assert!(
                (g.charges[i] - reaction).abs() < 1e-9 * reaction.abs().max(1.0),
                "atom {i}: {} vs reaction potential {reaction}",
                g.charges[i]
            );
        }
    }

    #[test]
    fn dielectric_derivatives_equal_central_differences() {
        let atoms = atoms();
        for (name, params, spec) in configurations() {
            let g = gradient(&atoms, &params, &spec, true);
            let d = g.dielectrics.unwrap();
            for (which, value) in [(0, params.pdie), (1, params.sdie)] {
                let h = 1e-4 * value;
                let shifted = |s: f64| {
                    let mut p = params.clone();
                    if which == 0 {
                        p.pdie += s;
                    } else {
                        p.sdie += s;
                    }
                    energy(&atoms, &p, &spec)
                };
                let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
                assert!(
                    (d[which] - fd).abs() < 1e-6 * fd.abs() + 1e-9,
                    "{name}, {}: adjoint {} vs central difference {fd}",
                    ["pdie", "sdie"][which],
                    d[which]
                );
            }
        }
    }

    /// Without ions G scales as 1/ε when pdie and sdie scale together:
    /// pdie ∂ΔG/∂pdie + sdie ∂ΔG/∂sdie = −ΔG.
    #[test]
    fn dielectric_derivatives_satisfy_euler_without_ions() {
        let atoms = atoms();
        for (name, params, spec) in configurations() {
            let params = PbParams {
                ions: Vec::new(),
                ..params
            };
            let g = gradient(&atoms, &params, &spec, true);
            let [dp, ds] = g.dielectrics.unwrap();
            let e = g.solvation.energy_kcal;
            let euler = params.pdie * dp + params.sdie * ds;
            assert!((euler + e).abs() < 1e-9 * e.abs(), "{name}: {euler} vs {}", -e);
        }
    }

    /// Positions and radii on the spline surfaces, through three focusing
    /// levels and `mdh` with ions, against central differences
    /// (Richardson-extrapolated, as the force tests: the spl2 switch is C¹).
    #[test]
    fn position_and_radius_derivatives_equal_central_differences() {
        let atoms = atoms();
        for (surface, chgm, rtol) in [
            (SurfaceMethod::Spline2, ChargeMethod::Spline2, 1e-7),
            (SurfaceMethod::Spline2, ChargeMethod::Trilinear, 1e-7),
            // spl4 maps carry APBS' rounding noise (up to 2e-7).
            (SurfaceMethod::Spline4, ChargeMethod::Spline4, 5e-6),
        ] {
            let params = PbParams {
                surface,
                chgm,
                ..params()
            };
            let spec = focused();
            let g = geometry(&atoms, &params, &spec);
            let (dx, dr) = (g.positions.unwrap(), g.radii.unwrap());
            let scale = dx.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            let delta = 1e-4;
            let richardson = |shifted: &dyn Fn(f64) -> f64| {
                let central = |d: f64| (shifted(d) - shifted(-d)) / (2.0 * d);
                (4.0 * central(0.5 * delta) - central(delta)) / 3.0
            };
            for (n, adjoint) in dx.iter().enumerate() {
                for (l, &adjoint) in adjoint.iter().enumerate() {
                    let fd = richardson(&|d: f64| {
                        let mut a = atoms.clone();
                        a[n].position[l] += d;
                        energy(&a, &params, &spec)
                    });
                    assert!(
                        (adjoint - fd).abs() < rtol * scale,
                        "{surface:?}/{chgm:?} atom {n} axis {l}: adjoint {adjoint} vs {fd}"
                    );
                }
                let fd = richardson(&|d: f64| {
                    let mut a = atoms.clone();
                    a[n].radius += d;
                    energy(&a, &params, &spec)
                });
                assert!(
                    (dr[n] - fd).abs() < rtol * scale,
                    "{surface:?}/{chgm:?} atom {n} radius: adjoint {} vs {fd}",
                    dr[n]
                );
            }
        }
    }

    /// On one level with zero boundary values and W = S the position
    /// derivative is minus APBS' force, which is exact there.
    #[test]
    fn on_one_level_the_position_derivative_is_minus_the_force() {
        let atoms = atoms();
        let params = PbParams {
            surface: SurfaceMethod::Spline2,
            chgm: ChargeMethod::Spline2,
            bcfl: BoundaryCondition::Zero,
            ..params()
        };
        let spec = single();
        let g = geometry(&atoms, &params, &spec).positions.unwrap();
        let f = crate::pb::polar_forces(&CpuBackend::new(), &atoms, &params, &spec).unwrap();
        let scale = g.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        for (n, g) in g.iter().enumerate() {
            for (l, &g) in g.iter().enumerate() {
                let force = f.qf[n][l] + f.db[n][l] + f.ib[n][l];
                assert!(
                    (g + force).abs() < 1e-9 * scale,
                    "atom {n} axis {l}: {g} vs force {force}"
                );
            }
        }
    }

    #[test]
    fn unsupported_options_are_refused() {
        let atoms = atoms();
        let spec = focused();
        let refused = |params: PbParams, dielectrics: bool| {
            let wrt = Wrt {
                dielectric_constants: dielectrics,
                ..Wrt::default()
            };
            polar_gradient(&CpuBackend::new(), &atoms, &params, &spec, wrt).is_err()
        };
        let geometry_refused = |params: PbParams| {
            let wrt = Wrt {
                positions: true,
                ..Wrt::default()
            };
            polar_gradient(&CpuBackend::new(), &atoms, &params, &spec, wrt).is_err()
        };
        for surface in [SurfaceMethod::Mol, SurfaceMethod::Vdw, SurfaceMethod::Gaussian] {
            assert!(geometry_refused(PbParams { surface, ..params() }));
        }
        assert!(geometry_refused(PbParams {
            surface: SurfaceMethod::Spline2,
            bcfl: BoundaryCondition::Sdh,
            ..params()
        }));
        assert!(!geometry_refused(PbParams {
            surface: SurfaceMethod::Spline2,
            ..params()
        }));
        assert!(refused(
            PbParams {
                coulomb: CoulombMethod::Analytic,
                ..params()
            },
            false
        ));
        assert!(refused(
            PbParams {
                equation: Equation::Nonlinear,
                ..params()
            },
            false
        ));
        assert!(refused(
            PbParams {
                period: Some([60.0, 60.0]),
                ..params()
            },
            false
        ));
        for surface in [SurfaceMethod::Gaussian, SurfaceMethod::HarmonicMol] {
            let p = PbParams { surface, ..params() };
            assert!(!refused(p.clone(), false));
            assert!(refused(p, true));
        }
        let sdh = PbParams {
            bcfl: BoundaryCondition::Sdh,
            ..params()
        };
        assert!(!refused(sdh.clone(), false));
        assert!(refused(sdh, true));
        let per_atom = PbParams {
            atom_pdie: Some(vec![2.0; atoms.len()].into()),
            ions: Vec::new(),
            ..params()
        };
        assert!(!refused(per_atom.clone(), false));
        assert!(refused(per_atom, true));
        // npbe without ions is the linear equation.
        assert!(!refused(
            PbParams {
                equation: Equation::Nonlinear,
                ions: Vec::new(),
                ..params()
            },
            false
        ));
    }
}
