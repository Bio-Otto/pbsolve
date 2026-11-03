//! High-level drivers: focused LPBE solves and the polar solvation energy
//! ΔG_polar = ½ Σ q_i [u_solvated(r_i) − u_reference(r_i)], with both states
//! solved on identical grids so grid self-energies cancel exactly.

use crate::autogrid::GridSpec;
use crate::backend::Backend;
use crate::boundary::{analytic_boundary, focus_boundary, BoundaryMedium};
use crate::charge::spread_charges;
use crate::error::Result;
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::operator::{assemble_rhs, StencilOperator};
use crate::params::PbParams;
use crate::solver::{pcg, SolveStats, SolverOptions};
use crate::surface::CoefficientMaps;
use crate::units::{bjerrum_length_vacuum, kappa_bar_squared, kt_in_kcal_per_mol};
use std::sync::Arc;

/// Which dielectric/ionic environment to solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medium {
    /// Solute in solvent with mobile ions.
    Solvated,
    /// Homogeneous medium with ε = pdie and no ions (APBS: second `elec`
    /// block with `sdie = pdie`, no `ion`).
    Reference,
}

/// Potential (kT/e) on the finest grid.
#[derive(Clone, Debug)]
pub struct FieldSolution {
    pub grid: Grid,
    pub potential: Vec<f64>,
    /// One entry per grid level (coarse first).
    pub stats: Vec<SolveStats>,
}

#[derive(Clone, Debug)]
pub struct PolarSolvation {
    /// ΔG_polar in kcal/mol.
    pub energy_kcal: f64,
    /// ΔG_polar in kT.
    pub energy_kt: f64,
    /// ½ Σ q u in the solvated state (kT).
    pub solvated_kt: f64,
    /// ½ Σ q u in the reference state (kT).
    pub reference_kt: f64,
    pub stats_solvated: Vec<SolveStats>,
    pub stats_reference: Vec<SolveStats>,
    pub grid: GridSpec,
}

/// Dielectric and ion maps of one grid level for the given medium.
pub fn coefficient_maps(grid: &Grid, atoms: &[Atom], params: &PbParams, medium: Medium) -> CoefficientMaps {
    match medium {
        Medium::Solvated => CoefficientMaps::solvated(grid, atoms, params),
        Medium::Reference => CoefficientMaps::homogeneous(grid, params.pdie),
    }
}

fn boundary_medium(params: &PbParams, medium: Medium) -> BoundaryMedium {
    let lb0 = bjerrum_length_vacuum(params.temperature);
    match medium {
        Medium::Reference => BoundaryMedium {
            eps: params.pdie,
            eps_solute: params.pdie,
            kappa: 0.0,
            lb0,
        },
        Medium::Solvated => {
            let kbar2 = kappa_bar_squared(&params.ions, params.temperature);
            BoundaryMedium {
                eps: params.sdie,
                eps_solute: params.pdie,
                kappa: (kbar2 / params.sdie).sqrt(),
                lb0,
            }
        }
    }
}

/// Solves the LPBE on `grid` for given coefficient maps, node charges (e)
/// and Dirichlet data `g_ext` (boundary nodes; interior ignored).
/// Returns u = x + g_ext in kT/e.
///
/// This is the discretisation + solver alone, independent of how the maps
/// were built — used for "map injection" tests against APBS.
pub fn solve_with_maps<B: Backend>(
    be: &B,
    grid: &Grid,
    maps: CoefficientMaps,
    node_charge: &[f64],
    g_ext: &[f64],
    temperature: f64,
    opts: SolverOptions,
) -> Result<(Vec<f64>, SolveStats)> {
    let n = grid.len();
    for (name, len) in [
        ("eps_x", maps.eps_x.len()),
        ("eps_y", maps.eps_y.len()),
        ("eps_z", maps.eps_z.len()),
        ("kappa_sq", maps.kappa_sq.len()),
        ("node_charge", node_charge.len()),
        ("g_ext", g_ext.len()),
    ] {
        if len != n {
            return crate::error::invalid(format!("{name} has {len} values, grid has {n} nodes"));
        }
    }
    let bad_eps = [&maps.eps_x, &maps.eps_y, &maps.eps_z]
        .iter()
        .any(|m| m.iter().any(|e| e.is_nan() || *e <= 0.0));
    if bad_eps || maps.kappa_sq.iter().any(|k| k.is_nan() || *k < 0.0) {
        return crate::error::invalid("dielectric maps must be positive and kappa_sq non-negative");
    }
    let mut g = vec![0.0; n];
    for c in grid.boundary_indices() {
        g[c] = g_ext[c];
    }
    let lb0 = bjerrum_length_vacuum(temperature);
    let op = Arc::new(StencilOperator::new(grid, maps));
    let b = assemble_rhs(&op, node_charge, &g, lb0);
    let (mut u, stats) = pcg(be, op, &b, opts)?;
    for (ui, gi) in u.iter_mut().zip(&g) {
        *ui += gi;
    }
    Ok((u, stats))
}

/// One LPBE solve on `grid` with Dirichlet data `g_ext`. Returns u = x + g_ext.
fn solve_level<B: Backend>(
    be: &B,
    grid: &Grid,
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
    g_ext: Vec<f64>,
) -> Result<(Vec<f64>, SolveStats)> {
    let maps = coefficient_maps(grid, atoms, params, medium);
    let q = spread_charges(grid, atoms)?;
    let opts = SolverOptions {
        tol: params.tol,
        max_iter: params.max_iter,
    };
    solve_with_maps(be, grid, maps, &q, &g_ext, params.temperature, opts)
}

/// Focusing solve over all levels of `spec` (APBS mg-auto sequence, see
/// [`GridSpec::levels`]): the outermost level uses the analytic boundary
/// condition `bcfl`, every finer level takes its boundary from the previous one.
pub fn solve_focused<B: Backend>(
    be: &B,
    spec: &GridSpec,
    atoms: &[Atom],
    params: &PbParams,
    medium: Medium,
) -> Result<FieldSolution> {
    params.validate()?;
    let levels = spec.levels()?;
    let bm = boundary_medium(params, medium);
    let mut stats = Vec::with_capacity(levels.len());
    let mut prev: Option<(Grid, Vec<f64>)> = None;
    for grid in levels {
        let g = match &prev {
            None => analytic_boundary(&grid, atoms, params.bcfl, &bm),
            Some((pg, pu)) => focus_boundary(&grid, pg, pu)?,
        };
        let (u, s) = solve_level(be, &grid, atoms, params, medium, g)?;
        stats.push(s);
        prev = Some((grid, u));
    }
    let (grid, potential) = prev.expect("at least one level");
    Ok(FieldSolution { grid, potential, stats })
}

/// ½ Σ q_i u(r_i) in kT (trilinear interpolation, identical to APBS
/// `calcenergy total` with `chgm spl0`).
pub fn electrostatic_energy_kt(sol: &FieldSolution, atoms: &[Atom]) -> Result<f64> {
    let q = spread_charges(&sol.grid, atoms)?;
    Ok(0.5 * q.iter().zip(&sol.potential).map(|(a, b)| a * b).sum::<f64>())
}

/// Polar (electrostatic) solvation free energy on the given grid.
pub fn polar_solvation<B: Backend>(
    be: &B,
    atoms: &[Atom],
    params: &PbParams,
    spec: &GridSpec,
) -> Result<PolarSolvation> {
    let solv = solve_focused(be, spec, atoms, params, Medium::Solvated)?;
    let refr = solve_focused(be, spec, atoms, params, Medium::Reference)?;
    let solvated_kt = electrostatic_energy_kt(&solv, atoms)?;
    let reference_kt = electrostatic_energy_kt(&refr, atoms)?;
    let energy_kt = solvated_kt - reference_kt;
    Ok(PolarSolvation {
        energy_kcal: energy_kt * kt_in_kcal_per_mol(params.temperature),
        energy_kt,
        solvated_kt,
        reference_kt,
        stats_solvated: solv.stats,
        stats_reference: refr.stats,
        grid: spec.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::params::SurfaceMethod;

    /// Born ion: ΔG = −(l_B0 z² / 2a)(1/ε_in − 1/ε_out) kT.
    fn born(params: &PbParams, a: f64) -> (f64, f64) {
        let atoms = [Atom {
            position: [0.1, -0.05, 0.02],
            charge: 1.0,
            radius: a,
        }];
        let spec = GridSpec::auto(&atoms, 0.2, 1.7, 12.0).unwrap();
        let res = polar_solvation(&CpuBackend::new(), &atoms, params, &spec).unwrap();
        let lb0 = bjerrum_length_vacuum(params.temperature);
        let exact = -(lb0 / (2.0 * a)) * (1.0 / params.pdie - 1.0 / params.sdie);
        (res.energy_kt, exact)
    }

    #[test]
    fn born_ion_matches_analytic_solution() {
        for surface in [SurfaceMethod::Mol, SurfaceMethod::HarmonicMol] {
            let params = PbParams {
                surface,
                ..PbParams::default()
            };
            let (num, exact) = born(&params, 2.0);
            let err = (num - exact).abs() / exact.abs();
            assert!(
                err < 0.03,
                "{surface:?}: numeric {num:.3} kT vs analytic {exact:.3} kT ({:.2}%)",
                100.0 * err
            );
        }
    }
}
