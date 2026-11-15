//! Reaction-field decomposition of ΔG_polar ([`CoulombMethod::Analytic`],
//! not an APBS option).
//!
//! The discrete solvated state is A u = f: A the volume-scaled 7-point
//! operator ([`crate::operator`]), f = 4π l_B0 q the spread charges. With G
//! the analytic Coulomb potential of the charges in a homogeneous `pdie`
//! medium and χ a set of interior nodes that contains the solute, the
//! substitution u = v + χG is exact:
//!
//! ```text
//! A v = f − A(χG)
//! ```
//!
//! Inside χ the charges are the sources of G (−pdie ∇²G = 4π l_B0 ρ), so f is
//! replaced there by the discrete source of G, A_p G (A_p: ε = pdie, no
//! ions), and outside χ by 0, the continuous source:
//!
//! ```text
//! A v = χ·A_p G − A(χG) =: r
//! ```
//!
//! r vanishes wherever a node and its neighbours lie in χ, see only `pdie`
//! faces and no ions; it lives on a thin layer at the edge of χ and at
//! faces of other dielectric inside χ. v is the reaction field inside χ and
//! the total potential outside, and ΔG_polar = ½ Σ q_i [v − (1−χ)G](r_i):
//! the grid self-energy never appears, and no reference state is solved.
//! With χ the solute region this is the scheme of Cai et al. (2009); here χ
//! reaches [`margin`] beyond every atomic sphere, because the error of the
//! discrete jump of v across the edge of χ grows as the edge approaches a
//! charge (docs/VALIDATION.md).

use crate::backend::Backend;
use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::operator::StencilOperator;
use crate::surface::{mark_map, Sphere};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Distance (Å) by which the region χ extends beyond every atomic sphere:
/// 1 Å, and at least two node spacings. Larger values change ΔG_polar of
/// 1YCR by less than 0.02 kcal/mol at 0.5 and 0.25 Å.
pub fn margin(grid: &Grid) -> f64 {
    let h = grid.spacing.iter().copied().fold(0.0, f64::max);
    (2.0 * h).max(1.0)
}

/// Coulomb potential l_B0 Σ_j q_j / (eps |p − r_j|) (kT/e) of `atoms` at
/// `points`; infinite at a charge.
// shortcut: direct sum, O(points × atoms); a treecode or multilevel summation
// would make large proteins on fine grids cheaper.
pub fn coulomb_potential(atoms: &[Atom], eps: f64, lb0: f64, points: &[[f64; 3]]) -> Vec<f64> {
    let src: Vec<[f64; 4]> = atoms
        .iter()
        .filter(|a| a.charge != 0.0)
        .map(|a| [a.position[0], a.position[1], a.position[2], a.charge])
        .collect();
    let scale = lb0 / eps;
    points
        .par_iter()
        .with_min_len(64)
        .map(|p| {
            let mut sum = 0.0;
            for [x, y, z, q] in &src {
                let (dx, dy, dz) = (p[0] - x, p[1] - y, p[2] - z);
                sum += q / (dx * dx + dy * dy + dz * dz).sqrt();
            }
            scale * sum
        })
        .collect()
}

/// Coulomb energy ½ Σ_{i≠j} l_B0 q_i q_j / (eps r_ij) in kT, without
/// self-energies; pairs at one position are left out.
pub fn coulomb_energy_kt(atoms: &[Atom], eps: f64, lb0: f64) -> f64 {
    let charged: Vec<&Atom> = atoms.iter().filter(|a| a.charge != 0.0).collect();
    let sum: f64 = (0..charged.len())
        .into_par_iter()
        .map(|i| {
            let a = charged[i];
            let mut s = 0.0;
            for b in &charged[i + 1..] {
                let r2: f64 = (0..3).map(|k| (a.position[k] - b.position[k]).powi(2)).sum();
                if r2 > 0.0 {
                    s += b.charge / r2.sqrt();
                }
            }
            a.charge * s
        })
        .sum();
    lb0 / eps * sum
}

/// The nodes of `mask` and their six neighbours.
fn dilate(dims: [usize; 3], mask: &[bool]) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    let mut out = vec![false; mask.len()];
    out.par_chunks_mut(plane).enumerate().for_each(|(k, o)| {
        for j in 0..ny {
            for i in 0..nx {
                let c = i + nx * j + plane * k;
                o[i + nx * j] = mask[c]
                    || (i > 0 && mask[c - 1])
                    || (i + 1 < nx && mask[c + 1])
                    || (j > 0 && mask[c - nx])
                    || (j + 1 < ny && mask[c + nx])
                    || (k > 0 && mask[c - plane])
                    || (k + 1 < nz && mask[c + plane]);
            }
        }
    });
    out
}

/// The region χ of one grid level and the nodes where r may be nonzero.
pub(crate) struct Split {
    dims: [usize; 3],
    /// χ: interior nodes within `radius + margin` of an atom.
    chi: Vec<bool>,
    /// Interior nodes where r may be nonzero, increasing.
    layer: Vec<usize>,
}

impl Split {
    /// `inset`: how far inside an atom's radius the solvated maps may still
    /// differ from `pdie` without ions ([`crate::params::PbParams::pdie_inset`]).
    pub fn new(grid: &Grid, atoms: &[Atom], inset: f64) -> Self {
        let n = grid.len();
        let spheres = |d: f64| -> Vec<Sphere> {
            atoms
                .iter()
                .filter(|a| a.radius + d > 0.0)
                .map(|a| Sphere {
                    center: a.position,
                    radius: a.radius + d,
                })
                .collect()
        };
        let interior = |c: usize| {
            let (i, j, k) = grid.ijk(c);
            !grid.is_boundary(i, j, k)
        };
        let mark = |d: f64| -> Vec<bool> {
            let map = mark_map(grid, [0.0; 3], 0.0, &[(&spheres(d), 1.0)]);
            map.par_iter()
                .enumerate()
                .map(|(c, &m)| m == 1.0 && interior(c))
                .collect()
        };
        let chi = mark(margin(grid));
        // `deep`: interior nodes whose six edges lie inside one atomic sphere
        // shrunk by `inset`. Their faces are `pdie` on every surface and
        // interface (the probes of `mol` never enter an atomic sphere) and
        // ions are excluded there.
        let reach = grid.spacing.iter().copied().fold(0.0, f64::max) * (1.0 + 1e-6) + 1e-9 + inset;
        let not_deep: Vec<bool> = mark(-reach).into_par_iter().map(|d| !d).collect();
        // r_c = 0 unless c lies next to χ and c or a neighbour is not deep.
        let near_chi = dilate(grid.dims, &chi);
        let near_open = dilate(grid.dims, &not_deep);
        let layer = (0..n)
            .into_par_iter()
            .filter(|&c| near_chi[c] && near_open[c] && interior(c))
            .collect();
        Self {
            dims: grid.dims,
            chi,
            layer,
        }
    }

    /// r = χ·A_p G − A(χG) on the layer, as node charges r / (4π l_B0) for
    /// the right-hand side 4π l_B0 q − A g. `maps` are the level's
    /// (eps_x, eps_y, eps_z, κ̄²) on `be`; only their values on the layer are
    /// read.
    pub fn source<B: Backend>(
        &self,
        be: &B,
        grid: &Grid,
        maps: &[B::Vector; 4],
        atoms: &[Atom],
        pdie: f64,
        lb0: f64,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let [nx, ny, _] = self.dims;
        let strides = [1, nx, nx * ny];
        let nodes = &self.layer;
        let below = |a: usize| -> Vec<usize> { nodes.iter().map(|&c| c - strides[a]).collect() };
        // Faces c−½ and c+½ along each axis, and κ̄² at c.
        let mut lo = Vec::with_capacity(3);
        let mut hi = Vec::with_capacity(3);
        for (a, map) in maps[..3].iter().enumerate() {
            lo.push(be.gather(map, &below(a))?);
            hi.push(be.gather(map, nodes)?);
        }
        let kappa = be.gather(&maps[3], nodes)?;
        let chi = &self.chi;
        // The faces of node m: (ε, neighbour) for the six directions.
        let (lo, hi) = (&lo, &hi);
        let faces = |m: usize| -> [(usize, f64, usize); 6] {
            let c = nodes[m];
            let f = |a: usize| [(a, lo[a][m], c - strides[a]), (a, hi[a][m], c + strides[a])];
            let [x, y, z] = [f(0), f(1), f(2)];
            [x[0], x[1], y[0], y[1], z[0], z[1]]
        };
        // G is needed where a term of r has a nonzero coefficient.
        let mut need: Vec<usize> = (0..nodes.len())
            .into_par_iter()
            .flat_map_iter(|m| {
                let c = nodes[m];
                let mut out = Vec::new();
                if chi[c] {
                    let mut own = kappa[m] != 0.0;
                    for (_, eps, nb) in faces(m) {
                        if eps != pdie {
                            own = true;
                            out.push(nb);
                        } else if !chi[nb] {
                            out.push(nb);
                        }
                    }
                    if own {
                        out.push(c);
                    }
                } else {
                    out.extend(faces(m).into_iter().filter(|&(_, _, nb)| chi[nb]).map(|(_, _, nb)| nb));
                }
                out
            })
            .collect();
        need.par_sort_unstable();
        need.dedup();
        let points: Vec<[f64; 3]> = need
            .iter()
            .map(|&c| {
                let (i, j, k) = grid.ijk(c);
                grid.position(i, j, k)
            })
            .collect();
        let g = coulomb_potential(atoms, pdie, lb0, &points);
        if let Some(m) = g.iter().position(|v| !v.is_finite()) {
            return invalid(format!(
                "coulomb 'analytic': a charge lies on grid node {:?}, where the Coulomb potential is needed; \
                 shift the molecule or use coulomb 'grid'",
                points[m]
            ));
        }
        let at = |c: usize| g[need.binary_search(&c).expect("G at a needed node")];
        let coef = StencilOperator::face_coefficients(grid);
        let vol = grid.cell_volume();
        let scale = 1.0 / (4.0 * PI * lb0);
        let values = (0..nodes.len())
            .into_par_iter()
            .map(|m| {
                let c = nodes[m];
                let mut r = 0.0;
                if chi[c] {
                    // Σ c_a [(pdie − ε)(G_c − G_nb) − ε (1 − χ_nb) G_nb] − κ̄² V G_c
                    for (a, eps, nb) in faces(m) {
                        if eps != pdie {
                            r += coef[a] * (pdie - eps) * (at(c) - at(nb));
                        }
                        if !chi[nb] {
                            r -= coef[a] * eps * at(nb);
                        }
                    }
                    if kappa[m] != 0.0 {
                        r -= kappa[m] * vol * at(c);
                    }
                } else {
                    // Σ c_a ε χ_nb G_nb
                    for (a, eps, nb) in faces(m) {
                        if chi[nb] {
                            r += coef[a] * eps * at(nb);
                        }
                    }
                }
                scale * r
            })
            .collect();
        Ok((nodes.clone(), values))
    }

    /// The reaction field v − (1−χ) G at the energy's nodes, with v at those
    /// nodes.
    pub fn reaction_field(
        &self,
        grid: &Grid,
        atoms: &[Atom],
        pdie: f64,
        lb0: f64,
        nodes: &[usize],
        v: &[f64],
    ) -> Vec<f64> {
        let outside: Vec<usize> = nodes.iter().copied().filter(|&c| !self.chi[c]).collect();
        let points: Vec<[f64; 3]> = outside
            .iter()
            .map(|&c| {
                let (i, j, k) = grid.ijk(c);
                grid.position(i, j, k)
            })
            .collect();
        let g = coulomb_potential(atoms, pdie, lb0, &points);
        let mut next = 0;
        nodes
            .iter()
            .zip(v)
            .map(|(&c, v)| {
                if self.chi[c] {
                    *v
                } else {
                    next += 1;
                    v - g[next - 1]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autogrid::GridSpec;
    use crate::backend::Backend;
    use crate::cpu::CpuBackend;
    use crate::error::PbError;
    use crate::params::{CoulombMethod, InterfaceMethod, PbParams, SurfaceMethod};
    use crate::pb::polar_solvation;
    use crate::surface::CoefficientMaps;
    use crate::units::bjerrum_length_vacuum;

    #[test]
    fn coulomb_energy_is_the_pair_sum() {
        let atoms = [
            Atom {
                position: [0.0, 0.0, 0.0],
                charge: 1.0,
                radius: 1.5,
            },
            Atom {
                position: [3.0, 4.0, 0.0],
                charge: -0.5,
                radius: 0.0,
            },
            Atom {
                position: [3.0, 4.0, 0.0],
                charge: 0.25,
                radius: 0.0,
            },
        ];
        // 1·(−0.5)/5 + 1·0.25/5; the coincident pair is left out.
        let e = coulomb_energy_kt(&atoms, 2.0, 7.0);
        assert!((e - 7.0 / 2.0 * (-0.1 + 0.05)).abs() < 1e-15, "{e}");
        let g = coulomb_potential(&atoms, 2.0, 7.0, &[[0.0, 0.0, 5.0]]);
        let expected = 7.0 / 2.0 * (1.0 / 5.0 - 0.25 / 50f64.sqrt());
        assert!((g[0] - expected).abs() < 1e-14, "{g:?}");
    }

    /// Two overlapping atoms, a buried charge of radius 0 and a charge of
    /// radius 0 in the solvent, none on a node.
    fn molecule() -> Vec<Atom> {
        [
            ([0.13, -0.21, 0.07], 0.6, 1.8),
            ([1.71, 0.52, -0.33], -0.4, 1.5),
            ([0.62, 0.11, 0.48], 0.35, 0.0),
            ([-3.9, 0.37, 0.21], -0.25, 0.0),
        ]
        .map(|(position, charge, radius)| Atom {
            position,
            charge,
            radius,
        })
        .to_vec()
    }

    /// The sparse source on the layer equals χ·A_p G − A(χG) evaluated on
    /// every node, which vanishes off the layer, for every surface and
    /// interface, with and without ions.
    #[test]
    fn source_equals_the_dense_formula_and_vanishes_off_the_layer() {
        let lb0 = bjerrum_length_vacuum(298.15);
        let some = Grid::centered([0.05, 0.4, 0.1], [12.0, 10.5, 9.0], [25, 23, 21]).unwrap();
        // Two large spheres 0.1 Å apart across x = 0, between the nodes
        // x = ±0.25: nodes deep inside different spheres face each other
        // across the gap, whose face is solvent with `vdw`.
        let gap = [-4.05, 4.05].map(|x| Atom {
            position: [x + 0.013, 0.021, -0.017],
            charge: x.signum() * 0.5,
            radius: 4.0,
        });
        let across = Grid::centered([0.0; 3], [18.5, 10.0, 10.0], [38, 21, 21]).unwrap();
        // h = 0.2 Å < swin: the spline window reaches faces between nodes
        // deeper than h inside a sphere.
        let fine = Grid::centered([0.0; 3], [18.4, 10.0, 10.0], [93, 51, 51]).unwrap();
        for (atoms, grid, surface, interface, salt) in [
            (molecule(), &some, SurfaceMethod::Mol, InterfaceMethod::Sharp, 0.15),
            (molecule(), &some, SurfaceMethod::Mol, InterfaceMethod::Fractional, 0.15),
            (molecule(), &some, SurfaceMethod::Vdw, InterfaceMethod::Fractional, 0.0),
            (
                molecule(),
                &some,
                SurfaceMethod::HarmonicMol,
                InterfaceMethod::Sharp,
                0.15,
            ),
            (gap.to_vec(), &across, SurfaceMethod::Vdw, InterfaceMethod::Sharp, 0.0),
            (molecule(), &some, SurfaceMethod::Spline2, InterfaceMethod::Sharp, 0.15),
            (molecule(), &some, SurfaceMethod::Spline4, InterfaceMethod::Sharp, 0.0),
            (gap.to_vec(), &fine, SurfaceMethod::Spline2, InterfaceMethod::Sharp, 0.0),
        ] {
            let (atoms, grid) = (&atoms[..], grid);
            let n = grid.len();
            let params = PbParams {
                pdie: 2.0,
                surface,
                interface,
                ..PbParams::default()
            }
            .with_salt(salt, 1.2);
            let maps = CoefficientMaps::solvated(grid, atoms, &params);
            let be = CpuBackend::new();
            let on_be = [&maps.eps_x, &maps.eps_y, &maps.eps_z, &maps.kappa_sq].map(|m| be.upload(m).unwrap());
            let split = Split::new(grid, atoms, params.pdie_inset());
            let (nodes, q) = split.source(&be, grid, &on_be, atoms, params.pdie, lb0).unwrap();

            let points: Vec<[f64; 3]> = (0..n)
                .map(|c| {
                    let (i, j, k) = grid.ijk(c);
                    grid.position(i, j, k)
                })
                .collect();
            let g = coulomb_potential(atoms, params.pdie, lb0, &points);
            let chi_g: Vec<f64> = (0..n).map(|c| if split.chi[c] { g[c] } else { 0.0 }).collect();
            let solvated = StencilOperator::new(grid, maps.clone());
            let homogeneous = StencilOperator::new(grid, CoefficientMaps::homogeneous(grid, params.pdie));
            let (mut a_chi_g, mut ap_g) = (vec![0.0; n], vec![0.0; n]);
            solvated.apply(&chi_g, &mut a_chi_g);
            homogeneous.apply(&g, &mut ap_g);
            let dense: Vec<f64> = (0..n)
                .map(|c| if split.chi[c] { ap_g[c] } else { 0.0 } - a_chi_g[c])
                .collect();
            let scale = dense.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let sparse: std::collections::HashMap<usize, f64> =
                nodes.iter().zip(&q).map(|(&c, &q)| (c, 4.0 * PI * lb0 * q)).collect();
            for (c, d) in dense.iter().enumerate() {
                let r = sparse.get(&c).copied().unwrap_or(0.0);
                assert!(
                    (r - d).abs() <= 1e-11 * scale,
                    "{surface:?}/{interface:?}: node {c}: sparse {r} vs dense {d} (scale {scale})"
                );
            }
            let nonzero = dense.iter().filter(|v| v.abs() > 1e-9 * scale).count();
            assert!(
                nonzero > 500 && nodes.len() < n / 2,
                "{nonzero} nonzero, layer {}",
                nodes.len()
            );
        }
    }

    /// Outside χ, v is the total potential: the reaction field there is v − G.
    #[test]
    fn reaction_field_subtracts_g_outside_chi() {
        let atoms = molecule();
        let grid = Grid::centered([0.05, 0.4, 0.1], [12.0, 10.5, 9.0], [25, 23, 21]).unwrap();
        let split = Split::new(&grid, &atoms, 0.0);
        let inside = (0..grid.len()).find(|&c| split.chi[c]).unwrap();
        let outside = [0, grid.len() / 2 + 3].map(|c| (c..grid.len()).find(|&c| !split.chi[c]).unwrap());
        let nodes = [inside, outside[0], outside[1]];
        let v = [1.5, 2.5, -4.0];
        let points = nodes.map(|c| {
            let (i, j, k) = grid.ijk(c);
            grid.position(i, j, k)
        });
        let g = coulomb_potential(&atoms, 2.0, 7.0, &points);
        let rf = split.reaction_field(&grid, &atoms, 2.0, 7.0, &nodes, &v);
        assert_eq!(rf, vec![v[0], v[1] - g[1], v[2] - g[2]]);
    }

    fn solve(atoms: &[Atom], params: &PbParams, spacing: f64) -> crate::pb::PolarSolvation {
        let spec = GridSpec::auto(atoms, spacing, 1.7, 12.0).unwrap();
        polar_solvation(&CpuBackend::new(), atoms, params, &spec).unwrap()
    }

    /// Born ion: the decomposition is as accurate as the reference solve,
    /// solves once per level and reports the self-energy-free states.
    #[test]
    fn born_ion_with_the_analytic_coulomb_potential() {
        let atoms = [Atom {
            position: [0.1, -0.05, 0.02],
            charge: 1.0,
            radius: 2.0,
        }];
        let lb0 = bjerrum_length_vacuum(298.15);
        for interface in [InterfaceMethod::Sharp, InterfaceMethod::Fractional] {
            let grid = PbParams {
                interface,
                ..PbParams::default()
            };
            let analytic = PbParams {
                coulomb: CoulombMethod::Analytic,
                ..grid.clone()
            };
            let (a, b) = (solve(&atoms, &grid, 0.3), solve(&atoms, &analytic, 0.3));
            let exact = -(lb0 / 4.0) * (1.0 - 1.0 / 78.54);
            let (ea, eb) = ((a.energy_kt - exact) / exact, (b.energy_kt - exact) / exact);
            assert!(
                eb.abs() < 0.03 && eb.abs() <= ea.abs() + 0.002,
                "{interface:?}: grid {ea:+.4}, analytic {eb:+.4}"
            );
            assert_eq!(b.stats_reference.len(), 0);
            assert_eq!(b.stats_solvated.len(), a.stats_solvated.len());
            assert_eq!(b.reference_kt, 0.0, "one charge has no Coulomb energy");
            assert_eq!(b.solvated_kt, b.energy_kt);
        }
    }

    /// A fine grid that cuts through the solute (focusing on part of it):
    /// χ stays off the boundary, where v is the total potential, and both
    /// methods agree as closely as on a grid that holds the whole solute.
    #[test]
    fn a_fine_grid_through_the_solute() {
        let atoms = [
            Atom {
                position: [0.07, -0.04, 0.02],
                charge: 1.0,
                radius: 3.0,
            },
            Atom {
                position: [1.03, 0.51, -0.36],
                charge: -0.4,
                radius: 0.0,
            },
        ];
        let center = [0.0; 3];
        let spec = |fine: f64| GridSpec {
            center,
            dims: [33; 3],
            coarse_length: [40.0; 3],
            fine_length: [fine; 3],
        };
        let (cut, whole) = (spec(4.8), spec(16.0));
        let grid = PbParams::default().with_salt(0.1, 2.0);
        let analytic = PbParams {
            coulomb: CoulombMethod::Analytic,
            ..grid.clone()
        };
        let be = CpuBackend::new();
        let gap = |spec: &GridSpec| {
            let a = polar_solvation(&be, &atoms, &grid, spec).unwrap().energy_kt;
            let b = polar_solvation(&be, &atoms, &analytic, spec).unwrap().energy_kt;
            ((b - a) / a).abs()
        };
        let (cut, whole) = (gap(&cut), gap(&whole));
        assert!(
            cut < 0.02 && whole < 0.02,
            "relative gap: cut {cut:.4}, whole {whole:.4}"
        );
    }

    /// A charge on a node where G is needed cannot be decomposed: an error,
    /// not a NaN.
    #[test]
    fn a_charge_on_a_needed_node_is_an_error() {
        let atoms = [
            Atom {
                position: [0.0, 0.0, 0.0],
                charge: 1.0,
                radius: 2.0,
            },
            Atom {
                position: [4.0, 0.0, 0.0],
                charge: -1.0,
                radius: 0.0,
            },
        ];
        let spec = GridSpec::single([0.0; 3], [16.0; 3], [33, 33, 33]);
        let params = PbParams {
            coulomb: CoulombMethod::Analytic,
            ..PbParams::default()
        };
        match polar_solvation(&CpuBackend::new(), &atoms, &params, &spec) {
            Err(PbError::InvalidInput(msg)) => assert!(msg.contains("coulomb 'analytic'"), "{msg}"),
            other => panic!("expected an error, got {other:?}"),
        }
    }
}
