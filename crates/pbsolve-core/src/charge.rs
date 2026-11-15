// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (vpmg.c: bspline2, bspline4, dbspline2, dbspline4,
// fillcoChargeSpline2, fillcoPermanentMultipole), Copyright (c) 2010-2022 Battelle Memorial
// Institute and other APBS contributors, BSD-3-Clause. See THIRD_PARTY_NOTICES.md for the full
// license text.

//! Charge discretisation (APBS `chgm`): trilinear spreading (`spl0`) and the
//! B-splines of order 3 and 5 (`spl2`, `spl4`). The B-spline weights, their
//! node ranges and the order of the products follow APBS 3.4.1 (`vpmg.c`:
//! `bspline2`, `bspline4`, `fillcoChargeSpline2`, `fillcoPermanentMultipole`).

use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::Atom;
use crate::params::ChargeMethod;

/// Spreads point charges to the 8 surrounding nodes. Returns the node charge
/// in e (not a density). Every charge must lie strictly inside the grid so
/// that no charge lands on a Dirichlet boundary node.
pub fn spread_charges(grid: &Grid, atoms: &[Atom]) -> Result<Vec<f64>> {
    spread_charges_with(grid, atoms, ChargeMethod::Trilinear)
}

/// [`spread_charges`] with any [`ChargeMethod`]: every node a charge reaches
/// must be an interior node.
pub fn spread_charges_with(grid: &Grid, atoms: &[Atom], method: ChargeMethod) -> Result<Vec<f64>> {
    let mut q = vec![0.0; grid.len()];
    for_each_contribution(grid, atoms, method, |c, v| q[c] += v)?;
    Ok(q)
}

/// [`spread_charges`] as a sparse vector: the nodes that receive charge, in
/// increasing order, and their charges. Each node sums its contributions in
/// the order `spread_charges` adds them (atoms in order, then the 8 corners),
/// starting from 0.0, so the values are identical bit for bit; nodes not
/// listed are 0.0.
pub fn spread_charges_sparse(grid: &Grid, atoms: &[Atom]) -> Result<(Vec<usize>, Vec<f64>)> {
    spread_charges_sparse_with(grid, atoms, ChargeMethod::Trilinear)
}

/// [`spread_charges_sparse`] with any [`ChargeMethod`], equal bit for bit to
/// [`spread_charges_with`].
pub fn spread_charges_sparse_with(grid: &Grid, atoms: &[Atom], method: ChargeMethod) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut contrib: Vec<(usize, f64)> = Vec::with_capacity(8 * atoms.len());
    for_each_contribution(grid, atoms, method, |c, v| contrib.push((c, v)))?;
    // Stable: equal nodes keep their contribution order.
    contrib.sort_by_key(|&(c, _)| c);
    let mut nodes: Vec<usize> = Vec::with_capacity(contrib.len());
    let mut values: Vec<f64> = Vec::with_capacity(contrib.len());
    for (c, v) in contrib {
        match (nodes.last(), values.last_mut()) {
            (Some(&last), Some(sum)) if last == c => *sum += v,
            _ => {
                nodes.push(c);
                values.push(0.0 + v);
            }
        }
    }
    Ok((nodes, values))
}

/// B-spline of order 3 on [0, 3] (APBS `bspline2`).
pub(crate) fn bspline2(x: f64) -> f64 {
    let m2m = if (0.0..=2.0).contains(&x) {
        1.0 - (x - 1.0).abs()
    } else {
        0.0
    };
    let m2 = if (1.0..=3.0).contains(&x) {
        1.0 - (x - 2.0).abs()
    } else {
        0.0
    };
    if (0.0..=3.0).contains(&x) {
        0.5 * x * m2m + 0.5 * (3.0 - x) * m2
    } else {
        0.0
    }
}

/// B-spline of order 5 on [0, 5] (APBS `bspline4`).
pub(crate) fn bspline4(x: f64) -> f64 {
    const ONE6: f64 = 1.0 / 6.0;
    const ONE8: f64 = 1.0 / 8.0;
    const ONE24: f64 = 1.0 / 24.0;
    const THIRTEEN24: f64 = 13.0 / 24.0;
    const FOURTYSEVEN24: f64 = 47.0 / 24.0;
    const SEVENTEEN24: f64 = 17.0 / 24.0;
    if x > 0.0 && x <= 1.0 {
        let m = x * x;
        ONE24 * m * m
    } else if x > 1.0 && x <= 2.0 {
        let m = x - 1.0;
        let m2 = m * m;
        -ONE8 + ONE6 * x + m2 * (0.25 + ONE6 * m - ONE6 * m2)
    } else if x > 2.0 && x <= 3.0 {
        let m = x - 2.0;
        let m2 = m * m;
        -THIRTEEN24 + 0.5 * x + m2 * (-0.25 - 0.5 * m + 0.25 * m2)
    } else if x > 3.0 && x <= 4.0 {
        let m = x - 3.0;
        let m2 = m * m;
        FOURTYSEVEN24 - 0.5 * x + m2 * (-0.25 + 0.5 * m - ONE6 * m2)
    } else if x > 4.0 && x <= 5.0 {
        let m = x - 4.0;
        let m2 = m * m;
        SEVENTEEN24 - ONE6 * x + m2 * (0.25 - ONE6 * m + ONE24 * m2)
    } else {
        0.0
    }
}

/// The derivative of [`bspline2`] (APBS `dbspline2`).
pub(crate) fn dbspline2(x: f64) -> f64 {
    let m2m = if (0.0..=2.0).contains(&x) {
        1.0 - (x - 1.0).abs()
    } else {
        0.0
    };
    let m2 = if (1.0..=3.0).contains(&x) {
        1.0 - (x - 2.0).abs()
    } else {
        0.0
    };
    m2m - m2
}

/// The derivative of [`bspline4`] (APBS `dbspline4`).
pub(crate) fn dbspline4(x: f64) -> f64 {
    const ONE6: f64 = 1.0 / 6.0;
    const ONE3: f64 = 1.0 / 3.0;
    const TWO3: f64 = 2.0 / 3.0;
    const THIRTEEN6: f64 = 13.0 / 6.0;
    if x > 0.0 && x <= 1.0 {
        let m2 = x * x;
        ONE6 * x * m2
    } else if x > 1.0 && x <= 2.0 {
        let m = x - 1.0;
        let m2 = m * m;
        -ONE3 + 0.5 * x + m2 * (0.5 - TWO3 * m)
    } else if x > 2.0 && x <= 3.0 {
        let m = x - 2.0;
        let m2 = m * m;
        1.5 - 0.5 * x + m2 * (-1.5 + m)
    } else if x > 3.0 && x <= 4.0 {
        let m = x - 3.0;
        let m2 = m * m;
        1.0 - 0.5 * x + m2 * (1.5 - TWO3 * m)
    } else if x > 4.0 && x <= 5.0 {
        let m = x - 4.0;
        let m2 = m * m;
        -THIRTEEN6 + 0.5 * x + m2 * (-0.5 + ONE6 * m)
    } else {
        0.0
    }
}

/// Calls `f(node, charge · weight)` for every charged atom and every node of
/// its stencil, in atom order.
fn for_each_contribution(
    grid: &Grid,
    atoms: &[Atom],
    method: ChargeMethod,
    mut f: impl FnMut(usize, f64),
) -> Result<()> {
    for (n, a) in atoms.iter().enumerate() {
        if a.charge == 0.0 {
            continue;
        }
        let outside = || {
            invalid(format!(
                "atom {n} at {:?} is too close to the boundary of the grid \
                 [{:?} .. {:?}] for charge method {}; enlarge the (fine) grid",
                a.position,
                grid.origin,
                grid.upper(),
                method.as_str()
            ))
        };
        let (half, kernel): (i64, fn(f64) -> f64) = match method {
            ChargeMethod::Trilinear => {
                if !grid.contains_interior(a.position) {
                    return invalid(format!(
                        "atom {n} at {:?} lies outside the interior of the grid \
                         [{:?} .. {:?}]; enlarge the (fine) grid",
                        a.position,
                        grid.origin,
                        grid.upper()
                    ));
                }
                let (idx, w) = grid.trilinear(a.position).expect("checked above");
                for m in 0..8 {
                    f(idx[m], a.charge * w[m]);
                }
                continue;
            }
            ChargeMethod::Spline2 => (1, bspline2),
            ChargeMethod::Spline4 => (2, bspline4),
        };
        // Nodes floor(t) − half … ceil(t) + half per axis, t the position in
        // grid units, with weights B(half + ½ + (i − t)); all must be interior.
        let mut t = [0.0; 3];
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for ax in 0..3 {
            t[ax] = (a.position[ax] - grid.origin[ax]) / grid.spacing[ax];
            if !t[ax].is_finite() {
                return outside();
            }
            lo[ax] = t[ax].floor() as i64 - half;
            hi[ax] = t[ax].ceil() as i64 + half;
            if lo[ax] < 1 || hi[ax] > grid.dims[ax] as i64 - 2 {
                return outside();
            }
        }
        // One end node per axis gets weight 0 unless t is a node; skip those.
        let shift = half as f64 + 0.5;
        for i in lo[0]..=hi[0] {
            let mx = kernel(shift + (i as f64 - t[0]));
            if mx == 0.0 {
                continue;
            }
            for j in lo[1]..=hi[1] {
                let my = kernel(shift + (j as f64 - t[1]));
                if my == 0.0 {
                    continue;
                }
                for k in lo[2]..=hi[2] {
                    let mz = kernel(shift + (k as f64 - t[2]));
                    if mz == 0.0 {
                        continue;
                    }
                    let v = match method {
                        ChargeMethod::Spline2 => a.charge * mx * my * mz,
                        _ => mx * my * mz * a.charge,
                    };
                    f(grid.index(i as usize, j as usize, k as usize), v);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const METHODS: [ChargeMethod; 3] = [ChargeMethod::Trilinear, ChargeMethod::Spline2, ChargeMethod::Spline4];

    #[test]
    fn b_spline_derivatives_are_the_slopes_of_the_splines() {
        for (b, db, width) in [
            (bspline2 as fn(f64) -> f64, dbspline2 as fn(f64) -> f64, 3),
            (bspline4, dbspline4, 5),
        ] {
            for i in 1..(100 * width) {
                let x = 0.01 * i as f64 + 0.003; // off the knots
                let fd = (b(x + 1e-6) - b(x - 1e-6)) / 2e-6;
                assert!((db(x) - fd).abs() < 1e-8, "{width}: x = {x}: {} vs {fd}", db(x));
            }
            assert_eq!(db(-0.5), 0.0);
            assert_eq!(db(width as f64 + 0.5), 0.0);
        }
    }

    /// Charge, centre and spread of one point charge's node charges per axis.
    fn moments(g: &Grid, q: &[f64]) -> (f64, [f64; 3], [f64; 3]) {
        let total: f64 = q.iter().sum();
        let mut mean = [0.0; 3];
        let mut var = [0.0; 3];
        for (c, v) in q.iter().enumerate() {
            let (i, j, k) = g.ijk(c);
            let p = g.position(i, j, k);
            for a in 0..3 {
                mean[a] += v * p[a] / total;
            }
        }
        for (c, v) in q.iter().enumerate() {
            let (i, j, k) = g.ijk(c);
            let p = g.position(i, j, k);
            for a in 0..3 {
                var[a] += v * (p[a] - mean[a]).powi(2) / total;
            }
        }
        (total, mean, var)
    }

    #[test]
    fn b_splines_keep_charge_and_centre_and_spread_as_their_order() {
        let g = Grid::centered([0.3; 3], [8.0, 9.0, 10.0], [17, 19, 41]).unwrap();
        let mut s: u64 = 11;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..50 {
            let p = [rnd(), rnd(), rnd()].map(|v| 3.0 * (v - 0.5));
            let atom = Atom {
                position: p,
                charge: -0.7,
                radius: 1.0,
            };
            for (method, var, nodes) in [(ChargeMethod::Spline2, 0.25, 3), (ChargeMethod::Spline4, 5.0 / 12.0, 5)] {
                let q = spread_charges_with(&g, &[atom], method).unwrap();
                let (total, mean, spread) = moments(&g, &q);
                assert!((total + 0.7).abs() < 1e-14, "{method:?} {total}");
                for a in 0..3 {
                    assert!((mean[a] - p[a]).abs() < 1e-12, "{method:?} {mean:?} vs {p:?}");
                    let h2 = g.spacing[a] * g.spacing[a];
                    assert!(
                        (spread[a] - var * h2).abs() < 1e-12,
                        "{method:?} axis {a}: {}",
                        spread[a] / h2
                    );
                }
                assert!(q.iter().filter(|v| **v != 0.0).count() <= nodes * nodes * nodes);
            }
        }
    }

    #[test]
    fn b_spline_weights_on_a_node_are_the_exact_values() {
        let g = Grid::centered([0.0; 3], [10.0; 3], [21; 3]).unwrap();
        let atom = Atom {
            position: [0.0; 3],
            charge: 1.0,
            radius: 1.0,
        };
        let centre = g.index(10, 10, 10);
        let axis = |q: &[f64], d: usize| q[centre + d]; // x is the fastest index
        let q2 = spread_charges_with(&g, &[atom], ChargeMethod::Spline2).unwrap();
        let w2 = [1.0 / 8.0, 3.0 / 4.0];
        assert_eq!(q2[centre], w2[1] * w2[1] * w2[1]);
        assert_eq!(axis(&q2, 1), w2[0] * w2[1] * w2[1]);
        let q4 = spread_charges_with(&g, &[atom], ChargeMethod::Spline4).unwrap();
        let w4 = [1.0 / 384.0, 19.0 / 96.0, 115.0 / 192.0];
        for (d, w) in [(0, w4[2]), (1, w4[1]), (2, w4[0])] {
            let expect = w * w4[2] * w4[2];
            assert!(
                (axis(&q4, d) - expect).abs() < 1e-16,
                "offset {d}: {} vs {expect}",
                axis(&q4, d)
            );
        }
        assert_eq!(q4.iter().filter(|v| **v != 0.0).count(), 125);
        assert_eq!(q2.iter().filter(|v| **v != 0.0).count(), 27);
    }

    #[test]
    fn b_splines_need_room_to_the_boundary() {
        // Node spacing 0.5 from −5: a charge at t = 2.3 grid units from the
        // boundary reaches node 1 with spl2 (floor − 1) but node 0 with spl4.
        let g = Grid::centered([0.0; 3], [10.0; 3], [21; 3]).unwrap();
        let near = Atom {
            position: [-5.0 + 2.3 * 0.5, 0.0, 0.0],
            charge: 1.0,
            radius: 1.0,
        };
        assert!(spread_charges_with(&g, &[near], ChargeMethod::Trilinear).is_ok());
        assert!(spread_charges_with(&g, &[near], ChargeMethod::Spline2).is_ok());
        let err = spread_charges_with(&g, &[near], ChargeMethod::Spline4)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("too close to the boundary") && err.contains("spl4"),
            "{err}"
        );
        let far = Atom {
            position: [-5.0 + 3.0 * 0.5, 0.0, 0.0],
            ..near
        };
        assert!(spread_charges_with(&g, &[far], ChargeMethod::Spline4).is_ok());
        let upper = Atom {
            position: [0.0, 5.0 - 2.3 * 0.5, 0.0],
            ..near
        };
        assert!(spread_charges_sparse_with(&g, &[upper], ChargeMethod::Spline2).is_ok());
        assert!(spread_charges_sparse_with(&g, &[upper], ChargeMethod::Spline4).is_err());
        for name in ["spl0", "spl2", "spl4"] {
            let m = ChargeMethod::parse(name).unwrap();
            assert_eq!(m.as_str(), name);
            assert_eq!(m.energy_method() == m, name != "spl4");
        }
        assert!(ChargeMethod::parse("spl1").is_err());
    }

    #[test]
    fn spreading_conserves_charge() {
        let g = Grid::centered([0.0; 3], [10.0; 3], [21; 3]).unwrap();
        let atoms = [
            Atom {
                position: [0.13, -1.7, 2.2],
                charge: 0.42,
                radius: 1.5,
            },
            Atom {
                position: [-3.3, 0.9, -0.4],
                charge: -1.1,
                radius: 1.5,
            },
        ];
        let q = spread_charges(&g, &atoms).unwrap();
        assert!((q.iter().sum::<f64>() - (0.42 - 1.1)).abs() < 1e-12);
    }

    #[test]
    fn sparse_spreading_equals_dense_spreading_bit_for_bit() {
        let g = Grid::centered([0.0; 3], [10.0; 3], [21; 3]).unwrap();
        let mut s: u64 = 5;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        // Many atoms share cells (sums of several contributions per node), some
        // sit exactly on nodes (zero weights, -0.0 contributions) or carry no charge.
        let mut atoms: Vec<Atom> = (0..400)
            .map(|_| Atom {
                position: [rnd(), rnd(), rnd()].map(|v| 6.0 * (v - 0.5)),
                charge: rnd() - 0.5,
                radius: 1.5,
            })
            .collect();
        atoms.push(Atom {
            position: [0.5, -1.0, 1.5],
            charge: -0.7,
            radius: 1.5,
        });
        atoms.push(Atom {
            position: [0.25, 0.0, 0.0],
            charge: 0.0,
            radius: 1.5,
        });
        for method in METHODS {
            let dense = spread_charges_with(&g, &atoms, method).unwrap();
            let (nodes, values) = spread_charges_sparse_with(&g, &atoms, method).unwrap();
            assert!(nodes.windows(2).all(|w| w[0] < w[1]));
            let mut rebuilt = vec![0.0; g.len()];
            for (c, v) in nodes.iter().zip(&values) {
                rebuilt[*c] = *v;
            }
            assert!(
                dense.iter().zip(&rebuilt).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{method:?}"
            );
        }
        let dense = spread_charges(&g, &atoms).unwrap();
        let trilinear = spread_charges_with(&g, &atoms, ChargeMethod::Trilinear).unwrap();
        assert!(dense.iter().zip(&trilinear).all(|(a, b)| a.to_bits() == b.to_bits()));
        // Same error for a charge outside the interior.
        atoms.push(Atom {
            position: [4.9, 0.0, 0.0],
            charge: 1.0,
            radius: 1.5,
        });
        let e1 = spread_charges(&g, &atoms).unwrap_err().to_string();
        let e2 = spread_charges_sparse(&g, &atoms).unwrap_err().to_string();
        assert_eq!(e1, e2);
    }
}
