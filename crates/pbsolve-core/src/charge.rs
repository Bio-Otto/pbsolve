//! Charge discretisation (APBS `chgm spl0`: trilinear spreading).

use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::Atom;

/// Spreads point charges to the 8 surrounding nodes. Returns the node charge
/// in e (not a density). Every charge must lie strictly inside the grid so
/// that no charge lands on a Dirichlet boundary node.
pub fn spread_charges(grid: &Grid, atoms: &[Atom]) -> Result<Vec<f64>> {
    let mut q = vec![0.0; grid.len()];
    for_each_contribution(grid, atoms, |c, v| q[c] += v)?;
    Ok(q)
}

/// [`spread_charges`] as a sparse vector: the nodes that receive charge, in
/// increasing order, and their charges. Each node sums its contributions in
/// the order `spread_charges` adds them (atoms in order, then the 8 corners),
/// starting from 0.0, so the values are identical bit for bit; nodes not
/// listed are 0.0.
pub fn spread_charges_sparse(grid: &Grid, atoms: &[Atom]) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut contrib: Vec<(usize, f64)> = Vec::with_capacity(8 * atoms.len());
    for_each_contribution(grid, atoms, |c, v| contrib.push((c, v)))?;
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

/// Calls `f(node, charge · weight)` for every charged atom and stencil corner,
/// in atom order.
fn for_each_contribution(grid: &Grid, atoms: &[Atom], mut f: impl FnMut(usize, f64)) -> Result<()> {
    for (n, a) in atoms.iter().enumerate() {
        if a.charge == 0.0 {
            continue;
        }
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let dense = spread_charges(&g, &atoms).unwrap();
        let (nodes, values) = spread_charges_sparse(&g, &atoms).unwrap();
        assert!(nodes.windows(2).all(|w| w[0] < w[1]));
        let mut rebuilt = vec![0.0; g.len()];
        for (c, v) in nodes.iter().zip(&values) {
            rebuilt[*c] = *v;
        }
        assert!(dense.iter().zip(&rebuilt).all(|(a, b)| a.to_bits() == b.to_bits()));
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
