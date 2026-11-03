//! Charge discretisation (APBS `chgm spl0`: trilinear spreading).

use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::Atom;

/// Spreads point charges to the 8 surrounding nodes. Returns the node charge
/// in e (not a density). Every charge must lie strictly inside the grid so
/// that no charge lands on a Dirichlet boundary node.
pub fn spread_charges(grid: &Grid, atoms: &[Atom]) -> Result<Vec<f64>> {
    let mut q = vec![0.0; grid.len()];
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
            q[idx[m]] += a.charge * w[m];
        }
    }
    Ok(q)
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
}
