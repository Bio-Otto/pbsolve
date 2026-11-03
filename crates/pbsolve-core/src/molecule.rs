//! Atoms and PQR input.

use crate::error::{invalid, PbError, Result};
use std::path::Path;

/// A point charge with a radius: everything the PB model needs per atom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atom {
    /// Position in Å.
    pub position: [f64; 3],
    /// Partial charge in e.
    pub charge: f64,
    /// Dielectric-boundary radius in Å (0 = does not shape the surface).
    pub radius: f64,
}

/// Builds atoms from parallel arrays, validating shapes and values.
pub fn atoms_from_arrays(coords: &[[f64; 3]], charges: &[f64], radii: &[f64]) -> Result<Vec<Atom>> {
    if coords.len() != charges.len() || coords.len() != radii.len() {
        return invalid(format!(
            "length mismatch: {} coordinates, {} charges, {} radii",
            coords.len(),
            charges.len(),
            radii.len()
        ));
    }
    if coords.is_empty() {
        return invalid("no atoms given");
    }
    let mut atoms = Vec::with_capacity(coords.len());
    for (n, ((p, q), r)) in coords.iter().zip(charges).zip(radii).enumerate() {
        if !p.iter().all(|v| v.is_finite()) || !q.is_finite() || !r.is_finite() || *r < 0.0 {
            return invalid(format!("atom {n} has a non-finite value or a negative radius"));
        }
        atoms.push(Atom {
            position: *p,
            charge: *q,
            radius: *r,
        });
    }
    Ok(atoms)
}

/// Parses PQR text. Only ATOM/HETATM records are read; the last five
/// whitespace-separated fields are taken as x, y, z, charge, radius, which
/// covers both the chain-ID and no-chain-ID PQR dialects.
pub fn parse_pqr(text: &str) -> Result<Vec<Atom>> {
    let mut atoms = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let rec = line.split_whitespace().next().unwrap_or("");
        if rec != "ATOM" && rec != "HETATM" {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 {
            return Err(PbError::Parse {
                line: lineno + 1,
                msg: "too few fields".into(),
            });
        }
        let tail = &fields[fields.len() - 5..];
        let mut v = [0.0f64; 5];
        for (slot, tok) in v.iter_mut().zip(tail) {
            *slot = tok.parse::<f64>().map_err(|_| PbError::Parse {
                line: lineno + 1,
                msg: format!(
                    "cannot parse '{tok}' as a number (fixed-width PQR with touching columns is not supported)"
                ),
            })?;
        }
        atoms.push(Atom {
            position: [v[0], v[1], v[2]],
            charge: v[3],
            radius: v[4],
        });
    }
    if atoms.is_empty() {
        return invalid("PQR contains no ATOM/HETATM records");
    }
    Ok(atoms)
}

pub fn read_pqr(path: impl AsRef<Path>) -> Result<Vec<Atom>> {
    parse_pqr(&std::fs::read_to_string(path)?)
}

/// Axis-aligned bounding box of the atoms, optionally inflated by each radius.
pub fn bounding_box(atoms: &[Atom], include_radius: bool) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for a in atoms {
        let r = if include_radius { a.radius } else { 0.0 };
        for d in 0..3 {
            lo[d] = lo[d].min(a.position[d] - r);
            hi[d] = hi[d].max(a.position[d] + r);
        }
    }
    (lo, hi)
}

pub fn total_charge(atoms: &[Atom]) -> f64 {
    atoms.iter().map(|a| a.charge).sum()
}

#[inline]
pub(crate) fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pqr_with_and_without_chain() {
        let text = "\
REMARK test
ATOM      1  N   ALA A   1      -1.000   2.000   3.500  0.1414 1.8240
ATOM      2  CA  ALA     1       0.000   0.000   0.000 -0.0912 1.9080
TER
END";
        let atoms = parse_pqr(text).unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].position, [-1.0, 2.0, 3.5]);
        assert!((atoms[1].charge + 0.0912).abs() < 1e-12);
        assert!((atoms[1].radius - 1.908).abs() < 1e-12);
    }
}
