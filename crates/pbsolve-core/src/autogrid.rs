// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Portions derived from APBS 3.4.1 (mgparm.c, nosh.c), Copyright (c) 2010-2022 Battelle
// Memorial Institute and other APBS contributors, BSD-3-Clause.
// See THIRD_PARTY_NOTICES.md for the full license text.

//! Grid specification for a focusing calculation, equivalent to APBS
//! `mg-auto` (`dime`, `cglen`, `fglen`, `cgcent` = `fgcent`). `auto`
//! follows the heuristic of APBS' `psize` helper (cfac, fadd).
//!
//! The sequence of focusing levels ([`GridSpec::levels`]) reproduces APBS 3.x
//! `NOsh_setupCalcMGAUTO`: the spacing is never reduced by more than
//! [`APBS_REDFRAC`] per level, so strongly focused grids get intermediate levels.

use crate::error::{invalid, Result};
use crate::grid::Grid;
use crate::molecule::{bounding_box, Atom};

#[derive(Clone, Debug, PartialEq)]
pub struct GridSpec {
    /// Common centre of the coarse and fine grids (Å).
    pub center: [f64; 3],
    /// Nodes per axis, shared by both levels (APBS `dime`).
    pub dims: [usize; 3],
    /// Coarse box edge lengths (APBS `cglen`).
    pub coarse_length: [f64; 3],
    /// Fine box edge lengths (APBS `fglen`).
    pub fine_length: [f64; 3],
}

/// Node counts are rounded up to `DIME_MULTIPLE·k + 1`: fine enough that
/// different spacings give different grids, and still compatible with a
/// 3-level geometric multigrid.
pub const DIME_MULTIPLE: usize = 8;

/// APBS keeps a `dime` above [`APBS_MAX_FREE_DIME`] only if it has the form
/// `32·c + 1` (four multigrid levels); otherwise `MGparm_check` silently
/// replaces it by a *smaller* value. See [`apbs_effective_dime`].
pub const APBS_DIME_MULTIPLE: usize = 32;

/// Largest `dime` APBS accepts without requiring four multigrid levels.
pub const APBS_MAX_FREE_DIME: usize = 65;

/// APBS `VREDFRAC`: the grid spacing shrinks by at most this factor per focusing level.
pub const APBS_REDFRAC: f64 = 0.25;

/// How [`GridSpec::auto_with`] rounds node counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DimePolicy {
    /// `8k+1` nodes: finest granularity, multigrid friendly (default).
    /// Grids above 65 nodes per axis are generally *not* accepted unchanged by APBS.
    #[default]
    Pbsolve,
    /// Node counts APBS uses unchanged: `8k+1` up to 65, `32c+1` above
    /// (the form APBS' `psize` produces). Use this for APBS comparisons.
    Apbs,
}

impl DimePolicy {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "pbsolve" | "default" => Ok(Self::Pbsolve),
            "apbs" => Ok(Self::Apbs),
            other => invalid(format!("unknown dime policy '{other}' (expected 'pbsolve' or 'apbs')")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pbsolve => "pbsolve",
            Self::Apbs => "apbs",
        }
    }
}

/// Smallest admissible node count ≥ the count needed for `length` at `spacing`.
fn dime_for(length: f64, spacing: f64, policy: DimePolicy) -> usize {
    let n = (length / spacing - 1e-9).ceil() as usize + 1;
    let k = (n - 1).div_ceil(DIME_MULTIPLE).max(2);
    let n8 = DIME_MULTIPLE * k + 1;
    match policy {
        DimePolicy::Pbsolve => n8,
        DimePolicy::Apbs if n8 <= APBS_MAX_FREE_DIME => n8,
        DimePolicy::Apbs => APBS_DIME_MULTIPLE * (n - 1).div_ceil(APBS_DIME_MULTIPLE) + 1,
    }
}

/// The `dime` APBS 3.x actually uses for a requested value, replicating
/// `MGparm_check` (with the default `nlev = 4`):
///
/// * values whose `dime − 1` admits fewer than 4 multigrid levels and that
///   exceed 65 are reset to `32·⌊dime/32⌋ + 1` (e.g. 121 → 97, 81 → 65);
/// * values admitting no multigrid level at all are rejected (`None`).
pub fn apbs_effective_dime(dime: usize) -> Option<usize> {
    if dime < 3 {
        return None;
    }
    let ti = dime - 1;
    if ti == APBS_DIME_MULTIPLE {
        return Some(dime);
    }
    let mut t = ti;
    let mut halvings = 0i32;
    while t % 2 == 0 {
        halvings += 1;
        t /= 2;
    }
    let tnlev = halvings - 1;
    if dime > APBS_MAX_FREE_DIME && tnlev < 4 {
        let c = (dime / APBS_DIME_MULTIPLE).max(1);
        return Some(APBS_DIME_MULTIPLE * c + 1);
    }
    if tnlev <= 0 {
        return None;
    }
    Some(dime)
}

impl GridSpec {
    /// APBS psize-like automatic sizing with the default [`DimePolicy::Pbsolve`].
    pub fn auto(atoms: &[Atom], spacing: f64, cfac: f64, fadd: f64) -> Result<Self> {
        Self::auto_with(atoms, spacing, cfac, fadd, DimePolicy::Pbsolve)
    }

    /// APBS psize-like automatic sizing:
    /// fine = extent + `fadd`, coarse = max(`cfac`·extent, fine),
    /// dims chosen (per `policy`) so the fine spacing is ≤ `spacing`.
    pub fn auto_with(atoms: &[Atom], spacing: f64, cfac: f64, fadd: f64, policy: DimePolicy) -> Result<Self> {
        if atoms.is_empty() {
            return invalid("no atoms");
        }
        if spacing.is_nan() || spacing <= 0.0 || cfac < 1.0 || fadd < 0.0 {
            return invalid("need spacing > 0, cfac >= 1, fadd >= 0");
        }
        let (lo, hi) = bounding_box(atoms, true);
        let (clo, chi) = bounding_box(atoms, false);
        let center = [0, 1, 2].map(|a| 0.5 * (clo[a] + chi[a]));
        let extent = [0, 1, 2].map(|a| (hi[a] - lo[a]).max(1.0));
        let fine_length = [0, 1, 2].map(|a| extent[a] + fadd);
        let coarse_length = [0, 1, 2].map(|a| (cfac * extent[a]).max(fine_length[a]));
        let dims = [0, 1, 2].map(|a| dime_for(fine_length[a], spacing, policy));
        Ok(Self {
            center,
            dims,
            coarse_length,
            fine_length,
        })
    }

    /// Single-level grid (no focusing).
    pub fn single(center: [f64; 3], length: [f64; 3], dims: [usize; 3]) -> Self {
        Self {
            center,
            dims,
            coarse_length: length,
            fine_length: length,
        }
    }

    /// This spec with a lateral period (Å along x and y): the outermost grid
    /// spans exactly one period in x and y, and finer grids no more than one.
    /// Applying it twice changes nothing; `None` returns the spec unchanged.
    pub fn with_period(&self, period: Option<[f64; 2]>) -> Self {
        let mut spec = self.clone();
        if let Some(p) = period {
            for (a, &l) in p.iter().enumerate() {
                spec.coarse_length[a] = l;
                spec.fine_length[a] = spec.fine_length[a].min(l);
            }
        }
        spec
    }

    pub fn is_focused(&self) -> bool {
        (0..3).any(|a| (self.coarse_length[a] - self.fine_length[a]).abs() > 1e-9)
    }

    /// Outermost grid (solved with the analytic boundary condition).
    pub fn coarse(&self) -> Result<Grid> {
        Ok(self.levels()?.swap_remove(0))
    }

    /// Finest grid (where energies and potentials are evaluated).
    pub fn fine(&self) -> Result<Grid> {
        Ok(self.levels()?.pop().expect("levels() is never empty"))
    }

    pub fn fine_spacing(&self) -> [f64; 3] {
        [0, 1, 2].map(|a| self.fine_length[a] / (self.dims[a] - 1) as f64)
    }

    /// All focusing levels, coarse first, exactly as APBS `mg-auto` sets them up
    /// (`NOsh_setupCalcMGAUTO`): `n = 2` levels unless the spacing ratio
    /// fine/coarse is below [`APBS_REDFRAC`] on some axis, in which case
    /// `n = ⌈ln r / ln 0.25⌉ + 1`; level spacings follow `h_i = ρ·h_{i−1}` with
    /// `ρ = (h_fine/h_coarse)^{1/(n−1)}`. All levels share `center` (APBS'
    /// mesh repositioning is a no-op for concentric grids). An unfocused spec
    /// has a single level (APBS repeats the same grid once, with identical result).
    pub fn levels(&self) -> Result<Vec<Grid>> {
        for a in 0..3 {
            if self.fine_length[a] > self.coarse_length[a] + 1e-9 {
                return invalid("fine grid must not be larger than the coarse grid");
            }
            if self.dims[a] < 5 {
                return invalid(format!("grid needs at least 5 nodes per axis, got {:?}", self.dims));
            }
        }
        let n1 = [0, 1, 2].map(|a| (self.dims[a] - 1) as f64);
        let hc = [0, 1, 2].map(|a| self.coarse_length[a] / n1[a]);
        let mut levels = vec![Grid::from_spacing(self.center, hc, self.dims)?];
        if !self.is_focused() {
            return Ok(levels);
        }
        let hf = [0, 1, 2].map(|a| self.fine_length[a] / n1[a]);
        let nfocus = (0..3)
            .map(|a| {
                let r = hf[a] / hc[a];
                if r < APBS_REDFRAC {
                    (r.ln() / APBS_REDFRAC.ln()).ceil() as usize + 1
                } else {
                    2
                }
            })
            .max()
            .unwrap_or(2);
        let redrat = [0, 1, 2].map(|a| (hf[a] / hc[a]).powf(1.0 / (nfocus as f64 - 1.0)));
        let mut h = hc;
        for _ in 1..nfocus {
            h = [0, 1, 2].map(|a| redrat[a] * h[a]);
            levels.push(Grid::from_spacing(self.center, h, self.dims)?);
        }
        Ok(levels)
    }

    pub fn validate(&self) -> Result<()> {
        self.levels().map(|_| ())
    }

    /// The node counts APBS would actually use for this spec (see
    /// [`apbs_effective_dime`]); `None` if APBS would reject one of them.
    pub fn apbs_dims(&self) -> Option<[usize; 3]> {
        let d = self.dims.map(apbs_effective_dime);
        Some([d[0]?, d[1]?, d[2]?])
    }

    /// Errors unless APBS would run this spec on exactly these grids.
    pub fn check_apbs(&self) -> Result<()> {
        match self.apbs_dims() {
            Some(d) if d == self.dims => Ok(()),
            Some(d) => invalid(format!(
                "APBS would silently replace dime {:?} by {d:?} (dime > 65 must be 32c+1); \
                 build the grid with dime policy 'apbs' to compare with APBS",
                self.dims
            )),
            None => invalid(format!(
                "APBS rejects dime {:?} (no multigrid level possible)",
                self.dims
            )),
        }
    }

    /// The equivalent APBS `mg-auto` grid keywords (for validation runs).
    /// Numbers are written in shortest round-trip form so APBS reads exactly
    /// the same doubles. Fails if APBS would change the grid ([`Self::check_apbs`]).
    pub fn to_apbs(&self) -> Result<String> {
        self.check_apbs()?;
        let v = |x: [f64; 3]| format!("{:?} {:?} {:?}", x[0], x[1], x[2]);
        Ok(format!(
            "dime {} {} {}\ncglen {}\nfglen {}\ncgcent {}\nfgcent {}",
            self.dims[0],
            self.dims[1],
            self.dims[2],
            v(self.coarse_length),
            v(self.fine_length),
            v(self.center),
            v(self.center),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dime_is_multigrid_friendly_and_fine_enough() {
        for policy in [DimePolicy::Pbsolve, DimePolicy::Apbs] {
            for (len, h) in [(24.0, 0.5), (70.0, 0.5), (33.0, 0.25), (16.0, 0.35), (120.0, 0.33)] {
                let n = dime_for(len, h, policy);
                assert_eq!((n - 1) % DIME_MULTIPLE, 0);
                assert!(len / (n - 1) as f64 <= h + 1e-12);
            }
        }
    }

    #[test]
    fn apbs_dime_reset_matches_mgparm_check() {
        // Observed with APBS 3.4.1: 121 -> 97, 105 -> 97 ("Reset dime ... (nlev = 4)").
        let cases = [
            (33, Some(33)),
            (41, Some(41)),
            (49, Some(49)),
            (65, Some(65)),
            (81, Some(65)),
            (89, Some(65)),
            (97, Some(97)),
            (105, Some(97)),
            (113, Some(97)),
            (121, Some(97)),
            (129, Some(129)),
            (161, Some(161)),
            (59, None),
        ];
        for (n, want) in cases {
            assert_eq!(apbs_effective_dime(n), want, "dime {n}");
        }
    }

    #[test]
    fn apbs_policy_dims_are_kept_by_apbs() {
        for len in [10.0, 30.0, 47.3, 64.0, 90.0, 133.7] {
            for h in [0.2, 0.333, 0.5, 0.75] {
                let n = dime_for(len, h, DimePolicy::Apbs);
                assert_eq!(apbs_effective_dime(n), Some(n), "len {len} h {h} -> {n}");
            }
        }
    }

    #[test]
    fn focusing_levels_follow_apbs() {
        let c = [1.0, 2.0, 3.0];
        let mild = GridSpec {
            center: c,
            dims: [33; 3],
            coarse_length: [40.0; 3],
            fine_length: [30.0; 3],
        };
        let l = mild.levels().unwrap();
        assert_eq!(l.len(), 2);
        assert!((l[1].spacing[0] - 30.0 / 32.0).abs() < 1e-14);

        // ratio 0.1 < 0.25: ceil(ln 0.1 / ln 0.25) + 1 = 3 levels, ρ = sqrt(0.1).
        let strong = GridSpec {
            fine_length: [4.0, 30.0, 30.0],
            ..mild.clone()
        };
        let l = strong.levels().unwrap();
        assert_eq!(l.len(), 3);
        assert!((l[1].spacing[0] - 1.25 * 0.1f64.sqrt()).abs() < 1e-14);
        assert!((l[2].spacing[0] - 4.0 / 32.0).abs() < 1e-14);
        assert!((l[2].center()[0] - c[0]).abs() < 1e-12);

        let flat = GridSpec::single(c, [20.0; 3], [33; 3]);
        assert_eq!(flat.levels().unwrap().len(), 1);
    }

    #[test]
    fn to_apbs_round_trips_and_refuses_reset_grids() {
        let g = GridSpec {
            center: [0.1 + 0.2, -1.0 / 3.0, 7.0],
            dims: [97, 65, 41],
            coarse_length: [60.123456789, 50.0, 40.0],
            fine_length: [50.0, 40.0, 30.0],
        };
        let s = g.to_apbs().unwrap();
        let cg: Vec<f64> = s
            .lines()
            .nth(3)
            .unwrap()
            .split_whitespace()
            .skip(1)
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(cg, g.center.to_vec());
        let bad = GridSpec {
            dims: [121, 65, 41],
            ..g
        };
        assert!(bad.to_apbs().is_err());
    }
}
