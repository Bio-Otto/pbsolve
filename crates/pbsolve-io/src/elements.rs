//! Chemical elements and element inference for formats that do not store them.
//!
//! Preferred sources, in order: an explicit element column or atomic number;
//! the atom name (PDB conventions, CHARMM ion names); the mass. Masses alone
//! are unreliable under hydrogen-mass repartitioning, so they are the last resort.

/// (symbol, atomic number, standard atomic mass) for Z = 1..=54 plus common heavier ions.
const TABLE: &[(&str, u32, f64)] = &[
    ("H", 1, 1.008),
    ("He", 2, 4.0026),
    ("Li", 3, 6.94),
    ("Be", 4, 9.0122),
    ("B", 5, 10.81),
    ("C", 6, 12.011),
    ("N", 7, 14.007),
    ("O", 8, 15.999),
    ("F", 9, 18.998),
    ("Ne", 10, 20.180),
    ("Na", 11, 22.990),
    ("Mg", 12, 24.305),
    ("Al", 13, 26.982),
    ("Si", 14, 28.085),
    ("P", 15, 30.974),
    ("S", 16, 32.06),
    ("Cl", 17, 35.45),
    ("Ar", 18, 39.948),
    ("K", 19, 39.098),
    ("Ca", 20, 40.078),
    ("Sc", 21, 44.956),
    ("Ti", 22, 47.867),
    ("V", 23, 50.942),
    ("Cr", 24, 51.996),
    ("Mn", 25, 54.938),
    ("Fe", 26, 55.845),
    ("Co", 27, 58.933),
    ("Ni", 28, 58.693),
    ("Cu", 29, 63.546),
    ("Zn", 30, 65.38),
    ("Ga", 31, 69.723),
    ("Ge", 32, 72.630),
    ("As", 33, 74.922),
    ("Se", 34, 78.971),
    ("Br", 35, 79.904),
    ("Kr", 36, 83.798),
    ("Rb", 37, 85.468),
    ("Sr", 38, 87.62),
    ("Y", 39, 88.906),
    ("Zr", 40, 91.224),
    ("Mo", 42, 95.95),
    ("Ru", 44, 101.07),
    ("Rh", 45, 102.91),
    ("Pd", 46, 106.42),
    ("Ag", 47, 107.87),
    ("Cd", 48, 112.41),
    ("Sn", 50, 118.71),
    ("I", 53, 126.90),
    ("Xe", 54, 131.29),
    ("Cs", 55, 132.91),
    ("Ba", 56, 137.33),
    ("Gd", 64, 157.25),
    ("Pt", 78, 195.08),
    ("Au", 79, 196.97),
    ("Hg", 80, 200.59),
    ("Pb", 82, 207.2),
];

/// Standard capitalisation of an element symbol if it is known ("CL" -> "Cl").
pub fn normalize(sym: &str) -> Option<&'static str> {
    let s = sym.trim();
    if s.is_empty() {
        return None;
    }
    TABLE
        .iter()
        .find(|(e, _, _)| e.eq_ignore_ascii_case(s))
        .map(|(e, _, _)| *e)
}

pub fn from_atomic_number(z: i64) -> Option<&'static str> {
    TABLE.iter().find(|(_, n, _)| *n as i64 == z).map(|(e, _, _)| *e)
}

pub fn atomic_number(sym: &str) -> Option<u32> {
    TABLE.iter().find(|(e, _, _)| *e == sym).map(|(_, z, _)| *z)
}

pub fn mass(sym: &str) -> Option<f64> {
    TABLE.iter().find(|(e, _, _)| *e == sym).map(|(_, _, m)| *m)
}

/// Element whose standard mass is closest to `m` (within 0.6 amu), `None`
/// for massless particles (virtual sites).
pub fn from_mass(m: f64) -> Option<&'static str> {
    if m.is_nan() || m < 0.5 {
        return None;
    }
    let (e, _, mm) = TABLE
        .iter()
        .min_by(|a, b| (a.2 - m).abs().total_cmp(&(b.2 - m).abs()))
        .expect("table is not empty");
    ((mm - m).abs() < 0.6).then_some(*e)
}

/// Common ion residue/atom names (CHARMM, Amber, GROMACS) -> element.
fn ion_name(name: &str) -> Option<&'static str> {
    Some(match name.to_ascii_uppercase().trim_end_matches(['+', '-']) {
        "SOD" | "NA" | "NA+" | "NAI" => "Na",
        "POT" | "K" => "K",
        "CLA" | "CL" | "CLI" => "Cl",
        "CAL" | "CA2" => "Ca",
        "MG" | "MG2" => "Mg",
        "ZN" | "ZN2" => "Zn",
        "CES" | "CS" => "Cs",
        "LIT" | "LI" => "Li",
        "RUB" | "RB" => "Rb",
        "BR" => "Br",
        "IOD" => "I",
        _ => return None,
    })
}

/// Guesses the element from a PDB-style atom name.
///
/// * `raw4` is the 4-character name field (columns 13–16) if available: by
///   PDB convention two-letter elements start in column 13 ("FE  "), one-letter
///   ones in column 14 (" CA ").
/// * Monoatomic residues and HETATM records may carry two-letter elements
///   ("CL", "ZN"); in standard residues "CA" is a carbon.
pub fn guess_from_name(name: &str, raw4: Option<&str>, resname: &str, hetero: bool) -> Option<&'static str> {
    let n = name.trim();
    if n.is_empty() {
        return None;
    }
    let single_atom_residue = n.eq_ignore_ascii_case(resname.trim()) || ion_name(resname.trim()).is_some();
    if single_atom_residue {
        if let Some(e) = ion_name(n).or_else(|| ion_name(resname.trim())) {
            return Some(e);
        }
        if let Some(e) = normalize(n) {
            return Some(e);
        }
    }
    let letters: String = n.chars().skip_while(|c| c.is_ascii_digit()).collect();
    if letters.is_empty() {
        return None;
    }
    // A name shorter than 4 characters that starts in column 13 marks a
    // two-letter element ("FE  ", "ZN  "); 4-character names start there anyway.
    if let Some(r) = raw4 {
        let b = r.as_bytes();
        let left_justified = b.len() == 4 && b[0].is_ascii_alphabetic() && b[1].is_ascii_alphabetic() && b[3] == b' ';
        if left_justified && hetero {
            if let Some(e) = normalize(&r[..2]) {
                return Some(e);
            }
        }
    }
    normalize(&letters[..1])
}

/// Best available element for an atom: explicit symbol, then name, then mass.
pub fn infer(explicit: &str, name: &str, raw4: Option<&str>, resname: &str, hetero: bool, m: Option<f64>) -> String {
    if let Some(e) = normalize(explicit) {
        return e.to_string();
    }
    if m.is_some_and(|mm| mm < 0.5) {
        return String::new(); // massless particle: virtual site
    }
    let by_name = guess_from_name(name, raw4, resname, hetero);
    let by_mass = m.and_then(from_mass);
    match (by_name, by_mass) {
        // "CL1" with 35.45 amu is chlorine, "BR" with 79.9 amu bromine. Otherwise the
        // name wins: masses are shifted by hydrogen-mass repartitioning.
        (Some(a), Some(b)) if a != b && b.starts_with(a) => b.to_string(),
        (Some(a), _) => a.to_string(),
        (None, Some(b)) => b.to_string(),
        (None, None) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_pdb_conventions() {
        assert_eq!(guess_from_name("CA", Some(" CA "), "ALA", false), Some("C"));
        assert_eq!(guess_from_name("CA", Some("CA  "), "CA", true), Some("Ca"));
        assert_eq!(guess_from_name("FE", Some("FE  "), "HEM", true), Some("Fe"));
        assert_eq!(guess_from_name("1HB", Some("1HB "), "ALA", false), Some("H"));
        assert_eq!(guess_from_name("HB1", None, "ALA", false), Some("H"));
        assert_eq!(guess_from_name("SOD", None, "SOD", false), Some("Na"));
        assert_eq!(guess_from_name("Cl-", None, "Cl-", false), Some("Cl"));
        assert_eq!(guess_from_name("OW", None, "SOL", false), Some("O"));
    }

    #[test]
    fn masses_and_virtual_sites() {
        assert_eq!(from_mass(12.01), Some("C"));
        assert_eq!(from_mass(0.0), None);
        // hydrogen-mass repartitioning: an H of 3.024 amu is still an H by name
        assert_eq!(infer("", "HA", None, "ALA", false, Some(3.024)), "H");
        assert_eq!(infer("", "MW", None, "SOL", false, Some(0.0)), "");
        assert_eq!(infer("cl", "X", None, "LIG", true, None), "Cl");
        // HMR nitrogen (14.007 - 2.016) looks like carbon by mass
        assert_eq!(infer("", "N", None, "ALA", false, Some(11.991)), "N");
        assert_eq!(infer("", "CL1", None, "LIG", false, Some(35.45)), "Cl");
        assert_eq!(infer("", "NA", Some(" NA "), "HEM", true, Some(14.007)), "N");
    }
}
