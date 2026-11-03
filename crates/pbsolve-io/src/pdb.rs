//! PDB and PQR.
//!
//! PDB: fixed columns (wwPDB v3.3), `MODEL`/`ENDMDL` → frames, `CRYST1` → cell,
//! `CONECT` → bonds. Alternate locations: the first altloc seen for an atom is
//! kept (blank or 'A' in practice), others are skipped with a note. Elements
//! come from columns 77–78, else from the atom name (see [`crate::elements`]).
//!
//! PQR: whitespace-separated (`ATOM serial name resname [chain] resid x y z q r`,
//! as written by pdb2pqr and pbsolve); if that fails, PDB fixed columns for
//! coordinates and whitespace for charge/radius (touching negative numbers).

use crate::elements;
use crate::{col, parse_err, path_str, AtomInfo, Frame, Result, Structure};
use std::collections::HashMap;
use std::path::Path;

fn parse_f(path: &str, line: usize, s: &str, what: &str) -> Result<f64> {
    s.trim()
        .parse::<f64>()
        .or_else(|_| parse_err(path, line, format!("cannot parse {what} '{}'", s.trim())))
}

/// Residue number; hybrid-36 or overflow ("****") values fall back to the previous one.
fn parse_resid(s: &str) -> Option<i64> {
    s.trim().parse::<i64>().ok()
}

fn cryst1(line: &str) -> Option<[f64; 6]> {
    let v: Vec<f64> = [(6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54)]
        .iter()
        .map(|&(a, b)| col(line, a, b).parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    // 1 Å cubes are placeholders written by many tools for non-periodic systems
    (v[0] > 1.0 || v[1] > 1.0 || v[2] > 1.0).then(|| [v[0], v[1], v[2], v[3], v[4], v[5]])
}

pub fn read_pdb(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let mut s = Structure::default();
    let mut frames: Vec<Frame> = Vec::new();
    let mut cur: Vec<[f64; 3]> = Vec::new();
    let mut cell: Option<[f64; 6]> = None;
    let mut first_model_done = false;
    let mut in_model = false;
    let mut serial_to_idx: HashMap<i64, usize> = HashMap::new();
    let mut seen_alt: HashMap<(String, i64, String, String, String), char> = HashMap::new();
    let mut skipped_alt = 0usize;
    let mut last_resid = 0i64;
    let mut conect: Vec<(i64, i64)> = Vec::new();
    let mut atom_in_model = 0usize;

    let finish_model = |cur: &mut Vec<[f64; 3]>, cell: Option<[f64; 6]>, frames: &mut Vec<Frame>| {
        if !cur.is_empty() {
            frames.push(Frame {
                coords: std::mem::take(cur),
                cell,
                ..Frame::default()
            });
        }
    };

    for (ln, line) in text.lines().enumerate() {
        let ln = ln + 1;
        let rec = line.get(..6).unwrap_or(line).trim_end();
        match rec {
            "MODEL" => {
                in_model = true;
                atom_in_model = 0;
            }
            "ENDMDL" => {
                finish_model(&mut cur, cell, &mut frames);
                if !s.atoms.is_empty() {
                    first_model_done = true;
                }
                in_model = false;
            }
            "CRYST1" => cell = cryst1(line),
            "END" if !in_model && !cur.is_empty() => {
                finish_model(&mut cur, cell, &mut frames);
                first_model_done = true;
            }
            "ATOM" | "HETATM" => {
                let name = col(line, 12, 16).to_string();
                let altloc = line.as_bytes().get(16).map(|&c| c as char).unwrap_or(' ');
                let resname = col(line, 17, 21).to_string();
                let chain = col(line, 21, 22).to_string();
                let resid = parse_resid(col(line, 22, 26)).unwrap_or(last_resid);
                last_resid = resid;
                let icode = col(line, 26, 27).to_string();
                if altloc != ' ' {
                    let key = (chain.clone(), resid, icode.clone(), resname.clone(), name.clone());
                    let first = *seen_alt.entry(key).or_insert(altloc);
                    if first != altloc {
                        skipped_alt += 1;
                        continue;
                    }
                }
                let x = parse_f(&p, ln, col(line, 30, 38), "x")?;
                let y = parse_f(&p, ln, col(line, 38, 46), "y")?;
                let z = parse_f(&p, ln, col(line, 46, 54), "z")?;
                if first_model_done {
                    // later models: coordinates only, same atom order
                    if atom_in_model >= s.atoms.len() {
                        return parse_err(&p, ln, "model has more atoms than the first model");
                    }
                    atom_in_model += 1;
                    cur.push([x, y, z]);
                    continue;
                }
                let hetero = rec == "HETATM";
                let raw4 = line.get(12..16);
                let element = elements::infer(col(line, 76, 78), &name, raw4, &resname, hetero, None);
                if let Ok(serial) = col(line, 6, 11).parse::<i64>() {
                    serial_to_idx.insert(serial, s.atoms.len());
                }
                s.atoms.push(AtomInfo {
                    name,
                    resname,
                    resid,
                    icode,
                    chain,
                    segid: col(line, 72, 76).to_string(),
                    element,
                    hetero,
                    ..AtomInfo::default()
                });
                cur.push([x, y, z]);
            }
            "CONECT" => {
                let a = col(line, 6, 11).parse::<i64>();
                if let Ok(a) = a {
                    for (lo, hi) in [(11, 16), (16, 21), (21, 26), (26, 31)] {
                        if let Ok(b) = col(line, lo, hi).parse::<i64>() {
                            conect.push((a, b));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    finish_model(&mut cur, cell, &mut frames);
    if s.atoms.is_empty() {
        return parse_err(&p, 0, "no ATOM/HETATM records");
    }
    for (i, f) in frames.iter().enumerate() {
        if f.coords.len() != s.atoms.len() {
            return parse_err(
                &p,
                0,
                format!(
                    "model {} has {} atoms, model 1 has {}",
                    i + 1,
                    f.coords.len(),
                    s.atoms.len()
                ),
            );
        }
    }
    for (a, b) in conect {
        if let (Some(&i), Some(&j)) = (serial_to_idx.get(&a), serial_to_idx.get(&b)) {
            if i < j {
                s.bonds.push((i, j));
            }
        }
    }
    if skipped_alt > 0 {
        s.notes.push(format!(
            "skipped {skipped_alt} alternate-location atoms (kept the first altloc of each atom)"
        ));
    }
    s.frames = frames;
    Ok(s)
}

/// Splits "52A" into (52, "A").
fn split_resid(tok: &str) -> Option<(i64, String)> {
    let digits_end = tok
        .char_indices()
        .find(|&(i, c)| !(c.is_ascii_digit() || (i == 0 && c == '-')))
        .map(|(i, _)| i)
        .unwrap_or(tok.len());
    let n = tok[..digits_end].parse::<i64>().ok()?;
    Some((n, tok[digits_end..].to_string()))
}

fn pqr_whitespace(fields: &[&str]) -> Option<(AtomInfo, [f64; 3])> {
    let n = fields.len();
    if n < 10 {
        return None;
    }
    let nums: Vec<f64> = fields[n - 5..]
        .iter()
        .map(|t| t.parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    let (chain, rtok) = match n {
        10 => (String::new(), fields[4]),
        11 => (fields[4].to_string(), fields[5]),
        _ => return None,
    };
    let (resid, icode) = split_resid(rtok)?;
    Some((
        AtomInfo {
            name: fields[2].to_string(),
            resname: fields[3].to_string(),
            resid,
            icode,
            chain,
            hetero: fields[0] == "HETATM",
            charge: Some(nums[3]),
            radius: Some(nums[4]),
            ..AtomInfo::default()
        },
        [nums[0], nums[1], nums[2]],
    ))
}

fn pqr_fixed(line: &str) -> Option<(AtomInfo, [f64; 3])> {
    let x = col(line, 30, 38).parse::<f64>().ok()?;
    let y = col(line, 38, 46).parse::<f64>().ok()?;
    let z = col(line, 46, 54).parse::<f64>().ok()?;
    let rest: Vec<f64> = line
        .get(54..)?
        .split_whitespace()
        .take(2)
        .map(|t| t.parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    if rest.len() != 2 {
        return None;
    }
    Some((
        AtomInfo {
            name: col(line, 12, 16).to_string(),
            resname: col(line, 17, 21).to_string(),
            chain: col(line, 21, 22).to_string(),
            resid: col(line, 22, 26).parse().ok()?,
            icode: col(line, 26, 27).to_string(),
            hetero: line.starts_with("HETATM"),
            charge: Some(rest[0]),
            radius: Some(rest[1]),
            ..AtomInfo::default()
        },
        [x, y, z],
    ))
}

pub fn read_pqr(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let mut s = Structure::default();
    let mut coords = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let rec = fields.first().copied().unwrap_or("");
        if rec != "ATOM" && rec != "HETATM" {
            continue;
        }
        let Some((mut a, x)) = pqr_whitespace(&fields).or_else(|| pqr_fixed(line)) else {
            return parse_err(
                &p,
                ln + 1,
                "cannot parse PQR record (expected '... x y z charge radius')",
            );
        };
        a.element = elements::infer("", &a.name, None, &a.resname, a.hetero, None);
        s.atoms.push(a);
        coords.push(x);
    }
    if s.atoms.is_empty() {
        return parse_err(&p, 0, "no ATOM/HETATM records");
    }
    s.frames.push(Frame {
        coords,
        ..Frame::default()
    });
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, text: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pbsolve-io-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(name);
        std::fs::File::create(&p).unwrap().write_all(text.as_bytes()).unwrap();
        p
    }

    #[test]
    fn pdb_models_altlocs_elements() {
        let t = "\
CRYST1   50.000   60.000   70.000  90.00  90.00  90.00 P 1           1
MODEL        1
ATOM      1  N   ALA A   1      -1.000   2.000   3.500  1.00  0.00           N
ATOM      2  CA AALA A   1       0.000   0.000   0.000  0.50  0.00
ATOM      3  CA BALA A   1       0.100   0.000   0.000  0.50  0.00
HETATM    4 FE   HEM A 201      10.000  10.000  10.000  1.00  0.00
HETATM    5  NA  HEM A 201      11.000  10.000  10.000  1.00  0.00
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1      -1.500   2.000   3.500  1.00  0.00           N
ATOM      2  CA AALA A   1       0.500   0.000   0.000  0.50  0.00
ATOM      3  CA BALA A   1       0.600   0.000   0.000  0.50  0.00
HETATM    4 FE   HEM A 201      10.000  10.000  10.000  1.00  0.00
HETATM    5  NA  HEM A 201      11.000  10.000  10.000  1.00  0.00
ENDMDL
CONECT    4    5
END
";
        let s = read_pdb(&tmp("a.pdb", t)).unwrap();
        assert_eq!(s.atoms.len(), 4);
        assert_eq!(s.frames.len(), 2);
        assert_eq!(s.frames[1].coords[0], [-1.5, 2.0, 3.5]);
        assert_eq!(s.frames[1].coords[1], [0.5, 0.0, 0.0]);
        let el: Vec<&str> = s.atoms.iter().map(|a| a.element.as_str()).collect();
        assert_eq!(el, ["N", "C", "Fe", "N"]);
        assert_eq!(s.bonds, vec![(2, 3)]);
        assert_eq!(s.frames[0].cell.unwrap()[1], 60.0);
        assert_eq!(s.notes.len(), 1);
    }

    #[test]
    fn pqr_dialects() {
        let t = "\
ATOM      1  N   ALA A   1      -1.000   2.000   3.500  0.1414 1.8240
ATOM      2  CA  ALA     1       0.000   0.000   0.000 -0.0912 1.9080
ATOM      3  C   ALA A  52A      1.000   0.000   0.000  0.5973 1.9080
ATOM      4  O   ALA A  52     -10.000-100.000-200.000 -0.5679 1.6612
";
        let s = read_pqr(&tmp("a.pqr", t)).unwrap();
        assert_eq!(s.atoms.len(), 4);
        assert_eq!(s.atoms[0].chain, "A");
        assert_eq!(s.atoms[1].chain, "");
        assert_eq!((s.atoms[2].resid, s.atoms[2].icode.as_str()), (52, "A"));
        assert_eq!(s.frames[0].coords[3], [-10.0, -100.0, -200.0]);
        assert_eq!(s.atoms[3].charge, Some(-0.5679));
        assert_eq!(s.atoms[3].element, "O");
    }
}
