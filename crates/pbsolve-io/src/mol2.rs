//! Tripos mol2: atoms with partial charges, bonds. Several molecules with the
//! same atoms (e.g. docking poses) become frames; otherwise only the first is read.

use crate::elements;
use crate::{parse_err, path_str, AtomInfo, Frame, Result, Structure};
use std::path::Path;

/// Element from a SYBYL type ("C.3" -> C, "Cl" -> Cl, "LP"/"Du" -> none).
fn element_of_type(t: &str) -> Option<&'static str> {
    let base = t.split('.').next().unwrap_or("");
    if base.eq_ignore_ascii_case("LP") || base.eq_ignore_ascii_case("Du") || base.eq_ignore_ascii_case("Any") {
        return None;
    }
    elements::normalize(base)
}

pub fn read_mol2(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let mut s = Structure::default();
    let mut section = "";
    let mut coords: Vec<[f64; 3]> = Vec::new();
    let mut id_to_idx: std::collections::HashMap<String, usize> = Default::default();
    let mut done_first = false;
    for (ln, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(sec) = t.strip_prefix("@<TRIPOS>") {
            section = match sec {
                "MOLECULE" => "MOLECULE",
                "ATOM" => "ATOM",
                "BOND" => "BOND",
                _ => "",
            };
            if section == "MOLECULE" && !coords.is_empty() {
                s.frames.push(Frame {
                    coords: std::mem::take(&mut coords),
                    ..Frame::default()
                });
                done_first = true;
            }
            continue;
        }
        match section {
            "ATOM" => {
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() < 6 {
                    return parse_err(&p, ln + 1, "ATOM line needs id name x y z type");
                }
                let xyz: Vec<f64> = match f[2..5].iter().map(|v| v.parse::<f64>()).collect() {
                    Ok(v) => v,
                    Err(_) => return parse_err(&p, ln + 1, "bad coordinates"),
                };
                coords.push([xyz[0], xyz[1], xyz[2]]);
                if done_first {
                    continue;
                }
                let name = f[1].to_string();
                let resname = f.get(7).copied().unwrap_or("UNL").to_string();
                let element = element_of_type(f[5])
                    .map(str::to_string)
                    .unwrap_or_else(|| elements::infer("", &name, None, &resname, true, None));
                id_to_idx.insert(f[0].to_string(), s.atoms.len());
                s.atoms.push(AtomInfo {
                    name,
                    resid: f.get(6).and_then(|v| v.parse().ok()).unwrap_or(1),
                    resname,
                    element,
                    hetero: true,
                    charge: f.get(8).and_then(|v| v.parse().ok()),
                    atom_type: f[5].to_string(),
                    ..AtomInfo::default()
                });
            }
            "BOND" if !done_first => {
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() >= 3 {
                    if let (Some(&a), Some(&b)) = (id_to_idx.get(f[1]), id_to_idx.get(f[2])) {
                        s.bonds.push((a.min(b), a.max(b)));
                    }
                }
            }
            _ => {}
        }
    }
    if !coords.is_empty() {
        s.frames.push(Frame {
            coords,
            ..Frame::default()
        });
    }
    if s.atoms.is_empty() {
        return parse_err(&p, 0, "no @<TRIPOS>ATOM records");
    }
    let first_atoms = s.atoms.len();
    if let Some(k) = s.frames.iter().position(|f| f.coords.len() != first_atoms) {
        s.frames.truncate(k.max(1));
        s.notes.push(format!(
            "mol2 molecule {} differs from the first one; only molecules 1..{} were read",
            k + 1,
            k.max(1)
        ));
    }
    if s.atoms.iter().any(|a| a.charge.is_none()) {
        s.notes.push("some mol2 atoms have no charge column".into());
    }
    Ok(s)
}
