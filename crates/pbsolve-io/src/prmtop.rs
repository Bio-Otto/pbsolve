//! Amber prmtop (parm7, including CHAMBER files). Charges are stored ×18.2223
//! (Amber's internal unit); `RADII` are the PB/GB radii set by tleap
//! (`RADIUS_SET` is reported in the notes). Old pre-%FLAG files are rejected.

use crate::elements;
use crate::{format_err, parse_err, path_str, AtomInfo, Result, Structure};
use std::collections::HashMap;
use std::path::Path;

/// Amber's charge unit: q_prmtop = 18.2223 · q[e].
pub const AMBER_CHARGE_FACTOR: f64 = 18.2223;

#[derive(Clone, Copy, Debug)]
struct FortranFormat {
    per_line: usize,
    width: usize,
}

/// Parses "(20a4)", "(5E16.8)", "(10I8)", "(1a80)", "(3I8)" ...
fn parse_format(f: &str) -> Option<FortranFormat> {
    let inner = f.trim().trim_start_matches('(').trim_end_matches(')');
    let pos = inner.find(|c: char| c.is_ascii_alphabetic())?;
    let per_line = if pos == 0 { 1 } else { inner[..pos].parse().ok()? };
    let rest = &inner[pos + 1..];
    let width = rest.split('.').next()?.parse().ok()?;
    Some(FortranFormat { per_line, width })
}

struct Sections {
    data: HashMap<String, Vec<String>>,
}

impl Sections {
    fn parse(p: &str, text: &str) -> Result<Self> {
        let mut data: HashMap<String, Vec<String>> = HashMap::new();
        let mut cur: Option<(String, Option<FortranFormat>)> = None;
        for (ln, line) in text.lines().enumerate() {
            if let Some(rest) = line.strip_prefix("%FLAG") {
                let name = rest.trim().to_string();
                data.entry(name.clone()).or_default();
                cur = Some((name, None));
                continue;
            }
            if let Some(rest) = line.strip_prefix("%FORMAT") {
                let Some((_, fmt)) = cur.as_mut() else {
                    return parse_err(p, ln + 1, "%FORMAT before %FLAG");
                };
                *fmt = parse_format(rest);
                if fmt.is_none() {
                    return parse_err(p, ln + 1, format!("unsupported format {}", rest.trim()));
                }
                continue;
            }
            if line.starts_with("%VERSION") || line.starts_with("%COMMENT") {
                continue;
            }
            let Some((name, Some(fmt))) = cur.as_ref() else {
                continue;
            };
            let v = data.get_mut(name).expect("inserted at %FLAG");
            let mut i = 0;
            let n = line.len();
            let mut k = 0;
            while i < n && k < fmt.per_line {
                let end = (i + fmt.width).min(n);
                v.push(line.get(i..end).unwrap_or("").to_string());
                i = end;
                k += 1;
            }
        }
        Ok(Self { data })
    }

    fn strings(&self, flag: &str) -> Option<Vec<String>> {
        self.data
            .get(flag)
            .map(|v| v.iter().map(|s| s.trim().to_string()).collect())
    }

    fn floats(&self, p: &str, flag: &str) -> Result<Option<Vec<f64>>> {
        let Some(v) = self.data.get(flag) else { return Ok(None) };
        v.iter()
            .map(|s| {
                // Fortran "D" exponents
                s.trim()
                    .replace(['D', 'd'], "E")
                    .parse::<f64>()
                    .or_else(|_| parse_err(p, 0, format!("%FLAG {flag}: cannot parse '{}'", s.trim())))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    fn ints(&self, p: &str, flag: &str) -> Result<Option<Vec<i64>>> {
        let Some(v) = self.data.get(flag) else { return Ok(None) };
        v.iter()
            .map(|s| {
                s.trim()
                    .parse::<i64>()
                    .or_else(|_| parse_err(p, 0, format!("%FLAG {flag}: bad integer '{}'", s.trim())))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
}

/// Extra per-file data Amber users need (box, molecules, solvent).
#[derive(Clone, Debug, Default)]
pub struct PrmtopInfo {
    /// IFBOX > 0: periodic (mdcrd frames have a box line).
    pub periodic: bool,
    pub radius_set: String,
    /// Atoms per molecule (periodic topologies only).
    pub atoms_per_molecule: Vec<usize>,
}

pub fn read_prmtop(path: &Path) -> Result<Structure> {
    read_prmtop_info(path).map(|(mut s, info)| {
        s.periodic = Some(info.periodic);
        s
    })
}

pub fn read_prmtop_info(path: &Path) -> Result<(Structure, PrmtopInfo)> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    if !text.contains("%FLAG") {
        return format_err(format!("{p}: old-style (pre-Amber 7) prmtop files are not supported"));
    }
    let sec = Sections::parse(&p, &text)?;
    let pointers = sec
        .ints(&p, "POINTERS")?
        .ok_or_else(|| crate::IoError::Format(format!("{p}: no POINTERS")))?;
    let natom = pointers[0] as usize;
    let nres = pointers.get(11).copied().unwrap_or(0) as usize;
    let ifbox = pointers.get(27).copied().unwrap_or(0);

    let need = |v: Option<Vec<String>>, flag: &str| -> Result<Vec<String>> {
        match v {
            Some(x) if x.len() >= natom => Ok(x),
            _ => format_err(format!("{p}: %FLAG {flag} missing or shorter than NATOM={natom}")),
        }
    };
    let names = need(sec.strings("ATOM_NAME"), "ATOM_NAME")?;
    let charges = sec.floats(&p, "CHARGE")?.unwrap_or_default();
    if charges.len() < natom {
        return format_err(format!("{p}: %FLAG CHARGE missing or too short"));
    }
    let masses = sec.floats(&p, "MASS")?;
    let radii = sec.floats(&p, "RADII")?;
    let z = sec.ints(&p, "ATOMIC_NUMBER")?;
    let types = sec.strings("AMBER_ATOM_TYPE");
    let reslab = sec.strings("RESIDUE_LABEL").unwrap_or_default();
    let resptr = sec.ints(&p, "RESIDUE_POINTER")?.unwrap_or_default();
    let resnum = sec.ints(&p, "RESIDUE_NUMBER")?;
    let chainid = sec.strings("RESIDUE_CHAINID");
    if reslab.len() < nres || resptr.len() < nres {
        return format_err(format!("{p}: residue tables shorter than NRES={nres}"));
    }

    let mut s = Structure::default();
    for (r, &start) in resptr.iter().enumerate().take(nres) {
        let end = if r + 1 < nres {
            resptr[r + 1] as usize - 1
        } else {
            natom
        };
        for i in (start as usize - 1)..end {
            let mass = masses.as_ref().and_then(|m| m.get(i).copied());
            let el = match z.as_ref().and_then(|z| z.get(i).copied()) {
                Some(zz) if zz > 0 => elements::from_atomic_number(zz).unwrap_or("").to_string(),
                Some(_) => String::new(), // Z <= 0: extra point / virtual site
                None => elements::infer("", &names[i], None, &reslab[r], false, mass),
            };
            s.atoms.push(AtomInfo {
                name: names[i].clone(),
                resname: reslab[r].clone(),
                resid: resnum.as_ref().and_then(|v| v.get(r).copied()).unwrap_or(r as i64 + 1),
                chain: chainid.as_ref().and_then(|v| v.get(r).cloned()).unwrap_or_default(),
                element: el,
                charge: Some(charges[i] / AMBER_CHARGE_FACTOR),
                radius: radii.as_ref().and_then(|v| v.get(i).copied()),
                mass,
                atom_type: types.as_ref().and_then(|t| t.get(i).cloned()).unwrap_or_default(),
                ..AtomInfo::default()
            });
        }
    }
    if s.atoms.len() != natom {
        return format_err(format!(
            "{p}: residue pointers cover {} atoms, NATOM is {natom}",
            s.atoms.len()
        ));
    }
    for flag in ["BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"] {
        if let Some(b) = sec.ints(&p, flag)? {
            for t in b.chunks(3) {
                if t.len() == 3 {
                    let (i, j) = ((t[0] / 3) as usize, (t[1] / 3) as usize);
                    if i < natom && j < natom {
                        s.bonds.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
    }
    let radius_set = sec
        .strings("RADIUS_SET")
        .and_then(|v| v.first().cloned())
        .unwrap_or_default();
    if radii.is_none() {
        s.notes.push("prmtop has no RADII section".into());
    } else if !radius_set.is_empty() {
        s.notes.push(format!("prmtop radii: {radius_set}"));
    }
    let atoms_per_molecule = sec
        .ints(&p, "ATOMS_PER_MOLECULE")?
        .unwrap_or_default()
        .into_iter()
        .map(|n| n as usize)
        .collect();
    let info = PrmtopInfo {
        periodic: ifbox > 0,
        radius_set,
        atoms_per_molecule,
    };
    Ok((s, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fortran_formats() {
        let f = parse_format("(5E16.8)").unwrap();
        assert_eq!((f.per_line, f.width), (5, 16));
        let f = parse_format("(20a4)").unwrap();
        assert_eq!((f.per_line, f.width), (20, 4));
        let f = parse_format("(a80)").unwrap();
        assert_eq!((f.per_line, f.width), (1, 80));
    }
}
