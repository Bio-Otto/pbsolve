//! CHARMM/NAMD/X-PLOR PSF: atoms (segid, resid, resname, name, type, charge,
//! mass) and bonds. Standard, EXT, CHEQ and X-PLOR (string types) flavours.
//! Elements come from the atom name (CHARMM ion names understood), corrected
//! by the mass for two-letter elements.

use crate::elements;
use crate::{parse_err, path_str, AtomInfo, Result, Structure};
use std::path::Path;

fn section_count(line: &str, tag: &str) -> Option<usize> {
    let (n, rest) = line.trim().split_once(char::is_whitespace)?;
    rest.trim_start().starts_with(tag).then(|| n.parse().ok())?
}

pub fn read_psf(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    if !lines.first().is_some_and(|l| l.trim_start().starts_with("PSF")) {
        return parse_err(&p, 1, "not a PSF file (first line must start with 'PSF')");
    }
    let mut s = Structure::default();
    let mut i = 1;
    while i < lines.len() {
        if let Some(n) = section_count(lines[i], "!NATOM") {
            for k in 0..n {
                let ln = i + 1 + k;
                let Some(l) = lines.get(ln) else {
                    return parse_err(&p, ln + 1, "file ends inside !NATOM");
                };
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.len() < 8 {
                    return parse_err(
                        &p,
                        ln + 1,
                        "atom line needs id segid resid resname name type charge mass",
                    );
                }
                let (Ok(q), Ok(m)) = (f[6].parse::<f64>(), f[7].parse::<f64>()) else {
                    return parse_err(&p, ln + 1, "cannot parse charge/mass");
                };
                let (resid, icode) = match f[2].parse::<i64>() {
                    Ok(r) => (r, String::new()),
                    Err(_) => {
                        let d: String = f[2].chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
                        (d.parse().unwrap_or(0), f[2][d.len()..].to_string())
                    }
                };
                let element = elements::infer("", f[4], None, f[3], false, Some(m));
                s.atoms.push(AtomInfo {
                    segid: f[1].to_string(),
                    resid,
                    icode,
                    resname: f[3].to_string(),
                    name: f[4].to_string(),
                    atom_type: f[5].to_string(),
                    charge: Some(q),
                    mass: Some(m),
                    element,
                    ..AtomInfo::default()
                });
            }
            i += 1 + n;
            continue;
        }
        if let Some(n) = section_count(lines[i], "!NBOND") {
            let mut vals = Vec::with_capacity(2 * n);
            let mut ln = i + 1;
            while vals.len() < 2 * n && ln < lines.len() {
                for t in lines[ln].split_whitespace() {
                    match t.parse::<usize>() {
                        Ok(v) if v >= 1 => vals.push(v - 1),
                        _ => return parse_err(&p, ln + 1, format!("bad bond index '{t}'")),
                    }
                }
                ln += 1;
            }
            for b in vals.chunks(2).filter(|c| c.len() == 2) {
                s.bonds.push((b[0].min(b[1]), b[0].max(b[1])));
            }
            i = ln;
            continue;
        }
        i += 1;
    }
    if s.atoms.is_empty() {
        return parse_err(&p, 0, "no !NATOM section");
    }
    if s.bonds.iter().any(|&(a, b)| b >= s.atoms.len() || a == b) {
        return parse_err(&p, 0, "bond refers to a non-existent atom");
    }
    Ok(s)
}
