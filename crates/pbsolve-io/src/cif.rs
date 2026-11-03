//! PDBx/mmCIF: the `_atom_site` loop (author naming preferred, as in PDB
//! files), models → frames, `_cell` → cell. Alternate locations as in PDB.

use crate::elements;
use crate::{parse_err, path_str, AtomInfo, Frame, Result, Structure};
use std::collections::HashMap;
use std::path::Path;

/// CIF tokens: bare words, quoted strings and ';'-delimited text fields.
fn tokenize(text: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut lines = text.lines().enumerate().peekable();
    while let Some((ln, line)) = lines.next() {
        if let Some(first) = line.strip_prefix(';') {
            let mut buf = first.to_string();
            for (_, l) in lines.by_ref() {
                if l.starts_with(';') {
                    break;
                }
                buf.push('\n');
                buf.push_str(l);
            }
            out.push((buf, ln + 1));
            continue;
        }
        let b = line.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c.is_ascii_whitespace() {
                i += 1;
            } else if c == b'#' {
                break;
            } else if c == b'\'' || c == b'"' {
                // a quote closes only when followed by whitespace or end of line
                let mut j = i + 1;
                while j < b.len() && !(b[j] == c && (j + 1 == b.len() || b[j + 1].is_ascii_whitespace())) {
                    j += 1;
                }
                out.push((line[i + 1..j.min(b.len())].to_string(), ln + 1));
                i = j + 1;
            } else {
                let mut j = i;
                while j < b.len() && !b[j].is_ascii_whitespace() {
                    j += 1;
                }
                out.push((line[i..j].to_string(), ln + 1));
                i = j;
            }
        }
    }
    out
}

pub fn read_cif(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let toks = tokenize(&text);
    let mut cell: HashMap<String, f64> = HashMap::new();
    let mut cols: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i].0.as_str();
        if t.eq_ignore_ascii_case("loop_") {
            let mut names = Vec::new();
            i += 1;
            while i < toks.len() && toks[i].0.starts_with('_') {
                names.push(toks[i].0.clone());
                i += 1;
            }
            let mut vals = Vec::new();
            while i < toks.len() {
                let v = &toks[i].0;
                if v.starts_with('_') || v.eq_ignore_ascii_case("loop_") || v.starts_with("data_") {
                    break;
                }
                vals.push(v.clone());
                i += 1;
            }
            if names.first().is_some_and(|n| n.starts_with("_atom_site.")) {
                if vals.len() % names.len() != 0 {
                    return parse_err(&p, toks[i.saturating_sub(1)].1, "_atom_site loop has a ragged last row");
                }
                cols = names.iter().map(|n| n["_atom_site.".len()..].to_string()).collect();
                rows = vals.chunks(names.len()).map(|c| c.to_vec()).collect();
            }
            continue;
        }
        if let Some(key) = t.strip_prefix("_cell.") {
            if let Some((v, _)) = toks.get(i + 1) {
                if let Ok(x) = v.split('(').next().unwrap_or("").parse::<f64>() {
                    cell.insert(key.to_string(), x);
                }
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    if rows.is_empty() {
        return parse_err(&p, 0, "no _atom_site loop");
    }
    let idx = |name: &str| cols.iter().position(|c| c == name);
    let pick = |a: &str, b: &str| idx(a).or_else(|| idx(b));
    let (ix, iy, iz) = match (idx("Cartn_x"), idx("Cartn_y"), idx("Cartn_z")) {
        (Some(x), Some(y), Some(z)) => (x, y, z),
        _ => return parse_err(&p, 0, "_atom_site has no Cartn_x/y/z"),
    };
    let i_name = pick("auth_atom_id", "label_atom_id");
    let i_res = pick("auth_comp_id", "label_comp_id");
    let i_seq = pick("auth_seq_id", "label_seq_id");
    let i_chain = pick("auth_asym_id", "label_asym_id");
    let i_el = idx("type_symbol");
    let i_alt = idx("label_alt_id");
    let i_ins = idx("pdbx_PDB_ins_code");
    let i_grp = idx("group_PDB");
    let i_model = idx("pdbx_PDB_model_num");
    let get = |r: &Vec<String>, i: Option<usize>| -> String {
        i.map(|k| r[k].clone())
            .filter(|v| v != "?" && v != ".")
            .unwrap_or_default()
    };

    let mut s = Structure::default();
    let mut frames: Vec<Frame> = Vec::new();
    let mut model_of_frame: Vec<String> = Vec::new();
    let mut seen_alt: HashMap<(String, String, String, String), String> = HashMap::new();
    let mut skipped = 0usize;
    for r in &rows {
        let model = get(r, i_model);
        if model_of_frame.last() != Some(&model) {
            model_of_frame.push(model.clone());
            frames.push(Frame::default());
        }
        let first = frames.len() == 1;
        let name = get(r, i_name);
        let resname = get(r, i_res);
        let chain = get(r, i_chain);
        let seq = get(r, i_seq);
        let alt = get(r, i_alt);
        if !alt.is_empty() {
            let k = (chain.clone(), seq.clone(), resname.clone(), name.clone());
            let f = seen_alt.entry(k).or_insert_with(|| alt.clone()).clone();
            if f != alt {
                skipped += 1;
                continue;
            }
        }
        let num = |k: usize| r[k].split('(').next().unwrap_or("").parse::<f64>();
        let (Ok(x), Ok(y), Ok(z)) = (num(ix), num(iy), num(iz)) else {
            return parse_err(&p, 0, format!("bad coordinates for atom '{name}'"));
        };
        frames.last_mut().expect("pushed above").coords.push([x, y, z]);
        if first {
            let hetero = get(r, i_grp) == "HETATM";
            let element = elements::infer(&get(r, i_el), &name, None, &resname, hetero, None);
            s.atoms.push(AtomInfo {
                name,
                resname,
                resid: seq.parse().unwrap_or(0),
                icode: get(r, i_ins),
                chain,
                element,
                hetero,
                ..AtomInfo::default()
            });
        }
    }
    let n = s.atoms.len();
    if let Some(k) = frames.iter().position(|f| f.coords.len() != n) {
        return parse_err(&p, 0, format!("model {} has a different number of atoms", k + 1));
    }
    let c = [
        "length_a",
        "length_b",
        "length_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
    ]
    .map(|k| cell.get(k).copied());
    if let [Some(a), Some(b), Some(cc), Some(al), Some(be), Some(ga)] = c {
        if a > 1.0 {
            for f in &mut frames {
                f.cell = Some([a, b, cc, al, be, ga]);
            }
        }
    }
    if skipped > 0 {
        s.notes.push(format!(
            "skipped {skipped} alternate-location atoms (kept the first altloc)"
        ));
    }
    s.frames = frames;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn atom_site_loop() {
        let t = "data_TEST
_cell.length_a 40.0
_cell.length_b 50.0
_cell.length_c 60.0
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1.0 2.0 3.0 10 B 1
ATOM 2 C CA A ALA A 1 1.5 2.0 3.0 10 B 1
ATOM 3 C CA B ALA A 1 1.6 2.0 3.0 10 B 1
HETATM 4 ZN ZN . ZN C . 9.0 9.0 9.0 101 B 1
ATOM 1 N N . ALA A 1 1.1 2.0 3.0 10 B 2
ATOM 2 C CA A ALA A 1 1.6 2.0 3.0 10 B 2
ATOM 3 C CA B ALA A 1 1.7 2.0 3.0 10 B 2
HETATM 4 ZN ZN . ZN C . 9.1 9.0 9.0 101 B 2
#
";
        let d = std::env::temp_dir().join(format!("pbsolve-io-cif-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("a.cif");
        std::fs::File::create(&path).unwrap().write_all(t.as_bytes()).unwrap();
        let s = read_cif(&path).unwrap();
        assert_eq!(s.atoms.len(), 3);
        assert_eq!(s.frames.len(), 2);
        assert_eq!(s.atoms[0].resid, 10);
        assert_eq!(s.atoms[0].chain, "B");
        assert_eq!(s.atoms[2].element, "Zn");
        assert_eq!(s.frames[1].coords[2], [9.1, 9.0, 9.0]);
        assert_eq!(s.frames[0].cell.unwrap()[2], 60.0);
    }
}
