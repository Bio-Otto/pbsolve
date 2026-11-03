//! GROMACS topologies (.top/.itp) with the C-preprocessor subset GROMACS
//! uses (`#include`, `#define`, `#undef`, `#ifdef`, `#ifndef`, `#else`,
//! `#endif`). Atoms come from `[ atoms ]` of each `[ moleculetype ]`,
//! expanded by `[ molecules ]`; elements from `[ atomtypes ]` atomic numbers,
//! else from names/masses. Bonds from `[ bonds ]` (and `[ settles ]`).
//!
//! Include search order: the including file's directory, `include_dirs`,
//! `$GMXLIB`, `$GMXDATA/top`, `$CONDA_PREFIX/share/gromacs/top`,
//! `/usr/share/gromacs/top`, `/usr/local/gromacs/share/gromacs/top`.
//! A fully preprocessed topology (`gmx grompp -pp`) needs no search path.

use crate::elements;
use crate::{format_err, parse_err, AtomInfo, Result, Structure};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
struct MolAtom {
    atype: String,
    resnr: i64,
    resname: String,
    name: String,
    charge: f64,
    mass: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct MolType {
    atoms: Vec<MolAtom>,
    bonds: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct AtomType {
    z: Option<i64>,
    mass: f64,
}

fn search_dirs(extra: &[PathBuf]) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = extra.to_vec();
    if let Ok(v) = std::env::var("GMXLIB") {
        d.extend(std::env::split_paths(&v));
    }
    if let Ok(v) = std::env::var("GMXDATA") {
        d.push(PathBuf::from(v).join("top"));
    }
    if let Ok(v) = std::env::var("CONDA_PREFIX") {
        d.push(PathBuf::from(v).join("share/gromacs/top"));
    }
    d.push(PathBuf::from("/usr/share/gromacs/top"));
    d.push(PathBuf::from("/usr/local/gromacs/share/gromacs/top"));
    d
}

struct Pre<'a> {
    defines: HashMap<String, String>,
    dirs: &'a [PathBuf],
    lines: Vec<(String, String, usize)>, // (text, file, line)
    depth: usize,
}

impl Pre<'_> {
    fn find(&self, name: &str, here: &Path) -> Option<PathBuf> {
        let cands = std::iter::once(here.to_path_buf()).chain(self.dirs.iter().cloned());
        cands.map(|d| d.join(name)).find(|p| p.is_file())
    }

    fn run(&mut self, path: &Path) -> Result<()> {
        if self.depth > 32 {
            return format_err(format!("{}: #include nested too deeply", path.display()));
        }
        let text = std::fs::read_to_string(path)
            .map_err(|e| crate::IoError::Format(format!("cannot read {}: {e}", path.display())))?;
        let here = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let fname = path.display().to_string();
        // stack of (active, any_branch_taken)
        let mut stack: Vec<(bool, bool)> = Vec::new();
        let active = |st: &Vec<(bool, bool)>| st.iter().all(|(a, _)| *a);
        let mut cont = String::new();
        for (ln, raw) in text.lines().enumerate() {
            let mut line = raw.split(';').next().unwrap_or("").to_string();
            if let Some(stripped) = line.strip_suffix('\\') {
                cont.push_str(stripped);
                continue;
            }
            if !cont.is_empty() {
                line = std::mem::take(&mut cont) + &line;
            }
            let t = line.trim();
            if let Some(d) = t.strip_prefix('#') {
                let mut it = d.split_whitespace();
                let kw = it.next().unwrap_or("");
                let arg = it.next().unwrap_or("");
                match kw {
                    "ifdef" | "ifndef" => {
                        let def = self.defines.contains_key(arg);
                        let on = if kw == "ifdef" { def } else { !def };
                        stack.push((on, on));
                    }
                    "else" => match stack.last_mut() {
                        Some(top) => {
                            top.0 = !top.1;
                            top.1 = true;
                        }
                        None => return parse_err(&fname, ln + 1, "#else without #ifdef"),
                    },
                    "endif" => {
                        if stack.pop().is_none() {
                            return parse_err(&fname, ln + 1, "#endif without #ifdef");
                        }
                    }
                    _ if !active(&stack) => {}
                    "define" => {
                        let rest: Vec<&str> = it.collect();
                        self.defines.insert(arg.to_string(), rest.join(" "));
                    }
                    "undef" => {
                        self.defines.remove(arg);
                    }
                    "include" => {
                        let name = d.trim_start()["include".len()..].trim().trim_matches(['"', '<', '>']);
                        match self.find(name, &here) {
                            Some(p) => {
                                self.depth += 1;
                                self.run(&p)?;
                                self.depth -= 1;
                            }
                            None => {
                                return parse_err(
                                    &fname,
                                    ln + 1,
                                    format!(
                                        "#include \"{name}\" not found (searched the topology directory, \
                                         GMXLIB, GMXDATA/top, CONDA_PREFIX/share/gromacs/top); set GMXLIB or \
                                         pass a preprocessed topology from 'gmx grompp -pp processed.top'"
                                    ),
                                )
                            }
                        }
                    }
                    other => return parse_err(&fname, ln + 1, format!("unsupported directive #{other}")),
                }
                continue;
            }
            if active(&stack) && !t.is_empty() {
                self.lines.push((t.to_string(), fname.clone(), ln + 1));
            }
        }
        if !stack.is_empty() {
            return parse_err(&fname, 0, "unterminated #ifdef");
        }
        Ok(())
    }
}

fn num(f: &str, l: usize, s: &str) -> Result<f64> {
    s.parse::<f64>()
        .or_else(|_| parse_err(f, l, format!("cannot parse number '{s}'")))
}

pub fn read_top(path: &Path, include_dirs: &[PathBuf]) -> Result<Structure> {
    let dirs = search_dirs(include_dirs);
    let mut pre = Pre {
        defines: HashMap::new(),
        dirs: &dirs,
        lines: Vec::new(),
        depth: 0,
    };
    pre.run(path)?;

    let mut atomtypes: HashMap<String, AtomType> = HashMap::new();
    let mut moltypes: HashMap<String, MolType> = HashMap::new();
    let mut molecules: Vec<(String, usize)> = Vec::new();
    let mut section = String::new();
    let mut cur_mol: Option<String> = None;
    for (line, f, l) in &pre.lines {
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']', ' ']).trim().to_ascii_lowercase();
            continue;
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        match section.as_str() {
            "atomtypes" => {
                // name [btype] [at.num] mass charge ptype V W — ptype locates the columns. Search
                // from the right: type names may themselves be "S" (sulfur) or "A".
                let Some(p) = tok
                    .iter()
                    .enumerate()
                    .skip(1)
                    .rev()
                    .find(|(_, t)| matches!(**t, "A" | "S" | "V" | "D"))
                    .map(|(i, _)| i)
                else {
                    return parse_err(f, *l, "cannot find the particle type (A/S/V/D) in [ atomtypes ]");
                };
                if p < 3 {
                    return parse_err(f, *l, "too few columns in [ atomtypes ]");
                }
                let mass = num(f, *l, tok[p - 2])?;
                let z = if p >= 4 { tok[p - 3].parse::<i64>().ok() } else { None };
                atomtypes.insert(tok[0].to_string(), AtomType { z, mass });
            }
            "moleculetype" => {
                let name = tok[0].to_string();
                moltypes.insert(name.clone(), MolType::default());
                cur_mol = Some(name);
            }
            "atoms" => {
                let Some(m) = cur_mol.as_ref().and_then(|n| moltypes.get_mut(n)) else {
                    return parse_err(f, *l, "[ atoms ] outside a [ moleculetype ]");
                };
                if tok.len() < 7 {
                    return parse_err(f, *l, "[ atoms ] needs nr type resnr residue atom cgnr charge");
                }
                let nr: usize = tok[0].parse().or_else(|_| parse_err(f, *l, "bad atom number"))?;
                if nr != m.atoms.len() + 1 {
                    return parse_err(f, *l, "[ atoms ] numbers must be consecutive from 1");
                }
                m.atoms.push(MolAtom {
                    atype: tok[1].to_string(),
                    resnr: tok[2]
                        .trim_end_matches(|c: char| c.is_ascii_alphabetic())
                        .parse()
                        .unwrap_or(0),
                    resname: tok[3].to_string(),
                    name: tok[4].to_string(),
                    charge: num(f, *l, tok[6])?,
                    mass: tok.get(7).and_then(|t| t.parse::<f64>().ok()),
                });
            }
            "bonds" | "settles" => {
                let Some(m) = cur_mol.as_ref().and_then(|n| moltypes.get_mut(n)) else {
                    continue;
                };
                if section == "settles" {
                    // settles: OW index; H atoms are the next two
                    if let Ok(o) = tok[0].parse::<usize>() {
                        m.bonds.push((o - 1, o));
                        m.bonds.push((o - 1, o + 1));
                    }
                } else if tok.len() >= 2 {
                    if let (Ok(a), Ok(b)) = (tok[0].parse::<usize>(), tok[1].parse::<usize>()) {
                        m.bonds.push((a.min(b) - 1, a.max(b) - 1));
                    }
                }
            }
            "molecules" => {
                if tok.len() < 2 {
                    return parse_err(f, *l, "[ molecules ] needs name count");
                }
                let n: usize = tok[1].parse().or_else(|_| parse_err(f, *l, "bad molecule count"))?;
                molecules.push((tok[0].to_string(), n));
            }
            _ => {}
        }
    }
    if molecules.is_empty() {
        return format_err(format!("{}: no [ molecules ] section", path.display()));
    }
    let mut s = Structure::default();
    let mut unknown_types: HashSet<String> = HashSet::new();
    for (name, count) in &molecules {
        let Some(m) = moltypes.get(name) else {
            return format_err(format!("{}: molecule type '{name}' is not defined", path.display()));
        };
        for _ in 0..*count {
            let base = s.atoms.len();
            for a in &m.atoms {
                let at = atomtypes.get(&a.atype);
                if at.is_none() {
                    unknown_types.insert(a.atype.clone());
                }
                let mass = a.mass.or(at.map(|t| t.mass));
                let element = match at.and_then(|t| t.z) {
                    Some(z) if z > 0 => elements::from_atomic_number(z).unwrap_or("").to_string(),
                    Some(_) => String::new(),
                    None => elements::infer("", &a.name, None, &a.resname, false, mass),
                };
                s.atoms.push(AtomInfo {
                    name: a.name.clone(),
                    resname: a.resname.clone(),
                    resid: a.resnr,
                    segid: name.clone(),
                    element,
                    charge: Some(a.charge),
                    mass,
                    atom_type: a.atype.clone(),
                    ..AtomInfo::default()
                });
            }
            for &(i, j) in &m.bonds {
                if j < m.atoms.len() {
                    s.bonds.push((base + i, base + j));
                }
            }
        }
    }
    if !unknown_types.is_empty() && atomtypes.is_empty() {
        s.notes
            .push("no [ atomtypes ] found; elements were guessed from names and masses".into());
    }
    Ok(s)
}
