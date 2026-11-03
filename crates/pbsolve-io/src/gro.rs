//! GROMACS GRO (single or multi-frame). Positions in nm are converted to Å.
//! The coordinate field width follows the file's precision (`%8.3f` by
//! default, wider with `gmx editconf -ndec`), detected from the decimal points.

use crate::elements;
use crate::{col, parse_err, path_str, AtomInfo, Frame, Result, Structure};
use std::path::Path;

/// Width of one coordinate field: distance between the first two decimal points.
fn field_width(line: &str) -> Option<usize> {
    let tail = line.get(20..)?;
    let mut dots = tail.char_indices().filter(|&(_, c)| c == '.').map(|(i, _)| i);
    let a = dots.next()?;
    let b = dots.next()?;
    Some(b - a)
}

fn box_from_gro(line: &str) -> Option<[f64; 6]> {
    let v: Vec<f64> = line
        .split_whitespace()
        .map(|t| t.parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    match v.len() {
        3 => Some([10.0 * v[0], 10.0 * v[1], 10.0 * v[2], 90.0, 90.0, 90.0]),
        9 => {
            // v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
            let a = [v[0], v[3], v[4]];
            let b = [v[5], v[1], v[6]];
            let c = [v[7], v[8], v[2]];
            let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let ang = |u: [f64; 3], w: [f64; 3]| {
                ((u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (norm(u) * norm(w)))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            };
            Some([
                10.0 * norm(a),
                10.0 * norm(b),
                10.0 * norm(c),
                ang(b, c),
                ang(a, c),
                ang(a, b),
            ])
        }
        _ => None,
    }
}

fn parse_time(title: &str) -> Option<f64> {
    let i = title.find("t=")?;
    title[i + 2..].split_whitespace().next()?.parse::<f64>().ok()
}

pub fn read_gro(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut s = Structure::default();
    let mut pos = 0usize;
    while pos < lines.len() {
        if lines[pos].trim().is_empty() {
            pos += 1;
            continue;
        }
        let title = lines[pos];
        let Some(nline) = lines.get(pos + 1) else {
            return parse_err(&p, pos + 2, "missing atom count");
        };
        let n: usize = match nline.trim().parse() {
            Ok(n) => n,
            Err(_) => return parse_err(&p, pos + 2, format!("bad atom count '{}'", nline.trim())),
        };
        if pos + 2 + n >= lines.len() {
            return parse_err(&p, lines.len(), "file ends inside a frame");
        }
        let first = s.frames.is_empty();
        if !first && n != s.atoms.len() {
            return parse_err(
                &p,
                pos + 2,
                format!("frame has {n} atoms, first frame {}", s.atoms.len()),
            );
        }
        let w = field_width(lines[pos + 2]).unwrap_or(8);
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            let ln = pos + 2 + i;
            let l = lines[ln];
            let mut xyz = [0.0; 3];
            for (d, v) in xyz.iter_mut().enumerate() {
                let f = col(l, 20 + d * w, 20 + (d + 1) * w);
                *v = match f.parse::<f64>() {
                    Ok(x) => 10.0 * x,
                    Err(_) => return parse_err(&p, ln + 1, format!("cannot parse coordinate '{f}'")),
                };
            }
            coords.push(xyz);
            if first {
                let resname = col(l, 5, 10).to_string();
                let name = col(l, 10, 15).to_string();
                let element = elements::infer("", &name, None, &resname, false, None);
                s.atoms.push(AtomInfo {
                    resid: col(l, 0, 5).parse().unwrap_or(0),
                    resname,
                    name,
                    element,
                    ..AtomInfo::default()
                });
            }
        }
        let cell = box_from_gro(lines[pos + 2 + n]);
        s.frames.push(Frame {
            coords,
            cell,
            time: parse_time(title),
            step: None,
        });
        pos += 3 + n;
    }
    if s.frames.is_empty() {
        return parse_err(&p, 0, "empty GRO file");
    }
    s.notes
        .push("GRO has no elements; they were guessed from atom names".into());
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn two_frames_and_precision() {
        let t = "\
Test t=   1.00000 step= 500
    3
    1ALA      N    1   0.100   0.200   0.300
    1ALA     CA    2  -1.000  10.000 100.000
    2SOL     OW    3   0.000   0.000   0.000
   5.00000   5.00000   5.00000
Test t=   2.00000
    3
    1ALA      N    1   0.10000   0.20000   0.30500
    1ALA     CA    2  -1.00000  10.00000 100.00000
    2SOL     OW    3   0.00000   0.00000   0.00000
   5.00000   5.00000   5.00000
";
        let d = std::env::temp_dir().join(format!("pbsolve-io-gro-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("a.gro");
        std::fs::File::create(&path).unwrap().write_all(t.as_bytes()).unwrap();
        let s = read_gro(&path).unwrap();
        assert_eq!(s.atoms.len(), 3);
        assert_eq!(s.frames.len(), 2);
        assert_eq!(s.frames[0].coords[1], [-10.0, 100.0, 1000.0]);
        assert!((s.frames[1].coords[0][2] - 3.05).abs() < 1e-12);
        assert_eq!(s.frames[0].time, Some(1.0));
        assert_eq!(s.frames[0].cell.unwrap()[0], 50.0);
        assert_eq!(s.atoms[2].element, "O");
    }
}
