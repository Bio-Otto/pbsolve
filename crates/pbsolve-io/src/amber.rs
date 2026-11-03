//! Amber ASCII coordinates: inpcrd/rst7 (`6F12.7`, optional velocities and
//! box) and trajectories in mdcrd format (`10F8.3`, optional box line per frame).
//! Both are fixed-width: negative numbers may touch.

use crate::{format_err, parse_err, path_str, AtomInfo, Frame, FrameReader, Result, Structure};
use std::io::BufRead;
use std::path::Path;

fn fixed_values(line: &str, width: usize, out: &mut Vec<f64>) -> std::result::Result<(), String> {
    let l = line.trim_end();
    let mut i = 0;
    while i < l.len() {
        let end = (i + width).min(l.len());
        let f = l.get(i..end).ok_or("non-ASCII data")?.trim();
        if !f.is_empty() {
            out.push(f.parse::<f64>().map_err(|_| format!("cannot parse '{f}'"))?);
        }
        i = end;
    }
    Ok(())
}

/// inpcrd / rst7 (ASCII). NetCDF restarts are handled by [`crate::netcdf`].
pub fn read_inpcrd(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(b"CDF") {
        return crate::netcdf::read_restart(path);
    }
    let text = String::from_utf8(bytes).or_else(|_| parse_err(&p, 0, "not a text file"))?;
    let mut lines = text.lines();
    lines.next(); // title
    let head = lines.next().unwrap_or("");
    let mut it = head.split_whitespace();
    let n: usize = match it.next().and_then(|t| t.parse().ok()) {
        Some(n) => n,
        None => return parse_err(&p, 2, "expected the number of atoms"),
    };
    let time = it.next().and_then(|t| t.parse::<f64>().ok());
    let mut vals = Vec::with_capacity(3 * n + 6);
    for (k, l) in lines.enumerate() {
        fixed_values(l, 12, &mut vals).or_else(|e| parse_err(&p, k + 3, e))?;
    }
    if vals.len() < 3 * n {
        return parse_err(&p, 0, format!("expected {} coordinates, found {}", 3 * n, vals.len()));
    }
    let coords: Vec<[f64; 3]> = vals[..3 * n].chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let rest = vals.len() - 3 * n;
    // remaining: [velocities 3n] [box 6]
    let cell = match rest {
        6 => Some(&vals[3 * n..]),
        r if r == 3 * n + 6 => Some(&vals[6 * n..]),
        _ => None,
    }
    .map(|b| [b[0], b[1], b[2], b[3], b[4], b[5]]);
    Ok(Structure {
        atoms: vec![AtomInfo::default(); n],
        frames: vec![Frame {
            coords,
            cell,
            time,
            step: None,
        }],
        notes: vec!["inpcrd carries coordinates only; load a topology for atom data".into()],
        ..Structure::default()
    })
}

/// Streaming ASCII mdcrd reader.
pub struct MdcrdReader {
    path: String,
    reader: std::io::BufReader<std::fs::File>,
    n: usize,
    has_box: Option<bool>,
    line_no: usize,
    pending: Option<String>,
}

impl MdcrdReader {
    pub fn open(path: &Path, n_atoms: usize, has_box: Option<bool>) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(f);
        let mut title = String::new();
        reader.read_line(&mut title)?;
        if n_atoms == 0 {
            return format_err("mdcrd: number of atoms must be positive");
        }
        Ok(Self {
            path: path_str(path),
            reader,
            n: n_atoms,
            has_box,
            line_no: 1,
            pending: None,
        })
    }

    fn next_line(&mut self) -> Result<Option<String>> {
        if let Some(l) = self.pending.take() {
            return Ok(Some(l));
        }
        let mut l = String::new();
        if self.reader.read_line(&mut l)? == 0 {
            return Ok(None);
        }
        self.line_no += 1;
        Ok(Some(l))
    }
}

impl FrameReader for MdcrdReader {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let need = 3 * self.n;
        let mut vals = Vec::with_capacity(need);
        while vals.len() < need {
            let Some(l) = self.next_line()? else {
                if vals.is_empty() {
                    return Ok(None);
                }
                return parse_err(&self.path, self.line_no, "file ends inside a frame");
            };
            if l.trim().is_empty() {
                continue;
            }
            fixed_values(&l, 8, &mut vals).or_else(|e| parse_err(&self.path, self.line_no, e))?;
        }
        if vals.len() != need {
            return parse_err(
                &self.path,
                self.line_no,
                "frame does not end at a line break (wrong atom count?)",
            );
        }
        let coords = vals.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        // Box line: exactly 3 values. Detect once if not specified: a following
        // 3-value line is a box, except for 1-atom systems where the next
        // frame's first line also holds 3 values (then: no box unless told).
        let mut cell = None;
        let box_possible = self.has_box != Some(false);
        if box_possible {
            if let Some(l) = self.next_line()? {
                let mut b = Vec::new();
                let ok = fixed_values(&l, 8, &mut b).is_ok();
                let is_box = ok && b.len() == 3 && (self.has_box == Some(true) || need != 3);
                if is_box {
                    cell = Some([b[0], b[1], b[2], 90.0, 90.0, 90.0]);
                    self.has_box.get_or_insert(true);
                } else {
                    if self.has_box == Some(true) {
                        return parse_err(&self.path, self.line_no, "expected a box line");
                    }
                    self.has_box.get_or_insert(false);
                    self.pending = Some(l);
                }
            }
        }
        Ok(Some(Frame {
            coords,
            cell,
            ..Frame::default()
        }))
    }
}
