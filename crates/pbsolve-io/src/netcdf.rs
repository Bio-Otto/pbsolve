//! Minimal NetCDF-3 reader (classic and 64-bit-offset formats, written from
//! the public Unidata format specification) for Amber trajectories
//! (`Conventions = "AMBER"`) and restarts (`"AMBERRESTART"`).
//! NetCDF-4/HDF5 files are rejected with a conversion hint.

use crate::{format_err, path_str, AtomInfo, Frame, FrameReader, IoError, Result, Structure};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;

#[derive(Clone, Debug)]
enum AttrValue {
    Text(String),
    Numbers(Vec<f64>),
}

#[derive(Clone, Debug)]
struct Var {
    dims: Vec<usize>,
    nc_type: u32,
    begin: u64,
    vsize: u64,
    atts: HashMap<String, AttrValue>,
}

struct Header {
    numrecs: u64,
    dims: Vec<(String, u64)>,
    vars: HashMap<String, Var>,
    gatts: HashMap<String, AttrValue>,
    recsize: u64,
}

fn type_size(t: u32) -> Option<u64> {
    Some(match t {
        1 | 2 => 1,
        3 => 2,
        4 | 5 => 4,
        6 => 8,
        _ => return None,
    })
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    path: &'a str,
}

impl Cursor<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8]> {
        if self.pos + n > self.b.len() {
            return format_err(format!("{}: truncated NetCDF header", self.path));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.bytes(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let s = self.bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_be_bytes(a))
    }
    fn padded(&mut self, n: usize) -> Result<Vec<u8>> {
        let v = self.bytes(n)?.to_vec();
        let pad = (4 - n % 4) % 4;
        self.bytes(pad)?;
        Ok(v)
    }
    fn name(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(&self.padded(n)?).into_owned())
    }
    fn atts(&mut self) -> Result<HashMap<String, AttrValue>> {
        let tag = self.u32()?;
        let n = self.u32()?;
        let mut out = HashMap::new();
        if tag == 0 {
            return Ok(out);
        }
        if tag != NC_ATTRIBUTE {
            return format_err(format!("{}: bad attribute list tag {tag:#x}", self.path));
        }
        for _ in 0..n {
            let name = self.name()?;
            let t = self.u32()?;
            let count = self.u32()? as usize;
            let size = type_size(t).ok_or_else(|| IoError::Format(format!("{}: attribute type {t}", self.path)))?;
            let raw = self.padded(count * size as usize)?;
            let val = match t {
                2 => AttrValue::Text(String::from_utf8_lossy(&raw).trim_end_matches('\0').to_string()),
                _ => AttrValue::Numbers(decode(&raw, t, count)),
            };
            out.insert(name, val);
        }
        Ok(out)
    }
}

fn decode(raw: &[u8], t: u32, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| match t {
            1 => raw[i] as i8 as f64,
            3 => i16::from_be_bytes([raw[2 * i], raw[2 * i + 1]]) as f64,
            4 => i32::from_be_bytes(raw[4 * i..4 * i + 4].try_into().expect("4 bytes")) as f64,
            5 => f32::from_be_bytes(raw[4 * i..4 * i + 4].try_into().expect("4 bytes")) as f64,
            6 => f64::from_be_bytes(raw[8 * i..8 * i + 8].try_into().expect("8 bytes")),
            _ => f64::NAN,
        })
        .collect()
}

fn read_header(path: &Path, file: &mut std::fs::File) -> Result<Header> {
    let p = path_str(path);
    let len = file.metadata()?.len();
    // headers are small; read up to 1 MiB (large enough for any Amber file)
    let mut buf = vec![0u8; len.min(1 << 20) as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;
    if buf.starts_with(b"\x89HDF") {
        return Err(IoError::Unsupported(format!(
            "{p}: NetCDF-4/HDF5 is not supported; convert with cpptraj ('trajout out.nc netcdf')"
        )));
    }
    if !buf.starts_with(b"CDF") || buf.len() < 4 {
        return format_err(format!("{p}: not a NetCDF file"));
    }
    let version = buf[3];
    if version != 1 && version != 2 {
        return Err(IoError::Unsupported(format!(
            "{p}: NetCDF format variant {version} (CDF-5?) is not supported"
        )));
    }
    let mut c = Cursor {
        b: &buf,
        pos: 4,
        path: &p,
    };
    let numrecs_raw = c.u32()?;
    let tag = c.u32()?;
    let n = c.u32()?;
    let mut dims = Vec::new();
    if tag == NC_DIMENSION {
        for _ in 0..n {
            let name = c.name()?;
            let l = c.u32()? as u64;
            dims.push((name, l));
        }
    } else if tag != 0 {
        return format_err(format!("{p}: bad dimension list"));
    }
    let gatts = c.atts()?;
    let tag = c.u32()?;
    let n = c.u32()?;
    let mut vars = HashMap::new();
    let mut order = Vec::new();
    if tag == NC_VARIABLE {
        for _ in 0..n {
            let name = c.name()?;
            let nd = c.u32()? as usize;
            let mut vd = Vec::with_capacity(nd);
            for _ in 0..nd {
                vd.push(c.u32()? as usize);
            }
            let atts = c.atts()?;
            let nc_type = c.u32()?;
            let vsize = c.u32()? as u64;
            let begin = if version == 1 { c.u32()? as u64 } else { c.u64()? };
            if vd.iter().any(|&d| d >= dims.len()) {
                return format_err(format!("{p}: variable {name} uses an unknown dimension"));
            }
            order.push(name.clone());
            vars.insert(
                name,
                Var {
                    dims: vd,
                    nc_type,
                    begin,
                    vsize,
                    atts,
                },
            );
        }
    } else if tag != 0 {
        return format_err(format!("{p}: bad variable list"));
    }
    let record_dim = dims.iter().position(|(_, l)| *l == 0);
    let is_rec = |v: &Var| record_dim.is_some_and(|r| v.dims.first() == Some(&r));
    let rec_vars: Vec<&Var> = order.iter().map(|n| &vars[n]).filter(|v| is_rec(v)).collect();
    let recsize: u64 = if rec_vars.len() == 1 {
        // a single record variable is not padded
        let v = rec_vars[0];
        let per: u64 = v.dims[1..].iter().map(|&d| dims[d].1).product();
        per * type_size(v.nc_type).unwrap_or(4)
    } else {
        rec_vars.iter().map(|v| v.vsize).sum()
    };
    let numrecs = if numrecs_raw == u32::MAX {
        // streaming: derive from file size
        let first = rec_vars.iter().map(|v| v.begin).min().unwrap_or(len);
        (len - first).checked_div(recsize).unwrap_or(0)
    } else {
        numrecs_raw as u64
    };
    Ok(Header {
        numrecs,
        dims,
        vars,
        gatts,
        recsize,
    })
}

fn read_values(file: &mut std::fs::File, offset: u64, t: u32, count: usize) -> Result<Vec<f64>> {
    let size = type_size(t).ok_or_else(|| IoError::Format(format!("unsupported NetCDF type {t}")))? as usize;
    let mut raw = vec![0u8; count * size];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut raw)?;
    Ok(decode(&raw, t, count))
}

fn scale(v: &Var) -> f64 {
    match v.atts.get("scale_factor") {
        Some(AttrValue::Numbers(x)) if !x.is_empty() => x[0],
        _ => 1.0,
    }
}

fn conventions(h: &Header) -> String {
    match h.gatts.get("Conventions") {
        Some(AttrValue::Text(t)) => t.clone(),
        _ => String::new(),
    }
}

/// Amber NetCDF trajectory reader.
pub struct AmberNetCdf {
    path: String,
    file: std::fs::File,
    h: Header,
    n_atoms: usize,
    next: u64,
}

impl AmberNetCdf {
    pub fn open(path: &Path) -> Result<Self> {
        let p = path_str(path);
        let mut file = std::fs::File::open(path)?;
        let h = read_header(path, &mut file)?;
        let conv = conventions(&h);
        if !conv.split([',', ' ']).any(|c| c == "AMBER") {
            if conv.contains("AMBERRESTART") {
                return format_err(format!("{p}: this is an Amber NetCDF restart; read it as a structure"));
            }
            return format_err(format!("{p}: not an Amber NetCDF trajectory (Conventions = '{conv}')"));
        }
        let Some(coords) = h.vars.get("coordinates") else {
            return format_err(format!("{p}: no 'coordinates' variable"));
        };
        if coords.dims.len() != 3 {
            return format_err(format!("{p}: 'coordinates' must be (frame, atom, spatial)"));
        }
        let n_atoms = h.dims[coords.dims[1]].1 as usize;
        Ok(Self {
            path: p,
            file,
            h,
            n_atoms,
            next: 0,
        })
    }
}

impl FrameReader for AmberNetCdf {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n_atoms)
    }

    fn n_frames(&self) -> Option<usize> {
        Some(self.h.numrecs as usize)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.next >= self.h.numrecs {
            return Ok(None);
        }
        let k = self.next;
        self.next += 1;
        let rec = |v: &Var| v.begin + k * self.h.recsize;
        let cv = self.h.vars["coordinates"].clone();
        let raw = read_values(&mut self.file, rec(&cv), cv.nc_type, 3 * self.n_atoms)
            .map_err(|e| IoError::Format(format!("{}: frame {k}: {e}", self.path)))?;
        let s = scale(&cv);
        let coords = raw.chunks(3).map(|c| [s * c[0], s * c[1], s * c[2]]).collect();
        let mut cell = None;
        if let (Some(l), Some(a)) = (
            self.h.vars.get("cell_lengths").cloned(),
            self.h.vars.get("cell_angles").cloned(),
        ) {
            let lv = read_values(&mut self.file, rec(&l), l.nc_type, 3)?;
            let av = read_values(&mut self.file, rec(&a), a.nc_type, 3)?;
            cell = Some([lv[0], lv[1], lv[2], av[0], av[1], av[2]]);
        }
        let time = match self.h.vars.get("time").cloned() {
            Some(t) => read_values(&mut self.file, rec(&t), t.nc_type, 1)?.first().copied(),
            None => None,
        };
        Ok(Some(Frame {
            coords,
            cell,
            time,
            step: None,
        }))
    }
}

/// True if the file is an Amber NetCDF restart (`Conventions = "AMBERRESTART"`).
pub fn is_restart(path: &Path) -> Result<bool> {
    let mut file = std::fs::File::open(path)?;
    Ok(conventions(&read_header(path, &mut file)?).contains("AMBERRESTART"))
}

/// Amber NetCDF restart (`Conventions = "AMBERRESTART"`) as a coordinate-only structure.
pub fn read_restart(path: &Path) -> Result<Structure> {
    let p = path_str(path);
    let mut file = std::fs::File::open(path)?;
    let h = read_header(path, &mut file)?;
    if !conventions(&h).contains("AMBERRESTART") {
        return format_err(format!("{p}: not an Amber NetCDF restart"));
    }
    let Some(cv) = h.vars.get("coordinates") else {
        return format_err(format!("{p}: no 'coordinates' variable"));
    };
    let n = h.dims[cv.dims[0]].1 as usize;
    let raw = read_values(&mut file, cv.begin, cv.nc_type, 3 * n)?;
    let s = scale(cv);
    let coords = raw.chunks(3).map(|c| [s * c[0], s * c[1], s * c[2]]).collect();
    let mut cell = None;
    if let (Some(l), Some(a)) = (h.vars.get("cell_lengths"), h.vars.get("cell_angles")) {
        let lv = read_values(&mut file, l.begin, l.nc_type, 3)?;
        let av = read_values(&mut file, a.begin, a.nc_type, 3)?;
        cell = Some([lv[0], lv[1], lv[2], av[0], av[1], av[2]]);
    }
    Ok(Structure {
        atoms: vec![AtomInfo::default(); n],
        frames: vec![Frame {
            coords,
            cell,
            ..Frame::default()
        }],
        notes: vec!["restart carries coordinates only; load a topology for atom data".into()],
        ..Structure::default()
    })
}
