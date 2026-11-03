//! DCD trajectories (CHARMM, NAMD, OpenMM, X-PLOR). Fortran unformatted
//! records with 4-byte markers, either endianness. Supports the CHARMM unit
//! cell block, fixed atoms (only free atoms stored after the first frame) and
//! files still being written (frame count from the file size).

use crate::{format_err, path_str, Frame, FrameReader, IoError, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct DcdReader {
    path: String,
    file: std::io::BufReader<std::fs::File>,
    big_endian: bool,
    n_atoms: usize,
    free: Option<Vec<usize>>,
    charmm: bool,
    has_cell: bool,
    has_4d: bool,
    first_frame_pos: u64,
    frame_bytes_first: u64,
    frame_bytes: u64,
    n_frames: usize,
    next: usize,
    fixed_coords: Vec<[f64; 3]>,
    istart: i64,
    nsavc: i64,
    delta: f64,
}

fn i32_of(b: &[u8], big: bool) -> i32 {
    let a = [b[0], b[1], b[2], b[3]];
    if big {
        i32::from_be_bytes(a)
    } else {
        i32::from_le_bytes(a)
    }
}

impl DcdReader {
    pub fn open(path: &Path) -> Result<Self> {
        let p = path_str(path);
        let f = std::fs::File::open(path)?;
        let size = f.metadata()?.len();
        let mut file = std::io::BufReader::new(f);
        let mut m = [0u8; 4];
        file.read_exact(&mut m)?;
        let big_endian = match (i32::from_le_bytes(m), i32::from_be_bytes(m)) {
            (84, _) => false,
            (_, 84) => true,
            _ => return format_err(format!("{p}: not a DCD file (first record marker is not 84)")),
        };
        let mut hdr = [0u8; 84];
        file.read_exact(&mut hdr)?;
        if &hdr[..4] != b"CORD" {
            return format_err(format!("{p}: DCD header does not start with CORD"));
        }
        let icntrl: Vec<i32> = (0..20)
            .map(|k| i32_of(&hdr[4 + 4 * k..8 + 4 * k], big_endian))
            .collect();
        let mut r = Self {
            path: p.clone(),
            file,
            big_endian,
            n_atoms: 0,
            free: None,
            charmm: icntrl[19] != 0,
            has_cell: false,
            has_4d: false,
            first_frame_pos: 0,
            frame_bytes_first: 0,
            frame_bytes: 0,
            n_frames: 0,
            next: 0,
            fixed_coords: Vec::new(),
            istart: icntrl[1] as i64,
            nsavc: icntrl[2] as i64,
            delta: 0.0,
        };
        r.has_cell = r.charmm && icntrl[10] != 0;
        r.has_4d = r.charmm && icntrl[11] != 0;
        let namnf = icntrl[8].max(0) as usize;
        r.delta = if r.charmm {
            let b = &hdr[4 + 36..4 + 40];
            let a = [b[0], b[1], b[2], b[3]];
            (if big_endian {
                f32::from_be_bytes(a)
            } else {
                f32::from_le_bytes(a)
            }) as f64
        } else {
            let b = &hdr[4 + 36..4 + 44];
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            if big_endian {
                f64::from_be_bytes(a)
            } else {
                f64::from_le_bytes(a)
            }
        };
        r.marker(84)?;
        // title record
        let tlen = r.read_marker()?;
        let mut title = vec![0u8; tlen as usize];
        r.file.read_exact(&mut title)?;
        r.marker(tlen)?;
        // natoms record
        r.marker_expect(4)?;
        let n = r.read_i32()?;
        r.marker(4)?;
        if n <= 0 {
            return format_err(format!("{p}: DCD has {n} atoms"));
        }
        r.n_atoms = n as usize;
        if namnf > 0 {
            let nfree = r.n_atoms - namnf;
            r.marker_expect(4 * nfree as u32)?;
            let mut idx = Vec::with_capacity(nfree);
            for _ in 0..nfree {
                idx.push(r.read_i32()? as usize - 1);
            }
            r.marker(4 * nfree as u32)?;
            r.free = Some(idx);
        }
        r.first_frame_pos = r.file.stream_position()?;
        let cell_bytes = if r.has_cell { 48 + 8 } else { 0 };
        let rec = |k: usize| 3 * (4 * k as u64 + 8) + if r.has_4d { 4 * k as u64 + 8 } else { 0 };
        r.frame_bytes_first = cell_bytes + rec(r.n_atoms);
        r.frame_bytes = cell_bytes + rec(r.free.as_ref().map_or(r.n_atoms, |f| f.len()));
        let avail = size.saturating_sub(r.first_frame_pos);
        r.n_frames = if avail < r.frame_bytes_first {
            0
        } else {
            1 + ((avail - r.frame_bytes_first) / r.frame_bytes) as usize
        };
        // icntrl[0] (frame count) is ignored: NAMD updates it only at the end of a run.
        Ok(r)
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.file.read_exact(&mut b)?;
        Ok(i32_of(&b, self.big_endian))
    }

    fn read_marker(&mut self) -> Result<u32> {
        Ok(self.read_i32()? as u32)
    }

    fn marker(&mut self, want: u32) -> Result<()> {
        self.marker_expect(want)
    }

    fn marker_expect(&mut self, want: u32) -> Result<()> {
        let got = self.read_marker()?;
        if got != want {
            return format_err(format!(
                "{}: corrupt DCD record marker (expected {want}, found {got})",
                self.path
            ));
        }
        Ok(())
    }

    fn read_floats(&mut self, n: usize) -> Result<Vec<f32>> {
        self.marker_expect(4 * n as u32)?;
        let mut raw = vec![0u8; 4 * n];
        self.file.read_exact(&mut raw)?;
        self.marker_expect(4 * n as u32)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3]];
                if self.big_endian {
                    f32::from_be_bytes(a)
                } else {
                    f32::from_le_bytes(a)
                }
            })
            .collect())
    }

    /// Time of frame `k` in AKMA units × steps, converted to ps (CHARMM timestep unit 48.88821 fs).
    fn time_of(&self, k: usize) -> Option<f64> {
        (self.delta > 0.0 && self.nsavc > 0)
            .then(|| (self.istart as f64 + (k as i64 * self.nsavc) as f64) * self.delta * 0.048_888_21)
    }
}

impl FrameReader for DcdReader {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n_atoms)
    }

    fn n_frames(&self) -> Option<usize> {
        Some(self.n_frames)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.next >= self.n_frames {
            return Ok(None);
        }
        let k = self.next;
        let mut cell = None;
        if self.has_cell {
            self.marker_expect(48)?;
            let mut raw = [0u8; 48];
            self.file.read_exact(&mut raw)?;
            self.marker_expect(48)?;
            let v: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(c);
                    if self.big_endian {
                        f64::from_be_bytes(a)
                    } else {
                        f64::from_le_bytes(a)
                    }
                })
                .collect();
            // CHARMM order: A, gamma, B, beta, alpha, C; NAMD/OpenMM may store cosines.
            let ang = |x: f64| if x.abs() <= 1.0 { x.acos().to_degrees() } else { x };
            cell = Some([v[0], v[2], v[5], ang(v[4]), ang(v[3]), ang(v[1])]);
        }
        let nread = if k == 0 {
            self.n_atoms
        } else {
            self.free.as_ref().map_or(self.n_atoms, |f| f.len())
        };
        let x = self.read_floats(nread)?;
        let y = self.read_floats(nread)?;
        let z = self.read_floats(nread)?;
        if self.has_4d {
            let _ = self.read_floats(nread)?;
        }
        let coords: Vec<[f64; 3]> = if nread == self.n_atoms {
            (0..nread).map(|i| [x[i] as f64, y[i] as f64, z[i] as f64]).collect()
        } else {
            let mut c = self.fixed_coords.clone();
            for (j, &i) in self.free.as_ref().expect("free atoms").iter().enumerate() {
                c[i] = [x[j] as f64, y[j] as f64, z[j] as f64];
            }
            c
        };
        if k == 0 && self.free.is_some() {
            self.fixed_coords = coords.clone();
        }
        self.next += 1;
        Ok(Some(Frame {
            coords,
            cell,
            time: self.time_of(k),
            step: (self.nsavc > 0).then(|| self.istart + k as i64 * self.nsavc),
        }))
    }
}

impl DcdReader {
    /// Rewinds to the first frame.
    pub fn rewind(&mut self) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(self.first_frame_pos))
            .map_err(IoError::from)?;
        self.next = 0;
        Ok(())
    }
}
