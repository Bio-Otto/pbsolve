// SPDX-License-Identifier: MIT AND BSD-2-Clause
// The XTC decompression (`decompress_coords`, `BitReader`, `decode_ints`,
// `sizeofint(s)`, `MAGICINTS`) is a Rust port of xdrfile 1.1.4
// (Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel, BSD-2-Clause).
// See THIRD_PARTY_NOTICES.md for the full license text.

//! GROMACS XTC (compressed, single precision) and TRR (single or double
//! precision) trajectories. XDR data are big-endian; positions in nm → Å.

use crate::{format_err, path_str, Frame, FrameReader, IoError, Result};
use std::io::{BufReader, Read};
use std::path::Path;

const XTC_MAGIC: i32 = 1995;
/// GROMACS 2023+: same layout, 64-bit byte count for very large frames.
const XTC_MAGIC_2023: i32 = 2023;
const TRR_MAGIC: i32 = 1993;

struct Xdr {
    r: BufReader<std::fs::File>,
}

impl Xdr {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            r: BufReader::with_capacity(1 << 20, std::fs::File::open(path)?),
        })
    }

    /// `Ok(None)` at a clean end of file.
    fn try_i32(&mut self) -> Result<Option<i32>> {
        let mut b = [0u8; 4];
        let mut got = 0;
        while got < 4 {
            let k = self.r.read(&mut b[got..])?;
            if k == 0 {
                return if got == 0 {
                    Ok(None)
                } else {
                    format_err("truncated XDR value")
                };
            }
            got += k;
        }
        Ok(Some(i32::from_be_bytes(b)))
    }

    fn i32(&mut self) -> Result<i32> {
        self.try_i32()?
            .ok_or_else(|| IoError::Format("unexpected end of file".into()))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_be_bytes(b))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.i32()? as u32))
    }

    fn f64(&mut self) -> Result<f64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(f64::from_be_bytes(b))
    }

    fn real(&mut self, double: bool) -> Result<f64> {
        if double {
            self.f64()
        } else {
            self.f32().map(|v| v as f64)
        }
    }

    /// XDR opaque data of `n` bytes, padded to a multiple of 4.
    fn opaque(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n + (4 - n % 4) % 4];
        self.r.read_exact(&mut v)?;
        v.truncate(n);
        Ok(v)
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        std::io::copy(&mut self.r.by_ref().take(n as u64), &mut std::io::sink())?;
        Ok(())
    }
}

// ------------------------------------------------------------------ XTC

const MAGICINTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256, 322, 406, 512, 645,
    812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003, 16384, 20642, 26007, 32768, 41285,
    52015, 65536, 82570, 104031, 131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561, 832255, 1048576,
    1321122, 1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];
const FIRSTIDX: usize = 9;
const LASTIDX: usize = MAGICINTS.len();

fn sizeofint(size: u32) -> u32 {
    let mut num: u64 = 1;
    let mut bits = 0;
    while size as u64 >= num && bits < 32 {
        bits += 1;
        num <<= 1;
    }
    bits
}

fn sizeofints(sizes: &[u32; 3]) -> u32 {
    let mut bytes = [0u64; 32];
    let mut num_of_bytes = 1usize;
    bytes[0] = 1;
    for &s in sizes {
        let s = s as u64;
        let mut tmp: u64 = 0;
        let mut bytecnt = 0;
        while bytecnt < num_of_bytes {
            tmp += bytes[bytecnt] * s;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
            bytecnt += 1;
        }
        while tmp != 0 {
            bytes[bytecnt] = tmp & 0xff;
            bytecnt += 1;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }
    let mut num = 1u64;
    let mut bits = 0u32;
    num_of_bytes -= 1;
    while bytes[num_of_bytes] >= num {
        bits += 1;
        num *= 2;
    }
    bits + num_of_bytes as u32 * 8
}

struct BitReader<'a> {
    buf: &'a [u8],
    cnt: usize,
    lastbits: u32,
    lastbyte: u32,
}

impl BitReader<'_> {
    fn byte(&mut self) -> Result<u32> {
        let b = *self
            .buf
            .get(self.cnt)
            .ok_or_else(|| IoError::Format("XTC: compressed data ends early".into()))?;
        self.cnt += 1;
        Ok(b as u32)
    }

    fn bits(&mut self, mut n: u32) -> Result<i32> {
        let mask: u32 = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        let mut num: u32 = 0;
        while n >= 8 {
            self.lastbyte = (self.lastbyte << 8) | self.byte()?;
            num |= (self.lastbyte >> self.lastbits) << (n - 8);
            n -= 8;
        }
        if n > 0 {
            if self.lastbits < n {
                self.lastbits += 8;
                self.lastbyte = (self.lastbyte << 8) | self.byte()?;
            }
            self.lastbits -= n;
            num |= (self.lastbyte >> self.lastbits) & ((1u32 << n) - 1);
        }
        Ok((num & mask) as i32)
    }
}

fn decode_ints(br: &mut BitReader, mut num_of_bits: u32, sizes: &[u32; 3], nums: &mut [i32; 3]) -> Result<()> {
    let mut bytes = [0i32; 32];
    let mut nb = 0usize;
    while num_of_bits > 8 {
        bytes[nb] = br.bits(8)?;
        nb += 1;
        num_of_bits -= 8;
    }
    if num_of_bits > 0 {
        bytes[nb] = br.bits(num_of_bits)?;
        nb += 1;
    }
    for i in (1..3).rev() {
        let mut num: i64 = 0;
        for j in (0..nb).rev() {
            num = (num << 8) | bytes[j] as i64;
            let p = num / sizes[i] as i64;
            bytes[j] = p as i32;
            num -= p * sizes[i] as i64;
        }
        nums[i] = num as i32;
    }
    nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    Ok(())
}

/// xdrfile_decompress_coord_float. Returns positions in nm.
fn decompress_coords(x: &mut Xdr, natoms: usize, magic: i32) -> Result<Vec<[f32; 3]>> {
    let lsize = x.i32()?;
    if lsize as usize != natoms {
        return format_err(format!("XTC frame has {lsize} atoms, header says {natoms}"));
    }
    if natoms <= 9 {
        let mut out = Vec::with_capacity(natoms);
        for _ in 0..natoms {
            out.push([x.f32()?, x.f32()?, x.f32()?]);
        }
        return Ok(out);
    }
    let precision = x.f32()?;
    let minint = [x.i32()?, x.i32()?, x.i32()?];
    let maxint = [x.i32()?, x.i32()?, x.i32()?];
    let sizeint = [0, 1, 2].map(|d| (maxint[d] as i64 - minint[d] as i64 + 1) as u32);
    let mut bitsizeint = [0u32; 3];
    let bitsize = if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xff_ffff {
        bitsizeint = sizeint.map(sizeofint);
        0
    } else {
        sizeofints(&sizeint)
    };
    let mut smallidx = x.i32()? as usize;
    if !(FIRSTIDX..LASTIDX).contains(&smallidx) {
        return format_err(format!("XTC: corrupt smallidx {smallidx}"));
    }
    let maxidx = (smallidx + 8).min(LASTIDX);
    let _minidx = maxidx - 8;
    let mut smaller = MAGICINTS[(smallidx - 1).max(FIRSTIDX)] / 2;
    let mut smallnum = MAGICINTS[smallidx] / 2;
    let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];
    let nbytes = if magic == XTC_MAGIC_2023 {
        x.u64()? as usize
    } else {
        x.i32()? as usize
    };
    let data = x.opaque(nbytes)?;
    let mut br = BitReader {
        buf: &data,
        cnt: 0,
        lastbits: 0,
        lastbyte: 0,
    };
    let inv = 1.0f32 / precision;
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(natoms);
    let mut i = 0usize;
    let mut run: i32 = 0;
    let mut this = [0i32; 3];
    while i < natoms {
        if bitsize == 0 {
            for d in 0..3 {
                this[d] = br.bits(bitsizeint[d])?;
            }
        } else {
            decode_ints(&mut br, bitsize, &sizeint, &mut this)?;
        }
        i += 1;
        for d in 0..3 {
            this[d] += minint[d];
        }
        let mut prev = this;
        let flag = br.bits(1)?;
        let mut is_smaller = 0i32;
        if flag == 1 {
            run = br.bits(5)?;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            let mut k = 0;
            while k < run {
                let mut t = [0i32; 3];
                decode_ints(&mut br, smallidx as u32, &sizesmall, &mut t)?;
                i += 1;
                for d in 0..3 {
                    t[d] += prev[d] - smallnum;
                }
                if k == 0 {
                    // first and second atom are swapped (better compression of water)
                    std::mem::swap(&mut t, &mut prev);
                    out.push(prev.map(|v| v as f32 * inv));
                } else {
                    prev = t;
                }
                out.push(t.map(|v| v as f32 * inv));
                k += 3;
            }
        } else {
            out.push(this.map(|v| v as f32 * inv));
        }
        if is_smaller < 0 {
            smallidx -= 1;
            smallnum = smaller;
            smaller = if smallidx > FIRSTIDX {
                MAGICINTS[smallidx - 1] / 2
            } else {
                0
            };
        } else if is_smaller > 0 {
            smallidx += 1;
            smaller = smallnum;
            smallnum = MAGICINTS[smallidx] / 2;
        }
        if !(FIRSTIDX..LASTIDX).contains(&smallidx) {
            return format_err("XTC: corrupt compressed data");
        }
        sizesmall = [MAGICINTS[smallidx] as u32; 3];
    }
    if out.len() != natoms {
        return format_err(format!("XTC: decoded {} atoms, expected {natoms}", out.len()));
    }
    Ok(out)
}

fn box_to_cell(b: [[f64; 3]; 3]) -> Option<[f64; 6]> {
    let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let (a, bb, c) = (norm(b[0]), norm(b[1]), norm(b[2]));
    if a == 0.0 || bb == 0.0 || c == 0.0 {
        return None;
    }
    let ang = |u: [f64; 3], w: [f64; 3]| {
        ((u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (norm(u) * norm(w)))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    Some([
        10.0 * a,
        10.0 * bb,
        10.0 * c,
        ang(b[1], b[2]),
        ang(b[0], b[2]),
        ang(b[0], b[1]),
    ])
}

pub struct XtcReader {
    path: String,
    x: Xdr,
    n_atoms: usize,
}

impl XtcReader {
    pub fn open(path: &Path) -> Result<Self> {
        let p = path_str(path);
        let mut x = Xdr::open(path)?;
        let magic = x.i32()?;
        if magic != XTC_MAGIC && magic != XTC_MAGIC_2023 {
            return format_err(format!("{p}: not an XTC file (magic {magic})"));
        }
        let n = x.i32()?;
        if n <= 0 {
            return format_err(format!("{p}: XTC with {n} atoms"));
        }
        // reopen to start reading frames from the beginning
        Ok(Self {
            path: p,
            x: Xdr::open(path)?,
            n_atoms: n as usize,
        })
    }
}

impl FrameReader for XtcReader {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n_atoms)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let Some(magic) = self.x.try_i32()? else {
            return Ok(None);
        };
        let wrap = |e: IoError| IoError::Format(format!("{}: {e}", self.path));
        if magic != XTC_MAGIC && magic != XTC_MAGIC_2023 {
            return format_err(format!("{}: bad XTC frame magic {magic}", self.path));
        }
        let n = self.x.i32()? as usize;
        if n != self.n_atoms {
            return format_err(format!(
                "{}: frame with {n} atoms (first frame {})",
                self.path, self.n_atoms
            ));
        }
        let step = self.x.i32()? as i64;
        let time = self.x.f32()? as f64;
        let mut b = [[0.0; 3]; 3];
        for row in b.iter_mut() {
            for v in row.iter_mut() {
                *v = self.x.f32()? as f64;
            }
        }
        let c = decompress_coords(&mut self.x, n, magic).map_err(wrap)?;
        Ok(Some(Frame {
            coords: c
                .iter()
                .map(|v| [10.0 * v[0] as f64, 10.0 * v[1] as f64, 10.0 * v[2] as f64])
                .collect(),
            cell: box_to_cell(b),
            time: Some(time),
            step: Some(step),
        }))
    }
}

// ------------------------------------------------------------------ TRR

pub struct TrrReader {
    path: String,
    x: Xdr,
    n_atoms: usize,
}

impl TrrReader {
    pub fn open(path: &Path) -> Result<Self> {
        let p = path_str(path);
        let mut r = Self {
            path: p.clone(),
            x: Xdr::open(path)?,
            n_atoms: 0,
        };
        let h = r
            .header()?
            .ok_or_else(|| IoError::Format(format!("{p}: empty TRR file")))?;
        r.n_atoms = h.natoms;
        r.x = Xdr::open(path)?;
        Ok(r)
    }

    fn header(&mut self) -> Result<Option<TrrHeader>> {
        let Some(magic) = self.x.try_i32()? else {
            return Ok(None);
        };
        if magic != TRR_MAGIC {
            return format_err(format!("{}: not a TRR file (magic {magic})", self.path));
        }
        let slen = self.x.i32()?;
        let n = self.x.i32()? as usize;
        let version = self.x.opaque(n)?;
        if slen as usize != n + 1 || version != b"GMX_trn_file" {
            return format_err(format!("{}: bad TRR version string", self.path));
        }
        let mut sz = [0i32; 13];
        for v in sz.iter_mut() {
            *v = self.x.i32()?;
        }
        let [_ir, _e, box_size, vir_size, pres_size, _top, _sym, x_size, v_size, f_size, natoms, step, _nre] = sz;
        if natoms <= 0 {
            return format_err(format!("{}: TRR frame with {natoms} atoms", self.path));
        }
        let natoms = natoms as usize;
        let real = if box_size > 0 {
            box_size / 9
        } else if x_size > 0 {
            x_size / (3 * natoms as i32)
        } else if v_size > 0 {
            v_size / (3 * natoms as i32)
        } else if f_size > 0 {
            f_size / (3 * natoms as i32)
        } else {
            4
        };
        if real != 4 && real != 8 {
            return format_err(format!("{}: cannot determine TRR precision", self.path));
        }
        let double = real == 8;
        let time = self.x.real(double)?;
        let _lambda = self.x.real(double)?;
        Ok(Some(TrrHeader {
            natoms,
            step: step as i64,
            time,
            double,
            box_size: box_size as usize,
            vir_size: vir_size as usize,
            pres_size: pres_size as usize,
            x_size: x_size as usize,
            v_size: v_size as usize,
            f_size: f_size as usize,
        }))
    }
}

struct TrrHeader {
    natoms: usize,
    step: i64,
    time: f64,
    double: bool,
    box_size: usize,
    vir_size: usize,
    pres_size: usize,
    x_size: usize,
    v_size: usize,
    f_size: usize,
}

impl FrameReader for TrrReader {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n_atoms)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        loop {
            let Some(h) = self.header()? else {
                return Ok(None);
            };
            if h.natoms != self.n_atoms {
                return format_err(format!("{}: frame with {} atoms", self.path, h.natoms));
            }
            let mut cell = None;
            if h.box_size > 0 {
                let mut b = [[0.0; 3]; 3];
                for row in b.iter_mut() {
                    for v in row.iter_mut() {
                        *v = self.x.real(h.double)?;
                    }
                }
                cell = box_to_cell(b);
            }
            self.x.skip(h.vir_size + h.pres_size)?;
            if h.x_size == 0 {
                // velocity/force-only frame: skip it
                self.x.skip(h.v_size + h.f_size)?;
                continue;
            }
            let mut coords = Vec::with_capacity(h.natoms);
            for _ in 0..h.natoms {
                coords.push([
                    10.0 * self.x.real(h.double)?,
                    10.0 * self.x.real(h.double)?,
                    10.0 * self.x.real(h.double)?,
                ]);
            }
            self.x.skip(h.v_size + h.f_size)?;
            return Ok(Some(Frame {
                coords,
                cell,
                time: Some(h.time),
                step: Some(h.step),
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_sizes_match_xdrfile() {
        assert_eq!(sizeofint(0), 0);
        assert_eq!(sizeofint(1), 1);
        assert_eq!(sizeofint(255), 8);
        assert_eq!(sizeofint(256), 9);
        assert_eq!(sizeofints(&[8, 8, 8]), 10); // bits of the product 512, as in xdrfile
        assert_eq!(sizeofints(&[1000, 1000, 1000]), 30);
    }
}
