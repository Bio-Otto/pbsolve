//! # pbsolve-io
//!
//! Dependency-free readers for the structure, topology and trajectory files
//! produced by common simulation packages, so that pbsolve can take their
//! output directly:
//!
//! | kind | formats |
//! |---|---|
//! | structures | PDB (multi-model), PQR, mmCIF, GRO (multi-frame), mol2 |
//! | topologies (charges) | Amber prmtop, CHARMM/NAMD PSF, GROMACS top/itp |
//! | coordinates | Amber inpcrd/rst7 (ASCII and NetCDF) |
//! | trajectories | DCD (CHARMM/NAMD/OpenMM), XTC, TRR (GROMACS), Amber NetCDF and ASCII mdcrd |
//!
//! Everything is converted to Å, e and amu. Readers fail loudly on
//! malformed input; nothing is guessed silently except chemical elements of
//! formats that do not store them (documented per reader, see [`elements`]).

pub mod amber;
pub mod cif;
pub mod dcd;
pub mod elements;
pub mod gmxtop;
pub mod gro;
pub mod mol2;
pub mod netcdf;
pub mod pdb;
pub mod prmtop;
pub mod psf;
pub mod xdr;

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{path}: line {line}: {msg}")]
    Parse { path: String, line: usize, msg: String },
    #[error("{0}")]
    Format(String),
    #[error("{0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, IoError>;

pub(crate) fn parse_err<T>(path: &str, line: usize, msg: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: path.to_string(),
        line,
        msg: msg.into(),
    })
}

pub(crate) fn format_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(IoError::Format(msg.into()))
}

/// Per-atom data. Optional fields are `None` when the format does not carry them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtomInfo {
    pub name: String,
    pub resname: String,
    pub resid: i64,
    /// Insertion code ("" if none).
    pub icode: String,
    pub chain: String,
    /// Segment (PSF/PDB) or molecule-type name (GROMACS).
    pub segid: String,
    /// Element symbol in standard capitalisation ("C", "Cl"); "" if unknown (e.g. virtual sites).
    pub element: String,
    /// HETATM record (PDB/mmCIF).
    pub hetero: bool,
    /// Partial charge (e).
    pub charge: Option<f64>,
    /// Dielectric-boundary radius (Å), e.g. PQR or Amber `RADII`.
    pub radius: Option<f64>,
    /// Mass (amu).
    pub mass: Option<f64>,
    /// Force-field atom type, if known.
    pub atom_type: String,
}

/// One set of coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// Positions in Å.
    pub coords: Vec<[f64; 3]>,
    /// Unit cell a, b, c (Å), α, β, γ (degrees), if present.
    pub cell: Option<[f64; 6]>,
    /// Simulation time (ps), if present.
    pub time: Option<f64>,
    pub step: Option<i64>,
}

/// Atoms plus whatever the file provides: bonds and zero or more frames.
#[derive(Clone, Debug, Default)]
pub struct Structure {
    pub atoms: Vec<AtomInfo>,
    /// 0-based atom index pairs (topologies, mol2, PDB CONECT).
    pub bonds: Vec<(usize, usize)>,
    /// Coordinates stored in the file itself (structures; empty for pure topologies).
    pub frames: Vec<Frame>,
    /// Format name as detected ("pdb", "prmtop", ...).
    pub format: String,
    /// Non-fatal remarks (skipped alternate locations, guessed elements, ...).
    pub notes: Vec<String>,
    /// Whether the topology describes a periodic system (Amber IFBOX), if known.
    pub periodic: Option<bool>,
}

/// File formats understood by [`read_file`] and [`open_trajectory`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Pdb,
    Pqr,
    Cif,
    Gro,
    Mol2,
    Prmtop,
    Psf,
    GmxTop,
    Inpcrd,
    NetCdf,
    Mdcrd,
    Dcd,
    Xtc,
    Trr,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().trim_start_matches('.') {
            "pdb" | "ent" => Self::Pdb,
            "pqr" => Self::Pqr,
            "cif" | "mmcif" => Self::Cif,
            "gro" => Self::Gro,
            "mol2" => Self::Mol2,
            "prmtop" | "parm7" | "parm" => Self::Prmtop,
            "psf" => Self::Psf,
            "itp" | "gmxtop" => Self::GmxTop,
            "inpcrd" | "rst7" | "restrt" | "rst" => Self::Inpcrd,
            "nc" | "ncdf" | "netcdf" | "ncrst" => Self::NetCdf,
            "mdcrd" | "crd" | "trj" | "x" => Self::Mdcrd,
            "dcd" => Self::Dcd,
            "xtc" => Self::Xtc,
            "trr" => Self::Trr,
            other => return Err(IoError::Unsupported(format!("unknown format '{other}'"))),
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pdb => "pdb",
            Self::Pqr => "pqr",
            Self::Cif => "cif",
            Self::Gro => "gro",
            Self::Mol2 => "mol2",
            Self::Prmtop => "prmtop",
            Self::Psf => "psf",
            Self::GmxTop => "gmxtop",
            Self::Inpcrd => "inpcrd",
            Self::NetCdf => "netcdf",
            Self::Mdcrd => "mdcrd",
            Self::Dcd => "dcd",
            Self::Xtc => "xtc",
            Self::Trr => "trr",
        }
    }

    /// Guesses the format from the extension and, where extensions are
    /// ambiguous (`.top`, `.crd`, `.rst7`), from the first bytes of the file.
    pub fn detect(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let head = read_head(path, 512).unwrap_or_default();
        let is_cdf = head.starts_with(b"CDF");
        if is_cdf {
            return Ok(Self::NetCdf);
        }
        if head.starts_with(b"\x89HDF") {
            return Err(IoError::Unsupported(format!(
                "{}: NetCDF-4/HDF5 files are not supported; convert to NetCDF-3 \
                 (cpptraj: 'trajout out.nc netcdf') or pass an MDAnalysis Universe",
                path.display()
            )));
        }
        let text = String::from_utf8_lossy(&head);
        match ext.as_str() {
            "top" => {
                if text.trim_start().starts_with("%VERSION") || text.contains("%FLAG") {
                    Ok(Self::Prmtop)
                } else {
                    Ok(Self::GmxTop)
                }
            }
            "crd" => {
                // Amber inpcrd: title, then "natoms [time]". mdcrd: title, then data lines.
                let mut lines = text.lines().skip(1);
                let second = lines.next().unwrap_or("");
                let n = second.split_whitespace().count();
                if (1..=2).contains(&n)
                    && second
                        .split_whitespace()
                        .next()
                        .is_some_and(|t| t.parse::<u64>().is_ok())
                {
                    Ok(Self::Inpcrd)
                } else {
                    Ok(Self::Mdcrd)
                }
            }
            "" => format_err(format!("{}: cannot detect format without an extension", path.display())),
            other => Self::parse(other),
        }
    }

    /// True for formats [`open_trajectory`] reads frame by frame.
    pub fn is_trajectory(&self) -> bool {
        matches!(self, Self::Dcd | Self::Xtc | Self::Trr | Self::NetCdf | Self::Mdcrd)
    }
}

fn read_head(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        let k = f.read(&mut buf[got..])?;
        if k == 0 {
            break;
        }
        got += k;
    }
    buf.truncate(got);
    Ok(buf)
}

/// Options for [`read_file`].
#[derive(Clone, Debug, Default)]
pub struct ReadOptions {
    /// Extra directories searched for GROMACS `#include` files.
    pub include_dirs: Vec<std::path::PathBuf>,
}

/// Reads a structure, topology or coordinate file. Trajectory formats are
/// read completely (use [`open_trajectory`] to stream them).
pub fn read_file(path: &Path, format: Option<Format>, opts: &ReadOptions) -> Result<Structure> {
    let fmt = match format {
        Some(f) => f,
        None => Format::detect(path)?,
    };
    let mut s = match fmt {
        Format::Pdb => pdb::read_pdb(path)?,
        Format::Pqr => pdb::read_pqr(path)?,
        Format::Cif => cif::read_cif(path)?,
        Format::Gro => gro::read_gro(path)?,
        Format::Mol2 => mol2::read_mol2(path)?,
        Format::Prmtop => prmtop::read_prmtop(path)?,
        Format::Psf => psf::read_psf(path)?,
        Format::GmxTop => gmxtop::read_top(path, &opts.include_dirs)?,
        Format::Inpcrd => amber::read_inpcrd(path)?,
        Format::NetCdf if netcdf::is_restart(path)? => netcdf::read_restart(path)?,
        Format::NetCdf | Format::Mdcrd | Format::Dcd | Format::Xtc | Format::Trr => {
            let mut r = open_trajectory(path, Some(fmt), None, None)?;
            let mut frames = Vec::new();
            while let Some(f) = r.next_frame()? {
                frames.push(f);
            }
            let n = frames.first().map(|f| f.coords.len()).unwrap_or(0);
            Structure {
                atoms: vec![AtomInfo::default(); n],
                frames,
                ..Structure::default()
            }
        }
    };
    s.format = fmt.as_str().to_string();
    Ok(s)
}

/// Streaming access to the frames of a trajectory.
pub trait FrameReader: Send {
    /// Number of atoms per frame, if known before reading.
    fn n_atoms(&self) -> Option<usize>;
    /// Total number of frames, if cheaply known.
    fn n_frames(&self) -> Option<usize> {
        None
    }
    /// The next frame, or `None` at the end of the file.
    fn next_frame(&mut self) -> Result<Option<Frame>>;
}

/// Opens a trajectory. `n_atoms` is required for ASCII mdcrd and checked
/// otherwise; `mdcrd_box` says whether mdcrd frames carry a box line
/// (`None`: detect).
pub fn open_trajectory(
    path: &Path,
    format: Option<Format>,
    n_atoms: Option<usize>,
    mdcrd_box: Option<bool>,
) -> Result<Box<dyn FrameReader>> {
    let fmt = match format {
        Some(f) => f,
        None => Format::detect(path)?,
    };
    let r: Box<dyn FrameReader> = match fmt {
        Format::Dcd => Box::new(dcd::DcdReader::open(path)?),
        Format::Xtc => Box::new(xdr::XtcReader::open(path)?),
        Format::Trr => Box::new(xdr::TrrReader::open(path)?),
        Format::NetCdf => Box::new(netcdf::AmberNetCdf::open(path)?),
        Format::Mdcrd => {
            let n = n_atoms.ok_or_else(|| {
                IoError::Format("ASCII mdcrd files need the number of atoms (load the topology first)".into())
            })?;
            Box::new(amber::MdcrdReader::open(path, n, mdcrd_box)?)
        }
        Format::Pdb | Format::Pqr | Format::Gro | Format::Cif | Format::Mol2 | Format::Inpcrd => {
            let s = read_file(path, Some(fmt), &ReadOptions::default())?;
            Box::new(VecReader {
                n_atoms: s.atoms.len(),
                frames: s.frames.into_iter(),
            })
        }
        Format::Prmtop | Format::Psf | Format::GmxTop => {
            return format_err(format!("{}: a topology has no frames", path.display()))
        }
    };
    if let (Some(want), Some(have)) = (n_atoms, r.n_atoms()) {
        if want != have {
            return format_err(format!(
                "{}: trajectory has {have} atoms, topology has {want}",
                path.display()
            ));
        }
    }
    Ok(r)
}

/// Frames already in memory (multi-model structure files).
pub struct VecReader {
    n_atoms: usize,
    frames: std::vec::IntoIter<Frame>,
}

impl FrameReader for VecReader {
    fn n_atoms(&self) -> Option<usize> {
        Some(self.n_atoms)
    }
    fn n_frames(&self) -> Option<usize> {
        Some(self.frames.len())
    }
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        Ok(self.frames.next())
    }
}

/// Splits fixed-width fields, returning trimmed slices (missing columns -> "").
pub(crate) fn col(line: &str, start: usize, end: usize) -> &str {
    let b = line.as_bytes();
    let s = start.min(b.len());
    let e = end.min(b.len());
    line.get(s..e).unwrap_or("").trim()
}

pub(crate) fn path_str(p: &Path) -> String {
    p.display().to_string()
}
