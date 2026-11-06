//! Physical and numerical parameters. Names follow APBS keywords
//! (`pdie`, `sdie`, `srad`, `sdens`, `srfm`, `bcfl`, `ion`) so that an APBS
//! input file can be translated one-to-one for validation.

use crate::error::{invalid, Result};

/// Preconditioner of the conjugate-gradient solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preconditioner {
    /// Diagonal scaling. Iteration counts grow with grid refinement.
    Jacobi,
    /// Geometric multigrid V-cycle (Galerkin coarse operators, Chebyshev
    /// smoothing, see [`crate::multigrid`]) in f64. Falls back to Jacobi for
    /// grids it cannot coarsen far enough. Bit-identical to the default of
    /// pbsolve before mixed precision.
    Multigrid,
    /// The default: the same V-cycle evaluated in single precision (f32
    /// operators and vectors) inside the f64 conjugate-gradient iteration,
    /// which then uses the flexible (Polak–Ribière) update [Notay 2000]. The
    /// converged solution satisfies the f64 system to the same tolerance; the
    /// cycle moves half the data. Falls back to Jacobi like [`Self::Multigrid`].
    MultigridF32,
}

impl Preconditioner {
    /// Default of [`PbParams`], [`crate::solver::SolverOptions`] and the
    /// Python `Solver`.
    pub const DEFAULT: Self = Self::MultigridF32;

    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "jacobi" => Ok(Self::Jacobi),
            "multigrid" | "mg" => Ok(Self::Multigrid),
            "multigrid-f32" | "mg-f32" => Ok(Self::MultigridF32),
            other => invalid(format!(
                "unknown preconditioner '{other}' (expected 'multigrid', 'multigrid-f32' or 'jacobi')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Jacobi => "jacobi",
            Self::Multigrid => "multigrid",
            Self::MultigridF32 => "multigrid-f32",
        }
    }
}

/// Dielectric boundary definition (APBS `srfm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceMethod {
    /// Solvent-excluded (molecular) surface evaluated on the staggered
    /// (face-centred) meshes with a sharp ε jump. Analogue of APBS `srfm mol`.
    Mol,
    /// Union of van der Waals spheres, sharp jump.
    Vdw,
    /// Molecular surface evaluated on grid nodes; face dielectric is the
    /// harmonic mean of the two adjacent nodes (DelPhi/pbsa-style smoothing).
    /// Parity with APBS `srfm smol` is NOT claimed; see docs/VALIDATION.md.
    HarmonicMol,
}

impl SurfaceMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mol" => Ok(Self::Mol),
            "vdw" => Ok(Self::Vdw),
            "harmonic" | "harmonic-mol" | "hmol" => Ok(Self::HarmonicMol),
            other => invalid(format!(
                "unknown surface method '{other}' (expected 'mol', 'vdw' or 'harmonic')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mol => "mol",
            Self::Vdw => "vdw",
            Self::HarmonicMol => "harmonic",
        }
    }
}

/// Boundary condition on the outermost (coarse) grid (APBS `bcfl`).
/// Finer levels always take their boundary from the coarser solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// u = 0 on the boundary.
    Zero,
    /// Single Debye–Hückel sphere carrying the total charge.
    Sdh,
    /// Sum of Debye–Hückel spheres, one per atom (most accurate, default).
    Mdh,
    /// `Mdh` evaluated exactly at Chebyshev points of patches of the box
    /// faces and interpolated in between (error ≤ 1e-10 of the summed term
    /// magnitudes, [`crate::boundary::MDH_FAST_TOLERANCE`]). Not an APBS
    /// option; the closest APBS setting is `mdh`.
    MdhFast,
}

impl BoundaryCondition {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "zero" => Ok(Self::Zero),
            "sdh" => Ok(Self::Sdh),
            "mdh" => Ok(Self::Mdh),
            "mdh-fast" => Ok(Self::MdhFast),
            other => invalid(format!(
                "unknown boundary condition '{other}' (expected 'zero', 'sdh', 'mdh' or 'mdh-fast')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Sdh => "sdh",
            Self::Mdh => "mdh",
            Self::MdhFast => "mdh-fast",
        }
    }
}

/// One mobile ion species (APBS `ion charge .. conc .. radius ..`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IonSpecies {
    /// Valence in e.
    pub charge: f64,
    /// Concentration in mol/L.
    pub concentration: f64,
    /// Ion radius in Å (defines the ion-accessibility surface).
    pub radius: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PbParams {
    /// Solute dielectric constant.
    pub pdie: f64,
    /// Solvent dielectric constant.
    pub sdie: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Mobile ion species; empty for zero ionic strength.
    pub ions: Vec<IonSpecies>,
    /// Solvent probe radius in Å.
    pub srad: f64,
    /// Surface sphere density in points/Å² (molecular surface construction).
    pub sdens: f64,
    pub surface: SurfaceMethod,
    pub bcfl: BoundaryCondition,
    /// Relative residual tolerance ‖r‖/‖b‖ of the linear solver.
    pub tol: f64,
    /// Maximum number of linear-solver iterations per grid level.
    pub max_iter: usize,
    /// Preconditioner of the linear solver.
    pub preconditioner: Preconditioner,
}

impl Default for PbParams {
    fn default() -> Self {
        Self {
            pdie: 1.0,
            sdie: 78.54,
            temperature: 298.15,
            ions: Vec::new(),
            srad: 1.4,
            sdens: 10.0,
            surface: SurfaceMethod::Mol,
            bcfl: BoundaryCondition::Mdh,
            tol: 1.0e-8,
            max_iter: 20_000,
            preconditioner: Preconditioner::DEFAULT,
        }
    }
}

impl PbParams {
    /// Adds a symmetric 1:1 salt of the given ionic strength (mol/L).
    pub fn with_salt(mut self, ionic_strength: f64, ion_radius: f64) -> Self {
        self.ions.clear();
        if ionic_strength > 0.0 {
            self.ions.push(IonSpecies {
                charge: 1.0,
                concentration: ionic_strength,
                radius: ion_radius,
            });
            self.ions.push(IonSpecies {
                charge: -1.0,
                concentration: ionic_strength,
                radius: ion_radius,
            });
        }
        self
    }

    /// Ionic strength I = ½ Σ c z² in mol/L.
    pub fn ionic_strength(&self) -> f64 {
        let sum: f64 = self.ions.iter().map(|i| i.concentration * i.charge * i.charge).sum();
        0.5 * sum + 0.0 // `+ 0.0` turns the empty sum's −0.0 into 0.0
    }

    /// Radius used for the ion-accessibility surface: the largest radius of
    /// all listed ion species, as APBS' `Vpbe_getMaxIonRadius` (species with
    /// zero concentration count too).
    pub fn ion_exclusion_radius(&self) -> f64 {
        self.ions.iter().map(|i| i.radius).fold(0.0, f64::max)
    }

    pub fn validate(&self) -> Result<()> {
        if self.pdie.is_nan() || self.sdie.is_nan() || self.pdie <= 0.0 || self.sdie <= 0.0 {
            return invalid("pdie and sdie must be positive");
        }
        if self.temperature.is_nan() || self.temperature <= 0.0 {
            return invalid("temperature must be positive");
        }
        if self.srad < 0.0 || self.sdens <= 0.0 {
            return invalid("srad must be >= 0 and sdens > 0");
        }
        if self.tol.is_nan() || self.tol <= 0.0 || self.tol >= 1.0 || self.max_iter == 0 {
            return invalid("tol must be in (0, 1) and max_iter > 0");
        }
        for ion in &self.ions {
            if ion.concentration < 0.0 || ion.radius < 0.0 {
                return invalid("ion concentrations and radii must be non-negative");
            }
        }
        let net: f64 = self.ions.iter().map(|i| i.concentration * i.charge).sum();
        if net.abs() > 1.0e-6 {
            return invalid(format!("mobile ions are not electroneutral (Σ c·z = {net:.3e})"));
        }
        Ok(())
    }
}
