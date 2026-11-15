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
    /// APBS `srfm spl2`: van der Waals spheres smoothed by a cubic switch of
    /// half-width `swin` ([`crate::spline`]); differentiable, so forces work.
    Spline2,
    /// APBS `srfm spl4`: the same with a septic switch.
    Spline4,
    /// Gaussian dielectric (not an APBS option): ε from the atoms' Gaussian
    /// densities of width `gauss_sigma`·R, with no surface; the reference
    /// state keeps it inside the `gauss_contour` ([`crate::gaussian`]).
    Gaussian,
}

impl SurfaceMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mol" => Ok(Self::Mol),
            "vdw" => Ok(Self::Vdw),
            "harmonic" | "harmonic-mol" | "hmol" => Ok(Self::HarmonicMol),
            "spl2" => Ok(Self::Spline2),
            "spl4" => Ok(Self::Spline4),
            "gauss" => Ok(Self::Gaussian),
            other => invalid(format!(
                "unknown surface method '{other}' (expected 'mol', 'vdw', 'harmonic', 'spl2', 'spl4' or 'gauss')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mol => "mol",
            Self::Vdw => "vdw",
            Self::HarmonicMol => "harmonic",
            Self::Spline2 => "spl2",
            Self::Spline4 => "spl4",
            Self::Gaussian => "gauss",
        }
    }

    /// `spl2` or `spl4`: smooth in the atom positions.
    pub fn is_spline(&self) -> bool {
        matches!(self, Self::Spline2 | Self::Spline4)
    }

    /// The spline surfaces or `gauss`: solvated maps that vary smoothly
    /// with position, built on the host.
    pub fn is_smooth(&self) -> bool {
        self.is_spline() || *self == Self::Gaussian
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

/// Assignment of point charges to grid nodes (APBS `chgm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChargeMethod {
    /// Trilinear, 2 nodes per axis (APBS `spl0`, the default).
    #[default]
    Trilinear,
    /// B-spline of order 3 (piecewise quadratic, C¹), 3 nodes per axis
    /// (APBS `spl2`). The energy is taken with the same weights.
    Spline2,
    /// B-spline of order 5 (piecewise quartic, C³), 5 nodes per axis (APBS
    /// `spl4`). As in APBS, the energy is taken with trilinear weights.
    Spline4,
}

impl ChargeMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "spl0" => Ok(Self::Trilinear),
            "spl2" => Ok(Self::Spline2),
            "spl4" => Ok(Self::Spline4),
            other => invalid(format!(
                "unknown charge method '{other}' (expected 'spl0', 'spl2' or 'spl4')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trilinear => "spl0",
            Self::Spline2 => "spl2",
            Self::Spline4 => "spl4",
        }
    }

    /// The assignment whose weights give the energy ½ Σ q u (APBS
    /// `Vpmg_qfEnergy`: the source's own weights for `spl2`, trilinear otherwise).
    pub fn energy_method(&self) -> Self {
        match self {
            Self::Spline2 => Self::Spline2,
            _ => Self::Trilinear,
        }
    }
}

/// Dielectric constant of the faces that the molecular surface cuts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterfaceMethod {
    /// `pdie` or `sdie` by the side of the surface the face's midpoint lies
    /// on (APBS `srfm mol`, the default); first order.
    #[default]
    Sharp,
    /// Harmonic average of `pdie` and `sdie` weighted by the lengths of the
    /// face's grid edge inside and outside the solute, computed exactly
    /// from the spheres of the surface. Not an APBS option.
    Fractional,
}

impl InterfaceMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "sharp" => Ok(Self::Sharp),
            "fractional" => Ok(Self::Fractional),
            other => invalid(format!(
                "unknown interface method '{other}' (expected 'sharp' or 'fractional')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sharp => "sharp",
            Self::Fractional => "fractional",
        }
    }
}

/// The Poisson–Boltzmann equation of the solvated state (APBS `lpbe` /
/// `npbe`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Equation {
    /// Linearized: the mobile ions respond as κ̄² u (APBS `lpbe`, the default).
    #[default]
    Linear,
    /// Nonlinear: Boltzmann-distributed ions, κ̄² Σ_j (c_j q_j / 2I)
    /// sinh-like terms (APBS `npbe`), solved by Newton's method.
    Nonlinear,
    /// Nonlinear with ions of finite size: lattice-gas ions that each fill a
    /// cube of edge `ion_size` [Borukhov et al. 1997], all of one size (APBS
    /// `smpbe` with `size 1`, which APBS 3.4.1 disables).
    SizeModified,
}

impl Equation {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "lpbe" => Ok(Self::Linear),
            "npbe" => Ok(Self::Nonlinear),
            "smpbe" => Ok(Self::SizeModified),
            other => invalid(format!(
                "unknown equation '{other}' (expected 'lpbe', 'npbe' or 'smpbe')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "lpbe",
            Self::Nonlinear => "npbe",
            Self::SizeModified => "smpbe",
        }
    }
}

/// How ΔG_polar removes the Coulomb potential of the charges, and with it
/// the grid's self-energy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoulombMethod {
    /// Solve a reference state (ε = `pdie`, no ions) on the same grid and
    /// subtract it (APBS, the default): two solves per grid level.
    #[default]
    Grid,
    /// Reaction-field decomposition: the analytic Coulomb potential is
    /// subtracted near the solute and only the reaction field is solved
    /// for, one solve per grid level. Not an APBS option.
    Analytic,
}

impl CoulombMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "grid" => Ok(Self::Grid),
            "analytic" => Ok(Self::Analytic),
            other => invalid(format!(
                "unknown coulomb method '{other}' (expected 'grid' or 'analytic')"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Analytic => "analytic",
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
    /// Half-width (Å) of the switching window of the spline surfaces
    /// (APBS `swin`).
    pub swin: f64,
    /// Width of an atom's Gaussian density relative to its radius
    /// (`gauss` only; σ of Li et al. 2013).
    pub gauss_sigma: f64,
    /// Dielectric value of the contour inside which the reference state of
    /// `gauss` keeps the Gaussian ε (Li et al. 2014).
    pub gauss_contour: f64,
    /// Dielectric of the faces the surface cuts (`mol` and `vdw` only).
    pub interface: InterfaceMethod,
    /// How ΔG_polar removes the Coulomb potential (reference solve or
    /// reaction-field decomposition).
    pub coulomb: CoulombMethod,
    /// Linearized, nonlinear or size-modified Poisson–Boltzmann equation.
    pub equation: Equation,
    /// Edge (Å) of the lattice cell an ion fills ([`Equation::SizeModified`]
    /// only; 0 otherwise).
    pub ion_size: f64,
    pub bcfl: BoundaryCondition,
    /// Charge assignment to the grid.
    pub chgm: ChargeMethod,
    /// Relative residual tolerance ‖r‖/‖b‖ of the linear solver.
    pub tol: f64,
    /// Maximum number of linear-solver iterations per grid level.
    pub max_iter: usize,
    /// Preconditioner of the linear solver.
    pub preconditioner: Preconditioner,
    /// Interior dielectric of every atom (opt-in, not APBS; DelPhi-style
    /// multiple dielectrics): a face inside the molecular surface takes the
    /// value of the atom that owns it ([`crate::surface::atom_dielectric_faces`]),
    /// and `pdie` is the dielectric outside the surface in the reference state.
    /// `None`: `pdie` everywhere inside.
    pub atom_pdie: Option<std::sync::Arc<[f64]>>,
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
            swin: 0.3,
            gauss_sigma: 0.93,
            gauss_contour: 20.0,
            interface: InterfaceMethod::Sharp,
            coulomb: CoulombMethod::Grid,
            equation: Equation::Linear,
            ion_size: 0.0,
            bcfl: BoundaryCondition::Mdh,
            chgm: ChargeMethod::Trilinear,
            tol: 1.0e-8,
            max_iter: 20_000,
            preconditioner: Preconditioner::DEFAULT,
            atom_pdie: None,
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

    /// How far inside an atom's radius the solvated maps may still differ
    /// from the reference state (ε = `pdie`, no ions): the switching window
    /// of the spline surfaces, 0 for the sharp ones.
    pub fn pdie_inset(&self) -> f64 {
        if self.surface.is_spline() {
            self.swin
        } else {
            0.0
        }
    }

    /// Bulk volume fraction of the ions in lattice cells of edge `ion_size`:
    /// Σ_j c_j N_A a³ (0 for point ions).
    pub fn ion_volume_fraction(&self) -> f64 {
        let per_molar = crate::units::AVOGADRO * 1.0e-27 * self.ion_size.powi(3);
        self.ions.iter().map(|i| i.concentration * per_molar).sum()
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
        if self.interface == InterfaceMethod::Fractional
            && !matches!(self.surface, SurfaceMethod::Mol | SurfaceMethod::Vdw)
        {
            return invalid("interface 'fractional' needs surface 'mol' or 'vdw'");
        }
        if self.swin.is_nan() || self.swin <= 0.0 {
            return invalid("swin must be positive");
        }
        if !self.gauss_sigma.is_finite() || self.gauss_sigma <= 0.0 {
            return invalid("gauss_sigma must be positive");
        }
        if self.surface == SurfaceMethod::Gaussian {
            if self.gauss_contour.is_nan() || self.gauss_contour <= self.pdie {
                return invalid("gauss_contour must exceed pdie");
            }
            if self.coulomb == CoulombMethod::Analytic {
                return invalid("surface 'gauss' needs coulomb 'grid': its reference state is not a homogeneous pdie");
            }
            if self.equation == Equation::Nonlinear {
                return invalid(
                    "surface 'gauss' with equation 'npbe': the ions' Born penalty is bounded but the potential \
                     next to a point charge is not, so their Boltzmann density diverges at the charges inside \
                     the molecule; use 'lpbe' or 'smpbe' (bounded ion density)",
                );
            }
        }
        if let Some(eps) = &self.atom_pdie {
            if eps.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                return invalid("atom dielectrics must be positive and finite");
            }
            if !matches!(self.surface, SurfaceMethod::Mol | SurfaceMethod::Vdw)
                || self.interface != InterfaceMethod::Sharp
            {
                return invalid("atom dielectrics need surface 'mol' or 'vdw' with interface 'sharp'");
            }
            if self.coulomb == CoulombMethod::Analytic {
                return invalid("atom dielectrics need coulomb 'grid': the decomposition assumes pdie inside");
            }
        }
        if self.equation != Equation::Linear && self.coulomb == CoulombMethod::Analytic {
            return invalid(format!("equation '{}' needs coulomb 'grid'", self.equation.as_str()));
        }
        if self.equation == Equation::SizeModified {
            if self.ion_size.is_nan() || self.ion_size <= 0.0 {
                return invalid("equation 'smpbe' needs ion_size > 0");
            }
            let fraction = self.ion_volume_fraction();
            if fraction >= 1.0 {
                return invalid(format!(
                    "ions of size {} Å at these concentrations fill the solvent (volume fraction {fraction:.2})",
                    self.ion_size
                ));
            }
        } else if self.ion_size != 0.0 {
            return invalid("ion_size applies to equation 'smpbe'");
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
        if self.surface == SurfaceMethod::Gaussian {
            // One accessibility map for all species: their Born penalties z²/r must agree.
            let mut present = self.ions.iter().filter(|i| i.concentration > 0.0);
            if let Some(first) = present.next() {
                if first.radius <= 0.0 {
                    return invalid("surface 'gauss' needs ions of positive radius (their Born penalty)");
                }
                if present.any(|i| i.charge.abs() != first.charge.abs() || i.radius != first.radius) {
                    return invalid("surface 'gauss' needs ions of one valence and one radius");
                }
            }
        }
        Ok(())
    }
}
