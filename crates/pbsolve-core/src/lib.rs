//! # pbsolve-core
//!
//! Backend-agnostic, APBS-style finite-difference solver for the linearized
//! Poisson–Boltzmann equation (LPBE), written for MM-PBSA workloads.
//!
//! The equation is solved in APBS' dimensionless form, with the potential `u`
//! in units of kT/e and lengths in Å:
//!
//! ```text
//! -∇·(ε(r) ∇u(r)) + κ̄²(r) u(r) = 4π l_B0 Σ_i z_i δ(r - r_i)
//! ```
//!
//! where `l_B0 = e²/(4πε₀ k_B T)` is the vacuum Bjerrum length and
//! `κ̄² = ε_s κ²` is the modified Debye–Hückel parameter.
//!
//! Design rules that keep the project maintainable:
//!
//! * All physics (grids, surfaces, charges, boundary conditions, energies)
//!   lives here and is shared by every backend.
//! * A [`backend::Backend`] only has to provide a handful of kernels
//!   (stencil applications, grid transfers, dot, element-wise vector
//!   updates). The CG solver and the multigrid preconditioner are written once
//!   against it, so adding an accelerator never touches the physics.
//! * Reductions are deterministic on every backend, so results are
//!   bit-for-bit reproducible on a given platform, independent of the
//!   number of CPU threads.

pub mod autogrid;
pub mod backend;
pub mod batch;
pub mod boundary;
pub mod charge;
pub mod cpu;
pub mod error;
pub mod grid;
pub mod molecule;
pub mod multigrid;
pub mod operator;
pub mod params;
pub mod pb;
pub mod sasa;
pub mod solver;
pub mod spatial;
pub mod surface;
pub mod units;

pub use autogrid::{apbs_effective_dime, DimePolicy, GridSpec};
pub use backend::Backend;
pub use cpu::CpuBackend;
pub use error::{PbError, Result};
pub use grid::Grid;
pub use molecule::Atom;
pub use params::{BoundaryCondition, IonSpecies, PbParams, Preconditioner, SurfaceMethod};
pub use pb::{
    coefficient_maps, polar_solvation, solve_focused, solve_with_maps, FieldSolution, Medium, PolarSolvation, Timings,
};
pub use solver::{SolveStats, SolverOptions};
pub use surface::{CoefficientMaps, MolecularSurface};
