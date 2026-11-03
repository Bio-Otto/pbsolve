//! Physical constants (CODATA 2018 exact SI values) and unit conversions.
//!
//! Internal units: lengths in Å, charges in e, potential in kT/e,
//! energies in kT. Conversions to kcal/mol and kJ/mol happen at the edges.

use crate::params::IonSpecies;
use std::f64::consts::PI;

pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19; // C
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12; // F/m
pub const BOLTZMANN: f64 = 1.380_649e-23; // J/K
pub const AVOGADRO: f64 = 6.022_140_76e23; // 1/mol
pub const JOULE_PER_KCAL: f64 = 4184.0;

/// Vacuum Bjerrum length e²/(4πε₀ k_B T) in Å (≈ 560.4 Å at 298.15 K).
pub fn bjerrum_length_vacuum(temperature: f64) -> f64 {
    ELEMENTARY_CHARGE * ELEMENTARY_CHARGE / (4.0 * PI * VACUUM_PERMITTIVITY * BOLTZMANN * temperature) * 1.0e10
}

/// k_B T in kcal/mol.
pub fn kt_in_kcal_per_mol(temperature: f64) -> f64 {
    BOLTZMANN * AVOGADRO * temperature / JOULE_PER_KCAL
}

/// k_B T in kJ/mol (APBS reports energies in kJ/mol).
pub fn kt_in_kj_per_mol(temperature: f64) -> f64 {
    BOLTZMANN * AVOGADRO * temperature / 1000.0
}

/// Modified Debye–Hückel parameter κ̄² = ε_s κ² in Å⁻².
///
/// κ̄² = 4π l_B0 Σ_i n_i z_i², with n_i the number density in Å⁻³
/// (n_i = c_i[mol/L] · N_A · 10⁻²⁷).
pub fn kappa_bar_squared(ions: &[IonSpecies], temperature: f64) -> f64 {
    let lb0 = bjerrum_length_vacuum(temperature);
    let sum_cz2: f64 = ions.iter().map(|ion| ion.concentration * ion.charge * ion.charge).sum();
    4.0 * PI * lb0 * AVOGADRO * 1.0e-27 * sum_cz2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bjerrum_length_at_room_temperature() {
        let lb = bjerrum_length_vacuum(298.15);
        assert!((lb - 560.4).abs() < 0.5, "l_B0 = {lb}");
    }

    #[test]
    fn debye_length_of_150mm_salt() {
        let ions = [
            IonSpecies {
                charge: 1.0,
                concentration: 0.15,
                radius: 2.0,
            },
            IonSpecies {
                charge: -1.0,
                concentration: 0.15,
                radius: 2.0,
            },
        ];
        let kappa = (kappa_bar_squared(&ions, 298.15) / 78.54).sqrt();
        let debye = 1.0 / kappa;
        assert!((debye - 7.85).abs() < 0.1, "Debye length = {debye}");
    }
}
