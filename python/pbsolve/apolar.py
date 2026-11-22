"""Non-polar solvation energy as APBS ``apolar`` [Wagoner & Baker 2006].

``G_np = γ A + p V + ρ̄ Σ_i ∫ u_i^att θ dy``: surface tension times the
solvent-accessible surface area, pressure times the solvent-accessible
volume, and the attractive (Weeks–Chandler–Andersen) part of each atom's
Lennard-Jones interaction with water, integrated over the solvent. The
three terms equal APBS' to the digits it prints (docs/VALIDATION.md).

The defaults are the alkane protocol of Wagoner & Baker that APBS ships as
an example, in kcal/mol, except ``grid``: 0.3 Å there, 0.5 Å here (four times
faster for proteins; 1YCR's energy changes by 0.05 %).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

from . import _core

KJ_PER_KCAL = 4.184
GAMMA = 0.0085 / KJ_PER_KCAL
"""Surface tension of the alkane protocol, kcal/(mol·Å²) (APBS: 0.0085 kJ/(mol·Å²))."""
PRESS = 0.2394 / KJ_PER_KCAL
"""Solvent pressure of the alkane protocol, kcal/(mol·Å³) (APBS: 0.2394 kJ/(mol·Å³))."""
BCONC = 0.033428
"""Bulk number density of water, Å⁻³."""
WATER_RADIUS = 1.7683
"""Lennard-Jones radius R_min/2 of water oxygen (TIP3P), Å."""
WATER_EPSILON = 0.6364 / KJ_PER_KCAL
"""Lennard-Jones well depth of water oxygen (TIP3P), kcal/mol (APBS: 0.6364 kJ/mol)."""

__all__ = ["ApolarEnergy", "apolar_energy"]

_SMALL = 1e-9  # APBS skips a term whose coefficient is at most VSMALL.


@dataclass(frozen=True)
class ApolarEnergy:
    """Non-polar solvation energy and its parts (kcal/mol, Å², Å³).

    A part whose coefficient is 0 (at most 1e-9 in magnitude, as in APBS) is
    not computed: its energy is 0 and its area or volume NaN.
    """

    energy: float
    """γ A + p V + dispersion, kcal/mol."""
    surface: float
    """γ A, kcal/mol."""
    pressure_volume: float
    """p V, kcal/mol."""
    dispersion: float
    """The integrated WCA dispersion, kcal/mol."""
    area: float
    """Solvent-accessible surface area A, Å²."""
    volume: float
    """Solvent-accessible volume V, Å³."""
    atom_area: npt.NDArray[np.float64]
    """Area of each atom, Å²."""
    atom_dispersion: npt.NDArray[np.float64]
    """Dispersion energy of each atom, kcal/mol."""

    def residue_terms(self, residues: Sequence[Any]) -> dict[Any, tuple[float, float]]:
        """The surface (γ·A_i) and dispersion energies summed per residue.

        Parameters
        ----------
        residues : sequence, length N
            A residue label for every atom.

        Returns
        -------
        dict
            label → (surface, dispersion) in kcal/mol, in the order the labels
            first appear. They sum to :attr:`surface` and :attr:`dispersion`;
            :attr:`pressure_volume` belongs to no atom (the volume is the
            molecule's).
        """
        labels = list(residues)
        if len(labels) != self.atom_dispersion.size:
            raise ValueError(f"{len(labels)} residue labels for {self.atom_dispersion.size} atoms")
        gamma = self.surface / self.area if self.area > 0.0 else 0.0  # area is NaN when γ = 0
        surface = gamma * self.atom_area if gamma else np.zeros_like(self.atom_dispersion)
        out: dict[Any, list[float]] = {}
        for label, a, d in zip(labels, surface.tolist(), self.atom_dispersion.tolist()):
            acc = out.setdefault(label, [0.0, 0.0])
            acc[0] += a
            acc[1] += d
        return {k: (v[0], v[1]) for k, v in out.items()}


def apolar_energy(
    coords: npt.ArrayLike,
    radii: npt.ArrayLike,
    epsilons: Optional[npt.ArrayLike] = None,
    *,
    gamma: float = GAMMA,
    press: float = PRESS,
    bconc: float = BCONC,
    srad: float = 0.65,
    sdens: float = 100.0,
    dpos: float = 0.2,
    grid: float = 0.5,
    water_radius: float = WATER_RADIUS,
    water_epsilon: float = WATER_EPSILON,
) -> ApolarEnergy:
    """Non-polar solvation energy γA + pV + dispersion, as APBS ``apolar``.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Atom positions in Å.
    radii : array_like, shape (N,)
        Lennard-Jones radii R_min/2 in Å (APBS takes them from its parameter
        file); they also define the solvent-accessible surface and volume.
    epsilons : array_like, shape (N,), optional
        Lennard-Jones well depths in kcal/mol; needed when ``bconc`` is not 0.
    gamma : float
        Surface tension in kcal/(mol·Å²).
    press : float
        Solvent pressure in kcal/(mol·Å³).
    bconc : float
        Bulk solvent number density in Å⁻³ (0: no dispersion term).
    srad : float
        Solvent probe radius in Å.
    sdens : float
        Surface points per Å² of the largest atom sphere (area).
    dpos : float
        APBS ``dpos`` in Å; it only enlarges the reference sphere and the
        volume's box, as in APBS.
    grid : float
        Spacing in Å of the volume and dispersion integrals.
    water_radius, water_epsilon : float
        Lennard-Jones radius (Å) and well depth (kcal/mol) of water oxygen.

    Returns
    -------
    ApolarEnergy
        The energy and its parts.

    Raises
    ------
    ValueError
        On inconsistent shapes, non-positive spacings or densities, or a
        dispersion term without ``epsilons``.
    """
    x = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    r = np.ascontiguousarray(radii, dtype=np.float64).ravel()
    use = [abs(c) > _SMALL for c in (gamma, press, bconc)]
    if use[2] and epsilons is None:
        raise ValueError("the dispersion term (bconc != 0) needs the atoms' Lennard-Jones epsilons")
    e = np.zeros(len(r)) if epsilons is None else np.ascontiguousarray(epsilons, dtype=np.float64).ravel()
    area, volume, disp = _core._apolar_terms(
        x, r, e, srad, sdens, dpos, [grid] * 3, bconc, water_radius, water_epsilon, *use
    )
    atom_area = np.full(len(r), np.nan) if area is None else area
    atom_disp = np.zeros(len(r)) if disp is None else disp
    total_area = float(np.sum(atom_area)) if area is not None else float("nan")
    vol = float("nan") if volume is None else volume
    surface = gamma * total_area if area is not None else 0.0
    pv = press * vol if volume is not None else 0.0
    dispersion = float(np.sum(atom_disp))
    return ApolarEnergy(
        energy=surface + pv + dispersion,
        surface=surface,
        pressure_volume=pv,
        dispersion=dispersion,
        area=total_area,
        volume=vol,
        atom_area=atom_area,
        atom_dispersion=atom_disp,
    )
