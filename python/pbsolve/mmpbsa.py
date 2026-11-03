"""Solvation terms for single-trajectory MM-PBSA.

pbsolve provides the solvation part; the gas-phase molecular-mechanics
energies come from your MM engine (e.g. OpenMM)::

    ΔG_bind ≈ ΔE_MM + ΔG_polar + ΔG_nonpolar   (− TΔS, if estimated)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._core import GridSpec, Solver, sasa

#: Non-polar model of Sitkoff, Sharp & Honig (1994): ΔG_np = γ·SASA + β.
SITKOFF_GAMMA = 0.00542  # kcal/(mol·Å²)
SITKOFF_BETA = 0.92  # kcal/mol


def _xyz(coords: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.ascontiguousarray(coords, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N, 3), got {a.shape}")
    return a


def _vec(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(values, dtype=np.float64).reshape(-1)


def nonpolar_energy(
    coords: npt.ArrayLike,
    radii: npt.ArrayLike,
    gamma: float = SITKOFF_GAMMA,
    beta: float = SITKOFF_BETA,
    probe: float = 1.4,
    n_points: int = 960,
) -> float:
    """Non-polar solvation energy γ·SASA + β in kcal/mol."""
    area = float(np.sum(sasa(_xyz(coords), _vec(radii), probe=probe, n_points=n_points)))
    return gamma * area + beta


@dataclass(frozen=True)
class SolvationTerms:
    """Solvation free energy of one species (kcal/mol)."""

    polar: float
    nonpolar: float

    @property
    def total(self) -> float:
        return self.polar + self.nonpolar


@dataclass(frozen=True)
class PBSAFrame:
    """Solvation terms of complex, receptor and ligand for one frame."""

    complex: SolvationTerms
    receptor: SolvationTerms
    ligand: SolvationTerms

    @property
    def delta_polar(self) -> float:
        return self.complex.polar - self.receptor.polar - self.ligand.polar

    @property
    def delta_nonpolar(self) -> float:
        return self.complex.nonpolar - self.receptor.nonpolar - self.ligand.nonpolar

    @property
    def delta_solvation(self) -> float:
        return self.delta_polar + self.delta_nonpolar

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in ("complex", "receptor", "ligand"):
            terms: SolvationTerms = getattr(self, name)
            out[f"{name}_polar"] = terms.polar
            out[f"{name}_nonpolar"] = terms.nonpolar
        out["delta_polar"] = self.delta_polar
        out["delta_nonpolar"] = self.delta_nonpolar
        out["delta_solvation"] = self.delta_solvation
        return out


def pbsa_solvation_frame(
    solver: Solver,
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    radii: npt.ArrayLike,
    ligand_atoms: Sequence[int],
    receptor_atoms: Optional[Sequence[int]] = None,
    *,
    grid: Optional[GridSpec] = None,
    gamma: float = SITKOFF_GAMMA,
    beta: float = SITKOFF_BETA,
    probe: float = 1.4,
) -> PBSAFrame:
    """Single-trajectory MM-PBSA solvation terms for one frame.

    All three PB calculations share one grid (built from the complex unless
    ``grid`` is given), so discretisation errors cancel in the binding
    difference. ``receptor_atoms`` defaults to every atom not in the ligand.
    """
    xyz, q, r = _xyz(coords), _vec(charges), _vec(radii)
    n = xyz.shape[0]
    lig = np.asarray(ligand_atoms, dtype=np.int64)
    if receptor_atoms is None:
        mask = np.ones(n, dtype=bool)
        mask[lig] = False
        rec = np.nonzero(mask)[0]
    else:
        rec = np.asarray(receptor_atoms, dtype=np.int64)
    if lig.size == 0 or rec.size == 0:
        raise ValueError("ligand and receptor selections must both be non-empty")
    if np.intersect1d(lig, rec).size:
        raise ValueError("ligand and receptor selections overlap")

    if grid is None:
        grid = solver.make_grid(xyz, r)

    def terms(idx: Optional[npt.NDArray[np.int64]]) -> SolvationTerms:
        x, c, rr = (xyz, q, r) if idx is None else (xyz[idx], q[idx], r[idx])
        polar = solver.polar_solvation(x, c, rr, grid=grid).energy
        return SolvationTerms(polar=polar, nonpolar=nonpolar_energy(x, rr, gamma, beta, probe))

    return PBSAFrame(complex=terms(None), receptor=terms(rec), ligand=terms(lig))
