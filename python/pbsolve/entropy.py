"""Binding entropy from the fluctuations of the interaction energy.

The interaction entropy of Duan, Liu & Zhang (2016) estimates −TΔS of
binding from the snapshots of a single-trajectory MM-PBSA run:

    −TΔS_IE = kT ln ⟨exp(β (E_int − ⟨E_int⟩))⟩,

where E_int is the gas-phase interaction energy (Coulomb plus van der
Waals) between receptor and ligand in each snapshot. Its second-order
cumulant approximation (C2) is −TΔS_C2 = σ²/(2kT), with σ the standard
deviation of E_int. Both are ≥ 0. The exponential average converges very
slowly when σ is large: Ekberg & Ryde (2021) found IE impossible to converge
for σ > 15 kJ/mol (3.6 kcal/mol), recommend trusting C2 where the two
differ, and ask that σ always be reported and that both estimates be
checked over blocks of the trajectory. :func:`interaction_entropy` does all
of this; :func:`interaction_energy` gives E_int for force fields with
Lorentz–Berthelot combining rules (AMBER, CHARMM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._core import bjerrum_length, kt_kcal_per_mol

__all__ = ["InteractionEntropy", "interaction_energy", "interaction_entropy"]

#: σ (kcal/mol) above which IE cannot be converged (15 kJ/mol, Ekberg & Ryde 2021).
SIGMA_LIMIT = 15.0 / 4.184


@dataclass(frozen=True)
class InteractionEntropy:
    """−TΔS of binding from the interaction-energy fluctuations (kcal/mol)."""

    ie: float
    """−TΔS from the exponential average (interaction entropy)."""
    c2: float
    """−TΔS from the second-order cumulant, σ²/(2kT)."""
    sigma: float
    """Standard deviation of the interaction energy."""
    n: int
    """Number of snapshots."""
    ie_blocks: npt.NDArray[np.float64] = field(compare=False, repr=False)
    """IE of each consecutive block of snapshots."""
    c2_blocks: npt.NDArray[np.float64] = field(compare=False, repr=False)
    """C2 of each consecutive block of snapshots."""

    @property
    def reliable(self) -> bool:
        """Whether σ is at most 3.6 kcal/mol (15 kJ/mol), below which IE can converge."""
        return self.sigma <= SIGMA_LIMIT

    @property
    def ie_sem(self) -> float:
        """Standard error of IE from the blocks (each block's IE is biased low for small blocks)."""
        return _sem(self.ie_blocks)

    @property
    def c2_sem(self) -> float:
        """Standard error of C2 from the blocks."""
        return _sem(self.c2_blocks)


def _sem(values: npt.NDArray[np.float64]) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")


def _ie_c2(e: npt.NDArray[np.float64], kt: float) -> tuple[float, float]:
    d = (e - e.mean()) / kt
    # kT ln mean exp(d), evaluated without overflow.
    m = d.max()
    ie = kt * (m + np.log(np.mean(np.exp(d - m))))
    return float(ie), float(e.var() / (2.0 * kt))


def interaction_entropy(
    energies: npt.ArrayLike,
    temperature: float = 298.15,
    blocks: int = 5,
) -> InteractionEntropy:
    """−TΔS of binding from the interaction energy of every snapshot.

    Parameters
    ----------
    energies : array_like, shape (N,)
        Gas-phase receptor–ligand interaction energy (Coulomb plus van der
        Waals) of every snapshot in kcal/mol, e.g. from
        :func:`interaction_energy` or your MM engine.
    temperature : float
        Temperature in K.
    blocks : int
        Number of consecutive blocks for the uncertainties (at least 2
        snapshots each).

    Returns
    -------
    InteractionEntropy
        IE and C2 estimates of −TΔS, σ, and their values per block.
    """
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size < 2 or not np.all(np.isfinite(e)):
        raise ValueError("need at least two finite interaction energies")
    if blocks < 1 or e.size < 2 * blocks:
        raise ValueError(f"{e.size} snapshots cannot form {blocks} blocks of at least two")
    kt = kt_kcal_per_mol(temperature)
    ie, c2 = _ie_c2(e, kt)
    parts = [_ie_c2(b, kt) for b in np.array_split(e, blocks)]
    return InteractionEntropy(
        ie=ie,
        c2=c2,
        sigma=float(e.std()),
        n=int(e.size),
        ie_blocks=np.array([p[0] for p in parts]),
        c2_blocks=np.array([p[1] for p in parts]),
    )


def interaction_energy(
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    lj_radii: npt.ArrayLike,
    lj_epsilons: npt.ArrayLike,
    ligand_atoms: Sequence[int],
    receptor_atoms: Optional[Sequence[int]] = None,
    *,
    dielectric: float = 1.0,
) -> tuple[float, float]:
    """Coulomb and Lennard-Jones interaction energy between receptor and ligand.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Atom positions in Å.
    charges : array_like, shape (N,)
        Charges in e.
    lj_radii, lj_epsilons : array_like, shape (N,)
        Lennard-Jones R_min/2 (Å) and well depths (kcal/mol), e.g. from
        :func:`pbsolve.openmm_bridge.lennard_jones_from_system`. Pairs use
        R_min = R_i + R_j and ε = √(ε_i ε_j) (Lorentz–Berthelot, as AMBER and
        CHARMM; not OPLS or pair-specific corrections such as NBFIX).
    ligand_atoms : sequence of int
        The ligand's atoms.
    receptor_atoms : sequence of int, optional
        The receptor's atoms (default: every atom not in the ligand).
    dielectric : float
        Relative permittivity of the Coulomb term (1 in the gas phase).

    Returns
    -------
    coulomb, lennard_jones : float
        kcal/mol, every receptor–ligand pair, no cutoff.
    """
    x = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(charges, dtype=np.float64).ravel()
    r = np.asarray(lj_radii, dtype=np.float64).ravel()
    eps = np.asarray(lj_epsilons, dtype=np.float64).ravel()
    if not (q.size == r.size == eps.size == x.shape[0]):
        raise ValueError("coords, charges, lj_radii and lj_epsilons must describe the same atoms")
    lig = np.asarray(ligand_atoms, dtype=np.int64)
    if receptor_atoms is None:
        mask = np.ones(x.shape[0], dtype=bool)
        mask[lig] = False
        rec = np.nonzero(mask)[0]
    else:
        rec = np.asarray(receptor_atoms, dtype=np.int64)
    if lig.size == 0 or rec.size == 0 or np.intersect1d(lig, rec).size:
        raise ValueError("ligand and receptor must be non-empty and disjoint")
    coulomb_constant = bjerrum_length(298.15) * kt_kcal_per_mol(298.15)  # e²/4πε0, kcal·Å/mol
    elec = lj = 0.0
    for start in range(0, lig.size, 256):  # bounded memory for large ligands
        part = lig[start : start + 256]
        d = np.linalg.norm(x[rec, None, :] - x[None, part, :], axis=2)
        elec += coulomb_constant / dielectric * float(np.sum(np.outer(q[rec], q[part]) / d))
        rmin = r[rec, None] + r[None, part]
        e_ij = np.sqrt(np.outer(eps[rec], eps[part]))
        s6 = (rmin / d) ** 6
        lj += float(np.sum(e_ij * (s6 * s6 - 2.0 * s6)))
    return elec, lj
