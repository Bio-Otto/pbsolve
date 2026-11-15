"""Solvation terms for single-trajectory MM-PBSA.

pbsolve provides the solvation part; the gas-phase molecular-mechanics
energies come from your MM engine (e.g. OpenMM)::

    ΔG_bind ≈ ΔE_MM + ΔG_polar + ΔG_nonpolar   (− TΔS, if estimated)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._core import GridSpec, Solver, kt_kcal_per_mol, sasa

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


def _nonpolar_terms(
    coords: npt.NDArray[np.float64], radii: npt.NDArray[np.float64], gamma: float, beta: float, probe: float
) -> tuple[float, npt.NDArray[np.float64]]:
    """:func:`nonpolar_energy` (the same value) and γ·SASA of every atom."""
    area = sasa(coords, radii, probe=probe, n_points=960)
    return gamma * float(np.sum(area)) + beta, gamma * area


@dataclass(frozen=True)
class SolvationTerms:
    """Solvation free energy of one species (kcal/mol), also per atom."""

    polar: float
    nonpolar: float
    atom_polar: Optional[npt.NDArray[np.float64]] = field(default=None, compare=False, repr=False)
    """ΔG_polar of every atom (:attr:`PolarResult.atom_energies`); ``None``
    for the nonlinear equations with ions."""
    atom_nonpolar: Optional[npt.NDArray[np.float64]] = field(default=None, compare=False, repr=False)
    """γ·SASA of every atom; the constant β belongs to no atom."""
    reference: Optional[float] = field(default=None, repr=False)
    """Electrostatic energy ½Σqu of the reference state, kcal/mol. It contains
    the grid's self-energy, so only differences on one grid mean something
    (:attr:`PBSAFrame.delta_reference`)."""

    @property
    def total(self) -> float:
        """Polar plus non-polar term."""
        return self.polar + self.nonpolar


@dataclass(frozen=True)
class PBSAFrame:
    """Solvation terms of complex, receptor and ligand for one frame."""

    complex: SolvationTerms
    receptor: SolvationTerms
    ligand: SolvationTerms
    receptor_atoms: Optional[npt.NDArray[np.int64]] = field(default=None, compare=False, repr=False)
    """The receptor's atoms in the complex (indices)."""
    ligand_atoms: Optional[npt.NDArray[np.int64]] = field(default=None, compare=False, repr=False)
    """The ligand's atoms in the complex (indices)."""

    def atom_terms(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Every complex atom's share of :attr:`delta_polar` and :attr:`delta_nonpolar`.

        Returns
        -------
        polar, nonpolar : numpy.ndarray, shape (N,)
            The complex's per-atom terms minus those of the same atom in the
            receptor or the ligand, kcal/mol. ``polar`` sums to
            :attr:`delta_polar`; ``nonpolar`` sums to :attr:`delta_nonpolar`
            + β, since β belongs to no atom.

        Raises
        ------
        ValueError
            If the frame has no per-atom terms (the nonlinear equations with
            ions, or a frame not made by :func:`pbsa_solvation_frames`).
        """
        species = (self.complex, self.receptor, self.ligand)
        if self.receptor_atoms is None or self.ligand_atoms is None or any(
            t.atom_polar is None or t.atom_nonpolar is None for t in species
        ):
            raise ValueError("this frame has no per-atom terms (nonlinear equation with ions?)")
        out = []
        for part in ("atom_polar", "atom_nonpolar"):
            whole = getattr(self.complex, part).copy()
            whole[self.receptor_atoms] -= getattr(self.receptor, part)
            whole[self.ligand_atoms] -= getattr(self.ligand, part)
            out.append(whole)
        return out[0], out[1]

    def residue_terms(self, residues: Sequence[Any]) -> dict[Any, tuple[float, float]]:
        """:meth:`atom_terms` summed per residue.

        Parameters
        ----------
        residues : sequence, length N
            A residue label for every complex atom (e.g. ``"B:LEU:26"``).

        Returns
        -------
        dict
            label → (polar, nonpolar) share of the binding terms in kcal/mol,
            in the order the labels first appear.
        """
        polar, nonpolar = self.atom_terms()
        labels = list(residues)
        if len(labels) != polar.size:
            raise ValueError(f"{len(labels)} residue labels for {polar.size} atoms")
        out: dict[Any, list[float]] = {}
        for label, p, n in zip(labels, polar.tolist(), nonpolar.tolist()):
            acc = out.setdefault(label, [0.0, 0.0])
            acc[0] += p
            acc[1] += n
        return {k: (v[0], v[1]) for k, v in out.items()}

    @property
    def delta_polar(self) -> float:
        """Polar term of binding (complex − receptor − ligand)."""
        return self.complex.polar - self.receptor.polar - self.ligand.polar

    @property
    def delta_nonpolar(self) -> float:
        """Non-polar term of binding (complex − receptor − ligand)."""
        return self.complex.nonpolar - self.receptor.nonpolar - self.ligand.nonpolar

    @property
    def delta_solvation(self) -> float:
        """Solvation free energy of binding, polar plus non-polar."""
        return self.delta_polar + self.delta_nonpolar

    @property
    def delta_reference(self) -> Optional[float]:
        """Receptor–ligand interaction energy in the reference state (complex − receptor − ligand).

        All three share the complex's grid, so the grid self-energies cancel.
        With one interior dielectric this is the Coulomb energy divided by
        ``pdie`` (1YCR: within 0.03 % at 0.35 Å). With atom dielectrics it is
        the interaction screened by them; use it in place of the force
        field's Coulomb term, so that ``delta_reference + delta_polar`` is the
        electrostatic binding energy in the solvent. ``None`` if a species
        has no reference energy.
        """
        parts = (self.complex.reference, self.receptor.reference, self.ligand.reference)
        if any(p is None for p in parts):
            return None
        return parts[0] - parts[1] - parts[2]

    def as_dict(self) -> dict[str, float]:
        """All terms as a flat dict (``complex_polar``, ..., ``delta_solvation``, ``delta_reference``)."""
        out: dict[str, float] = {}
        for name in ("complex", "receptor", "ligand"):
            terms: SolvationTerms = getattr(self, name)
            out[f"{name}_polar"] = terms.polar
            out[f"{name}_nonpolar"] = terms.nonpolar
        out["delta_polar"] = self.delta_polar
        out["delta_nonpolar"] = self.delta_nonpolar
        out["delta_solvation"] = self.delta_solvation
        if self.delta_reference is not None:
            out["delta_reference"] = self.delta_reference
        return out


def residue_dielectrics(
    charges: npt.ArrayLike,
    residues: Sequence[Any],
    *,
    charged: float,
    neutral: float,
    threshold: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Interior dielectric of every atom from the net charge of its residue.

    Residue-type dielectrics put a higher ε inside charged residues, whose
    side chains reorganise more around a charge, than in the rest of the
    protein [Wang, Zhang, Rocchia & Alexov 2013]; for MM/GBSA, Liu, Peng &
    Zhang (2019) fitted 2.7 (charged) and 1.1 (other residues) to
    protein–protein binding. Pass the result as ``dielectrics`` to
    :meth:`Solver.polar_solvation` or :func:`pbsa_solvation_frames`.

    Parameters
    ----------
    charges : array_like, shape (N,)
        Atomic charges in e.
    residues : sequence, length N
        A label per atom that is the same for the atoms of one residue, e.g.
        ``f"{chain}:{resname}:{resid}"``.
    charged, neutral : float
        ε of the atoms of charged residues and of all other atoms. No
        default: the published values were fitted for generalized Born.
    threshold : float
        A residue whose charges add up to at least this magnitude (e) is
        charged.

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    q = _vec(charges)
    if len(residues) != q.size:
        raise ValueError(f"{len(residues)} residue labels for {q.size} charges")
    for name, eps in (("charged", charged), ("neutral", neutral)):
        if not (math.isfinite(eps) and eps > 0.0):
            raise ValueError(f"{name} dielectric must be positive, got {eps}")
    net: dict[Any, float] = {}
    for label, c in zip(residues, q):
        net[label] = net.get(label, 0.0) + float(c)
    return np.array([charged if abs(net[label]) >= threshold else neutral for label in residues])


def _selections(
    n: int, ligand_atoms: Sequence[int], receptor_atoms: Optional[Sequence[int]]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
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
    return lig, rec


def pbsa_solvation_frames(
    solver: Solver,
    frames: Iterable[npt.ArrayLike],
    charges: npt.ArrayLike,
    radii: npt.ArrayLike,
    ligand_atoms: Sequence[int],
    receptor_atoms: Optional[Sequence[int]] = None,
    *,
    grid: Optional[GridSpec] = None,
    workers: Optional[int] = None,
    gamma: float = SITKOFF_GAMMA,
    beta: float = SITKOFF_BETA,
    probe: float = 1.4,
    dielectrics: Optional[npt.ArrayLike] = None,
) -> list[PBSAFrame]:
    """Single-trajectory MM-PBSA solvation terms for several frames.

    Each frame contributes three PB calculations (complex, receptor, ligand)
    on one grid: ``grid`` if given, otherwise the automatic grid of that
    frame's complex. All calculations of the given frames are solved
    up to ``workers`` at a time (default: 2 on the CPU, 3 on CUDA; on CUDA
    no more than fit in the free GPU memory, see ``Solver.batch_workers``),
    so pass frames in chunks rather than one by one. The results do not
    depend on ``workers`` and equal those of :func:`pbsa_solvation_frame`
    bit for bit. Every term also comes per atom
    (:meth:`PBSAFrame.atom_terms`, :meth:`PBSAFrame.residue_terms`).
    ``dielectrics`` gives every complex atom its own interior dielectric
    (see :meth:`Solver.polar_solvation` and :func:`residue_dielectrics`);
    receptor and ligand keep their atoms' values.
    """
    q, r = _vec(charges), _vec(radii)
    eps = None if dielectrics is None else _vec(dielectrics)
    if eps is not None and eps.shape != q.shape:
        raise ValueError(f"{eps.size} dielectrics for {q.size} atoms")
    coords = [_xyz(x) for x in frames]
    if not coords:
        return []
    lig, rec = _selections(coords[0].shape[0], ligand_atoms, receptor_atoms)
    species = (None, rec, lig)
    systems, grids, atom_eps = [], [], []
    for xyz in coords:
        if xyz.shape[0] != q.shape[0]:
            raise ValueError(f"frame has {xyz.shape[0]} atoms, charges have {q.shape[0]}")
        g = grid if grid is not None else solver.make_grid(xyz, r)
        for idx in species:
            systems.append((xyz, q, r) if idx is None else (xyz[idx], q[idx], r[idx]))
            grids.append(g)
            atom_eps.append(None if eps is None else eps if idx is None else eps[idx])
    polar = solver.polar_solvation_many(
        systems, grid=grids, workers=workers, dielectrics=None if eps is None else atom_eps
    )

    kt = kt_kcal_per_mol(solver.params["temperature"])
    out = []
    for k in range(len(coords)):
        terms = []
        for j in range(3):
            x, _, rr = systems[3 * k + j]
            res = polar[3 * k + j]
            nonpolar, atom_nonpolar = _nonpolar_terms(x, rr, gamma, beta, probe)
            terms.append(
                SolvationTerms(res.energy, nonpolar, res.atom_energies, atom_nonpolar, reference=res.reference_kt * kt)
            )
        out.append(PBSAFrame(terms[0], terms[1], terms[2], rec, lig))
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
    dielectrics: Optional[npt.ArrayLike] = None,
) -> PBSAFrame:
    """Single-trajectory MM-PBSA solvation terms for one frame.

    All three PB calculations share one grid (built from the complex unless
    ``grid`` is given), so discretisation errors cancel in the binding
    difference. ``receptor_atoms`` defaults to every atom not in the ligand;
    ``dielectrics`` as for :func:`pbsa_solvation_frames`.
    """
    return pbsa_solvation_frames(
        solver, [coords], charges, radii, ligand_atoms, receptor_atoms,
        grid=grid, workers=1, gamma=gamma, beta=beta, probe=probe, dielectrics=dielectrics,
    )[0]
