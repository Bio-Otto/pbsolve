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
    *,
    zmem: Optional[float] = None,
    lmem: float = 40.0,
    membrane_switch: float = 0.3,
) -> float:
    """Non-polar solvation energy γ·SASA + β in kcal/mol.

    With ``zmem`` the area in a membrane from ``zmem`` to ``zmem + lmem``
    does not count: every exposed point of the surface is weighted by the
    switch of GBSW (Im, Feig & Brooks 2003), 0 inside the membrane and 1 in
    the water, smooth over ``membrane_switch`` Å on each side of the
    surfaces (see :func:`sasa`).
    """
    area = float(np.sum(sasa(_xyz(coords), _vec(radii), probe, n_points, zmem, lmem, membrane_switch)))
    return gamma * area + beta


def _nonpolar_terms(
    coords: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    gamma: float,
    beta: float,
    probe: float,
    membrane: Optional[tuple[float, float]] = None,
    membrane_switch: float = 0.3,
) -> tuple[float, npt.NDArray[np.float64]]:
    """:func:`nonpolar_energy` (the same value) and γ·SASA of every atom."""
    zmem, lmem = membrane if membrane is not None else (None, 40.0)
    area = sasa(coords, radii, probe, 960, zmem, lmem, membrane_switch)
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
    nonpolar_membrane: Optional[tuple[float, float]] = None,
    membrane_switch: float = 0.3,
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
    receptor and ligand keep their atoms' values. ``nonpolar_membrane=(zmem,
    lmem)`` leaves the surface buried in a membrane out of the non-polar
    terms (:func:`nonpolar_energy`; for example the solver's own membrane,
    ``(solver.params["zmem"], solver.params["lmem"])``), with the switch
    half-width ``membrane_switch`` (Å).
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
            nonpolar, atom_nonpolar = _nonpolar_terms(x, rr, gamma, beta, probe, nonpolar_membrane, membrane_switch)
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
    nonpolar_membrane: Optional[tuple[float, float]] = None,
    membrane_switch: float = 0.3,
) -> PBSAFrame:
    """Single-trajectory MM-PBSA solvation terms for one frame.

    All three PB calculations share one grid (built from the complex unless
    ``grid`` is given), so discretisation errors cancel in the binding
    difference. ``receptor_atoms`` defaults to every atom not in the ligand;
    ``dielectrics``, ``nonpolar_membrane`` and ``membrane_switch`` as for
    :func:`pbsa_solvation_frames`.
    """
    return pbsa_solvation_frames(
        solver, [coords], charges, radii, ligand_atoms, receptor_atoms,
        grid=grid, workers=1, gamma=gamma, beta=beta, probe=probe, dielectrics=dielectrics,
        nonpolar_membrane=nonpolar_membrane, membrane_switch=membrane_switch,
    )[0]


def _groups(labels: Sequence[Any]) -> tuple[list[Any], npt.NDArray[np.int64]]:
    """The distinct labels in order of first appearance, and every atom's index among them."""
    index: dict[Any, int] = {}
    ids = np.array([index.setdefault(label, len(index)) for label in labels], dtype=np.int64)
    return list(index), ids


def _pair_systems(
    xyz: npt.NDArray[np.float64], q: npt.NDArray[np.float64], r: npt.NDArray[np.float64], ids: npt.NDArray[np.int64]
) -> tuple[list[tuple[Any, Any, Any]], list[int]]:
    """All charges, then the charges of each charged group reversed (uncharged groups have no field)."""
    flipped = [b for b in range(int(ids.max()) + 1) if np.any(q[ids == b] != 0.0)]
    systems = [(xyz, q, r)]
    for b in flipped:
        reversed_b = q.copy()
        reversed_b[ids == b] *= -1.0
        systems.append((xyz, reversed_b, r))
    return systems, flipped


def _pair_matrix(results: Sequence[Any], ids: npt.NDArray[np.int64], flipped: Sequence[int]) -> npt.NDArray[np.float64]:
    """E[a, b] = ½ Σ_{i∈a} q_i φ_b(r_i) from the atom energies of :func:`_pair_systems`' calculations.

    By linearity, with the charges of b reversed every atom's energy changes by
    q_i φ_b(r_i) outside b, while inside b the sum of the two is q_i φ_b(r_i).
    """
    base = results[0]
    if base.atom_energies is None:
        raise ValueError("pair terms need per-atom energies, which the nonlinear equations with ions do not have")
    if base.membrane_potential_energy != 0.0:
        raise ValueError(
            "pair terms leave out the energy in the membrane potential, which is linear in the charges; "
            "use membrane_potential=0"
        )
    n = int(ids.max()) + 1
    e0 = base.atom_energies
    m = np.zeros((n, n))
    for res, b in zip(results[1:], flipped):
        m[:, b] = 0.5 * np.bincount(ids, weights=e0 - res.atom_energies, minlength=n)
        m[b, b] = 0.5 * np.bincount(ids, weights=e0 + res.atom_energies, minlength=n)[b]
    return m


def residue_pair_terms(
    solver: Solver,
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    radii: npt.ArrayLike,
    residues: Sequence[Any],
    *,
    grid: Optional[GridSpec] = None,
    workers: Optional[int] = None,
    dielectrics: Optional[npt.ArrayLike] = None,
) -> tuple[list[Any], npt.NDArray[np.float64]]:
    """ΔG_polar of one structure split into pairs of residues.

    ``E[a, b] = ½ Σ_{i∈a} q_i φ_b(r_i)``: half the energy of residue a's
    charges in the reaction field φ_b of residue b's charges. Row a sums to
    residue a's share (its atoms' :attr:`PolarResult.atom_energies`), and
    all entries sum to ΔG_polar. The two residues of a pair interact by
    ``E[a, b] + E[b, a]``; the two halves agree up to the focusing and the
    boundary values. One extra PB calculation per charged residue gives its
    field (the same structure with that residue's charges reversed); all of
    them share one grid and are solved ``workers`` at a time.

    Parameters
    ----------
    solver : Solver
        Any settings with per-atom energies (not the nonlinear equations with
        ions, nor a membrane potential).
    coords, charges, radii : array_like
        As for :meth:`Solver.polar_solvation`.
    residues : sequence, length N
        A residue label for every atom (e.g. ``"B:LEU:26"``).
    grid : GridSpec, optional
        The grid of all calculations; by default the automatic grid of the
        structure.
    workers : int, optional
        As for :meth:`Solver.polar_solvation_many`.
    dielectrics : array_like, shape (N,), optional
        As for :meth:`Solver.polar_solvation`.

    Returns
    -------
    labels : list
        The residues in order of first appearance.
    terms : numpy.ndarray, shape (len(labels), len(labels))
        ``E`` in kcal/mol.
    """
    xyz, q, r = _xyz(coords), _vec(charges), _vec(radii)
    if len(residues) != q.size:
        raise ValueError(f"{len(residues)} residue labels for {q.size} atoms")
    labels, ids = _groups(residues)
    systems, flipped = _pair_systems(xyz, q, r, ids)
    g = grid if grid is not None else solver.make_grid(xyz, r)
    eps = None if dielectrics is None else [_vec(dielectrics)] * len(systems)
    results = solver.polar_solvation_many(systems, grid=g, workers=workers, dielectrics=eps)
    return labels, _pair_matrix(results, ids, flipped)


def pbsa_pair_terms(
    solver: Solver,
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    radii: npt.ArrayLike,
    ligand_atoms: Sequence[int],
    residues: Sequence[Any],
    receptor_atoms: Optional[Sequence[int]] = None,
    *,
    grid: Optional[GridSpec] = None,
    workers: Optional[int] = None,
    dielectrics: Optional[npt.ArrayLike] = None,
) -> tuple[list[Any], npt.NDArray[np.float64]]:
    """The polar binding term of one frame split into pairs of residues.

    :func:`residue_pair_terms` of the complex minus those of the receptor and
    of the ligand, each placed at its residues; all three on the complex's
    grid (or ``grid``), as :func:`pbsa_solvation_frame` computes them. The
    entries sum to :attr:`PBSAFrame.delta_polar` and row a to residue a's
    polar share in :meth:`PBSAFrame.residue_terms`. A receptor–ligand pair
    gets only the complex's terms, since the two are apart in the other
    states. The cost is one PB calculation per charged residue of each
    species on top of the three of :func:`pbsa_solvation_frame`.

    Parameters
    ----------
    solver : Solver
        As for :func:`residue_pair_terms`.
    coords, charges, radii : array_like
        The complex, as for :func:`pbsa_solvation_frame`.
    ligand_atoms : sequence of int
        The ligand's atoms in the complex.
    residues : sequence, length N
        A residue label for every complex atom.
    receptor_atoms : sequence of int, optional
        Default: every atom not in the ligand.
    grid : GridSpec, optional
        The grid of all calculations; by default the complex's automatic grid.
    workers : int, optional
        As for :meth:`Solver.polar_solvation_many`.
    dielectrics : array_like, shape (N,), optional
        Interior dielectric of every complex atom; receptor and ligand keep
        their atoms' values.

    Returns
    -------
    labels : list
        The complex's residues in order of first appearance.
    terms : numpy.ndarray, shape (len(labels), len(labels))
        ΔΔG_polar per residue pair in kcal/mol.
    """
    xyz, q, r = _xyz(coords), _vec(charges), _vec(radii)
    if len(residues) != q.size:
        raise ValueError(f"{len(residues)} residue labels for {q.size} atoms")
    eps = None if dielectrics is None else _vec(dielectrics)
    lig, rec = _selections(xyz.shape[0], ligand_atoms, receptor_atoms)
    labels, ids = _groups(residues)
    g = grid if grid is not None else solver.make_grid(xyz, r)
    parts, systems, atom_eps = [], [], []
    for idx in (None, rec, lig):
        sel = slice(None) if idx is None else idx
        sys_, flipped = _pair_systems(xyz[sel], q[sel], r[sel], ids[sel])
        parts.append((len(systems), len(sys_), sel, flipped))
        systems += sys_
        atom_eps += [None if eps is None else eps[sel]] * len(sys_)
    results = solver.polar_solvation_many(systems, grid=g, workers=workers, dielectrics=None if eps is None else atom_eps)
    total = np.zeros((len(labels), len(labels)))
    for k, (start, count, sel, flipped) in enumerate(parts):
        m = _pair_matrix(results[start:start + count], ids[sel], flipped)
        present = np.arange(m.shape[0])  # group b of the species is complex group b (ids keep the complex numbering)
        total[np.ix_(present, present)] += m if k == 0 else -m
    return labels, total
