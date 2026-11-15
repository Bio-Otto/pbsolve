"""Bridge between OpenMM and pbsolve (requires the ``openmm`` package).

Typical use inside an OpenMM-based MM-GBSA/MM-PBSA tool::

    from pbsolve.openmm_bridge import OpenMMPBSA

    pbsa = OpenMMPBSA(system, topology, ligand_atoms=lig_idx,
                      solver=pbsolve.Solver(grid_spacing=0.5, ionic_strength=0.15))
    for positions in frames:                  # OpenMM Quantity or Å array
        terms = pbsa.frame(positions)         # PBSAFrame
        dg = delta_e_mm + terms.delta_solvation

The System must describe the solute only (vacuum or implicit solvent):
strip explicit water and ions before building it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

try:
    import openmm
    from openmm import unit
except ImportError as exc:  # pragma: no cover - exercised only without OpenMM
    raise ImportError(
        "pbsolve.openmm_bridge requires OpenMM: conda install -c conda-forge openmm"
    ) from exc

from ._core import Solver
from .mmpbsa import SITKOFF_BETA, SITKOFF_GAMMA, PBSAFrame, pbsa_solvation_frame

from .radii import BONDI, FALLBACK_RADIUS, PARSE, assign_radii  # noqa: E402  (BONDI etc. re-exported)

__all__ = [
    "BONDI",
    "FALLBACK_RADIUS",
    "PARSE",
    "OpenMMPBSA",
    "charges_from_system",
    "lennard_jones_from_system",
    "positions_angstrom",
    "radii_from_topology",
]

_WATER = {"HOH", "WAT", "TIP3", "TIP4", "TIP5", "SPC", "T3P", "T4P"}


def charges_from_system(system: "openmm.System") -> npt.NDArray[np.float64]:
    """Partial charges (e) from the system's single NonbondedForce."""
    forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(forces) != 1:
        raise ValueError(f"expected exactly one NonbondedForce, found {len(forces)}")
    nb = forces[0]
    return np.array(
        [nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(nb.getNumParticles())],
        dtype=np.float64,
    )


def lennard_jones_from_system(system: "openmm.System") -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Lennard-Jones radii and well depths for :func:`pbsolve.apolar_energy`.

    Parameters
    ----------
    system : openmm.System
        A system with exactly one NonbondedForce.

    Returns
    -------
    radii : numpy.ndarray, shape (N,)
        R_min/2 = 2^(1/6) σ/2 in Å. Atoms with ε = 0 get radius 0: force fields
        give them a placeholder σ, and APBS' AMBER parameters have radius 0
        there too.
    epsilons : numpy.ndarray, shape (N,)
        Well depths in kcal/mol.
    """
    forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(forces) != 1:
        raise ValueError(f"expected exactly one NonbondedForce, found {len(forces)}")
    nb = forces[0]
    params = [nb.getParticleParameters(i) for i in range(nb.getNumParticles())]
    sigma = np.array([p[1].value_in_unit(unit.angstrom) for p in params])
    eps = np.array([p[2].value_in_unit(unit.kilocalorie_per_mole) for p in params])
    radii = np.where(eps > 0.0, 2.0 ** (1.0 / 6.0) * sigma / 2.0, 0.0)
    return radii, eps


def radii_from_topology(topology: "openmm.app.Topology", scheme: str = "mbondi2") -> npt.NDArray[np.float64]:
    """Dielectric radii (Å) from elements and bonds.

    Schemes: ``bondi``, ``mbondi`` (H on C/N 1.3, on O/S 0.8), ``mbondi2``
    (H on N 1.3, other H 1.2) and ``parse``. Particles without an element
    (virtual sites) get radius 0 and do not shape the dielectric boundary.
    """
    atoms = list(topology.atoms())
    elements = [a.element.symbol if a.element is not None else "" for a in atoms]
    bonds = [(a.index, b.index) for a, b in topology.bonds()]
    return assign_radii(elements, scheme, bonds if bonds else None)


def positions_angstrom(positions: Union["unit.Quantity", npt.ArrayLike]) -> npt.NDArray[np.float64]:
    """OpenMM positions (Quantity, any length unit) or a plain Å array -> (N, 3) Å."""
    if hasattr(positions, "value_in_unit"):
        positions = positions.value_in_unit(unit.angstrom)
    return np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3))


class OpenMMPBSA:
    """MM-PBSA solvation terms for an OpenMM complex (single-trajectory protocol)."""

    def __init__(
        self,
        system: "openmm.System",
        topology: "openmm.app.Topology",
        ligand_atoms: Sequence[int],
        receptor_atoms: Optional[Sequence[int]] = None,
        solver: Optional[Solver] = None,
        radii: Union[str, npt.ArrayLike] = "mbondi2",
        gamma: float = SITKOFF_GAMMA,
        beta: float = SITKOFF_BETA,
        probe: float = 1.4,
    ) -> None:
        if any(res.name.upper() in _WATER for res in topology.residues()):
            raise ValueError("explicit water found in the topology; strip solvent before MM-PBSA")
        self.charges = charges_from_system(system)
        self.radii = (
            radii_from_topology(topology, radii)
            if isinstance(radii, str)
            else np.ascontiguousarray(radii, dtype=np.float64)
        )
        n = topology.getNumAtoms()
        if self.charges.size != n or self.radii.size != n:
            raise ValueError(f"topology has {n} atoms but got {self.charges.size} charges / {self.radii.size} radii")
        self.ligand_atoms = np.asarray(ligand_atoms, dtype=np.int64)
        self.receptor_atoms = None if receptor_atoms is None else np.asarray(receptor_atoms, dtype=np.int64)
        self.solver = solver if solver is not None else Solver()
        self.gamma, self.beta, self.probe = gamma, beta, probe

    def frame(self, positions: Union["unit.Quantity", npt.ArrayLike]) -> PBSAFrame:
        """Solvation terms of one frame (OpenMM positions in any length unit, or a plain array in Å)."""
        return pbsa_solvation_frame(
            self.solver,
            positions_angstrom(positions),
            self.charges,
            self.radii,
            self.ligand_atoms,
            self.receptor_atoms,
            gamma=self.gamma,
            beta=self.beta,
            probe=self.probe,
        )

    def trajectory(self, frames: Iterable[Union["unit.Quantity", npt.ArrayLike]]) -> list[PBSAFrame]:
        """Solvation terms of every frame, one :meth:`frame` call each."""
        return [self.frame(p) for p in frames]
