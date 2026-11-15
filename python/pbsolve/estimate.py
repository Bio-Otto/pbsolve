"""Numerical error estimates of ΔG_polar.

With a sharp dielectric boundary, ΔG_polar depends on where the molecular
surface falls between the grid nodes: moving the grid by a fraction of a
spacing changes the energy of a protein by a few tenths of a percent at
0.5 Å. :func:`estimate_error` measures this by solving on several placements
of the grid and, given two spacings, extrapolates the mean to zero spacing.
The discretisation is first order (observed orders 1.0–1.2 on the analytic
references in ``docs/VALIDATION.md``), so the extrapolation assumes order 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._core import GridSpec, Solver

__all__ = ["ErrorEstimate", "estimate_error"]


@dataclass
class ErrorEstimate:
    """ΔG_polar (or ΔΔG of association) with its numerical uncertainty, kcal/mol."""

    energy: float
    """Mean over the placements at the finest spacing."""
    sd: float
    """Standard deviation over those placements: the spread of a single calculation."""
    sem: float
    """Standard error of :attr:`energy`, ``sd / √placements``."""
    energies: list[float]
    """One value per placement at the finest spacing; the first is the unshifted grid."""
    spacings: list[float]
    """Fine-grid spacing of each set of placements (mean over the axes), coarsest first."""
    means: list[float]
    """Mean over the placements at each spacing."""
    offsets: list[list[float]]
    """How far each placement at the finest spacing moved the automatic grid's
    centre, in Å (at most half a spacing per axis; the first is zero)."""
    extrapolated: Optional[float] = None
    """The last two means extrapolated to zero spacing at first order (two spacings or more)."""

    @property
    def placements(self) -> int:
        """Number of placements per spacing."""
        return len(self.energies)

    @property
    def discretisation_error(self) -> Optional[float]:
        """:attr:`energy` − :attr:`extrapolated`, the estimated error of the mean at the finest spacing."""
        return None if self.extrapolated is None else self.energy - self.extrapolated

    def format(self) -> str:
        """One line of text."""
        text = (
            f"{self.energy:.3f} ± {self.sem:.3f} kcal/mol (mean ± sem of {self.placements} placements; "
            f"sd {self.sd:.3f}) at h = {self.spacings[-1]:.3f} Å"
        )
        if self.extrapolated is not None:
            text += f"; extrapolated to h = 0: {self.extrapolated:.3f} (error of the mean {self.discretisation_error:+.3f})"
        return text


def _with_spacing(solver: Solver, spacing: float) -> Solver:
    params = dict(solver.params)
    params.pop("ionic_strength")  # the ions are listed in params["ions"]
    params["grid_spacing"] = spacing
    return Solver(platform=solver.platform, device=solver.device, **params)


def estimate_error(
    solver: Solver,
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    radii: npt.ArrayLike,
    *,
    placements: int = 8,
    spacings: Optional[Sequence[float]] = None,
    parts: Optional[Sequence[npt.ArrayLike]] = None,
    seed: int = 0,
    workers: Optional[int] = None,
    dielectrics: Optional[npt.ArrayLike] = None,
) -> ErrorEstimate:
    """ΔG_polar averaged over grid placements, with its spread and, given two
    spacings, its extrapolation to zero spacing.

    The automatic grid of each spacing is moved by random fractions of its
    fine spacing along every axis (the first placement is the grid itself),
    and each placement is solved with :meth:`Solver.polar_solvation_many`.
    The cost is ``placements`` calculations per spacing.

    Parameters
    ----------
    solver : Solver
        The settings; its ``grid_spacing`` is used unless ``spacings`` is given.
    coords, charges, radii : array_like
        The molecule, as for :meth:`Solver.polar_solvation`.
    placements : int
        Grid placements per spacing, at least 2.
    spacings : sequence of float, optional
        Fine-grid spacings in Å. With two or more, the means at the two finest
        are extrapolated to zero spacing at first order.
    parts : sequence of array_like, optional
        Boolean masks or index arrays of atoms (e.g. receptor and ligand).
        The quantity becomes ΔΔG = ΔG(all atoms) − Σ ΔG(part), all on the
        grid of the whole molecule, as in MM-PBSA.
    seed : int
        Seed of the random placements; the same seed gives the same result.
    workers : int, optional
        Calculations solved concurrently (as for :meth:`Solver.polar_solvation_many`).
    dielectrics : array_like, optional
        Interior dielectric of every atom (as for :meth:`Solver.polar_solvation`).

    Returns
    -------
    ErrorEstimate

    Raises
    ------
    ValueError
        If ``placements < 2`` or the arrays do not match.
    """
    if placements < 2:
        raise ValueError("placements must be at least 2")
    coords = np.asarray(coords, dtype=float)
    charges = np.asarray(charges, dtype=float)
    radii = np.asarray(radii, dtype=float)
    selections = [np.arange(len(charges))]
    for p in parts or ():
        sel = np.asarray(p)
        if sel.dtype == bool:
            if sel.shape != charges.shape:
                raise ValueError("a mask in parts must have one entry per atom")
            sel = np.flatnonzero(sel)
        sel = sel.astype(int)
        if sel.size and (sel.min() < 0 or sel.max() >= len(charges)):
            raise ValueError("indices in parts must refer to the atoms")
        selections.append(sel)
    hs = sorted({float(h) for h in spacings}, reverse=True) if spacings else [None]
    rng = np.random.default_rng(seed)
    shifts = [np.zeros(3)] + [rng.uniform(-0.5, 0.5, 3) for _ in range(placements - 1)]
    used, values, offsets = [], [], []
    for h in hs:
        s = solver if h is None else _with_spacing(solver, h)
        base = s.make_grid(coords, radii)
        step = np.asarray(base.fine_spacing)
        offsets = [list(f * step) for f in shifts]
        grids = [
            GridSpec(list(np.asarray(base.center) + d), base.dims, base.coarse_length, base.fine_length)
            for d in offsets
        ]
        systems = [(coords[i], charges[i], radii[i]) for i in selections]
        eps = None if dielectrics is None else [np.asarray(dielectrics, dtype=float)[i] for i in selections]
        results = s.polar_solvation_many(
            [x for _ in grids for x in systems],
            grid=[g for g in grids for _ in systems],
            workers=workers,
            dielectrics=None if eps is None else [e for _ in grids for e in eps],
        )
        e = [r.energy for r in results]
        k = len(systems)
        values.append([e[n * k] - sum(e[n * k + 1 : (n + 1) * k]) for n in range(len(grids))])
        used.append(float(np.mean(step)))
    last = np.asarray(values[-1])
    means = [float(np.mean(v)) for v in values]
    extrapolated = None
    if len(means) >= 2:
        (h1, m1), (h2, m2) = zip(used[-2:], means[-2:])
        extrapolated = m2 + (m2 - m1) * h2 / (h1 - h2)
    sd = float(np.std(last, ddof=1))
    return ErrorEstimate(
        energy=means[-1],
        sd=sd,
        sem=sd / math.sqrt(len(last)),
        energies=[float(v) for v in last],
        spacings=used,
        means=means,
        offsets=offsets,
        extrapolated=extrapolated,
    )
