"""Poisson–Boltzmann pKa values and titration (ROADMAP 4.21).

The method of Bashford & Karplus (1990). Every titratable group (a *site*)
has a neutral and a charged state, which differ in the charges of a few atoms
by Δq_i (charged minus neutral). With q_0 the charges of the molecule with
every site neutral and G the electrostatic Green's function (the reaction
field R from the Poisson–Boltzmann equation plus Coulomb's law in ``pdie``),
the energy of a protonation microstate x (x_i = 1: site i charged) is

    E(q_0 + Σ x_i Δq_i) = E(q_0) + Σ_i x_i [Δq_iᵀ G q_0 + ½ Δq_iᵀ G Δq_i] + ½ Σ_{i≠j} x_i x_j W_ij,

W_ij = Δq_iᵀ G Δq_j. The same bracket, computed for the site's *model
compound* (by default its residue alone, in water), turns the model pKa into
the site's intrinsic pKa in the molecule:

    pKa_intr,i = pKa_model,i ± ΔG_i / (RT ln 10),    ΔG_i = [Δq_iᵀ G q_0 + ½ Δq_iᵀ G Δq_i]_molecule − [same]_model,

with + for acids (the charged state is the deprotonated one) and − for
bases. The free energy of microstate x at a pH is then

    G(x) = Σ_i x_i s_i RT ln10 (pKa_intr,i − pH) + ½ Σ_{i≠j} W_ij x_i x_j,    s_i = +1 (acid), −1 (base),

and the titration follows from its Boltzmann average over x: exactly for
up to :data:`EXACT_SITES` sites, by Metropolis Monte Carlo beyond.

The reaction-field terms come from :meth:`pbsolve.Solver.polar_gradient`:
its charge derivative for the charges Δq_i is R Δq_i at every atom (R
symmetrised, exact through focusing), so one call per site and environment
gives the self term, the background term and every W_ij. The molecule and
the model compounds share the molecule's grid, so discretisation errors
cancel in the difference. Coulomb's law adds the rest analytically; its
terms between atoms of a model compound cancel and are left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._core import GridSpec, Solver, bjerrum_length, kt_kcal_per_mol

__all__ = ["EXACT_SITES", "Site", "SiteEnergies", "Titration", "site_energies", "titrate"]

LN10 = float(np.log(10.0))

#: Up to this many sites :func:`titrate` sums over all 2^N microstates.
EXACT_SITES = 14


@dataclass(frozen=True)
class Site:
    """A titratable group with a neutral and a charged state."""

    #: Label, e.g. "ASP A 52".
    name: str
    #: Indices of the atoms whose charge differs between the states.
    atoms: Sequence[int]
    #: Their charges (e) in the neutral state.
    neutral: Sequence[float]
    #: Their charges (e) in the charged state.
    charged: Sequence[float]
    #: pKa of the model compound in water.
    model_pka: float
    #: True if the charged state is the deprotonated one (Asp, Glu, Tyr, Cys,
    #: C terminus), False if it is the protonated one (Lys, Arg, His, N terminus).
    acid: bool
    #: Atoms of the model compound (they must include ``atoms``); None: ``atoms`` alone.
    model_atoms: Optional[Sequence[int]] = None


@dataclass
class SiteEnergies:
    """The energies of the sites in a molecule (:func:`site_energies`), kcal/mol."""

    sites: list
    #: ΔΔG of the self term ½ Δqᵀ G Δq, molecule minus model compound.
    born: np.ndarray
    #: ΔΔG of the background term Δqᵀ G q_0 (all sites neutral), molecule minus model compound.
    background: np.ndarray
    #: W_ij between the charged states, symmetric, zero diagonal.
    interactions: np.ndarray
    #: pKa of every site with all others neutral.
    intrinsic_pka: np.ndarray
    temperature: float

    @property
    def signs(self) -> np.ndarray:
        """+1 for acids, −1 for bases."""
        return np.array([1.0 if s.acid else -1.0 for s in self.sites])


@dataclass
class Titration:
    """Protonation of every site over a range of pH (:func:`titrate`)."""

    ph: np.ndarray
    #: Fraction of each site protonated, shape (pH values, sites).
    protonated: np.ndarray
    #: pH where each site is half protonated (NaN if it does not cross 0.5 in the range).
    pka: np.ndarray
    #: Net charge of the sites (e), relative to all of them neutral, per pH.
    charge: np.ndarray
    method: str
    #: Monte Carlo only: standard error of each fraction (block averages).
    error: Optional[np.ndarray] = field(default=None)


def _unit_coulomb(a: np.ndarray, b: np.ndarray, qa: np.ndarray, qb: np.ndarray) -> float:
    """Σ_{a,b} q_a q_b / r_ab (e²/Å) over two disjoint sets of atoms."""
    if len(qa) == 0 or len(qb) == 0:
        return 0.0
    r = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(np.sum(np.outer(qa, qb) / r))


def site_energies(
    solver: Solver,
    coords,
    charges,
    radii,
    sites: Sequence[Site],
    grid: Optional[GridSpec] = None,
) -> SiteEnergies:
    """Intrinsic pKa values and interactions of titratable sites.

    Parameters
    ----------
    solver : Solver
        The Poisson–Boltzmann options (``pdie`` is also the dielectric of
        Coulomb's law between the charges).
    coords : array_like, shape (N, 3)
        Atom positions in Å.
    charges : array_like, shape (N,)
        Charges with every site in its neutral state.
    radii : array_like, shape (N,)
        Atomic radii in Å; the dielectric boundary is the same in every
        protonation state.
    sites : sequence of Site
    grid : GridSpec, optional
        The grid of every calculation (molecule and model compounds); by
        default the solver's automatic grid of the molecule.

    Returns
    -------
    SiteEnergies
    """
    x = np.asarray(coords, dtype=np.float64)
    q0 = np.asarray(charges, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    n = len(sites)
    if grid is None:
        grid = solver.make_grid(x, r)
    params = solver.params
    temperature = params["temperature"]
    coulomb = bjerrum_length(temperature) * kt_kcal_per_mol(temperature) / params["pdie"]
    owner = np.full(len(q0), -1)
    for k, s in enumerate(sites):
        _check(s, len(q0))
        if (owner[list(s.atoms)] >= 0).any():
            raise ValueError(f"site {s.name} shares atoms with site {sites[owner[list(s.atoms)].max()].name}")
        owner[list(s.atoms)] = k
        neutral = np.asarray(s.neutral, dtype=np.float64)
        if not np.allclose(q0[list(s.atoms)], neutral, rtol=0.0, atol=1e-12):
            raise ValueError(f"site {s.name}: `charges` must hold its neutral state")
    delta = [np.asarray(s.charged, dtype=np.float64) - np.asarray(s.neutral, dtype=np.float64) for s in sites]
    born = np.zeros(n)
    background = np.zeros(n)
    w = np.zeros((n, n))
    for i, s in enumerate(sites):
        atoms = np.asarray(s.atoms)
        dq = np.zeros(len(q0))
        dq[atoms] = delta[i]
        # The reaction field of Δq_i at every atom of the molecule (kcal/mol/e).
        phi = solver.polar_gradient(x, dq, r, grid=grid).charges
        model = np.asarray(s.model_atoms if s.model_atoms is not None else s.atoms)
        phi_model = solver.polar_gradient(x[model], dq[model], r[model], grid=grid).charges
        born[i] = 0.5 * (dq @ phi - dq[model] @ phi_model)
        outside = np.setdiff1d(np.arange(len(q0)), model)
        background[i] = (q0 @ phi - q0[model] @ phi_model) + coulomb * _unit_coulomb(
            x[atoms], x[outside], delta[i], q0[outside]
        )
        for j, t in enumerate(sites):
            if j != i:
                other = np.asarray(t.atoms)
                w[i, j] = delta[j] @ phi[other] + coulomb * _unit_coulomb(x[atoms], x[other], delta[i], delta[j])
    # R is symmetric up to the solver tolerance; average the two halves.
    w = 0.5 * (w + w.T)
    kt = kt_kcal_per_mol(temperature)
    signs = np.array([1.0 if s.acid else -1.0 for s in sites])
    intrinsic = np.array([s.model_pka for s in sites]) + signs * (born + background) / (kt * LN10)
    return SiteEnergies(list(sites), born, background, w, intrinsic, temperature)


def _check(site: Site, n_atoms: int) -> None:
    atoms = list(site.atoms)
    if not atoms or len(set(atoms)) != len(atoms):
        raise ValueError(f"site {site.name}: `atoms` must be distinct and not empty")
    if len(site.neutral) != len(atoms) or len(site.charged) != len(atoms):
        raise ValueError(f"site {site.name}: `neutral` and `charged` need one charge per atom")
    if min(atoms) < 0 or max(atoms) >= n_atoms:
        raise ValueError(f"site {site.name}: atom index out of range")
    if site.model_atoms is not None and not set(atoms) <= set(site.model_atoms):
        raise ValueError(f"site {site.name}: `model_atoms` must include `atoms`")


def _free_energies(energies: SiteEnergies, states: np.ndarray, ph: float) -> np.ndarray:
    """G(x) in kcal/mol for microstates x (rows of 0/1), up to a constant."""
    kt = kt_kcal_per_mol(energies.temperature)
    h = energies.signs * kt * LN10 * (energies.intrinsic_pka - ph)
    return states @ h + 0.5 * np.einsum("ki,ij,kj->k", states, energies.interactions, states)


def titrate(
    energies: SiteEnergies,
    ph: Sequence[float] = tuple(np.round(np.arange(0.0, 14.01, 0.1), 10)),
    method: str = "auto",
    sweeps: int = 20000,
    equilibration: int = 2000,
    seed: int = 0,
) -> Titration:
    """Protonation of the sites as a function of pH.

    Parameters
    ----------
    energies : SiteEnergies
        From :func:`site_energies`.
    ph : sequence of float, optional
        pH values (default 0–14 in steps of 0.1).
    method : {"auto", "exact", "mc"}
        "exact" sums over all 2^N microstates (N ≤ 20); "mc" samples them by
        Metropolis Monte Carlo (single-site flips and, for pairs that
        interact by more than 2 pK units, pair flips); "auto" is "exact" up to
        :data:`EXACT_SITES` sites.
    sweeps, equilibration : int, optional
        Monte Carlo sweeps (every site once per sweep) averaged and discarded.
    seed : int, optional
        Random seed (the result is reproducible).

    Returns
    -------
    Titration
    """
    ph = np.asarray(ph, dtype=np.float64)
    n = len(energies.sites)
    if method == "auto":
        method = "exact" if n <= EXACT_SITES else "mc"
    acid = np.array([s.acid for s in energies.sites])
    error = None
    if method == "exact":
        if n > 20:
            raise ValueError("method 'exact' is limited to 20 sites")
        states = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        kt = kt_kcal_per_mol(energies.temperature)
        charged = np.empty((len(ph), n))
        for k, p in enumerate(ph):
            g = _free_energies(energies, states, p)
            weights = np.exp(-(g - g.min()) / kt)
            charged[k] = weights @ states / weights.sum()
    elif method == "mc":
        charged, error = _monte_carlo(energies, ph, sweeps, equilibration, seed)
    else:
        raise ValueError(f"unknown method {method!r} (expected 'auto', 'exact' or 'mc')")
    protonated = np.where(acid[None, :], 1.0 - charged, charged)
    signs = np.where(acid, -1.0, 1.0)
    return Titration(ph, protonated, _half_points(ph, protonated), charged @ signs, method, error)


def _half_points(ph: np.ndarray, protonated: np.ndarray) -> np.ndarray:
    """The pH where each site's protonation crosses 0.5 (first crossing, linear interpolation)."""
    out = np.full(protonated.shape[1], np.nan)
    for i in range(protonated.shape[1]):
        f = protonated[:, i] - 0.5
        cross = np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)
        cross = [c for c in cross if f[c] != f[c + 1]]
        if cross:
            c = cross[0]
            out[i] = ph[c] + (ph[c + 1] - ph[c]) * f[c] / (f[c] - f[c + 1])
    return out


def _monte_carlo(energies: SiteEnergies, ph: np.ndarray, sweeps: int, equilibration: int, seed: int):
    """Metropolis sampling, one chain per pH value (vectorised over the chains)."""
    rng = np.random.default_rng(seed)
    kt = kt_kcal_per_mol(energies.temperature)
    n = len(energies.sites)
    w = energies.interactions
    h = energies.signs[None, :] * kt * LN10 * (energies.intrinsic_pka[None, :] - ph[:, None])
    chains = len(ph)
    x = (h < 0).astype(np.float64)  # start from the lower single-site energy
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(w[i, j]) > 2.0 * kt * LN10]
    total = np.zeros((chains, n))
    blocks = 20
    block_sums = np.zeros((blocks, chains, n))
    per_block = max(1, sweeps // blocks)
    for sweep in range(equilibration + sweeps):
        for k in rng.permutation(n):
            field_k = h[:, k] + x @ w[:, k]
            de = (1.0 - 2.0 * x[:, k]) * field_k
            accept = rng.random(chains) < np.exp(-np.clip(de, 0.0, None) / kt)
            x[accept, k] = 1.0 - x[accept, k]
        for i, j in pairs:
            fi = h[:, i] + x @ w[:, i] - x[:, j] * w[j, i]
            fj = h[:, j] + x @ w[:, j] - x[:, i] * w[i, j]
            di, dj = 1.0 - 2.0 * x[:, i], 1.0 - 2.0 * x[:, j]
            de = di * fi + dj * fj + w[i, j] * ((1.0 - x[:, i]) * (1.0 - x[:, j]) - x[:, i] * x[:, j])
            accept = rng.random(chains) < np.exp(-np.clip(de, 0.0, None) / kt)
            x[accept, i] = 1.0 - x[accept, i]
            x[accept, j] = 1.0 - x[accept, j]
        if sweep >= equilibration:
            s = sweep - equilibration
            total += x
            if s // per_block < blocks:
                block_sums[s // per_block] += x
    mean = total / sweeps
    used = min(blocks, sweeps // per_block)
    block_means = block_sums[:used] / per_block
    error = block_means.std(axis=0, ddof=1) / np.sqrt(used) if used > 1 else None
    return mean, error
