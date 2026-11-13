"""Analytic reference energies for testing discretisations.

:func:`kirkwood_energy` is the polar solvation energy of point charges
inside a dielectric sphere [Kirkwood 1934], the model pbsolve solves for one
sphere with zero-radius charges inside it: interior ``pdie``, exterior
``sdie``, and ions beyond the Stern radius ``radius + ion_radius`` (pbsolve's
ion-accessibility rule). The reference state is ``pdie`` everywhere without
ions, as in :meth:`pbsolve.Solver.polar_solvation`, and the physical
constants are pbsolve's (CODATA 2018).

Kirkwood JG. Theory of solutions of molecules containing widely separated
charges with special application to zwitterions. J Chem Phys 2, 351 (1934).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ._core import bjerrum_length, kt_kcal_per_mol

__all__ = ["kirkwood_energy"]

_AVOGADRO = 6.02214076e23


def _reaction_coefficients(n_max: int, pdie: float, sdie: float, kappa_a: float, stern: float) -> np.ndarray:
    """β_n of the reaction field Σ β_n (r ρ/a²)ⁿ Pₙ(cos γ) / a, in units of q/(4πε₀)."""
    n = np.arange(n_max + 1, dtype=float)
    if kappa_a > 0.0:
        # a·d ln kₙ(κr)/dr at r = b, from k_{n-1}/k_n by forward recurrence
        # (kₙ itself overflows for large n; its ratios do not).
        x = kappa_a * stern
        log_d = np.empty(n_max + 1)
        log_d[0] = -1.0 - 1.0 / x
        ratio = x / (1.0 + x)  # k_0/k_1
        for k in range(1, n_max + 1):
            log_d[k] = -ratio - (k + 1) / x
            ratio = 1.0 / (ratio + (2 * k + 1) / x)
        robin = kappa_a * log_d
    else:
        robin = -(n + 1) / stern
    # Between the sphere and the Stern radius φ = D (δ uⁿ + u⁻ⁿ⁻¹), u = r/a.
    delta = stern ** -(2 * n + 1) * (n + 1 + robin * stern) / (n - robin * stern)
    t = sdie * (n * delta - (n + 1)) / (delta + 1.0)
    return ((n + 1) + t / pdie) / (n * pdie - t)


def kirkwood_energy(
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    radius: float,
    *,
    center: npt.ArrayLike = (0.0, 0.0, 0.0),
    pdie: float = 1.0,
    sdie: float = 78.54,
    ionic_strength: float = 0.0,
    ion_radius: float = 2.0,
    temperature: float = 298.15,
    tol: float = 1e-14,
) -> float:
    """Polar solvation energy of charges inside a sphere, kcal/mol [Kirkwood 1934].

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Charge positions in Å, strictly inside the sphere.
    charges : array_like, shape (N,)
        Charges in e.
    radius : float
        Sphere radius a in Å (the dielectric boundary).
    center : array_like, shape (3,)
        Sphere centre in Å.
    pdie, sdie : float
        Dielectric constants inside and outside.
    ionic_strength : float
        1:1 salt in mol/L, beyond the Stern radius ``radius + ion_radius``.
    ion_radius : float
        Ion radius in Å.
    temperature : float
        Temperature in K.
    tol : float
        Relative truncation of the multipole series.

    Returns
    -------
    float
        ½ Σᵢⱼ qᵢ qⱼ φ_RF(rᵢ, rⱼ) in kcal/mol.

    Raises
    ------
    ValueError
        If a charge is not strictly inside the sphere.
    """
    pos = np.asarray(coords, dtype=float).reshape(-1, 3) - np.asarray(center, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    if len(q) != len(pos):
        raise ValueError("coords and charges differ in length")
    rho = np.linalg.norm(pos, axis=1) / radius
    if rho.size and rho.max() >= 1.0:
        raise ValueError("every charge must lie strictly inside the sphere")
    lb = bjerrum_length(temperature)
    kappa = math.sqrt(8.0 * math.pi * lb * _AVOGADRO * 1e-27 * ionic_strength / sdie)
    # Terms fall off as (ρᵢρⱼ)ⁿ with |β_n| bounded, so this order meets ``tol``.
    rmax = rho.max() if rho.size else 0.0
    n_max = 0 if rmax == 0.0 else max(1, math.ceil(math.log(tol) / (2.0 * math.log(rmax))))
    beta = _reaction_coefficients(n_max, pdie, sdie, kappa * radius, 1.0 + ion_radius / radius)

    unit = pos / np.where(rho > 0.0, rho * radius, 1.0)[:, None]
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    qq = np.outer(q, q)
    rr = np.outer(rho, rho)
    p_prev, p = np.ones_like(cos), cos  # Legendre P_0, P_1
    total = beta[0] * qq.sum()
    power = np.ones_like(rr)
    for n in range(1, n_max + 1):
        power = power * rr
        total += beta[n] * np.sum(qq * power * p)
        p_prev, p = p, ((2 * n + 1) * cos * p - n * p_prev) / (n + 1)
    return 0.5 * lb / radius * total * kt_kcal_per_mol(temperature)
