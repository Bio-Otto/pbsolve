"""Analytic reference energies for testing discretisations.

:func:`kirkwood_energy` is the polar solvation energy of point charges
inside a dielectric sphere [Kirkwood 1934], :func:`two_sphere_energy` the
same for two spheres (a converged multipole solution, semi-analytic). Both
describe the model pbsolve solves for uncharged spheres with zero-radius
charges inside them: interior ``pdie``, exterior ``sdie``, and ions beyond
the Stern radius ``radius + ion_radius`` (pbsolve's ion-accessibility rule).
The reference state is ``pdie`` everywhere without ions, as in
:meth:`pbsolve.Solver.polar_solvation`, and the physical constants are
pbsolve's (CODATA 2018).

Kirkwood JG. Theory of solutions of molecules containing widely separated
charges with special application to zwitterions. J Chem Phys 2, 351 (1934).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._core import bjerrum_length, kt_kcal_per_mol

__all__ = ["kirkwood_energy", "two_sphere_energy"]

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


def _kappa(ionic_strength: float, sdie: float, temperature: float) -> float:
    return math.sqrt(8.0 * math.pi * bjerrum_length(temperature) * _AVOGADRO * 1e-27 * ionic_strength / sdie)


def _legendre(n_max: int, m: int, x: np.ndarray) -> np.ndarray:
    """Orthonormal associated Legendre functions P̄ₗᵐ(x) on [−1, 1], l = m … n_max."""
    sin = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    p = np.full_like(x, math.sqrt(0.5))
    for k in range(1, m + 1):
        p = -math.sqrt((2 * k + 1) / (2 * k)) * sin * p
    out = np.empty((n_max - m + 1,) + x.shape)
    out[0] = p
    if n_max > m:
        out[1] = math.sqrt(2 * m + 3) * x * p
    for l in range(m + 2, n_max + 1):
        a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
        b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
        out[l - m] = a * (x * out[l - m - 1] - b * out[l - m - 2])
    return out


def _log_k(n_max: int, x: np.ndarray) -> np.ndarray:
    """log kₙ(x) for n = 0 … n_max (up to a constant), from the ratios k_{n-1}/kₙ."""
    out = np.empty((n_max + 1,) + np.shape(x))
    out[0] = -x - np.log(x)
    ratio = x / (1.0 + x)
    for n in range(1, n_max + 1):
        out[n] = out[n - 1] - np.log(ratio)
        ratio = 1.0 / (ratio + (2 * n + 1) / x)
    return out


def _log_derivatives(n_max: int, x: float) -> tuple[np.ndarray, np.ndarray]:
    """x·iₙ'(x)/iₙ(x) and x·kₙ'(x)/kₙ(x), n = 0 … n_max."""
    n = np.arange(n_max + 1)
    ti = np.empty(n_max + 1)  # i_{n+1}/i_n by backward recurrence (Miller)
    t = 0.0
    for k in range(n_max + 40 + int(x), -1, -1):
        t = 1.0 / ((2 * k + 3) / x + t)
        if k <= n_max:
            ti[k] = t
    tk = np.empty(n_max + 1)  # k_{n-1}/k_n
    ratio = x / (1.0 + x)
    tk[0] = 0.0
    for k in range(1, n_max + 1):
        tk[k] = ratio
        ratio = 1.0 / (ratio + (2 * k + 1) / x)
    dk = np.where(n == 0, -1.0 - x, -x * tk - (n + 1))
    return x * ti + n, dk


def _radial(n_max, pdie, sdie, stern, lam_in, lam_out):
    """Response of one sphere per order n, in u = r/a with the ion-free shell 1 < u < stern.

    Inside: (M/pdie) u^-(n+1) + B uⁿ; shell: C uⁿ + D u^-(n+1); outside: F R + E S with the
    incoming and outgoing radial functions R(stern) = S(stern) = 1 and logarithmic
    derivatives ``lam_in``, ``lam_out`` (d ln/du). Returns E and B for (M, F) = (1, 0) and (0, 1).
    """
    out = np.empty((n_max + 1, 2, 2))
    for n in range(n_max + 1):
        g, d = stern ** -n, stern ** -(n + 1)  # C = C'·g with C' = C·sternⁿ
        m = np.array(
            [
                [1.0, -g, -1.0, 0.0],
                [n * pdie, -n * sdie * g, (n + 1) * sdie, 0.0],
                [0.0, 1.0, d, -1.0],
                [0.0, n / stern, -(n + 1) * d / stern, -lam_out[n]],
            ]
        )
        rhs = np.array([[-1.0 / pdie, 0.0], [n + 1.0, 0.0], [0.0, 1.0], [0.0, lam_in[n]]])
        sol = np.linalg.solve(m, rhs)  # rows B, C', D, E; columns M, F
        out[n] = sol[[3, 0]]
    return out  # [n, (E, B), (M, F)]


def two_sphere_energy(
    coords: npt.ArrayLike,
    charges: npt.ArrayLike,
    centers: npt.ArrayLike,
    radii: npt.ArrayLike,
    *,
    pdie: float = 1.0,
    sdie: float = 78.54,
    ionic_strength: float = 0.0,
    ion_radius: float = 2.0,
    temperature: float = 298.15,
    order: int | None = None,
    sphere_pdie: Sequence[float] | None = None,
) -> float:
    """Polar solvation energy of charges inside two dielectric spheres, kcal/mol.

    The model of :func:`kirkwood_energy` with two spheres. The multipole
    expansions of both spheres, coupled through re-expansions along the line
    between the centres, are solved per azimuthal order and truncated at
    ``order``. With the spheres far apart the energy approaches the sum of
    the two Kirkwood energies.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Charge positions in Å; each must lie strictly inside one sphere.
    charges : array_like, shape (N,)
        Charges in e.
    centers : array_like, shape (2, 3)
        Sphere centres in Å.
    radii : array_like, shape (2,)
        Sphere radii in Å (the dielectric boundaries).
    pdie, sdie, ionic_strength, ion_radius, temperature
        As for :func:`kirkwood_energy`. The spheres must not overlap, nor,
        with salt, their ion-free shells.
    order : int, optional
        Highest multipole order; by default chosen for a relative accuracy of
        about 1e-12 from the geometry (at most 150).
    sphere_pdie : sequence of 2 float, optional
        A dielectric for the inside of each sphere instead of ``pdie``; the
        reference state then keeps them and has ``pdie`` outside the spheres
        (as ``Solver.polar_solvation`` with ``dielectrics``).

    Returns
    -------
    float
        ½ Σᵢ qᵢ (φ_solvated − φ_reference)(rᵢ) in kcal/mol.

    Raises
    ------
    ValueError
        If a charge is in neither sphere or the spheres (shells) overlap.
    """
    pos = np.asarray(coords, dtype=float).reshape(-1, 3)
    q = np.asarray(charges, dtype=float).reshape(-1)
    c = np.asarray(centers, dtype=float).reshape(2, 3)
    a = np.asarray(radii, dtype=float).reshape(2)
    if len(q) != len(pos):
        raise ValueError("coords and charges differ in length")
    kappa = _kappa(ionic_strength, sdie, temperature)
    b = a + ion_radius if kappa > 0.0 else a.copy()
    axis = c[1] - c[0]
    dist = float(np.linalg.norm(axis))
    if dist <= b.sum():
        raise ValueError("the spheres (with salt: their ion-free shells) overlap")
    owner = np.full(len(q), -1)
    rel = np.empty_like(pos)
    for i in range(2):
        r = pos - c[i]
        inside = np.linalg.norm(r, axis=1) < a[i]
        owner[inside], rel[inside] = i, r[inside]
    if np.any(owner < 0):
        raise ValueError("every charge must lie strictly inside one of the spheres")

    # A frame with z along c₁ − c₀; both expansions share it, so a translation keeps m.
    ez = axis / dist
    ex = np.cross(ez, [1.0, 0.0, 0.0] if abs(ez[0]) < 0.9 else [0.0, 1.0, 0.0])
    ex /= np.linalg.norm(ex)
    local = rel @ np.column_stack([ex, np.cross(ez, ex), ez])
    rho = np.linalg.norm(local, axis=1)
    cos_t = np.where(rho > 0.0, local[:, 2] / np.where(rho > 0.0, rho, 1.0), 1.0)
    phi = np.arctan2(local[:, 1], local[:, 0])
    rho_a = rho / a[owner]

    if order is None:
        ratio = max([*(rho_a[owner == i].max(initial=0.0) for i in range(2)), b[0] / (dist - b[1]), b[1] / (dist - b[0])])
        order = 8 if ratio <= 0.0 else min(150, max(8, math.ceil(math.log(1e-12) / math.log(ratio))))
    n_max = order
    stern = b / a
    lb = bjerrum_length(temperature)
    if sphere_pdie is None:
        potential = _two_sphere_potential(q, owner, rho_a, cos_t, phi, a, b, stern, dist, n_max, (pdie, pdie), sdie, kappa)
        # minus the reference state's Coulomb potential of the other sphere's charges
        cross = owner[:, None] != owner[None, :]
        rij = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        coulomb = np.where(cross, q[None, :] / (pdie * np.where(cross, rij, 1.0)), 0.0).sum(axis=1)
        return 0.5 * lb * float(q @ (potential - coulomb)) * kt_kcal_per_mol(temperature)
    eps_in = (float(sphere_pdie[0]), float(sphere_pdie[1]))
    solvated = _two_sphere_potential(q, owner, rho_a, cos_t, phi, a, b, stern, dist, n_max, eps_in, sdie, kappa)
    reference = _two_sphere_potential(q, owner, rho_a, cos_t, phi, a, b, stern, dist, n_max, eps_in, pdie, 0.0)
    return 0.5 * lb * float(q @ (solvated - reference)) * kt_kcal_per_mol(temperature)


def _two_sphere_potential(q, owner, rho_a, cos_t, phi, a, b, stern, dist, n_max, eps_in, eps_out, kappa):
    """Reaction plus transmitted potential at the charges (e/Å) of :func:`two_sphere_energy`:
    interiors ``eps_in``, the shells and outside ``eps_out`` with screening ``kappa`` beyond r = b."""
    resp = []
    for i in range(2):
        if kappa > 0.0:
            li, lk = _log_derivatives(n_max, kappa * b[i])
            lam_in, lam_out = li / stern[i], lk / stern[i]  # d ln/du at u = stern
        else:
            n = np.arange(n_max + 1)
            lam_in, lam_out = n / stern[i], -(n + 1) / stern[i]
        resp.append(_radial(n_max, eps_in[i], eps_out, stern[i], lam_in, lam_out))

    # Outgoing radial functions S of the other sphere at Gauss nodes on this sphere's r = b.
    x, w = np.polynomial.legendre.leggauss(2 * n_max + 40)
    geo = []
    for i, sign in ((0, 1.0), (1, -1.0)):  # field of sphere 1−i seen from sphere i
        j = 1 - i
        zr = b[i] * x - sign * dist  # z of the node relative to the other centre
        r_other = np.sqrt(b[i] ** 2 * (1.0 - x * x) + zr * zr)
        if kappa > 0.0:
            s_out = np.exp(_log_k(n_max, kappa * r_other) - _log_k(n_max, np.array([kappa * b[j]])))
        else:
            s_out = (b[j] / r_other)[None, :] ** (np.arange(n_max + 1)[:, None] + 1)
        geo.append((zr / r_other, s_out))

    potential = np.zeros(len(q))  # reaction + transmitted potential at the charges, e/Å
    for m in range(n_max + 1):
        k = n_max - m + 1
        ylm = _legendre(n_max, m, cos_t) * np.exp(1j * m * phi) / math.sqrt(2.0 * math.pi)  # Y_nm at the charges
        ns = np.arange(m, n_max + 1)
        moments, trans, er, br = [], [], [], []
        for i in range(2):
            sel = owner == i
            pw = rho_a[sel][None, :] ** ns[:, None]
            moments.append(4.0 * math.pi / ((2 * ns + 1) * a[i]) * ((pw * np.conj(ylm[:, sel])) @ q[sel]))
            cos_o, s_out = geo[i]
            g_nodes = s_out[m:] * _legendre(n_max, m, cos_o)  # [n, node]
            trans.append((_legendre(n_max, m, x) * w) @ g_nodes.T)  # T[l, n]
            er.append(resp[i][m:, 0])
            br.append(resp[i][m:, 1])
        # E_i = α_i M_i + γ_i T_i E_j
        mat = np.eye(2 * k, dtype=complex)
        mat[:k, k:] = -er[0][:, 1][:, None] * trans[0]
        mat[k:, :k] = -er[1][:, 1][:, None] * trans[1]
        rhs = np.concatenate([er[0][:, 0] * moments[0], er[1][:, 0] * moments[1]])
        e_out = np.linalg.solve(mat, rhs)
        for i in range(2):
            f_in = trans[i] @ (e_out[k:] if i == 0 else e_out[:k])
            coef = br[i][:, 0] * moments[i] + br[i][:, 1] * f_in
            sel = owner == i
            val = (coef[:, None] * rho_a[sel][None, :] ** ns[:, None] * ylm[:, sel]).sum(axis=0)
            potential[sel] += (1.0 if m == 0 else 2.0) * val.real
    return potential
