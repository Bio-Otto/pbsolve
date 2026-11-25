"""ΔG_polar as a differentiable JAX function (ROADMAP 4.20).

:func:`polar_energy` returns ΔG_polar (kcal/mol) with a custom
vector–Jacobian product from :meth:`pbsolve.Solver.polar_gradient`: the exact
derivatives of the discrete energy with respect to the charges, the
positions, the radii and the dielectric constants, through every focusing
level and the boundary condition. The solver runs on the host through
``jax.pure_callback``, so the function works under ``jax.jit``, ``jax.grad``
and ``jax.vmap`` (one solve per element). JAX is an optional dependency,
imported only here; enable ``jax_enable_x64`` for double precision.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._core import GridSpec, Solver

__all__ = ["polar_energy"]

SPLINES = ("spl2", "spl4")
SURFACES_WITH_DIELECTRIC_DERIVATIVES = ("mol", "vdw", "spl2", "spl4")


def _solver(solver: Solver, pdie: float, sdie: float) -> Solver:
    """`solver` with these dielectric constants (a copy only if they differ)."""
    p = solver.params
    if pdie == p["pdie"] and sdie == p["sdie"]:
        return solver
    return Solver(platform=solver.platform, device=solver.device, **{**p, "pdie": pdie, "sdie": sdie})


def polar_energy(
    solver: Solver,
    coords,
    charges,
    radii,
    pdie=None,
    sdie=None,
    grid: Optional[GridSpec] = None,
):
    """ΔG_polar in kcal/mol, differentiable with respect to every argument.

    The derivatives with respect to ``charges`` are always available, those
    with respect to ``coords`` and ``radii`` with a spline surface
    (``surface`` "spl2" or "spl4"), those with respect to ``pdie`` and
    ``sdie`` with ``surface`` "mol", "vdw", "spl2" or "spl4" (see
    :meth:`pbsolve.Solver.polar_gradient`). The others are NaN: JAX does not
    tell the function which derivatives will be used, so it computes the
    available ones in the forward pass.

    Parameters
    ----------
    solver : Solver
        The calculation's options.
    coords : array_like, shape (N, 3)
        Atom positions in Å.
    charges : array_like, shape (N,)
        Atomic charges in e.
    radii : array_like, shape (N,)
        Atomic radii in Å.
    pdie, sdie : scalar, optional
        Dielectric constants in place of the solver's.
    grid : GridSpec, optional
        The grid; by default the solver's automatic grid of the atoms, which
        moves with them. Fix it (``solver.make_grid``) when optimising
        positions or radii, so that the energy is a smooth function of them.

    Returns
    -------
    jax.Array
        ΔG_polar, a scalar of the dtype of ``charges``.
    """
    params = solver.params
    surface = params["surface"]
    geometry = surface in SPLINES and params["bcfl"] != "sdh"
    dielectric = (pdie is not None or sdie is not None) and surface in SURFACES_WITH_DIELECTRIC_DERIVATIVES
    coords, charges, radii = jnp.asarray(coords), jnp.asarray(charges), jnp.asarray(radii)
    dtype = charges.dtype
    pdie = jnp.asarray(params["pdie"] if pdie is None else pdie, dtype)
    sdie = jnp.asarray(params["sdie"] if sdie is None else sdie, dtype)
    n = charges.shape[0]
    wrt = (["positions", "radii"] if geometry else []) + (["dielectric_constants"] if dielectric else [])

    def host(x, q, r, p, s, gradient):
        s_ = _solver(solver, float(p), float(s))
        x, q, r = (np.asarray(a, dtype=np.float64) for a in (x, q, r))
        if not gradient:
            return np.asarray(s_.polar_solvation(x, q, r, grid=grid).energy, dtype)
        g = s_.polar_gradient(x, q, r, grid=grid, wrt=wrt)
        nan = np.nan
        return (
            np.asarray(g.energy, dtype),
            np.asarray(g.positions if geometry else np.full((n, 3), nan), dtype),
            np.asarray(g.charges, dtype),
            np.asarray(g.radii if geometry else np.full(n, nan), dtype),
            np.asarray(g.pdie if dielectric else nan, dtype),
            np.asarray(g.sdie if dielectric else nan, dtype),
        )

    scalar = jax.ShapeDtypeStruct((), dtype)
    shapes = (
        scalar,
        jax.ShapeDtypeStruct((n, 3), dtype),
        jax.ShapeDtypeStruct((n,), dtype),
        jax.ShapeDtypeStruct((n,), dtype),
        scalar,
        scalar,
    )

    @jax.custom_vjp
    def energy(x, q, r, p, s):
        return jax.pure_callback(lambda *a: host(*a, False), scalar, x, q, r, p, s, vmap_method="sequential")

    def forward(x, q, r, p, s):
        e, *grads = jax.pure_callback(lambda *a: host(*a, True), shapes, x, q, r, p, s, vmap_method="sequential")
        return e, tuple(grads)

    def backward(grads, cotangent):
        return tuple(cotangent * g for g in grads)

    energy.defvjp(forward, backward)
    return energy(coords.astype(dtype), charges, radii.astype(dtype), pdie, sdie)
