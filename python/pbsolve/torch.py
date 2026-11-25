"""ΔG_polar as a differentiable PyTorch function (ROADMAP 4.20).

:func:`polar_energy` returns ΔG_polar (kcal/mol) as a 0-d tensor whose
gradient flows, through :meth:`pbsolve.Solver.polar_gradient`, to the charges,
the positions, the radii and the dielectric constants: the exact derivatives
of the discrete energy, through every focusing level and the boundary
condition. PyTorch is an optional dependency, imported only here.

The solver works on host arrays: tensors on a GPU are copied to the host and
the gradients back (a few kilobytes per molecule; the linear solves dominate).
Only first derivatives are available (no double backward).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ._core import GridSpec, Solver

__all__ = ["polar_energy"]


def _host(value) -> np.ndarray:
    """A tensor or array as a float64 NumPy array on the host."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def _like(array, value):
    """`array` as a tensor with the dtype and device of `value` (float64 on the CPU for non-tensors)."""
    if isinstance(value, torch.Tensor):
        return torch.as_tensor(np.asarray(array), dtype=value.dtype, device=value.device)
    return torch.as_tensor(np.asarray(array), dtype=torch.float64)


def _solver_with(solver: Solver, pdie=None, sdie=None) -> Solver:
    """`solver`, or a copy of it with other dielectric constants (same platform and device)."""
    if pdie is None and sdie is None:
        return solver
    params = dict(solver.params)
    if pdie is not None:
        params["pdie"] = float(_host(pdie))
    if sdie is not None:
        params["sdie"] = float(_host(sdie))
    return Solver(platform=solver.platform, device=solver.device, **params)


class _PolarEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, charges, radii, pdie, sdie, solver, grid):
        need = ctx.needs_input_grad
        wrt = [name for name, n in (("positions", 0), ("radii", 2)) if need[n]]
        if need[3] or need[4]:
            wrt.append("dielectric_constants")
        s = _solver_with(solver, pdie, sdie)
        g = s.polar_gradient(_host(coords), _host(charges), _host(radii), grid=grid, wrt=wrt)
        ctx.grads = (
            _like(g.positions, coords) if need[0] else None,
            _like(g.charges, charges) if need[1] else None,
            _like(g.radii, radii) if need[2] else None,
            _like(g.pdie, pdie) if need[3] else None,
            _like(g.sdie, sdie) if need[4] else None,
        )
        like = next((v for v in (charges, coords, radii) if isinstance(v, torch.Tensor)), None)
        return _like(g.energy, like) if like is not None else torch.tensor(g.energy, dtype=torch.float64)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        return (*(None if d is None else grad * d for d in ctx.grads), None, None)


def polar_energy(
    solver: Solver,
    coords,
    charges,
    radii,
    pdie=None,
    sdie=None,
    grid: Optional[GridSpec] = None,
) -> torch.Tensor:
    """ΔG_polar in kcal/mol as a differentiable 0-d tensor.

    Gradients flow to every input that requires them: ``charges`` always,
    ``coords`` and ``radii`` with a spline surface (``surface`` "spl2" or
    "spl4"), ``pdie`` and ``sdie`` with ``surface`` "mol", "vdw", "spl2" or
    "spl4". The solver's other options apply, and asking for a derivative it
    cannot give raises ValueError (see :meth:`pbsolve.Solver.polar_gradient`).

    Parameters
    ----------
    solver : Solver
        The calculation's options.
    coords : torch.Tensor or array_like, shape (N, 3)
        Atom positions in Å.
    charges : torch.Tensor or array_like, shape (N,)
        Atomic charges in e.
    radii : torch.Tensor or array_like, shape (N,)
        Atomic radii in Å.
    pdie, sdie : torch.Tensor or float, optional
        Dielectric constants in place of the solver's (0-d tensors to
        differentiate with respect to them).
    grid : GridSpec, optional
        The grid; by default the solver's automatic grid of the atoms, which
        moves with them. Fix it (``solver.make_grid``) when optimising
        positions or radii, so that the energy is a smooth function of them.

    Returns
    -------
    torch.Tensor
        ΔG_polar, with the dtype and device of ``charges`` (or the first
        tensor among the inputs).
    """
    return _PolarEnergy.apply(coords, charges, radii, pdie, sdie, solver, grid)
