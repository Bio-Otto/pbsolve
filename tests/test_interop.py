"""ΔG_polar as a differentiable PyTorch and JAX function (`pbsolve.torch`, `pbsolve.jax`; ROADMAP 4.20).

Skipped where PyTorch or JAX is not installed (they are optional).
"""

import numpy as np
import pytest

import pbsolve

from conftest import synthetic_molecule

KW = dict(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, pdie=2.0, tol=1e-12)
ALL = ("positions", "radii", "dielectric_constants")


def reference(surface="spl2"):
    x, q, r = synthetic_molecule()
    s = pbsolve.Solver(**KW, surface=surface, chgm="spl2")
    grid = s.make_grid(x, r)
    wrt = ALL if surface.startswith("spl") else ("dielectric_constants",)
    return s, grid, (x, q, r), s.polar_gradient(x, q, r, grid=grid, wrt=wrt)


def test_torch_gradients_are_those_of_polar_gradient():
    torch = pytest.importorskip("torch")
    import pbsolve.torch as pt

    s, grid, (x, q, r), ref = reference()
    t = {k: torch.tensor(v, requires_grad=True) for k, v in dict(x=x, q=q, r=r).items()}
    p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(78.54, dtype=torch.float64, requires_grad=True)
    e = pt.polar_energy(s, t["x"], t["q"], t["r"], pdie=p, sdie=w, grid=grid)
    assert e.dtype == torch.float64 and e.shape == () and e.item() == ref.energy
    e.backward()
    assert np.array_equal(t["x"].grad.numpy(), ref.positions)
    assert np.array_equal(t["q"].grad.numpy(), ref.charges)
    assert np.array_equal(t["r"].grad.numpy(), ref.radii)
    assert p.grad.item() == ref.pdie and w.grad.item() == ref.sdie
    # The output gradient scales them (the chain rule), and plain arrays are constants.
    q2 = torch.tensor(q, requires_grad=True)
    (3.0 * pt.polar_energy(s, x, q2, r, grid=grid)).backward()
    assert np.allclose(q2.grad.numpy(), 3.0 * ref.charges, rtol=1e-15)


def test_torch_keeps_dtype_and_refuses_what_the_surface_cannot_give():
    torch = pytest.importorskip("torch")
    import pbsolve.torch as pt

    s, grid, (x, q, r), ref = reference(surface="mol")
    q32 = torch.tensor(q, dtype=torch.float32, requires_grad=True)
    e = pt.polar_energy(s, x, q32, r, grid=grid)
    e.backward()
    assert e.dtype == torch.float32 and q32.grad.dtype == torch.float32
    assert np.allclose(q32.grad.numpy(), ref.charges, rtol=1e-6)
    with pytest.raises(ValueError, match="spl2"):
        pt.polar_energy(s, torch.tensor(x, requires_grad=True), q, r, grid=grid)


def test_jax_gradients_jit_and_vmap():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import pbsolve.jax as pj

    s, grid, (x, q, r), ref = reference()

    def f(x, q, r, p, w):
        return pj.polar_energy(s, x, q, r, pdie=p, sdie=w, grid=grid)

    args = (jnp.array(x), jnp.array(q), jnp.array(r), 2.0, 78.54)
    assert float(f(*args)) == ref.energy
    gx, gq, gr, gp, gw = jax.grad(f, argnums=(0, 1, 2, 3, 4))(*args)
    assert np.array_equal(gx, ref.positions) and np.array_equal(gq, ref.charges)
    assert np.array_equal(gr, ref.radii) and float(gp) == ref.pdie and float(gw) == ref.sdie
    energy_of = jax.jit(lambda q: pj.polar_energy(s, x, q, r, grid=grid))
    assert float(energy_of(jnp.array(q))) == ref.energy
    # ΔG is quadratic in the charges: half the charges, a quarter of the energy.
    batch = jax.vmap(energy_of)(jnp.stack([jnp.array(q), 0.5 * jnp.array(q)]))
    assert float(batch[0]) == ref.energy and float(batch[1]) == pytest.approx(ref.energy / 4, rel=1e-9)


def test_jax_marks_unavailable_derivatives_as_nan():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import pbsolve.jax as pj

    s, grid, (x, q, r), ref = reference(surface="mol")
    gx, gq = jax.grad(lambda x, q: pj.polar_energy(s, x, q, r, grid=grid), argnums=(0, 1))(jnp.array(x), jnp.array(q))
    assert np.isnan(gx).all()
    assert np.array_equal(gq, ref.charges)
