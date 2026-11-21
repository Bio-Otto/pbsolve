"""Implicit membrane (`zmem`, `lmem`, `mdie`, `membrane_profile`, ...)."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli
from pbsolve.apbs import apbs_unit_factor, find_apbs
from pbsolve.dx import read_dx

from conftest import synthetic_molecule, write_pqr

LB, KT = pbsolve.bjerrum_length(298.15), pbsolve.kt_kcal_per_mol(298.15)
GREENE = [(0.0, 80.0), (5.0, 20.0), (10.0, 4.0), (15.0, 2.0), (20.0, 1.0)]
HELIX = Path(shutil.which("apbs") or "/nonexistent").resolve().parent.parent / "share/apbs/examples/helix"


def layered_reaction(z0, layers, sdie, kbar2):
    """Reaction potential (kT/e) at a unit charge at z0 in a stratified medium: `layers`
    [(z_lo, z_hi, ε)] without ions between half-spaces of water (sdie, κ̄²). Hankel transform
    in the plane; the log-derivative y = ε g'/g of the decaying solutions is carried through
    the layers, and g(z0) = 2 l_B / (y_below − y_above)."""
    from scipy.integrate import quad

    eps0 = next(e for a, b, e in layers if a <= z0 <= b)

    def ys(k):
        lw = math.sqrt(k * k + kbar2 / sdie)
        y = sdie * lw
        for a, b, e in layers:
            hi = min(b, z0)
            if hi > a:
                r, t = y / (e * k), math.tanh(k * (hi - a))
                y = e * k * (r + t) / (1.0 + r * t)
            if hi == z0:
                break
        below, y = y, -sdie * lw
        for a, b, e in reversed(layers):
            lo = max(a, z0)
            if lo < b:
                r, t = y / (e * k), math.tanh(k * (b - lo))
                y = e * k * (r - t) / (1.0 - r * t)
            if lo == z0:
                break
        return below, y

    def f(k):
        below, above = ys(k)
        return (2.0 * LB / (below - above) - LB / (eps0 * k)) * k

    return quad(f, 0.0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-11)[0]


def test_layered_reference_reproduces_the_image_series():
    # A charge in the middle of a slab (Parsegian): u = −(2 l_B / ε_m L) ln(1 − Δ), Δ = (ε_m − ε_w)/(ε_m + ε_w).
    em, ew, thick = 2.0, 78.54, 30.0
    d = (em - ew) / (em + ew)
    exact = -(2.0 * LB / (em * thick)) * math.log(1.0 - d)
    assert layered_reaction(0.0, [(-15.0, 15.0, em)], ew, 0.0) == pytest.approx(exact, rel=1e-10)


def _greene(z, bottom, thick):
    from scipy.interpolate import PchipInterpolator

    depth, eps = zip(*GREENE)
    s = np.minimum(z - bottom, bottom + thick - z)
    return np.where(s >= 20.0, 1.0, PchipInterpolator(depth, eps)(np.clip(s, 0.0, 20.0)))


@pytest.mark.parametrize("profile", ["uniform", "greene2019"])
@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_a_charge_in_the_membrane_matches_the_layered_solution(profile, salt):
    """A point charge in the middle of the membrane, no solute; pdie = ε there, so the
    grid self-energy cancels. Measured −0.02 … +0.08 % (docs/VALIDATION.md)."""
    if profile == "uniform":
        kw, layers = dict(zmem=-15.0, lmem=30.0, mdie=2.0, pdie=2.0), [(-15.0, 15.0, 2.0)]
    else:
        edges = np.linspace(-25.0, 25.0, 5001)
        eps = _greene(0.5 * (edges[1:] + edges[:-1]), -25.0, 50.0)
        kw, layers = dict(zmem=-25.0, lmem=50.0, membrane_profile="greene2019", pdie=1.0), list(zip(edges[:-1], edges[1:], eps))
    s = pbsolve.Solver(platform="CPU", ionic_strength=salt, **kw)
    x, q, r = np.array([[0.0, 0.0, 0.0]]), np.array([1.0]), np.array([0.0])
    grid = pbsolve.GridSpec([0.1, -0.05, 0.03], [49] * 3, [150.0] * 3, [24.0] * 3)
    kbar2 = s.maps(x, q, r, grid=grid)["kappa_bar_sq"]
    exact = 0.5 * layered_reaction(0.0, layers, 78.54, kbar2) * KT
    assert s.polar_solvation(x, q, r, grid=grid).energy == pytest.approx(exact, rel=2e-3)


def test_the_profile_is_scipys_pchip():
    from scipy.interpolate import PchipInterpolator

    # Not monotone, unequal steps, interior slopes from both neighbours, and SciPy's end-slope limit at depth 0.
    points = [(0.0, 10.0), (1.0, 9.0), (2.0, 29.0), (5.0, 15.0), (7.0, 4.0), (10.0, 2.0), (14.0, 3.0), (18.0, 6.0)]
    s = pbsolve.Solver(platform="CPU", pdie=1.0, zmem=-17.3, lmem=40.0, membrane_profile=points)
    grid = pbsolve.GridSpec([0.0, 0.0, 0.0], [9, 9, 97], [8.0, 8.0, 48.0], [8.0, 8.0, 48.0])
    m = s.maps(np.zeros((1, 3)), np.zeros(1), np.zeros(1), grid=grid)  # no solute
    z = m["origin"][2] + np.arange(97) * m["spacing"][2]
    depth = np.minimum(z + 17.3, 22.7 - z)
    inside = depth >= 0.0
    p = PchipInterpolator(*zip(*points))
    expected = np.where(depth >= 18.0, 6.0, p(np.clip(depth, 0.0, 18.0)))
    np.testing.assert_allclose(m["eps_x"][4, 4, inside], expected[inside], rtol=1e-13)
    assert np.all(m["eps_x"][4, 4, ~inside] == 78.54)
    assert np.all(m["kappa_sq"][4, 4, inside] == 0.0)


def test_a_membrane_outside_the_grid_changes_nothing():
    x, q, r = synthetic_molecule()
    plain = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15)
    far = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, zmem=1.0e4)
    a, b = plain.polar_solvation(x, q, r), far.polar_solvation(x, q, r)
    assert (a.energy, a.solvated_kt) == (b.energy, b.solvated_kt)
    near = pbsolve.Solver(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, zmem=-5.0, lmem=10.0)
    c = near.polar_solvation(x, q, r)
    assert c.reference_kt == a.reference_kt and c.energy != a.energy  # the reference state has no membrane


def test_fill_keeps_the_water_of_a_pore():
    # A ring of atoms around the vertical axis spans the membrane: the pore inside is enclosed.
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring = [[6.0 * math.cos(a), 6.0 * math.sin(a), z] for a in angles for z in np.arange(-12.0, 12.1, 2.0)]
    x = np.array(ring)
    q, r = np.zeros(len(x)), np.full(len(x), 2.2)
    grid = pbsolve.GridSpec([0.0, 0.0, 0.0], [49] * 3, [36.0] * 3, [36.0] * 3)
    kw = dict(platform="CPU", ionic_strength=0.1, zmem=-8.0, lmem=16.0)
    centre = (24, 24, 24)  # the axis is half a spacing away
    plain = pbsolve.Solver(**kw).maps(x, q, r, grid=grid)
    filled = pbsolve.Solver(membrane_fill=True, **kw).maps(x, q, r, grid=grid)
    water = pbsolve.Solver(platform="CPU", ionic_strength=0.1).maps(x, q, r, grid=grid)
    assert water["kappa_sq"][centre] > 0.0 and water["eps_x"][centre] == 78.54
    assert plain["kappa_sq"][centre] == 0.0 and plain["eps_x"][centre] == 2.0
    assert filled["kappa_sq"][centre] == water["kappa_sq"][centre] and filled["eps_x"][centre] == 78.54
    corner = (2, 2, 24)
    assert filled["kappa_sq"][corner] == 0.0 and filled["eps_x"][corner] == 2.0
    # Outside the ring, fill changes nothing.
    h, o = np.asarray(plain["spacing"]), np.asarray(plain["origin"])
    i, j = np.meshgrid(np.arange(49), np.arange(49), indexing="ij")
    rho = np.hypot(o[0] + i * h[0] - (o[0] + 49 * h[0] / 2), o[1] + j * h[1] - (o[1] + 49 * h[1] / 2))
    for key in ("eps_x", "eps_y", "eps_z", "kappa_sq"):
        assert np.array_equal(filled[key][rho > 9.0], plain[key][rho > 9.0]), key
    assert not np.array_equal(filled["kappa_sq"], plain["kappa_sq"])
    # A cone of 3 Å keeps the axis, but not the rest of the pore.
    cone = pbsolve.Solver(membrane_exclusion=(3.0, 3.0), **kw).maps(x, q, r, grid=grid)
    assert cone["kappa_sq"][centre] > 0.0 and cone["kappa_sq"][(24, 30, 24)] == 0.0


def test_invalid_membranes():
    x, q, r = synthetic_molecule()
    for kw in (
        dict(lmem=0.0),
        dict(mdie=-1.0),
        dict(membrane_exclusion=(-1.0, 0.0)),
        dict(membrane_profile=[(1.0, 2.0), (5.0, 3.0)]),
        dict(membrane_profile=[(0.0, 2.0), (0.0, 3.0)]),
        dict(membrane_profile=[(0.0, 2.0)]),
        dict(membrane_profile="nonsense"),
        dict(surface="gauss"),
        dict(coulomb="analytic"),
    ):
        with pytest.raises(ValueError):
            pbsolve.Solver(zmem=-10.0, **kw)
    s = pbsolve.Solver(platform="CPU", zmem=-10.0)
    with pytest.raises(ValueError, match="pdie"):
        s.polar_solvation(x, q, r, dielectrics=np.full(len(q), 2.0))
    with pytest.raises(ValueError, match="membrane"):
        pbsolve.Solver(platform="CPU", surface="spl2", zmem=-10.0).polar_forces(x, q, r)
    with pytest.raises(ValueError, match="membrane"):
        pbsolve.apbs.elec_from_solver(s, s.make_grid(x, r), "x")


def test_parameters_round_trip_and_the_command_line(tmp_path, capsys):
    s = pbsolve.Solver(zmem=-20.0, lmem=38.0, membrane_profile="greene2019", membrane_exclusion=(1.0, 2.0), membrane_fill=True)
    p = s.params
    assert (p["zmem"], p["lmem"], p["membrane_exclusion"], p["membrane_fill"]) == (-20.0, 38.0, (1.0, 2.0), True)
    assert p["membrane_profile"] == GREENE
    params = dict(p)
    params.pop("ionic_strength")
    assert pbsolve.Solver(**params).params == p
    assert pbsolve.Solver().params["zmem"] is None
    x, q, r = synthetic_molecule()
    pqr = tmp_path / "m.pqr"
    write_pqr(pqr, x, q, r)
    args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--json", "--zmem", "-4", "--lmem", "8"]
    assert cli(args + ["--membrane-profile", "0:80,4:2"]) == 0
    out = json.loads(capsys.readouterr().out)
    mol = pbsolve.io.load(str(pqr))
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.6, zmem=-4.0, lmem=8.0, membrane_profile=[(0, 80), (4, 2)])
    assert out["polar_kcal_mol"] == py.polar_solvation(mol.coords, mol.charges, mol.radii).energy
    assert cli(args + ["--mdie", "3", "--membrane-exclusion", "1.5", "2.5", "--membrane-fill"]) == 0
    out = json.loads(capsys.readouterr().out)
    kw = dict(mdie=3.0, membrane_exclusion=(1.5, 2.5), membrane_fill=True)
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.6, zmem=-4.0, lmem=8.0, **kw)
    assert out["polar_kcal_mol"] == py.polar_solvation(mol.coords, mol.charges, mol.radii).energy
    assert out["polar_kcal_mol"] != pbsolve.Solver(platform="CPU", grid_spacing=0.6, zmem=-4.0, lmem=8.0, mdie=3.0).polar_solvation(
        mol.coords, mol.charges, mol.radii
    ).energy
    assert cli(args + ["--membrane-profile", "4,2"]) == 2
    assert "--membrane-profile" in capsys.readouterr().err


def _draw_membrane2(tmp_path):
    source = HELIX / "draw_membrane2.c"
    cc = shutil.which("cc") or shutil.which("gcc")
    if find_apbs() is None or cc is None or not source.exists():
        pytest.skip("needs APBS with its examples/helix/draw_membrane2.c and a C compiler")
    exe = tmp_path / "draw_membrane2"
    subprocess.run([cc, "-O2", "-w", "-o", str(exe), str(source), "-lm"], check=True)
    return exe


def _apbs_maps(work, pqr, center, dime, glen, srfm, membrane=None, energy=False):
    """APBS mg-dummy maps of one level and, with `membrane`, draw_membrane2 on them; with
    `energy` the solvated energy (kJ/mol) of APBS solving those maps."""
    c = " ".join(repr(v) for v in center)
    common = f"""  dime {dime} {dime} {dime}
  glen {glen} {glen} {glen}
  gcent {c}
  mol 1
  lpbe
  bcfl zero
  ion charge 1 conc 0.1 radius 2.0
  ion charge -1 conc 0.1 radius 2.0
  pdie 2.0
  sdie 78.54
  srfm {srfm}
  chgm spl0
  srad 1.4
  swin 0.3
  sdens 10.0
  temp 298.15
"""
    names = ("dielx", "diely", "dielz", "kappa", "charge")
    writes = "".join(f"  write {n} dx {n}_L\n" for n in names)
    (work / "d.in").write_text(f"read\n  mol pqr {pqr}\nend\nelec name s\n  mg-dummy\n{common}{writes}end\nquit\n")
    subprocess.run([find_apbs(), "d.in"], cwd=work, capture_output=True, check=True)
    suffix = "_L"
    if membrane is not None:
        exe, zmem, lmem, r_bottom, r_top = membrane
        args = [str(exe), "dielx_L.dx", repr(zmem), repr(lmem), "2.0", "0.0", "0.1", repr(r_top), repr(r_bottom)]
        subprocess.run(args, cwd=work, capture_output=True, check=True)
        suffix = "_Lm"
    maps = {n: read_dx(work / f"{n}{suffix}.dx")[0] for n in names[:4]}
    if not energy:
        return maps, None
    reads = (
        f"  diel dx dielx{suffix}.dx diely{suffix}.dx dielz{suffix}.dx\n  kappa dx kappa{suffix}.dx\n"
        f"  charge dx charge{suffix}.dx\n"
    )
    use = "  usemap diel 1\n  usemap kappa 1\n  usemap charge 1\n  calcenergy total\n"
    (work / "e.in").write_text(f"read\n  mol pqr {pqr}\n{reads}end\nelec name e\n  mg-manual\n{common}{use}end\nquit\n")
    out = subprocess.run([find_apbs(), "e.in"], cwd=work, capture_output=True, text=True, check=True).stdout
    return maps, float([ln for ln in out.splitlines() if "Total electrostatic energy" in ln][-1].split("=")[1].split()[0])


@pytest.mark.parametrize(
    "helix, zmem, lmem, radii, dime, glen, surface",
    [
        ("Membrane-helix-0.pqr", -19.87, 39.6, (3.1, 5.3), 65, 70.0, "mol"),
        ("Membrane-helix-4.pqr", -20.0, 40.0, (0.0, 0.0), 33, 64.0, "spl2"),  # surfaces on node planes (centre z = 0)
    ],
)
def test_maps_equal_apbs_draw_membrane2(tmp_path, helix, zmem, lmem, radii, dime, glen, surface):
    exe = _draw_membrane2(tmp_path)
    pqr = HELIX / helix
    x, q, r = pbsolve.read_pqr(str(pqr))
    center = [float(0.5 * (x[:, a].max() + x[:, a].min())) for a in range(2)] + [0.0]
    theirs, _ = _apbs_maps(tmp_path, pqr, center, dime, glen, "spl2" if surface == "spl2" else "mol", (exe, zmem, lmem, *radii))
    s = pbsolve.Solver(platform="CPU", pdie=2.0, ionic_strength=0.1, surface=surface, zmem=zmem, lmem=lmem, membrane_exclusion=radii)
    ours = s.maps(x, q, r, grid=pbsolve.GridSpec(center, [dime] * 3, [glen] * 3, [glen] * 3))
    ours["kappa"] = ours["kappa_sq"] / ours["kappa_bar_sq"]
    for key, name in (("eps_x", "dielx"), ("eps_y", "diely"), ("eps_z", "dielz"), ("kappa", "kappa")):
        np.testing.assert_allclose(ours[key], theirs[name], rtol=0, atol=1e-6 * np.abs(theirs[name]).max(), err_msg=name)
    assert np.sum(ours["kappa"] == 0.0) > 0.4 * ours["kappa"].size  # the membrane is there


def test_membrane_effect_matches_apbs(tmp_path):
    """APBS solves its draw_membrane2 maps (usemap), pbsolve its own: the membrane's effect on the
    solvated energy agrees to 1.3e-4. The DX route itself (7 digits) costs ~1e-5 of each energy,
    with or without membrane (measured)."""
    exe = _draw_membrane2(tmp_path)
    pqr = HELIX / "Membrane-helix-8.pqr"
    x, q, r = pbsolve.read_pqr(str(pqr))
    center = [float(0.5 * (x[:, a].max() + x[:, a].min())) for a in range(3)]
    grid = pbsolve.GridSpec(center, [65] * 3, [70.0] * 3, [70.0] * 3)
    energies = {}
    for name, membrane in (("water", None), ("membrane", (exe, -20.0, 40.0, 0.0, 0.0))):
        work = tmp_path / name
        work.mkdir()
        _, energies[name] = _apbs_maps(work, pqr, center, 65, 70.0, "mol", membrane, energy=True)
    kw = dict(platform="CPU", pdie=2.0, ionic_strength=0.1, bcfl="zero", tol=1e-12)
    ours = {
        name: pbsolve.Solver(**kw, **extra).polar_solvation(x, q, r, grid=grid).solvated_kt
        * pbsolve.kt_kj_per_mol(298.15)
        * apbs_unit_factor()
        for name, extra in (("water", {}), ("membrane", dict(zmem=-20.0, lmem=40.0)))
    }
    effect = energies["membrane"] - energies["water"]
    assert ours["membrane"] - ours["water"] == pytest.approx(effect, rel=5e-4)
    assert effect > 100.0


@pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")
@pytest.mark.parametrize("extra", [dict(), dict(membrane_profile="greene2019", membrane_fill=True), dict(equation="npbe")])
def test_cuda_matches_cpu(extra):
    x, q, r = synthetic_molecule()
    kw = dict(grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, zmem=-6.0, lmem=12.0, **extra)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(x, q, r)
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(x, q, r)
    assert gpu.platform == "CUDA"
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-10)


def test_cli_membrane_fill(tmp_path, capsys):
    # A charged ring around the axis: with --membrane-fill its pore stays water.
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    x = np.array([[6.0 * math.cos(a), 6.0 * math.sin(a), z] for a in angles for z in np.arange(-8.0, 8.1, 2.0)])
    q = np.where(np.arange(len(x)) % 2 == 0, 0.3, -0.3)
    r = np.full(len(x), 2.2)
    pqr = tmp_path / "ring.pqr"
    write_pqr(pqr, x, q, r)
    args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.7", "--json", "--zmem", "-6", "--lmem", "12"]
    energies = []
    for extra in ([], ["--membrane-fill"]):
        assert cli(args + extra) == 0
        energies.append(json.loads(capsys.readouterr().out)["polar_kcal_mol"])
    mol = pbsolve.io.load(str(pqr))
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.7, zmem=-6.0, lmem=12.0, membrane_fill=True)
    assert energies[1] == py.polar_solvation(mol.coords, mol.charges, mol.radii).energy != energies[0]


def _slab_potential(z, v, kappa, em=2.0, ew=78.54, bottom=-15.0, top=15.0):
    """Membrane potential across a uniform slab between salt water, the bath below at v (Roux 1997;
    APBSmem eq. 8): linear in the slab, exponential in the baths, v/2 in the middle."""
    a = em * v / (2.0 * em + ew * kappa * (top - bottom))
    if z < bottom:
        return v - a * math.exp(kappa * (z - bottom))
    if z > top:
        return a * math.exp(-kappa * (z - top))
    return (v - a) + (2.0 * a - v) * (z - bottom) / (top - bottom)


def _kt_per_e(mv, temperature=298.15):
    return mv * 1e-3 / (1.380649e-23 * temperature / 1.602176634e-19)


def test_the_membrane_potential_field_matches_the_slab_solution():
    """A probe charge (radius 0: no cavity) across a uniform slab: E_V = q φ_mp(z). Exact in the
    middle by symmetry; elsewhere within the placement of the surfaces in a cell (measured ≤ 0.4 %
    of V at 0.34 Å)."""
    kw = dict(platform="CPU", ionic_strength=0.15, zmem=-15.0, lmem=30.0, mdie=2.0)
    kbar2 = pbsolve.Solver(**kw).maps(np.zeros((1, 3)), np.array([1.0]), np.array([0.0]))["kappa_bar_sq"]
    kappa = math.sqrt(kbar2 / 78.54)
    s = pbsolve.Solver(membrane_potential=100.0, **kw)
    v = _kt_per_e(100.0)
    for z in (-40.0, -16.0, -10.0, 0.0, 7.5, 20.0):
        grid = pbsolve.GridSpec([0.13, -0.07, 0.0], [129] * 3, [150.0] * 3, [24.0, 24.0, 44.0 + 2 * abs(z)])
        e = s.polar_solvation(np.array([[0.13, -0.07, z]]), np.array([1.0]), np.array([0.0]), grid=grid)
        phi = e.membrane_potential_energy / KT
        assert phi == pytest.approx(_slab_potential(z, v, kappa), abs=(1e-7 if z == 0.0 else 5e-3) * v), z


def test_the_membrane_potential_energy_is_linear_and_complete():
    x, q, r = synthetic_molecule()
    kw = dict(platform="CPU", grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, zmem=-6.0, lmem=12.0)
    e = {mv: pbsolve.Solver(membrane_potential=mv, **kw).polar_solvation(x, q, r) for mv in (0.0, 60.0, -60.0, 120.0)}
    # One more solve per solvated level (the solvated and reference states have as many levels).
    assert e[0.0].membrane_potential_energy == 0.0 and len(e[60.0].iterations) == 3 * len(e[0.0].iterations) // 2
    assert e[120.0].membrane_potential_energy == pytest.approx(2.0 * e[60.0].membrane_potential_energy, rel=1e-9)
    assert e[-60.0].membrane_potential_energy == pytest.approx(-e[60.0].membrane_potential_energy, rel=1e-9)
    assert e[60.0].energy == pytest.approx(e[0.0].energy + e[60.0].membrane_potential_energy, abs=1e-9)
    assert e[60.0].atom_energies.sum() == pytest.approx(e[60.0].energy, abs=1e-9)
    assert e[60.0].membrane_potential_energy != 0.0
    # An ion deep in the bath below gets q V; one far above nothing.
    s = pbsolve.Solver(platform="CPU", ionic_strength=0.15, zmem=-15.0, lmem=30.0, membrane_potential=100.0)
    for z, want in ((-45.0, 1.0), (45.0, 0.0)):
        grid = pbsolve.GridSpec([0.0, 0.0, z], [97] * 3, [150.0] * 3, [24.0] * 3)
        res = s.polar_solvation(np.array([[0.0, 0.0, z]]), np.array([1.0]), np.array([1.5]), grid=grid)
        assert res.membrane_potential_energy / (KT * _kt_per_e(100.0)) == pytest.approx(want, abs=1e-3)
    # The potential map holds the membrane's field; its energy is ½Σqu + Σqφ_mp.
    s0, s1 = pbsolve.Solver(**kw), pbsolve.Solver(membrane_potential=60.0, **kw)
    g = s0.make_grid(x, r)
    u0, _, _, e0 = s0.potential(x, q, r, grid=g)
    u1, _, _, e1 = s1.potential(x, q, r, grid=g)
    assert (e1 - e0) * KT == pytest.approx(e[60.0].membrane_potential_energy, rel=1e-9)
    assert np.abs(u1 - u0).max() > 1.0
    _, _, _, r0 = s1.potential(x, q, r, grid=g, reference=True)
    assert r0 == s0.potential(x, q, r, grid=g, reference=True)[3]  # the reference state has no membrane


def test_invalid_membrane_potentials(tmp_path, capsys):
    for kw, match in (
        (dict(membrane_potential=50.0, ionic_strength=0.1), "needs a membrane"),
        (dict(membrane_potential=50.0, zmem=-10.0), "needs ions"),
        (dict(membrane_potential=50.0, zmem=-10.0, ionic_strength=0.1, equation="npbe"), "lpbe"),
        (dict(membrane_potential=math.inf, zmem=-10.0, ionic_strength=0.1), "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            pbsolve.Solver(**kw)
    s = pbsolve.Solver(membrane_potential=-70.0, zmem=-10.0, ionic_strength=0.1)
    p = dict(s.params)
    assert p["membrane_potential"] == -70.0
    p.pop("ionic_strength")
    assert pbsolve.Solver(**p).params == s.params
    x, q, r = synthetic_molecule()
    pqr = tmp_path / "m.pqr"
    write_pqr(pqr, x, q, r)
    args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.6", "--json", "--ionic-strength", "0.15",
            "--zmem", "-6", "--lmem", "12", "--membrane-potential", "-70"]
    assert cli(args) == 0
    out = json.loads(capsys.readouterr().out)
    mol = pbsolve.io.load(str(pqr))
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.6, ionic_strength=0.15, zmem=-6.0, lmem=12.0, membrane_potential=-70.0)
    assert out["polar_kcal_mol"] == py.polar_solvation(mol.coords, mol.charges, mol.radii).energy


@pytest.mark.skipif("CUDA" not in pbsolve.available_platforms(), reason="no CUDA device")
@pytest.mark.parametrize("extra", [dict(), dict(period=(30.0, 32.0)), dict(membrane_profile="greene2019", membrane_fill=True)])
def test_membrane_potential_cuda_matches_cpu(extra):
    x, q, r = synthetic_molecule()
    kw = dict(grid_spacing=0.6, fadd=10.0, ionic_strength=0.15, zmem=-6.0, lmem=12.0, membrane_potential=80.0, **extra)
    cpu = pbsolve.Solver(platform="CPU", **kw).polar_solvation(x, q, r)
    gpu = pbsolve.Solver(platform="CUDA", **kw).polar_solvation(x, q, r)
    assert gpu.iterations == cpu.iterations
    assert gpu.membrane_potential_energy == pytest.approx(cpu.membrane_potential_energy, rel=1e-11, abs=1e-12)
    assert gpu.energy == pytest.approx(cpu.energy, rel=1e-12)


def _switch(z, bottom, thick, w):
    """GBSW's membrane switch (Im, Feig & Brooks 2003, Eq. 13), written out independently."""
    u = abs(z - (bottom + thick / 2)) - thick / 2
    if u <= -w:
        return 0.0
    if u >= w:
        return 1.0
    return 0.5 + 3 * u / (4 * w) - u**3 / (4 * w**3)


def test_the_nonpolar_area_leaves_the_membrane_out():
    x, q, r = synthetic_molecule()
    plain = pbsolve.sasa(x, r)
    # A membrane far away is no membrane, bit for bit.
    assert np.array_equal(pbsolve.sasa(x, r, zmem=100.0, lmem=20.0), plain)
    # One sphere across the top surface: dA = 2πR dz (Archimedes), so the area is 2πR ∫ H dz.
    for z0, w in ((5.5, 0.3), (6.8, 2.5), (4.0, 2.5)):
        a = pbsolve.sasa(np.array([[0.2, -0.1, z0]]), np.array([1.6]), zmem=-6.0, lmem=12.0, membrane_switch=w)[0]
        zz = np.linspace(z0 - 3.0, z0 + 3.0, 600_001)
        exact = 2 * math.pi * 3.0 * np.trapezoid([_switch(z, -6.0, 12.0, w) for z in zz], zz)
        assert a == pytest.approx(exact, rel=2e-4, abs=1e-6), (z0, w)
    # The molecule across the membrane: less area, and the energy is γ·area + β.
    kw = dict(zmem=-6.0, lmem=12.0, membrane_switch=0.3)
    area = pbsolve.sasa(x, r, **kw)
    assert 0.0 < area.sum() < 0.8 * plain.sum() and np.all(area <= plain + 1e-12)
    assert pbsolve.nonpolar_energy(x, r, **kw) == pytest.approx(0.00542 * area.sum() + 0.92, rel=1e-14)
    assert pbsolve.nonpolar_energy(x, r) == pytest.approx(0.00542 * plain.sum() + 0.92, rel=1e-14)
    for bad in (dict(zmem=0.0, lmem=0.0), dict(zmem=math.nan), dict(zmem=0.0, membrane_switch=-1.0)):
        with pytest.raises(ValueError):
            pbsolve.sasa(x, r, **bad)


def test_mmpbsa_nonpolar_terms_in_a_membrane(tmp_path, capsys):
    x, q, r = synthetic_molecule()
    lig = np.arange(len(q) - 10, len(q))
    solver = pbsolve.Solver(platform="CPU", grid_spacing=0.7, ionic_strength=0.15, zmem=-6.0, lmem=12.0)
    plain = pbsolve.pbsa_solvation_frame(solver, x, q, r, lig)
    mem = pbsolve.pbsa_solvation_frame(solver, x, q, r, lig, nonpolar_membrane=(-6.0, 12.0), membrane_switch=1.0)
    for a, b, idx in ((plain.complex, mem.complex, slice(None)), (plain.ligand, mem.ligand, lig)):
        assert b.polar == a.polar
        assert b.nonpolar == pytest.approx(pbsolve.nonpolar_energy(x[idx], r[idx], zmem=-6.0, lmem=12.0, membrane_switch=1.0), rel=1e-14)
        assert b.atom_nonpolar.sum() + 0.92 == pytest.approx(b.nonpolar, rel=1e-14)
        assert b.nonpolar < a.nonpolar
    # The command line: --nonpolar-membrane takes the --zmem/--lmem slab; --membrane-potential is reported.
    pqr = tmp_path / "m.pqr"
    write_pqr(pqr, x, q, r)
    mol = pbsolve.io.load(str(pqr))
    args = ["solvation", str(pqr), "--platform", "CPU", "--spacing", "0.7", "--json", "--ionic-strength", "0.15",
            "--zmem", "-6", "--lmem", "12"]
    assert cli(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["nonpolar_kcal_mol"] == pbsolve.nonpolar_energy(mol.coords, mol.radii)
    assert "membrane_potential_kcal_mol" not in out
    assert cli(args + ["--nonpolar-membrane", "--membrane-switch", "1.0", "--membrane-potential", "-40"]) == 0
    out = json.loads(capsys.readouterr().out)
    want = pbsolve.nonpolar_energy(mol.coords, mol.radii, zmem=-6.0, lmem=12.0, membrane_switch=1.0)
    assert out["nonpolar_kcal_mol"] == want
    py = pbsolve.Solver(platform="CPU", grid_spacing=0.7, ionic_strength=0.15, zmem=-6.0, lmem=12.0, membrane_potential=-40.0)
    res = py.polar_solvation(mol.coords, mol.charges, mol.radii)
    assert out["membrane_potential_kcal_mol"] == res.membrane_potential_energy != 0.0
    assert cli(["solvation", str(pqr), "--platform", "CPU", "--nonpolar-membrane"]) == 2
    assert "--nonpolar-membrane needs a membrane" in capsys.readouterr().err
    # mmpbsa: two frames of a multi-model PDB.
    traj = tmp_path / "traj.pdb"
    with open(traj, "w") as fh:
        for m, shift in enumerate((0.0, 0.4), 1):
            fh.write(f"MODEL     {m:4d}\n")
            for n, (a, b, c) in enumerate(mol.coords + shift):
                fh.write(f"ATOM  {n + 1:5d}  C   LIG A{mol.resids[n]:4d}    {a:8.3f}{b:8.3f}{c:8.3f}  1.00  0.00           C\n")
            fh.write("ENDMDL\n")
    csv = tmp_path / "frames.csv"
    sel = f"index {len(q) - 10}:{len(q) - 1}"
    margs = ["mmpbsa", str(pqr), str(traj), "--ligand", sel, "--platform", "CPU", "--spacing", "0.7", "--csv", str(csv),
             "--ionic-strength", "0.15", "--zmem", "-6", "--lmem", "12", "--nonpolar-membrane", "--membrane-switch", "1.0"]
    assert cli(margs) == 0
    capsys.readouterr()
    import csv as csvmod

    with open(csv) as fh:
        rows = list(csvmod.DictReader(fh))
    keep = ~mol.solvent_mask()
    top = mol.subset(keep)
    ligand = np.nonzero(top.select(sel))[0]
    for row, xyz in zip(rows, pbsolve.io.frames(mol.n_atoms, str(traj))):
        f = pbsolve.pbsa_solvation_frame(solver, np.asarray(xyz)[keep], top.charges, top.radii, ligand,
                                         nonpolar_membrane=(-6.0, 12.0), membrane_switch=1.0)
        assert float(row["delta_nonpolar"]) == pytest.approx(f.delta_nonpolar, abs=1e-9)
