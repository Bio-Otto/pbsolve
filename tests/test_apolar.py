"""Non-polar solvation energy (`pbsolve.apolar_energy`) as APBS `apolar`."""

from pathlib import Path

import numpy as np
import pytest

import pbsolve
from pbsolve.apbs import find_apbs, read_parm

from conftest import synthetic_molecule

ALKANES = Path(__file__).parent / "data" / "apbs_alkanes"
KJ = 4.184

# APBS' examples/alkanes/README.md (APBS 3.0, identical since 1.4): SASA (Å²), SAV (Å³),
# WCA (kJ/mol) as printed with %g, and the total non-polar energy (kJ/mol) printed with %1.12E.
APBS_ALKANES = {
    "2-methylbutane": (214.202, 253.665, -48.1507, 1.439739455792e01),
    "butane": (193.855, 217.863, -41.7207, 1.208346456826e01),
    "cyclohexane": (221.799, 267.435, -52.3691, 1.354016672221e01),
    "cyclopentane": (193.638, 217.998, -44.471, 9.363673200142e00),
    "ethane": (139.427, 140.346, -25.3612, 9.422717598546e00),
    "hexane": (250.291, 298.053, -57.0807, 1.640068943201e01),
    "isobutane": (192.744, 218.943, -40.8218, 1.323144287435e01),
    "methane": (105.42, 95.985, -15.9805, 7.894367190329e00),
    "neopentane": (210.755, 251.127, -47.4149, 1.449633815052e01),
    "pentane": (222.524, 258.93, -49.4003, 1.447900211546e01),
    "propane": (170.391, 183.573, -33.4721, 1.192358496286e01),
}


def alkane(name):
    """Coordinates, radii and well depths (kcal/mol) as APBS assigns them from parm.dat."""
    parm = read_parm(ALKANES / "parm.dat")
    xyz, r, e = [], [], []
    # The files are not column-aligned; APBS reads whitespace-separated fields.
    for f in (line.split() for line in (ALKANES / f"{name}.pdb").read_text().splitlines()):
        if f and f[0] in ("ATOM", "HETATM"):
            _, radius, eps = parm[(f[3], f[2])]
            xyz.append([float(v) for v in f[5:8]])
            r.append(radius)
            e.append(eps / KJ)
    return np.array(xyz), np.array(r), np.array(e)


@pytest.mark.parametrize("name", sorted(APBS_ALKANES))
def test_alkanes_reproduce_apbs(name):
    # Wagoner & Baker's alkane protocol as APBS runs it (grid 0.3 Å, sdens 100).
    area, volume, wca, total = APBS_ALKANES[name]
    e = pbsolve.apolar_energy(*alkane(name), grid=0.3)
    assert e.area == pytest.approx(area, abs=0.5e-3) and e.volume == pytest.approx(volume, abs=0.5e-3)
    assert e.dispersion * KJ == pytest.approx(wca, abs=0.5e-4)
    assert e.energy * KJ == pytest.approx(total, rel=1e-12)  # every printed digit


def test_parts_and_units():
    x, r, e = alkane("hexane")
    full = pbsolve.apolar_energy(x, r, e)
    assert full.energy == pytest.approx(full.surface + full.pressure_volume + full.dispersion, rel=1e-15)
    assert full.surface == pytest.approx(pbsolve.apolar.GAMMA * full.area, rel=1e-15)
    assert full.atom_area.sum() == pytest.approx(full.area, rel=1e-15)
    assert full.atom_dispersion.sum() == pytest.approx(full.dispersion, rel=1e-15)
    # The terms are linear in their coefficients; a zero coefficient skips its term.
    double = pbsolve.apolar_energy(x, r, 4 * e, gamma=2 * pbsolve.apolar.GAMMA, press=0.0, bconc=2 * pbsolve.apolar.BCONC)
    assert double.surface == pytest.approx(2 * full.surface, rel=1e-14)
    assert double.dispersion == pytest.approx(4 * full.dispersion, rel=1e-14)  # 2 ρ̄ · √(4 ε ε_w)
    assert np.isnan(double.volume) and double.pressure_volume == 0.0
    area_only = pbsolve.apolar_energy(x, r, gamma=1.0, press=0.0, bconc=0.0)
    assert area_only.energy == pytest.approx(full.area, rel=1e-15) and not area_only.atom_dispersion.any()


def test_bad_input():
    x, r, e = alkane("methane")
    with pytest.raises(ValueError, match="epsilons"):
        pbsolve.apolar_energy(x, r)
    for kw in (dict(grid=0.0), dict(sdens=-1.0), dict(srad=-0.1)):
        with pytest.raises(ValueError):
            pbsolve.apolar_energy(x, r, e, **kw)
    with pytest.raises(ValueError, match="epsilons"):
        pbsolve.apolar_energy(x, r, e[:2])


@pytest.mark.skipif(find_apbs() is None, reason="APBS not installed")
def test_apbs_parity_with_radius_zero_atoms():
    # Atoms of radius 0 have no area but bury surface points of others within srad.
    import pbsolve.apbs as A

    x, q, r = synthetic_molecule()
    e = np.random.default_rng(3).uniform(0.01, 0.2, len(r))
    kw = dict(srad=1.4, sdens=10.0, grid=0.5)
    ours, apbs = pbsolve.apolar_energy(x, r, e, **kw), A.apolar_energy(x, r, e, **kw)
    for part in ("area", "volume", "dispersion", "energy"):
        assert getattr(ours, part) == pytest.approx(getattr(apbs, part), rel=1e-12), part
    np.testing.assert_allclose(ours.atom_area, apbs.atom_area, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(ours.atom_dispersion, apbs.atom_dispersion, rtol=1e-11, atol=1e-13)
    assert (r == 0).any() and (ours.atom_area[r == 0] == 0).all()
