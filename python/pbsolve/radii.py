"""Dielectric radius schemes from elements (and, for hydrogens, the bonded atom).

Schemes (as in Amber's ``set default PBRadii``):

* ``bondi`` — Bondi (1964) van der Waals radii;
* ``mbondi`` — Bondi, with H bonded to C or N 1.3 Å and H bonded to O or S 0.8 Å;
* ``mbondi2`` — Bondi, with H bonded to N 1.3 Å (Onufriev, Bashford & Case 2004);
* ``parse`` — PARSE radii (Sitkoff, Sharp & Honig 1994) for H, C, N, O, S.

Particles without an element (virtual sites, extra points) get radius 0 and
do not shape the dielectric boundary.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

#: Bondi (1964) radii as used by Amber (Å).
BONDI = {
    "H": 1.2, "C": 1.7, "N": 1.55, "O": 1.5, "F": 1.5, "Si": 2.1,
    "P": 1.85, "S": 1.8, "Cl": 1.7, "Br": 1.85, "I": 1.98,
}
#: PARSE radii (Sitkoff, Sharp & Honig 1994); other elements fall back to Bondi.
PARSE = {"H": 1.0, "C": 1.7, "N": 1.5, "O": 1.4, "S": 1.85}
#: Amber's radius for elements missing from the tables.
FALLBACK_RADIUS = 1.5

SCHEMES = ("bondi", "mbondi", "mbondi2", "parse")


def hydrogen_partners(
    elements: Sequence[str],
    bonds: Optional[npt.ArrayLike] = None,
    coords: Optional[npt.ArrayLike] = None,
    cutoff: float = 1.4,
) -> list[str]:
    """Element bonded to each hydrogen ('' for non-hydrogens or if unknown).

    Uses ``bonds`` when given, otherwise the nearest heavy atom within
    ``cutoff`` Å of each hydrogen in ``coords``.
    """
    el = list(elements)
    out = [""] * len(el)
    hs = [i for i, e in enumerate(el) if e == "H"]
    if not hs:
        return out
    b = np.asarray(bonds if bonds is not None else np.zeros((0, 2)), dtype=np.int64).reshape(-1, 2)
    if b.size:
        for i, j in b:
            if el[i] == "H" and el[j] not in ("H", ""):
                out[i] = el[j]
            elif el[j] == "H" and el[i] not in ("H", ""):
                out[j] = el[i]
        return out
    if coords is None:
        return out
    xyz = np.asarray(coords, dtype=np.float64)
    heavy = np.array([i for i, e in enumerate(el) if e not in ("H", "")], dtype=np.int64)
    if heavy.size == 0:
        return out
    hx = xyz[hs]
    # chunked nearest-neighbour search (memory bounded)
    for start in range(0, len(hs), 2048):
        d2 = ((hx[start : start + 2048, None, :] - xyz[heavy][None, :, :]) ** 2).sum(-1)
        k = d2.argmin(axis=1)
        for m, (kk, dd) in enumerate(zip(k, d2[np.arange(len(k)), k])):
            if dd <= cutoff * cutoff:
                out[hs[start + m]] = el[heavy[kk]]
    return out


def assign_radii(
    elements: Sequence[str],
    scheme: str = "mbondi2",
    bonds: Optional[npt.ArrayLike] = None,
    coords: Optional[npt.ArrayLike] = None,
) -> npt.NDArray[np.float64]:
    """Radii (Å) for each atom from its element (see module docstring)."""
    scheme = scheme.lower()
    if scheme not in SCHEMES:
        raise ValueError(f"unknown radius scheme '{scheme}' (expected one of {', '.join(SCHEMES)})")
    el = list(elements)
    partner = hydrogen_partners(el, bonds, coords) if scheme in ("mbondi", "mbondi2") else [""] * len(el)
    table = PARSE if scheme == "parse" else BONDI
    out = np.zeros(len(el), dtype=np.float64)
    for i, e in enumerate(el):
        if not e:
            continue
        r = table.get(e, BONDI.get(e, FALLBACK_RADIUS))
        if e == "H" and scheme == "mbondi2":
            r = 1.3 if partner[i] == "N" else 1.2
        elif e == "H" and scheme == "mbondi":
            r = {"C": 1.3, "N": 1.3, "O": 0.8, "S": 0.8}.get(partner[i], 1.2)
        out[i] = r
    return out
