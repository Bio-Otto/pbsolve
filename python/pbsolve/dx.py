"""OpenDX scalar-field I/O in the layout written by APBS (readable by VMD, PyMOL, Chimera)."""

from __future__ import annotations

import os
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

PathLike = Union[str, "os.PathLike[str]"]


def write_dx(
    path: PathLike,
    data: npt.ArrayLike,
    origin: Sequence[float],
    spacing: Sequence[float],
    comment: str = "written by pbsolve",
) -> None:
    """Write an (nx, ny, nz) array (z fastest in the file, as in APBS)."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("data must be a 3-D array of shape (nx, ny, nz)")
    nx, ny, nz = arr.shape
    flat = arr.reshape(-1)  # C order == x slowest, z fastest
    with open(path, "w", encoding="ascii") as fh:
        for line in comment.splitlines():
            fh.write(f"# {line}\n")
        fh.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        fh.write("origin {:.6e} {:.6e} {:.6e}\n".format(*origin))
        fh.write(f"delta {spacing[0]:.6e} 0.000000e+00 0.000000e+00\n")
        fh.write(f"delta 0.000000e+00 {spacing[1]:.6e} 0.000000e+00\n")
        fh.write(f"delta 0.000000e+00 0.000000e+00 {spacing[2]:.6e}\n")
        fh.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        fh.write(f"object 3 class array type double rank 0 items {flat.size} data follows\n")
        for start in range(0, flat.size, 3):
            fh.write(" ".join(f"{v:.6e}" for v in flat[start : start + 3]) + "\n")
        fh.write('attribute "dep" string "positions"\n')
        fh.write('object "regular positions regular connections" class field\n')
        fh.write('component "positions" value 1\ncomponent "connections" value 2\ncomponent "data" value 3\n')


def read_dx(path: PathLike) -> tuple[npt.NDArray[np.float64], list[float], list[float]]:
    """Read an OpenDX file written by APBS or pbsolve. Returns (data, origin, spacing)."""
    dims: list[int] = []
    origin: list[float] = []
    spacing: list[float] = []
    values: list[float] = []
    reading = False
    with open(path, encoding="ascii") as fh:
        for line in fh:
            tok = line.split()
            if not tok or tok[0] == "#":
                continue
            if reading:
                if tok[0] in ("attribute", "object", "component"):
                    reading = False
                    continue
                values.extend(float(t) for t in tok)
            elif tok[0] == "object" and "gridpositions" in tok:
                dims = [int(t) for t in tok[-3:]]
            elif tok[0] == "origin":
                origin = [float(t) for t in tok[1:4]]
            elif tok[0] == "delta":
                spacing.append(max(float(t) for t in tok[1:4]))
            elif tok[0] == "object" and "array" in tok:
                reading = True
    data = np.asarray(values, dtype=np.float64).reshape(dims)
    return data, origin, spacing
