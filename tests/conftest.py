import numpy as np
import pytest


def synthetic_molecule(n: int = 30, seed: int = 7):
    """Compact random cluster with crevices, two zero-radius atoms and net charge −1."""
    rng = np.random.default_rng(seed)
    pts = [np.zeros(3)]
    while len(pts) < n:
        p = pts[rng.integers(len(pts))] + rng.normal(size=3) * 1.6
        if min(np.linalg.norm(p - q) for q in pts) > 1.2:
            pts.append(p)
    coords = np.round(np.array(pts), 3)
    radii = np.round(rng.uniform(1.2, 2.0, n), 4)
    radii[5] = radii[17 % n] = 0.0
    q = rng.normal(size=n) * 0.4
    q = np.round(q - (q.sum() + 1.0) / n, 4)
    return coords, q, radii


def write_pqr(path, coords, charges, radii):
    with open(path, "w") as fh:
        for i, (p, q, r) in enumerate(zip(coords, charges, radii), 1):
            fh.write(f"ATOM  {i:5d}  C   LIG A   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f} {q:7.4f} {r:6.4f}\n")
        fh.write("END\n")
    return path


@pytest.fixture
def molecule():
    return synthetic_molecule()


@pytest.fixture
def molecule_pqr(tmp_path, molecule):
    return write_pqr(tmp_path / "mol.pqr", *molecule)
