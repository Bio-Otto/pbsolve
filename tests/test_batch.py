"""Concurrent batches: `polar_solvation_many`, `polar_solvation_batch` with
workers, `pbsa_solvation_frames`, `pbsolve mmpbsa --workers` and the worker
count (`batch_workers`).

Concurrency must only change the throughput: every result has to equal the
sequential one bit for bit, in input order, on every platform.
"""

import csv
import dataclasses

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli

PLATFORMS = pbsolve.available_platforms()


@pytest.fixture(scope="module")
def peptide_frames():
    """Chain B of 1YCR (p53 peptide) and five perturbed copies."""
    coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
    keep = pbsolve.io.load("tests/data/1ycr_amber_opt_pdb2pqr.pqr").select("chain B")
    xyz, q, r = coords[keep], charges[keep], radii[keep]
    rng = np.random.default_rng(5)
    frames = [xyz + rng.normal(0.0, 0.25, xyz.shape) for _ in range(5)]
    return frames, q, r


def solver(platform="CPU", **kw):
    return pbsolve.Solver(platform=platform, grid_spacing=0.6, ionic_strength=0.15, fadd=10.0, **kw)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_many_equals_sequential_calls_bit_for_bit(peptide_frames, platform):
    frames, q, r = peptide_frames
    s = solver(platform)
    systems = [(x, q, r) for x in frames] + [(frames[0][:40], q[:40], r[:40])]
    sequential = [s.polar_solvation(*sys_) for sys_ in systems]
    for workers in (1, 2, 3, 7):
        many = s.polar_solvation_many(systems, workers=workers)
        assert [m.energy for m in many] == [x.energy for x in sequential], workers
        assert [m.iterations for m in many] == [x.iterations for x in sequential]
        assert [m.grid.dims for m in many] == [x.grid.dims for x in sequential]
        assert all(m.platform == platform for m in many)


def test_batch_with_workers_equals_single_frames(peptide_frames):
    frames, q, r = peptide_frames
    s = solver()
    grid = s.make_grid(frames[0], r)
    single = [s.polar_solvation(x, q, r, grid=grid).energy for x in frames]
    for workers in (None, 1, 3):
        batch = s.polar_solvation_batch(np.stack(frames), q, r, grid=grid, workers=workers)
        assert batch.tolist() == single


def test_grids_shared_per_system_or_automatic(peptide_frames):
    frames, q, r = peptide_frames
    s = solver()
    systems = [(x, q, r) for x in frames[:3]]
    shared = s.make_grid(frames[0], r)
    own = [s.make_grid(x, r) for x, _, _ in systems]
    with_shared = s.polar_solvation_many(systems, grid=shared, workers=2)
    assert all(m.grid.dims == shared.dims for m in with_shared)
    assert [m.energy for m in with_shared] == [s.polar_solvation(*x, grid=shared).energy for x in systems]
    per_system = s.polar_solvation_many(systems, grid=own, workers=2)
    automatic = s.polar_solvation_many(systems, workers=2)
    assert [m.energy for m in per_system] == [m.energy for m in automatic]


def test_invalid_batch_arguments(peptide_frames):
    frames, q, r = peptide_frames
    s = solver()
    systems = [(x, q, r) for x in frames[:2]]
    with pytest.raises(ValueError, match="workers"):
        s.polar_solvation_many(systems, workers=0)
    with pytest.raises(ValueError, match="1 grids given for 2 systems"):
        s.polar_solvation_many(systems, grid=[s.make_grid(frames[0], r)])
    with pytest.raises(ValueError, match="GridSpec"):
        s.polar_solvation_many(systems, grid="auto")
    with pytest.raises(ValueError, match="system 1"):
        s.polar_solvation_many([(frames[0], q, r), (frames[1], q[:-1], r)])
    assert s.polar_solvation_many([]) == []


def test_failing_system_is_named_and_others_are_unaffected(peptide_frames):
    frames, q, r = peptide_frames
    s = solver()
    grid = s.make_grid(frames[0], r)
    far = frames[1] + 500.0  # outside the grid
    with pytest.raises(ValueError, match="system 1: .*outside"):
        s.polar_solvation_many([(frames[0], q, r), (far, q, r), (frames[2], q, r)], grid=grid, workers=3)
    # The solver stays usable afterwards.
    assert s.polar_solvation_many([(frames[0], q, r)], grid=grid, workers=3)[0].energy == (
        s.polar_solvation(frames[0], q, r, grid=grid).energy
    )


@pytest.mark.parametrize("platform", PLATFORMS)
def test_batch_workers(peptide_frames, platform):
    frames, q, r = peptide_frames
    s = solver(platform)
    grid = s.make_grid(frames[0], r)
    # A small grid: the default (CPU 2, CUDA 3) or what is asked for.
    assert s.batch_workers(grid) == (3 if platform == "CUDA" else 2)
    assert s.batch_workers([grid, grid], workers=5) == 5
    assert s.batch_workers(grid, workers=1) == 1
    # 1025^3 nodes need ~140 GB per calculation: on a GPU only one runs at a
    # time (and fails if it does not fit); the CPU ignores device memory.
    huge = pbsolve.GridSpec([0.0, 0.0, 0.0], [1025, 1025, 1025], [400.0, 400.0, 400.0])
    assert s.batch_workers([grid, huge], workers=3) == (1 if platform == "CUDA" else 3)
    with pytest.raises(ValueError, match="workers"):
        s.batch_workers(grid, workers=0)
    with pytest.raises(ValueError, match="GridSpec"):
        s.batch_workers("auto")


def test_timings_are_reported(peptide_frames):
    frames, q, r = peptide_frames
    res = solver().polar_solvation(frames[0], q, r)
    t = res.timings
    assert set(t) == {"maps", "boundary", "assembly", "setup", "solve", "energy", "total"}
    assert all(v >= 0.0 for v in t.values())
    parts = sum(v for k, v in t.items() if k != "total")
    assert t["total"] == pytest.approx(parts)
    assert t["solve"] > 0.0


@pytest.mark.parametrize("platform", PLATFORMS)
def test_pbsa_frames_equal_frame_by_frame(platform):
    coords, charges, radii = pbsolve.read_pqr("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
    mol = pbsolve.io.load("tests/data/1ycr_amber_opt_pdb2pqr.pqr")
    # A small "complex": the p53 peptide (ligand) and the MDM2 residues near it.
    lig_mask = mol.select("chain B")
    centre = coords[lig_mask].mean(axis=0)
    near = np.linalg.norm(coords - centre, axis=1) < 14.0
    sel = near | lig_mask
    xyz, q, r = coords[sel], charges[sel], radii[sel]
    lig = np.nonzero(lig_mask[sel])[0]
    rng = np.random.default_rng(9)
    frames = [xyz + rng.normal(0.0, 0.2, xyz.shape) for _ in range(3)]
    s = solver(platform)
    one_by_one = [pbsolve.pbsa_solvation_frame(s, x, q, r, lig).as_dict() for x in frames]
    for workers in (None, 1, 4):
        together = [f.as_dict() for f in pbsolve.pbsa_solvation_frames(s, frames, q, r, lig, workers=workers)]
        assert together == one_by_one
    grid = s.make_grid(frames[0], r)
    fixed = pbsolve.pbsa_solvation_frames(s, frames, q, r, lig, grid=grid, workers=2)
    assert [f.as_dict() for f in fixed] == [pbsolve.pbsa_solvation_frame(s, x, q, r, lig, grid=grid).as_dict() for x in frames]
    assert pbsolve.pbsa_solvation_frames(s, [], q, r, lig) == []


def test_mmpbsa_cli_output_does_not_depend_on_workers(tmp_path):
    # The synthetic GROMACS fixture has no charges: give it some through a PQR.
    gro = pbsolve.io.load("tests/data/synthetic.gro")
    rng = np.random.default_rng(3)
    mol = dataclasses.replace(gro, charges=rng.uniform(-0.4, 0.4, gro.n_atoms), radii=np.full(gro.n_atoms, 1.6))
    pqr = tmp_path / "synthetic.pqr"
    mol.write_pqr(pqr)
    args = ["mmpbsa", str(pqr), "tests/data/synthetic.xtc", "--ligand", "resid 2", "--platform", "CPU",
            "--spacing", "0.7"]
    out = {}
    for workers in (1, 3):
        path = tmp_path / f"w{workers}.csv"
        assert cli(args + ["--workers", str(workers), "--csv", str(path)]) == 0
        with open(path) as fh:
            out[workers] = list(csv.DictReader(fh))
    assert len(out[1]) == sum(1 for _ in pbsolve.io.frames(gro.n_atoms, "tests/data/synthetic.xtc"))
    assert out[1] == out[3]
    assert [row["frame"] for row in out[1]] == [str(k) for k in range(len(out[1]))]
    assert cli(args + ["--workers", "0"]) == 2
