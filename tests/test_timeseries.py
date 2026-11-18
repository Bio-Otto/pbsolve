"""Uncertainty over correlated frames (`pbsolve.timeseries`, `pbsolve mmpbsa`)."""

import dataclasses
import math

import numpy as np
import pytest

import pbsolve
from pbsolve.__main__ import main as cli
from pbsolve.timeseries import _t_quantile, block_averages, block_bootstrap, statistical_inefficiency


def ar1(n, phi, seed, mean=0.0):
    """Autoregressive series x_i = φ x_{i−1} + noise with unit variance: g = (1 + φ)/(1 − φ)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n) * math.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + mean


def ar1_sem(n, phi):
    """Exact standard error of the mean of n frames of `ar1`."""
    t = np.arange(1, n)
    return math.sqrt((1.0 + 2.0 * np.sum((1.0 - t / n) * phi**t)) / n)


def reference_inefficiency(x, mintime=3):
    """Chodera et al. (2007) term by term (pymbar's loop): direct sums, no FFT."""
    n = len(x)
    dx = x - x.mean()
    var = np.mean(dx * dx)
    g = 1.0
    for t in range(1, n - 1):
        c = np.sum(dx[: n - t] * dx[t:]) / ((n - t) * var)
        if c <= 0.0 and t > mintime:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def test_the_inefficiency_is_the_term_by_term_sum():
    # Equal to pymbar 4.0.3's statistical_inefficiency to 6e-16 (docs/VALIDATION.md).
    for phi in (0.0, 0.5, 0.95):
        for n in (8, 60, 700):
            x = ar1(n, phi, seed=n) * 2.3 + 7.0
            assert statistical_inefficiency(x) == pytest.approx(reference_inefficiency(x), rel=1e-12)
    # Negative correlations before lag 4 count; g never drops below 1.
    alternating = np.array([1.0, -1.0] * 20)
    assert statistical_inefficiency(alternating) == 1.0
    assert statistical_inefficiency(np.full(10, 3.0)) == 1.0


def test_long_series_recover_the_exact_inefficiency_and_error():
    for phi, tol in ((0.0, 0.06), (0.5, 0.1), (0.9, 0.2)):
        x = ar1(20_000, phi, seed=1, mean=5.0)
        st = pbsolve.series_statistics(x)
        g = (1.0 + phi) / (1.0 - phi)
        assert st.inefficiency == pytest.approx(g, rel=tol)
        assert st.tau == pytest.approx(0.5 * (st.inefficiency - 1.0))
        assert st.n_eff == pytest.approx(20_000 / st.inefficiency)
        assert st.sem == pytest.approx(ar1_sem(20_000, phi), rel=tol)
        assert st.sem_blocking == pytest.approx(ar1_sem(20_000, phi), rel=0.15)
        assert st.sem_independent == pytest.approx(st.sd / math.sqrt(20_000))
        assert st.enough


def test_the_interval_covers_the_true_mean():
    """Student-t intervals with N_eff − 1 degrees of freedom: 94–98 % coverage on AR(1)
    series from ~20 effective frames on (docs/VALIDATION.md); here φ = 0.8, 1000 frames."""
    hits = 0
    for seed in range(200):
        lo, hi = pbsolve.series_statistics(ar1(1000, 0.8, seed)).ci
        hits += lo <= 0.0 <= hi
    assert 0.90 <= hits / 200 <= 0.99, hits


def test_blocking_curve():
    x = ar1(1000, 0.0, seed=4)
    levels = block_averages(x)
    assert [m for _, m, _ in levels[:4]] == [1000, 500, 250, 125]
    assert [b for b, _, _ in levels[:4]] == [1, 2, 4, 8]
    assert levels[0][2] == pytest.approx(np.std(x, ddof=1) / math.sqrt(1000))
    assert levels[-1][1] == 3  # 1000 → … → 7 → 3 (odd values dropped)


def test_block_bootstrap():
    x = ar1(4000, 0.0, seed=8, mean=1.0)
    st = pbsolve.series_statistics(x)
    lo, hi = block_bootstrap(x)
    assert (hi - lo) / 2 == pytest.approx(1.96 * st.sem, rel=0.15)
    assert block_bootstrap(x) == (lo, hi)  # seeded
    assert block_bootstrap(x, seed=1) != (lo, hi)
    med = block_bootstrap(x, np.median, block=10, resamples=300)
    assert med[0] < np.median(x) < med[1]
    y = ar1(4000, 0.9, seed=8)
    lo, hi = block_bootstrap(y)
    assert hi - lo > 3.0 * (2 * 1.96 * np.std(y, ddof=1) / math.sqrt(4000))  # correlated: much wider
    with pytest.raises(ValueError):
        block_bootstrap(x, confidence=1.0)
    # The default block is 4 g (it covered best on AR(1) series).
    assert block_bootstrap(y) == block_bootstrap(y, block=math.ceil(4 * statistical_inefficiency(y)))


def test_t_quantiles():
    # SciPy 1.18: t.ppf(0.975, ν) for ν = 2, 3, 10, 50.
    for nu, exact, tol in ((2, 4.302652730, 8e-3), (3, 3.182446305, 1.3e-3), (10, 2.228138852, 1e-5), (50, 2.008559112, 1e-8)):
        assert _t_quantile(0.975, nu) == pytest.approx(exact, rel=tol)


def test_frames_needed_and_errors():
    x = ar1(500, 0.7, seed=2)
    st = pbsolve.series_statistics(x, confidence=0.9)
    assert st.sd == pytest.approx(np.std(x, ddof=1)) and st.mean == pytest.approx(x.mean())
    half = 0.5 * (st.ci[1] - st.ci[0])
    assert half == pytest.approx(_t_quantile(0.95, st.n_eff - 1.0) * st.sem)  # two-sided 90 %
    assert st.frames_for(st.sem) in (500, 501)  # g (SD/SEM)² = N
    assert st.frames_for(st.sem / 2) == pytest.approx(4 * 500, rel=1e-3)
    assert st.confidence == 0.9 and st.ci[0] < st.mean < st.ci[1]
    with pytest.raises(ValueError):
        st.frames_for(0.0)
    few = pbsolve.series_statistics([1.0, 2.0, 1.5])
    assert not few.enough and math.isnan(few.sem_blocking)
    assert math.isnan(pbsolve.series_statistics([1.0, 2.0]).ci[0])  # 1 degree of freedom: no interval
    for bad in ([1.0], [[1.0, 2.0]], [1.0, math.nan]):
        with pytest.raises(ValueError):
            pbsolve.series_statistics(bad)
    with pytest.raises(ValueError):
        pbsolve.series_statistics(x, confidence=0.0)


def test_mmpbsa_reports_the_correlated_uncertainty(tmp_path, capsys):
    """40 frames of the synthetic molecule whose atoms drift (AR(1) displacements), as a multi-model PDB."""
    gro = pbsolve.io.load("tests/data/synthetic.gro")
    rng = np.random.default_rng(3)
    mol = dataclasses.replace(gro, charges=rng.uniform(-0.4, 0.4, gro.n_atoms), radii=np.full(gro.n_atoms, 1.6))
    pqr = tmp_path / "synthetic.pqr"
    mol.write_pqr(pqr)
    shift = np.zeros_like(mol.coords)
    frames = []
    for _ in range(40):
        shift = 0.8 * shift + 0.3 * rng.normal(size=shift.shape)
        frames.append(mol.coords + shift)
    traj = tmp_path / "traj.pdb"
    with open(traj, "w") as fh:
        for m, xyz in enumerate(frames, 1):
            fh.write(f"MODEL     {m:4d}\n")
            for n, (x, y, z) in enumerate(xyz):
                fh.write(f"ATOM  {n + 1:5d}  C   LIG A{mol.resids[n]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            fh.write("ENDMDL\n")
    csv = tmp_path / "res.csv"
    per_frame = tmp_path / "frames.csv"
    args = ["mmpbsa", str(pqr), str(traj), "--ligand", "resid 2", "--platform", "CPU", "--spacing", "0.7",
            "--target-sem", "0.05", "--per-residue", str(csv), "--csv", str(per_frame)]
    assert cli(args) == 0
    out = capsys.readouterr().out
    import csv as csvmod

    with open(per_frame) as fh:
        rows = list(csvmod.DictReader(fh))
    assert len(rows) == 40
    for key in ("delta_polar", "delta_nonpolar", "delta_solvation"):
        st = pbsolve.series_statistics([float(r[key]) for r in rows])
        line = next(row for row in out.splitlines() if row.strip().startswith(f"{key} ") and "SEM" in row and "CI" in row)
        assert f"SEM {st.sem:.4f}" in line and f"frames for SEM ≤ 0.05: {st.frames_for(0.05)}" in line, line
        assert f"[{st.ci[0]:.4f}, {st.ci[1]:.4f}]" in line and f"N_eff {st.n_eff:.1f}" in line
    assert "fewer than 20 effective frames" in out
    # Per residue: the correlated standard error of each residue's total over the frames.
    solver = pbsolve.Solver(platform="CPU", grid_spacing=0.7)
    # The files' values (coordinates to 0.001 Å, the PQR's charges and radii)
    # without the solvent, as the command line takes them.
    full = pbsolve.io.load(str(pqr))
    keep = ~full.solvent_mask()
    top = full.subset(keep)
    read = [np.array(f, dtype=np.float64)[keep] for f in pbsolve.io.frames(full.n_atoms, str(traj))]
    lig = np.nonzero(top.select("resid 2"))[0]
    labels = [f"{c}:{n}:{i}" for c, n, i in zip(top.chains, top.resnames, top.resids)]
    per = {}
    for f in pbsolve.pbsa_solvation_frames(solver, read, top.charges, top.radii, lig):
        for label, (p, n) in f.residue_terms(labels).items():
            per.setdefault(label, []).append(p + n)
    with open(csv) as fh:
        table = {r["residue"]: r for r in csvmod.DictReader(fh)}
    for label, totals in per.items():
        row = table[label]
        assert float(row["sem_solvation_correlated"]) == pytest.approx(pbsolve.series_statistics(totals).sem, rel=1e-9)
        assert float(row["sem_solvation"]) == pytest.approx(np.std(totals, ddof=1) / math.sqrt(40), rel=1e-9)
    assert any(float(r["sem_solvation_correlated"]) > 1.2 * float(r["sem_solvation"]) for r in table.values())
    assert cli(args[:9] + ["--target-sem", "0"]) == 2
    assert "--target-sem must be positive" in capsys.readouterr().err
