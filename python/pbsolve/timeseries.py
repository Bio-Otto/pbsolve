"""Uncertainty of averages over correlated frames.

Consecutive frames of a trajectory are correlated, so the standard error of
an average is larger than SD/√N. :func:`series_statistics` estimates it
from the statistical inefficiency g of the series (Chodera et al. 2007, the
estimator of pymbar): the number of frames per independent sample,

    g = 1 + 2 Σ_{t≥1} (1 − t/N) C(t),

summed over the normalised autocorrelation C(t) until it first drops to
zero after lag 3. The integrated correlation time is τ = (g − 1)/2 frames,
the effective number of frames N_eff = N/g, the standard error of the mean
SD·√(g/N), and its confidence interval uses Student's t with N_eff − 1
degrees of freedom. Block averaging (Flyvbjerg & Petersen 1989) is an
independent check, and :func:`block_bootstrap` (moving blocks, Künsch 1989)
gives intervals for statistics other than the mean. On autoregressive
series with known answers the interval covers the true mean in 94–98 % of
the cases from about 20 effective frames on, and less with fewer
(docs/VALIDATION.md, *Uncertainty over frames*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

__all__ = ["SeriesStatistics", "block_averages", "block_bootstrap", "series_statistics", "statistical_inefficiency"]

#: Effective frames below which the correlation time, and hence the error, is itself uncertain.
MIN_EFFECTIVE_FRAMES = 20.0


def _series(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("need a one-dimensional series of at least 2 finite values")
    return x


def statistical_inefficiency(values: npt.ArrayLike, mintime: int = 3) -> float:
    """Statistical inefficiency g ≥ 1 of a series (frames per independent sample).

    Chodera et al. (2007): g = 1 + 2 Σ_t (1 − t/N) C(t), with C(t) the
    autocorrelation at lag t (mean over the N − t pairs, normalised by the
    variance), summed until C(t) ≤ 0 for a lag t > `mintime`. A constant series has g = 1.
    """
    x = _series(values)
    n = x.size
    dx = x - x.mean()
    var = float(np.mean(dx * dx))
    if var == 0.0:
        return 1.0
    spectrum = np.fft.rfft(dx, 2 * n)
    cov = np.fft.irfft(spectrum * np.conj(spectrum), 2 * n)[:n] / (n - np.arange(n))
    c = cov / var
    g = 1.0
    for t in range(1, n - 1):
        if c[t] <= 0.0 and t > mintime:
            break
        g += 2.0 * c[t] * (1.0 - t / n)
    return max(float(g), 1.0)


def block_averages(values: npt.ArrayLike) -> list[tuple[int, int, float]]:
    """Flyvbjerg–Petersen blocking: (block length, number of blocks, standard error) per level.

    Each level averages pairs of the previous one (a last odd value is
    dropped); the standard error √(var/(n − 1)) of the block means rises with
    the block length until the blocks are independent and then levels off.
    """
    x = _series(values)
    out = []
    length = 1
    while x.size >= 2:
        out.append((length, x.size, math.sqrt(float(x.var()) / (x.size - 1))))
        m = x.size // 2 * 2
        x = 0.5 * (x[0:m:2] + x[1:m:2])
        length *= 2
    return out


def _t_quantile(p: float, nu: float) -> float:
    """Student-t quantile, Cornish–Fisher expansion (Abramowitz & Stegun 26.7.5):
    within 0.12 % for p = 0.975 and ν ≥ 3, 0.75 % at ν = 2."""
    z = NormalDist().inv_cdf(p)
    g1 = (z**3 + z) / 4.0
    g2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / 96.0
    g3 = (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / 384.0
    g4 = (79.0 * z**9 + 776.0 * z**7 + 1482.0 * z**5 - 1920.0 * z**3 - 945.0 * z) / 92160.0
    return z + g1 / nu + g2 / nu**2 + g3 / nu**3 + g4 / nu**4


def block_bootstrap(
    values: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.mean,
    *,
    block: Optional[int] = None,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of `statistic` from a moving-block bootstrap (Künsch 1989).

    Each resample joins randomly chosen runs of `block` consecutive frames
    (default: 4 g, which covered best on autoregressive series) to the
    length of the series. For the mean prefer the t interval of
    :func:`series_statistics`: the bootstrap covers less when there are few
    effective frames.
    """
    x = _series(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")
    n = x.size
    b = block if block is not None else math.ceil(4.0 * statistical_inefficiency(x))
    b = max(1, min(int(b), n))
    k = math.ceil(n / b)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(resamples, k))
    runs = starts[:, :, None] + np.arange(b)[None, None, :]
    stats = np.array([statistic(x[r.ravel()[:n]]) for r in runs])
    lo, hi = np.quantile(stats, [(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0])
    return float(lo), float(hi)


@dataclass(frozen=True)
class SeriesStatistics:
    """Average of a series of correlated frames and its uncertainty (units of the series)."""

    mean: float
    sd: float
    """Standard deviation (N − 1 in the denominator)."""
    n: int
    """Number of frames."""
    inefficiency: float
    """Statistical inefficiency g: frames per independent sample."""
    sem: float
    """Standard error of the mean, SD·√(g/N)."""
    ci: tuple[float, float]
    """Confidence interval of the mean (Student's t, N_eff − 1 degrees of freedom); NaN below 3 effective frames."""
    confidence: float
    sem_blocking: float
    """Standard error from block averaging: the largest over levels with at least 16 blocks (an independent check)."""
    blocks: list[tuple[int, int, float]] = field(compare=False, repr=False)
    """The blocking curve, :func:`block_averages`."""

    @property
    def tau(self) -> float:
        """Integrated correlation time in frames, (g − 1)/2."""
        return 0.5 * (self.inefficiency - 1.0)

    @property
    def n_eff(self) -> float:
        """Effective number of independent frames, N/g."""
        return self.n / self.inefficiency

    @property
    def sem_independent(self) -> float:
        """SD/√N, the standard error if the frames were independent."""
        return self.sd / math.sqrt(self.n)

    @property
    def enough(self) -> bool:
        """At least 20 effective frames, so g and the error are themselves reliable."""
        return self.n_eff >= MIN_EFFECTIVE_FRAMES

    def frames_for(self, sem: float) -> int:
        """Frames needed for a standard error of `sem` at this SD and correlation: g·(SD/sem)²."""
        if not sem > 0.0:
            raise ValueError("the target standard error must be positive")
        return max(1, math.ceil(self.inefficiency * (self.sd / sem) ** 2))


def series_statistics(values: npt.ArrayLike, confidence: float = 0.95) -> SeriesStatistics:
    """Mean of a series of frames with its correlation-corrected uncertainty.

    Examples
    --------
    >>> stats = pbsolve.series_statistics([f.delta_solvation for f in frames])
    >>> print(f"{stats.mean:.2f} ± {stats.sem:.2f} kcal/mol, {stats.n_eff:.0f} independent frames")
    """
    x = _series(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")
    n = x.size
    g = statistical_inefficiency(x)
    sd = float(np.std(x, ddof=1))
    sem = sd * math.sqrt(g / n)
    mean = float(x.mean())
    nu = n / g - 1.0
    if nu >= 2.0:
        half = _t_quantile(0.5 + confidence / 2.0, nu) * sem
        ci = (float(mean - half), float(mean + half))
    else:
        ci = (math.nan, math.nan)
    blocks = block_averages(x)
    usable = [s for _, m, s in blocks if m >= 16]
    sem_blocking = max(usable) if usable else math.nan
    return SeriesStatistics(mean, sd, n, g, sem, ci, confidence, sem_blocking, blocks)
