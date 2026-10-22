"""Confidence and credible intervals for success probabilities and agent scores.

Binomial intervals (Wilson, Clopper-Pearson, Beta-Binomial credible) treat all
attempts as exchangeable Bernoulli trials. They are valid for a *single task* or
for i.i.d. attempts, **not** for a benchmark score averaged over tasks with
repeated runs; for that use the bootstrap family, which respects the
tasks-by-runs structure (see :func:`bootstrap_ci`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
Statistic = Callable[[FloatArray], FloatArray]
Scheme = Literal["tasks", "runs", "two-stage"]
Method = Literal["percentile", "bca"]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval."""

    estimate: float
    lo: float
    hi: float
    level: float
    method: str

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


def _check(successes: int, trials: int, level: float) -> None:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials >= 1")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")


def wilson(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Wilson score interval (Wilson, 1927)."""
    _check(successes, trials, level)
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    lo = 0.0 if successes == 0 else max(0.0, float(centre - half))
    hi = 1.0 if successes == trials else min(1.0, float(centre + half))
    return Interval(p, lo, hi, level, "wilson")


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Exact (Clopper-Pearson, 1934) interval via Beta quantiles; conservative."""
    _check(successes, trials, level)
    a = (1 - level) / 2
    lo = 0.0 if successes == 0 else float(stats.beta.ppf(a, successes, trials - successes + 1))
    hi = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1 - a, successes + 1, trials - successes))
    )
    return Interval(successes / trials, lo, hi, level, "clopper-pearson")


def beta_binomial(
    successes: int,
    trials: int,
    level: float = 0.95,
    prior: tuple[float, float] = (0.5, 0.5),
) -> Interval:
    """Equal-tailed Bayesian credible interval with a Beta(a, b) prior (Jeffreys by default).

    The point estimate is the posterior mean.
    """
    _check(successes, trials, level)
    a0, b0 = prior
    if a0 <= 0 or b0 <= 0:
        raise ValueError("prior parameters must be positive")
    a, b = a0 + successes, b0 + trials - successes
    q = (1 - level) / 2
    lo, hi = (float(x) for x in stats.beta.ppf([q, 1 - q], a, b))
    return Interval(a / (a + b), lo, hi, level, f"beta-binomial({a0:g},{b0:g})")


def resample(
    outcomes: FloatArray, n_boot: int, scheme: Scheme, rng: np.random.Generator
) -> FloatArray:
    """Bootstrap replicates of a ``(tasks, runs)`` matrix, shape ``(n_boot, tasks, runs)``.

    * ``"tasks"``: resample whole tasks (rows) with replacement. This is the cluster
      bootstrap and treats the observed per-task run set as fixed.
    * ``"runs"``: resample runs within each task, keeping the task set fixed. It
      captures run noise only and *ignores* task sampling variance.
    * ``"two-stage"``: resample tasks, then runs within each drawn task. Known to
      overstate variance (run noise is counted twice); kept for comparison.
    """
    t, r = outcomes.shape
    if scheme == "runs":
        cols = rng.integers(0, r, size=(n_boot, t, r))
        rows = np.arange(t)[None, :, None]
        return outcomes[rows, cols]
    rows_idx = rng.integers(0, t, size=(n_boot, t))
    drawn = outcomes[rows_idx]
    if scheme == "tasks":
        return drawn
    cols = rng.integers(0, r, size=(n_boot, t, r))
    b = np.arange(n_boot)[:, None, None]
    ti = np.arange(t)[None, :, None]
    return drawn[b, ti, cols]


def bootstrap_ci(
    outcomes: ArrayLike,
    statistic: Statistic,
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    scheme: Scheme = "tasks",
    method: Method = "percentile",
    seed: int | np.random.Generator = 0,
) -> Interval:
    """Bootstrap interval for ``statistic`` of a ``(tasks, runs)`` outcome matrix.

    ``statistic`` maps an array of shape ``(..., tasks, runs)`` to ``(...)`` so
    that all replicates are evaluated in one vectorised call. BCa uses a jackknife
    over tasks for the acceleration and requires ``scheme="tasks"``.
    """
    arr = np.asarray(outcomes, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("outcomes must be 2-D (tasks, runs)")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    theta = float(statistic(arr))
    reps = np.asarray(statistic(resample(arr, n_boot, scheme, rng)), dtype=np.float64)
    a = (1 - level) / 2
    if method == "percentile":
        lo, hi = (float(x) for x in np.quantile(reps, [a, 1 - a]))
        return Interval(theta, lo, hi, level, f"bootstrap-{scheme}-percentile")
    if scheme != "tasks":
        raise ValueError("BCa is only implemented for scheme='tasks'")
    prop = (np.sum(reps < theta) + 0.5 * np.sum(reps == theta)) / n_boot
    z0 = float(stats.norm.ppf(np.clip(prop, 1 / (2 * n_boot), 1 - 1 / (2 * n_boot))))
    t = arr.shape[0]
    jack = np.array([float(statistic(np.delete(arr, i, axis=0))) for i in range(t)])
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d**2) ** 1.5)
    acc = float(np.sum(d**3) / denom) if denom > 0 else 0.0
    zq = stats.norm.ppf([a, 1 - a])
    adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
    lo, hi = (float(x) for x in np.quantile(reps, adj))
    return Interval(theta, lo, hi, level, "bootstrap-tasks-bca")
