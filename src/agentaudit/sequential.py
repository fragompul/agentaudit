"""Anytime-valid sequential evaluation via betting confidence sequences.

Tasks are evaluated one at a time (in random order). After each task we maintain a
confidence sequence (CS) for the mean of a ``[0, 1]``-valued quantity such that, with
probability at least ``1 - alpha``, *every* interval in the sequence contains the true
mean simultaneously. Hence we may inspect the CS after every task and stop as soon as
a decision is reached without inflating the error rate (Ville's inequality), unlike
repeatedly applying a fixed-sample interval ("peeking").

Construction: the hedged capital process of Waudby-Smith and Ramdas (2024,
JRSSB, "Estimating means of bounded random variables by betting") with the predictable
plug-in bet, evaluated on a grid of candidate means. The estimand is the mean of the
task distribution (tasks treated as i.i.d. draws, as in the rest of the library).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Decision = Literal["a_better", "b_better", "equivalent", "undecided"]


def confidence_sequence(
    x: ArrayLike, alpha: float = 0.05, *, grid_size: int = 2001, cap: float = 0.5
) -> tuple[FloatArray, FloatArray]:
    """Time-uniform ``(lo, hi)`` bounds for the mean of ``[0, 1]`` observations ``x``.

    Returns arrays of length ``len(x)``; the bounds are intersected over time so they are
    monotone (lo non-decreasing, hi non-increasing).
    """
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim != 1 or xs.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    if np.any(xs < 0) or np.any(xs > 1):
        raise ValueError("observations must lie in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = xs.size
    t = np.arange(1, n + 1)
    # predictable plug-in mean/variance estimates use data up to t-1 only
    prev_mu_all = (0.5 + np.cumsum(xs)) / (t + 1)
    sq = (xs - prev_mu_all) ** 2
    sigma2_prev = (0.25 + np.concatenate([[0.0], np.cumsum(sq)[:-1]])) / t
    lam = np.sqrt(2 * np.log(2 / alpha) / (sigma2_prev * t * np.log1p(t)))
    m = np.linspace(0.0, 1.0, grid_size)[1:-1]
    lam_p = np.minimum(lam[:, None], cap / m[None, :])
    lam_m = np.minimum(lam[:, None], cap / (1 - m)[None, :])
    inc = xs[:, None] - m[None, :]
    log_kp = np.cumsum(np.log1p(lam_p * inc), axis=0)
    log_km = np.cumsum(np.log1p(-lam_m * inc), axis=0)
    log_k = np.logaddexp(log_kp, log_km) - np.log(2.0)  # hedged: average of both bets
    keep = log_k < np.log(1 / alpha)
    lo = np.empty(n)
    hi = np.empty(n)
    cur_lo, cur_hi = 0.0, 1.0
    for i in range(n):
        idx = np.flatnonzero(keep[i])
        if idx.size:
            cur_lo = max(cur_lo, float(m[idx[0]]))
            cur_hi = min(cur_hi, float(m[idx[-1]]))
        lo[i], hi[i] = cur_lo, max(cur_hi, cur_lo)
    return lo, hi


@dataclass(frozen=True)
class SequentialResult:
    decision: Decision
    n_used: int
    diff_lo: float
    diff_hi: float
    stopped_early: bool


def sequential_compare(
    task_diffs: ArrayLike,
    *,
    alpha: float = 0.05,
    margin: float = 0.0,
    min_tasks: int = 10,
) -> SequentialResult:
    """Anytime-valid paired comparison over a stream of per-task score differences.

    ``task_diffs`` are values in ``[-1, 1]`` (mean score of A minus B on each task, in the
    order the tasks were evaluated; randomise the order beforehand). Stops at the first task
    ``n >= min_tasks`` where the CS for the mean difference excludes the margin band
    (``a_better`` / ``b_better``) or lies entirely inside ``[-margin, margin]`` (``equivalent``,
    only possible if ``margin > 0``). The overall error rate is controlled at ``alpha``.
    """
    d = np.asarray(task_diffs, dtype=np.float64)
    if np.any(d < -1) or np.any(d > 1):
        raise ValueError("task differences must lie in [-1, 1]")
    lo01, hi01 = confidence_sequence((d + 1) / 2, alpha)
    lo, hi = 2 * lo01 - 1, 2 * hi01 - 1
    for i in range(min_tasks - 1, d.size):
        if lo[i] > margin:
            return SequentialResult("a_better", i + 1, float(lo[i]), float(hi[i]), i + 1 < d.size)
        if hi[i] < -margin:
            return SequentialResult("b_better", i + 1, float(lo[i]), float(hi[i]), i + 1 < d.size)
        if margin > 0 and lo[i] >= -margin and hi[i] <= margin:
            return SequentialResult("equivalent", i + 1, float(lo[i]), float(hi[i]), i + 1 < d.size)
    return SequentialResult("undecided", d.size, float(lo[-1]), float(hi[-1]), False)


def sequential_precision(
    x: ArrayLike, *, half_width: float, alpha: float = 0.05, min_n: int = 10
) -> tuple[int, float, float]:
    """First ``n`` at which the CS half-width for a mean score is at most ``half_width``.

    Returns ``(n, lo, hi)``; ``n`` is ``len(x)`` if the target precision was never reached.
    """
    xs = np.asarray(x, dtype=np.float64)
    lo, hi = confidence_sequence(xs, alpha)
    ok = np.flatnonzero((hi - lo) / 2 <= half_width)
    ok = ok[ok >= min_n - 1]
    i = int(ok[0]) if ok.size else xs.size - 1
    return i + 1, float(lo[i]), float(hi[i])
