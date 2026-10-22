"""Point metrics for repeated-run agent evaluation.

Notation: a task ``i`` was attempted ``n_i`` times with ``c_i`` successes.
``pass@k`` (Chen et al., 2021) is the probability that at least one of ``k``
attempts drawn without replacement from the ``n`` observed ones succeeds;
``pass^k`` (Yao et al., 2024, tau-bench) is the probability that all ``k`` succeed.
Both estimators below are unbiased for their population counterparts
``1 - (1 - p)^k`` and ``p^k`` when attempts are i.i.d. Bernoulli(p).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _counts(successes: ArrayLike, attempts: ArrayLike) -> tuple[FloatArray, FloatArray]:
    c = np.asarray(successes, dtype=np.float64)
    n = np.asarray(attempts, dtype=np.float64)
    c, n = np.broadcast_arrays(c, n)
    if np.any(n < 1) or np.any(c < 0) or np.any(c > n):
        raise ValueError("need 0 <= successes <= attempts and attempts >= 1")
    return c, n


def _check_k(n: FloatArray, k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if np.any(n < k):
        raise ValueError("k cannot exceed the number of attempts")


def pass_hat_k(successes: ArrayLike, attempts: ArrayLike, k: int) -> FloatArray:
    """Unbiased per-task estimate of ``p^k`` (all ``k`` attempts succeed).

    Equals ``C(c, k) / C(n, k)``, computed as a stable product. Requires ``k <= n``.
    """
    c, n = _counts(successes, attempts)
    _check_k(n, k)
    out = np.ones_like(c)
    for j in range(k):
        out = out * np.clip(c - j, 0.0, None) / (n - j)
    return out


def pass_at_k(successes: ArrayLike, attempts: ArrayLike, k: int) -> FloatArray:
    """Unbiased per-task estimate of ``1 - (1 - p)^k`` (at least one of ``k`` succeeds).

    Equals ``1 - C(n - c, k) / C(n, k)`` (Chen et al., 2021). Requires ``k <= n``.
    """
    c, n = _counts(successes, attempts)
    _check_k(n, k)
    miss = np.ones_like(c)
    for j in range(k):
        miss = miss * np.clip(n - c - j, 0.0, None) / (n - j)
    return 1.0 - miss


def success_rate(outcomes: ArrayLike) -> float:
    """Mean success over a ``(tasks, runs)`` matrix (NaN = missing, ignored)."""
    arr = np.asarray(outcomes, dtype=np.float64)
    return float(np.nanmean(arr))


def task_counts(outcomes: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Per-task ``(successes, attempts)`` from a ``(tasks, runs)`` matrix; NaN = missing."""
    arr = np.asarray(outcomes, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("outcomes must be 2-D (tasks, runs)")
    attempts = np.sum(~np.isnan(arr), axis=1).astype(np.float64)
    successes = np.nansum(arr, axis=1)
    return successes, attempts


def mean_pass_at_k(outcomes: ArrayLike, k: int) -> float:
    """Benchmark-level pass@k: mean over tasks of the unbiased per-task estimate."""
    c, n = task_counts(outcomes)
    return float(np.mean(pass_at_k(c, n, k)))


def mean_pass_hat_k(outcomes: ArrayLike, k: int) -> float:
    """Benchmark-level pass^k: mean over tasks of the unbiased per-task estimate."""
    c, n = task_counts(outcomes)
    return float(np.mean(pass_hat_k(c, n, k)))
