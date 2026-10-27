"""Paired comparison of agents evaluated on the same tasks.

The unit of analysis is the *task*: for agents A and B with ``R`` runs per task, the
per-task difference ``d_i = mean_r A[i, r] - mean_r B[i, r]`` is one independent
observation. Treating every run as independent (pooling ``T * R`` pairs) inflates
the effective sample size and the type-I error; all tests here avoid that.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from agentaudit.results import ResultSet

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PairedComparison:
    """Outcome of comparing agent ``a`` against agent ``b`` (positive diff favours ``a``)."""

    a: str
    b: str
    diff: float
    ci_lo: float
    ci_hi: float
    p_bootstrap: float
    p_permutation: float
    p_mcnemar: float
    cohens_dz: float
    prob_superiority: float
    p_adjusted: float = float("nan")

    @property
    def significant_ci(self) -> bool:
        """True when the paired-bootstrap interval excludes zero."""
        return self.ci_lo > 0 or self.ci_hi < 0


def task_diffs(a: FloatArray, b: FloatArray) -> FloatArray:
    """Per-task mean differences of two ``(tasks, runs)`` matrices."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("agents must share the same tasks")
    out: FloatArray = np.nanmean(a, axis=1) - np.nanmean(b, axis=1)
    return out


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    ``only_a`` = tasks solved by A but not B, ``only_b`` = solved by B but not A. Under
    H0 each discordant pair is a fair coin, so the p-value is a two-sided binomial test.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))


def paired_bootstrap(
    diffs: ArrayLike,
    *,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap over tasks for the mean difference: ``(lo, hi, p_two_sided)``.

    The p-value uses the bootstrap distribution of the centred statistic
    (``mean(d*) - mean(d)``), i.e. how often a resampled mean is at least as extreme
    as zero after re-centring at the observed mean.
    """
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = (float(x) for x in np.quantile(means, [a, 1 - a]))
    obs = float(d.mean())
    shifted = means - obs
    p = (1 + np.sum(np.abs(shifted) >= abs(obs) - 1e-15)) / (n_boot + 1)
    return lo, hi, float(min(1.0, p))


def sign_flip_permutation(
    diffs: ArrayLike, *, n_perm: int = 20_000, seed: int = 0, exact_below: int = 14
) -> float:
    """Paired permutation (sign-flip) test based on ``|sum(d)|``.

    Exact (conditionally on ``|d|``) under the *strong* null that every ``d_t`` is symmetric about zero, which
    holds when the two agents are exchangeable within each task. It is not exact for the weaker null
    ``E[d] = 0`` with skewed ``d``; see :func:`studentized_sign_flip`. Zeros are dropped. Up to
    ``exact_below`` non-zero differences all ``2^n`` sign patterns are enumerated; beyond that the
    p-value is Monte Carlo, ``(1 + #{|S*| >= |S|}) / (n_perm + 1)`` with ``n_perm`` random sign patterns
    drawn from ``default_rng(seed)``.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 1.0
    obs = abs(d.sum())
    if n <= exact_below:
        signs = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        sums = np.abs(((2 * signs - 1) * d).sum(axis=1))
        return float(np.mean(sums >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    sums = np.abs((signs * d).sum(axis=1))
    return float((1 + np.sum(sums >= obs - 1e-12)) / (n_perm + 1))


def studentized_sign_flip(diffs: ArrayLike, *, n_perm: int = 10_000, seed: int = 0) -> float:
    """Sign-flip test of the paired *t* statistic (a reference method for the weak null).

    The plain sign-flip test of ``sum(d)`` is exact only when each ``d_t`` is symmetric about zero. Flipping
    signs of the studentised mean ``mean(d) / (sd(d) / sqrt(n))`` stays exact under symmetry and is
    asymptotically valid for the weaker null ``E[d] = 0`` with skewed ``d`` (Janssen, 1997).
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n < 2:
        return 1.0

    def tstat(x: FloatArray) -> FloatArray:
        sd = x.std(axis=-1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = x.mean(axis=-1) / (sd / np.sqrt(n))
        out: FloatArray = np.where(sd > 1e-12, t, 0.0)
        return out

    obs = abs(float(tstat(d)))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    t_perm = np.abs(tstat(signs * d))
    return float((1 + np.sum(t_perm >= obs - 1e-12)) / (n_perm + 1))


def prob_superiority(diffs: ArrayLike) -> float:
    """P(task-level A beats B), ties counted as one half (a paired common-language effect size)."""
    d = np.asarray(diffs, dtype=np.float64)
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def cohens_dz(diffs: ArrayLike) -> float:
    """Standardised paired effect size ``mean(d) / sd(d)`` (``nan`` if sd is zero)."""
    d = np.asarray(diffs, dtype=np.float64)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 1e-12 else float("nan")


def compare(
    a: FloatArray,
    b: FloatArray,
    *,
    names: tuple[str, str] = ("A", "B"),
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Full paired comparison of two ``(tasks, runs)`` score matrices."""
    d = task_diffs(a, b)
    lo, hi, p_boot = paired_bootstrap(d, level=level, n_boot=n_boot, seed=seed)
    solved_a = np.nanmean(a, axis=1) > 0.5
    solved_b = np.nanmean(b, axis=1) > 0.5
    return PairedComparison(
        a=names[0],
        b=names[1],
        diff=float(d.mean()),
        ci_lo=lo,
        ci_hi=hi,
        p_bootstrap=p_boot,
        p_permutation=sign_flip_permutation(d, seed=seed),
        p_mcnemar=mcnemar_exact(
            int(np.sum(solved_a & ~solved_b)), int(np.sum(solved_b & ~solved_a))
        ),
        cohens_dz=cohens_dz(d),
        prob_superiority=prob_superiority(d),
    )


def holm(pvalues: ArrayLike) -> FloatArray:
    """Holm-Bonferroni adjusted p-values (strong FWER control)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvalues: ArrayLike) -> FloatArray:
    """Benjamini-Hochberg adjusted p-values (FDR control under independence/PRDS)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rev_rank, i in enumerate(order):
        rank = m - rev_rank
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def pairwise(
    results: ResultSet,
    *,
    correction: str = "holm",
    n_boot: int = 5_000,
    seed: int = 0,
    test: str = "permutation",
) -> list[PairedComparison]:
    """All pairwise comparisons with multiplicity-adjusted p-values.

    ``test`` selects the p-value that is adjusted: ``"permutation"``, ``"bootstrap"``
    or ``"mcnemar"``.
    """
    key = {"permutation": "p_permutation", "bootstrap": "p_bootstrap", "mcnemar": "p_mcnemar"}
    if test not in key:
        raise ValueError(f"unknown test {test!r}")
    if correction not in {"holm", "bh", "none"}:
        raise ValueError(f"unknown correction {correction!r}")
    out = [
        compare(results.model(x), results.model(y), names=(x, y), n_boot=n_boot, seed=seed)
        for x, y in combinations(results.models, 2)
    ]
    if not out:
        return out
    raw = np.array([getattr(c, key[test]) for c in out])
    adj = (
        holm(raw)
        if correction == "holm"
        else benjamini_hochberg(raw)
        if correction == "bh"
        else raw
    )
    return [
        PairedComparison(**{**c.__dict__, "p_adjusted": float(q)})
        for c, q in zip(out, adj, strict=True)
    ]
