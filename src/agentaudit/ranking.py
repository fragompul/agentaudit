"""Uncertainty of agent rankings.

The same resampled task set is applied to every agent in each bootstrap replicate,
preserving the pairing that makes leaderboard comparisons far more precise than
comparing marginal intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from agentaudit.compare import pairwise
from agentaudit.results import ResultSet

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RankingUncertainty:
    """Bootstrap distribution of ranks (rank 1 = best score)."""

    models: tuple[str, ...]
    point_ranks: FloatArray  # (M,)
    rank_probs: FloatArray  # (M, M): P(model i has rank j + 1)
    rank_lo: FloatArray
    rank_hi: FloatArray
    prob_better: FloatArray  # (M, M): P(score_i > score_j) across replicates

    def prob_top_k(self, k: int) -> FloatArray:
        """P(model is among the ``k`` best), per model."""
        out: FloatArray = self.rank_probs[:, :k].sum(axis=1)
        return out

    def table(self) -> list[dict[str, float | str]]:
        order = np.argsort(self.point_ranks)
        return [
            {
                "model": self.models[i],
                "rank": float(self.point_ranks[i]),
                "rank_lo": float(self.rank_lo[i]),
                "rank_hi": float(self.rank_hi[i]),
                "p_best": float(self.rank_probs[i, 0]),
            }
            for i in order
        ]


def _ranks_desc(scores: FloatArray) -> FloatArray:
    """Rank 1 = highest score, ties get the average rank (last axis)."""
    out: FloatArray = stats.rankdata(-scores, axis=-1, method="average")
    return out


def rank_uncertainty(
    results: ResultSet, *, n_boot: int = 4000, level: float = 0.95, seed: int = 0
) -> RankingUncertainty:
    """Bootstrap the task set to get the distribution of every agent's rank."""
    if results.n_models < 2:
        raise ValueError("need at least two models")
    task_means = results.task_means()  # (M, T)
    m, t = task_means.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t, size=(n_boot, t))
    boot_scores = task_means[:, idx].mean(axis=2).T  # (B, M)
    # break exact ties at random so each replicate assigns every integer rank exactly once
    ranks = _ranks_desc(boot_scores + rng.random(boot_scores.shape) * 1e-9)
    rank_probs = np.stack([np.mean(ranks == j + 1, axis=0) for j in range(m)], axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(ranks, [a, 1 - a], axis=0)
    wins = boot_scores[:, :, None] > boot_scores[:, None, :]
    ties = boot_scores[:, :, None] == boot_scores[:, None, :]
    prob_better = (wins + 0.5 * ties).mean(axis=0)
    return RankingUncertainty(
        results.models,
        _ranks_desc(task_means.mean(axis=1)),
        rank_probs,
        lo,
        hi,
        prob_better,
    )


def indistinguishable_matrix(
    results: ResultSet, *, alpha: float = 0.05, correction: str = "holm", seed: int = 0
) -> NDArray[np.bool_]:
    """Symmetric boolean matrix: ``True`` where two agents are statistically indistinguishable.

    Two agents are indistinguishable when the multiplicity-adjusted paired
    sign-flip test fails to reject at level ``alpha``. The diagonal is ``True``.
    """
    names = list(results.models)
    out = np.eye(len(names), dtype=bool)
    for c in pairwise(results, correction=correction, n_boot=200, seed=seed):
        i, j = names.index(c.a), names.index(c.b)
        out[i, j] = out[j, i] = not c.p_adjusted < alpha
    return out


def fraction_indistinguishable(results: ResultSet, *, alpha: float = 0.05) -> float:
    """Fraction of agent pairs that cannot be told apart at level ``alpha`` (Holm-adjusted)."""
    mat = indistinguishable_matrix(results, alpha=alpha)
    iu = np.triu_indices(mat.shape[0], k=1)
    return float(mat[iu].mean())


def split_half_rank_correlation(
    results: ResultSet, *, n_splits: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Mean and 5th percentile of Kendall's tau between rankings from disjoint task halves.

    A direct measure of ranking *reliability*: values near 1 mean a fresh task sample would
    reproduce the ordering, values near 0 mean the leaderboard order is mostly noise.
    """
    tm = results.task_means()
    t = tm.shape[1]
    if t < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_splits):
        perm = rng.permutation(t)
        a, b = perm[: t // 2], perm[t // 2 : 2 * (t // 2)]
        tau = stats.kendalltau(tm[:, a].mean(axis=1), tm[:, b].mean(axis=1)).statistic
        if not np.isnan(tau):
            taus.append(tau)
    if not taus:
        return float("nan"), float("nan")
    return float(np.mean(taus)), float(np.quantile(taus, 0.05))
