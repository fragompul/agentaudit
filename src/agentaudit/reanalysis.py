"""Re-analysis of public single-run leaderboards (one binary outcome per agent and task).

With one run per task the paired design reduces to McNemar's exact test on the tasks where
two agents disagree. These helpers quantify how many published pairwise differences are
statistically distinguishable once multiplicity is accounted for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from agentaudit.compare import benjamini_hochberg, holm

BoolArray = NDArray[np.bool_]


def mcnemar_matrix(x: BoolArray) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """All-pairs exact McNemar p-values for an ``(agents, tasks)`` boolean matrix.

    Returns ``(p, only_a)`` where ``only_a[i, j]`` is the number of tasks solved by ``i`` but not
    ``j``. The diagonal p-value is 1.
    """
    xi = x.astype(np.int64)
    only = xi @ (1 - xi).T  # only[i, j] = solved by i and not by j
    n = only + only.T
    k = np.minimum(only, only.T)
    with np.errstate(invalid="ignore"):
        p = np.where(n > 0, np.minimum(1.0, 2 * stats.binom.cdf(k, np.maximum(n, 1), 0.5)), 1.0)
    np.fill_diagonal(p, 1.0)
    return p.astype(np.float64), only


def reanalyze_single_run(
    x: BoolArray, names: list[str], *, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Summary of pairwise distinguishability and rank uncertainty for a single-run leaderboard."""
    m, t = x.shape
    score = x.mean(axis=1)
    order = np.argsort(-score, kind="stable")
    p, _ = mcnemar_matrix(x)
    iu = np.triu_indices(m, k=1)
    pv = p[iu]
    gaps = np.abs(score[iu[0]] - score[iu[1]])
    holm_p, bh_p = holm(pv), benjamini_hochberg(pv)
    sig_raw, sig_holm, sig_bh = pv < alpha, holm_p < alpha, bh_p < alpha

    bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    by_gap = []
    for lo, hi in bins:
        sel = (gaps >= lo) & (gaps < hi)
        by_gap.append(
            {
                "gap_lo": lo,
                "gap_hi": min(hi, 1.0),
                "pairs": int(sel.sum()),
                "frac_distinguishable_holm": float(sig_holm[sel].mean()) if sel.any() else None,
                "frac_distinguishable_raw": float(sig_raw[sel].mean()) if sel.any() else None,
            }
        )
    # neighbours on the leaderboard (what a "state of the art" claim compares)
    adj = [(int(order[i]), int(order[i + 1])) for i in range(m - 1)]
    adj_p = np.array([p[a, b] for a, b in adj])
    adj_holm = holm(adj_p) if len(adj_p) else adj_p
    # Holm over *all* pairs, then read off the adjacent ones: the adjacent family is chosen after
    # seeing the scores, so only this version has a family fixed before looking at the data.
    pos = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu[0], iu[1], strict=True))}
    adj_all = np.array([holm_p[pos[(min(a, b), max(a, b))]] for a, b in adj])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t, size=(n_boot, t))
    boot = x[:, idx].mean(axis=2).T  # (B, M)
    jitter = rng.random(boot.shape) * 1e-9
    ranks = stats.rankdata(-(boot + jitter), axis=1)
    top = order[: min(10, m)]
    return {
        "n_agents": m,
        "n_tasks": t,
        "alpha": alpha,
        "n_pairs": int(pv.size),
        "frac_distinguishable_raw": float(sig_raw.mean()),
        "frac_distinguishable_holm": float(sig_holm.mean()),
        "frac_distinguishable_bh": float(sig_bh.mean()),
        "by_score_gap": by_gap,
        "adjacent_pairs": {
            "n": len(adj),
            "frac_distinguishable_holm_within_adjacent": float(np.mean(adj_holm < alpha)),
            "frac_distinguishable_raw": float(np.mean(adj_p < alpha)),
            "n_distinguishable_holm_all_pairs": int(np.sum(adj_all < alpha)),
            "n_distinguishable_raw": int(np.sum(adj_p < alpha)),
        },
        "score_range": [float(np.min(score)), float(np.max(score))],
        "top10": [
            {
                "name": names[i],
                "score": float(score[i]),
                "rank_lo": float(np.quantile(ranks[:, i], 0.025)),
                "rank_hi": float(np.quantile(ranks[:, i], 0.975)),
                "p_best": float(np.mean(ranks[:, i] == 1)),
            }
            for i in top
        ],
        "median_gap_needed": float(np.median(gaps[sig_holm])) if sig_holm.any() else None,
        "min_gap_distinguishable": float(np.min(gaps[sig_holm])) if sig_holm.any() else None,
    }
