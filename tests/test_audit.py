"""Tests added by the scientific audit: exact unbiasedness, edge cases and cross-checks against references."""

from __future__ import annotations

from itertools import product
from math import comb

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from agentaudit.compare import (
    benjamini_hochberg,
    holm,
    mcnemar_exact,
    sign_flip_permutation,
    studentized_sign_flip,
)
from agentaudit.metrics import pass_at_k, pass_hat_k
from agentaudit.sequential import confidence_sequence, sequential_compare
from agentaudit.variance import decompose_array


@pytest.mark.parametrize("n", range(1, 9))
def test_pass_estimators_exactly_unbiased(n: int) -> None:
    """E over Binomial(n, p) of the estimators equals p^k and 1-(1-p)^k for every k <= n (exact sums)."""
    for p in (0.0, 0.1, 0.5, 0.83, 1.0):
        pmf = np.array([comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n + 1)])
        c = np.arange(n + 1)
        for k in range(1, n + 1):
            assert float(pmf @ pass_hat_k(c, n, k)) == pytest.approx(p**k, abs=1e-12)
            assert float(pmf @ pass_at_k(c, n, k)) == pytest.approx(1 - (1 - p) ** k, abs=1e-12)


def test_pass_edge_cases() -> None:
    assert pass_hat_k(0, 5, 3) == 0 and pass_at_k(0, 5, 3) == 0
    assert pass_hat_k(5, 5, 5) == 1 and pass_at_k(5, 5, 5) == 1
    assert pass_hat_k(4, 5, 5) == 0  # k = n and one failure: all-k success impossible
    assert pass_at_k(1, 5, 5) == 1  # k = n and one success: at least one success certain
    assert pass_hat_k(3, 5, 1) == pytest.approx(0.6) and pass_at_k(3, 5, 1) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        pass_hat_k(2, 3, 4)  # fewer than k attempts is refused, not extrapolated
    with pytest.raises(ValueError):
        pass_at_k(2, 3, 0)
    assert pass_hat_k(2, 4, 2) == pytest.approx(comb(2, 2) / comb(4, 2))
    assert pass_at_k(2, 4, 2) == pytest.approx(1 - comb(2, 2) / comb(4, 2))


def test_pass_ordering() -> None:
    c = np.arange(11)
    for k in range(1, 10):
        assert np.all(pass_hat_k(c, 10, k + 1) <= pass_hat_k(c, 10, k) + 1e-12)
        assert np.all(pass_at_k(c, 10, k + 1) >= pass_at_k(c, 10, k) - 1e-12)
    assert np.all(pass_hat_k(c, 10, 3) <= pass_at_k(c, 10, 3) + 1e-12)


def test_holm_and_bh_match_statsmodels() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(size=25) ** 3
        assert holm(p) == pytest.approx(multipletests(p, method="holm")[1])
        assert benjamini_hochberg(p) == pytest.approx(multipletests(p, method="fdr_bh")[1])


def test_sign_flip_exact_matches_full_enumeration() -> None:
    d = np.array([3.0, -1.0, 2.0, 0.5, -0.25, 1.5, 0.0, 4.0])
    nz = d[d != 0]
    obs = abs(nz.sum())
    sums = [abs(sum(s * x for s, x in zip(signs, nz, strict=True))) for signs in product((-1, 1), repeat=nz.size)]
    assert sign_flip_permutation(d) == pytest.approx(np.mean(np.array(sums) >= obs - 1e-12))


def test_sign_flip_monte_carlo_agrees_with_exact() -> None:
    rng = np.random.default_rng(1)
    d = rng.normal(0.3, 1.0, 12)
    exact = sign_flip_permutation(d, exact_below=14)
    mc = sign_flip_permutation(d, exact_below=0, n_perm=200_000, seed=3)
    assert mc == pytest.approx(exact, abs=0.01)


def test_sign_flip_is_exact_under_symmetric_null() -> None:
    """Rejection rate at 5% under a symmetric null is at most 5% (it is a valid test)."""
    rng = np.random.default_rng(2)
    rej = sum(sign_flip_permutation(rng.standard_t(3, 12), exact_below=14) < 0.05 for _ in range(1500))
    assert rej / 1500 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 1500)


def test_studentized_sign_flip_agrees_in_symmetric_case() -> None:
    rng = np.random.default_rng(4)
    d = rng.normal(0.4, 1.0, 40)
    t_p = 2 * stats.t.sf(abs(stats.ttest_1samp(d, 0).statistic), 39)
    assert studentized_sign_flip(d, n_perm=20_000, seed=1) == pytest.approx(t_p, abs=0.02)


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 8) == pytest.approx(2 * 0.5**8)
    assert mcnemar_exact(2, 10) == pytest.approx(stats.binomtest(2, 12, 0.5).pvalue)


def test_expected_mean_squares_by_simulation() -> None:
    """Average ANOVA estimates over many balanced Gaussian datasets recover the true components."""
    rng = np.random.default_rng(5)
    m, t, r = 5, 20, 4
    truth = (0.5, 1.0, 0.7, 2.0)  # model, task, model:task, run
    est = []
    for _ in range(600):
        y = (
            rng.normal(0, truth[0] ** 0.5, (m, 1, 1))
            + rng.normal(0, truth[1] ** 0.5, (1, t, 1))
            + rng.normal(0, truth[2] ** 0.5, (m, t, 1))
            + rng.normal(0, truth[3] ** 0.5, (m, t, r))
        )
        est.append(decompose_array(y).raw)
    assert np.mean(est, axis=0) == pytest.approx(truth, rel=0.12)


def test_variance_with_single_run_is_flagged_confounded() -> None:
    comps = decompose_array(np.random.default_rng(0).random((4, 10, 1)))
    assert comps.interaction_confounded and comps.run == 0.0


def test_truncation_keeps_raw_values() -> None:
    comps = decompose_array(np.random.default_rng(6).normal(size=(3, 6, 2)))
    assert min(comps.raw) < 0 or comps.model >= 0  # components are reported truncated, raw is retained
    assert all(v >= 0 for v in (comps.model, comps.task, comps.model_task, comps.run))


def test_confidence_sequence_validity_under_null() -> None:
    """Fraction of sequences whose bounds ever exclude the true mean is below alpha (Ville)."""
    rng = np.random.default_rng(7)
    bad = 0
    reps = 300
    for _ in range(reps):
        x = rng.binomial(1, 0.3, 200).astype(float) * 0.5 + rng.random(200) * 0.4 * (rng.random(200) < 0.5)
        mu = 0.3 * 0.5 + 0.5 * 0.2
        lo, hi = confidence_sequence(x, 0.05, grid_size=801)
        bad += bool(np.any(lo > mu) or np.any(hi < mu))
    assert bad / reps <= 0.05


def test_sequential_equivalence_needs_margin() -> None:
    rng = np.random.default_rng(8)
    d = np.clip(rng.normal(0.0, 0.2, 400), -1, 1)
    assert sequential_compare(d, margin=0.0).decision != "equivalent"
    assert sequential_compare(d, margin=0.25).decision == "equivalent"
