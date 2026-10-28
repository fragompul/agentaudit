"""Independent checks of the BCa, two-stage and beta-binomial profile intervals (final audit)."""

import numpy as np
import pytest
from scipy import optimize, stats

from agentaudit.intervals import (
    _bb_loglik,
    beta_binomial_hierarchical,
    bootstrap_ci,
    resample,
)


def mean_stat(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=(-2, -1))


def test_task_resampling_keeps_task_rows_intact() -> None:
    rng = np.random.default_rng(0)
    y = np.arange(20, dtype=float).reshape(5, 4)  # row identifies the task
    rep = resample(y, 50, "tasks", rng)
    rows = {tuple(r) for r in y}
    assert all(tuple(r) in rows for b in rep for r in b)


def test_two_stage_variance_counts_run_noise_twice() -> None:
    """Bootstrap variance of the mean: tasks -> V_m / T; two-stage adds W / (T R)."""
    rng = np.random.default_rng(1)
    t, r = 30, 5
    y = (rng.random((t, 1)) * 0.6 + 0.2 > rng.random((t, r))).astype(float)
    ybar = y.mean(axis=1)
    v_m = ybar.var()  # plug-in variance of task means
    w = np.mean(y.var(axis=1))  # mean plug-in within-task variance
    n = 40000
    v_tasks = resample(y, n, "tasks", rng).mean(axis=(1, 2)).var()
    v_two = resample(y, n, "two-stage", rng).mean(axis=(1, 2)).var()
    assert v_tasks == pytest.approx(v_m / t, rel=0.03)
    assert v_two == pytest.approx(v_m / t + w / (t * r), rel=0.03)
    assert v_two > v_tasks


def test_two_stage_unbalanced_designs_via_per_task_counts() -> None:
    # runs-only resampling keeps the task set fixed: variance is W / (T R) exactly
    rng = np.random.default_rng(2)
    y = (rng.random((12, 6)) < 0.4).astype(float)
    v = resample(y, 40000, "runs", rng).mean(axis=(1, 2)).var()
    assert v == pytest.approx(np.mean(y.var(axis=1)) / (12 * 6), rel=0.04)


def test_bca_matches_scipy_on_fixed_cases() -> None:
    rng = np.random.default_rng(3)
    for p in (0.2, 0.5):
        y = (rng.random((40, 1)) < p).astype(float)  # one run: tasks are exchangeable units
        ours = bootstrap_ci(y, mean_stat, n_boot=20000, method="bca", seed=5)
        ref = stats.bootstrap(
            (y[:, 0],), np.mean, n_resamples=20000, method="BCa", random_state=5
        ).confidence_interval
        assert ours.lo == pytest.approx(ref.low, abs=0.015)
        assert ours.hi == pytest.approx(ref.high, abs=0.015)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_bca_degenerate_samples(value: float) -> None:
    y = np.full((10, 3), value)
    ci = bootstrap_ci(y, mean_stat, n_boot=500, method="bca")
    assert ci.lo == ci.hi == value


def test_bca_small_and_discrete_samples_are_ordered_and_in_range() -> None:
    y = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    ci = bootstrap_ci(y, mean_stat, n_boot=2000, method="bca")
    assert 0 <= ci.lo <= ci.hi <= 1
    one_hit = np.zeros((25, 1))
    one_hit[0, 0] = 1.0
    ci = bootstrap_ci(one_hit, mean_stat, n_boot=2000, method="bca")
    assert ci.lo <= ci.estimate <= ci.hi


def test_bb_loglik_matches_scipy_betabinom_up_to_binomial_coefficient() -> None:
    c = np.array([0.0, 2.0, 5.0, 5.0])
    n = np.array([5.0, 5.0, 5.0, 10.0])
    mu, kappa = 0.4, 3.0
    ref = stats.betabinom.logpmf(c, n, mu * kappa, (1 - mu) * kappa)
    from scipy.special import gammaln

    coef = gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1)
    assert _bb_loglik(mu, kappa, c, n) == pytest.approx(float(np.sum(ref - coef)))


def test_profile_interval_reduces_to_binomial_likelihood_interval_for_one_run() -> None:
    c = np.array([1.0] * 12 + [0.0] * 28)
    n = np.ones(40)
    ci = beta_binomial_hierarchical(c, n)
    k, m = 12, 40

    def gap(p: float) -> float:
        ll = lambda q: k * np.log(q) + (m - k) * np.log(1 - q)  # noqa: E731
        return ll(k / m) - ll(p) - stats.chi2.ppf(0.95, 1) / 2

    lo = optimize.brentq(gap, 1e-6, k / m)
    hi = optimize.brentq(gap, k / m, 1 - 1e-6)
    assert ci.lo == pytest.approx(lo, abs=2e-3)
    assert ci.hi == pytest.approx(hi, abs=2e-3)


def test_profile_interval_boundary_solutions() -> None:
    zeros = beta_binomial_hierarchical(np.zeros(20), np.full(20, 5.0))
    assert zeros.lo <= 1e-5 and 0 < zeros.hi < 0.5
    ones = beta_binomial_hierarchical(np.full(20, 5.0), np.full(20, 5.0))
    assert ones.hi >= 1 - 1e-5 and 0.5 < ones.lo < 1


def test_profile_interval_is_wider_under_overdispersion() -> None:
    n = np.full(40, 10.0)
    homog = np.full(40, 5.0)
    hetero = np.array([0.0, 10.0] * 20)
    w_h = beta_binomial_hierarchical(homog, n)
    w_x = beta_binomial_hierarchical(hetero, n)
    assert (w_x.hi - w_x.lo) > (w_h.hi - w_h.lo)
