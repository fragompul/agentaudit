import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from statsmodels.stats.proportion import proportion_confint

from agentaudit.intervals import (
    beta_binomial,
    bootstrap_ci,
    clopper_pearson,
    resample,
    wilson,
)
from agentaudit.metrics import batch_pass_at_k, batch_pass_hat_k, batch_success

cases = st.integers(1, 200).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n)))


@given(cases, st.sampled_from([0.8, 0.9, 0.95, 0.99]))
def test_wilson_and_cp_match_statsmodels(sc: tuple[int, int], level: float) -> None:
    c, n = sc
    w, cp = wilson(c, n, level), clopper_pearson(c, n, level)
    ref_w = proportion_confint(c, n, alpha=1 - level, method="wilson")
    ref_cp = proportion_confint(c, n, alpha=1 - level, method="beta")
    assert (w.lo, w.hi) == pytest.approx(ref_w, abs=1e-9)
    assert (cp.lo, cp.hi) == pytest.approx(ref_cp, abs=1e-9)


@given(cases)
def test_intervals_contain_estimate_and_nest(sc: tuple[int, int]) -> None:
    c, n = sc
    for fn in (wilson, clopper_pearson):
        narrow, wide = fn(c, n, 0.8), fn(c, n, 0.99)
        assert 0 <= narrow.lo <= narrow.estimate <= narrow.hi <= 1
        assert wide.lo <= narrow.lo + 1e-12 and wide.hi >= narrow.hi - 1e-12


@given(cases)
def test_beta_binomial_properties(sc: tuple[int, int]) -> None:
    c, n = sc
    iv = beta_binomial(c, n)
    assert 0 <= iv.lo < iv.hi <= 1
    assert iv.lo <= iv.estimate <= iv.hi
    assert iv.estimate == pytest.approx((c + 0.5) / (n + 1))


def test_beta_binomial_uniform_prior_closed_form() -> None:
    iv = beta_binomial(0, 10, 0.95, prior=(1, 1))
    # posterior is Beta(1, 11), whose CDF is 1 - (1 - x)^11
    assert iv.hi == pytest.approx(1 - 0.025 ** (1 / 11))
    assert iv.lo == pytest.approx(1 - 0.975 ** (1 / 11))


def test_binomial_validation() -> None:
    for fn in (wilson, clopper_pearson, beta_binomial):
        with pytest.raises(ValueError):
            fn(5, 3)
        with pytest.raises(ValueError):
            fn(1, 3, 1.5)
    with pytest.raises(ValueError):
        beta_binomial(1, 3, prior=(0, 1))


def test_edge_counts() -> None:
    assert wilson(0, 10).lo == 0.0 and wilson(10, 10).hi == 1.0
    assert clopper_pearson(0, 10).lo == 0.0 and clopper_pearson(10, 10).hi == 1.0
    assert clopper_pearson(0, 10).hi == pytest.approx(1 - 0.025 ** (1 / 10))


def test_resample_shapes_and_semantics() -> None:
    rng = np.random.default_rng(0)
    m = np.arange(12, dtype=float).reshape(4, 3)
    assert resample(m, 5, "tasks", rng).shape == (5, 4, 3)
    runs = resample(m, 50, "runs", rng)
    # runs scheme keeps each row's values within that row
    for t in range(4):
        assert set(np.unique(runs[:, t, :])) <= set(m[t])
    tasks = resample(m, 50, "tasks", rng)
    # tasks scheme keeps whole rows intact
    assert all(tuple(row) in {tuple(r) for r in m} for rep in tasks for row in rep)
    assert resample(m, 5, "two-stage", rng).shape == (5, 4, 3)


def test_bootstrap_validation_and_determinism() -> None:
    m = (np.random.default_rng(1).random((30, 4)) < 0.6).astype(float)
    a = bootstrap_ci(m, batch_success, n_boot=300, seed=7)
    b = bootstrap_ci(m, batch_success, n_boot=300, seed=7)
    assert a == b
    with pytest.raises(ValueError):
        bootstrap_ci(m[0], batch_success)
    with pytest.raises(ValueError):
        bootstrap_ci(m, batch_success, level=1.2)
    with pytest.raises(ValueError):
        bootstrap_ci(m, batch_success, method="bca", scheme="runs")


def test_bootstrap_schemes_and_bca_cover_estimate() -> None:
    rng = np.random.default_rng(3)
    p_task = rng.beta(1.5, 1.0, size=60)
    m = (rng.random((60, 5)) < p_task[:, None]).astype(float)
    widths = {}
    for scheme in ("tasks", "runs", "two-stage"):
        iv = bootstrap_ci(m, batch_success, n_boot=500, scheme=scheme, seed=0)  # type: ignore[arg-type]
        assert iv.lo <= iv.estimate <= iv.hi
        widths[scheme] = iv.width
    # run-only resampling ignores task variance, so it must be the narrowest here
    assert widths["runs"] < widths["tasks"] < widths["two-stage"] * 1.05
    bca = bootstrap_ci(m, batch_success, n_boot=500, method="bca", seed=0)
    assert bca.lo < bca.estimate < bca.hi


@settings(max_examples=15, deadline=None)
@given(st.integers(2, 6), st.integers(0, 10_000))
def test_bootstrap_pass_stats_bounded(k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = (rng.random((25, 8)) < 0.5).astype(float)
    for stat in (batch_pass_hat_k(k), batch_pass_at_k(k)):
        iv = bootstrap_ci(m, stat, n_boot=100, seed=seed)
        assert 0 <= iv.lo <= iv.hi <= 1


def test_task_bootstrap_coverage_simulation() -> None:
    """The task bootstrap should cover the true mean near the nominal level."""
    rng = np.random.default_rng(11)
    hits, reps = 0, 150
    true_mean = 1.5 / 2.5
    for i in range(reps):
        p = rng.beta(1.5, 1.0, size=80)
        m = (rng.random((80, 5)) < p[:, None]).astype(float)
        hits += bootstrap_ci(m, batch_success, n_boot=300, seed=i).contains(true_mean)
    assert 0.88 <= hits / reps <= 0.995
