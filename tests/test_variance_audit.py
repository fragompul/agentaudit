"""Independent checks of the variance-component and allocation formulas (final audit, section E)."""

from math import sqrt

import numpy as np

from agentaudit.power import DiffVariance, optimal_runs
from agentaudit.variance import decompose_array


def _simulate(rng, m, t, r, s_m, s_t, s_mt, s_e):
    a = rng.normal(0, sqrt(s_m), (m, 1, 1))
    b = rng.normal(0, sqrt(s_t), (1, t, 1))
    ab = rng.normal(0, sqrt(s_mt), (m, t, 1))
    e = rng.normal(0, sqrt(s_e), (m, t, r))
    return a + b + ab + e


def test_anova_estimators_are_unbiased_before_truncation() -> None:
    rng = np.random.default_rng(0)
    truth = (0.05, 0.30, 0.10, 0.40)
    raws = np.array([decompose_array(_simulate(rng, 5, 30, 4, *truth)).raw for _ in range(3000)])
    se = raws.std(axis=0, ddof=1) / sqrt(len(raws))
    assert np.all(np.abs(raws.mean(axis=0) - truth) < 4 * se)


def test_difference_variance_formula_matches_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    _, s_t, s_mt, s_e = 0.0, 0.3, 0.1, 0.4
    t, r = 25, 3
    diffs = []
    for _ in range(20000):
        y = _simulate(rng, 2, t, r, 1e-12, s_t, s_mt, s_e)
        diffs.append(y[0].mean() - y[1].mean())
    formula = 2 * (s_mt / t + s_e / (t * r))
    assert abs(np.var(diffs, ddof=1) / formula - 1) < 0.04


def test_negative_components_are_truncated_but_raw_values_are_kept() -> None:
    rng = np.random.default_rng(2)
    seen_negative = False
    for _ in range(200):
        c = decompose_array(_simulate(rng, 3, 10, 2, 0.0, 0.0, 0.0, 1.0))
        assert all(v >= 0 for v in (c.model, c.task, c.model_task, c.run))
        seen_negative |= min(c.raw[:3]) < 0
    assert seen_negative  # with null components the raw ANOVA estimate is often negative


def test_optimal_runs_matches_brute_force_integer_search() -> None:
    for s_t in (0.01, 0.1, 0.5, 2.0):
        for s_e in (0.05, 0.5, 3.0):
            for c_task in (0.0, 1.0, 10.0, 250.0):
                dv = DiffVariance(s_t, s_e)

                def f(r: int, s_t: float = s_t, s_e: float = s_e, c_task: float = c_task) -> float:
                    return (s_t + s_e / r) * (c_task + r * 1.0)

                best = min(range(1, 5000), key=lambda r: (f(r), r))
                got = optimal_runs(dv, c_task, 1.0)
                assert abs(f(got) - f(best)) <= 1e-12 * f(best), (s_t, s_e, c_task, got, best)


def test_optimal_runs_boundaries() -> None:
    assert optimal_runs(DiffVariance(0.2, 0.0), 10.0) == 1  # no run noise
    assert optimal_runs(DiffVariance(0.2, 1.0), 0.0) == 1  # tasks free
    assert optimal_runs(DiffVariance(0.0, 1.0), 10.0) >= 10**6  # no interaction: unbounded
    assert optimal_runs(DiffVariance(1e-9, 1.0), 10.0) > 1000  # near-zero component: huge R*
