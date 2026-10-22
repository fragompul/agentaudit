import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from numpy.typing import NDArray

from agentaudit.results import ResultSet
from agentaudit.variance import decompose, decompose_array, decompose_single

F = NDArray[np.float64]


def gaussian(m: int, t: int, r: int, comps: tuple[float, float, float, float], seed: int) -> F:
    sm, st, smt, se = comps
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, np.sqrt(sm), (m, 1, 1))
        + rng.normal(0, np.sqrt(st), (1, t, 1))
        + rng.normal(0, np.sqrt(smt), (m, t, 1))
        + rng.normal(0, np.sqrt(se), (m, t, r))
    )


def test_recovers_components_on_average() -> None:
    true = (0.05, 0.30, 0.10, 0.20)
    est = np.mean(
        [
            [decompose_array(gaussian(6, 40, 4, true, s)).raw[i] for i in range(4)]
            for s in range(400)
        ],
        axis=0,
    )
    assert est == pytest.approx(true, abs=0.015)  # unbiased before truncation


def test_variance_of_difference_matches_simulation() -> None:
    true = (0.0, 0.3, 0.1, 0.2)
    t, r = 30, 3
    diffs = []
    for s in range(3000):
        y = gaussian(2, t, r, true, s)
        diffs.append(y[0].mean() - y[1].mean())
    comps = decompose_array(gaussian(40, 200, 6, true, 123))  # precise estimate
    assert comps.var_difference(t, r) == pytest.approx(np.var(diffs), rel=0.1)
    assert comps.var_mean(t, r) == pytest.approx(
        np.var([gaussian(1, t, r, true, s)[0].mean() for s in range(3000)]), rel=0.12
    )


def test_single_agent_matches_reml() -> None:
    rng = np.random.default_rng(2)
    t, r = 60, 5
    y = rng.normal(0, 0.5, (t, 1)) + rng.normal(0, 0.3, (t, r))
    sv = decompose_single(y)
    df = pd.DataFrame({"y": y.ravel(), "g": np.repeat(np.arange(t), r)})
    fit = smf.mixedlm("y ~ 1", df, groups=df["g"]).fit(reml=True)
    assert sv.task == pytest.approx(float(fit.cov_re.iloc[0, 0]), rel=1e-3)
    assert sv.run == pytest.approx(float(fit.scale), rel=1e-3)
    assert 0 < sv.icc < 1
    assert sv.var_mean(10, 2) == pytest.approx(sv.task / 10 + sv.run / 20)


def test_g_coefficient_increases_with_budget() -> None:
    c = decompose_array(gaussian(8, 50, 5, (0.1, 0.3, 0.1, 0.3), 5))
    assert c.g_coefficient(50, 5) > c.g_coefficient(10, 1)
    assert 0 <= c.g_coefficient() <= 1
    assert sum(c.shares().values()) == pytest.approx(1.0)


def test_confounding_flag_and_validation() -> None:
    c = decompose_array(gaussian(4, 20, 1, (0.1, 0.2, 0.1, 0.1), 0))
    assert c.interaction_confounded and c.run == 0.0
    with pytest.raises(ValueError):
        decompose_array(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        decompose_array(np.zeros((1, 3, 2)))
    with pytest.raises(ValueError):
        decompose_single(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        decompose_single(np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        decompose_array(np.full((2, 2, 2), np.nan))


def test_resultset_wrapper_and_zero_variance() -> None:
    rs = ResultSet.from_array(np.ones((3, 5, 4)))
    c = decompose(rs)
    assert c.total == 0 and c.shares()["task"] == 0.0
    assert np.isnan(c.g_coefficient())
