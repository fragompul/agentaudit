from math import comb

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentaudit.metrics import (
    mean_pass_at_k,
    mean_pass_hat_k,
    pass_at_k,
    pass_hat_k,
    success_rate,
    task_counts,
)


@given(st.integers(1, 30).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n))), st.data())
def test_matches_binomial_coefficients(nc: tuple[int, int], data: st.DataObject) -> None:
    n, c = nc
    k = data.draw(st.integers(1, n))
    assert pass_hat_k(c, n, k) == pytest.approx(comb(c, k) / comb(n, k))
    assert pass_at_k(c, n, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k))


@given(st.integers(2, 25), st.data())
def test_monotone_in_k_and_ordered(n: int, data: st.DataObject) -> None:
    c = data.draw(st.integers(0, n))
    ph = [float(pass_hat_k(c, n, k)) for k in range(1, n + 1)]
    pa = [float(pass_at_k(c, n, k)) for k in range(1, n + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(ph, ph[1:], strict=False))
    assert all(a <= b + 1e-12 for a, b in zip(pa, pa[1:], strict=False))
    assert all(h <= a + 1e-12 for h, a in zip(ph, pa, strict=True))
    assert ph[0] == pytest.approx(c / n)


def test_unbiasedness_by_simulation() -> None:
    rng = np.random.default_rng(0)
    p, n, k = 0.7, 8, 3
    c = rng.binomial(n, p, size=200_000)
    assert pass_hat_k(c, n, k).mean() == pytest.approx(p**k, abs=3e-3)
    assert pass_at_k(c, n, k).mean() == pytest.approx(1 - (1 - p) ** k, abs=3e-3)


def test_validation() -> None:
    with pytest.raises(ValueError):
        pass_at_k(3, 2, 1)
    with pytest.raises(ValueError):
        pass_hat_k(1, 2, 3)
    with pytest.raises(ValueError):
        pass_hat_k(1, 2, 0)
    with pytest.raises(ValueError):
        task_counts([1, 0])


def test_matrix_aggregates_with_missing() -> None:
    m = np.array([[1, 1, 0], [0, 0, np.nan]])
    c, n = task_counts(m)
    assert c.tolist() == [2, 0] and n.tolist() == [3, 2]
    assert success_rate(m) == pytest.approx(2 / 5)
    assert mean_pass_hat_k(m, 2) == pytest.approx((1 / 3 + 0) / 2)
    assert mean_pass_at_k(m, 2) == pytest.approx((1.0 + 0) / 2)
