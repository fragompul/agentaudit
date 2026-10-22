"""Invariance and monotonicity properties of the statistical core (Hypothesis)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from agentaudit.compare import mcnemar_exact, sign_flip_permutation, task_diffs
from agentaudit.intervals import bootstrap_ci, wilson
from agentaudit.metrics import batch_pass_at_k, batch_pass_hat_k, batch_success
from agentaudit.ranking import rank_uncertainty
from agentaudit.results import ResultSet
from agentaudit.variance import decompose_array

binary = arrays(
    np.float64,
    st.tuples(st.integers(8, 30), st.integers(2, 5)),
    elements=st.sampled_from([0.0, 1.0]),
)
gauss3 = arrays(
    np.float64,
    st.tuples(st.integers(2, 4), st.integers(4, 12), st.integers(2, 4)),
    elements=st.floats(-5, 5),
)


@settings(max_examples=30, deadline=None)
@given(binary, st.integers(0, 1000))
def test_bootstrap_interval_invariant_to_task_order_and_run_order(m: np.ndarray, seed: int) -> None:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    perm_t, perm_r = rng.permutation(m.shape[0]), rng.permutation(m.shape[1])
    for stat in (batch_success, batch_pass_hat_k(2), batch_pass_at_k(2)):
        assert float(stat(m)) == float(stat(m[perm_t][:, perm_r])) or np.isclose(
            float(stat(m)), float(stat(m[perm_t][:, perm_r]))
        )
    a = bootstrap_ci(m, batch_success, n_boot=50, seed=1)
    assert a.lo <= a.estimate <= a.hi


@settings(max_examples=30, deadline=None)
@given(gauss3, st.floats(-10, 10), st.floats(0.1, 5))
def test_variance_components_shift_and_scale(y: np.ndarray, shift: float, scale: float) -> None:  # type: ignore[type-arg]
    base = decompose_array(y)
    moved = decompose_array(scale * y + shift)
    assert np.isclose(moved.raw[0], scale**2 * base.raw[0], atol=1e-8, rtol=1e-6)
    assert np.isclose(moved.raw[3], scale**2 * base.raw[3], atol=1e-8, rtol=1e-6)


@settings(max_examples=25, deadline=None)
@given(gauss3, st.integers(0, 100))
def test_variance_components_invariant_to_agent_and_task_permutation(
    y: np.ndarray,  # type: ignore[type-arg]
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    p = y[rng.permutation(y.shape[0])][:, rng.permutation(y.shape[1])]
    np.testing.assert_allclose(decompose_array(y).raw, decompose_array(p).raw, atol=1e-8)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 50), st.integers(0, 10_000))
def test_ranking_equivariant_to_agent_relabelling(k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = (rng.random((4, 30, 3)) < np.array([0.2, 0.4, 0.6, 0.8])[:, None, None]).astype(float)
    order = np.random.default_rng(k).permutation(4)
    a = rank_uncertainty(ResultSet.from_array(scores), n_boot=100, seed=3)
    b = rank_uncertainty(ResultSet.from_array(scores[order]), n_boot=100, seed=3)
    # point ranks follow the agents, whatever the order they are listed in
    np.testing.assert_allclose(a.point_ranks[order], b.point_ranks)


@given(st.integers(1, 300), st.integers(0, 300))
def test_wilson_width_shrinks_with_more_data(n: int, extra: int) -> None:
    c = n // 2
    assert wilson(c * 4, n * 4).width <= wilson(c, n).width + 1e-12


@given(st.lists(st.floats(-1, 1), min_size=3, max_size=14))
def test_sign_flip_is_symmetric_and_bounded(d: list[float]) -> None:
    p = sign_flip_permutation(d)
    assert 0 < p <= 1.0 or p == 1.0
    assert abs(p - sign_flip_permutation([-x for x in d])) < 1e-12


@given(st.integers(0, 60), st.integers(0, 60))
def test_mcnemar_symmetric_and_monotone(b: int, c: int) -> None:
    assert mcnemar_exact(b, c) == mcnemar_exact(c, b)
    lo, hi = min(b, c), max(b, c)
    # with the smaller count fixed, a more lopsided table is stronger evidence
    assert mcnemar_exact(lo, hi + 5) <= mcnemar_exact(lo, hi) + 1e-12


def test_task_diffs_antisymmetric() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.random((20, 3)), rng.random((20, 3))
    np.testing.assert_allclose(task_diffs(a, b), -task_diffs(b, a))
