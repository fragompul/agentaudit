import numpy as np
import pytest

from agentaudit.sequential import confidence_sequence, sequential_compare, sequential_precision


def test_cs_shrinks_and_is_monotone() -> None:
    x = (np.random.default_rng(0).random(400) < 0.6).astype(float)
    lo, hi = confidence_sequence(x)
    assert (np.diff(lo) >= -1e-12).all() and (np.diff(hi) <= 1e-12).all()
    assert hi[-1] - lo[-1] < hi[20] - lo[20]
    assert lo[-1] < 0.6 < hi[-1]


def test_time_uniform_coverage() -> None:
    """The CS must contain the truth at *all* times with prob >= 1 - alpha (peeking is safe)."""
    rng = np.random.default_rng(1)
    miss = 0
    reps = 300
    for _ in range(reps):
        x = (rng.random(150) < 0.3).astype(float)
        lo, hi = confidence_sequence(x, 0.1, grid_size=401)
        miss += bool(np.any((lo > 0.3) | (hi < 0.3)))
    assert miss / reps <= 0.1


def test_sequential_type_i_error_and_power() -> None:
    rng = np.random.default_rng(2)
    false_pos = 0
    reps = 200
    for _ in range(reps):
        d = rng.choice([-1.0, 0.0, 1.0], p=[0.15, 0.7, 0.15], size=200)
        false_pos += sequential_compare(d, alpha=0.1).decision in ("a_better", "b_better")
    assert false_pos / reps <= 0.1
    d = rng.choice([-1.0, 0.0, 1.0], p=[0.05, 0.6, 0.35], size=400)
    res = sequential_compare(d)
    assert res.decision == "a_better" and res.stopped_early and res.n_used < 400
    assert res.diff_lo > 0


def test_equivalence_and_undecided() -> None:
    d = np.zeros(300)
    assert sequential_compare(d, margin=0.2).decision == "equivalent"
    assert sequential_compare(np.array([1.0, -1.0] * 5)).decision == "undecided"
    flipped = sequential_compare(-np.ones(60))
    assert flipped.decision == "b_better"


def test_precision_stop_and_validation() -> None:
    x = (np.random.default_rng(3).random(2000) < 0.5).astype(float)
    n, lo, hi = sequential_precision(x, half_width=0.1)
    assert (hi - lo) / 2 <= 0.1 and n < 2000
    n2, _, _ = sequential_precision(x[:15], half_width=0.001)
    assert n2 == 15
    for bad in (np.array([]), np.array([[0.1]]), np.array([1.5])):
        with pytest.raises(ValueError):
            confidence_sequence(bad)
    with pytest.raises(ValueError):
        confidence_sequence([0.5], alpha=1.0)
    with pytest.raises(ValueError):
        sequential_compare([2.0])
