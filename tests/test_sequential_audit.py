"""Independent checks of the betting confidence sequence (final audit, section A)."""

from itertools import product

import numpy as np

from agentaudit.sequential import confidence_sequence, log_capital, sequential_compare


def _reference_log_capital(x: list[float], m: float, alpha: float, cap: float = 0.5) -> list[float]:
    """Plain-Python re-derivation of the hedged capital process at a single candidate mean."""
    kp = km = 1.0
    out = []
    mu_hat_sum, sq_sum = 0.5, 0.0
    for i, xi in enumerate(x, start=1):
        sigma2 = (0.25 + sq_sum) / i  # uses x_1..x_{i-1} only
        lam = np.sqrt(2 * np.log(2 / alpha) / (sigma2 * i * np.log(1 + i)))
        kp *= 1 + min(lam, cap / m) * (xi - m)
        km *= 1 - min(lam, cap / (1 - m)) * (xi - m)
        out.append(float(np.log((kp + km) / 2)))
        mu_hat_sum += xi
        sq_sum += (xi - mu_hat_sum / (i + 1)) ** 2
    return out


def test_vectorised_capital_matches_reference() -> None:
    rng = np.random.default_rng(0)
    x = rng.random(25)
    grid, logk = log_capital(x, 0.05, grid_size=101)
    for j in (3, 40, 77):
        ref = _reference_log_capital(list(x), float(grid[j + 1]), 0.05)
        assert np.allclose(logk[:, j], ref, atol=1e-10)


def test_capital_is_nonnegative_for_extreme_observations() -> None:
    for x in (np.zeros(50), np.ones(50), np.tile([0.0, 1.0], 25), np.tile([1.0, 0.0], 25)):
        _, logk = log_capital(x, 0.05, grid_size=201)
        assert np.isfinite(logk).all()


def test_capital_is_a_martingale_by_exact_enumeration() -> None:
    """E[K_n(m)] = 1 when E x = m, summing over every path of a 3-point law."""
    support = np.array([0.0, 0.5, 1.0])
    for m, probs in ((0.3, [0.5, 0.4, 0.1]), (0.5, [0.25, 0.5, 0.25]), (0.6, [0.2, 0.4, 0.4])):
        assert abs(float(np.dot(support, probs)) - m) < 1e-12
        total = 0.0
        for path in product(range(3), repeat=6):
            p = float(np.prod([probs[i] for i in path]))
            total += p * np.exp(_reference_log_capital([support[i] for i in path], m, 0.05)[-1])
        assert abs(total - 1.0) < 1e-9


def test_bets_are_predictable_prefix_invariance() -> None:
    rng = np.random.default_rng(1)
    x = rng.random(60)
    lo, hi = confidence_sequence(x, 0.05, grid_size=201)
    y = x.copy()
    y[30:] = 1.0 - y[30:]
    lo2, hi2 = confidence_sequence(y, 0.05, grid_size=201)
    assert np.array_equal(lo[:30], lo2[:30]) and np.array_equal(hi[:30], hi2[:30])


def test_grid_refinement_changes_bounds_by_at_most_grid_cells() -> None:
    x = (np.random.default_rng(2).random(200) < 0.4).astype(float)
    lo_c, hi_c = confidence_sequence(x, grid_size=401)
    lo_f, hi_f = confidence_sequence(x, grid_size=4001)
    assert np.max(np.abs(lo_c - lo_f)) <= 2 / 400 and np.max(np.abs(hi_c - hi_f)) <= 2 / 400
    assert (lo_c <= lo_f + 1e-12).all() or np.max(lo_f - lo_c) <= 1 / 400


def test_equivalence_requires_whole_sequence_inside_band() -> None:
    # point estimate 0.04 sits inside a 0.05 band but the sequence is still wide: undecided
    d = np.array([0.04] * 12)
    d[::2] = 0.9
    d[1::2] = -0.82
    res = sequential_compare(d, margin=0.05)
    assert res.decision == "undecided"
    assert not (res.diff_lo >= -0.05 and res.diff_hi <= 0.05)


def test_false_equivalence_requires_true_diff_outside_band_to_be_excluded() -> None:
    # true mean 0.15 (outside margin 0.10): equivalence needs the CS to exclude 0.15 at some time
    rng = np.random.default_rng(3)
    wrong = 0
    reps = 300
    for _ in range(reps):
        d = rng.choice([-1.0, 0.0, 1.0], p=[0.1, 0.65, 0.25], size=250)  # mean 0.15
        wrong += sequential_compare(d, margin=0.10).decision == "equivalent"
    assert wrong / reps <= 0.05
