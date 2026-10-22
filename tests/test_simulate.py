import numpy as np
import pytest

from agentaudit.simulate import SyntheticPopulation


def test_quadrature_matches_monte_carlo() -> None:
    pop = SyntheticPopulation((0.0, 1.0), sigma_task=1.2, sigma_interaction=0.4)
    rng = np.random.default_rng(0)
    p = pop.sample_probs(400_000, rng)
    assert pop.true_mean(1) == pytest.approx(p[1].mean(), abs=2e-3)
    assert pop.true_pass_hat_k(0, 3) == pytest.approx((p[0] ** 3).mean(), abs=2e-3)
    assert pop.true_pass_at_k(0, 3) == pytest.approx((1 - (1 - p[0]) ** 3).mean(), abs=2e-3)
    assert pop.true_diff(1, 0) > 0


def test_symmetric_agent_has_half_mean() -> None:
    assert SyntheticPopulation((0.0,)).true_mean(0) == pytest.approx(0.5, abs=1e-9)


def test_sample_is_seeded_and_shaped() -> None:
    pop = SyntheticPopulation((0.0, 0.5, 1.0), names=("a", "b", "c"))
    r1, r2 = pop.sample(30, 4, 7), pop.sample(30, 4, 7)
    np.testing.assert_array_equal(r1.scores, r2.scores)
    assert r1.models == ("a", "b", "c") and r1.scores.shape == (3, 30, 4)
    assert set(np.unique(r1.scores)) <= {0.0, 1.0}
    assert SyntheticPopulation((0.0, 1.0)).model_names == ("agent0", "agent1")
