import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentaudit.power import (
    DiffVariance,
    mde,
    optimal_runs,
    plan,
    power,
    required_tasks,
    shift_probs,
    simulated_power,
)
from agentaudit.variance import decompose_array

dvs = st.builds(DiffVariance, st.floats(0.001, 0.5), st.floats(0.001, 0.5))


@settings(deadline=None, max_examples=40)
@given(dvs, st.floats(0.02, 0.3), st.integers(1, 10))
def test_required_tasks_is_minimal(dv: DiffVariance, delta: float, r: int) -> None:
    t = required_tasks(delta, r, dv)
    assert power(delta, t, r, dv) >= 0.8
    if t > 2:
        assert power(delta, t - 1, r, dv) < 0.8


@given(dvs, st.integers(5, 200), st.integers(1, 10))
def test_power_monotone(dv: DiffVariance, t: int, r: int) -> None:
    assert power(0.05, t, r, dv) <= power(0.1, t, r, dv) + 1e-12
    assert power(0.05, t, r, dv) <= power(0.05, t + 20, r, dv) + 1e-12
    assert power(0.05, t, r, dv) <= power(0.05, t, r + 3, dv) + 1e-12
    assert power(0.0, t, r, dv) == pytest.approx(0.05, abs=1e-9)


def test_mde_inverts_power() -> None:
    dv = DiffVariance(0.1, 0.2)
    d = mde(100, 3, dv)
    assert power(d, 100, 3, dv) == pytest.approx(0.8, abs=1e-4)


def test_fixed_attempt_budget_prefers_more_tasks() -> None:
    dv = DiffVariance(0.05, 0.2)
    rows = plan(0.05, dv, runs_grid=(1, 2, 5, 10))
    attempts = [r.total_attempts for r in rows]
    assert attempts == sorted(attempts)  # more runs per task => more attempts needed in total
    assert rows[0].n_tasks > rows[-1].n_tasks  # but fewer tasks


def test_optimal_runs_closed_form_and_edges() -> None:
    dv = DiffVariance(0.04, 0.16)
    # R* = sqrt(0.16 * 9 / (0.04 * 1)) = 6
    assert optimal_runs(dv, cost_task=9, cost_run=1) == 6
    assert optimal_runs(dv, cost_task=0, cost_run=1) == 1
    assert optimal_runs(DiffVariance(0.0, 0.1)) == 10**6
    assert optimal_runs(DiffVariance(0.1, 0.0), cost_task=5) == 1
    with pytest.raises(ValueError):
        optimal_runs(dv, cost_run=0)
    cost = [r.cost for r in plan(0.05, dv, cost_task=9, cost_run=1, runs_grid=(1, 3, 6, 12, 24))]
    assert min(cost) in (cost[1], cost[2], cost[3])


def test_analytic_power_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    pa = rng.beta(2, 2, size=5000)
    pb = shift_probs(pa, -0.06)
    pb = np.clip(pb + rng.normal(0, 0.05, pb.shape), 0, 1)  # task-specific interaction
    sim_dv = DiffVariance(
        float(np.var(pa - pb)),
        float(np.mean(pa * (1 - pa) + pb * (1 - pb))),
    )
    delta = float(np.mean(pa - pb))
    t, r = 60, 3
    assert simulated_power(pa, pb, t, r, n_sim=3000) == pytest.approx(
        power(delta, t, r, sim_dv), abs=0.04
    )


def test_pilot_estimator_recovers_components() -> None:
    rng = np.random.default_rng(1)
    t, r = 400, 6
    sm, se = 0.02, 0.1
    inter = rng.normal(0, np.sqrt(sm), (2, t, 1))
    y = inter + rng.normal(0, np.sqrt(se), (2, t, r))
    dv = DiffVariance.from_pilot(y[0], y[1])
    assert dv.task == pytest.approx(2 * sm, abs=0.012)
    assert dv.run == pytest.approx(2 * se, abs=0.03)
    comps = DiffVariance.from_components(decompose_array(y))
    assert comps.run == pytest.approx(dv.run, rel=0.2)
    with pytest.raises(ValueError):
        DiffVariance.from_pilot(y[0], y[1][:10])


def test_shift_probs_and_validation() -> None:
    p = np.random.default_rng(2).beta(2, 3, 1000)
    assert shift_probs(p, 0.1).mean() == pytest.approx(p.mean() + 0.1, abs=1e-6)
    with pytest.raises(ValueError):
        simulated_power([0.5, 0.5], [0.5], 5, 2)
    with pytest.raises(ValueError):
        power(0.1, 1, 1, DiffVariance(0.1, 0.1))
    with pytest.raises(ValueError):
        required_tasks(0.0, 1, DiffVariance(0.1, 0.1))
    with pytest.raises(ValueError):
        required_tasks(1e-5, 1, DiffVariance(0.1, 0.1), max_tasks=100)
    assert power(0.1, 10, 1, DiffVariance(0.0, 0.0)) == 1.0
    assert power(0.0, 10, 1, DiffVariance(0.0, 0.0)) == 0.05
    # identical agents with zero noise never reject
    assert simulated_power([0.5] * 4, [0.5] * 4, 6, 2, n_sim=50) < 0.5
