"""Power planning: how many tasks and how many runs per task are needed?

Let ``d`` be the per-task difference of two agents' mean scores. With ``T`` tasks and
``R`` runs per task, the variance of the benchmark-level difference is

    Var = (s_t + s_e / R) / T,

where ``s_t`` is the variance of the *true* per-task difference (model-by-task
interaction, ``2 * sigma2_model_task``) and ``s_e`` the run-noise variance of one
paired run difference (``2 * sigma2_run``). Two consequences drive the planner:

* for a fixed number of attempts ``B = T R`` the variance is ``(R s_t + s_e) / B``, which
  *increases* with ``R``: when tasks are free, spend the budget on more tasks, not repeats;
* when a task is costly to obtain (cost ``c_task``) relative to a run (``c_run``), the
  variance-minimising number of runs per task is ``R* = sqrt(s_e c_task / (s_t c_run))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from agentaudit.variance import VarianceComponents, decompose_single

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiffVariance:
    """Variance components of a paired per-task difference."""

    task: float  # s_t: variance of the true per-task difference
    run: float  # s_e: run-noise variance of one paired run difference

    def var_mean(self, n_tasks: int, n_runs: int) -> float:
        return (self.task + self.run / n_runs) / n_tasks

    @classmethod
    def from_components(cls, comps: VarianceComponents) -> DiffVariance:
        return cls(2.0 * comps.model_task, 2.0 * comps.run)

    @classmethod
    def from_pilot(cls, a: FloatArray, b: FloatArray) -> DiffVariance:
        """Estimate from a pilot of two agents on the same ``(tasks, runs)`` grid."""
        if a.shape != b.shape:
            raise ValueError("pilot matrices must have the same shape")
        sv = decompose_single(a - b)
        return cls(sv.task, sv.run)


def power(delta: float, n_tasks: int, n_runs: int, dv: DiffVariance, alpha: float = 0.05) -> float:
    """Power of the two-sided paired t-test to detect a true mean difference ``delta``."""
    if n_tasks < 2 or n_runs < 1:
        raise ValueError("need n_tasks >= 2 and n_runs >= 1")
    se = sqrt(dv.var_mean(n_tasks, n_runs))
    if se == 0:
        return 1.0 if delta != 0 else alpha
    df = n_tasks - 1
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    ncp = abs(delta) / se
    val = float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))
    if not np.isfinite(val):  # scipy's noncentral t overflows for very large ncp
        val = float(stats.norm.cdf(ncp - tcrit) + stats.norm.cdf(-ncp - tcrit))
    return min(1.0, val)


def mde(
    n_tasks: int, n_runs: int, dv: DiffVariance, alpha: float = 0.05, target_power: float = 0.8
) -> float:
    """Minimum detectable effect (in score units) at the given power."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power(mid, n_tasks, n_runs, dv, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def required_tasks(
    delta: float,
    n_runs: int,
    dv: DiffVariance,
    alpha: float = 0.05,
    target_power: float = 0.8,
    max_tasks: int = 1_000_000,
) -> int:
    """Smallest ``T`` reaching ``target_power`` for effect ``delta`` with ``n_runs`` per task."""
    if delta == 0:
        raise ValueError("delta must be non-zero")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(target_power)
    t = max(2, int(ceil((za + zb) ** 2 * (dv.task + dv.run / n_runs) / delta**2)))
    while t > 2 and power(delta, t - 1, n_runs, dv, alpha) >= target_power:
        t -= 1
    while power(delta, t, n_runs, dv, alpha) < target_power:
        t += 1
    if t > max_tasks:
        raise ValueError("target not reachable within max_tasks")
    return t


def optimal_runs(dv: DiffVariance, cost_task: float = 1.0, cost_run: float = 1.0) -> int:
    """Integer ``R >= 1`` minimising variance for a fixed budget ``T (c_task + R c_run)``."""
    if cost_task < 0 or cost_run <= 0:
        raise ValueError("costs must satisfy cost_task >= 0, cost_run > 0")
    if dv.task <= 0:
        return 10**6  # no task-level signal: only run noise matters, repeats are the only lever
    if dv.run <= 0 or cost_task == 0:
        return 1
    r_star = sqrt(dv.run * cost_task / (dv.task * cost_run))

    def f(r: int) -> float:
        return (dv.task + dv.run / r) * (cost_task + r * cost_run)

    cands = {max(1, int(np.floor(r_star))), max(1, int(np.ceil(r_star)))}
    return min(cands, key=f)


@dataclass(frozen=True)
class PlanRow:
    n_runs: int
    n_tasks: int
    total_attempts: int
    cost: float


def plan(
    delta: float,
    dv: DiffVariance,
    *,
    runs_grid: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    alpha: float = 0.05,
    target_power: float = 0.8,
    cost_task: float = 0.0,
    cost_run: float = 1.0,
) -> list[PlanRow]:
    """Tasks needed for each candidate number of runs, with total attempts and cost."""
    rows = []
    for r in runs_grid:
        t = required_tasks(delta, r, dv, alpha, target_power)
        rows.append(PlanRow(r, t, t * r, t * (cost_task + r * cost_run)))
    return rows


def simulated_power(
    p_a: ArrayLike,
    p_b: ArrayLike,
    n_tasks: int,
    n_runs: int,
    *,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Monte-Carlo power of the paired t-test on task means.

    ``p_a`` / ``p_b`` are per-task success probabilities of the two agents in a task
    population (paired by index); each simulation resamples ``n_tasks`` tasks with
    replacement and draws ``n_runs`` Bernoulli runs per agent and task.
    """
    pa, pb = np.asarray(p_a, dtype=np.float64), np.asarray(p_b, dtype=np.float64)
    if pa.shape != pb.shape or pa.ndim != 1:
        raise ValueError("p_a and p_b must be 1-D and equally long")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, pa.size, size=(n_sim, n_tasks))
    ya = rng.binomial(n_runs, pa[idx]) / n_runs
    yb = rng.binomial(n_runs, pb[idx]) / n_runs
    d = ya - yb
    sd = d.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = d.mean(axis=1) / (sd / sqrt(n_tasks))
    pvals = 2 * stats.t.sf(np.abs(tstat), n_tasks - 1)
    reject = np.where(sd > 0, pvals < alpha, d.mean(axis=1) != 0)
    return float(reject.mean())


def shift_probs(p: ArrayLike, delta: float, tol: float = 1e-10) -> FloatArray:
    """Shift per-task probabilities on the logit scale so the mean rises by exactly ``delta``."""
    p0 = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    target = float(np.clip(p0.mean() + delta, 1e-6, 1 - 1e-6))
    logit = np.log(p0 / (1 - p0))
    lo, hi = -30.0, 30.0
    out: FloatArray = p0
    for _ in range(200):
        mid = (lo + hi) / 2
        out = 1 / (1 + np.exp(-(logit + mid)))
        if out.mean() < target:
            lo = mid
        else:
            hi = mid
        if abs(out.mean() - target) < tol:
            break
    return out
