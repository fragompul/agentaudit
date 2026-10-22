"""Variance decomposition (generalizability theory) for repeated agent evaluations.

For scores ``Y[m, t, r]`` of agent ``m`` on task ``t`` in run ``r`` we use the crossed
random-effects model

    Y = mu + a_m + b_t + (ab)_mt + e_mtr,

with variance components ``sigma2_model, sigma2_task, sigma2_model_task, sigma2_run``.
Estimation is by the method of moments from the balanced ANOVA mean squares
(expected mean squares below), which is unbiased before truncation at zero.
For binary scores this is a linear-probability decomposition: it is exact for
the variance of *means*, which is what the standard errors and power planner need,
but the components are not latent-scale quantities.

    E[MS_model]      = s_e + R s_mt + T R s_m
    E[MS_task]       = s_e + R s_mt + M R s_t
    E[MS_model:task] = s_e + R s_mt
    E[MS_run]        = s_e
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from agentaudit.results import ResultSet

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VarianceComponents:
    """Estimated variance components (truncated at 0) of the crossed model."""

    model: float
    task: float
    model_task: float
    run: float
    n_models: int
    n_tasks: int
    n_runs: int
    raw: tuple[float, float, float, float]
    interaction_confounded: bool = False

    @property
    def total(self) -> float:
        return self.model + self.task + self.model_task + self.run

    def shares(self) -> dict[str, float]:
        tot = self.total
        if tot <= 0:
            return {"model": 0.0, "task": 0.0, "model_task": 0.0, "run": 0.0}
        return {
            "model": self.model / tot,
            "task": self.task / tot,
            "model_task": self.model_task / tot,
            "run": self.run / tot,
        }

    def var_difference(self, n_tasks: int, n_runs: int) -> float:
        """Variance of the difference between two agents' benchmark means.

        Task main effects cancel because both agents face the same tasks; what remains is
        the model-by-task interaction and run noise: ``2 (s_mt / T + s_e / (T R))``.
        """
        return 2.0 * (self.model_task / n_tasks + self.run / (n_tasks * n_runs))

    def var_mean(self, n_tasks: int, n_runs: int) -> float:
        """Variance of one agent's benchmark mean (tasks drawn at random)."""
        return (self.task + self.model_task) / n_tasks + self.run / (n_tasks * n_runs)

    def g_coefficient(self, n_tasks: int | None = None, n_runs: int | None = None) -> float:
        """Generalizability coefficient for *relative* agent comparisons.

        ``s_m / (s_m + s_mt / T + s_e / (T R))``: the fraction of observed between-agent
        variance that is true skill difference rather than task-sampling or run noise.
        """
        t = n_tasks or self.n_tasks
        r = n_runs or self.n_runs
        denom = self.model + self.model_task / t + self.run / (t * r)
        return self.model / denom if denom > 0 else float("nan")


def decompose(results: ResultSet) -> VarianceComponents:
    """ANOVA estimator of the crossed variance components (balanced data required)."""
    return decompose_array(results.scores)


def decompose_array(y: FloatArray) -> VarianceComponents:
    """Same as :func:`decompose` for a raw ``(models, tasks, runs)`` array (any real values)."""
    if y.ndim != 3 or np.any(np.isnan(y)):
        raise ValueError("variance decomposition requires balanced (no-NaN) 3-D data")
    m, t, r = y.shape
    if m < 2 or t < 2:
        raise ValueError("need at least 2 models and 2 tasks")
    grand = y.mean()
    ym = y.mean(axis=(1, 2))
    yt = y.mean(axis=(0, 2))
    ymt = y.mean(axis=2)
    ss_m = t * r * np.sum((ym - grand) ** 2)
    ss_t = m * r * np.sum((yt - grand) ** 2)
    ss_mt = r * np.sum((ymt - ym[:, None] - yt[None, :] + grand) ** 2)
    ms_m = ss_m / (m - 1)
    ms_t = ss_t / (t - 1)
    ms_mt = ss_mt / ((m - 1) * (t - 1))
    if r > 1:
        ss_e = np.sum((y - ymt[:, :, None]) ** 2)
        ms_e = ss_e / (m * t * (r - 1))
        s_e = float(ms_e)
        s_mt = (ms_mt - ms_e) / r
        confounded = False
    else:
        s_e = 0.0
        s_mt = float(ms_mt)
        confounded = True
    s_m = (ms_m - ms_mt) / (t * r)
    s_t = (ms_t - ms_mt) / (m * r)
    raw = (float(s_m), float(s_t), float(s_mt), s_e)
    return VarianceComponents(
        model=max(raw[0], 0.0),
        task=max(raw[1], 0.0),
        model_task=max(raw[2], 0.0),
        run=max(raw[3], 0.0),
        n_models=m,
        n_tasks=t,
        n_runs=r,
        raw=raw,
        interaction_confounded=confounded,
    )


@dataclass(frozen=True)
class SingleAgentVariance:
    """One-way random-effects decomposition (task, run-within-task) for one agent."""

    task: float
    run: float
    n_tasks: int
    n_runs: int

    @property
    def icc(self) -> float:
        """Intraclass correlation: share of variance attributable to task difficulty."""
        tot = self.task + self.run
        return self.task / tot if tot > 0 else float("nan")

    def var_mean(self, n_tasks: int, n_runs: int) -> float:
        return self.task / n_tasks + self.run / (n_tasks * n_runs)


def decompose_single(scores: FloatArray) -> SingleAgentVariance:
    """Variance of a single agent's ``(tasks, runs)`` matrix into task and run components."""
    if scores.ndim != 2 or np.any(np.isnan(scores)):
        raise ValueError("need a balanced 2-D (tasks, runs) matrix")
    t, r = scores.shape
    if t < 2 or r < 2:
        raise ValueError("need at least 2 tasks and 2 runs")
    ybar = scores.mean(axis=1)
    msb = r * np.sum((ybar - scores.mean()) ** 2) / (t - 1)
    msw = np.sum((scores - ybar[:, None]) ** 2) / (t * (r - 1))
    return SingleAgentVariance(max(float((msb - msw) / r), 0.0), float(msw), t, r)
